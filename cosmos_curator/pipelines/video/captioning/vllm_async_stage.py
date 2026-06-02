# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU-owning caption stage using in-process ``AsyncLLM``.

Architecture
------------

The async pipeline reuses sync's ``VllmPrepStage`` (CPU-side windowing,
deterministic resize, tokenization).  All that is async-specific here is
the GPU-side captioning stage, which runs an in-process
:class:`vllm.v1.engine.async_llm.AsyncLLM` engine and pumps requests
concurrently through ``engine.generate`` under an actor-wide
``asyncio.Semaphore``.

Lifecycle:

    +----------------------+        +-----------------------------+
    |  sync VllmPrepStage  |  --->  |   VllmAsyncCaptionStage     |
    |  (CPU, smart_resize, |        |   (in-process AsyncLLM,     |
    |   plugin.make_llm_   |        |    asyncio orchestrator     |
    |   input -> dict)     |        |    via vllm_caption_async)  |
    +----------------------+        +-----------------------------+

Per-pipe-task fan-out:

    run_continuous
       |
       v
    _register_task -> vllm_caption_async( request iterator )
                                 |
                                 +-- per-window asyncio task
                                 +-- per-window asyncio task
                                 +-- ...
                                 |   (each gates engine.generate
                                 |    via actor-wide Semaphore)
                                 |
                                 v
                           on_window_done callback
                                 |
                                 v
                        scatter caption + free buffer

The stage hands ``vllm_caption_async`` an ``Iterator[VllmCaptionRequest]``
(built lazily by :meth:`VllmAsyncCaptionStage._iter_requests`) instead
of a materialized list - ``_AsyncCaptioner`` becomes the single owner
of every per-window payload alias the moment ``_pending`` registers a
request.

Only a single ``asyncio.Task`` lives in ``tracker.pending`` per pipe task:
the inner per-window concurrency is owned by ``vllm_caption_async``.
"""

import asyncio
import contextlib
import dataclasses
import itertools
import json
import logging
import os
import time
import warnings
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, ClassVar

import attrs
import ray

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.infra.gpu_start_helper import gpu_stage_cleanup, gpu_stage_startup
from cosmos_curator.core.utils.infra.performance_utils import StagePerfStats
from cosmos_curator.core.utils.infra.tracing import TracedSpan, traced_span
from cosmos_curator.core.utils.misc.logging_utils import make_tagged_logger
from cosmos_curator.core.utils.model import pixi_utils
from cosmos_curator.core.utils.pixi_runtime_envs import PixiRuntimeEnv
from cosmos_curator.models.vllm_model_ids import get_vllm_model_id
from cosmos_curator.pipelines.video.captioning.single_inference import SingleInferenceCaptionStage
from cosmos_curator.pipelines.video.utils.data_model import (
    CaptionOutcome,
    Clip,
    SplitPipeTask,
    VllmAsyncConfig,
    VllmCaptionRequest,
    Window,
    get_video_from_task,
)
from cosmos_xenna.ray_utils.continuous_stage import (
    ContinuousInterface,
    ContinuousTaskInput,
    ContinuousTaskOutput,
)
from cosmos_xenna.ray_utils.runtime_envs import CondaEnv, RuntimeEnv
from cosmos_xenna.utils.exception_utils import unwrap_taskgroup_exception_group
from cosmos_xenna.utils.gpu import get_gpu_ids_from_cuda_env_vars

if TYPE_CHECKING:
    from transformers import AutoProcessor
    from vllm import SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

    from cosmos_curator.models.vllm_interface import VllmWindowResult
    from cosmos_curator.models.vllm_plugin import VllmPlugin

if pixi_utils.is_running_in_env("unified"):
    from vllm.utils.gc_utils import freeze_gc_heap
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.exceptions import EngineDeadError

    from cosmos_curator.core.utils.misc.memfd import buffer_as_memfd_path
    from cosmos_curator.models.qwen_vl import QWEN_VARIANTS_NEED_RAW_FRAMES
    from cosmos_curator.models.vllm_interface import (
        _get_vllm_plugin,
        make_model_inputs,
        vllm_caption_async,
    )
    from cosmos_curator.models.vllm_interface import (
        sampling_params as build_sampling_params,
    )
    from cosmos_curator.pipelines.video.captioning.vllm_caption_stage import (
        _free_vllm_inputs,
        _get_stage2_prompts,
        _get_windows_from_tasks,
        _normalize_vllm_result,
    )
    from cosmos_curator.pipelines.video.utils.decoder_utils import get_frame_count
    from cosmos_curator.pipelines.video.utils.vision_process import fetch_video, read_video_cpu
    from cosmos_curator.pipelines.video.utils.windowing_types import WindowFrameInfo
else:

    def freeze_gc_heap() -> None:  # type: ignore[misc]
        """No-op fallback when vLLM is not installed."""

    class EngineDeadError(Exception):  # type: ignore[no-redef]
        """Fallback EngineDeadError when vLLM is not installed."""


_module_logger = make_tagged_logger("[asyncvLLM]")


class _VllmAsyncModel(ModelInterface):
    """Model interface that registers vllm_async weights for download."""

    def __init__(self, model_variant: str) -> None:
        """Initialize with a registered model variant key."""
        self._model_id = get_vllm_model_id(model_variant)

    @property
    def conda_env_name(self) -> str:
        """Return the conda environment where vLLM is installed."""
        return "unified"

    @property
    def model_id_names(self) -> list[str]:
        """Return the HuggingFace model ID for weight download."""
        return [self._model_id]

    def setup(self) -> None:
        """No-op - the AsyncLLM engine loads model weights during stage_setup."""


_MAX_ACCEPT_PER_LOOP: int = 4
"""Upper bound on tasks pulled from the input queue per ``run_continuous`` loop tick.

Bounding the per-tick intake keeps registration latency low without
starving the in-flight reaping / emission code paths in the same loop.
"""

_INPUT_GET_TIMEOUT_S: float = 0.1
"""Maximum block time on ``input_queue.get()`` when nothing is in flight.

Mirrors the framework's ``_collect_continuous_async`` polling cadence so
``stop_event`` is observed within ~100 ms even when no new input arrives.
"""


@attrs.define(eq=False)
class _ContinuousTaskTracker:
    """In-flight bookkeeping for one ``ContinuousTaskInput`` (one ``SplitPipeTask``).

    Enforces the 1-task-in / 1-task-out emission contract of
    ``run_continuous`` while internal per-window fan-out / stage-2
    refinement / retries / semaphore gating all live inside
    :func:`vllm_caption_async`.  ``pending`` holds the single
    ``asyncio.Task`` driving captioning for every window contained in
    this pipe task; the tracker only tracks completion of that task and
    the per-window scatter that ``on_window_done`` performs.

    See ``docs/curator/guides/vllm-async-captioning.md`` for the full
    discussion of the 1:1 contract.
    """

    task_input: ContinuousTaskInput
    pending: set[asyncio.Task[Any]] = attrs.field(factory=set)
    wall_start: float = attrs.field(factory=time.time)

    def all_done(self) -> bool:
        """Return ``True`` once the per-task ``vllm_caption_async`` future has completed."""
        return not self.pending


@attrs.define(frozen=True)
class _VllmAsyncStageMode:
    """Pre-resolved mode-dependent parameters for ``VllmAsyncCaptionStage``."""

    N_ACTORS_SEMAPHORE_LIMIT: ClassVar[int] = 256
    DP_BATCH_MULTIPLIER: ClassVar[int] = 3
    DP_BATCH_FLOOR: ClassVar[int] = 8

    gpus_per_actor: float
    stage_batch_size: int
    semaphore_limit: int
    executor_backend: str
    is_dp_mode: bool


def _resolve_mode(config: VllmAsyncConfig) -> _VllmAsyncStageMode:
    """Resolve all mode-dependent parameters from config."""
    if config.data_parallel_size > 1:
        total = config.total_gpus
        return _VllmAsyncStageMode(
            gpus_per_actor=total,
            stage_batch_size=max(
                _VllmAsyncStageMode.DP_BATCH_MULTIPLIER * total,
                _VllmAsyncStageMode.DP_BATCH_FLOOR,
            ),
            semaphore_limit=_VllmAsyncStageMode.N_ACTORS_SEMAPHORE_LIMIT * total,
            executor_backend=config.distributed_executor_backend,
            is_dp_mode=True,
        )

    backend = "mp" if config.num_gpus == 1 else config.distributed_executor_backend
    return _VllmAsyncStageMode(
        gpus_per_actor=config.num_gpus,
        stage_batch_size=1,
        semaphore_limit=_VllmAsyncStageMode.N_ACTORS_SEMAPHORE_LIMIT,
        executor_backend=backend,
        is_dp_mode=False,
    )


def _vllm_async_collect_gpu_trace_attributes(stage: CuratorStage) -> dict[str, str | int | float | bool]:
    """Build OTel attributes for GPU visibility (cached in ``stage_setup``, applied to later spans).

    Uses :func:`cosmos_xenna.utils.gpu.get_gpu_ids_from_cuda_env_vars` for
    ``CUDA_VISIBLE_DEVICES`` parsing (ints and MIG-style UUID tokens).  If that
    list is empty but Ray is initialized, falls back to ``ray.get_gpu_ids()``.
    Never raises.
    """
    out: dict[str, str | int | float | bool] = {}
    try:
        out["stage.requested_gpus"] = float(stage.resources.gpus)
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd:
            out["stage.cuda_visible_devices"] = cvd
        visible = get_gpu_ids_from_cuda_env_vars()
        if visible:
            out["stage.visible_gpu_ids"] = ",".join(str(x) for x in visible)
        elif ray.is_initialized():
            ray_ids = ray.get_gpu_ids()
            if ray_ids:
                out["stage.ray_gpu_ids"] = ",".join(str(i) for i in ray_ids)
        if ray.is_initialized():
            node_id = ray.get_runtime_context().get_node_id()
            if node_id is not None:
                ns = str(node_id).strip()
                if ns:
                    out["stage.ray_node_id"] = ns
    except Exception as e:  # noqa: BLE001 - tracing must never break stage_setup
        _module_logger.debug("GPU trace attribute collection failed: {}", e, exc_info=True)
    return out


@dataclasses.dataclass(frozen=True)
class CaptionSingleOptions:
    """Sampling overrides applied only by ``VllmAsyncCaptionStage.caption_single``.

    All fields default to ``None`` so the in-process AsyncLLM engine
    keeps the per-window batch path's ``SamplingParams`` derived from
    ``serve_config.sampling_config``. ``sampling_fps`` falls back to
    ``2.0`` (the historical per-event Qwen default) when ``None``.
    """

    sampling_fps: float | None = None
    max_tokens: int | None = None


# Module-level singleton used as the constructor default. Frozen
# dataclasses are safe to share across instances; using a singleton
# satisfies ruff B008 (no mutable function-call defaults).
_DEFAULT_CAPTION_SINGLE_OPTIONS = CaptionSingleOptions()


class VllmAsyncCaptionStage(SingleInferenceCaptionStage, ContinuousInterface):  # type: ignore[misc]
    """GPU stage that runs an in-process ``AsyncLLM`` engine for video captioning.

    The stage is a thin :class:`ContinuousInterface` shell:

    - Per-actor ``asyncio.Semaphore`` gates ``engine.generate`` concurrency.
    - Per-pipe-task ``_ContinuousTaskTracker`` enforces the 1-in / 1-out
      emission contract.
    - Per-window orchestration (request lifecycle, retries, stage-2
      refinement, output decode) lives inside
      :func:`cosmos_curator.models.vllm_interface.vllm_caption_async`.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        serve_config: VllmAsyncConfig,
        model_name: str,
        max_concurrent_requests: int = 0,
        stage_batch_size: int = 0,
        log_stats: bool = False,
        verbose: bool = False,
        stage2_caption: bool = False,
        stage2_prompt_text: str | None = None,
        keep_mp4: bool = False,
        caption_single_options: CaptionSingleOptions = _DEFAULT_CAPTION_SINGLE_OPTIONS,
    ) -> None:
        """Initialize stage with engine config.

        Args:
            serve_config: vLLM async-engine configuration.
            model_name: Human-readable model identifier (used in logs).
            max_concurrent_requests: ``asyncio.Semaphore`` ceiling for
                ``engine.generate``. ``0`` selects mode-based default.
            stage_batch_size: Tasks per ``process_data`` call. ``0``
                selects mode-based default.
            log_stats: Record stage performance statistics on tasks.
            verbose: Verbose vLLM and stage logging.
            stage2_caption: Enable stage-2 caption refinement.
            stage2_prompt_text: Override prompt for stage-2 refinement.
            keep_mp4: Retain ``window.mp4_bytes`` after captioning.
            caption_single_options: Sampling overrides applied only by
                :meth:`caption_single`. The default empty struct keeps
                the per-window batch path's ``SamplingParams`` and the
                historical 2.0 fps decode fallback.

        """
        super().__init__()
        self._model_name = model_name
        # Sync's VllmPrepStage stores per-window LLM inputs under
        # vllm_config.model_variant (e.g. "qwen") - that key is what we
        # READ from window.model_input.  The user-facing caption JSON key
        # for the vllm_async backend stays "vllm_async" regardless of which
        # underlying model variant ran, so we WRITE captions / token counts
        # under a separate constant key.  Two distinct keys preserves
        # backward-compatible output schema without modifying sync prep.
        self._input_key = serve_config.model_variant
        self._caption_key = "vllm_async"
        self._max_concurrent_requests = max_concurrent_requests
        self._verbose = verbose
        self._log_stats = log_stats
        self._serve_config = serve_config
        self._mode = _resolve_mode(serve_config)
        self._stage_batch_size = stage_batch_size
        self._keep_mp4 = keep_mp4
        self._vllm_model = _VllmAsyncModel(serve_config.model_variant)
        self._engine: AsyncLLM | None = None
        self._request_counter: itertools.count[int] = itertools.count()
        self._log_tag = f"[asyncvLLM:{serve_config.model_variant}]"
        self._logger = make_tagged_logger(self._log_tag)
        self._stage2_caption = stage2_caption
        self._stage2_prompt_text = stage2_prompt_text
        self._plugin: VllmPlugin | None = None
        self._processor: AutoProcessor | None = None
        self._gpu_trace_attributes: dict[str, str | int | float | bool] = {}
        # caption_single state (one-shot consumers like PerEventCaptionStage).
        # The dataclass is the public surface; helpers read the unpacked
        # private fields so per-call code paths stay readable.
        self._caption_single_sampling_fps: float = (
            caption_single_options.sampling_fps if caption_single_options.sampling_fps is not None else 2.0
        )
        self._caption_single_max_tokens = caption_single_options.max_tokens
        self._caption_single_sampling_params: SamplingParams | None = None
        self._caption_single_runner: asyncio.Runner | None = None
        self._caption_single_request_counter: int = 0

    def __getstate__(self) -> dict[str, Any]:
        """Exclude non-serializable and derived objects from pickling."""
        state = self.__dict__.copy()
        state.pop("_logger", None)
        state.pop("_request_counter", None)
        state.pop("_mode", None)  # derived from _serve_config
        state.pop("_sampling_params", None)  # rebuilt in stage_setup from _serve_config
        state.pop("_processor", None)  # loaded in stage_setup
        state.pop("_plugin", None)  # resolved in stage_setup
        state.pop("_caption_single_sampling_params", None)  # rebuilt in stage_setup
        state.pop("_caption_single_runner", None)  # asyncio.Runner not picklable
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore instance state and recreate non-serializable objects."""
        self.__dict__.update(state)
        self._mode = _resolve_mode(self._serve_config)
        self._logger = make_tagged_logger(self._log_tag)
        self._request_counter = itertools.count()
        self._processor = None  # loaded in stage_setup
        self._plugin = None  # resolved in stage_setup
        self._caption_single_sampling_params = None  # built in stage_setup
        self._caption_single_runner = None  # built lazily in caption_single
        self._caption_single_request_counter = getattr(self, "_caption_single_request_counter", 0)
        if not hasattr(self, "_gpu_trace_attributes"):
            self._gpu_trace_attributes = {}

    def secondary_name(self) -> str:
        """Return the user-facing caption key (preserves original stage-tag string)."""
        return self._caption_key

    @property
    def model(self) -> ModelInterface:
        """Return the model interface for automatic weight download."""
        return self._vllm_model

    @property
    def resources(self) -> CuratorStageResource:
        """Declare CPU and GPU resources for Xenna scheduling."""
        return CuratorStageResource(cpus=1.0, gpus=self._mode.gpus_per_actor)

    @property
    def conda_env_name(self) -> str:
        """Use the unified environment (vllm + transformers packages live there)."""
        return "unified"

    # Env vars to unset in the worker process.
    #
    #   VLLM_ATTENTION_BACKEND       - use AsyncEngineArgs instead
    #   VLLM_WORKER_MULTIPROC_METHOD - Docker default is for sync vLLM; let
    #                                  async vLLM use its executor default
    _UNSET_VLLM_ENV_VARS: tuple[str, ...] = (
        "VLLM_ATTENTION_BACKEND",
        "VLLM_WORKER_MULTIPROC_METHOD",
    )

    @property
    def env_info(self) -> RuntimeEnv | None:
        """Build and inject the complete env var set into the Ray worker."""

        class _PixiRuntimeEnv(RuntimeEnv):
            def to_ray_runtime_env(self) -> ray.runtime_env.RuntimeEnv:
                return PixiRuntimeEnv(
                    self.conda.name if self.conda else "",
                    env_vars=self.extra_env_vars,
                )

        # Empty string effectively unsets stale vars: vLLM's envs.py
        # lambdas treat "" the same as "not set" for boolean/choice vars.
        env: dict[str, str] = dict.fromkeys(self._UNSET_VLLM_ENV_VARS, "")

        env["VLLM_LOGGING_LEVEL"] = "DEBUG" if self._verbose else "INFO"
        env["VLLM_LOGGING_PREFIX"] = f"{self._log_tag} "

        if not self._verbose:
            env["TQDM_DISABLE"] = "1"

        # Redirect vLLM caches (torch.compile, deep_gemm, model registry,
        # etc.) to /tmp/ so they land on fast local storage instead of the
        # home directory, which may be slow NFS on cloud workers.
        env["VLLM_CACHE_ROOT"] = "/tmp/vllm"  # noqa: S108

        # User overrides applied last so they can override any built-in.
        if self._serve_config.extra_env_vars:
            env.update(json.loads(self._serve_config.extra_env_vars))

        rt_env = _PixiRuntimeEnv(CondaEnv(self.conda_env_name))
        rt_env.extra_env_vars = env
        return rt_env

    @property
    def stage_batch_size(self) -> int:
        """Tasks per ``process_data()`` call."""
        if self._stage_batch_size > 0:
            return self._stage_batch_size
        return self._mode.stage_batch_size

    @property
    def _effective_max_concurrent_requests(self) -> int:
        """Resolve concurrency limit for ``asyncio.Semaphore``."""
        if self._max_concurrent_requests > 0:
            return self._max_concurrent_requests
        return self._mode.semaphore_limit

    def _configure_vllm_environment(self) -> None:
        """Apply env vars to ``os.environ`` and tune Python loggers."""
        rt = self.env_info
        env_vars = rt.extra_env_vars if rt else {}
        for key, value in env_vars.items():
            if value == "":
                removed = os.environ.pop(key, None)
                if removed is not None:
                    self._logger.info("Removed stale env var {}={}", key, removed)
            else:
                os.environ[key] = value

        vllm_log_level = logging.DEBUG if self._verbose else logging.INFO
        logging.getLogger("vllm").setLevel(vllm_log_level)

        # Suppress OTLP exporter internal retry noise (WARNING/ERROR from
        # OTLPSpanExporter when no collector runs on localhost:4318).
        # cosmos-curator uses its own TracerProvider; these loggers are noise.
        logging.getLogger("opentelemetry.exporter.otlp.proto.http").setLevel(logging.CRITICAL)

        self._logger.info(
            "vLLM environment configured: env_vars={}, vllm_log_level={}",
            {k: v for k, v in env_vars.items() if v != ""},
            logging.getLevelName(vllm_log_level),
        )

    def stage_setup(self) -> None:
        """Resolve the plugin, load processor, build ``AsyncEngineArgs``, construct ``AsyncLLM``."""
        self._logger.info("stage_setup starting")
        self._configure_vllm_environment()

        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)

        # Resolve the plugin and load its processor.  The processor is
        # used by stage-2 refinement (`make_refined_llm_request`) - one
        # processor per actor.
        self._plugin = _get_vllm_plugin(self._serve_config.model_variant)
        vllm_config = self._serve_config.to_vllm_config()
        self._processor = self._plugin.processor(vllm_config)
        self._logger.info("Plugin {} loaded; processor ready", type(self._plugin).__name__)

        # Plugin builds AsyncEngineArgs (per-variant constants + user knobs +
        # async-only mm_processor_kwargs invariants).  Single source of truth
        # for sync (`model()`) and async (`model_async()`).
        engine_args = self._plugin.model_async(self._serve_config)

        # Override executor backend with mode-resolved value.
        # In N-actors mode with num_gpus=1, _resolve_mode() auto-selects
        # "mp" to enable async_scheduling (+22% throughput).
        if engine_args.distributed_executor_backend != self._mode.executor_backend:
            self._logger.info(
                "Executor backend auto-selected: {} -> {} (mode={})",
                engine_args.distributed_executor_backend,
                self._mode.executor_backend,
                "N-actors" if not self._mode.is_dp_mode else "DP",
            )
            engine_args = dataclasses.replace(engine_args, distributed_executor_backend=self._mode.executor_backend)

        self._logger.info(
            "Mode: {} | gpus_per_actor={} batch={} sem={} backend={}",
            "DP" if self._mode.is_dp_mode else "N-actors",
            self._mode.gpus_per_actor,
            self.stage_batch_size,
            self._effective_max_concurrent_requests,
            self._mode.executor_backend,
        )

        start_time = time.monotonic()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*trust_remote_code.*Auto classes.*It has no effect here and is ignored.*",
            )
            self._engine = AsyncLLM.from_engine_args(engine_args)
        elapsed = time.monotonic() - start_time
        self._logger.info(
            "AsyncLLM engine ready model={} startup={:.1f}s",
            engine_args.model,
            elapsed,
        )

        # Freeze all GC-tracked objects into the oldest generation so that
        # the cyclic GC does not repeatedly scan the millions of long-lived
        # model weight tensors during inference.  This reduces GC pause
        # jitter without affecting correctness - new short-lived objects
        # (request buffers, caption strings) are still collected normally.
        freeze_gc_heap()
        self._logger.debug("GC heap frozen after engine init")

        self._sampling_params = build_sampling_params(self._serve_config.sampling_config)
        self._logger.info("SamplingParams: {}", self._sampling_params)

        # Build a separate SamplingParams instance for caption_single so
        # per-call overrides (max_tokens) don't pollute the per-window
        # batch params used by run_continuous.
        self._caption_single_sampling_params = build_sampling_params(self._serve_config.sampling_config)
        if self._caption_single_max_tokens is not None:
            self._caption_single_sampling_params.max_tokens = int(self._caption_single_max_tokens)

        self._logger.info(
            "Engine config: prefix_caching={} mm_cache_type={} chunked_prefill={}",
            engine_args.enable_prefix_caching,
            self._serve_config.mm_processor_cache_type or "lru",
            engine_args.enable_chunked_prefill,
        )

        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=False)

        self._gpu_trace_attributes = _vllm_async_collect_gpu_trace_attributes(self)
        TracedSpan.current().set_attributes(self._gpu_trace_attributes)

    def _require_engine(self) -> "AsyncLLM":
        """Return the ``AsyncLLM`` engine, raising if not initialised."""
        engine = self._engine
        if engine is None:
            msg = "AsyncLLM engine not initialized; call stage_setup() before generating captions."
            raise RuntimeError(msg)
        return engine

    def _make_request_id(self) -> str:
        """Build a monotonically-increasing engine-side request_id."""
        return f"caption-{next(self._request_counter)}"

    def _gather_inputs(
        self,
        pipe_tasks: list[SplitPipeTask],
    ) -> tuple[list[Window], list[str]]:
        """Collect ``(windows, clip_uuids)`` for all clips in ``pipe_tasks``.

        Reuses sync's :func:`_get_windows_from_tasks` so the prep-stage
        output is consumed identically in both pipelines.  Intentionally
        does NOT materialize a parallel ``model_inputs`` list - per-
        window LLM input dicts are accessed lazily by
        :meth:`_iter_requests` so the captioner becomes the sole owner
        of every payload alias once dispatch starts.
        """
        return _get_windows_from_tasks(pipe_tasks)

    def _iter_requests(
        self,
        windows: list[Window],
        clip_uuids: list[str],
        stage2_prompts: list[str | None],
    ) -> Iterator[VllmCaptionRequest]:
        """Yield one :class:`VllmCaptionRequest` per window without retaining a list."""
        for window, clip_uuid, stage2_prompt in zip(windows, clip_uuids, stage2_prompts, strict=True):
            request_id = self._make_request_id()
            payload = window.model_input.get(self._input_key)
            if payload is None:
                self._logger.warning(
                    "Window for clip {} [{}, {}] has no model_input[{!r}]; emitting VLLM_UNKNOWN_CAPTION for this slot",
                    clip_uuid,
                    window.start_frame,
                    window.end_frame,
                    self._input_key,
                )
                yield VllmCaptionRequest(
                    request_id=request_id,
                    inputs={},
                    stage2_prompt=stage2_prompt,
                )
            else:
                yield VllmCaptionRequest(
                    request_id=request_id,
                    inputs=payload,
                    stage2_prompt=stage2_prompt,
                )

    def _scatter_one(
        self,
        idx: int,
        result: "VllmWindowResult",
        windows: list[Window],
        clip_uuids: list[str],
    ) -> None:
        """Write one window's caption back and drop its input buffer.

        Mirrors sync's ``_scatter_captions`` for a single index, then
        releases ``window.model_input`` and ``window.mp4_bytes`` so memory
        is reclaimed as soon as each window finishes instead of piling up
        until the whole pipe task completes.
        """
        window = windows[idx]
        clip_uuid = clip_uuids[idx]
        outcome = _normalize_vllm_result(result)
        if outcome.text is not None:
            window.caption[self._caption_key] = outcome.text
        window.token_counts[self._caption_key] = result.token_counts
        window.caption_status = outcome.outcome.value
        window.caption_failure_reason = outcome.failure_reason if outcome.outcome == CaptionOutcome.ERROR else None
        # Drop the cached input dict (written by sync prep under the input
        # key) and the underlying frame buffer immediately so memory does not
        # pile up while the slowest sibling is still captioning.  When
        # ``self._keep_mp4`` is True a downstream stage (preview, persistence)
        # still needs the bytes - mirrors sync ``_free_vllm_inputs`` semantics.
        window.model_input.pop(self._input_key, None)
        if not self._keep_mp4:
            window.mp4_bytes.drop()
        if self._verbose:
            self._logger.info(
                "Caption for clip {} window [{}, {}]: {}",
                clip_uuid,
                window.start_frame,
                window.end_frame,
                outcome.text,
            )

    def _record_window_error(
        self,
        idx: int,
        phase: str,
        exc: Exception,
        clips: list[Clip],
        window_indices: list[int],
    ) -> None:
        """Record the per-window ``clip.errors``."""
        key = f"{self._caption_key}_caption_{window_indices[idx]}"
        if phase != "stage1":
            key = f"{key}_{phase}"
        clips[idx].errors[key] = str(exc)

    def _log_window_error(
        self,
        idx: int,
        phase: str,
        exc: Exception,
        windows: list[Window],
        clip_uuids: list[str],
    ) -> None:
        """Log a per-window failure for diagnostics."""
        window = windows[idx]
        # ``opt(exception=exc)`` attaches the exception object explicitly so
        # loguru renders the traceback even though we are outside the except
        # block at this point (the callback fires after vllm_caption_async
        # has already swallowed and contained the failure).
        self._logger.opt(exception=exc).warning(
            "Caption {} failed for clip {} window [{}, {}]: {}",
            phase,
            clip_uuids[idx],
            window.start_frame,
            window.end_frame,
            exc,
        )

    async def _caption_pipe_tasks(
        self,
        pipe_tasks: list[SplitPipeTask],
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Drive captioning for every window in ``pipe_tasks`` through ``vllm_caption_async``.

        Builds the per-window :class:`VllmCaptionRequest` stream inline
        via :meth:`_iter_requests` (a generator) and hands it to
        :func:`vllm_caption_async`.  Crucially, this method NEVER
        materializes a parallel ``model_inputs`` list - the captioner
        is the sole owner of every payload alias from the moment each
        request enters ``_AsyncCaptioner._pending``.
        """
        engine = self._require_engine()
        if self._plugin is None or self._processor is None:
            msg = "Plugin/processor not initialized; call stage_setup() first."
            raise RuntimeError(msg)

        windows, clip_uuids = self._gather_inputs(pipe_tasks)
        if not windows:
            return

        clips: list[Clip] = []
        window_indices: list[int] = []
        for task in pipe_tasks:
            for clip in get_video_from_task(task).clips:
                if not clip.windows:
                    continue
                clips.extend([clip] * len(clip.windows))
                window_indices.extend(range(len(clip.windows)))
        if len(clips) != len(windows):
            msg = (
                f"clips parallel array misaligned with windows "
                f"(len(clips)={len(clips)} vs len(windows)={len(windows)}); "
                "_get_windows_from_tasks filter has drifted"
            )
            raise RuntimeError(msg)

        vllm_config = self._serve_config.to_vllm_config()
        stage2_vllm = attrs.evolve(
            vllm_config,
            stage2_caption=self._stage2_caption,
            stage2_prompt_text=self._stage2_prompt_text,
        )
        stage2_prompts = _get_stage2_prompts(stage2_vllm, len(windows))

        with traced_span(
            "VllmAsyncCaptionStage.caption_pipe_tasks",
            attributes={
                "stage.num_windows": len(windows),
                "stage.num_pipe_tasks": len(pipe_tasks),
                **self._gpu_trace_attributes,
            },
        ):

            def _on_window_error(idx: int, phase: str, exc: Exception) -> None:
                try:
                    self._record_window_error(idx, phase, exc, clips, window_indices)
                except Exception as record_exc:  # noqa: BLE001 - per-window containment
                    self._logger.opt(exception=record_exc).warning(
                        "Failed to record clip.errors for window idx={} phase={}; original exc was: {}",
                        idx,
                        phase,
                        exc,
                    )
                self._log_window_error(idx, phase, exc, windows, clip_uuids)

            try:
                await vllm_caption_async(
                    self._iter_requests(windows, clip_uuids, stage2_prompts),
                    engine,
                    self._processor,
                    self._sampling_params,
                    vllm_config,
                    semaphore=semaphore,
                    max_retries=self._serve_config.max_retries,
                    request_id_factory=self._make_request_id,
                    on_window_done=lambda idx, result: self._scatter_one(idx, result, windows, clip_uuids),
                    on_window_error=_on_window_error,
                )
            finally:
                # Defence in depth: even if vllm_caption_async returned
                # early (e.g. cancellation), drop any per-window inputs
                # that on_window_done never got to clear.  Mirrors sync's
                # final _free_vllm_inputs call - ``keep_mp4`` propagates
                # the same downstream contract so preview / persistence
                # stages can still consume the clip bytes.
                _free_vllm_inputs(windows, self._input_key, keep_mp4=self._keep_mp4)

    def _decode_video_for_caption_single(self, video_bytes: bytes) -> tuple[Any, dict[str, Any]]:
        """Decode whole-clip mp4 bytes into a frame tensor + HF video metadata.

        Mirrors :meth:`VllmCaptionStage._decode_video_for_caption_single`:
        Qwen3-VL needs raw uint8 TCHW frames + HF's video processor;
        everything else uses the 28-aligned float16 ``fetch_video`` path.
        """
        total_frames = get_frame_count(video_bytes)
        if total_frames <= 0:
            msg = "video bytes contain 0 decodable frames"
            raise RuntimeError(msg)
        sampling_fps = self._caption_single_sampling_fps
        # ``WindowFrameInfo.end`` is inclusive (see windowing_types.py and
        # ``windowing_utils.make_windows_for_video``); ``end=total_frames``
        # would request a frame at out-of-range index ``total_frames``.
        window_range = [WindowFrameInfo(start=0, end=total_frames - 1)]
        needs_raw_frames = self._serve_config.model_variant in QWEN_VARIANTS_NEED_RAW_FRAMES
        with buffer_as_memfd_path(video_bytes, name="vllm-async-caption-single") as path:
            if needs_raw_frames:
                video_tensor, _frame_counts = read_video_cpu(
                    path,
                    sampling_fps,
                    0,
                    window_range,
                )
            else:
                video_tensor, _frame_counts = fetch_video(
                    path,
                    sampling_fps=sampling_fps,
                    window_range=window_range,
                    do_preprocess=True,
                    preprocess_dtype="float16",
                )

        num_sampled_frames = int(video_tensor.shape[0]) if video_tensor.ndim >= 1 else 0
        duration_s = float(num_sampled_frames) / sampling_fps if sampling_fps > 0 else 0.0
        video_metadata: dict[str, Any] = {
            "total_num_frames": num_sampled_frames,
            "fps": float(sampling_fps),
            "duration": duration_s,
            "video_backend": "opencv_dynamic",
            "frames_indices": list(range(num_sampled_frames)),
            "do_sample_frames": False,
        }
        return video_tensor, video_metadata

    async def _generate_caption_single_async(
        self,
        prompt_payload: dict[str, Any],
        request_id: str,
    ) -> str:
        """Drive a single ``engine.generate`` async iterator to completion.

        Lighter than :class:`_AsyncCaptioner._drive_request` (no traced
        span, no token-count plumbing, no stage-2 refinement, no retry)
        because per-event callers don't need the per-window observability.
        """
        engine = self._engine
        params = self._caption_single_sampling_params
        if engine is None or params is None:
            msg = "vllm_async engine not initialised; call stage_setup first."
            raise RuntimeError(msg)
        final_output = None
        async for output in engine.generate(
            prompt=prompt_payload,  # type: ignore[arg-type]
            sampling_params=params,
            request_id=request_id,
        ):
            final_output = output
        if final_output is None or not final_output.outputs:
            msg = "AsyncLLM engine returned no outputs for caption_single."
            raise RuntimeError(msg)
        text = final_output.outputs[0].text
        if not text or not str(text).strip():
            finish = final_output.outputs[0].finish_reason
            msg = f"AsyncLLM engine returned empty caption (finish_reason={finish!r})"
            raise RuntimeError(msg)
        return str(text).strip()

    def caption_single(self, prompt: str, video_bytes: bytes) -> str:
        """Implement :class:`SingleInferenceCaptionStage` for one-shot consumers.

        Synchronous wrapper around the async engine: decodes the clip,
        builds a per-window LLM input dict via the same plugin path as
        the per-window batch, and drives one ``engine.generate`` call to
        completion via a lazily-created ``asyncio.Runner``. Raises
        ``RuntimeError`` on engine / decode failures or empty responses;
        callers (typically :class:`PerEventCaptionStage`) catch and
        record on ``clip.errors``.

        Note: continuous-mode state (``_register_task`` / trackers /
        in-flight semaphore) is not used here. ``caption_single`` is
        only safe to call when the per-window pipeline is *not*
        actively driving ``run_continuous`` on the same engine; per-event
        consumers always own their own actor instance.
        """
        if self._engine is None:
            msg = "vllm_async engine not initialised; call stage_setup before caption_single."
            raise RuntimeError(msg)
        if self._processor is None:
            msg = "Processor not initialised; call stage_setup before caption_single."
            raise RuntimeError(msg)
        if self._caption_single_sampling_params is None:
            msg = "caption_single sampling params not built; call stage_setup first."
            raise RuntimeError(msg)
        if self._caption_single_runner is None:
            self._caption_single_runner = asyncio.Runner()

        video_tensor, video_metadata = self._decode_video_for_caption_single(video_bytes)
        vllm_config = self._serve_config.to_vllm_config()
        llm_inputs = make_model_inputs(
            videos=[video_tensor],
            metadata=[video_metadata],
            config=vllm_config,
            processor=self._processor,
            prompt=prompt,
        )
        if not llm_inputs:
            msg = "make_model_inputs produced no inputs for caption_single."
            raise RuntimeError(msg)

        self._caption_single_request_counter += 1
        request_id = f"caption-single-{self._caption_single_request_counter}"
        return self._caption_single_runner.run(
            self._generate_caption_single_async(llm_inputs[0], request_id),
        )

    def _register_task(
        self,
        trackers: dict[str, _ContinuousTaskTracker],
        task_input: ContinuousTaskInput,
        semaphore: asyncio.Semaphore,
        output_queue: "asyncio.Queue[ContinuousTaskOutput]",
    ) -> None:
        """Register a continuous-mode task and spawn its captioning coroutine."""
        # Pin wall_start at the moment the actor takes ownership of the task so
        # the perf row attributes the extraction cost (and, on the early-exit
        # branch, the only work done) to this stage instead of reporting ~0s.
        entry_time = time.time()
        pipe_tasks: list[SplitPipeTask] = []
        for item in task_input.data:
            if not isinstance(item, SplitPipeTask):
                msg = f"VllmAsyncCaptionStage expected SplitPipeTask in task_input.data, got {type(item).__name__}"
                raise TypeError(msg)
            pipe_tasks.append(item)

        has_windows = any(clip.windows for task in pipe_tasks for clip in get_video_from_task(task).clips)
        if not has_windows:
            self._logger.debug(
                "task {} produced no windows; emitting synchronously",
                task_input.task_id,
            )
            if self._log_stats:
                # Synthesize an instantaneous tracker so the perf row exists
                # for tasks that bypass the inference loop entirely.
                self._record_stage_perf(_ContinuousTaskTracker(task_input=task_input, wall_start=entry_time))
            output_queue.put_nowait(self._build_output(task_input))
            return

        tracker = _ContinuousTaskTracker(task_input=task_input, wall_start=entry_time)
        trackers[task_input.task_id] = tracker
        tracker.pending.add(asyncio.create_task(self._caption_pipe_tasks(pipe_tasks, semaphore)))

    @staticmethod
    def _build_output(task_input: ContinuousTaskInput) -> ContinuousTaskOutput:
        """Construct the ``ContinuousTaskOutput`` that mirrors a ``ContinuousTaskInput``.

        Single source of truth for the input-to-output field copy used by both
        the synchronous zero-window emit in ``_register_task`` and the
        per-tick emit in ``_emit_completed_tasks``.  Keeps the two sites in
        lockstep if a new field is later added to ``ContinuousTaskOutput``.

        ``out_data`` is the **same list** carried by ``task_input.data``:
        captioning mutates each contained ``SplitPipeTask`` in place (window
        captions / token counts / errors are written onto the existing
        ``Window`` instances), so passing the input list through verbatim
        carries all stage results downstream without any copy.
        """
        return ContinuousTaskOutput(
            task_id=task_input.task_id,
            out_data=task_input.data,
            timing=task_input.timing,
            object_sizes=task_input.object_sizes,
        )

    def _record_stage_perf(self, tracker: _ContinuousTaskTracker) -> None:
        """Attach this stage's per-task perf entry to the first ``SplitPipeTask``.

        Continuous mode keeps many tasks in flight inside a single actor;
        writing to every contained ``SplitPipeTask`` would cause
        ``_summarize_perf_stats`` to multi-count residence time.  Mirrors
        the convention used by sync ``VllmCaptionStage`` (writes to
        ``tasks[0]`` only).
        """
        pipe_tasks = [t for t in tracker.task_input.data if isinstance(t, SplitPipeTask)]
        if not pipe_tasks:
            return
        stage_perf = getattr(pipe_tasks[0], "stage_perf", None)
        if stage_perf is None:
            return
        wall_end = time.time()
        stage_perf[type(self).__name__] = StagePerfStats(
            process_time=wall_end - tracker.wall_start,
            wall_start=tracker.wall_start,
            wall_end=wall_end,
        )

    def _drain_input_queue(
        self,
        input_queue: "asyncio.Queue[ContinuousTaskInput]",
        trackers: dict[str, _ContinuousTaskTracker],
        semaphore: asyncio.Semaphore,
        output_queue: "asyncio.Queue[ContinuousTaskOutput]",
    ) -> None:
        """Drain up to ``_MAX_ACCEPT_PER_LOOP`` already-buffered items without blocking.

        Called after a successful blocking ``input_queue.get()`` to absorb
        additional ready tasks in the same loop tick.  Shutdown is driven
        exclusively by ``stop_event`` - there is no in-band sentinel.
        """
        for _ in range(_MAX_ACCEPT_PER_LOOP):
            try:
                task_input = input_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            # ``output_queue`` is forwarded so zero-window tasks emit
            # synchronously rather than waiting for the next loop tick.
            self._register_task(trackers, task_input, semaphore, output_queue)

    def _reap_tracker_done(
        self,
        trackers: dict[str, _ContinuousTaskTracker],
        done: set["asyncio.Task[None]"],
    ) -> None:
        """Mark ``done`` caption tasks complete on every tracker and surface errors.

        Extracted from ``_await_and_reap`` so that the main loop can wait
        on caption futures and the input-queue ``get`` task in the same
        ``asyncio.wait`` set - the loop already knows which subset of
        futures are caption tasks, so it passes them in directly and
        skips the ``trackers`` -> ``pending`` re-collection that
        ``_await_and_reap`` still performs for the final-drain path.

        Retrieves every exception so asyncio does not log "Task exception
        was never retrieved" at GC; ``Task.exception()`` raises
        ``CancelledError`` on cancelled tasks, so the explicit
        try/except keeps the drain watertight even during actor teardown.
        """
        if not done:
            return
        for tracker in trackers.values():
            tracker.pending -= done
        exceptions: list[BaseException] = []
        for task in done:
            try:
                exc = task.exception()
            except asyncio.CancelledError as cancelled:
                exc = cancelled
            if exc is not None:
                exceptions.append(exc)
        if exceptions:
            raise unwrap_taskgroup_exception_group(BaseExceptionGroup("vLLM caption tasks failed", exceptions))

    async def _await_and_reap(
        self,
        trackers: dict[str, _ContinuousTaskTracker],
    ) -> None:
        """Wait for at least one in-flight caption task and re-raise errors.

        Used by the final-drain block of ``run_continuous`` after
        ``stop_event`` fires; the steady-state main loop bypasses this
        helper and calls ``_reap_tracker_done`` directly so that input
        arrival and caption completion share a single ``asyncio.wait``.
        """
        all_pending = {task for tracker in trackers.values() for task in tracker.pending}
        if not all_pending:
            return
        done, _ = await asyncio.wait(all_pending, return_when=asyncio.FIRST_COMPLETED)
        self._reap_tracker_done(trackers, done)

    def _emit_completed_tasks(
        self,
        trackers: dict[str, _ContinuousTaskTracker],
        output_queue: "asyncio.Queue[ContinuousTaskOutput]",
    ) -> None:
        """Emit ``ContinuousTaskOutput`` for every fully completed pipeline task."""
        completed_ids = [task_id for task_id, tr in trackers.items() if tr.all_done()]
        for task_id in completed_ids:
            tracker = trackers.pop(task_id)
            if self._log_stats:
                self._record_stage_perf(tracker)
            output_queue.put_nowait(self._build_output(tracker.task_input))

    async def run_continuous(
        self,
        input_queue: "asyncio.Queue[ContinuousTaskInput]",
        output_queue: "asyncio.Queue[ContinuousTaskOutput]",
        stop_event: asyncio.Event,
    ) -> None:
        """Drive the GPU stage in continuous mode."""
        trackers: dict[str, _ContinuousTaskTracker] = {}
        semaphore = asyncio.Semaphore(self._effective_max_concurrent_requests)
        # ``get_task`` is reused across iterations so a single ``get()``
        # never gets dropped on the floor: if no caption work is pending,
        # we still block on input arrival; once work IS pending, the
        # ``get`` task lives inside the same ``asyncio.wait`` set so the
        # loop wakes on EITHER (a) any caption completion or (b) new
        # input arrival, whichever comes first.
        get_task: asyncio.Task[ContinuousTaskInput] | None = None

        TracedSpan.current().set_attributes(self._gpu_trace_attributes)
        self._logger.info(
            "run_continuous starting (semaphore={})",
            self._effective_max_concurrent_requests,
        )

        try:
            while not stop_event.is_set():
                if get_task is None:
                    get_task = asyncio.create_task(input_queue.get(), name="vllm-async-continuous-get")

                pending_captions: set[asyncio.Task[None]] = {
                    task for tracker in trackers.values() for task in tracker.pending
                }
                wait_set: set[asyncio.Future[Any]] = {*pending_captions, get_task}

                try:
                    async with asyncio.timeout(_INPUT_GET_TIMEOUT_S):
                        done, _pending = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
                except TimeoutError:
                    # No completions and no new input within the timeout:
                    # re-check ``stop_event`` and rebuild the wait set.
                    # ``get_task`` is intentionally NOT cancelled here -
                    # we want to keep the same pending ``get()`` across
                    # idle ticks so a late arrival is not lost.
                    continue

                if get_task in done:
                    # Register the freshly-arrived input, then absorb any
                    # already-buffered burst behind it via NOWAIT drain
                    # (capped at ``_MAX_ACCEPT_PER_LOOP`` so registration
                    # work cannot starve caption reaping or ``stop_event``
                    # observation in the same tick).  ``get_task`` is
                    # cleared so the next iteration recreates it.
                    task_input = get_task.result()
                    self._register_task(trackers, task_input, semaphore, output_queue)
                    self._drain_input_queue(input_queue, trackers, semaphore, output_queue)
                    get_task = None
                    done = done - {get_task} if get_task is not None else done

                # Surface any caption-task completions and propagate errors.
                # The only non-caption future in ``wait_set`` is
                # ``get_task`` and we already removed it from ``done``
                # above, so every remaining element here is a caption
                # ``asyncio.Task[None]`` -- the cast tells mypy that.
                completed_captions: set[asyncio.Task[None]] = {fut for fut in done if isinstance(fut, asyncio.Task)}
                self._reap_tracker_done(trackers, completed_captions)
                self._emit_completed_tasks(trackers, output_queue)
        finally:
            # ``stop_event`` fired (or an exception is propagating).
            # Cancel the still-pending ``get_task`` so it does not leak
            # an "asyncio task was destroyed but it is pending" warning;
            # ``suppress`` covers both the normal cancellation path and
            # the rare race where Xenna already cancelled us.
            if get_task is not None and not get_task.done():
                get_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await get_task

        # stop_event fired - drain any in-flight work before exiting.
        while trackers:
            if any(tr.pending for tr in trackers.values()):
                await self._await_and_reap(trackers)
            self._emit_completed_tasks(trackers, output_queue)

        self._logger.info("run_continuous exiting")

    def destroy(self) -> None:
        """Shut down the ``AsyncLLM`` engine and release GPU memory.

        ``gpu_stage_cleanup`` always runs (even if ``engine.shutdown()``
        raises) so GPU memory is reliably released on actor teardown.
        """
        try:
            if self._engine is not None:
                self._logger.info("destroy: shutting down AsyncLLM engine")
                self._engine.shutdown()  # type: ignore[no-untyped-call]
                self._engine = None
            self._processor = None
            self._plugin = None
            if self._caption_single_runner is not None:
                self._caption_single_runner.close()
                self._caption_single_runner = None
        finally:
            gpu_stage_cleanup(self.__class__.__name__)
