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

"""GPU-owning caption stages using in-process ``AsyncLLM``."""

import asyncio
import collections
import concurrent.futures
import itertools
import json
import logging
import os
import time
import warnings
from typing import TYPE_CHECKING, Any, ClassVar

import attrs
import numpy as np
import numpy.typing as npt
import ray

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.infra.gpu_start_helper import gpu_stage_cleanup, gpu_stage_startup
from cosmos_curator.core.utils.infra.performance_utils import StagePerfStats, StageTimer
from cosmos_curator.core.utils.infra.tracing import StatusCode, TracedSpan, traced_span
from cosmos_curator.core.utils.misc.logging_utils import make_tagged_logger
from cosmos_curator.core.utils.misc.memfd import buffer_as_memfd_path
from cosmos_curator.core.utils.model import conda_utils
from cosmos_curator.core.utils.pixi_runtime_envs import PixiRuntimeEnv
from cosmos_curator.models.prompts import build_refinement_prompt_text, get_prompt
from cosmos_curator.models.vllm_model_ids import get_vllm_model_id
from cosmos_curator.pipelines.video.captioning.vllm_async_config import VllmAsyncConfig, VllmAsyncPrepConfig
from cosmos_curator.pipelines.video.utils.data_model import (
    Clip,
    SplitPipeTask,
    TokenCounts,
    Window,
    get_video_from_task,
)
from cosmos_curator.pipelines.video.utils.decoder_utils import (
    decode_video_cpu_frame_ids,
    get_avg_frame_rate,
    get_frame_count,
)
from cosmos_curator.pipelines.video.utils.vision_process import smart_nframes
from cosmos_curator.pipelines.video.utils.windowing_utils import (
    compute_windows,
    window_source_time_trace_attributes,
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
    from vllm.config import CompilationConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.multimodal.processing.inputs import ProcessorInputs
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

if conda_utils.is_running_in_env("unified"):
    from transformers import AutoProcessor
    from vllm.config import CompilationConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.utils.gc_utils import freeze_gc_heap
    from vllm.v1.engine.async_llm import AsyncLLM

    from cosmos_curator.models.vllm_interface import sampling_params as build_sampling_params
else:

    def freeze_gc_heap() -> None:  # type: ignore[misc]
        """No-op fallback when vLLM is not installed."""


_module_logger = make_tagged_logger("[asyncvLLM]")


def _build_render_payload(
    prompt_text: str,
    decoded_rgb_frames: npt.NDArray[np.uint8],
    mm_processor_kwargs_json: str,
) -> dict[str, Any]:
    """Build a fresh renderer payload from stable prompt primitives."""
    payload: dict[str, Any] = {
        "prompt": prompt_text,
        "multi_modal_data": {"video": [decoded_rgb_frames]},
    }
    if mm_processor_kwargs_json:
        payload["mm_processor_kwargs"] = json.loads(mm_processor_kwargs_json)
    return payload


def resolve_model_path(model_id: str) -> str:
    """Resolve a model ID to a local path with pre-downloaded weights."""
    from cosmos_curator.core.utils.model.model_utils import get_local_dir_for_weights_name  # noqa: PLC0415

    local_dir = get_local_dir_for_weights_name(model_id)
    if local_dir.exists():
        _module_logger.info("Reusing cached model weights at {}", local_dir)
        return str(local_dir)

    msg = (
        f"Pre-downloaded model weights not found for '{model_id}'. "
        f"Expected path: {local_dir}. "
        f"Ensure model weights are downloaded before launching vllm async engine "
        f"(e.g. via the model downloader stage or manual placement)."
    )
    raise FileNotFoundError(msg)


def _build_engine_args(config: VllmAsyncConfig, model_path: str) -> "AsyncEngineArgs":
    """Convert a ``VllmAsyncConfig`` into ``AsyncEngineArgs`` for in-process ``AsyncLLM``."""
    tp_size = int(config.num_gpus)

    limit_mm: dict[str, Any] | None = None
    if config.limit_mm_per_prompt:
        limit_mm = json.loads(config.limit_mm_per_prompt)

    mm_kwargs: dict[str, Any] | None = None
    if config.mm_processor_kwargs:
        mm_kwargs = json.loads(config.mm_processor_kwargs)

    comp_config: CompilationConfig | None = None
    if config.cudagraph_mode:
        comp_config = CompilationConfig(cudagraph_mode=config.cudagraph_mode)  # type: ignore[arg-type]

    prefill_threshold = config.long_prefill_token_threshold

    return AsyncEngineArgs(
        model=model_path,
        served_model_name=[config.model_variant],
        tensor_parallel_size=tp_size,
        data_parallel_size=max(1, config.data_parallel_size),
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len if config.max_model_len > 0 else None,  # type: ignore[arg-type]
        dtype=config.dtype,
        quantization=config.quantization or None,  # type: ignore[arg-type]
        max_num_batched_tokens=config.max_num_batched_tokens if config.max_num_batched_tokens > 0 else None,
        max_num_seqs=config.max_num_seqs if config.max_num_seqs > 0 else None,
        enforce_eager=config.enforce_eager,
        trust_remote_code=config.trust_remote_code,
        enable_prefix_caching=True,
        limit_mm_per_prompt=limit_mm,  # type: ignore[arg-type]
        kv_cache_dtype=config.kv_cache_dtype,  # type: ignore[arg-type]
        compilation_config=comp_config,  # type: ignore[arg-type]
        mm_encoder_tp_mode=config.mm_encoder_tp_mode or None,  # type: ignore[arg-type]
        mm_processor_cache_gb=config.mm_processor_cache_gb,
        mm_processor_cache_type=config.mm_processor_cache_type or None,  # type: ignore[arg-type]
        disable_log_stats=config.disable_log_stats,
        enable_log_requests=config.enable_log_requests,
        async_scheduling=config.async_scheduling,
        enable_chunked_prefill=config.enable_chunked_prefill,
        disable_chunked_mm_input=config.disable_chunked_mm_input,
        long_prefill_token_threshold=prefill_threshold,
        stream_interval=config.stream_interval,
        distributed_executor_backend=config.distributed_executor_backend or None,
        skip_mm_profiling=config.skip_mm_profiling,
        use_tqdm_on_load=False,
        mm_processor_kwargs=mm_kwargs,
    )


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


class VllmAsyncPrepStage(CuratorStage):
    """CPU-only prep stage: windowing, frame decode, and ``TextPrompt`` build."""

    def __init__(
        self,
        *,
        prep_config: VllmAsyncPrepConfig,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the prep stage with config."""
        super().__init__()
        self._timer = StageTimer(self)
        self._prep_config = prep_config
        self._model_variant = "vllm_async"
        self._verbose = verbose
        self._log_stats = log_stats
        self._vllm_model = _VllmAsyncModel(prep_config.model_variant)
        self._log_tag = f"[asyncvLLM-prep:{prep_config.model_variant}]"
        self._logger = make_tagged_logger(self._log_tag)
        self._processor: AutoProcessor | None = None
        self._prompt_template: str | None = None
        self._decode_workers: int = 1

    def __getstate__(self) -> dict[str, Any]:
        """Exclude non-serializable objects from pickling."""
        state = self.__dict__.copy()
        state.pop("_logger", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore instance state and recreate non-serializable objects."""
        self.__dict__.update(state)
        self._logger = make_tagged_logger(self._log_tag)

    def secondary_name(self) -> str:
        """Return the model variant for logging."""
        return self._model_variant

    @property
    def model(self) -> ModelInterface:
        """Return the model interface for automatic weight download."""
        return self._vllm_model

    @property
    def resources(self) -> CuratorStageResource:
        """Declare CPU resources for Xenna scheduling."""
        return CuratorStageResource(cpus=0.5)

    @property
    def conda_env_name(self) -> str:
        """Use the unified environment (AutoProcessor for chat template rendering)."""
        return "unified"

    def stage_setup(self) -> None:
        """Load AutoProcessor and prompt template."""
        if self._prep_config.decode_workers > 0:
            self._decode_workers = self._prep_config.decode_workers
        else:
            self._decode_workers = max(1, (os.cpu_count() or 1) // 10)
        self._logger.info(
            "stage_setup starting (decode_workers={}, host_cpus={})",
            self._decode_workers,
            os.cpu_count(),
        )

        model_id = get_vllm_model_id(self._prep_config.model_variant)
        model_path = resolve_model_path(model_id)

        self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)  # type: ignore[no-untyped-call]
        self._logger.info("AutoProcessor loaded from {}", model_path)

        self._prompt_template = self._compute_prompt_template()

    def _compute_prompt_template(self) -> str:
        """Build the tokenized prompt text once, using a bare video placeholder."""
        if self._processor is None:
            msg = "AutoProcessor not initialized; call stage_setup first."
            raise RuntimeError(msg)

        prompt = get_prompt(self._prep_config.prompt_variant, self._prep_config.prompt_text, verbose=self._verbose)
        instruction = prompt.strip()
        messages: list[dict[str, str | list[dict[str, str]]]] = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": instruction},
                ],
            },
        ]
        prompt_text: str = self._processor.apply_chat_template(  # type: ignore[attr-defined]
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        self._logger.debug("Prompt template computed ({} chars)", len(prompt_text))
        return prompt_text

    def _build_prompt(self) -> str:
        """Return the precomputed chat-template prompt for one window."""
        if self._prompt_template is None:
            msg = "Prompt template not initialized; call stage_setup first."
            raise RuntimeError(msg)
        return self._prompt_template

    def _create_windows_and_decode(self, clip: Clip) -> list[Window]:
        """Create windows from ``clip.encoded_data`` and decode frames via memfd."""
        clip_data = clip.encoded_data.resolve()
        if clip_data is None:
            self._logger.warning("Clip {} has no encoded_data, skipping", clip.uuid)
            clip.errors["encoded_data"] = "empty"
            return []

        with buffer_as_memfd_path(clip_data, name="vllm-prep-clip") as video_path:
            native_fps = get_avg_frame_rate(video_path)
            total_native = get_frame_count(clip_data)
            window_infos = compute_windows(
                total_native,
                self._prep_config.window_size,
                self._prep_config.remainder_threshold,
            )
            if not window_infos:
                self._logger.debug("Clip {} produced 0 windows (total_native={})", clip.uuid, total_native)
                return []

            all_indices: list[int] = []
            frame_counts: list[int] = []
            for wi in window_infos:
                n_native = wi.end - wi.start + 1
                n_sampled = smart_nframes(self._prep_config.sample_fps, n_native, native_fps)
                indices = np.linspace(wi.start, wi.end, n_sampled, dtype=np.int32).tolist()
                all_indices.extend(indices)
                frame_counts.append(n_sampled)

            all_frames = decode_video_cpu_frame_ids(
                video_path,
                np.array(all_indices, dtype=np.int32),
                num_threads=2,
            )

        windows: list[Window] = []
        offset = 0
        total_decoded = all_frames.shape[0]
        for wi, count in zip(window_infos, frame_counts, strict=True):
            actual = min(count, total_decoded - offset)
            if actual <= 0:
                self._logger.warning(
                    "Clip {} window [{}, {}]: no frames remaining (expected {}, decoded={}), skipping",
                    clip.uuid,
                    wi.start,
                    wi.end,
                    count,
                    total_decoded,
                )
                continue
            if actual < count:
                self._logger.warning(
                    "Clip {} window [{}, {}]: expected {} frames, got {} (PyAV frame drop)",
                    clip.uuid,
                    wi.start,
                    wi.end,
                    count,
                    actual,
                )

            frames_slice = all_frames[offset : offset + actual]
            offset += actual
            try:
                window = Window(start_frame=wi.start, end_frame=wi.end)
                prompt_text = self._build_prompt()
                frames_shape = tuple(frames_slice.shape)
                window.model_input[self._model_variant] = {
                    "prompt": prompt_text,
                    "video_frames": frames_slice,
                    "frames_shape": frames_shape,
                }
                # Append only after the window is fully assembled so failed
                # windows never leak as partially-built orphans onto clip.windows.
                clip.windows.append(window)
                windows.append(window)
                self._logger.debug(
                    "Window [{}, {}]: frames_shape={}",
                    wi.start,
                    wi.end,
                    frames_shape,
                )
            except Exception as exc:  # noqa: BLE001
                clip.errors[f"vllm_async_prep_window_{wi.start}_{wi.end}"] = f"window assembly failed: {exc}"
                self._logger.warning(
                    "window assembly failed for clip {} window [{}, {}]: {}",
                    clip.uuid,
                    wi.start,
                    wi.end,
                    exc,
                    exc_info=True,
                )

        if self._prep_config.keep_mp4:
            self._extract_mp4_bytes_for_windows(clip_data, windows)

        del clip_data
        return windows

    def _extract_mp4_bytes_for_windows(
        self,
        clip_data: npt.NDArray[np.uint8],
        windows: list[Window],
    ) -> None:
        """Extract per-window MP4 bytes for ``PreviewStage`` compatibility."""
        from cosmos_curator.pipelines.video.utils.windowing_utils import split_video_into_windows  # noqa: PLC0415

        mp4_bytes_list, _, window_infos = split_video_into_windows(
            clip_data,
            window_size=self._prep_config.window_size,
            remainder_threshold=self._prep_config.remainder_threshold,
            return_bytes=True,
            return_video_frames=False,
        )
        # Build a lookup from (start, end) -> mp4 bytes so that skipped
        # windows (from frame drops) don't misalign the assignment.
        mp4_by_range: dict[tuple[int, int], bytes] = {}
        for wi, mp4_bytes in zip(window_infos, mp4_bytes_list, strict=True):
            if mp4_bytes is not None:
                mp4_by_range[(wi.start, wi.end)] = mp4_bytes

        for window in windows:
            mp4_data = mp4_by_range.get((window.start_frame, window.end_frame))
            if mp4_data is not None:
                window.mp4_bytes = mp4_data  # type: ignore[assignment]

    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask]:
        """Create windows from clip data, decode frames, and build caption inputs."""
        for task in tasks:
            major_size = task.get_major_size()
            self._timer.reinit(self, major_size)
            video = get_video_from_task(task)

            with (
                self._timer.time_process(),
                concurrent.futures.ThreadPoolExecutor(
                    max_workers=self._decode_workers,
                ) as pool,
            ):
                futures = {pool.submit(self._create_windows_and_decode, clip): clip for clip in video.clips}
                for fut in concurrent.futures.as_completed(futures):
                    clip = futures[fut]
                    try:
                        fut.result()
                    except Exception as exc:  # noqa: BLE001
                        clip.errors["vllm_async_prep"] = f"windowing+decode failed: {exc}"
                        self._logger.warning(
                            "Clip {} prep failed: {}",
                            clip.uuid,
                            exc,
                            exc_info=True,
                        )

            stage_perf = getattr(task, "stage_perf", None)
            if self._log_stats and stage_perf is not None:
                stage_name, stage_perf_stats = self._timer.log_stats()
                stage_perf[stage_name] = stage_perf_stats

        return tasks


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
class _PreparedWindow:
    """Stable raw inputs for a single window awaiting GPU inference."""

    clip: Clip
    window_index: int
    window: Window
    prompt_text: str
    decoded_rgb_frames: npt.NDArray[np.uint8]
    sampling_params: "SamplingParams"
    frames_shape: tuple[int, ...]


@attrs.define(eq=False)
class _ContinuousTaskTracker:
    """In-flight bookkeeping for one ``ContinuousTaskInput`` (one ``SplitPipeTask``).

    ``pending`` collects the asyncio tasks driving stage-1 generation and
    optional stage-2 refinement for every window in the pipeline task.
    ``stage2_queue`` buffers stage-1 captions awaiting refinement so the
    main loop can spawn them in batch on the next tick.

    Why the tracker exists: the 1:1 contract
    -----------------------------------------
    Cosmos-Curate's stage interface defines a strict 1-task-in / 1-task-out
    contract: every ``CuratorStage`` (sync ``process_data`` or continuous
    ``run_continuous``) consumes one ``PipelineTask`` and must emit either
    the same task or ``None``.  ``SplitPipeTask`` carries
    ``list[Video] -> list[Clip] -> list[Window]``, so a single input task
    fans out to N (often 10-100+) per-window GPU inference units inside
    this stage, but only ONE ``ContinuousTaskOutput`` may leave the stage
    - and only after EVERY window in the task is done.

    The tracker is the bookkeeping that bridges N-window fan-out to
    1-task fan-in: it owns every per-window asyncio task and the stage-2
    queue tied to one ``ContinuousTaskInput`` so :meth:`all_done` can
    decide when the whole pipeline task is ready to emit.

    Cost the contract imposes
    ------------------------------------------------------
    - **Memory growth**: each tracker holds N ``_PreparedWindow`` objects
      alive (each a multi-MB ``decoded_rgb_frames`` ndarray + ``prompt_text`` +
      a rendered prompt during inference).  Until ``all_done()`` returns
      True, none of them can be released, even for windows whose stage-1
      and stage-2 captions are already written back into the task.
    - **OOM risk**: with K concurrent tasks in flight on the actor,
      memory footprint is O(K * N * frame_buffer_bytes).  K is bounded by
      the input queue and the framework's continuous concurrency, but N
      is unbounded - a single long video can pin the actor for the
      duration of its slowest window.
    - **Head-of-line blocking**: 1 slow / failed window of N delays
      emission of the entire task; downstream writer / metadata stages
      stay idle waiting for the remaining 99% of work that is already
      complete.
    - **Complexity**: the tracker, ``stage2_queue``, ``all_done``
      reaping, and the unconditional emit in :meth:`run_continuous` only
      exist to bridge fan-out -> fan-in.  Per-window emission would
      collapse this into a single forward-and-forget primitive.
    """

    task_input: ContinuousTaskInput
    pending: set[asyncio.Task[None]] = attrs.field(factory=set)
    stage2_queue: collections.deque[tuple["_PreparedWindow", str]] = attrs.field(factory=collections.deque)
    wall_start: float = attrs.field(factory=time.time)

    def all_done(self) -> bool:
        """Return ``True`` once every window has emitted a final caption.

        Gate that enforces the 1:1 contract: the task can only leave the
        stage after BOTH the per-window inference set (``pending``) and
        the stage-2 refinement backlog (``stage2_queue``) are empty.  A
        single straggling window keeps the entire ``SplitPipeTask`` (and
        all its already-completed siblings' frame buffers) pinned in
        memory.
        """
        return not self.pending and not self.stage2_queue


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
        total = int(config.total_gpus)
        return _VllmAsyncStageMode(
            gpus_per_actor=config.total_gpus,
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


class VllmAsyncCaptionStage(CuratorStage, ContinuousInterface):  # type: ignore[misc]
    """GPU stage that runs an in-process ``AsyncLLM`` engine for video captioning."""

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
    ) -> None:
        """Initialize stage with engine config."""
        super().__init__()
        self._model_name = model_name
        self._model_variant = "vllm_async"
        self._max_concurrent_requests = max_concurrent_requests
        self._verbose = verbose
        self._log_stats = log_stats
        self._serve_config = serve_config
        self._mode = _resolve_mode(serve_config)
        self._stage_batch_size = stage_batch_size
        self._vllm_model = _VllmAsyncModel(serve_config.model_variant)
        self._engine: AsyncLLM | None = None
        self._request_counter: itertools.count[int] = itertools.count()
        self._log_tag = f"[asyncvLLM:{serve_config.model_variant}]"
        self._logger = make_tagged_logger(self._log_tag)
        self._stage2_caption = stage2_caption
        self._stage2_prompt_text = stage2_prompt_text
        self._stage2_processor: AutoProcessor | None = None
        self._gpu_trace_attributes: dict[str, str | int | float | bool] = {}
        # asyncio.Lock serialising calls to the in-process vLLM Renderer.
        # HF tokenizers raise RuntimeError("Already borrowed") under
        # concurrent use; the lock + asyncio.to_thread dispatch keeps the
        # event loop free while ensuring at most one renderer call is
        # in flight inside this actor at any moment.
        self._render_lock: asyncio.Lock = asyncio.Lock()

    def __getstate__(self) -> dict[str, Any]:
        """Exclude non-serializable and derived objects from pickling."""
        state = self.__dict__.copy()
        state.pop("_logger", None)
        state.pop("_render_lock", None)  # asyncio.Lock not picklable
        state.pop("_request_counter", None)
        state.pop("_mode", None)  # derived from _serve_config
        state.pop("_sampling_params", None)  # rebuilt in stage_setup from _serve_config
        state.pop("_stage2_processor", None)  # loaded in stage_setup
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore instance state and recreate non-serializable objects."""
        self.__dict__.update(state)
        self._mode = _resolve_mode(self._serve_config)
        self._logger = make_tagged_logger(self._log_tag)
        self._render_lock = asyncio.Lock()
        self._request_counter = itertools.count()
        self._stage2_processor = None  # loaded in stage_setup
        if not hasattr(self, "_gpu_trace_attributes"):
            self._gpu_trace_attributes = {}

    def secondary_name(self) -> str:
        """Return the model variant for logging."""
        return self._model_variant

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
    #   VLLM_USE_V1                  - removed in v0.17.0
    #   VLLM_ATTENTION_BACKEND       - removed; use AsyncEngineArgs
    #   VLLM_WORKER_MULTIPROC_METHOD - irrelevant with "ray" executor
    _UNSET_VLLM_ENV_VARS: tuple[str, ...] = (
        "VLLM_USE_V1",
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
        # Mirror env_info env vars into os.environ.
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
        """Construct the in-process ``AsyncLLM`` engine and build ``SamplingParams``."""
        self._logger.info("stage_setup starting")
        self._configure_vllm_environment()

        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)

        model_id = get_vllm_model_id(self._serve_config.model_variant)
        model_path = resolve_model_path(model_id)

        if self._stage2_caption:
            self._stage2_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)  # type: ignore[no-untyped-call]
            self._logger.info("AutoProcessor loaded for stage-2 refinement from {}", model_path)

        engine_args = _build_engine_args(self._serve_config, model_path)

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
            engine_args.distributed_executor_backend = self._mode.executor_backend

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
        self._logger.info("AsyncLLM engine ready model={} startup={:.1f}s", model_path, elapsed)

        # Freeze all GC-tracked objects into the oldest generation so that
        # the cyclic GC does not repeatedly scan the millions of long-lived
        # model weight tensors during inference.  This reduces GC pause
        # jitter without affecting correctness - new short-lived objects
        # (request buffers, caption strings) are still collected normally.
        freeze_gc_heap()
        self._logger.debug("GC heap frozen after engine init")

        self._sampling_params = build_sampling_params(self._serve_config.sampling_config)
        self._logger.info("SamplingParams: {}", self._sampling_params)

        self._logger.info(
            "Engine config: prefix_caching={} mm_cache_gb={} mm_cache_type={} chunked_prefill={}",
            engine_args.enable_prefix_caching,
            self._serve_config.mm_processor_cache_gb,
            self._serve_config.mm_processor_cache_type or "lru",
            engine_args.enable_chunked_prefill,
        )

        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=False)

        self._gpu_trace_attributes = _vllm_async_collect_gpu_trace_attributes(self)
        TracedSpan.current().set_attributes(self._gpu_trace_attributes)

    def _mark_window_success(self, window: Window, caption: str) -> None:
        """Record a successful async caption result on a window."""
        window.caption[self._model_variant] = caption
        window.caption_status = "success"
        window.caption_failure_reason = None

    def _mark_window_error(self, window: Window) -> None:
        """Record an async caption failure for downstream writer gating."""
        window.caption_status = "error"
        window.caption_failure_reason = "exception"

    def _extract_prepared_windows(
        self,
        task: SplitPipeTask,
    ) -> list[_PreparedWindow]:
        """Read flat prep-stage primitives into a :class:`_PreparedWindow` per window.

        ``VllmAsyncPrepStage`` populates ``window.model_input[variant]`` with
        the flat shape ``{"prompt": <str>, "video_frames": <ndarray>,
        "frames_shape": <tuple>}`` - three stable primitives, none of which
        the renderer can mutate in place.  This method copies the references
        into a :class:`_PreparedWindow` and evicts the cached dict so any
        downstream mutation (vLLM / Transformers) cannot leak back into
        upstream state.  Any per-window extraction failure (missing keys
        from a producer-side bug, unexpected types, etc.) is caught,
        recorded on ``clip.errors``, and skipped; the loop continues with
        remaining windows so one bad window cannot poison its siblings.
        """
        result: list[_PreparedWindow] = []
        variant = self._model_variant
        for clip in task.video.clips:
            for wi, window in enumerate(clip.windows):
                cached = window.model_input.get(variant)
                if cached is None:
                    continue
                try:
                    result.append(
                        _PreparedWindow(
                            clip=clip,
                            window_index=wi,
                            window=window,
                            prompt_text=cached["prompt"],
                            decoded_rgb_frames=cached["video_frames"],
                            sampling_params=self._sampling_params,
                            frames_shape=cached["frames_shape"],
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    clip.errors[f"{variant}_caption_{wi}"] = f"input extraction failed: {exc}"
                    self._mark_window_error(window)
                    self._logger.warning(
                        "input extraction failed for clip {} window {} frames=[{}, {}]: {}",
                        clip.uuid,
                        wi,
                        window.start_frame,
                        window.end_frame,
                        exc,
                        exc_info=True,
                    )
                finally:
                    # Renderer mutates dicts in place; release upstream ref on every path.
                    window.model_input.pop(variant, None)
        return result

    async def _render_payload(
        self,
        prompt_text: str,
        video_frames: npt.NDArray[np.uint8],
    ) -> "ProcessorInputs":
        """Render a fresh prompt payload on a worker thread under ``_render_lock``.

        The renderer is not thread-safe (HF tokenizers raise
        ``RuntimeError("Already borrowed")`` under concurrent use), so the
        ``asyncio.Lock`` enforces single-flight semantics inside this actor.
        ``asyncio.to_thread`` keeps the event loop free for ``AsyncLLM.generate``
        result polling while the blocking renderer call executes.
        """
        engine = self._require_engine()
        payload = _build_render_payload(
            prompt_text,
            video_frames,
            self._serve_config.mm_processor_kwargs,
        )
        async with self._render_lock:
            rendered_list = await asyncio.to_thread(engine.renderer.render_cmpl, [payload])  # type: ignore[list-item]
        rendered: ProcessorInputs = rendered_list[0]
        return rendered  # type: ignore[no-any-return]

    async def _generate_and_assign(
        self,
        pw: _PreparedWindow,
        semaphore: asyncio.Semaphore,
        stage2_queue: "collections.deque[tuple[_PreparedWindow, str]]",
    ) -> None:
        """Render + generate a caption for a single window and assign it.

        If renderer or engine fails, ``_await_and_reap`` re-raises the exception,
        ``run_continuous`` exits, and Xenna restarts the actor.
        """
        with traced_span(
            "VllmAsyncCaptionStage.generate_and_assign",
            attributes={
                "window.index": pw.window_index,
                "window.clip_uuid": str(pw.clip.uuid),
                "window.start_frame": pw.window.start_frame,
                "window.end_frame": pw.window.end_frame,
                "window.clip_source": pw.clip.source_video,
                **self._gpu_trace_attributes,
                **window_source_time_trace_attributes(pw.clip, pw.window),
            },
        ):
            rendered_prompt = await self._render_payload(pw.prompt_text, pw.decoded_rgb_frames)
            try:
                async with semaphore:
                    caption, tc = await self._generate_caption_async(
                        rendered_prompt,
                        pw.sampling_params,
                        pw.frames_shape,
                        pw.clip.source_video,
                        pw.window_index,
                    )
            finally:
                # Drop the rendered prompt eagerly even on error so sibling
                # in-flight tasks do not stack up multimodal tensors during
                # actor teardown.  Mirrors `_stage2_refine_and_assign`.
                del rendered_prompt

            pw.window.token_counts[self._model_variant] = tc

            if self._stage2_caption and self._stage2_processor is not None:
                stage2_queue.append((pw, caption))
                return

            self._mark_window_success(pw.window, caption)
            if self._verbose:
                self._logger.info(
                    "Caption for clip {} window {} frames=[{}, {}]: {}",
                    pw.clip.uuid,
                    pw.window_index,
                    pw.window.start_frame,
                    pw.window.end_frame,
                    caption,
                )

    async def _stage2_refine_and_assign(
        self,
        pw: _PreparedWindow,
        stage1_caption: str,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Run stage-2 refinement on a window and assign the final caption.

        As with stage 1, exceptions are not caught - they propagate up to
        ``_await_and_reap`` and trigger an actor restart.
        """
        with traced_span(
            "VllmAsyncCaptionStage.stage2_refine",
            attributes={
                "window.index": pw.window_index,
                "window.clip_uuid": str(pw.clip.uuid),
                "window.start_frame": pw.window.start_frame,
                "window.end_frame": pw.window.end_frame,
                "window.clip_source": pw.clip.source_video,
                **window_source_time_trace_attributes(pw.clip, pw.window),
            },
        ):
            refined_prompt_text = build_refinement_prompt_text(
                self._stage2_processor,
                stage1_caption,
                self._stage2_prompt_text,
            )
            stage2_rendered = await self._render_payload(refined_prompt_text, pw.decoded_rgb_frames)
            try:
                async with semaphore:
                    caption, s2_tc = await self._generate_caption_async(
                        stage2_rendered,
                        pw.sampling_params,
                        pw.frames_shape,
                        pw.clip.source_video,
                        pw.window_index,
                    )
                    # Accumulate stage-2 tokens on top of stage-1 counts already stored.
                    existing = pw.window.token_counts.get(self._model_variant, TokenCounts())
                    pw.window.token_counts[self._model_variant] = TokenCounts(
                        existing.prompt_tokens + s2_tc.prompt_tokens,
                        existing.output_tokens + s2_tc.output_tokens,
                    )
            finally:
                del stage2_rendered

            self._mark_window_success(pw.window, caption)
            if self._verbose:
                self._logger.info(
                    "Caption for clip {} window {} frames=[{}, {}]: {}",
                    pw.clip.uuid,
                    pw.window_index,
                    pw.window.start_frame,
                    pw.window.end_frame,
                    caption,
                )

    def _require_engine(self) -> "AsyncLLM":
        """Return the ``AsyncLLM`` engine, raising if not initialised."""
        engine = self._engine
        if engine is None:
            msg = "AsyncLLM engine not initialized; call stage_setup() before generating captions."
            raise RuntimeError(msg)
        return engine

    async def _generate_caption_async(
        self,
        rendered_prompt: "ProcessorInputs",
        sampling_params: "SamplingParams",
        frames_shape: tuple[int, ...],
        clip_source: str,
        window_index: int,
    ) -> tuple[str, TokenCounts]:
        """Submit a pre-rendered prompt to the ``AsyncLLM`` engine and return the caption.

        Returns:
            Tuple of (caption_text, token_counts).

        """
        request_id = f"caption-{next(self._request_counter)}"

        with traced_span(
            "VllmAsyncCaptionStage.generate",
            attributes={
                "inference.request_id": request_id,
                "inference.clip_source": clip_source,
                "inference.window_index": window_index,
            },
        ) as span:
            engine = self._require_engine()
            final_output = None
            async for output in engine.generate(
                prompt=rendered_prompt,  # type: ignore[arg-type]  # ProcessorInputs accepted at runtime; vLLM stubs omit it
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                final_output = output
            if final_output is None or not final_output.outputs:
                msg = f"AsyncLLM engine returned no outputs. model={self._model_name!r} frames_shape={frames_shape}"
                span.set_status(StatusCode.ERROR, msg)
                raise RuntimeError(msg)
            out0 = final_output.outputs[0]
            caption_text = out0.text

            prompt_tokens = len(final_output.prompt_token_ids) if final_output.prompt_token_ids else 0
            generated_tokens = len(out0.token_ids) if out0.token_ids else 0
            span.set_attributes(
                {
                    "inference.prompt_tokens": prompt_tokens,
                    "inference.generated_tokens": generated_tokens,
                    "inference.finish_reason": str(out0.finish_reason),
                    "inference.caption_length": len(caption_text) if caption_text else 0,
                }
            )

            if not caption_text or not caption_text.strip():
                msg = (
                    f"AsyncLLM engine returned empty caption."
                    f" finish_reason={out0.finish_reason!r}"
                    f" prompt_tokens={prompt_tokens}"
                    f" generated_tokens={generated_tokens}"
                    f" min_tokens={sampling_params.min_tokens}"
                    f" cumulative_logprob={out0.cumulative_logprob}"
                    f" frames_shape={frames_shape}"
                )
                span.set_status(StatusCode.ERROR, msg)
                raise RuntimeError(msg)
            return str(caption_text).strip(), TokenCounts(prompt_tokens, generated_tokens)

    def _register_task(
        self,
        trackers: dict[str, _ContinuousTaskTracker],
        task_input: ContinuousTaskInput,
        semaphore: asyncio.Semaphore,
        output_queue: "asyncio.Queue[ContinuousTaskOutput]",
    ) -> None:
        """Register a continuous-mode task and drive its prepared windows."""
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

        prepared = [pw for t in pipe_tasks for pw in self._extract_prepared_windows(t)]
        if not prepared:
            self._logger.debug(
                "task {} produced no prepared windows; emitting synchronously",
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
        for pw in prepared:
            tracker.pending.add(asyncio.create_task(self._generate_and_assign(pw, semaphore, tracker.stage2_queue)))

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

        ``actor_idle_time`` / ``input_data_size_mb`` / RSS deltas are left
        at their zero defaults: they are derived from
        ``StageTimer.reinit/log_stats`` book-keeping that has no meaningful
        per-task definition when N tasks overlap on one actor.

        Aggregation note: with N tasks in flight per actor, the summed
        ``process_time`` across tasks is total **task-seconds** spent in
        this stage and may exceed wall-clock duration by up to N. Use the
        min/max-aggregated ``wall_start`` / ``wall_end`` for the wall span.
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

    def _spawn_stage2_tasks(
        self,
        trackers: dict[str, _ContinuousTaskTracker],
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Spawn pending stage-2 refinement tasks across all in-flight trackers."""
        for tracker in trackers.values():
            while tracker.stage2_queue:
                pw, stage1_caption = tracker.stage2_queue.popleft()
                tracker.pending.add(asyncio.create_task(self._stage2_refine_and_assign(pw, stage1_caption, semaphore)))

    async def _await_and_reap(
        self,
        trackers: dict[str, _ContinuousTaskTracker],
    ) -> None:
        """Wait for at least one in-flight task to complete and re-raise errors."""
        all_pending = {task for tracker in trackers.values() for task in tracker.pending}
        if not all_pending:
            return
        done, _ = await asyncio.wait(all_pending, return_when=asyncio.FIRST_COMPLETED)
        # Drop completed tasks from each tracker's pending set in one pass.
        for tracker in trackers.values():
            tracker.pending -= done
        # Retrieve every exception so asyncio does not log "Task exception was
        # never retrieved" at GC; Task.exception() raises CancelledError on
        # cancelled tasks, so the explicit loop with try/except keeps the drain
        # watertight even during actor teardown.
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

        TracedSpan.current().set_attributes(self._gpu_trace_attributes)
        self._logger.info(
            "run_continuous starting (semaphore={})",
            self._effective_max_concurrent_requests,
        )

        while not stop_event.is_set():
            self._spawn_stage2_tasks(trackers, semaphore)

            has_pending = any(tr.pending for tr in trackers.values())
            if has_pending:
                await self._await_and_reap(trackers)

            self._emit_completed_tasks(trackers, output_queue)

            if has_pending:
                # Opportunistically pull more work without blocking.
                self._drain_input_queue(input_queue, trackers, semaphore, output_queue)
                continue

            # Nothing pending: block on the input queue until either a task
            # arrives or the stop-event timeout fires.
            try:
                async with asyncio.timeout(_INPUT_GET_TIMEOUT_S):
                    task_input = await input_queue.get()
            except TimeoutError:
                continue
            self._register_task(trackers, task_input, semaphore, output_queue)
            self._drain_input_queue(input_queue, trackers, semaphore, output_queue)

        # stop_event fired - drain any in-flight work before exiting.
        while trackers:
            self._spawn_stage2_tasks(trackers, semaphore)
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
            self._stage2_processor = None
        finally:
            gpu_stage_cleanup(self.__class__.__name__)
