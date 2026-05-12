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

"""Per-event VLM captioning stage.

Runs a VLM on each clip's transcoded mp4 bytes + SAM3 object summary to
produce a structured list of "events" that reference SAM3 object IDs. Designed
to be run after ``SAM3BBoxStage`` and independently of the default caption
stage.

Four backends are available:

* ``qwen`` (default) — local Qwen2.5-VL via vLLM (sync). Single-GPU, no API
  key. Variant menu is ``qwen | qwen3_vl_30b | qwen3_vl_30b_fp8``.
* ``gemini`` — remote Gemini API. CPU-only stage, requires
  ``gemini.api_key`` in the project config.
* ``openai`` — any OpenAI-compatible chat-completion endpoint (e.g. vLLM
  serving an OpenAI-compatible API). CPU-only stage, requires
  ``openai.<endpoint_key>.api_key`` (default key ``caption``).
* ``vllm_async`` — in-process ``AsyncLLM`` engine for any HF model id
  supported by vLLM (including Qwen3-VL-235B-A22B-Instruct[-FP8] with
  TP/DP). Requires GPUs equal to ``config.total_gpus``.

Switch between them via the ``backend`` constructor argument (or
``--event-caption-backend`` on the pipeline CLIs).
"""

import asyncio
import base64
import json
import os
import re
from importlib import resources as importlib_resources
from typing import TYPE_CHECKING, Any, Literal

import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.core.utils.misc.memfd import buffer_as_memfd_path
from cosmos_curator.core.utils.model import conda_utils
from cosmos_curator.models.qwen_vl import QWEN_VARIANTS_NEED_RAW_FRAMES, QwenUtils, QwenVL
from cosmos_curator.pipelines.common.api_caption_utils import (
    create_openai_client_and_resolve_model,
    normalize_openai_response_with_detail,
    openai_error_result_from_exception,
)
from cosmos_curator.pipelines.common.api_stage_async_utils import destroy_api_clients
from cosmos_curator.pipelines.video.captioning.vllm_async_config import VllmAsyncConfig

# ``_VllmAsyncModel`` is a thin ``ModelInterface`` wrapper with no heavy deps,
# so it must be importable from the driver (default env) where ``__init__``
# runs. The vLLM-touching helpers stay inside the gated block below.
from cosmos_curator.pipelines.video.captioning.vllm_async_stage import _VllmAsyncModel
from cosmos_curator.pipelines.video.utils.data_model import CaptionOutcome, CaptionResult, Clip, SplitPipeTask

# Type-checking imports for symbols used purely as annotations.  The
# corresponding runtime imports live in the ``conda_utils.is_running_in_env``
# block below so this module still loads in CPU-only environments.
if TYPE_CHECKING:
    from transformers import AutoProcessor
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

# Heavy ML / API SDKs only live in the ``unified`` env; guard imports so this
# module loads elsewhere (e.g. CPU-only test collection environments).
if conda_utils.is_running_in_env("unified"):
    import numpy as np
    import openai
    import tenacity
    from google import genai
    from google.genai import types as genai_types
    from transformers import AutoProcessor
    from vllm.v1.engine.async_llm import AsyncLLM

    from cosmos_curator.core.utils.infra.gpu_start_helper import gpu_stage_cleanup, gpu_stage_startup
    from cosmos_curator.models.vllm_interface import sampling_params as build_sampling_params
    from cosmos_curator.models.vllm_model_ids import get_vllm_model_id
    from cosmos_curator.pipelines.video.captioning.vllm_async_stage import (
        _build_engine_args,
        _build_render_payload,
        resolve_model_path,
    )
    from cosmos_curator.pipelines.video.utils.decoder_utils import (
        decode_video_cpu_frame_ids,
        get_avg_frame_rate,
        get_frame_count,
    )
    from cosmos_curator.pipelines.video.utils.vision_process import fetch_video, read_video_cpu, smart_nframes
    from cosmos_curator.pipelines.video.utils.windowing_types import WindowFrameInfo


Backend = Literal["qwen", "gemini", "openai", "vllm_async"]
_BACKEND_VALUES: tuple[Backend, ...] = ("qwen", "gemini", "openai", "vllm_async")
_OPENAI_ENDPOINT_KEYS: tuple[str, ...] = ("caption", "enhance", "filter", "classifier")


_DEFAULT_PROMPT_RESOURCE = "traffic_surveillance.md"
_DEFAULT_EVENT_PROMPT = (
    importlib_resources.files("cosmos_curator.pipelines.video.captioning.prompts")
    .joinpath(_DEFAULT_PROMPT_RESOURCE)
    .read_text(encoding="utf-8")
)

# Max length of the raw-response preview logged when extraction yields 0 events.
_RAW_PREVIEW_MAX_CHARS = 1500

# Low default fps — per-event captioning only needs coarse temporal coverage,
# and higher fps multiplies the vision-token budget.
_QWEN_DEFAULT_SAMPLING_FPS = 2.0
# Gemini inline-data cap (≈20 MiB); over this we route through the Files API.
_GEMINI_INLINE_LIMIT_BYTES = 20 * 1024 * 1024


def _build_instances_block(clip: Clip) -> str:
    """Serialize the clip's SAM3 instances as a compact JSON block for the VLM.

    The annotated video carries spatial grounding; this block is text-level
    grounding only. Each entry keeps four fields (``object_id``, ``class``,
    ``start_time_s``, ``end_time_s``) — ``num_frames`` is dropped to keep
    the prompt short.
    """
    instances = clip.sam3_instances or []
    payload: dict[str, Any] = {
        "instances": [
            {
                "object_id": entry.get("object_id"),
                "class": entry.get("prompt", "?"),
                "start_time_s": entry.get("start_time_s"),
                "end_time_s": entry.get("end_time_s"),
            }
            for entry in instances
            if isinstance(entry.get("object_id"), int)
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=False)


def _build_prompt(user_template: str, clip: Clip) -> str:
    """Append clip duration + instances JSON to the user prompt template.

    Duration is explicit so the VLM doesn't hallucinate out-of-range times.
    """
    instances_block = _build_instances_block(clip)
    duration_s = float(clip.duration) if clip.duration else 0.0
    return (
        f"{user_template.strip()}\n\n"
        f"======================================================================\n"
        f"CLIP DURATION: {duration_s:.2f} seconds.\n"
        f"All ``start_time`` / ``end_time`` values MUST lie within [0.0, {duration_s:.2f}].\n"
        f"======================================================================\n"
        f"TRACKED INSTANCES (id -> class -> visibility interval in seconds)\n"
        f"======================================================================\n"
        f"{instances_block}\n"
    )


def _extract_events_payload(text: str) -> list[Any]:
    """Pull the ``events`` array out of a model response.

    Accepts either ``{"events": [...], ...}`` (extra keys ignored) or a bare
    ``[...]`` array. Returns ``[]`` on unparseable input.
    """
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    # ``strict=False`` tolerates literal newlines/tabs inside long caption
    # strings (Gemini 2.5 Flash occasionally forgets to escape them).
    try:
        parsed = json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match is None:
            match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
        if match is None:
            return []
        try:
            parsed = json.loads(match.group(0), strict=False)
        except json.JSONDecodeError:
            return []

    if isinstance(parsed, dict):
        events = parsed.get("events", [])
        return events if isinstance(events, list) else []
    if isinstance(parsed, list):
        return parsed
    return []


class PerEventCaptionStage(CuratorStage):
    """Generate per-event captions referencing SAM3 object IDs.

    Runs after ``SAM3BBoxStage``. Reads the clip's transcoded mp4 bytes
    directly (no prior windowing required) and populates ``clip.sam3_events``.

    Supports two backends: local Qwen2.5-VL (default) and remote Gemini.
    """

    def __init__(  # noqa: PLR0913, PLR0915  # flat config surface keeps CLI wiring straightforward
        self,
        *,
        backend: Backend = "qwen",
        prompt_text: str | None = None,
        # --- Qwen-only ---
        qwen_variant: str = "qwen",
        qwen_sampling_fps: float = _QWEN_DEFAULT_SAMPLING_FPS,
        qwen_max_output_tokens: int = 4096,
        # FP8 requires native FP8 hardware (Hopper/Ada); Ampere falls back to
        # Marlin FP8 which crashes on Qwen2.5-VL's vision encoder shape.
        qwen_fp8: bool = False,
        # QwenVL's built-in sampling is near-greedy (T=0.1, top_p=0.001), which
        # collapses to one cookie-cutter event under heavy vision context.
        # ``None`` keeps the QwenVL default; pass T≈0.3-0.7 + top_p≈0.9 for
        # richer event lists.
        qwen_temperature: float | None = None,
        qwen_top_p: float | None = None,
        qwen_top_k: int | None = None,
        # --- Gemini-only ---
        gemini_model_name: str = "models/gemini-2.5-flash",
        # 16384 gives slack for thinking + a multi-event response. 4096 was
        # empirically too tight: responses truncated mid-event.
        gemini_max_output_tokens: int = 16384,
        # 8 attempts x exponential backoff (base 4 s, max 120 s) + jitter gives
        # a ~9 min worst-case per clip, which straddles typical Flash 503
        # spikes without being wasteful.
        gemini_max_retries: int = 8,
        gemini_retry_delay_seconds: float = 4.0,
        gemini_retry_delay_max_seconds: float = 120.0,
        # Uniform jitter de-synchronises concurrent workers so they don't all
        # retry into the same overloaded server wave.
        gemini_retry_jitter_seconds: float = 5.0,
        gemini_max_video_size_bytes: int = _GEMINI_INLINE_LIMIT_BYTES,
        # Gemini defaults are ~1 fps / low-res, too coarse for sub-second
        # events and too small to OCR ``#id`` overlay labels. 4 fps + HIGH
        # stays under the per-request budget for a 30 s clip.
        gemini_video_fps: float = 4.0,
        gemini_media_resolution: str = "high",
        # -1 = dynamic, 0 = disabled, N = hard cap. Flash needs some thinking
        # to separate "queued vehicles" from "vehicles in contact", so disable
        # only if thinking is starving ``max_output_tokens``.
        gemini_thinking_budget: int = -1,
        # --- OpenAI-compatible-only ---
        openai_model_name: str = "auto",
        openai_max_output_tokens: int = 8192,
        openai_max_retries: int = 3,
        openai_retry_delay_seconds: float = 1.0,
        # Reuses an existing OpenAIConfig endpoint slot (no new schema field);
        # users can point per-event at a separate endpoint via this knob.
        openai_endpoint_key: str = "caption",
        # --- vllm_async-only ---
        # Whole-config object built upstream via build_vllm_async_config(...,
        # prefix="event-caption-"). None when backend != "vllm_async".
        vllm_async_config: VllmAsyncConfig | None = None,
        vllm_async_sampling_fps: float = 2.0,
        vllm_async_max_output_tokens: int = 4096,
        # --- Common ---
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialise the stage.

        Args:
            backend: ``"qwen"`` (local single-GPU vLLM), ``"gemini"`` (remote
                API), ``"openai"`` (any OpenAI-compatible chat-completion
                endpoint), or ``"vllm_async"`` (in-process AsyncLLM engine
                supporting TP/DP, including Qwen3-VL-235B[-FP8]).
            prompt_text: User prompt template; falls back to the bundled
                traffic-surveillance default. The instances JSON block is
                always appended at prompt-build time.
            qwen_variant: Qwen model variant.
            qwen_sampling_fps: Decode-to-tensor fps for Qwen input.
            qwen_max_output_tokens: Qwen max_new_tokens.
            qwen_fp8: Enable FP8 (Hopper/Ada only).
            qwen_temperature: Override sampling temperature. ``None`` = QwenVL
                default (near-greedy, which collapses under heavy vision load;
                try 0.3-0.7 for richer event lists).
            qwen_top_p: Override nucleus ``top_p``. ``None`` = QwenVL default.
            qwen_top_k: Override ``top_k``. ``None`` = QwenVL default.
            gemini_model_name: e.g. ``"models/gemini-2.5-pro"``.
            gemini_max_output_tokens: Gemini max_output_tokens.
            gemini_max_retries: Per-clip retry budget for transient errors.
            gemini_retry_delay_seconds: Base of the exponential backoff.
            gemini_retry_delay_max_seconds: Cap on the exponential tail.
            gemini_retry_jitter_seconds: Uniform jitter added per retry.
            gemini_max_video_size_bytes: Gemini inline-data cap; larger clips
                go through the Files API.
            gemini_video_fps: Video sampling fps (Gemini default ~1 fps is too
                coarse for sub-second events).
            gemini_media_resolution: ``"low"`` / ``"medium"`` / ``"high"``.
                ``"high"`` is required to OCR ``#id`` overlay labels.
            gemini_thinking_budget: ``-1`` = dynamic, ``0`` = disabled, ``N``
                = hard cap.
            openai_model_name: Model name passed in OpenAI-compatible
                chat-completion requests. ``"auto"`` queries ``/v1/models``
                and picks the first entry.
            openai_max_output_tokens: ``max_tokens`` for OpenAI requests.
            openai_max_retries: Per-clip retry budget for transient errors.
            openai_retry_delay_seconds: Fixed delay between retries.
            openai_endpoint_key: Which ``openai.<key>`` block in the project
                config supplies api_key/base_url. Defaults to ``"caption"``
                (reuses the per-window captioner's endpoint).
            vllm_async_config: Whole ``VllmAsyncConfig`` built upstream
                (``build_vllm_async_config(args, sampling_config,
                prefix="event-caption-")``). Required when
                ``backend="vllm_async"``.
            vllm_async_sampling_fps: Decode fps for the annotated mp4 before
                feeding it to the AsyncLLM engine.
            vllm_async_max_output_tokens: ``max_tokens`` for the per-event
                AsyncLLM ``generate`` request (overrides the engine's
                sampling-config max).
            verbose: Emit per-clip logs.
            log_stats: Record stage performance stats.

        """
        super().__init__()
        if backend not in _BACKEND_VALUES:
            msg = f"backend must be one of {_BACKEND_VALUES}, got {backend!r}"
            raise ValueError(msg)

        self._timer = StageTimer(self)
        self._backend: Backend = backend
        self._prompt_template = prompt_text if prompt_text is not None else _DEFAULT_EVENT_PROMPT
        self._verbose = verbose
        self._log_stats = log_stats

        # Qwen state.
        self._qwen_variant = qwen_variant
        self._qwen_sampling_fps = qwen_sampling_fps
        self._qwen_max_output_tokens = qwen_max_output_tokens
        self._qwen_fp8 = qwen_fp8
        self._qwen_temperature = qwen_temperature
        self._qwen_top_p = qwen_top_p
        self._qwen_top_k = qwen_top_k
        self._qwen_utils: QwenUtils | None = None
        self._qwen_model: QwenVL | None = None

        # Gemini state.
        self._gemini_model_name = gemini_model_name
        self._gemini_max_output_tokens = gemini_max_output_tokens
        self._gemini_max_retries = gemini_max_retries
        self._gemini_retry_delay_seconds = gemini_retry_delay_seconds
        self._gemini_retry_delay_max_seconds = gemini_retry_delay_max_seconds
        self._gemini_retry_jitter_seconds = gemini_retry_jitter_seconds
        self._gemini_max_video_size_bytes = gemini_max_video_size_bytes
        self._gemini_video_fps = float(gemini_video_fps)
        self._gemini_media_resolution = gemini_media_resolution.strip().lower()
        if self._gemini_media_resolution not in ("low", "medium", "high"):
            msg = f"gemini_media_resolution must be one of 'low', 'medium', 'high'; got {gemini_media_resolution!r}"
            raise ValueError(msg)
        self._gemini_thinking_budget = int(gemini_thinking_budget)
        self._gemini_api_key: str | None = None
        self._gemini_client: genai.Client | None = None
        # Populated after each Gemini call for diagnostic logging.
        self._last_gemini_finish_reasons: list[str] = []
        self._last_gemini_usage_metadata: object | None = None

        # OpenAI-compatible state.
        if openai_endpoint_key not in _OPENAI_ENDPOINT_KEYS:
            msg = f"openai_endpoint_key must be one of {_OPENAI_ENDPOINT_KEYS}, got {openai_endpoint_key!r}"
            raise ValueError(msg)
        self._openai_model_name = openai_model_name
        self._openai_max_output_tokens = openai_max_output_tokens
        self._openai_max_retries = max(1, openai_max_retries)
        self._openai_retry_delay_seconds = openai_retry_delay_seconds
        self._openai_endpoint_key = openai_endpoint_key
        self._openai_client: openai.OpenAI | None = None
        self._openai_async_client: Any | None = None
        self._openai_runner: asyncio.Runner | None = None

        # vllm_async state.
        if backend == "vllm_async" and vllm_async_config is None:
            msg = "vllm_async_config is required when backend='vllm_async'"
            raise ValueError(msg)
        self._vllm_async_config = vllm_async_config
        self._vllm_async_sampling_fps = vllm_async_sampling_fps
        self._vllm_async_max_output_tokens = vllm_async_max_output_tokens
        self._vllm_async_engine: AsyncLLM | None = None
        self._vllm_async_processor: AutoProcessor | None = None
        self._vllm_async_sampling_params: SamplingParams | None = None
        self._vllm_async_runner: asyncio.Runner | None = None
        self._vllm_async_request_counter: int = 0
        self._vllm_async_model_iface: _VllmAsyncModel | None = None
        if backend == "vllm_async":
            assert vllm_async_config is not None
            self._vllm_async_model_iface = _VllmAsyncModel(vllm_async_config.model_variant)

        if self._backend == "gemini":
            from cosmos_curator.core.utils.config.config import load_config  # noqa: PLC0415

            config = load_config()
            if config.gemini is None or not config.gemini.api_key:
                msg = "Gemini API key missing from config file (required by PerEventCaptionStage backend=gemini)."
                raise RuntimeError(msg)
            self._gemini_api_key = config.gemini.api_key
        elif self._backend == "qwen":
            # Construct eagerly so the framework sees it via ``self.model``.
            self._qwen_model = QwenVL(
                model_variant=self._qwen_variant,
                fp8=self._qwen_fp8,
                max_output_tokens=self._qwen_max_output_tokens,
            )

    @property
    def resources(self) -> CuratorStageResource:
        """Return resource requirements (GPU for Qwen / vllm_async, CPU for Gemini / OpenAI)."""
        if self._backend == "qwen":
            return CuratorStageResource(gpus=1.0)
        if self._backend == "vllm_async":
            assert self._vllm_async_config is not None
            return CuratorStageResource(cpus=1.0, gpus=self._vllm_async_config.total_gpus)
        return CuratorStageResource(cpus=1.0)

    @property
    def conda_env_name(self) -> str:
        """Run in the unified env (vllm + google-genai + openai all live there)."""
        return "unified"

    @property
    def model(self) -> ModelInterface | None:  # type: ignore[override]
        """Expose the relevant ModelInterface so weights are auto-downloaded."""
        if self._backend == "vllm_async":
            return self._vllm_async_model_iface
        return self._qwen_model

    def stage_setup(self) -> None:
        """Initialise the selected backend (inside the remote worker).

        ``super().stage_setup()`` invokes ``self.model.setup()`` when
        ``self.model`` is set — for the Qwen backend this constructs the
        vLLM engine. Gemini and OpenAI expose no model so the base call is
        a no-op and we just create the client. For ``vllm_async`` the
        framework downloads weights via ``_VllmAsyncModel`` but engine init
        happens here.
        """
        super().stage_setup()
        if self._backend == "qwen":
            assert self._qwen_model is not None
            self._qwen_utils = QwenUtils(model_variant=self._qwen_variant)
            self._qwen_utils.setup()
            self._apply_qwen_sampling_overrides()
        elif self._backend == "gemini":
            assert self._gemini_api_key is not None
            self._gemini_client = genai.Client(api_key=self._gemini_api_key)
        elif self._backend == "openai":
            self._setup_openai()
        elif self._backend == "vllm_async":
            self._setup_vllm_async()

    def _apply_qwen_sampling_overrides(self) -> None:
        """Mutate ``QwenVL.sampling_params`` in place to apply any overrides."""
        assert self._qwen_model is not None
        params = self._qwen_model.sampling_params
        if params is None:
            return
        changes: list[str] = []
        if self._qwen_temperature is not None:
            params.temperature = float(self._qwen_temperature)
            changes.append(f"temperature={params.temperature}")
        if self._qwen_top_p is not None:
            params.top_p = float(self._qwen_top_p)
            changes.append(f"top_p={params.top_p}")
        if self._qwen_top_k is not None:
            params.top_k = int(self._qwen_top_k)
            changes.append(f"top_k={params.top_k}")
        if changes:
            logger.info(f"[PerEventCaptionStage] Qwen sampling overrides applied: {', '.join(changes)}")

    def secondary_name(self) -> str:
        """Return the backend label for logging."""
        return self._backend

    # ------------------------------------------------------------------
    # Gemini backend
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_gemini_text(response: object) -> str:
        """Extract plain text from a Gemini response.

        Raises ``RuntimeError`` with diagnostic context (``finish_reason``,
        block reason, token usage) when no text was produced — most often
        caused by thinking tokens consuming the full output budget.
        """
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        candidates = getattr(response, "candidates", None) or []
        collected: list[str] = []
        finish_reasons: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None) or candidate
            parts = getattr(content, "parts", None) or []
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    collected.append(part_text.strip())
            fr = getattr(candidate, "finish_reason", None)
            if fr is not None:
                finish_reasons.append(str(fr))
        result = "\n".join(collected).strip()
        if result:
            return result
        usage = getattr(response, "usage_metadata", None)
        block = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
        msg = (
            f"Gemini response contained no text "
            f"(finish_reasons={finish_reasons or 'n/a'}, block_reason={block!s}, usage={usage!s})"
        )
        raise RuntimeError(msg)

    def _with_video_metadata(self, part: "genai_types.Part") -> "genai_types.Part":
        """Attach the configured ingest-fps to a built Part (no-op on old SDKs)."""
        video_metadata_cls = getattr(genai_types, "VideoMetadata", None)
        if video_metadata_cls is None:
            return part
        try:
            video_metadata = video_metadata_cls(fps=self._gemini_video_fps)
        except (TypeError, ValueError):
            logger.warning(
                f"[PerEventCaptionStage] VideoMetadata(fps={self._gemini_video_fps}) not supported "
                "by installed google-genai; falling back to Gemini's default sampling fps."
            )
            return part
        # ``Part`` is effectively immutable; rebuild preserving the source.
        kwargs: dict[str, Any] = {"video_metadata": video_metadata}
        if getattr(part, "inline_data", None) is not None:
            kwargs["inline_data"] = part.inline_data
        if getattr(part, "file_data", None) is not None:
            kwargs["file_data"] = part.file_data
        return genai_types.Part(**kwargs)

    def _build_inline_video_part(self, clip_mp4_bytes: bytes) -> "genai_types.Part":
        """Build a video ``Part`` that embeds the mp4 bytes inline (≤20 MB)."""
        inline_data = genai_types.Blob(data=clip_mp4_bytes, mime_type="video/mp4")
        return self._with_video_metadata(genai_types.Part(inline_data=inline_data))

    def _build_uploaded_video_part(self, clip_mp4_bytes: bytes, clip_uuid: str) -> "genai_types.Part":
        """Upload via the Files API and reference by URI (for clips > 20 MB).

        Uploaded files are GC'd server-side after 48 h, so we don't delete.
        """
        if self._gemini_client is None:
            msg = "Gemini client not initialised; call stage_setup first."
            raise RuntimeError(msg)
        # Use a memfd-backed path so we don't touch the real filesystem.
        with buffer_as_memfd_path(clip_mp4_bytes, name=f"per-event-caption-{clip_uuid}") as mp4_path:
            uploaded = self._gemini_client.files.upload(
                file=str(mp4_path),
                config={"mime_type": "video/mp4"},
            )
        file_uri = getattr(uploaded, "uri", None)
        if not isinstance(file_uri, str) or not file_uri:
            msg = f"Gemini Files API upload did not return a usable URI (got {uploaded!r})"
            raise RuntimeError(msg)
        file_data = genai_types.FileData(file_uri=file_uri, mime_type="video/mp4")
        return self._with_video_metadata(genai_types.Part(file_data=file_data))

    def _build_video_part(self, clip_mp4_bytes: bytes, clip_uuid: str) -> "genai_types.Part":
        """Route inline vs. Files API based on ``_gemini_max_video_size_bytes``."""
        if len(clip_mp4_bytes) <= self._gemini_max_video_size_bytes:
            return self._build_inline_video_part(clip_mp4_bytes)
        logger.info(
            f"[PerEventCaptionStage] clip {clip_uuid}: mp4 bytes {len(clip_mp4_bytes)} exceed "
            f"inline limit {self._gemini_max_video_size_bytes}; uploading via Files API"
        )
        return self._build_uploaded_video_part(clip_mp4_bytes, clip_uuid)

    def _resolve_media_resolution(self) -> object | None:
        """Look up the configured ``MediaResolution`` enum value (``None`` if unsupported)."""
        enum_cls = getattr(genai_types, "MediaResolution", None)
        if enum_cls is None:
            return None
        name_map = {
            "low": "MEDIA_RESOLUTION_LOW",
            "medium": "MEDIA_RESOLUTION_MEDIUM",
            "high": "MEDIA_RESOLUTION_HIGH",
        }
        attr_name = name_map.get(self._gemini_media_resolution)
        if attr_name is None:
            return None
        return getattr(enum_cls, attr_name, None)

    def _call_gemini(self, clip_mp4_bytes: bytes, prompt: str, clip_uuid: str) -> str:
        client = self._gemini_client
        if client is None:
            msg = "Gemini client not initialized; call stage_setup first."
            raise RuntimeError(msg)

        content = genai_types.Content(
            parts=[
                self._build_video_part(clip_mp4_bytes, clip_uuid),
                genai_types.Part(text=prompt),
            ]
        )
        config_kwargs: dict[str, Any] = {
            "max_output_tokens": self._gemini_max_output_tokens,
            "response_mime_type": "application/json",
        }
        media_resolution = self._resolve_media_resolution()
        if media_resolution is not None:
            config_kwargs["media_resolution"] = media_resolution
        thinking_cfg_cls = getattr(genai_types, "ThinkingConfig", None)
        if thinking_cfg_cls is not None:
            try:
                config_kwargs["thinking_config"] = thinking_cfg_cls(thinking_budget=self._gemini_thinking_budget)
            except (TypeError, ValueError):
                # Older SDKs reject ``-1`` (dynamic); fall back to disabled.
                logger.warning(
                    f"[PerEventCaptionStage] thinking_budget={self._gemini_thinking_budget} "
                    "not supported by installed google-genai; falling back to 0 (thinking disabled)."
                )
                config_kwargs["thinking_config"] = thinking_cfg_cls(thinking_budget=0)
        generate_kwargs: dict[str, Any] = {
            "model": self._gemini_model_name,
            "contents": content,
            "config": genai_types.GenerateContentConfig(**config_kwargs),
        }

        # Retry 5xx / 429 / transport hiccups only; leave 4xx auth/bad-prompt
        # errors to fail fast.
        def _is_transient(exc: BaseException) -> bool:
            from google.genai import errors as genai_errors  # noqa: PLC0415

            if isinstance(exc, genai_errors.ServerError):
                return True
            code = getattr(exc, "code", None)
            if isinstance(code, int) and code in (408, 429, 500, 502, 503, 504):
                return True
            name = type(exc).__name__
            return name in {"APIError", "ConnectionError", "TimeoutError", "ReadTimeout"}

        def _log_retry(retry_state: "tenacity.RetryCallState") -> None:
            outcome = retry_state.outcome
            exc = outcome.exception() if outcome is not None else None
            next_wait = getattr(retry_state.next_action, "sleep", "?") if retry_state.next_action else "?"
            logger.warning(
                f"[PerEventCaptionStage] gemini call attempt "
                f"{retry_state.attempt_number}/{self._gemini_max_retries} failed "
                f"({type(exc).__name__ if exc else 'unknown'}: {exc}); "
                f"sleeping {next_wait}s before retry"
            )

        # Exponential backoff + uniform jitter so concurrent workers don't
        # retry in lockstep into the same overloaded server wave.
        @tenacity.retry(
            stop=tenacity.stop_after_attempt(self._gemini_max_retries),
            wait=tenacity.wait_exponential(
                multiplier=self._gemini_retry_delay_seconds,
                max=self._gemini_retry_delay_max_seconds,
            )
            + tenacity.wait_random(0, self._gemini_retry_jitter_seconds),
            retry=tenacity.retry_if_exception(_is_transient),
            reraise=True,
            before_sleep=_log_retry,
        )
        def _call() -> str:
            response = client.models.generate_content(**generate_kwargs)
            text = self._extract_gemini_text(response)
            # Stash diagnostics for ``_log_empty_events`` to surface on failure.
            self._last_gemini_finish_reasons = [
                str(getattr(c, "finish_reason", "")) for c in (getattr(response, "candidates", None) or [])
            ]
            self._last_gemini_usage_metadata = getattr(response, "usage_metadata", None)
            return text

        return _call()

    # ------------------------------------------------------------------
    # OpenAI-compatible backend
    # ------------------------------------------------------------------

    def _setup_openai(self) -> None:
        """Construct sync + async OpenAI clients from the project config."""
        from cosmos_curator.core.utils.config.config import maybe_load_config  # noqa: PLC0415

        config = maybe_load_config()
        endpoint = (
            getattr(config.openai, self._openai_endpoint_key, None)
            if config is not None and config.openai is not None
            else None
        )
        if endpoint is None or not endpoint.api_key:
            msg = (
                f"OpenAI {self._openai_endpoint_key} configuration not found. "
                f"Provide openai.{self._openai_endpoint_key}.api_key in "
                "~/.config/cosmos_curator/config.yaml (required by "
                "PerEventCaptionStage backend=openai)."
            )
            raise RuntimeError(msg)

        self._openai_client, self._openai_model_name = create_openai_client_and_resolve_model(
            openai,
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            model_name=self._openai_model_name,
            endpoint_label=f"OpenAI {self._openai_endpoint_key}",
        )
        client_kwargs: dict[str, Any] = {"api_key": endpoint.api_key}
        if endpoint.base_url:
            client_kwargs["base_url"] = endpoint.base_url
        self._openai_async_client = openai.AsyncOpenAI(**client_kwargs)
        self._openai_runner = asyncio.Runner()

    async def _generate_openai_caption_async(
        self,
        clip_mp4_bytes: bytes,
        prompt: str,
    ) -> tuple[CaptionResult, str | None]:
        """Generate a caption result for one clip via the async OpenAI client."""
        client = self._openai_async_client
        if client is None:
            msg = "OpenAI async client not initialised; call stage_setup before generating captions."
            raise RuntimeError(msg)

        video_b64 = base64.b64encode(bytes(clip_mp4_bytes)).decode("utf-8")
        content_parts: list[dict[str, Any]] = [
            {
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
            },
            {"type": "text", "text": prompt},
        ]
        request_kwargs: dict[str, Any] = {
            "model": self._openai_model_name,
            "messages": [{"role": "user", "content": content_parts}],
            "max_tokens": self._openai_max_output_tokens,
        }

        async def _call() -> object:
            async for attempt in tenacity.AsyncRetrying(
                stop=tenacity.stop_after_attempt(self._openai_max_retries),
                wait=tenacity.wait_fixed(self._openai_retry_delay_seconds),
                retry=tenacity.retry_if_not_exception_type(
                    (openai.AuthenticationError, openai.NotFoundError, openai.BadRequestError),
                ),
                reraise=True,
            ):
                with attempt:
                    return await client.chat.completions.create(**request_kwargs)
            msg = "OpenAI async retry loop exited without a result."
            raise RuntimeError(msg)

        try:
            response = await _call()
        except Exception as exc:  # noqa: BLE001
            timeout_error = getattr(openai, "APITimeoutError", None)
            return openai_error_result_from_exception(exc, timeout_error_type=timeout_error)
        return normalize_openai_response_with_detail(response)

    def _call_openai(self, clip_mp4_bytes: bytes, prompt: str) -> str:
        """Drive one OpenAI clip request synchronously and return caption text.

        Raises ``RuntimeError`` when the API returns no usable caption text
        (caller logs the failure on ``clip.errors``).
        """
        if self._openai_runner is None:
            msg = "OpenAI runner not initialised; call stage_setup before generating captions."
            raise RuntimeError(msg)
        result, detail = self._openai_runner.run(self._generate_openai_caption_async(clip_mp4_bytes, prompt))
        if result.outcome == CaptionOutcome.BLOCKED:
            msg = "OpenAI request blocked by content filter."
            raise RuntimeError(msg)
        if result.text is None:
            msg = detail or f"OpenAI request produced no caption text (outcome={result.outcome.value!r})."
            raise RuntimeError(msg)
        return result.text

    # ------------------------------------------------------------------
    # vllm_async backend
    # ------------------------------------------------------------------

    # Mirrors VllmAsyncCaptionStage._UNSET_VLLM_ENV_VARS — both vars are
    # legacy sync-vLLM defaults that confuse the AsyncLLM engine.
    _UNSET_VLLM_ENV_VARS: tuple[str, ...] = (
        "VLLM_ATTENTION_BACKEND",
        "VLLM_WORKER_MULTIPROC_METHOD",
    )

    def _configure_vllm_async_environment(self) -> None:
        """Apply the minimal env-var hygiene needed by the AsyncLLM engine.

        Lighter than ``VllmAsyncCaptionStage._configure_vllm_environment`` —
        the per-event stage runs at most one engine.generate per clip and
        does not need the full ``RuntimeEnv`` plumbing or per-actor logging
        prefix machinery.
        """
        for var in self._UNSET_VLLM_ENV_VARS:
            stale = os.environ.pop(var, None)
            if stale is not None and self._verbose:
                logger.info(f"[PerEventCaptionStage] removed stale env var {var}={stale}")
        # Steer vLLM caches off potentially-slow NFS home directories.
        os.environ.setdefault("VLLM_CACHE_ROOT", "/tmp/vllm")  # noqa: S108

    def _setup_vllm_async(self) -> None:
        """Construct the in-process AsyncLLM engine for per-clip generation."""
        config = self._vllm_async_config
        if config is None:
            msg = "vllm_async_config not set; backend='vllm_async' requires a config."
            raise RuntimeError(msg)

        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)
        self._configure_vllm_async_environment()

        model_id = get_vllm_model_id(config.model_variant)
        model_path = resolve_model_path(model_id)
        self._vllm_async_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)  # type: ignore[no-untyped-call]

        engine_args = _build_engine_args(config, model_path)
        logger.info(
            f"[PerEventCaptionStage] booting AsyncLLM engine "
            f"variant={config.model_variant!r} num_gpus={config.num_gpus} "
            f"data_parallel_size={config.data_parallel_size} "
            f"distributed_executor_backend={config.distributed_executor_backend!r}"
        )
        self._vllm_async_engine = AsyncLLM.from_engine_args(engine_args)

        params = build_sampling_params(config.sampling_config)
        params.max_tokens = self._vllm_async_max_output_tokens
        self._vllm_async_sampling_params = params
        # Per-clip generate is sync-shaped from the framework's POV.
        self._vllm_async_runner = asyncio.Runner()

        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=False)

    def _build_vllm_async_prompt(self, prompt: str) -> str:
        """Render the chat-template prompt for one clip via the AutoProcessor."""
        processor = self._vllm_async_processor
        if processor is None:
            msg = "vllm_async processor not initialised; call stage_setup first."
            raise RuntimeError(msg)
        messages: list[dict[str, str | list[dict[str, str]]]] = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": prompt.strip()},
                ],
            },
        ]
        prompt_text: str = processor.apply_chat_template(  # type: ignore[attr-defined]
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt_text

    def _decode_vllm_async_frames(self, clip_mp4_bytes: bytes) -> "np.ndarray[Any, np.dtype[np.uint8]]":
        """Decode the annotated mp4 to a uint8 [T, H, W, 3] frame stack.

        Mirrors ``VllmAsyncPrepStage._create_windows_and_decode`` for a
        single window covering the whole clip.
        """
        total_native = get_frame_count(clip_mp4_bytes)
        if total_native <= 0:
            msg = "clip has 0 decodable frames"
            raise RuntimeError(msg)

        with buffer_as_memfd_path(clip_mp4_bytes, name="per-event-vllm-async-clip") as video_path:
            native_fps = get_avg_frame_rate(video_path)
            n_sampled = smart_nframes(self._vllm_async_sampling_fps, total_native, native_fps)
            indices = np.linspace(0, total_native - 1, n_sampled, dtype=np.int32)
            return decode_video_cpu_frame_ids(video_path, indices, num_threads=2)

    async def _generate_vllm_async_caption(
        self,
        rendered_prompt: object,
        request_id: str,
    ) -> str:
        """Drive a single ``engine.generate`` async iterator to completion."""
        engine = self._vllm_async_engine
        params = self._vllm_async_sampling_params
        if engine is None or params is None:
            msg = "vllm_async engine not initialised; call stage_setup first."
            raise RuntimeError(msg)
        final_output = None
        async for output in engine.generate(
            prompt=rendered_prompt,  # type: ignore[arg-type]
            sampling_params=params,
            request_id=request_id,
        ):
            final_output = output
        if final_output is None or not final_output.outputs:
            msg = "AsyncLLM engine returned no outputs."
            raise RuntimeError(msg)
        text = final_output.outputs[0].text
        if not text or not text.strip():
            msg = f"AsyncLLM engine returned empty caption (finish_reason={final_output.outputs[0].finish_reason!r})"
            raise RuntimeError(msg)
        return str(text).strip()

    def _call_vllm_async(self, clip_mp4_bytes: bytes, prompt: str) -> str:
        """Generate one per-event caption via the in-process AsyncLLM engine."""
        if self._vllm_async_runner is None or self._vllm_async_engine is None:
            msg = "vllm_async backend not initialised; call stage_setup first."
            raise RuntimeError(msg)
        config = self._vllm_async_config
        assert config is not None

        prompt_text = self._build_vllm_async_prompt(prompt)
        frames = self._decode_vllm_async_frames(clip_mp4_bytes)
        payload = _build_render_payload(prompt_text, frames, config.mm_processor_kwargs)
        rendered_list = self._vllm_async_engine.renderer.render_cmpl([payload])
        rendered_prompt = rendered_list[0]
        self._vllm_async_request_counter += 1
        request_id = f"per-event-{self._vllm_async_request_counter}"
        return self._vllm_async_runner.run(self._generate_vllm_async_caption(rendered_prompt, request_id))

    # ------------------------------------------------------------------
    # Qwen backend
    # ------------------------------------------------------------------

    def _call_qwen(self, clip_mp4_bytes: bytes, prompt: str) -> str:
        qwen_utils = self._qwen_utils
        qwen_model = self._qwen_model
        if qwen_utils is None or qwen_model is None:
            msg = "Qwen backend not initialized; call stage_setup first."
            raise RuntimeError(msg)

        # Decode mp4 → frame tensor via a memfd-backed path (kept in RAM).
        total_frames = get_frame_count(clip_mp4_bytes)
        if total_frames <= 0:
            msg = "clip has 0 decodable frames"
            raise RuntimeError(msg)
        window_range = [WindowFrameInfo(start=0, end=total_frames - 1)]
        # Variants in ``QWEN_VARIANTS_NEED_RAW_FRAMES`` (Qwen3-VL family)
        # want raw uint8 TCHW + HF's own video processor; everything else
        # uses our 28-aligned float16 ``fetch_video`` path (bfloat16 crashes
        # vLLM's numpy path).
        needs_raw_frames = self._qwen_variant in QWEN_VARIANTS_NEED_RAW_FRAMES
        with buffer_as_memfd_path(clip_mp4_bytes, name="per-event-caption-clip") as path:
            if needs_raw_frames:
                video_tensor, _frame_counts = read_video_cpu(
                    path,
                    self._qwen_sampling_fps,
                    0,
                    window_range,
                )
            else:
                video_tensor, _frame_counts = fetch_video(
                    path,
                    sampling_fps=self._qwen_sampling_fps,
                    window_range=window_range,
                    do_preprocess=True,
                    preprocess_dtype="float16",
                )

        # Qwen3-VL requires HF-style video metadata; Qwen2.5-VL ignores it.
        # We pre-sample here so the processor skips its own sampling step.
        num_sampled_frames = int(video_tensor.shape[0]) if video_tensor.ndim >= 1 else 0
        duration_s = float(num_sampled_frames) / self._qwen_sampling_fps if self._qwen_sampling_fps > 0 else 0.0
        video_metadata = {
            "total_num_frames": num_sampled_frames,
            "fps": float(self._qwen_sampling_fps),
            "duration": duration_s,
            "video_backend": "opencv_dynamic",
            "frames_indices": list(range(num_sampled_frames)),
            "do_sample_frames": False,
        }

        llm_input = qwen_utils.generate_llm_inputs(
            prompt=prompt,
            video_inputs=video_tensor,
            video_metadata=video_metadata,
        )
        outputs = qwen_model.generate([llm_input], generate_stage2_caption=False, batch_size=1)
        if not outputs:
            return ""
        return str(outputs[0])

    # ------------------------------------------------------------------
    # Common orchestration
    # ------------------------------------------------------------------

    def _process_clip(self, clip: Clip) -> None:  # noqa: C901  # 4-way backend dispatch + skip / retry guards
        if not clip.sam3_instances:
            if self._verbose:
                logger.debug(f"[PerEventCaptionStage] clip {clip.uuid}: no SAM3 instances; skipping")
            return

        # The annotated ``tracked.mp4`` is the only video the VLM sees — the
        # ``#id`` overlay is load-bearing for the prompt. Fail fast rather
        # than silently falling back to the raw clip if SAM3 skipped it.
        annotated = clip.sam3_annotated_video.resolve()
        if annotated is None:
            clip.errors["per_event_caption"] = "missing_annotated_video"
            logger.warning(f"[PerEventCaptionStage] clip {clip.uuid}: sam3_annotated_video missing; skipping")
            return
        mp4_bytes = annotated.tobytes()
        if self._verbose:
            logger.debug(
                f"[PerEventCaptionStage] clip {clip.uuid}: feeding annotated video to VLM ({len(mp4_bytes)} bytes)"
            )

        prompt = _build_prompt(self._prompt_template, clip)
        try:
            if self._backend == "qwen":
                raw = self._call_qwen(mp4_bytes, prompt)
            elif self._backend == "gemini":
                raw = self._call_gemini(mp4_bytes, prompt, str(clip.uuid))
            elif self._backend == "openai":
                raw = self._call_openai(mp4_bytes, prompt)
            else:  # vllm_async
                raw = self._call_vllm_async(mp4_bytes, prompt)
        except Exception as exc:  # noqa: BLE001
            clip.errors["per_event_caption"] = f"api_error: {exc!r}"
            logger.exception(f"[PerEventCaptionStage] clip {clip.uuid}: {self._backend} call failed")
            return

        events = _extract_events_payload(raw)
        clip.sam3_events = events
        if not events:
            self._log_empty_events(clip, raw)
        elif self._verbose:
            logger.info(f"[PerEventCaptionStage] clip {clip.uuid}: {len(events)} events (backend={self._backend})")

    @staticmethod
    def _looks_truncated(raw: str) -> bool:
        """Heuristic: response got chopped mid-JSON (usually MAX_TOKENS)."""
        stripped = raw.strip().rstrip("` \n")
        if not stripped:
            return False
        return stripped[-1] not in {"}", "]"}

    def _log_empty_events(self, clip: Clip, raw: str) -> None:
        """Log the raw model response when no events were parsed."""
        preview = raw.strip().replace("\n", " ")
        if len(preview) > _RAW_PREVIEW_MAX_CHARS:
            preview = preview[:_RAW_PREVIEW_MAX_CHARS] + "...[truncated]"
        truncated = self._looks_truncated(raw)
        diag_parts = [f"raw_len={len(raw)}"]
        if self._backend == "gemini":
            if self._last_gemini_finish_reasons:
                diag_parts.append(f"finish_reasons={self._last_gemini_finish_reasons}")
            if self._last_gemini_usage_metadata is not None:
                diag_parts.append(f"usage={self._last_gemini_usage_metadata!s}")
        diagnostics = ", ".join(diag_parts)
        hint = ""
        if truncated:
            hint = (
                " [likely MAX_TOKENS truncation — raise --event-caption-gemini-max-output-tokens "
                "or lower --event-caption-gemini-thinking-budget]"
            )
        logger.warning(
            f"[PerEventCaptionStage] clip {clip.uuid}: extracted 0 events from "
            f"{self._backend} response ({diagnostics}){hint}: {preview!r}"
        )
        clip.errors.setdefault(
            "per_event_caption",
            "truncated_response" if truncated else "empty_or_unparseable_response",
        )

    def destroy(self) -> None:
        """Tear down per-backend clients / engines on actor stop.

        Sync ``qwen`` and ``gemini`` paths hold no resources that need
        explicit teardown beyond garbage collection. ``openai`` mirrors
        ``OpenAICaptionStage.destroy``; ``vllm_async`` mirrors
        ``VllmAsyncCaptionStage.destroy`` while keeping
        ``gpu_stage_cleanup`` in a ``finally`` so GPU memory is reliably
        released even if the engine shutdown raises.
        """
        if self._backend == "openai":
            destroy_api_clients(
                async_client=self._openai_async_client,
                runner=self._openai_runner,
                sync_client=self._openai_client,
            )
            self._openai_async_client = None
            self._openai_runner = None
            self._openai_client = None
        elif self._backend == "vllm_async":
            try:
                if self._vllm_async_engine is not None:
                    logger.info("[PerEventCaptionStage] destroy: shutting down AsyncLLM engine")
                    self._vllm_async_engine.shutdown()  # type: ignore[no-untyped-call]
                    self._vllm_async_engine = None
                self._vllm_async_processor = None
                if self._vllm_async_runner is not None:
                    self._vllm_async_runner.close()
                    self._vllm_async_runner = None
            finally:
                gpu_stage_cleanup(self.__class__.__name__)

    @nvtx.annotate("PerEventCaptionStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask]:  # type: ignore[override]
        """Generate per-event captions for every clip in every task."""
        for task in tasks:
            major_size = task.get_major_size()
            self._timer.reinit(self, major_size)
            with self._timer.time_process():
                for video in task.videos:
                    for clip in video.clips:
                        self._process_clip(clip)

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats
        return tasks
