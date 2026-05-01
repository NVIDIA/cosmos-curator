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

Two backends are available:

* ``qwen`` (default) — local Qwen2.5-VL via vLLM. Requires a GPU and the
  ``unified`` conda environment. Fully offline, no API key.
* ``gemini`` — remote Gemini API. CPU-only stage, but requires a
  ``gemini.api_key`` entry in the project config file and internet access.

Switch between them via the ``backend`` constructor argument (or
``--event-caption-backend`` on the pipeline CLIs).
"""

import json
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
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask

# ``google-genai``, ``tenacity``, and ``vision_process`` (torchvision) only
# live in the ``unified`` env; guard imports so this module loads elsewhere.
if conda_utils.is_running_in_env("unified") or TYPE_CHECKING:
    import tenacity
    from google import genai
    from google.genai import types as genai_types

    from cosmos_curator.pipelines.video.utils.decoder_utils import get_frame_count
    from cosmos_curator.pipelines.video.utils.vision_process import fetch_video, read_video_cpu
    from cosmos_curator.pipelines.video.utils.windowing_utils import WindowFrameInfo


Backend = Literal["qwen", "gemini"]


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

    def __init__(  # noqa: PLR0913  # flat config surface keeps CLI wiring straightforward
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
        # --- Common ---
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialise the stage.

        Args:
            backend: ``"qwen"`` (local GPU) or ``"gemini"`` (remote API).
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
            verbose: Emit per-clip logs.
            log_stats: Record stage performance stats.

        """
        super().__init__()
        if backend not in ("qwen", "gemini"):
            msg = f"backend must be 'qwen' or 'gemini', got {backend!r}"
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

        if self._backend == "gemini":
            from cosmos_curator.core.utils.config.config import load_config  # noqa: PLC0415

            config = load_config()
            if config.gemini is None or not config.gemini.api_key:
                msg = "Gemini API key missing from config file (required by PerEventCaptionStage backend=gemini)."
                raise RuntimeError(msg)
            self._gemini_api_key = config.gemini.api_key
        else:
            # Construct eagerly so the framework sees it via ``self.model``.
            self._qwen_model = QwenVL(
                model_variant=self._qwen_variant,
                fp8=self._qwen_fp8,
                max_output_tokens=self._qwen_max_output_tokens,
            )

    @property
    def resources(self) -> CuratorStageResource:
        """Return resource requirements (GPU for Qwen, CPU for Gemini)."""
        if self._backend == "qwen":
            return CuratorStageResource(gpus=1.0)
        return CuratorStageResource(cpus=1.0)

    @property
    def conda_env_name(self) -> str:
        """Run in the unified env (vllm + google-genai both live there)."""
        return "unified"

    @property
    def model(self) -> ModelInterface | None:  # type: ignore[override]
        """Expose the Qwen model so the framework downloads weights / drives setup()."""
        return self._qwen_model

    def stage_setup(self) -> None:
        """Initialise the selected backend (inside the remote worker).

        ``super().stage_setup()`` invokes ``self.model.setup()`` when
        ``self.model`` is set — for the Qwen backend this constructs the
        vLLM engine. Gemini exposes no model so the base call is a no-op
        and we just create the client.
        """
        super().stage_setup()
        if self._backend == "qwen":
            assert self._qwen_model is not None
            self._qwen_utils = QwenUtils(model_variant=self._qwen_variant)
            self._qwen_utils.setup()
            self._apply_qwen_sampling_overrides()
        else:
            assert self._gemini_api_key is not None
            self._gemini_client = genai.Client(api_key=self._gemini_api_key)

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
        window_range = [WindowFrameInfo(start=0, end=total_frames)]
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

    def _process_clip(self, clip: Clip) -> None:
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
            else:
                raw = self._call_gemini(mp4_bytes, prompt, str(clip.uuid))
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
