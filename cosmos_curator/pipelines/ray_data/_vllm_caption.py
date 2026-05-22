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

"""Ray Data vLLM captioning helpers.

The public Ray Data pipeline uses Ray's vLLM processor for scheduling,
continuous batching, and engine lifecycle, but intentionally reuses the existing
Xenna Qwen preparation path for model-specific prompt and multimodal input
construction. Rows carry decoded video frames as Arrow-friendly bytes, shape,
and dtype columns until the Ray vLLM engine actor rebuilds the nested
``multimodal_data`` request immediately before inference.
"""

import asyncio
import json
import logging
import sys
import time
import types
import uuid
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any, Protocol, cast

import attrs
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import ray

from cosmos_curator.core.utils.model.model_utils import get_local_dir_for_weights_name
from cosmos_curator.core.utils.pixi_runtime_envs import PixiRuntimeEnv, ray_data_gpu_runtime_env
from cosmos_curator.core.utils.storage.storage_utils import StorageWriter
from cosmos_curator.models.vllm_model_ids import get_vllm_model_id
from cosmos_curator.pipelines.ray_data._summary_writer import write_summary_from_rows
from cosmos_curator.pipelines.video.utils.data_model import VllmConfig, VllmSamplingConfig, WindowConfig

if TYPE_CHECKING:
    from transformers import AutoProcessor

logger = logging.getLogger(__name__)

_MODEL_VARIANT = "qwen"
_OK_CAPTION_STATUSES = {"success", "truncated"}
_PROCESSOR_CACHE: dict[str, object] = {}
_CAPTION_METADATA_WRITER_CACHE: dict[str, StorageWriter] = {}
_VIDEO_TUPLE_LENGTH = 2


class _VllmInputsModule(Protocol):
    TextPrompt: object
    TokensPrompt: object
    data: types.ModuleType


class _RayDataLlmStage(Protocol):
    fn: type[object]
    map_batches_kwargs: dict[str, Any]


class _RayDataLlmProcessor(Protocol):
    def get_stage_by_name(self, name: str) -> _RayDataLlmStage: ...


def make_default_vllm_config() -> VllmConfig:
    """Build the Ray Data Qwen config."""
    return VllmConfig(model_variant=_MODEL_VARIANT, preprocess=False, num_gpus=1, batch_size=32)


def make_default_window_config() -> WindowConfig:
    """Build the first-version Qwen window config to match Xenna defaults."""
    return WindowConfig(model_does_preprocess=False, preprocess_dtype="float16")


def qwen_model_id() -> str:
    """Return the Qwen model ID used by the Ray Data captioning path."""
    return get_vllm_model_id(_MODEL_VARIANT)


def qwen_model_source() -> str:
    """Return the Xenna local-cache model path for Qwen."""
    return str(get_local_dir_for_weights_name(qwen_model_id()))


def _max_caption_workers(total_visible_gpus: int, num_gpus_per_worker: int) -> int:
    """Return the maximum vLLM replicas this fixed Ray cluster can run."""
    if num_gpus_per_worker <= 0:
        msg = f"num_gpus_per_worker must be positive, got {num_gpus_per_worker}"
        raise ValueError(msg)
    if total_visible_gpus <= 0:
        return 0
    return total_visible_gpus // num_gpus_per_worker


def sampling_params_dict(config: VllmSamplingConfig | None = None) -> dict[str, Any]:
    """Convert ``VllmSamplingConfig`` into Ray processor row sampling params."""
    return attrs.asdict(config or VllmSamplingConfig())


def _patch_vllm_inputs_data_namespace() -> None:
    """Patch Ray's expected vLLM prompt namespace for newer vLLM versions.

    Ray's vLLM batch stage calls ``vllm.inputs.data.TextPrompt`` and
    ``TokensPrompt``. The pinned vLLM exposes those classes from ``vllm.inputs``.
    Keep this compatibility local to the Ray Data captioning actor.
    """
    import vllm.inputs as vllm_inputs  # noqa: PLC0415

    if hasattr(vllm_inputs, "data"):
        return

    vllm_inputs_module = cast("_VllmInputsModule", vllm_inputs)
    data_module = types.ModuleType("vllm.inputs.data")
    data_module.__dict__["TextPrompt"] = vllm_inputs_module.TextPrompt
    data_module.__dict__["TokensPrompt"] = vllm_inputs_module.TokensPrompt
    sys.modules["vllm.inputs.data"] = data_module
    vllm_inputs_module.data = data_module


def _to_numpy_tensor(value: object) -> npt.NDArray[np.generic]:
    """Convert model-frame tensors to contiguous numpy-compatible arrays."""
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    if hasattr(value, "detach"):
        tensor = cast("Any", value).detach().cpu().contiguous()
        if str(tensor.dtype) == "torch.bfloat16":
            tensor = tensor.float()
        return np.ascontiguousarray(cast("npt.NDArray[np.generic]", tensor.numpy()))
    return np.ascontiguousarray(np.asarray(value))


def _add_video_frame_payload(row: dict[str, Any], frames: object) -> None:
    """Store video frames using Arrow-native fields before Ray's LLM wrapper."""
    array = _to_numpy_tensor(frames)
    row["video_frame_bytes"] = pa.scalar(array.tobytes(), type=pa.large_binary())
    row["video_frame_shape"] = list(array.shape)
    row["video_frame_dtype"] = str(array.dtype)


def _add_ray_multimodal_columns(row: dict[str, Any], multimodal_data: dict[str, Any]) -> None:
    """Store vLLM video inputs as Arrow-native columns for Ray Data."""
    video_items = multimodal_data.get("video")
    if not isinstance(video_items, list) or len(video_items) != 1:
        msg = "Ray Data captioning expects exactly one video item per request"
        raise ValueError(msg)

    video_item = video_items[0]
    metadata = None
    if isinstance(video_item, tuple):
        if len(video_item) != _VIDEO_TUPLE_LENGTH:
            msg = "Ray Data captioning expects video tuple inputs as (frames, metadata)"
            raise ValueError(msg)
        frames, metadata = video_item
    else:
        frames = video_item

    _add_video_frame_payload(row, frames)
    if metadata is not None:
        row["video_metadata"] = metadata


def _add_ray_llm_columns(
    row: dict[str, Any],
    model_input: dict[str, Any],
    sampling_params: dict[str, Any],
) -> None:
    """Adapt one Xenna-style vLLM input to Ray's vLLM engine row contract."""
    token_ids = model_input.get("prompt_token_ids")
    prompt = model_input.get("prompt")
    if token_ids is not None:
        row["tokenized_prompt"] = token_ids
    elif prompt is not None:
        row["prompt"] = prompt
    else:
        msg = "vLLM model input must include prompt_token_ids or prompt"
        raise ValueError(msg)

    if "multi_modal_data" in model_input:
        _add_ray_multimodal_columns(row, model_input["multi_modal_data"])
    elif "multimodal_data" in model_input:
        _add_ray_multimodal_columns(row, model_input["multimodal_data"])

    if "mm_processor_kwargs" in model_input:
        row["mm_processor_kwargs"] = model_input["mm_processor_kwargs"]
    if "multi_modal_uuids" in model_input:
        row["multimodal_uuids"] = model_input["multi_modal_uuids"]
    elif "multimodal_uuids" in model_input:
        row["multimodal_uuids"] = model_input["multimodal_uuids"]

    row["sampling_params"] = dict(sampling_params)


def _assemble_ray_multimodal_data(row: dict[str, Any]) -> None:
    """Rebuild vLLM multimodal input from Ray Data frame payload columns."""
    if "video_frame_bytes" not in row:
        return

    frame_bytes = row.pop("video_frame_bytes")
    if hasattr(frame_bytes, "as_py"):
        frame_bytes = cast("Any", frame_bytes).as_py()
    if not isinstance(frame_bytes, bytes | bytearray | memoryview):
        msg = f"Expected video_frame_bytes to be a bytes-like value, got {type(frame_bytes).__name__}"
        raise TypeError(msg)

    shape = tuple(int(dim) for dim in row.pop("video_frame_shape"))
    dtype = np.dtype(cast("str", row.pop("video_frame_dtype")))
    expected_nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    actual_nbytes = len(frame_bytes)
    if actual_nbytes != expected_nbytes:
        msg = (
            "Video frame payload byte size does not match shape and dtype: "
            f"actual={actual_nbytes} expected={expected_nbytes} shape={shape} dtype={dtype}"
        )
        raise ValueError(msg)

    frames = np.frombuffer(frame_bytes, dtype=dtype).reshape(shape).copy()
    metadata = row.pop("video_metadata", None)
    video_item = (frames, metadata) if metadata is not None else frames
    row["multimodal_data"] = {"video": [video_item]}


def _arrow_data_column_to_rows(batch: pa.Table, data_column: str) -> list[dict[str, Any]]:
    """Extract processor payload rows from an Arrow struct column without combining chunks."""
    if data_column not in batch.column_names:
        msg = f"[Internal] {data_column} not found in batch {batch}"
        raise ValueError(msg)

    rows: list[dict[str, Any]] = []
    for chunk in batch.column(data_column).chunks:
        for row in chunk.to_pylist():
            if not isinstance(row, dict):
                msg = f"[Internal] {data_column} row must be a dict, got {type(row).__name__}"
                raise TypeError(msg)
            rows.append(cast("dict[str, Any]", row))
    return rows


def _install_vllm_engine_stage_shim(processor: object) -> None:  # noqa: C901
    """Install local Ray/vLLM engine-stage fixes for captioning."""
    from ray.llm._internal.batch.stages.vllm_engine_stage import vLLMEngineStageUDF  # noqa: PLC0415

    try:
        stage = cast("_RayDataLlmProcessor", processor).get_stage_by_name("vLLMEngineStage")
    except Exception as exc:
        msg = "Ray vLLM processor no longer exposes a vLLMEngineStage stage"
        raise RuntimeError(msg) from exc
    if not isinstance(stage.fn, type) or not issubclass(stage.fn, vLLMEngineStageUDF):
        msg = f"Ray vLLMEngineStage uses unexpected UDF type {stage.fn!r}"
        raise TypeError(msg)

    class ArrowFramePayloadEngineStageUDF(vLLMEngineStageUDF):  # type: ignore[misc, valid-type]
        def __init__(self, *args: object, **kwargs: object) -> None:
            _patch_vllm_inputs_data_namespace()
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]

        async def __call__(self, batch: dict[str, Any] | pa.Table) -> AsyncIterator[dict[str, Any]]:
            if isinstance(batch, pa.Table):
                if batch.num_rows == 0:
                    yield {}
                    return
                batch = {self.data_column: _arrow_data_column_to_rows(batch, self.data_column)}

            inputs: object = batch.get(self.data_column)
            if hasattr(inputs, "tolist"):
                inputs = cast("Any", inputs).tolist()
            if isinstance(inputs, list) and all(isinstance(row, dict) and "caption_windows" in row for row in inputs):
                yield {self.data_column: await self._generate_clip_rows(cast("list[dict[str, Any]]", inputs))}
                return

            async for output in super().__call__(batch):
                yield output

        async def _generate_with_error_handling(self, row: dict[str, Any], batch_uuid: uuid.UUID) -> dict[str, Any]:
            try:
                _assemble_ray_multimodal_data(row)
                return await super()._generate_with_error_handling(row, batch_uuid)  # type: ignore[misc]
            finally:
                row.pop("multimodal_data", None)

        async def _generate_clip_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:  # noqa: C901
            batch_uuid = uuid.uuid4()
            batch_start_time = time.perf_counter()
            index_to_window: dict[int, tuple[int, int]] = {}
            normal_windows: list[dict[str, Any]] = []

            for clip_index, row in enumerate(rows):
                windows = row.get("caption_windows", [])
                if not isinstance(windows, list):
                    msg = f"caption_windows must be a list, got {type(windows).__name__}"
                    raise TypeError(msg)
                for window_index, window in enumerate(windows):
                    if not isinstance(window, dict):
                        msg = f"caption_windows entries must be dicts, got {type(window).__name__}"
                        raise TypeError(msg)
                    if window.get("caption_skip", False) or window.get("__inference_error__", "") != "":
                        continue
                    index_to_window[len(normal_windows)] = (clip_index, window_index)
                    normal_windows.append(window)

            self.validate_inputs(normal_windows)
            tasks = []
            for request_index, window in enumerate(normal_windows):
                window[self.IDX_IN_BATCH_COLUMN] = request_index
                tasks.append(asyncio.create_task(self._generate_with_error_handling(window, batch_uuid)))

            pending_windows = set(index_to_window)
            for task in asyncio.as_completed(tasks):
                output = await task
                request_index = output.pop(self.IDX_IN_BATCH_COLUMN)
                if request_index not in pending_windows:
                    msg = f"The window request {request_index} is output twice."
                    raise ValueError(msg)
                pending_windows.remove(request_index)
                clip_index, window_index = index_to_window[request_index]
                window = cast("dict[str, Any]", rows[clip_index]["caption_windows"][window_index])
                window.pop(self.IDX_IN_BATCH_COLUMN, None)
                window.update(output)

            if pending_windows:
                msg = f"The window requests {pending_windows} were not output."
                raise ValueError(msg)

            batch_time_taken = time.perf_counter() - batch_start_time
            logger.info(
                "[vLLM] Elapsed time for clip batch %s with %d clips and %d windows: %s",
                batch_uuid.hex,
                len(rows),
                len(normal_windows),
                batch_time_taken,
            )

            if normal_windows and not getattr(self, "engine_kwargs", {}).get("disable_log_stats", False):
                await self.llm.engine.do_log_stats()

            return rows

    stage.fn = ArrowFramePayloadEngineStageUDF
    stage.map_batches_kwargs["batch_format"] = "pyarrow"


def make_caption_window_rows_fn(
    vllm_config: VllmConfig | None = None,
    window_config: WindowConfig | None = None,
    sampling_config: VllmSamplingConfig | None = None,
) -> Callable[[dict[str, Any]], list[dict[str, Any]]]:
    """Create a ``flat_map`` function that emits one row per caption window."""
    resolved_vllm_config = vllm_config or make_default_vllm_config()
    resolved_window_config = window_config or make_default_window_config()
    resolved_sampling_params = sampling_params_dict(sampling_config or resolved_vllm_config.sampling_config)
    resolved_processor: object | None = None
    resolved_prompt: str | None = None

    def _get_processor() -> object:
        nonlocal resolved_processor
        if resolved_processor is None:
            resolved_processor = _auto_processor(resolved_vllm_config)
        return resolved_processor

    def _get_prompt() -> str:
        nonlocal resolved_prompt
        if resolved_prompt is None:
            from cosmos_curator.models.prompts import get_prompt  # noqa: PLC0415

            resolved_prompt = get_prompt(resolved_vllm_config.prompt_variant, resolved_vllm_config.prompt_text)
        return resolved_prompt

    def _make_window_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
        from cosmos_curator.models.vllm_interface import make_metadata, make_model_inputs  # noqa: PLC0415
        from cosmos_curator.pipelines.video.utils.windowing_utils import split_video_into_windows  # noqa: PLC0415

        _, window_frames, window_infos = split_video_into_windows(
            row["clip_bytes"],
            window_size=resolved_window_config.window_size,
            remainder_threshold=resolved_window_config.remainder_threshold,
            sampling_fps=resolved_window_config.sampling_fps,
            model_does_preprocess=resolved_window_config.model_does_preprocess,
            preprocess_dtype=resolved_window_config.preprocess_dtype,
            return_bytes=False,
            return_video_frames=True,
        )

        decoded_windows = []
        for window_index, (frame_tensor, window_info) in enumerate(
            zip(window_frames, window_infos, strict=True),
        ):
            if frame_tensor is None:
                logger.warning(
                    "Caption window %s for clip %s has no decoded frames",
                    window_index,
                    row["clip_uuid"],
                )
                continue
            decoded_windows.append((window_index, window_info, frame_tensor))

        if decoded_windows:
            frames = [frame for _, _, frame in decoded_windows]
            metadata = make_metadata(frames, resolved_window_config)
            model_inputs = make_model_inputs(
                frames,
                metadata,
                resolved_vllm_config,
                cast("AutoProcessor", _get_processor()),
                _get_prompt(),
            )
            output_rows: list[dict[str, Any]] = []
            for (window_index, window_info, _), model_input in zip(decoded_windows, model_inputs, strict=True):
                request_row = {
                    **_clip_base_row(row),
                    "window_index": window_index,
                    "start_frame": window_info.start,
                    "end_frame": window_info.end,
                    "caption_skip": False,
                    "caption_error": None,
                }
                _add_ray_llm_columns(request_row, model_input, resolved_sampling_params)
                output_rows.append(request_row)

            return output_rows

        return [
            {
                **_clip_base_row(row),
                "window_index": -1,
                "start_frame": None,
                "end_frame": None,
                "caption_skip": True,
                "caption_status": "error",
                "caption_failure_reason": "exception",
                "caption_error": "no_caption_windows",
                # Ray's LLM stage treats non-empty inference errors as pass-through rows,
                # so no-window clips avoid prompt validation and vLLM inference.
                "__inference_error__": "no_caption_windows",
                "qwen_caption": None,
                "qwen_prompt_tokens": 0,
                "qwen_output_tokens": 0,
            }
        ]

    return _make_window_rows


def make_caption_clip_rows_fn(
    vllm_config: VllmConfig | None = None,
    window_config: WindowConfig | None = None,
    sampling_config: VllmSamplingConfig | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a ``map`` function that attaches caption windows to one clip row."""
    make_window_rows = make_caption_window_rows_fn(
        vllm_config=vllm_config,
        window_config=window_config,
        sampling_config=sampling_config,
    )

    def _make_clip_row(row: dict[str, Any]) -> dict[str, Any]:
        return {**_clip_base_row(row), "caption_windows": make_window_rows(row)}

    return _make_clip_row


def _auto_processor(vllm_config: VllmConfig) -> object:
    cache_key = f"{vllm_config.model_variant}:{vllm_config.copy_weights_to or ''}"
    if cache_key not in _PROCESSOR_CACHE:
        from cosmos_curator.models.vllm_interface import auto_processor  # noqa: PLC0415

        _PROCESSOR_CACHE[cache_key] = auto_processor(vllm_config)
    return _PROCESSOR_CACHE[cache_key]


def caption_window_rows(
    ds: ray.data.Dataset,
    *,
    model_source: str,
    caption_workers: int,
    vllm_config: VllmConfig | None = None,
    window_config: WindowConfig | None = None,
) -> ray.data.Dataset:
    """Add Qwen captions to MP4 clip rows and return one row per clip."""
    resolved_vllm_config = vllm_config or make_default_vllm_config()
    resolved_window_config = window_config or make_default_window_config()

    if caption_workers <= 0:
        msg = "Captioning requires at least one visible Ray GPU; rerun with --no-generate-captions to split only."
        raise RuntimeError(msg)

    clip_ds = ds.map(
        make_caption_clip_rows_fn(
            vllm_config=resolved_vllm_config,
            window_config=resolved_window_config,
            sampling_config=resolved_vllm_config.sampling_config,
        ),
        num_cpus=resolved_vllm_config.num_cpus_for_prepare,
        runtime_env=PixiRuntimeEnv("unified"),
    )

    processor = _build_processor(
        model_source=model_source,
        caption_workers=caption_workers,
        vllm_config=resolved_vllm_config,
    )
    return processor(clip_ds).map(
        make_normalize_caption_clip_output_fn(resolved_vllm_config.sampling_config),
        num_cpus=0.25,
    )


def _build_processor(
    *,
    model_source: str,
    caption_workers: int,
    vllm_config: VllmConfig,
) -> Callable[[ray.data.Dataset], ray.data.Dataset]:
    from ray.data.llm import build_processor, vLLMEngineProcessorConfig  # noqa: PLC0415

    engine_kwargs: dict[str, Any] = {
        # Mirrors cosmos_curator.models.vllm_qwen.VllmQwen.model without importing
        # vLLM-backed classes on the driver.
        "limit_mm_per_prompt": {"images": 0, "video": 1},
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.85,
        "mm_processor_kwargs": {
            "do_resize": vllm_config.preprocess,
            "do_rescale": vllm_config.preprocess,
            "do_normalize": vllm_config.preprocess,
        },
        "mm_processor_cache_gb": 0.0 if vllm_config.disable_mmcache else 4.0,
        "max_num_batched_tokens": 32768,
        "tensor_parallel_size": vllm_config.num_gpus,
        "trust_remote_code": False,
        "compilation_config": {"cudagraph_mode": "piecewise"},
        "performance_mode": vllm_config.performance_mode,
    }
    if vllm_config.fp8:
        engine_kwargs["quantization"] = "fp8"

    processor = build_processor(
        cast(
            "Any",
            vLLMEngineProcessorConfig(
                model_source=model_source,
                batch_size=vllm_config.batch_size,
                concurrency=(1, caption_workers),
                runtime_env=ray_data_gpu_runtime_env("unified"),
                should_continue_on_error=True,
                chat_template_stage=False,
                tokenize_stage=False,
                detokenize_stage=False,
                prepare_image_stage=False,
                prepare_multimodal_stage=False,
                engine_kwargs=engine_kwargs,
            ),
        ),
    )
    _install_vllm_engine_stage_shim(processor)
    return cast("Callable[[ray.data.Dataset], ray.data.Dataset]", processor)


def make_normalize_caption_output_fn(
    sampling_config: VllmSamplingConfig | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a normalizer from Ray processor output to metadata-window rows."""
    max_tokens = (sampling_config or VllmSamplingConfig()).max_tokens

    def _normalize(row: dict[str, Any]) -> dict[str, Any]:
        if row.get("caption_skip", False):
            return _make_skipped_caption_output(row)

        error = str(row.get("__inference_error__", "") or "")
        text = str(row.get("generated_text", "") or "").strip()
        prompt_tokens = _safe_int(row.get("num_input_tokens", 0))
        output_tokens = _safe_int(row.get("num_generated_tokens", 0))

        if error or not text:
            status = "error"
            failure_reason = "exception"
            caption: str | None = None
            prompt_tokens = 0
            output_tokens = 0
        elif max_tokens is not None and output_tokens >= max_tokens:
            status = "truncated"
            failure_reason = None
            caption = text
        else:
            status = "success"
            failure_reason = None
            caption = text

        return {
            **_clip_base_row(row),
            "window_index": _safe_int(row.get("window_index", -1)),
            "start_frame": row.get("start_frame"),
            "end_frame": row.get("end_frame"),
            "caption_skip": False,
            "caption_status": status,
            "caption_failure_reason": failure_reason,
            "caption_error": error or None,
            "qwen_caption": caption,
            "qwen_prompt_tokens": prompt_tokens,
            "qwen_output_tokens": output_tokens,
        }

    return _normalize


def make_normalize_caption_clip_output_fn(
    sampling_config: VllmSamplingConfig | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a normalizer for one clip row with nested caption windows."""
    normalize_window = make_normalize_caption_output_fn(sampling_config)

    def _normalize(row: dict[str, Any]) -> dict[str, Any]:
        windows = row.get("caption_windows", [])
        if not isinstance(windows, list):
            msg = f"caption_windows must be a list, got {type(windows).__name__}"
            raise TypeError(msg)
        return {
            **_clip_base_row(row),
            "caption_windows": [normalize_window(cast("dict[str, Any]", window)) for window in windows],
        }

    return _normalize


def write_captioned_metadata_and_summary(
    ds: ray.data.Dataset,
    *,
    input_video_path: str,
    output_path: str,
    num_input_videos: int,
) -> int:
    """Run captioning, write final per-clip metadata, and write ``summary.json``."""
    pipeline_start = time.monotonic()
    clip_rows = _write_caption_metadata(ds, output_path).take_all()
    pipeline_run_time_minutes = (time.monotonic() - pipeline_start) / 60

    return write_summary_from_rows(
        clip_rows,
        input_video_path=input_video_path,
        output_path=output_path,
        num_input_videos=num_input_videos,
        pipeline_run_time_minutes=pipeline_run_time_minutes,
    )


def _caption_metadata_writer(output_path: str) -> StorageWriter:
    writer = _CAPTION_METADATA_WRITER_CACHE.get(output_path)
    if writer is None:
        writer = StorageWriter(output_path)
        _CAPTION_METADATA_WRITER_CACHE[output_path] = writer
    return writer


def _write_caption_metadata_row(row: dict[str, Any], *, output_path: str) -> dict[str, Any]:
    rows = row.get("caption_windows", [])
    if not isinstance(rows, list):
        msg = f"caption_windows must be a list, got {type(rows).__name__}"
        raise TypeError(msg)
    if not rows:
        msg = "caption metadata row must contain at least one caption window"
        raise ValueError(msg)

    window_rows = cast("list[dict[str, Any]]", rows)
    window_rows.sort(key=lambda row: _safe_int(row.get("window_index", -1)))
    metadata, clip_row = _make_clip_metadata_rows(window_rows)
    _caption_metadata_writer(output_path).write_str_to(
        f"metas/v0/{clip_row['clip_uuid']}.json", json.dumps(metadata, indent=4)
    )
    return clip_row


def _write_caption_metadata(ds: ray.data.Dataset, output_path: str) -> ray.data.Dataset:
    def _write_row(row: dict[str, Any]) -> dict[str, Any]:
        return _write_caption_metadata_row(row, output_path=output_path)

    return ds.map(
        _write_row,
        num_cpus=0.25,
    )


def _make_clip_metadata_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    base = rows[0]
    windows: list[dict[str, Any]] = []
    total_prompt_tokens = 0
    total_output_tokens = 0
    num_caption_windows = 0
    has_caption = False

    for row in rows:
        if row.get("caption_skip", False):
            continue

        curr_window: dict[str, Any] = {
            "start_frame": row["start_frame"],
            "end_frame": row["end_frame"],
            "caption_status": row["caption_status"],
            "caption_failure_reason": row["caption_failure_reason"],
        }
        caption = row.get("qwen_caption")
        if caption is not None:
            curr_window["qwen_caption"] = caption

        if row.get("caption_status") in _OK_CAPTION_STATUSES:
            prompt_tokens = int(row.get("qwen_prompt_tokens", 0))
            output_tokens = int(row.get("qwen_output_tokens", 0))
            curr_window["qwen_prompt_tokens"] = prompt_tokens
            curr_window["qwen_output_tokens"] = output_tokens
            total_prompt_tokens += prompt_tokens
            total_output_tokens += output_tokens
            num_caption_windows += 1
            has_caption = True
        elif row.get("caption_error"):
            curr_window["errors"] = {"qwen": row["caption_error"]}

        windows.append(curr_window)

    metadata: dict[str, Any] = {
        "span_uuid": base["clip_uuid"],
        "source_video": base["video_path"],
        "duration_span": [base["clip_start_s"], base["clip_end_s"]],
        "width_source": base["width_source"],
        "height_source": base["height_source"],
        "framerate_source": base["framerate_source"],
        "clip_location": base["clip_location"],
        "width": base["width"],
        "height": base["height"],
        "framerate": base["framerate"],
        "num_frames": base["num_frames"],
        "video_codec": base["video_codec"],
        "num_bytes": base["num_bytes"],
        "windows": windows,
        "filtered_windows": [],
        "valid": bool(windows),
        "has_caption": has_caption,
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
        "num_caption_windows": num_caption_windows,
    }

    clip_row = {
        "video_path": base["video_path"],
        "video_size": base["video_size"],
        "duration_s": base["duration_s"],
        "clip_uuid": base["clip_uuid"],
        "clip_start_s": base["clip_start_s"],
        "clip_end_s": base["clip_end_s"],
        "clip_location": base["clip_location"],
        "has_caption": has_caption,
        "num_caption_windows": num_caption_windows,
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
    }
    return metadata, clip_row


def _clip_base_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "video_path": row["video_path"],
        "video_size": row["video_size"],
        "duration_s": row["duration_s"],
        "clip_uuid": row["clip_uuid"],
        "clip_start_s": row["clip_start_s"],
        "clip_end_s": row["clip_end_s"],
        "clip_location": row["clip_location"],
        "width_source": row["width_source"],
        "height_source": row["height_source"],
        "framerate_source": row["framerate_source"],
        "width": row["width"],
        "height": row["height"],
        "framerate": row["framerate"],
        "num_frames": row["num_frames"],
        "video_codec": row["video_codec"],
        "num_bytes": row["num_bytes"],
    }


def _make_skipped_caption_output(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **_clip_base_row(row),
        "window_index": _safe_int(row.get("window_index", -1)),
        "start_frame": row.get("start_frame"),
        "end_frame": row.get("end_frame"),
        "caption_skip": True,
        "caption_status": "error",
        "caption_failure_reason": "exception",
        "caption_error": row.get("caption_error") or "no_caption_windows",
        "qwen_caption": None,
        "qwen_prompt_tokens": 0,
        "qwen_output_tokens": 0,
    }


def _safe_int(value: object) -> int:
    try:
        converted: int = int(cast("Any", value))
    except (TypeError, ValueError):
        return 0
    else:
        return converted
