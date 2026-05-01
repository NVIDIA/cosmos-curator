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

"""vLLM-based image captioning helpers and stages."""

import logging
from typing import Any, Literal, cast

import attrs
import numpy as np
import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.infra.gpu_start_helper import gpu_stage_startup
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.core.utils.model import conda_utils
from cosmos_curator.models.all_models import get_all_models_by_id
from cosmos_curator.models.vllm_model_ids import get_vllm_model_id
from cosmos_curator.models.vllm_sentinels import VLLM_UNKNOWN_CAPTION
from cosmos_curator.pipelines.image.captioning.image_prep_utils import (
    DEFAULT_PREP_MAX_PIXELS,
    DEFAULT_PREP_MIN_PIXELS,
)
from cosmos_curator.pipelines.image.utils.data_model import ImagePipeTask
from cosmos_curator.pipelines.video.utils.data_model import (
    CaptionOutcome,
    CaptionResult,
    VllmConfig,
    WindowConfig,
)

IMAGE_FACTOR = 28

if conda_utils.is_running_in_env("unified"):
    import torch
    from PIL import Image as PILImage
    from torchvision import transforms  # type: ignore[import-untyped]
    from torchvision.transforms import InterpolationMode  # type: ignore[import-untyped]

    from cosmos_curator.models.prompts import get_prompt, get_stage2_prompt
    from cosmos_curator.models.vllm_interface import (
        auto_processor,
        make_metadata,
        make_model_inputs,
        sampling_params,
        vllm_caption,
        vllm_model,
    )
    from cosmos_curator.pipelines.video.utils.vision_process import smart_resize

    vllm_logger = logging.getLogger("vllm")
    vllm_logger.setLevel(logging.ERROR)  # Suppress warnings and info from vLLM


class _ImageVllmModelInfo(ModelInterface):
    """Model interface for image vLLM caption stage (model_id_names + env only)."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        self._vllm_config = vllm_config

    @property
    def conda_env_name(self) -> str:
        return "unified"

    @property
    def model_id_names(self) -> list[str]:
        model_variant = get_vllm_model_id(self._vllm_config.model_variant)
        models = get_all_models_by_id()
        model = models.get(model_variant)
        if model is None:
            msg = f"Model not found for {self._vllm_config.model_variant} -> {model_variant}"
            raise ValueError(msg)
        model_id = model.get("model_id")
        if model_id is None:
            msg = f"Model ID not found for variant {self._vllm_config.model_variant} -> {model_variant}"
            raise ValueError(msg)
        return [cast("str", model_id)]

    def setup(self) -> None:
        pass


def _image_frame_to_tensor(frame: np.ndarray[Any, Any]) -> Any:  # noqa: ANN401
    """Convert one RGB frame to (1, C, H, W) tensor in [0, 1] for client-side preprocess."""
    pil = PILImage.fromarray(frame, mode="RGB")
    to_tensor = transforms.ToTensor()
    return to_tensor(pil).unsqueeze(0)  # (1, C, H, W)


def _image_frame_to_model_preprocess_tensor(frame: np.ndarray[Any, Any]) -> Any:  # noqa: ANN401
    """Convert one RGB frame to (1, C, H, W) uint8 tensor for model-side preprocess."""
    tensor = torch.from_numpy(frame).permute(2, 0, 1).contiguous()
    return tensor.unsqueeze(0)


def _resize_image_tensor(
    tensor: Any,  # noqa: ANN401
    min_pixels: int,
    max_pixels: int,
) -> Any:  # noqa: ANN401
    """Resize (1, C, H, W) tensor to target size via smart_resize (same logic as video pipeline)."""
    height, width = int(tensor.shape[2]), int(tensor.shape[3])
    if height <= 0 or width <= 0:
        msg = f"Invalid image dimensions for resize: {height}x{width}"
        raise ValueError(msg)
    resized_h, resized_w = smart_resize(
        height, width, factor=IMAGE_FACTOR, min_pixels=min_pixels, max_pixels=max_pixels
    )
    # Fail fast if smart_resize ever returns 0 or negative (before clamp)
    if resized_h <= 0 or resized_w <= 0:
        msg = (
            f"smart_resize produced invalid dimensions ({resized_h}x{resized_w}) "
            f"for input {height}x{width} (min_pixels={min_pixels}, max_pixels={max_pixels})"
        )
        raise ValueError(msg)
    # Clamp small positive values so resize() never gets a dimension below IMAGE_FACTOR
    resized_h = max(IMAGE_FACTOR, resized_h)
    resized_w = max(IMAGE_FACTOR, resized_w)
    return transforms.functional.resize(
        tensor,
        [resized_h, resized_w],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )


_PREVIEW_MAX_LEN = 80


@attrs.frozen
class _ImagePrepParams:
    """Parameters for prepare_image_model_input (reduces argument count)."""

    min_pixels: int
    max_pixels: int
    window_config: WindowConfig
    vllm_config: VllmConfig
    processor: Any
    prompt: str


def prepare_image_model_input(image_frame: np.ndarray[Any, Any], params: _ImagePrepParams) -> dict[str, Any]:
    """Prepare one decoded frame for captioning.

    Args:
        image_frame: One decoded RGB frame to process.
        params: Prep parameters (min/max pixels, window_config, vllm_config, processor, prompt).

    Returns:
        Dictionary with keys:
        - "model_input": Model input dict for vLLM
        - "height": Image height after resize
        - "width": Image width after resize

    Raises:
        ValueError: If resize fails.

    """
    if params.vllm_config.preprocess:
        tensor = _image_frame_to_model_preprocess_tensor(image_frame)
    else:
        tensor = _image_frame_to_tensor(image_frame)
    tensor = _resize_image_tensor(tensor, params.min_pixels, params.max_pixels)
    height = int(tensor.shape[2])
    width = int(tensor.shape[3])
    metadata_list = make_metadata([tensor], params.window_config)
    model_inputs = make_model_inputs(
        [tensor],
        metadata_list,
        params.vllm_config,
        params.processor,
        params.prompt,
    )
    return {
        "model_input": model_inputs[0],
        "height": height,
        "width": width,
    }


def caption_images(
    model_inputs: list[dict[str, Any]],
    vllm_config: VllmConfig,
    llm: Any,  # noqa: ANN401
    processor: Any,  # noqa: ANN401
    sampling_params: Any,  # noqa: ANN401
) -> list[Any]:
    """Generate captions for a batch of image model inputs.

    Args:
        model_inputs: List of model input dicts from prepare_image_model_input.
        vllm_config: vLLM configuration.
        llm: vLLM LLM instance.
        processor: AutoProcessor instance.
        sampling_params: vLLM SamplingParams instance.

    Returns:
        List of raw vLLM results, one per model input.

    """
    n_inputs = len(model_inputs)
    if vllm_config.stage2_caption:
        stage2_prompts = cast(
            "list[str | None]",
            [get_stage2_prompt(vllm_config.stage2_prompt_text)] * n_inputs,
        )
    else:
        stage2_prompts = cast("list[str | None]", [None] * n_inputs)

    return vllm_caption(
        model_inputs,
        llm,
        processor,
        sampling_params,
        vllm_config,
        max_inflight_requests=0,
        inflight_batching=True,
        stage2_prompts=stage2_prompts,
    )


def _collect_caption_inputs(
    tasks: list[ImagePipeTask],
    variant: str,
    *,
    result_target: Literal["caption", "filter_caption"] = "caption",
    result_key: str | None = None,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Collect valid model inputs and their task indices for captioning/filter-captioning."""
    model_inputs_list: list[dict[str, Any]] = []
    valid_indices: list[int] = []
    storage_key = result_key or variant
    for i, task in enumerate(tasks):
        if task.image.is_filtered:
            continue
        if result_target == "caption" and task.image.has_caption():
            continue
        if result_target == "filter_caption" and storage_key in task.image.filter_captions:
            continue
        inp = task.image.model_input.get(variant)
        if inp is None:
            if "caption_prep" not in task.image.errors:
                logger.warning(
                    "Skipping caption for %s: no model_input[%r] (prep did not populate or set error)",
                    task.session_id,
                    variant,
                )
            continue
        if "caption_prep" in task.image.errors:
            continue
        model_inputs_list.append(inp)
        valid_indices.append(i)
    return model_inputs_list, valid_indices


def _normalize_vllm_result(result: Any) -> CaptionResult:  # noqa: ANN401
    """Map a raw vLLM interface result to a normalized image caption result."""
    text = result.text.strip()
    if text == VLLM_UNKNOWN_CAPTION:
        return CaptionResult(outcome=CaptionOutcome.ERROR, failure_reason="exception")
    if result.finish_reason == "length":
        if text:
            return CaptionResult(outcome=CaptionOutcome.TRUNCATED, text=text)
        return CaptionResult(outcome=CaptionOutcome.ERROR, failure_reason="exception")
    if text:
        return CaptionResult(outcome=CaptionOutcome.SUCCESS, text=text)
    return CaptionResult(outcome=CaptionOutcome.ERROR, failure_reason="exception")


def _scatter_captions(  # noqa: PLR0913
    tasks: list[ImagePipeTask],
    valid_indices: list[int],
    results: list[Any],
    model_variant: str,
    *,
    verbose: bool,
    result_target: Literal["caption", "filter_caption"] = "caption",
    result_key: str | None = None,
) -> None:
    """Write normalized caption/filter-caption outputs into tasks at valid_indices."""
    storage_key = result_key or model_variant
    for idx, raw_result in zip(valid_indices, results, strict=True):
        image = tasks[idx].image
        result = _normalize_vllm_result(raw_result)
        if result.text is not None:
            if result_target == "caption":
                image.caption = result.text
                image.captions[model_variant] = result.text
            else:
                image.filter_captions[storage_key] = result.text
        if result_target == "caption":
            image.token_counts[model_variant] = raw_result.token_counts
            image.caption_status = result.outcome.value
            image.caption_failure_reason = result.failure_reason if result.outcome == CaptionOutcome.ERROR else None
        else:
            image.token_counts[storage_key] = raw_result.token_counts
            image.filter_caption_status[storage_key] = result.outcome.value
            image.filter_caption_failure_reason[storage_key] = (
                result.failure_reason if result.outcome == CaptionOutcome.ERROR else None
            )
        if verbose:
            preview = (
                raw_result.text[:_PREVIEW_MAX_LEN] + "..."
                if len(raw_result.text) > _PREVIEW_MAX_LEN
                else raw_result.text
            )
            logger.info(f"Caption for {tasks[idx].session_id}: {preview}")


def _clear_model_inputs(tasks: list[ImagePipeTask]) -> None:
    """Drop prep payloads before returning tasks to a non-unified downstream stage."""
    for task in tasks:
        task.image.model_input.clear()


class ImageVllmPrepStage(CuratorStage):
    """Prep stage: decode image bytes, resize, and build vLLM model input (image modality only).

    Output is always image modality (multi_modal_data["image"], content type "image").
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        caption_prep_min_pixels: int | None = None,
        caption_prep_max_pixels: int | None = None,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Store config and init timer; processor set in stage_setup."""
        self._timer = StageTimer(self)
        self._vllm_config = vllm_config
        self._min_pixels = caption_prep_min_pixels if caption_prep_min_pixels is not None else DEFAULT_PREP_MIN_PIXELS
        self._max_pixels = caption_prep_max_pixels if caption_prep_max_pixels is not None else DEFAULT_PREP_MAX_PIXELS
        self._verbose = verbose
        self._log_stats = log_stats
        self._processor: Any = None
        self._window_config = WindowConfig(sampling_fps=1.0)

    @property
    def resources(self) -> CuratorStageResource:
        """CPU resources for prep (no GPU)."""
        return CuratorStageResource(cpus=max(0.25, self._vllm_config.num_cpus_for_prepare))

    @property
    def conda_env_name(self) -> str:
        """Unified env for torch/vLLM."""
        return "unified"

    def stage_setup(self) -> None:
        """Load processor for the configured model variant."""
        self._processor = auto_processor(self._vllm_config)

    @nvtx.annotate("ImageVllmPrepStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[ImagePipeTask]) -> list[ImagePipeTask] | None:
        """Decode, resize, and build vLLM model input for each task (image modality)."""
        if self._processor is None:
            msg = "processor not initialized, call stage_setup() first"
            raise RuntimeError(msg)

        prompt = get_prompt(
            self._vllm_config.prompt_variant,
            self._vllm_config.prompt_text,
            verbose=self._verbose,
        )
        variant = self._vllm_config.model_variant
        prep_params = _ImagePrepParams(
            min_pixels=self._min_pixels,
            max_pixels=self._max_pixels,
            window_config=self._window_config,
            vllm_config=self._vllm_config,
            processor=self._processor,
            prompt=prompt,
        )

        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            image = task.image
            if image.is_filtered:
                continue
            if image.image_data is None:
                image.errors["caption_prep"] = "no image_data"
                continue
            if len(image.image_data.frames) == 0:
                image.errors["caption_prep"] = "image_data has no frames"
                continue
            with self._timer.time_process():
                try:
                    result = prepare_image_model_input(image.image_data.frames[0], prep_params)
                    image.model_input[variant] = result["model_input"]
                    image.height = result["height"]
                    image.width = result["width"]
                except Exception as e:  # noqa: BLE001
                    image.errors["caption_prep"] = str(e)
                    logger.warning(f"Caption prep failed for {task.session_id}: {e}")
                    continue
            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats
        return tasks


class ImageVllmCaptionStage(CuratorStage):
    """Caption stage: run vLLM on prepped model inputs (image modality only) and write caption."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        result_target: Literal["caption", "filter_caption"] = "caption",
        result_key: str | None = None,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Store config and init timer; model/sampling/processor set in stage_setup."""
        self._timer = StageTimer(self)
        self._vllm_config = vllm_config
        self._result_target = result_target
        self._result_key = result_key
        self._verbose = verbose
        self._log_stats = log_stats
        self._llm: Any = None
        self._sampling_params: Any = None
        self._processor: Any = None
        self._model = _ImageVllmModelInfo(vllm_config)

    @property
    def resources(self) -> CuratorStageResource:
        """GPU resources for caption model."""
        return CuratorStageResource(gpus=self._vllm_config.num_gpus)

    @property
    def conda_env_name(self) -> str:
        """Unified env for vLLM."""
        return "unified"

    @property
    def model(self) -> ModelInterface:
        """Model interface for pipeline build (model_id_names, conda_env_name)."""
        return self._model

    def stage_setup(self) -> None:
        """Load vLLM model, sampling params, and processor."""
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)
        self._llm = vllm_model(self._vllm_config)
        self._sampling_params = sampling_params(self._vllm_config.sampling_config)
        self._processor = auto_processor(self._vllm_config)
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=False)

    @nvtx.annotate("ImageVllmCaptionStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[ImagePipeTask]) -> list[ImagePipeTask] | None:
        """Run vLLM caption on prepped inputs and scatter normalized caption outputs."""
        if self._llm is None or self._sampling_params is None or self._processor is None:
            msg = "stage not initialized, call stage_setup() first"
            raise RuntimeError(msg)

        variant = self._vllm_config.model_variant
        model_inputs_list, valid_indices = _collect_caption_inputs(
            tasks,
            variant,
            result_target=self._result_target,
            result_key=self._result_key,
        )
        if not model_inputs_list:
            logger.warning("No valid model inputs for captioning")
            _clear_model_inputs(tasks)
            return tasks

        major_size = sum(tasks[i].get_major_size() for i in valid_indices)
        self._timer.reinit(self, major_size)
        with self._timer.time_process():
            results = caption_images(
                model_inputs_list,
                self._vllm_config,
                self._llm,
                self._processor,
                self._sampling_params,
            )
        _scatter_captions(
            tasks,
            valid_indices,
            results,
            variant,
            verbose=self._verbose,
            result_target=self._result_target,
            result_key=self._result_key,
        )
        _clear_model_inputs(tasks)

        if self._log_stats and tasks and valid_indices:
            stage_name, stage_perf_stats = self._timer.log_stats()
            tasks[valid_indices[0]].stage_perf[stage_name] = stage_perf_stats

        return tasks
