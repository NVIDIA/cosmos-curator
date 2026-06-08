# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""vLLM plugin for Nemotron Nano 12B v2 model."""

import os
import secrets
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, RequestOutput
from vllm.config import CompilationConfig
from vllm.engine.arg_utils import AsyncEngineArgs

from cosmos_curator.models.vllm_plugin import VllmPlugin
from cosmos_curator.pipelines.video.utils.data_model import VllmAsyncConfig, VllmCaptionRequest, VllmConfig

# Constants tuned similarly to existing plugins
GPU_MEMORY_UTILIZATION = 0.9
MAX_NUM_BATCHED_TOKENS = 32768
MAX_MODEL_LEN = 32768
TRUST_REMOTE_CODE = True
LIMIT_MM_PER_PROMPT_VIDEO = {"video": 1}
LIMIT_MM_PER_PROMPT_IMAGE = {"image": 1}

_DEFAULT_REFINE_PROMPT = (
    """
    Improve and refine following video description.
    Focus on highlighting the key visual and sensory elements.
    Ensure the description is clear, precise, and paints a compelling
    picture of the scene.
    """.strip()
    + "\n"
)

# Constants for tensor dimensions and channel counts
EXPECTED_TENSOR_DIMENSIONS = 4
EXPECTED_NUMPY_DIMENSIONS = 4
EXPECTED_CHANNELS = 3


def _validate_numpy_array(
    array: npt.NDArray[np.uint8],
    *,
    clip_float_for_image: bool = False,
) -> npt.NDArray[np.uint8]:
    """Validate and normalize numpy array format."""
    if array.ndim != EXPECTED_NUMPY_DIMENSIONS:
        msg = f"Expected 4D numpy array (T, H, W, C), got shape {array.shape}"
        raise ValueError(msg)

    if array.shape[-1] != EXPECTED_CHANNELS:
        msg = f"Expected channels-last format (T, H, W, 3), got shape {array.shape}."
        raise ValueError(msg)

    return _normalize_dtype(array, clip_float_for_image=clip_float_for_image)


def _float_to_uint8_video(arr: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    """Original video behavior: scale by 255 only when values in [0, 1]."""
    if arr.max() <= 1.0:
        return (arr * 255).astype(np.uint8)
    return arr.astype(np.uint8)


def _float_to_uint8_image(arr: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    """Image path: expect [0, 1] or [0, 255]; normalize to [0, 1] then scale and clip to uint8."""
    arr_f = arr.astype(np.float32)
    if arr_f.size > 0:
        mx = float(np.max(arr_f))
        if mx > 255.0:  # noqa: PLR2004
            msg = f"Float array has max {mx} (expected [0, 1] or [0, 255])"
            raise ValueError(msg)
        if mx > 1.0:
            arr_f = arr_f / 255.0
    return np.clip(arr_f * 255, 0, 255).astype(np.uint8)


def _normalize_dtype(
    array: npt.NDArray[np.uint8],
    *,
    clip_float_for_image: bool = False,
) -> npt.NDArray[np.uint8]:
    """Normalize array dtype to uint8. Video and image paths use separate float handling."""
    if array.dtype == np.uint8:
        return array
    if array.dtype in (np.float32, np.float16):
        return _float_to_uint8_image(array) if clip_float_for_image else _float_to_uint8_video(array)
    return array.astype(np.uint8)


def _convert_tensor_to_numpy(
    tensor: torch.Tensor,
    *,
    clip_float_for_image: bool = False,
) -> npt.NDArray[np.uint8]:
    """Convert torch.Tensor (T, C, H, W) to numpy (T, H, W, C)."""
    if tensor.ndim != EXPECTED_TENSOR_DIMENSIONS:
        msg = f"Expected 4D torch.Tensor (T, C, H, W), got shape {tensor.shape}"
        raise ValueError(msg)

    video_np = tensor.permute(0, 2, 3, 1).cpu().numpy()
    return _normalize_dtype(video_np, clip_float_for_image=clip_float_for_image)


def _convert_video_format(
    video_inputs: torch.Tensor | npt.NDArray[np.uint8] | None,
    *,
    clip_float_for_image: bool = False,
) -> npt.NDArray[np.uint8] | None:
    """Convert torch.Tensor (T, C, H, W) or np.ndarray to vLLM format (T, H, W, C)."""
    retval: npt.NDArray[np.uint8] | None = None
    if video_inputs is None:
        retval = None
    elif isinstance(video_inputs, torch.Tensor):
        retval = _convert_tensor_to_numpy(video_inputs, clip_float_for_image=clip_float_for_image)
    else:  # isinstance(video_inputs, np.ndarray):
        retval = _validate_numpy_array(video_inputs, clip_float_for_image=clip_float_for_image)

    return retval


def make_prompt(
    message: dict[str, Any],
    frames: torch.Tensor | npt.NDArray[np.uint8],
    metadata: dict[str, Any],
    processor: AutoProcessor,
    *,
    use_image: bool = False,
) -> dict[str, Any]:
    """Make a prompt for the Nemotron Nano 12B v2 model.

    Args:
        message: The message to use for the prompt.
        frames: Frames as torch.Tensor (T, C, H, W) or numpy (T, H, W, C) uint8;
            refinement path passes numpy from multi_modal_data["video"][0].
        metadata: The metadata of the video clip or image.
        processor: The processor to use for the prompt.
        use_image: When True, select image modality (multi_modal_data["image"]);
            when False, select video (multi_modal_data["video"] with metadata).
            Aligned with VllmConfig.use_image_input.

    Returns:
        A prompt for the Nemotron Nano 12B v2 model.

    """
    video_np = _convert_video_format(frames, clip_float_for_image=use_image)
    if video_np is None:
        msg = "convert_video_format returned None"
        raise ValueError(msg)
    prompt_ids = processor.apply_chat_template(  # type: ignore[attr-defined]
        [message], add_generation_prompt=True, tokenize=True, return_tensors="pt"
    )[0].tolist()

    if use_image:
        single_frame = video_np[0]  # (H, W, C) uint8
        if single_frame.ndim != 3 or single_frame.shape[-1] != EXPECTED_CHANNELS:  # noqa: PLR2004
            msg = f"Expected single frame (H, W, C) with C={EXPECTED_CHANNELS}, got shape {single_frame.shape}"
            raise ValueError(msg)
        pil_image = Image.fromarray(single_frame.copy())
        multi_modal_data: dict[str, Any] = {"image": [pil_image]}
    else:
        nemotron_metadata = {
            "total_num_frames": frames.shape[0],
            "fps": metadata["fps"],
            "duration": metadata["duration"],
            "frames_indices": metadata["frames_indices"],
            "video_backend": metadata["video_backend"],
        }
        multi_modal_data = {"video": (video_np, nemotron_metadata)}

    return {
        "prompt_token_ids": prompt_ids,
        "multi_modal_data": multi_modal_data,
    }


def make_message(text_input: str, *, use_image: bool = False) -> dict[str, Any]:
    """Create a chat message structure for Nemotron Nano 12B v2.

    Args:
        text_input: The text input to create a message for.
        use_image: When True, select image modality (content type "image"); when False,
            select video. Aligned with VllmConfig.use_image_input.

    Returns:
        A chat message structure for Nemotron Nano 12B v2.

    """
    content_type = "image" if use_image else "video"
    return {
        "role": "user",
        "content": [{"type": content_type}, {"type": "text", "text": text_input}],
    }


class VllmNemotronNano12Bv2VL(VllmPlugin):
    """Nemotron Nano 12B v2 vLLM model variant plugin."""

    @classmethod
    def model_variant(cls) -> str:
        """Return the model variant name."""
        return "nemotron"

    @classmethod
    def model(cls, config: VllmConfig) -> LLM:
        """Instantiate the vLLM model for Nemotron Nano 12B v2.

        Args:
            config: Configuration for the model.

        Returns:
            The vLLM model.

        """
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        limit_mm = LIMIT_MM_PER_PROMPT_IMAGE if config.use_image_input else LIMIT_MM_PER_PROMPT_VIDEO
        return LLM(
            model=str(cls.model_path(config)),
            trust_remote_code=TRUST_REMOTE_CODE,
            tensor_parallel_size=config.num_gpus,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            limit_mm_per_prompt=limit_mm,
            compilation_config={"cudagraph_mode": "piecewise"},
            performance_mode=config.performance_mode,
        )

    @classmethod
    def model_async(cls, config: VllmAsyncConfig) -> AsyncEngineArgs:
        """Build ``AsyncEngineArgs`` for Nemotron in-process ``AsyncLLM``.

        Mirrors :meth:`model` - reads from module-scope constants.
        """
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        gpu_mem_util = (
            config.gpu_memory_utilization if config.gpu_memory_utilization is not None else GPU_MEMORY_UTILIZATION
        )
        return AsyncEngineArgs(
            model=str(cls.model_path(config.to_vllm_config())),
            served_model_name=[config.model_variant],
            tensor_parallel_size=int(config.num_gpus),
            data_parallel_size=max(1, config.data_parallel_size),
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=TRUST_REMOTE_CODE,
            limit_mm_per_prompt=LIMIT_MM_PER_PROMPT_VIDEO,  # type: ignore[arg-type]
            max_num_seqs=config.max_num_seqs if config.max_num_seqs > 0 else None,
            enforce_eager=config.enforce_eager,
            kv_cache_dtype=config.kv_cache_dtype,  # type: ignore[arg-type]
            mm_encoder_tp_mode=config.mm_encoder_tp_mode or None,  # type: ignore[arg-type]
            mm_processor_cache_type=config.mm_processor_cache_type or None,  # type: ignore[arg-type]
            async_scheduling=config.async_scheduling,
            enable_chunked_prefill=config.enable_chunked_prefill,
            disable_chunked_mm_input=config.disable_chunked_mm_input,
            long_prefill_token_threshold=config.long_prefill_token_threshold,
            stream_interval=config.stream_interval,
            distributed_executor_backend=config.distributed_executor_backend,
            skip_mm_profiling=config.skip_mm_profiling,
            disable_log_stats=config.disable_log_stats,
            enable_log_requests=config.enable_log_requests,
            mm_processor_cache_gb=0.0 if config.disable_mmcache else 4.0,
            mm_processor_kwargs={
                "do_sample_frames": False,
                "do_resize": config.model_preprocess_enabled,
                "do_rescale": config.model_preprocess_enabled,
                "do_normalize": config.model_preprocess_enabled,
            },
            compilation_config=CompilationConfig(cudagraph_mode="piecewise"),  # type: ignore[arg-type]
            enable_prefix_caching=True,
            use_tqdm_on_load=False,
        )

    @classmethod
    def processor(cls, config: VllmConfig) -> AutoProcessor:
        """Return the AutoProcessor for the model."""
        processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            str(cls.model_path(config)),
            trust_remote_code=TRUST_REMOTE_CODE,
            use_fast=False,  # No fast processor available for nemotron, be explicit to silence warnings.
        )
        return cast("AutoProcessor", processor)

    @staticmethod
    def make_llm_input(
        prompt: str,
        frames: torch.Tensor,
        metadata: dict[str, Any],
        processor: AutoProcessor,
        config: VllmConfig,
    ) -> dict[str, Any]:
        """Make LLM inputs for the model (video or image).

        Args:
            prompt: The prompt to use for the LLM.
            frames: The frames to use for the LLM.
            metadata: The metadata to use for the LLM (video path; unused when use_image_input).
            processor: The AutoProcessor to use for the LLM.
            config: vLLM config. config.use_image_input: when True, select image modality
                (content type "image", multi_modal_data["image"], first frame as PIL);
                when False, select video. Aligned with VllmConfig.use_image_input
                (pipelines/video/utils/data_model.py).

        Returns:
            A dictionary containing the LLM inputs.

        """
        message = make_message(prompt, use_image=config.use_image_input)
        return make_prompt(message, frames, metadata, processor, use_image=config.use_image_input)

    @staticmethod
    def make_refined_llm_request(
        request: VllmCaptionRequest,
        processor: AutoProcessor,
        refine_prompt: str | None = None,
    ) -> VllmCaptionRequest:
        """Make a refined LLM request.

        Args:
            request: The request to refine.
            processor: The processor to use for the stage 2 prompt
            refine_prompt: An optional prompt to use to refine the caption. If
                None, the default refine prompt will be used.

        Returns:
            A refined LLM request.

        """
        _refine_prompt = _DEFAULT_REFINE_PROMPT if refine_prompt is None else refine_prompt

        if request.caption is None:
            msg = "Request caption is None"
            raise ValueError(msg)

        if "multi_modal_data" not in request.inputs:
            msg = "Message does not contain multi_modal_data"
            raise ValueError(msg)

        final_prompt = _refine_prompt + request.caption
        mm_data = request.inputs["multi_modal_data"]
        if "image" in mm_data and "video" in mm_data:
            msg = "multi_modal_data must contain one of 'image' or 'video', not both"
            raise ValueError(msg)
        if "image" not in mm_data and "video" not in mm_data:
            msg = "multi_modal_data must contain 'image' or 'video'"
            raise ValueError(msg)

        use_image = "image" in mm_data
        if use_image:
            pil_img = mm_data["image"][0]
            arr = np.array(pil_img)
            frames = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            inputs = make_prompt(make_message(final_prompt, use_image=True), frames, {}, processor, use_image=True)
        else:
            video_frames = mm_data["video"][0]
            nemotron_metadata = mm_data["video"][1]
            metadata = nemotron_metadata if isinstance(nemotron_metadata, dict) else nemotron_metadata.__dict__
            inputs = make_prompt(
                make_message(final_prompt, use_image=False), video_frames, metadata, processor, use_image=False
            )

        return VllmCaptionRequest(
            request_id=secrets.token_hex(8),
            inputs=inputs,
        )

    @staticmethod
    def decode(vllm_output: RequestOutput) -> str:
        """Decode vLLM output into a caption (extract <answer> section)."""
        return str(vllm_output.outputs[0].text)
