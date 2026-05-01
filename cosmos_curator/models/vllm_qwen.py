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
"""Qwen vLLM plugin."""

import secrets
from typing import TYPE_CHECKING, Any, TypedDict, cast

import torch
from transformers import AutoProcessor
from vllm import LLM, RequestOutput

from cosmos_curator.models.vllm_plugin import VllmPlugin
from cosmos_curator.pipelines.video.utils.data_model import VllmCaptionRequest, VllmConfig

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods


MAX_MODEL_LEN = 32768
GPU_MEMORY_UTILIZATION = 0.85
MAX_NUM_BATCHED_TOKENS = 32768
DEFAULT_BATCH_SIZE = 16
TRUST_REMOTE_CODE = False
LIMIT_MM_PER_PROMPT_VIDEO = {"images": 0, "video": 1}
LIMIT_MM_PER_PROMPT_IMAGE = {"images": 1, "video": 0}

_DEFAULT_REFINE_PROMPT = """
Improve and refine following video description. Focus on highlighting the key visual and sensory elements.
Ensure the description is clear, precise, and paints a compelling picture of the scene.
"""


class QwenContentType(TypedDict):  # noqa: D101
    type: str


class QwenContentTypeText(TypedDict):  # noqa: D101
    type: str
    text: str


class QwenMessage(TypedDict):  # noqa: D101
    role: str
    content: list[QwenContentType | QwenContentTypeText]


def make_message(
    text_input: str,
    *,
    use_image: bool = False,
) -> QwenMessage:
    """Create a message for the Qwen model.

    Args:
        text_input: The text input to create a message for.
        use_image: If True, use image content type (for image pipeline); else video.

    Returns:
        A message for the Qwen model.

    """
    content_type = "image" if use_image else "video"
    return QwenMessage(
        role="user",
        content=[
            QwenContentType(type=content_type),
            QwenContentTypeText(type="text", text=text_input),
        ],
    )


def make_prompt(
    message: QwenMessage,
    data: torch.Tensor | list[tuple[torch.Tensor, dict[str, Any]]],
    processor: AutoProcessor,
    *,
    use_image: bool = False,
) -> dict[str, Any]:
    """Make a prompt for the Qwen model.

    Args:
        message: The message to use to create the prompt
        data: The data to use for the prompt (video: list of (tensor, metadata); image: tensor 1,C,H,W).
        processor: The processor to use for the prompt.
        use_image: If True, pass data under multi_modal_data["image"] for image pipeline.

    Returns:
        A prompt for the Qwen model.

    """
    prompt_ids = processor.apply_chat_template(  # type: ignore[attr-defined]
        [message],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )[0].tolist()

    if use_image:
        # Single image: data is frames tensor (1, C, H, W); pass as-is for processor image path.
        multi_modal_data: dict[str, Any] = {"image": data}
    else:
        multi_modal_data = {"video": data}

    return {
        "prompt_token_ids": prompt_ids,
        "multi_modal_data": multi_modal_data,
    }


class VllmQwen(VllmPlugin):
    """Qwen vLLM model variant plugin."""

    @classmethod
    def model(cls, config: VllmConfig) -> LLM:
        """Instantiate the vLLM model.

        Args:
            config: Configuration for the model.

        Returns:
            The vLLM model.

        """
        quantization: QuantizationMethods | None = None
        if config.fp8:
            quantization = "fp8"

        mm_processor_kwargs = {
            "do_resize": config.preprocess,
            "do_rescale": config.preprocess,
            "do_normalize": config.preprocess,
        }

        limit_mm = LIMIT_MM_PER_PROMPT_IMAGE if config.use_image_input else LIMIT_MM_PER_PROMPT_VIDEO
        return LLM(
            model=str(cls.model_path(config)),
            limit_mm_per_prompt=limit_mm,
            quantization=quantization,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            mm_processor_kwargs=mm_processor_kwargs,
            mm_processor_cache_gb=0.0 if config.disable_mmcache else 4.0,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            tensor_parallel_size=config.num_gpus,
            trust_remote_code=TRUST_REMOTE_CODE,
            compilation_config={"cudagraph_mode": "piecewise"},
            performance_mode=config.performance_mode,
        )

    @classmethod
    def processor(cls, config: VllmConfig) -> AutoProcessor:
        """Return the AutoProcessor for the model."""
        processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            cls.model_path(config),
            trust_remote_code=TRUST_REMOTE_CODE,
            use_fast=True,
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
        """Make LLM inputs for the model.

        Args:
            prompt: The prompt to use for the LLM.
            frames: The frames to use for the LLM (video: T,C,H,W; image: 1,C,H,W).
            metadata: The metadata to use for the LLM.
            processor: The AutoProcessor to use for the LLM.
            config: vLLM config; config.use_image_input selects image vs video.

        Returns:
            A dictionary containing the LLM inputs.

        """
        message = make_message(prompt, use_image=config.use_image_input)
        data = frames if config.use_image_input else [(frames, metadata)]
        return make_prompt(message, data, processor, use_image=config.use_image_input)

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

        final_prompt = _refine_prompt + request.caption

        if "multi_modal_data" not in request.inputs:
            msg = "Message does not contain multi_modal_data"
            raise ValueError(msg)

        mm_data = request.inputs["multi_modal_data"]
        if "image" in mm_data and "video" in mm_data:
            msg = "multi_modal_data must contain one of 'image' or 'video', not both"
            raise ValueError(msg)
        if "image" not in mm_data and "video" not in mm_data:
            msg = "multi_modal_data must contain 'image' or 'video'"
            raise ValueError(msg)

        key = "image" if "image" in mm_data else "video"
        use_image = key == "image"
        message = make_message(final_prompt, use_image=use_image)
        inputs = make_prompt(message, mm_data[key], processor, use_image=use_image)

        return VllmCaptionRequest(
            request_id=secrets.token_hex(8),
            inputs=inputs,
        )

    @staticmethod
    def decode(vllm_output: RequestOutput) -> str:
        """Decode vllm output into a caption."""
        return str(vllm_output.outputs[0].text)


class VllmQwen7B(VllmQwen):
    """Qwen-7B vLLM model variant plugin."""

    @staticmethod
    def model_variant() -> str:
        """Return the model variant name."""
        return "qwen"


class VllmQwen3VL(VllmQwen):
    """Qwen3-VL vLLM model variant plugin base class."""

    @classmethod
    def model(cls, config: VllmConfig) -> LLM:
        """Instantiate the vLLM model.

        Args:
            config: Configuration for the model.

        Returns:
            The vLLM model.

        """
        limit_mm = LIMIT_MM_PER_PROMPT_IMAGE if config.use_image_input else LIMIT_MM_PER_PROMPT_VIDEO
        return LLM(
            model=str(cls.model_path(config)),
            limit_mm_per_prompt=limit_mm,
            max_model_len=MAX_MODEL_LEN,
            pipeline_parallel_size=1,
            mm_processor_cache_gb=0.0 if config.disable_mmcache else 4.0,
            tensor_parallel_size=config.num_gpus,
            trust_remote_code=TRUST_REMOTE_CODE,
            compilation_config={"cudagraph_mode": "piecewise"},
            performance_mode=config.performance_mode,
        )

    @staticmethod
    def make_llm_input(
        prompt: str,
        frames: torch.Tensor,
        metadata: dict[str, Any],
        processor: AutoProcessor,
        config: VllmConfig,
    ) -> dict[str, Any]:
        """Make LLM inputs for the model.

        Args:
            prompt: The prompt to use for the LLM.
            frames: The frames to use for the LLM (video: T,C,H,W; image: 1,C,H,W).
            metadata: The metadata to use for the LLM.
            processor: The AutoProcessor to use for the LLM.
            config: vLLM config; config.use_image_input selects image vs video.

        Returns:
            A dictionary containing the LLM inputs.

        """
        message = make_message(prompt, use_image=config.use_image_input)
        data = frames if config.use_image_input else [(frames, metadata)]
        return make_prompt(message, data, processor, use_image=config.use_image_input)


class VllmQwen3VL30B(VllmQwen3VL):
    """Qwen3-VL-30B-A3B-Instruct vLLM model variant plugin."""

    @staticmethod
    def model_variant() -> str:
        """Return the model variant name."""
        return "qwen3_vl_30b"


class VllmQwen3VL30BFP8(VllmQwen3VL):
    """Qwen3-VL-30B-A3B-Instruct-FP8 vLLM model variant plugin."""

    @staticmethod
    def model_variant() -> str:
        """Return the model variant name."""
        return "qwen3_vl_30b_fp8"


class VllmQwen3VL235B(VllmQwen3VL):
    """Qwen3-VL-235B-A22B-Instruct vLLM model variant plugin."""

    @staticmethod
    def model_variant() -> str:
        """Return the model variant name."""
        return "qwen3_vl_235b"


class VllmQwen3VL235BFP8(VllmQwen3VL):
    """Qwen3-VL-235B-A22B-Instruct-FP8 vLLM model variant plugin."""

    @staticmethod
    def model_variant() -> str:
        """Return the model variant name."""
        return "qwen3_vl_235b_fp8"


class VllmQwen3527B(VllmQwen3VL):
    """Qwen3.5-27B vLLM model variant plugin.

    This model supports video inputs in spite of not having VL in the model name.
    """

    @staticmethod
    def model_variant() -> str:
        """Return the model variant name."""
        return "qwen3_5_27b"
