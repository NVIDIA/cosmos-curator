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
"""vLLM plugin for Cosmos-Reason1 vision-language model."""

import re
import secrets
from typing import TYPE_CHECKING, Any, cast

import torch
from transformers import AutoProcessor
from vllm import LLM, RequestOutput
from vllm.config import CompilationConfig
from vllm.engine.arg_utils import AsyncEngineArgs

from cosmos_curator.models.vllm_plugin import VllmPlugin
from cosmos_curator.pipelines.video.utils.data_model import VllmAsyncConfig, VllmCaptionRequest, VllmConfig

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods


# Constants tuned similarly to existing plugins
GPU_MEMORY_UTILIZATION = 0.85
MAX_NUM_BATCHED_TOKENS = 32768
MAX_MODEL_LEN = 32768
TRUST_REMOTE_CODE = False
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


def _extract_from_reasoning_format(text: str) -> str:
    """Extract the <answer>...</answer> content if present.

    Falls back to the original text if the reasoning format is missing.
    """
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def make_message(text_input: str, *, use_image: bool = False) -> list[dict[str, Any]]:
    """Create a chat message structure for Cosmos-Reason1.

    The system prompt instructs the reasoning format. The user content
    includes a placeholder (image or video) and the user's text prompt.

    Args:
        text_input: The user text prompt.
        use_image: When True, select image modality (content type "image"); when False,
            select video. Aligned with VllmConfig.use_image_input.

    """
    content_type = "image" if use_image else "video"
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the question in the following format: "
                "<think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": content_type},
                {"type": "text", "text": text_input},
            ],
        },
    ]


def make_prompt(
    message: list[dict[str, Any]],
    frames: torch.Tensor,
    metadata: dict[str, Any],
    processor: AutoProcessor,
    *,
    use_image: bool = False,
) -> dict[str, Any]:
    """Create a prompt payload for vLLM using the processor chat template.

    Args:
        message: Chat message structure (e.g. from make_message) for the processor.
        frames: Video or image tensor (1, C, H, W) or (T, C, H, W).
        metadata: Video metadata (used when use_image is False).
        processor: HF AutoProcessor with apply_chat_template.
        use_image: When True, select image modality (multi_modal_data["image"]);
            when False, select video (multi_modal_data["video"] with metadata).
            Aligned with VllmConfig.use_image_input.

    Returns:
        Dict with "prompt" and "multi_modal_data" for vLLM.

    """
    apply_chat_template = getattr(processor, "apply_chat_template", None)
    if apply_chat_template is None:
        msg = "Processor does not support apply_chat_template"
        raise ValueError(msg)

    prompt_str = apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    )

    if use_image:
        # HF image processor expects tensor or list of tensors, not (tensor, metadata) tuples.
        multi_modal_data: dict[str, Any] = {"image": frames}
    else:
        multi_modal_data = {"video": [(frames, metadata)]}

    return {
        "prompt": cast("str", prompt_str),
        "multi_modal_data": multi_modal_data,
    }


class VllmCosmosReason1VL(VllmPlugin):
    """Cosmos-Reason1 vLLM model variant plugin."""

    @staticmethod
    def model_variant() -> str:
        """Return the model variant name."""
        return "cosmos_r1"

    @classmethod
    def model(cls, config: VllmConfig) -> LLM:
        """Instantiate the vLLM model for Cosmos-Reason1.

        Args:
            config: Configuration for the model.

        Returns:
            The vLLM model.

        """
        quantization: QuantizationMethods | None = None
        if config.fp8:
            quantization = "fp8"

        mm_processor_kwargs = {
            "do_resize": config.model_preprocess_enabled,
            "do_rescale": config.model_preprocess_enabled,
            "do_normalize": config.model_preprocess_enabled,
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
    def model_async(cls, config: VllmAsyncConfig) -> AsyncEngineArgs:
        """Build ``AsyncEngineArgs`` for Cosmos-Reason1 in-process ``AsyncLLM``.

        Mirrors :meth:`model` - reads from module-scope constants.
        """
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
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
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
            quantization="fp8" if config.fp8 else None,
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
            cls.model_path(config),
            trust_remote_code=TRUST_REMOTE_CODE,
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
                (content type "image", multi_modal_data["image"]); when False, select video.
                Aligned with VllmConfig.use_image_input (pipelines/video/utils/data_model.py).

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
            image_data = mm_data["image"]
            inputs = make_prompt(make_message(final_prompt, use_image=True), image_data, {}, processor, use_image=True)
        else:
            video_data = mm_data["video"][0]
            video_frames, video_metadata = video_data
            inputs = make_prompt(make_message(final_prompt), video_frames, video_metadata, processor)
        if "mm_processor_kwargs" in request.inputs:
            inputs["mm_processor_kwargs"] = request.inputs["mm_processor_kwargs"]

        return VllmCaptionRequest(
            request_id=secrets.token_hex(8),
            inputs=inputs,
        )

    @staticmethod
    def decode(vllm_output: RequestOutput) -> str:
        """Decode vLLM output into a caption (extract <answer> section)."""
        return _extract_from_reasoning_format(vllm_output.outputs[0].text)
