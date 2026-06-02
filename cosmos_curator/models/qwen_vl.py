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

"""Qwen Video Model."""

import logging
import re
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger
from nvtx import nvtx  # type: ignore[import-untyped]

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.utils.misc import grouping
from cosmos_curator.core.utils.model import model_utils, pixi_utils

_QWEN2_5_VL_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Variant keys must match entries in ``cosmos_curator/configs/all_models.json``
# so the weight downloader can resolve them. The 30B MoE variants are the
# recommended choice on Hopper-class GPUs; the FP8 variant is a W8A8 PTQ of
# the same checkpoint and delivers ~2x throughput on native-FP8 hardware.
_QWEN_VARIANTS_INFO = {
    "qwen": _QWEN2_5_VL_MODEL_ID,
    "qwen3_vl_30b": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "qwen3_vl_30b_fp8": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
}

# Variants that skip our 28-aligned ``fetch_video`` preprocessing and instead
# expect raw uint8 TCHW frames + HF's native video processor. Qwen3-VL uses a
# 32-aligned patch grid that our preprocessor would corrupt; let HF resize/
# rescale/normalize itself. Single source of truth — both ``QwenVL.__init__``
# (for ``model_does_preprocess`` defaulting) and ``PerEventCaptionStage._call_qwen``
# (for picking the decoder path) key off this set.
QWEN_VARIANTS_NEED_RAW_FRAMES: frozenset[str] = frozenset({"qwen3_vl_30b", "qwen3_vl_30b_fp8"})

_DEFAULT_STAGE2_PROMPT = """
Improve and refine following video description. Focus on highlighting the key visual and sensory elements.
Ensure the description is clear, precise, and paints a compelling picture of the scene.\n
"""

# pyright: reportMissingImports=false
if pixi_utils.is_running_in_env("unified"):
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    if TYPE_CHECKING:
        from vllm.model_executor.layers.quantization import QuantizationMethods

    vllm_logger = logging.getLogger("vllm")
    vllm_logger.setLevel(logging.ERROR)  # Suppress warnings and info from vLLM


class QwenUtils:
    """Utility class for handling Qwen model inputs and message formatting."""

    def __init__(
        self,
        model_variant: str = "qwen",
    ) -> None:
        """Initialize the QwenUtils class.

        Args:
            model_variant: The variant of the Qwen model to use.

        """
        self.weight_file = model_utils.get_local_dir_for_weights_name(_QWEN_VARIANTS_INFO[model_variant])
        self.processor: AutoProcessor | None = None
        self._prompt_template_cache: dict[str, str] = {}

    def setup(self) -> None:
        """Set up the Qwen model.

        This method initializes the model and its configuration for processing video and text data.
        It also sets up the image processor for preprocessing video frames if needed.

        """
        self.processor = AutoProcessor.from_pretrained(self.weight_file)  # type: ignore[no-untyped-call]

    @staticmethod
    def create_message(
        text_input: str,
    ) -> list[dict[str, str | list[dict[str, str]]]]:
        """Create a message for the Qwen model.

        Args:
            text_input: The text input to create a message for.

        Returns:
            List of messages for the Qwen model.

        """
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                    },
                    {
                        "type": "text",
                        "text": text_input,
                    },
                ],
            },
        ]

    @nvtx.annotate("Generate LLM inputs")  # type: ignore[untyped-decorator]
    def generate_llm_inputs(
        self,
        prompt: str,
        video_inputs: torch.Tensor | None = None,
        video_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate inputs for the Qwen language model from video and text data.

        Processes video and text inputs to create the input for the Qwen model. It handles both video and
        image inputs, decoding video and applying preprocessing if needed, and creates a structured
        input dictionary containing the processed prompt and multimodal data. The applied chat template
        is cached per prompt string so repeated calls with the same prompt (e.g. captioning) reuse it;
        different prompts (e.g. type then filter in combined prep) each get their own template.

        Args:
            prompt: Text prompt to be included with the input.
            video_inputs: Pre-processed video inputs. If None, and video data is to be passed to
                          the model, then video cannot be None.
            video_metadata: Optional HF-style video metadata dict
                (``fps``, ``total_num_frames``, ``duration``,
                ``frames_indices``, ``do_sample_frames``, ``video_backend``).
                Required by Qwen3-VL; ignored by Qwen2.5-VL.

        Returns:
            dict containing:
                - "prompt": The processed text prompt with chat template applied
                - "multi_modal_data": Dictionary containing processed "image" and/or "video" inputs

        """
        if video_inputs is None:
            error_msg = "No input frames provided, cannot call process_vision_info"
            raise ValueError(error_msg)

        if prompt in self._prompt_template_cache:
            text_prompt = self._prompt_template_cache[prompt]
        else:
            messages = self.create_message(prompt)
            assert self.processor is not None
            text_prompt = self.processor.apply_chat_template(  # type: ignore[attr-defined]
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            self._prompt_template_cache[prompt] = text_prompt

        # Qwen3-VL requires a ``(tensor, metadata)`` tuple; Qwen2.5-VL accepts
        # either and discards the metadata. Always tuple when we have it.
        video_item: torch.Tensor | tuple[torch.Tensor, dict[str, Any]] = video_inputs
        if video_metadata is not None:
            video_item = (video_inputs, video_metadata)

        mm_data = {}
        mm_data["video"] = [video_item]
        return {
            "prompt": text_prompt,
            "multi_modal_data": mm_data,
        }


class QwenVL(ModelInterface):
    """Interface for Qwen vision-language model for video understanding and captioning."""

    def __init__(  # noqa: PLR0913
        self,
        model_variant: str = "qwen",
        *,
        fp8: bool = True,
        max_output_tokens: int = 8192,
        model_does_preprocess: bool | None = None,
        stage2_prompt_text: str | None = None,
        disable_mmcache: bool = False,
        num_gpus: int = 1,
    ) -> None:
        """Initialize the QwenVL model.

        Args:
            model_variant: The variant of the Qwen model to use.
            fp8: Whether to use FP8 quantization.
            max_output_tokens: The maximum number of tokens to generate.
            model_does_preprocess: Whether vLLM's HF mm-processor should
                resize/rescale/normalize the video. ``None`` picks a
                per-variant default: ``False`` for Qwen2.5-VL (28-aligned
                patch grid; callers pre-process with ``fetch_video``) and
                ``True`` for Qwen3-VL (32-aligned grid; let HF do it).
            stage2_prompt_text: The prompt for the stage 2 caption.
            disable_mmcache: Whether to disable the MM cache.
            num_gpus: Number of GPUs to use for processing.

        """
        super().__init__()
        self._weights_name = _QWEN_VARIANTS_INFO[model_variant]
        self.weight_file = str(model_utils.get_local_dir_for_weights_name(self._weights_name))
        self.fp8 = fp8
        self.max_output_tokens = max_output_tokens
        if model_does_preprocess is None:
            model_does_preprocess = model_variant in QWEN_VARIANTS_NEED_RAW_FRAMES
        self.model_does_preprocess = model_does_preprocess
        self.disable_mmcache = disable_mmcache
        self.llm: LLM | None = None
        self.model_variant = model_variant
        self.sampling_params: SamplingParams | None = None
        self.num_gpus = num_gpus
        self.pattern = (
            r"(<\|im_start\|>user\s*<\|vision_start\|><\|video_pad\|><\|vision_end\|>\s*)(.*?)(\s*<\|im_end\|>)"
        )
        self.stage2_prompt: str = _DEFAULT_STAGE2_PROMPT
        if stage2_prompt_text is not None:
            self.stage2_prompt = stage2_prompt_text

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name.

        Returns:
            The conda environment name.

        """
        return "unified"

    @property
    def model_id_names(self) -> list[str]:
        """Get the model ID names.

        Returns:
            A list of model ID names.

        """
        return [self._weights_name]

    @nvtx.annotate("Setup Qwen model")  # type: ignore[untyped-decorator]
    def setup(self) -> None:
        """Set up the Qwen model.

        This method initializes the model and its configuration for processing video and text data.
        It also sets up the image processor for preprocessing video frames if needed.

        """
        logger.info("Setting up Qwen model")
        mm_processor_kwargs = {
            "do_resize": self.model_does_preprocess,
            "do_rescale": self.model_does_preprocess,
            "do_normalize": self.model_does_preprocess,
        }

        quantization: QuantizationMethods | None = None
        if self.fp8:
            quantization = "fp8"

        self.llm = LLM(
            model=self.weight_file,
            limit_mm_per_prompt={"image": 0, "video": 1},
            quantization=quantization,
            max_model_len=32768,
            gpu_memory_utilization=0.85,
            mm_processor_kwargs=mm_processor_kwargs,
            mm_processor_cache_gb=0.0 if self.disable_mmcache else 4.0,
            max_num_batched_tokens=32768,
            tensor_parallel_size=self.num_gpus,
        )
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=self.max_output_tokens,
            stop_token_ids=[],
        )

        logger.info(
            "CUDA graph enabled for sequences smaller than 16k tokens; adjust accordingly for even longer sequences",
        )

    @nvtx.annotate("Qwen Generate tokens")  # type: ignore[untyped-decorator]
    def generate(
        self,
        videos: list[dict[str, Any]],
        *,
        generate_stage2_caption: bool = False,
        batch_size: int = 16,
    ) -> list[str]:
        """Generate text for a list of videos.

        Args:
            videos: List of input dictionaries for the LLM.
            generate_stage2_caption: Whether to generate a stage 2 caption.
            batch_size: Batch size for processing.

        Returns:
            List of generated captions.

        """
        generated_text = []
        for batch_videos in grouping.split_by_chunk_size(videos, batch_size):
            llm_inputs = list(batch_videos)

            try:
                assert self.llm is not None
                assert self.sampling_params is not None
                outputs = self.llm.generate(
                    llm_inputs,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )

                if generate_stage2_caption:
                    for i, out in enumerate(outputs):
                        updated_prompt = self.stage2_prompt + out.outputs[0].text
                        llm_inputs[i]["prompt"] = re.sub(
                            self.pattern,
                            rf"\1{updated_prompt}\3",
                            llm_inputs[i]["prompt"],
                            flags=re.DOTALL,
                        )

                    outputs = self.llm.generate(
                        llm_inputs,
                        sampling_params=self.sampling_params,
                        use_tqdm=False,
                    )

                generated_text.extend([out.outputs[0].text for out in outputs])

            except Exception as e:
                logger.exception(f"Error generating text for batch of {len(batch_videos)}: {e}")
                raise

        return generated_text
