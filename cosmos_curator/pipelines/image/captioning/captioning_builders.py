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

"""Builder functions for image captioning stages."""

import attrs

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.pipelines.common.model_constraints import MODEL_VARIANTS_REQUIRING_PREPROCESS
from cosmos_curator.pipelines.image.captioning.image_api_caption_stages import (
    ImageGeminiCaptionStage,
    ImageOpenAICaptionStage,
    ImageOpenAIPrepStage,
)
from cosmos_curator.pipelines.image.captioning.image_vllm_stages import ImageVllmCaptionStage, ImageVllmPrepStage
from cosmos_curator.pipelines.video.utils.data_model import VllmConfig, VllmSamplingConfig

IMAGE_CAPTION_ALGOS: frozenset[str] = frozenset(
    {"qwen", "qwen3_5_27b", "qwen3_vl_30b", "qwen3_vl_30b_fp8", "qwen3_vl_235b", "qwen3_vl_235b_fp8"}
    | {"nemotron", "cosmos_r1", "cosmos_r2", "openai", "gemini"}
)


@attrs.define(frozen=True)
class ImageCaptioningConfig:
    """Configuration for image captioning (prep + vLLM caption)."""

    caption_algo: str = "qwen"
    num_gpus: int = 1
    num_prep_workers_per_node: int = 2
    batch_size: int = 4
    max_output_tokens: int = 8192
    prompt_variant: str = "image"
    prompt_text: str | None = None
    stage2_caption: bool = False
    stage2_prompt_text: str | None = None
    caption_prep_min_pixels: int | None = None
    caption_prep_max_pixels: int | None = None
    openai_raw_image: bool = False
    openai_model_name: str = "auto"
    openai_caption_retries: int = 3
    openai_retry_delay_seconds: float = 1.0
    gemini_model_name: str = "models/gemini-2.5-pro"
    gemini_caption_retries: int = 3
    gemini_retry_delay_seconds: float = 1.0
    verbose: bool = False
    perf_profile: bool = False


def build_image_captioning_stages(config: ImageCaptioningConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Build prep + caption stages for image captioning."""
    if config.caption_algo not in IMAGE_CAPTION_ALGOS:
        msg = f"caption_algo must be one of {sorted(IMAGE_CAPTION_ALGOS)}, got {config.caption_algo!r}"
        raise ValueError(msg)

    if config.caption_algo == "gemini":
        return [
            CuratorStageSpec(
                ImageGeminiCaptionStage(
                    model_name=config.gemini_model_name,
                    prompt_variant=config.prompt_variant,
                    prompt_text=config.prompt_text,
                    max_output_tokens=config.max_output_tokens,
                    max_caption_retries=config.gemini_caption_retries,
                    retry_delay_seconds=config.gemini_retry_delay_seconds,
                    batch_size=config.batch_size,
                    verbose=config.verbose,
                    log_stats=config.perf_profile,
                )
            )
        ]

    if config.caption_algo == "openai":
        stages: list[CuratorStage | CuratorStageSpec] = []
        if not config.openai_raw_image:
            stages.append(
                CuratorStageSpec(
                    ImageOpenAIPrepStage(
                        caption_prep_min_pixels=config.caption_prep_min_pixels,
                        caption_prep_max_pixels=config.caption_prep_max_pixels,
                        verbose=config.verbose,
                        log_stats=config.perf_profile,
                    ),
                    num_workers_per_node=config.num_prep_workers_per_node,
                )
            )
        stages.append(
            CuratorStageSpec(
                ImageOpenAICaptionStage(
                    model_name=config.openai_model_name,
                    prompt_variant=config.prompt_variant,
                    prompt_text=config.prompt_text,
                    max_output_tokens=config.max_output_tokens,
                    max_caption_retries=config.openai_caption_retries,
                    retry_delay_seconds=config.openai_retry_delay_seconds,
                    batch_size=config.batch_size,
                    verbose=config.verbose,
                    log_stats=config.perf_profile,
                )
            )
        )
        return stages

    model_does_preprocess = config.caption_algo in MODEL_VARIANTS_REQUIRING_PREPROCESS
    vllm_config = VllmConfig(
        model_variant=config.caption_algo,
        use_image_input=True,
        num_gpus=config.num_gpus,
        batch_size=config.batch_size,
        prompt_variant=config.prompt_variant,
        prompt_text=config.prompt_text,
        sampling_config=VllmSamplingConfig(max_tokens=config.max_output_tokens),
        stage2_caption=config.stage2_caption,
        stage2_prompt_text=config.stage2_prompt_text,
        preprocess=model_does_preprocess,
    )
    return [
        CuratorStageSpec(
            ImageVllmPrepStage(
                vllm_config=vllm_config,
                caption_prep_min_pixels=config.caption_prep_min_pixels,
                caption_prep_max_pixels=config.caption_prep_max_pixels,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            ),
            num_workers_per_node=config.num_prep_workers_per_node,
        ),
        CuratorStageSpec(
            ImageVllmCaptionStage(
                vllm_config=vllm_config,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            ),
            num_setup_attempts_python=None,
        ),
    ]
