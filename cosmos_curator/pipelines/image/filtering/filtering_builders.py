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

"""Builder functions for local image semantic filtering and classification."""

import attrs

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.pipelines.common.filter_prompts import (
    IMAGE_FILTER_CRITERIA,
    IMAGE_TYPE_LABELS,
    get_image_filter_prompt,
)
from cosmos_curator.pipelines.common.model_constraints import resolve_preprocess_mode
from cosmos_curator.pipelines.common.semantic_filter_postprocess import custom_categories_union, read_categories_file
from cosmos_curator.pipelines.image.captioning.image_api_caption_stages import (
    ImageGeminiCaptionStage,
    ImageOpenAICaptionStage,
    ImageOpenAIPrepStage,
)
from cosmos_curator.pipelines.image.captioning.image_vllm_stages import ImageVllmCaptionStage, ImageVllmPrepStage
from cosmos_curator.pipelines.image.filtering.filter_stages import ImageClassifierStage, ImageSemanticFilterStage
from cosmos_curator.pipelines.video.utils.data_model import VllmConfig, VllmSamplingConfig


@attrs.define(frozen=True)
class ImageSemanticFilterConfig:
    """Configuration for local image semantic filtering."""

    enabled: bool = False
    score_only: bool = False
    model_variant: str = "qwen"
    filter_categories: str | None = None
    prompt_variant: str = "default"
    rejection_threshold: float = 0.5
    batch_size: int = 16
    max_output_tokens: int = 8192
    num_gpus: int = 1
    openai_model_name: str = "auto"
    openai_max_caption_retries: int = 3
    openai_retry_delay_seconds: float = 1.0
    gemini_model_name: str = "models/gemini-2.5-pro"
    gemini_max_caption_retries: int = 3
    gemini_retry_delay_seconds: float = 1.0
    caption_prep_min_pixels: int | None = None
    caption_prep_max_pixels: int | None = None
    num_prep_workers_per_node: int = 2
    verbose: bool = False
    perf_profile: bool = False


@attrs.define(frozen=True)
class ImageClassifierConfig:
    """Configuration for local image classification/filtering."""

    enabled: bool = False
    model_variant: str = "qwen"
    rejection_threshold: float = 0.5
    batch_size: int = 16
    max_output_tokens: int = 8192
    num_gpus: int = 1
    openai_model_name: str = "auto"
    openai_max_caption_retries: int = 3
    openai_retry_delay_seconds: float = 1.0
    gemini_model_name: str = "models/gemini-2.5-pro"
    gemini_max_caption_retries: int = 3
    gemini_retry_delay_seconds: float = 1.0
    caption_prep_min_pixels: int | None = None
    caption_prep_max_pixels: int | None = None
    num_prep_workers_per_node: int = 2
    verbose: bool = False
    perf_profile: bool = False
    type_allow: str | None = None
    type_block: str | None = None
    custom_categories: bool = False
    type_allow_file: str | None = None
    type_block_file: str | None = None


def _make_image_vllm_config(
    *,
    model_variant: str,
    prompt_text: str,
    num_gpus: int,
    batch_size: int,
    max_output_tokens: int,
) -> VllmConfig:
    return VllmConfig(
        model_variant=model_variant,
        use_image_input=True,
        num_gpus=num_gpus,
        batch_size=batch_size,
        prompt_variant="image",
        prompt_text=prompt_text,
        preprocess_mode=resolve_preprocess_mode(model_variant),
        sampling_config=VllmSamplingConfig(max_tokens=max_output_tokens),
    )


def _build_semantic_filter_stages(config: ImageSemanticFilterConfig) -> list[CuratorStage | CuratorStageSpec]:
    prompt_text = get_image_filter_prompt(config.prompt_variant, config.filter_categories, verbose=config.verbose)
    if config.model_variant in {"openai", "gemini"}:
        return _build_external_semantic_filter_stages(config, prompt_text)

    filter_caption_key = f"semantic:{config.model_variant}"
    vllm_config = _make_image_vllm_config(
        model_variant=config.model_variant,
        prompt_text=prompt_text,
        num_gpus=config.num_gpus,
        batch_size=config.batch_size,
        max_output_tokens=config.max_output_tokens,
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
                result_target="filter_caption",
                result_key=filter_caption_key,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            ),
            num_setup_attempts_python=None,
        ),
        CuratorStageSpec(
            ImageSemanticFilterStage(
                model_variant=config.model_variant,
                filter_caption_key=filter_caption_key,
                user_prompt=config.filter_categories,
                filter_variant=config.prompt_variant,
                rejection_threshold=config.rejection_threshold,
                criteria_by_variant=IMAGE_FILTER_CRITERIA,
                score_only=config.score_only,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            )
        ),
    ]


def _build_classifier_stages(config: ImageClassifierConfig) -> list[CuratorStage | CuratorStageSpec]:
    type_allow = config.type_allow or read_categories_file(config.type_allow_file)
    type_block = config.type_block or read_categories_file(config.type_block_file)
    has_custom = config.custom_categories or bool(config.type_allow_file or config.type_block_file)
    prompt_text = get_image_filter_prompt(
        "type",
        custom_categories_union(type_allow, type_block) if has_custom else None,
        verbose=config.verbose,
    )
    if config.model_variant in {"openai", "gemini"}:
        return _build_external_classifier_stages(
            config,
            prompt_text,
            type_allow,
            type_block,
            has_custom=has_custom,
        )

    filter_caption_key = f"classifier:{config.model_variant}"
    vllm_config = _make_image_vllm_config(
        model_variant=config.model_variant,
        prompt_text=prompt_text,
        num_gpus=config.num_gpus,
        batch_size=config.batch_size,
        max_output_tokens=config.max_output_tokens,
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
                result_target="filter_caption",
                result_key=filter_caption_key,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            ),
            num_setup_attempts_python=None,
        ),
        CuratorStageSpec(
            ImageClassifierStage(
                model_variant=config.model_variant,
                filter_caption_key=filter_caption_key,
                rejection_threshold=config.rejection_threshold,
                type_allow=type_allow,
                type_block=type_block,
                custom_categories=has_custom,
                valid_type_labels=IMAGE_TYPE_LABELS,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            )
        ),
    ]


def _build_external_semantic_filter_stages(
    config: ImageSemanticFilterConfig,
    prompt_text: str,
) -> list[CuratorStage | CuratorStageSpec]:
    """Build image semantic filtering stages for external OpenAI/Gemini endpoints."""
    filter_caption_key = f"semantic:{config.model_variant}"
    stages: list[CuratorStage | CuratorStageSpec] = []
    if config.model_variant == "openai":
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
                    model_variant="openai",
                    prompt_variant=config.prompt_variant,
                    prompt_text=prompt_text,
                    max_output_tokens=config.max_output_tokens,
                    batch_size=config.batch_size,
                    endpoint_key="filter",
                    result_target="filter_caption",
                    result_key=filter_caption_key,
                    max_caption_retries=config.openai_max_caption_retries,
                    retry_delay_seconds=config.openai_retry_delay_seconds,
                    verbose=config.verbose,
                    log_stats=config.perf_profile,
                )
            )
        )
    else:
        stages.append(
            CuratorStageSpec(
                ImageGeminiCaptionStage(
                    model_variant="gemini",
                    model_name=config.gemini_model_name,
                    prompt_variant=config.prompt_variant,
                    prompt_text=prompt_text,
                    max_output_tokens=config.max_output_tokens,
                    batch_size=config.batch_size,
                    result_target="filter_caption",
                    result_key=filter_caption_key,
                    max_caption_retries=config.gemini_max_caption_retries,
                    retry_delay_seconds=config.gemini_retry_delay_seconds,
                    verbose=config.verbose,
                    log_stats=config.perf_profile,
                )
            )
        )
    stages.append(
        CuratorStageSpec(
            ImageSemanticFilterStage(
                model_variant=config.model_variant,
                filter_caption_key=filter_caption_key,
                user_prompt=config.filter_categories,
                filter_variant=config.prompt_variant,
                rejection_threshold=config.rejection_threshold,
                criteria_by_variant=IMAGE_FILTER_CRITERIA,
                score_only=config.score_only,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            )
        )
    )
    return stages


def _build_external_classifier_stages(
    config: ImageClassifierConfig,
    prompt_text: str,
    type_allow: str | None,
    type_block: str | None,
    *,
    has_custom: bool,
) -> list[CuratorStage | CuratorStageSpec]:
    """Build image classifier stages for external OpenAI/Gemini endpoints."""
    filter_caption_key = f"classifier:{config.model_variant}"
    stages: list[CuratorStage | CuratorStageSpec] = []
    if config.model_variant == "openai":
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
                    model_variant="openai",
                    prompt_variant="type",
                    prompt_text=prompt_text,
                    max_output_tokens=config.max_output_tokens,
                    batch_size=config.batch_size,
                    endpoint_key="classifier",
                    result_target="filter_caption",
                    result_key=filter_caption_key,
                    max_caption_retries=config.openai_max_caption_retries,
                    retry_delay_seconds=config.openai_retry_delay_seconds,
                    verbose=config.verbose,
                    log_stats=config.perf_profile,
                )
            )
        )
    else:
        stages.append(
            CuratorStageSpec(
                ImageGeminiCaptionStage(
                    model_variant="gemini",
                    model_name=config.gemini_model_name,
                    prompt_variant="type",
                    prompt_text=prompt_text,
                    max_output_tokens=config.max_output_tokens,
                    batch_size=config.batch_size,
                    result_target="filter_caption",
                    result_key=filter_caption_key,
                    max_caption_retries=config.gemini_max_caption_retries,
                    retry_delay_seconds=config.gemini_retry_delay_seconds,
                    verbose=config.verbose,
                    log_stats=config.perf_profile,
                )
            )
        )
    stages.append(
        CuratorStageSpec(
            ImageClassifierStage(
                model_variant=config.model_variant,
                filter_caption_key=filter_caption_key,
                rejection_threshold=config.rejection_threshold,
                type_allow=type_allow,
                type_block=type_block,
                custom_categories=has_custom,
                valid_type_labels=IMAGE_TYPE_LABELS,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            )
        )
    )
    return stages


def build_image_filter_classifier_stages(
    filter_config: ImageSemanticFilterConfig | None = None,
    classifier_config: ImageClassifierConfig | None = None,
) -> list[CuratorStage | CuratorStageSpec]:
    """Build stages for local image semantic filtering and/or classification."""
    if filter_config is None and classifier_config is None:
        msg = "At least one of filter_config or classifier_config is required"
        raise ValueError(msg)
    stages: list[CuratorStage | CuratorStageSpec] = []
    if filter_config is not None and filter_config.enabled:
        stages.extend(_build_semantic_filter_stages(filter_config))
    if classifier_config is not None and classifier_config.enabled:
        stages.extend(_build_classifier_stages(classifier_config))
    return stages
