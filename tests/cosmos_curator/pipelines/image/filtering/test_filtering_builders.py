# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for image semantic filter/classifier stage builders."""

import pytest

from cosmos_curator.core.interfaces.stage_interface import CuratorStageSpec
from cosmos_curator.core.utils.config.config import ConfigFileData, Gemini
from cosmos_curator.pipelines.image.captioning import image_api_caption_stages
from cosmos_curator.pipelines.image.captioning.image_api_caption_stages import (
    ImageGeminiCaptionStage,
    ImageOpenAICaptionStage,
    ImageOpenAIPrepStage,
)
from cosmos_curator.pipelines.image.captioning.image_vllm_stages import ImageVllmCaptionStage, ImageVllmPrepStage
from cosmos_curator.pipelines.image.filtering.filter_stages import ImageClassifierStage, ImageSemanticFilterStage
from cosmos_curator.pipelines.image.filtering.filtering_builders import (
    ImageClassifierConfig,
    ImageSemanticFilterConfig,
    build_image_filter_classifier_stages,
)


def test_build_image_filter_stages_returns_local_vllm_and_shared_postprocess() -> None:
    """Semantic filter builder should return local vLLM prep/inference plus shared postprocess."""
    stages = build_image_filter_classifier_stages(filter_config=ImageSemanticFilterConfig(enabled=True))

    assert len(stages) == 3
    assert isinstance(stages[0], CuratorStageSpec)
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[2], CuratorStageSpec)
    assert isinstance(stages[0].stage, ImageVllmPrepStage)
    assert isinstance(stages[1].stage, ImageVllmCaptionStage)
    assert isinstance(stages[2].stage, ImageSemanticFilterStage)
    assert stages[1].stage._result_target == "filter_caption"


def test_build_image_classifier_stages_returns_local_vllm_and_shared_postprocess() -> None:
    """Classifier builder should return local vLLM prep/inference plus shared postprocess."""
    stages = build_image_filter_classifier_stages(
        classifier_config=ImageClassifierConfig(enabled=True, custom_categories=True, type_allow="planet_earth")
    )

    assert len(stages) == 3
    assert isinstance(stages[0], CuratorStageSpec)
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[2], CuratorStageSpec)
    assert isinstance(stages[0].stage, ImageVllmPrepStage)
    assert isinstance(stages[1].stage, ImageVllmCaptionStage)
    assert isinstance(stages[2].stage, ImageClassifierStage)
    assert stages[1].stage._result_target == "filter_caption"


def test_build_image_filter_classifier_requires_a_config() -> None:
    """Builder should require at least one filter or classifier config."""
    with pytest.raises(ValueError, match="At least one of filter_config or classifier_config is required"):
        build_image_filter_classifier_stages()


def test_build_image_filter_classifier_stages_orders_filter_before_classifier() -> None:
    """Semantic filter stages should run before classifier stages and use distinct caption keys."""
    stages = build_image_filter_classifier_stages(
        filter_config=ImageSemanticFilterConfig(enabled=True, model_variant="qwen"),
        classifier_config=ImageClassifierConfig(enabled=True, model_variant="qwen"),
    )

    assert len(stages) == 6
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[2], CuratorStageSpec)
    assert isinstance(stages[2].stage, ImageSemanticFilterStage)
    assert isinstance(stages[4], CuratorStageSpec)
    assert isinstance(stages[5], CuratorStageSpec)
    assert isinstance(stages[5].stage, ImageClassifierStage)
    assert stages[1].stage._result_key == "semantic:qwen"
    assert stages[2].stage._filter_caption_key == "semantic:qwen"
    assert stages[4].stage._result_key == "classifier:qwen"
    assert stages[5].stage._filter_caption_key == "classifier:qwen"


def test_build_image_filter_stages_returns_openai_endpoint_chain() -> None:
    """OpenAI semantic filter should build prep + endpoint caption + postprocess."""
    stages = build_image_filter_classifier_stages(
        filter_config=ImageSemanticFilterConfig(enabled=True, model_variant="openai")
    )

    assert len(stages) == 3
    assert isinstance(stages[0], CuratorStageSpec)
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[2], CuratorStageSpec)
    assert isinstance(stages[0].stage, ImageOpenAIPrepStage)
    assert isinstance(stages[1].stage, ImageOpenAICaptionStage)
    assert stages[1].stage._endpoint_key == "filter"
    assert stages[1].stage.stage_batch_size == 16
    assert stages[1].stage._result_target == "filter_caption"
    assert stages[1].stage._result_key == "semantic:openai"
    assert isinstance(stages[2].stage, ImageSemanticFilterStage)
    assert stages[2].stage._filter_caption_key == "semantic:openai"


def test_build_image_filter_stages_returns_gemini_endpoint_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini semantic filter should build endpoint caption + postprocess."""
    monkeypatch.setattr(image_api_caption_stages, "load_config", lambda: ConfigFileData(gemini=Gemini(api_key="k")))
    stages = build_image_filter_classifier_stages(
        filter_config=ImageSemanticFilterConfig(enabled=True, model_variant="gemini")
    )

    assert len(stages) == 2
    assert isinstance(stages[0], CuratorStageSpec)
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[0].stage, ImageGeminiCaptionStage)
    assert stages[0].stage.stage_batch_size == 16
    assert stages[0].stage._result_target == "filter_caption"
    assert stages[0].stage._result_key == "semantic:gemini"
    assert isinstance(stages[1].stage, ImageSemanticFilterStage)
    assert stages[1].stage._filter_caption_key == "semantic:gemini"


def test_build_image_classifier_stages_returns_openai_endpoint_chain() -> None:
    """OpenAI classifier should build prep + endpoint caption + postprocess."""
    stages = build_image_filter_classifier_stages(
        classifier_config=ImageClassifierConfig(
            enabled=True,
            model_variant="openai",
            custom_categories=True,
            type_allow="planet_earth",
        )
    )

    assert len(stages) == 3
    assert isinstance(stages[0], CuratorStageSpec)
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[2], CuratorStageSpec)
    assert isinstance(stages[0].stage, ImageOpenAIPrepStage)
    assert isinstance(stages[1].stage, ImageOpenAICaptionStage)
    assert stages[1].stage._endpoint_key == "classifier"
    assert stages[1].stage.stage_batch_size == 16
    assert stages[1].stage._result_target == "filter_caption"
    assert stages[1].stage._result_key == "classifier:openai"
    assert isinstance(stages[2].stage, ImageClassifierStage)
    assert stages[2].stage._filter_caption_key == "classifier:openai"


def test_build_image_classifier_stages_returns_gemini_endpoint_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini classifier should build endpoint caption + postprocess."""
    monkeypatch.setattr(image_api_caption_stages, "load_config", lambda: ConfigFileData(gemini=Gemini(api_key="k")))
    stages = build_image_filter_classifier_stages(
        classifier_config=ImageClassifierConfig(
            enabled=True,
            model_variant="gemini",
            custom_categories=True,
            type_allow="planet_earth",
        )
    )

    assert len(stages) == 2
    assert isinstance(stages[0], CuratorStageSpec)
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[0].stage, ImageGeminiCaptionStage)
    assert stages[0].stage.stage_batch_size == 16
    assert stages[0].stage._result_target == "filter_caption"
    assert stages[0].stage._result_key == "classifier:gemini"
    assert isinstance(stages[1].stage, ImageClassifierStage)
    assert stages[1].stage._filter_caption_key == "classifier:gemini"
