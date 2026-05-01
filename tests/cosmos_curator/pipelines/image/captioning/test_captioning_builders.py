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

"""Tests for image captioning stage builders."""

import pytest

from cosmos_curator.core.interfaces.stage_interface import CuratorStageSpec
from cosmos_curator.core.utils.config.config import ConfigFileData, Gemini
from cosmos_curator.pipelines.image.captioning import image_api_caption_stages
from cosmos_curator.pipelines.image.captioning.captioning_builders import (
    ImageCaptioningConfig,
    build_image_captioning_stages,
)
from cosmos_curator.pipelines.image.captioning.image_api_caption_stages import (
    ImageGeminiCaptionStage,
    ImageOpenAICaptionStage,
    ImageOpenAIPrepStage,
)
from cosmos_curator.pipelines.image.captioning.image_vllm_stages import ImageVllmCaptionStage, ImageVllmPrepStage


def test_build_image_captioning_stages_returns_prep_and_caption_specs() -> None:
    """The captioning builder should return prep and caption stage specs."""
    stages = build_image_captioning_stages(
        ImageCaptioningConfig(
            caption_algo="qwen",
            num_gpus=1,
            num_prep_workers_per_node=4,
            batch_size=8,
            max_output_tokens=123,
            prompt_variant="image",
        )
    )

    assert len(stages) == 2
    assert all(isinstance(stage, CuratorStageSpec) for stage in stages)
    assert isinstance(stages[0].stage, ImageVllmPrepStage)
    assert isinstance(stages[1].stage, ImageVllmCaptionStage)
    assert stages[0].num_workers_per_node == 4
    assert stages[1].num_setup_attempts_python is None


def test_build_image_captioning_stages_rejects_unknown_algorithm() -> None:
    """The captioning builder should reject unsupported image caption algorithms."""
    with pytest.raises(ValueError, match="caption_algo must be one of"):
        build_image_captioning_stages(ImageCaptioningConfig(caption_algo="not-a-model"))


def test_build_image_captioning_stages_enables_model_preprocess_for_qwen3() -> None:
    """Qwen3 image captioning should force model-side preprocessing like the video pipeline."""
    stages = build_image_captioning_stages(ImageCaptioningConfig(caption_algo="qwen3_vl_30b"))

    prep_stage = stages[0]
    caption_stage = stages[1]
    assert isinstance(prep_stage, CuratorStageSpec)
    assert isinstance(caption_stage, CuratorStageSpec)
    assert isinstance(prep_stage.stage, ImageVllmPrepStage)
    assert isinstance(caption_stage.stage, ImageVllmCaptionStage)
    assert prep_stage.stage._vllm_config.preprocess is True
    assert caption_stage.stage._vllm_config.preprocess is True


def test_build_image_captioning_stages_leaves_qwen25_preprocess_disabled() -> None:
    """Qwen2.5 should keep the existing image-pipeline preprocess behavior."""
    stages = build_image_captioning_stages(ImageCaptioningConfig(caption_algo="qwen"))

    prep_stage = stages[0]
    assert isinstance(prep_stage, CuratorStageSpec)
    assert isinstance(prep_stage.stage, ImageVllmPrepStage)
    assert prep_stage.stage._vllm_config.preprocess is False


def test_build_image_captioning_stages_returns_gemini_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini should use a single image-native API caption stage."""
    monkeypatch.setattr(image_api_caption_stages, "load_config", lambda: ConfigFileData(gemini=Gemini(api_key="k")))
    stages = build_image_captioning_stages(ImageCaptioningConfig(caption_algo="gemini"))

    assert len(stages) == 1
    assert isinstance(stages[0], CuratorStageSpec)
    assert isinstance(stages[0].stage, ImageGeminiCaptionStage)


def test_build_image_captioning_stages_returns_openai_prep_and_caption_by_default() -> None:
    """OpenAI should default to local-preprocessed payloads before captioning."""
    stages = build_image_captioning_stages(ImageCaptioningConfig(caption_algo="openai"))

    assert len(stages) == 2
    assert isinstance(stages[0], CuratorStageSpec)
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[0].stage, ImageOpenAIPrepStage)
    assert isinstance(stages[1].stage, ImageOpenAICaptionStage)


def test_build_image_captioning_stages_openai_raw_image_skips_prep() -> None:
    """OpenAI raw-image mode should bypass the local preprocessing stage."""
    stages = build_image_captioning_stages(
        ImageCaptioningConfig(
            caption_algo="openai",
            openai_raw_image=True,
        )
    )

    assert len(stages) == 1
    assert isinstance(stages[0], CuratorStageSpec)
    assert isinstance(stages[0].stage, ImageOpenAICaptionStage)
