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

"""Tests for the captioning stage builder dispatch paths."""

import pytest

from cosmos_curator.pipelines.video.captioning.captioning_builders import (
    CaptioningConfig,
    OpenAIConfig,
    _build_captioning_caption_stage,
    _build_captioning_prep_stage,
    build_captioning_stages,
)
from cosmos_curator.pipelines.video.captioning.gemini_caption_stage import ApiPrepStage
from cosmos_curator.pipelines.video.captioning.openai_caption_stage import OpenAICaptionStage
from cosmos_curator.pipelines.video.utils.data_model import WindowConfig


def _default_openai_config() -> CaptioningConfig:
    """Return a minimal CaptioningConfig for the openai backend."""
    return CaptioningConfig(
        backend=OpenAIConfig(),
        window_config=WindowConfig(),
    )


# ---------------------------------------------------------------------------
# _build_captioning_prep_stage
# ---------------------------------------------------------------------------


def test_build_prep_stage_openai_returns_api_prep_stage() -> None:
    """The openai prep stage should be an ApiPrepStage."""
    stage = _build_captioning_prep_stage(_default_openai_config())
    assert isinstance(stage, ApiPrepStage)


def test_build_prep_stage_openai_uses_configured_cpus() -> None:
    """ApiPrepStage should receive num_cpus_for_prepare from OpenAIConfig."""
    cfg = CaptioningConfig(
        backend=OpenAIConfig(num_cpus_for_prepare=5.0),
        window_config=WindowConfig(),
    )
    stage = _build_captioning_prep_stage(cfg)
    assert isinstance(stage, ApiPrepStage)
    assert stage._num_cpus_for_prepare == 5.0


# ---------------------------------------------------------------------------
# _build_captioning_caption_stage
# ---------------------------------------------------------------------------


def test_build_caption_stage_openai_returns_openai_stage() -> None:
    """The openai caption stage should be an OpenAICaptionStage."""
    stage = _build_captioning_caption_stage(_default_openai_config())
    assert isinstance(stage, OpenAICaptionStage)


def test_build_caption_stage_openai_forwards_config_params() -> None:
    """OpenAICaptionStage should receive parameters from OpenAIConfig."""
    cfg = CaptioningConfig(
        backend=OpenAIConfig(
            model_name="my-custom-model",
            max_output_tokens=4096,
            caption_retries=5,
            retry_delay_seconds=2.0,
            batch_size=6,
        ),
        window_config=WindowConfig(),
    )
    stage = _build_captioning_caption_stage(cfg)
    assert isinstance(stage, OpenAICaptionStage)
    assert stage._model_name == "my-custom-model"
    assert stage._max_output_tokens == 4096
    assert stage._max_caption_retries == 5
    assert stage._retry_delay_seconds == 2.0
    assert stage._batch_size == 6


def test_build_caption_stage_unsupported_backend_raises() -> None:
    """NotImplementedError for an unrecognized backend config type."""
    cfg = CaptioningConfig(
        backend="not_a_real_backend",  # type: ignore[arg-type]
        window_config=WindowConfig(),
    )
    with pytest.raises(NotImplementedError, match="Unsupported caption backend type"):
        _build_captioning_caption_stage(cfg)


# ---------------------------------------------------------------------------
# build_captioning_stages (full pipeline)
# ---------------------------------------------------------------------------


def test_build_stages_base_count() -> None:
    """Base openai build_captioning_stages should produce exactly 2 stages: prep + caption."""
    stages = build_captioning_stages(_default_openai_config())
    assert len(stages) == 2
    assert isinstance(stages[0], ApiPrepStage)
    assert isinstance(stages[1], OpenAICaptionStage)


def test_build_stages_with_previews() -> None:
    """Enabling previews should add a PreviewStage (3 total)."""
    cfg = CaptioningConfig(
        backend=OpenAIConfig(),
        window_config=WindowConfig(),
        generate_previews=True,
    )
    stages = build_captioning_stages(cfg)
    assert len(stages) == 3
    # Preview is inserted between prep and caption
    assert isinstance(stages[0], ApiPrepStage)
    assert isinstance(stages[2], OpenAICaptionStage)
