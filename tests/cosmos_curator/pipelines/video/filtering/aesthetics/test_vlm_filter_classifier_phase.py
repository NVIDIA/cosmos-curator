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

"""Unit tests for build_vllm_filter_classifier_stages and classifier metadata."""

import pathlib
import uuid
from unittest.mock import MagicMock, patch

import pytest

from cosmos_curator.core.interfaces.stage_interface import CuratorStageSpec
from cosmos_curator.pipelines.video.captioning.gemini_caption_stage import ApiPrepStage, GeminiCaptionStage
from cosmos_curator.pipelines.video.captioning.openai_caption_stage import OpenAICaptionStage
from cosmos_curator.pipelines.video.captioning.vllm_caption_stage import VllmCaptionStage, VllmPrepStage
from cosmos_curator.pipelines.video.filtering.aesthetics.aesthetics_builders import (
    VideoClassifierConfig,
    VlmFilterConfig,
    build_vllm_filter_classifier_stages,
)
from cosmos_curator.pipelines.video.filtering.aesthetics.semantic_filter_stages import (
    VllmFilteringStage,
    VllmVideoClassifierStage,
)
from cosmos_curator.pipelines.video.utils.data_model import Clip, Video, Window

_GEMINI_LOAD_CONFIG = "cosmos_curator.pipelines.video.captioning.gemini_caption_stage.load_config"


@pytest.fixture
def mock_gemini_config() -> MagicMock:
    """Return a minimal config mock that satisfies GeminiCaptionStage.__init__."""
    cfg = MagicMock()
    cfg.gemini = MagicMock()
    cfg.gemini.api_key = "test-key"
    return cfg


def test_vlm_filter_classifier_requires_at_least_one_config() -> None:
    """Builder must receive at least one of filter_config or classifier_config."""
    with pytest.raises(ValueError, match="At least one of filter_config or classifier_config is required"):
        build_vllm_filter_classifier_stages()


def test_vlm_filter_classifier_build_stages_filter_only() -> None:
    """Filter-only builds VllmPrepStage + VllmCaptionStage + VllmFilteringStage (3 stages)."""
    stages = build_vllm_filter_classifier_stages(filter_config=VlmFilterConfig())
    assert len(stages) == 3
    assert isinstance(stages[0], VllmPrepStage)
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[1].stage, VllmCaptionStage)
    assert isinstance(stages[2], CuratorStageSpec)
    assert isinstance(stages[2].stage, VllmFilteringStage)


def test_vlm_filter_classifier_build_stages_classifier_only() -> None:
    """Classifier-only builds VllmPrepStage + VllmCaptionStage + VllmVideoClassifierStage (3 stages)."""
    stages = build_vllm_filter_classifier_stages(classifier_config=VideoClassifierConfig())
    assert len(stages) == 3
    assert isinstance(stages[0], VllmPrepStage)
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[1].stage, VllmCaptionStage)
    assert isinstance(stages[2], CuratorStageSpec)
    assert isinstance(stages[2].stage, VllmVideoClassifierStage)


def test_vlm_filter_classifier_build_stages_both() -> None:
    """Both configs: two sets of (VllmPrepStage + VllmCaptionStage + stage) = 6 stages."""
    stages = build_vllm_filter_classifier_stages(
        filter_config=VlmFilterConfig(),
        classifier_config=VideoClassifierConfig(),
    )
    assert len(stages) == 6
    # Filter set
    assert isinstance(stages[0], VllmPrepStage)
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[1].stage, VllmCaptionStage)
    assert isinstance(stages[2], CuratorStageSpec)
    assert isinstance(stages[2].stage, VllmFilteringStage)
    # Classifier set
    assert isinstance(stages[3], VllmPrepStage)
    assert isinstance(stages[4], CuratorStageSpec)
    assert isinstance(stages[4].stage, VllmCaptionStage)
    assert isinstance(stages[5], CuratorStageSpec)
    assert isinstance(stages[5].stage, VllmVideoClassifierStage)


def test_vlm_filter_build_stages_openai_endpoint() -> None:
    """OpenAI filter endpoint builds ApiPrepStage + OpenAICaptionStage(filter) + VllmFilteringStage."""
    stages = build_vllm_filter_classifier_stages(filter_config=VlmFilterConfig(endpoint="openai", batch_size=7))
    assert len(stages) == 3
    assert isinstance(stages[0], ApiPrepStage)
    assert isinstance(stages[1], OpenAICaptionStage)
    assert stages[1]._endpoint_key == "filter"
    assert stages[1]._model_variant == "openai"
    assert stages[1]._batch_size == 7
    assert isinstance(stages[2], CuratorStageSpec)
    assert isinstance(stages[2].stage, VllmFilteringStage)


def test_vlm_filter_build_stages_gemini_endpoint(mock_gemini_config: MagicMock) -> None:
    """Gemini filter endpoint builds ApiPrepStage + GeminiCaptionStage + VllmFilteringStage."""
    with patch(_GEMINI_LOAD_CONFIG, return_value=mock_gemini_config):
        stages = build_vllm_filter_classifier_stages(filter_config=VlmFilterConfig(endpoint="gemini", batch_size=5))
    assert len(stages) == 3
    assert isinstance(stages[0], ApiPrepStage)
    assert isinstance(stages[1], GeminiCaptionStage)
    assert stages[1]._model_variant == "gemini"
    assert stages[1]._batch_size == 5
    assert isinstance(stages[2], CuratorStageSpec)
    assert isinstance(stages[2].stage, VllmFilteringStage)


def test_vlm_classifier_build_stages_openai_endpoint() -> None:
    """OpenAI classifier endpoint builds ApiPrepStage + OpenAICaptionStage(classifier) + VllmVideoClassifierStage."""
    stages = build_vllm_filter_classifier_stages(
        classifier_config=VideoClassifierConfig(endpoint="openai", batch_size=9)
    )
    assert len(stages) == 3
    assert isinstance(stages[0], ApiPrepStage)
    assert isinstance(stages[1], OpenAICaptionStage)
    assert stages[1]._endpoint_key == "classifier"
    assert stages[1]._model_variant == "openai"
    assert stages[1]._batch_size == 9
    assert isinstance(stages[2], CuratorStageSpec)
    assert isinstance(stages[2].stage, VllmVideoClassifierStage)


def test_vlm_classifier_build_stages_gemini_endpoint(mock_gemini_config: MagicMock) -> None:
    """Gemini classifier endpoint builds ApiPrepStage + GeminiCaptionStage + VllmVideoClassifierStage."""
    with patch(_GEMINI_LOAD_CONFIG, return_value=mock_gemini_config):
        stages = build_vllm_filter_classifier_stages(
            classifier_config=VideoClassifierConfig(endpoint="gemini", batch_size=4)
        )
    assert len(stages) == 3
    assert isinstance(stages[0], ApiPrepStage)
    assert isinstance(stages[1], GeminiCaptionStage)
    assert stages[1]._model_variant == "gemini"
    assert stages[1]._batch_size == 4
    assert isinstance(stages[2], CuratorStageSpec)
    assert isinstance(stages[2].stage, VllmVideoClassifierStage)


def test_external_endpoint_forwards_max_output_tokens(mock_gemini_config: MagicMock) -> None:
    """max_output_tokens from config is forwarded to the caption stage for both endpoints."""
    custom_tokens = 1024
    openai_stages = build_vllm_filter_classifier_stages(
        filter_config=VlmFilterConfig(endpoint="openai", max_output_tokens=custom_tokens)
    )
    assert openai_stages[1]._max_output_tokens == custom_tokens

    with patch(_GEMINI_LOAD_CONFIG, return_value=mock_gemini_config):
        gemini_stages = build_vllm_filter_classifier_stages(
            filter_config=VlmFilterConfig(endpoint="gemini", max_output_tokens=custom_tokens)
        )
    assert gemini_stages[1]._max_output_tokens == custom_tokens


def test_qwen_video_classifier_stage_custom_categories_requires_allow_or_block() -> None:
    """custom_categories=True requires at least one of type_allow or type_block."""
    with pytest.raises(ValueError, match="custom_categories=True requires at least one of type_allow or type_block"):
        VllmVideoClassifierStage(model_variant="qwen", custom_categories=True)


def test_qwen_video_classifier_stage_custom_categories_builds_valid_labels() -> None:
    """With custom_categories=True, allow+block union becomes the only categories."""
    stage = VllmVideoClassifierStage(
        model_variant="qwen",
        custom_categories=True,
        type_allow="penguins",
        type_block="polar_bears",
    )
    assert stage._valid_type_labels == ("penguins", "polar_bears")


def test_vlm_filter_classifier_build_stages_custom_categories() -> None:
    """Builder with custom_categories returns 3 stages including VllmVideoClassifierStage."""
    stages = build_vllm_filter_classifier_stages(
        classifier_config=VideoClassifierConfig(
            custom_categories=True,
            type_allow="penguins",
            type_block="polar_bears",
        ),
    )
    assert len(stages) == 3
    assert isinstance(stages[0], VllmPrepStage)
    classifier_spec = stages[2]
    assert isinstance(classifier_spec, CuratorStageSpec)
    assert isinstance(classifier_spec.stage, VllmVideoClassifierStage)


def test_vlm_filter_classifier_build_stages_default_categories() -> None:
    """When custom_categories=False, builder returns 3 stages with VllmVideoClassifierStage."""
    stages = build_vllm_filter_classifier_stages(classifier_config=VideoClassifierConfig())
    assert len(stages) == 3
    assert isinstance(stages[0], VllmPrepStage)
    classifier_spec = stages[2]
    assert isinstance(classifier_spec, CuratorStageSpec)
    assert isinstance(classifier_spec.stage, VllmVideoClassifierStage)


def test_vlm_filter_classifier_custom_categories_union_only_allow() -> None:
    """Custom categories with only allow list returns 3 stages."""
    stages = build_vllm_filter_classifier_stages(
        classifier_config=VideoClassifierConfig(
            custom_categories=True,
            type_allow="planet_earth,mountains",
            type_block=None,
        ),
    )
    assert len(stages) == 3
    classifier_spec = stages[2]
    assert isinstance(classifier_spec, CuratorStageSpec)
    assert isinstance(classifier_spec.stage, VllmVideoClassifierStage)


def test_vlm_filter_classifier_custom_categories_union_only_block() -> None:
    """Custom categories with only block list returns 3 stages."""
    stages = build_vllm_filter_classifier_stages(
        classifier_config=VideoClassifierConfig(
            custom_categories=True,
            type_allow=None,
            type_block="space",
        ),
    )
    assert len(stages) == 3
    classifier_spec = stages[2]
    assert isinstance(classifier_spec, CuratorStageSpec)
    assert isinstance(classifier_spec.stage, VllmVideoClassifierStage)


def _make_video_with_one_clip_one_window() -> tuple[Video, int]:
    """Minimal Video with one clip and one filter_window for _type_mode_filter_clip."""
    window = Window(start_frame=0, end_frame=124)
    clip = Clip(
        uuid=uuid.uuid4(),
        source_video="test.mp4",
        span=(0.0, 5.0),
    )
    clip.filter_windows.append(window)
    video = Video(input_video=pathlib.Path("test.mp4"), clips=[clip])
    return video, 0


def test_qwen_video_classifier_rejection_reasons_only_allow_no_and_block_yes() -> None:
    """Rejection reasons must only include allow-list 'no' and block-list 'yes', not block 'no'."""
    stage = VllmVideoClassifierStage(
        model_variant="qwen",
        custom_categories=True,
        type_allow="planet_earth,mountains",
        type_block="space",
    )
    video, clip_idx = _make_video_with_one_clip_one_window()
    all_issues: set[str] = set()
    result_json = '{"planet_earth": "no", "mountains": "no", "space": "no"}'
    stage._type_mode_filter_clip(video, clip_idx, [(0, result_json)], all_issues)
    cap = video.clips[clip_idx].filter_windows[0].caption["qwen_rejection_reasons"]
    assert "planet_earth" in cap
    assert "no" in cap
    assert "mountains" in cap
    assert "space" not in cap


def test_qwen_video_classifier_rejection_reasons_block_yes_included() -> None:
    """Rejection reasons include block-list categories when model says yes."""
    stage = VllmVideoClassifierStage(
        model_variant="qwen",
        custom_categories=True,
        type_allow="planet_earth",
        type_block="space",
    )
    video, clip_idx = _make_video_with_one_clip_one_window()
    all_issues: set[str] = set()
    result_json = '{"planet_earth": "no", "space": "yes"}'
    stage._type_mode_filter_clip(video, clip_idx, [(0, result_json)], all_issues)
    cap = video.clips[clip_idx].filter_windows[0].caption["qwen_rejection_reasons"]
    assert "space" in cap
    assert "yes" in cap
    assert "planet_earth" in cap
    assert "no" in cap


def test_qwen_video_classifier_custom_categories_empty_gets_unclassified() -> None:
    """When custom_categories=True and no category is yes, classification is ['unclassified']."""
    stage = VllmVideoClassifierStage(
        model_variant="qwen",
        custom_categories=True,
        type_allow="planet_earth,mountains",
        type_block="space",
    )
    video, clip_idx = _make_video_with_one_clip_one_window()
    all_issues: set[str] = set()
    result_json = '{"planet_earth": "no", "mountains": "no", "space": "no"}'
    stage._type_mode_filter_clip(video, clip_idx, [(0, result_json)], all_issues)
    assert video.clips[clip_idx].qwen_type_classification == ["unclassified"]


def test_qwen_video_classifier_custom_categories_match_not_other() -> None:
    """When custom_categories=True and at least one category is yes, classification lists them."""
    stage = VllmVideoClassifierStage(
        model_variant="qwen",
        custom_categories=True,
        type_allow="planet_earth,mountains",
        type_block="space",
    )
    video, clip_idx = _make_video_with_one_clip_one_window()
    all_issues: set[str] = set()
    result_json = '{"planet_earth": "yes", "mountains": "no", "space": "no"}'
    stage._type_mode_filter_clip(video, clip_idx, [(0, result_json)], all_issues)
    assert video.clips[clip_idx].qwen_type_classification == ["planet_earth"]


def _make_video_three_clips_middle_no_windows() -> Video:
    """Video with 3 clips where the middle clip has no filter_windows."""
    clips = []
    for i in range(3):
        clip = Clip(uuid=uuid.uuid4(), source_video="test.mp4", span=(float(i * 5), float((i + 1) * 5)))
        if i != 1:
            clip.filter_windows.append(Window(start_frame=0, end_frame=124))
        clips.append(clip)
    return Video(input_video=pathlib.Path("test.mp4"), clips=clips)


def test_vlm_filtering_stage_middle_clip_no_windows_correct_error_assignment() -> None:
    """Middle clip with no windows must get the error, not the clip after it (index binding fix)."""
    stage = VllmFilteringStage(model_variant="qwen", user_prompt="slideshow")
    video = _make_video_three_clips_middle_no_windows()
    original_clips = list(video.clips)

    # Clips 0 and 2 have windows; clip 1 does not — mapping skips index 1.
    mapping = {0: (0, 0), 1: (2, 0)}
    captions = enumerate(['{"slideshow": "no"}', '{"slideshow": "no"}'])

    stage._filter_clips(video, mapping, captions)

    # Clips 0 and 2 passed; clip 1 (no windows) should be filtered with the right error.
    assert len(video.clips) == 2
    assert len(video.filtered_clips) == 1
    errored = video.filtered_clips[0]
    assert errored.errors.get("qwen") == "all_windows_failed_preparation"
    # Must be the original middle clip, not the clip that followed it.
    assert errored is original_clips[1]


def test_vlm_classifier_stage_middle_clip_no_windows_correct_error_assignment() -> None:
    """Middle clip with no windows must get the error, not the clip after it (index binding fix)."""
    stage = VllmVideoClassifierStage(
        model_variant="qwen",
        custom_categories=True,
        type_allow="planet_earth",
        clear_model_input_after=False,
    )
    video = _make_video_three_clips_middle_no_windows()
    original_clips = list(video.clips)

    # Clips 0 and 2 have windows; clip 1 does not — mapping skips index 1.
    mapping = {0: (0, 0), 1: (2, 0)}
    captions = enumerate(['{"planet_earth": "yes"}', '{"planet_earth": "yes"}'])

    stage._filter_clips(video, mapping, captions)

    # Clips 0 and 2 passed; clip 1 (no windows) should be filtered with the right error.
    assert len(video.clips) == 2
    assert len(video.filtered_clips) == 1
    errored = video.filtered_clips[0]
    assert errored.errors.get("qwen") == "all_windows_failed_preparation"
    # Must be the original middle clip, not the clip that followed it.
    assert errored is original_clips[1]


def test_vlm_filtering_stage_score_only_middle_clip_no_windows_kept_with_error() -> None:
    """Score-only semantic filtering should keep errored clips instead of moving them to filtered_clips."""
    stage = VllmFilteringStage(model_variant="qwen", user_prompt="slideshow", score_only=True)
    video = _make_video_three_clips_middle_no_windows()
    original_clips = list(video.clips)

    mapping = {0: (0, 0), 1: (2, 0)}
    captions = enumerate(['{"slideshow": "no"}', '{"slideshow": "no"}'])

    stage._filter_clips(video, mapping, captions)

    assert len(video.clips) == 3
    assert len(video.filtered_clips) == 0
    errored = video.clips[2]
    assert errored.errors.get("qwen") == "all_windows_failed_preparation"
    assert errored is original_clips[1]


def test_vlm_filtering_stage_marks_malformed_model_output_on_window() -> None:
    """Malformed semantic-filter output should filter the clip and store the error on its window."""
    stage = VllmFilteringStage(model_variant="qwen", user_prompt="slideshow")
    video, clip_idx = _make_video_with_one_clip_one_window()

    stage._filter_clips(video, {0: (clip_idx, 0)}, enumerate(["not valid json"]))

    assert len(video.clips) == 0
    assert len(video.filtered_clips) == 1
    assert video.filtered_clips[0].filter_windows[0].errors.get("qwen") == "malformed_model_output"
    assert video.filtered_clips[0].qwen_rejection_stage == "semantic"


def test_vlm_filtering_stage_malformed_score_only_keeps_clip() -> None:
    """Malformed semantic-filter output with score_only should keep the clip."""
    stage = VllmFilteringStage(model_variant="qwen", user_prompt="slideshow", score_only=True)
    video, clip_idx = _make_video_with_one_clip_one_window()

    stage._filter_clips(video, {0: (clip_idx, 0)}, enumerate(["not valid json"]))

    assert len(video.clips) == 1
    assert len(video.filtered_clips) == 0
    assert video.clips[0].filter_windows[0].errors.get("qwen") == "malformed_model_output"


def test_vlm_classifier_stage_marks_malformed_model_output_on_window() -> None:
    """Malformed classifier output should filter the clip and store the error on its window."""
    stage = VllmVideoClassifierStage(
        model_variant="qwen",
        custom_categories=True,
        type_allow="planet_earth",
        clear_model_input_after=False,
    )
    video, clip_idx = _make_video_with_one_clip_one_window()

    stage._filter_clips(video, {0: (clip_idx, 0)}, enumerate(["not valid json"]))

    assert len(video.clips) == 0
    assert len(video.filtered_clips) == 1
    assert video.filtered_clips[0].filter_windows[0].errors.get("qwen") == "malformed_model_output"
    assert video.filtered_clips[0].qwen_rejection_stage == "classifier"


def test_qwen_video_classifier_default_categories_empty_not_unclassified() -> None:
    """When custom_categories=False and no category is yes, classification is [] (no default fallback)."""
    stage = VllmVideoClassifierStage(
        model_variant="qwen",
        custom_categories=False,
        type_allow="nature_environment",
        type_block=None,
    )
    video, clip_idx = _make_video_with_one_clip_one_window()
    all_issues: set[str] = set()
    result_json = '{"nature_environment": "no"}'
    stage._type_mode_filter_clip(video, clip_idx, [(0, result_json)], all_issues)
    assert video.clips[clip_idx].qwen_type_classification == []
