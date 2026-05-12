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

"""Test the vLLM-based filter and classifier stages."""

import pathlib
import uuid

import pytest

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.runner_interface import RunnerInterface
from cosmos_curator.pipelines.common.filter_prompts import VIDEO_TYPE_LABELS
from cosmos_curator.pipelines.video.clipping.clip_extraction_stages import (  # type: ignore[import-untyped]
    ClipTranscodingStage,
)
from cosmos_curator.pipelines.video.filtering.aesthetics.aesthetics_builders import (
    VideoClassifierConfig,
    VlmFilterConfig,
    build_vllm_filter_classifier_stages,
)
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video  # type: ignore[import-untyped]

# Fill in after running the classifier GPU test manually to lock in expected classifications.
EXPECTED_CLASSIFICATIONS: list[str] = ["movie/film_scene", "nature_environment"]


@pytest.fixture
def sample_filtering_task(sample_video_data: bytes) -> SplitPipeTask:
    """Fixture to create a sample embedding task."""
    clips = []
    for start, end in [(0, 3), (11, 14)]:
        clip = Clip(
            uuid=uuid.uuid5(uuid.NAMESPACE_URL, f"sample_video.mp4#{start}-{end}"),
            source_video="sample_video.mp4",
            span=(start, end),
        )
        clips.append(clip)

    video = Video(
        input_video=pathlib.Path("sample_video.mp4"),
        encoded_data=sample_video_data,
        clips=clips,
    )
    return SplitPipeTask(
        session_id="test-session",
        video=video,
    )


@pytest.mark.env("unified")
def test_generate_embedding(sample_filtering_task: SplitPipeTask, sequential_runner: RunnerInterface) -> None:
    """Test the vLLM filtering result."""
    filtering_prompt = "blue car"
    stages = [
        ClipTranscodingStage(encoder="libopenh264"),
        *build_vllm_filter_classifier_stages(filter_config=VlmFilterConfig(filter_categories=filtering_prompt)),
    ]
    tasks = run_pipeline([sample_filtering_task], stages, runner=sequential_runner)

    assert tasks is not None
    assert len(tasks) > 0

    passing_clips = tasks[0].video.clips
    failing_clips = tasks[0].video.filtered_clips

    # Total clips (passing + filtered) should equal input count
    assert len(passing_clips) + len(failing_clips) == 2

    # Each clip should have filter_windows with qwen rejection reasons
    for clip in passing_clips + failing_clips:
        assert len(clip.filter_windows) > 0
        assert "qwen_rejection_reasons" in clip.filter_windows[0].caption


@pytest.mark.env("unified")
def test_qwen_video_classifier_classifications(
    sample_filtering_task: SplitPipeTask, sequential_runner: RunnerInterface
) -> None:
    """Test that the vLLM video classifier sets qwen_type_classification on each clip.

    Run with: cosmos-curator local launch --curator-path . -- pixi run --as-is -e unified pytest -m env
        tests/cosmos_curator/pipelines/video/filtering/aesthetics/test_qwen_filter.py
        -k test_qwen_video_classifier_classifications -v

    When EXPECTED_CLASSIFICATIONS is empty, we only assert structure (classification is a list
    of valid VIDEO_TYPE_LABELS). Fill in EXPECTED_CLASSIFICATIONS after a manual run to lock
    in the expected result for the test clip.
    """
    stages = [
        ClipTranscodingStage(encoder="libopenh264"),
        *build_vllm_filter_classifier_stages(classifier_config=VideoClassifierConfig()),
    ]
    tasks = run_pipeline([sample_filtering_task], stages, runner=sequential_runner)

    assert tasks is not None
    assert len(tasks) > 0

    all_clips = tasks[0].video.clips + tasks[0].video.filtered_clips
    assert len(all_clips) >= 1

    for clip in all_clips:
        assert clip.qwen_type_classification is not None
        assert isinstance(clip.qwen_type_classification, list)
        for label in clip.qwen_type_classification:
            assert label in VIDEO_TYPE_LABELS, f"{label!r} not in VIDEO_TYPE_LABELS"

    first_clip = tasks[0].video.clips[0] if tasks[0].video.clips else all_clips[0]
    actual = sorted(first_clip.qwen_type_classification or [])

    if EXPECTED_CLASSIFICATIONS:
        assert actual == sorted(EXPECTED_CLASSIFICATIONS)
