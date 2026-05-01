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
"""Functional tests for TransNetV2ClipExtractionStage."""

import pytest

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.runner_interface import RunnerInterface
from cosmos_curator.pipelines.video.clipping.frame_extraction_stages import VideoFrameExtractionStage
from cosmos_curator.pipelines.video.clipping.transnetv2_extraction_stages import TransNetV2ClipExtractionStage
from cosmos_curator.pipelines.video.utils.data_model import SplitPipeTask


@pytest.mark.env("cosmos-curator")
def test_transnetv2_requires_frame_extraction(
    sample_splitting_task: SplitPipeTask, sequential_runner: RunnerInterface
) -> None:
    """Test that TransNetV2 stage skips processing (yields zero clips) if frames were not extracted."""
    # Run TransNetV2 without prior frame extraction: it should skip processing
    result_tasks = run_pipeline([sample_splitting_task], [TransNetV2ClipExtractionStage()], runner=sequential_runner)
    assert len(result_tasks) == 1
    assert len(result_tasks[0].video.clips) == 0


@pytest.mark.env("cosmos-curator")
def test_transnetv2_default_extraction(
    sample_splitting_task: SplitPipeTask, sequential_runner: RunnerInterface
) -> None:
    """Test default extraction pipeline produces clips."""
    stages = [
        VideoFrameExtractionStage(log_stats=True),
        TransNetV2ClipExtractionStage(log_stats=True),
    ]
    result_tasks = run_pipeline([sample_splitting_task], stages, runner=sequential_runner)
    # Verify task returned
    assert result_tasks is not None
    assert len(result_tasks) == 1
    video = result_tasks[0].video
    # Verify clips were extracted
    assert hasattr(video, "clips")
    assert len(video.clips) > 0
    # Verify clip spans are valid
    duration = video.metadata.duration
    for clip in video.clips:
        assert hasattr(clip, "uuid")
        assert clip.source_video == str(video.input_video)
        start, end = clip.span
        # Ensure spans are within video duration
        assert 0.0 <= start < end <= duration
    # Verify stats recorded
    assert "TransNetV2ClipExtractionStage" in result_tasks[0].stage_perf


@pytest.mark.env("cosmos-curator")
def test_transnetv2_no_transitions_entire_scene_false(
    sample_splitting_task: SplitPipeTask, sequential_runner: RunnerInterface
) -> None:
    """Test that no clips are extracted when no transitions and entire_scene_as_clip=False."""
    stages = [
        VideoFrameExtractionStage(),
        TransNetV2ClipExtractionStage(threshold=1.0, entire_scene_as_clip=False),
    ]
    result_tasks = run_pipeline([sample_splitting_task], stages, runner=sequential_runner)
    video = result_tasks[0].video
    assert len(video.clips) == 0


@pytest.mark.env("cosmos-curator")
def test_transnetv2_entire_scene_when_no_transitions(
    sample_splitting_task: SplitPipeTask, sequential_runner: RunnerInterface
) -> None:
    """Test that entire scene is returned as one clip when no transitions and entire_scene_as_clip=True."""
    stages = [
        VideoFrameExtractionStage(),
        TransNetV2ClipExtractionStage(threshold=1.0, entire_scene_as_clip=True, crop_s=0.0),
    ]
    result_tasks = run_pipeline([sample_splitting_task], stages, runner=sequential_runner)
    video = result_tasks[0].video
    assert len(video.clips) == 1
    start, end = video.clips[0].span
    duration = video.metadata.duration
    assert start == pytest.approx(0.0, abs=1e-2)
    assert end == pytest.approx(duration, abs=1e-2)


@pytest.mark.env("cosmos-curator")
def test_transnetv2_limit_clips(sample_splitting_task: SplitPipeTask, sequential_runner: RunnerInterface) -> None:
    """Test that limit_clips parameter limits the number of extracted clips."""
    stages = [
        VideoFrameExtractionStage(),
        TransNetV2ClipExtractionStage(limit_clips=1),
    ]
    result_tasks = run_pipeline([sample_splitting_task], stages, runner=sequential_runner)
    video = result_tasks[0].video
    assert len(video.clips) == 1
