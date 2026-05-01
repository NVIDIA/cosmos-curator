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
"""Functional test for aesthetic score filtering stages.

This test verifies the aesthetic scoring and filtering stages using a sample video.
The expected aesthetic score values were obtained by running the aesthetic filter pipeline
on the sample video (ForBiggerBlazes.mp4) and capturing the actual values produced.
These values serve as a regression test to ensure the aesthetic scoring algorithm
maintains consistency across code changes.
"""

import pytest

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.runner_interface import RunnerInterface
from cosmos_curator.pipelines.video.filtering.aesthetics.aesthetic_filter_stages import (
    AestheticFilterStage,
)
from cosmos_curator.pipelines.video.utils.data_model import SplitPipeTask

EXPECTED_AESTHETIC_SCORE_MEAN: float = 4.8575
EXPECTED_AESTHETIC_SCORE_MIN: float = 3.7989
TOLERANCE: float = 0.002


@pytest.mark.env("unified")
def test_aesthetic_filter_setup() -> None:
    """Test that the aesthetic filter stage can be set up properly."""
    aesthetic_filter_stage = AestheticFilterStage(
        score_threshold=0.0,  # Set to 0 to ensure no filtering happens during score testing
        reduction="mean",
        log_stats=True,
    )
    # Set up the stage
    aesthetic_filter_stage.stage_setup()

    # Verify the model is set up
    assert aesthetic_filter_stage.model is not None

    # Clean up
    aesthetic_filter_stage.destroy()


@pytest.mark.env("unified")
def test_aesthetic_score_calculation_mean(
    sample_filtering_task: SplitPipeTask, sequential_runner: RunnerInterface
) -> None:
    """Test that aesthetic scores are calculated correctly with mean reduction.

    Args:
        sample_filtering_task: Sample task with video data
        sequential_runner: Runner for sequential test execution

    """
    stage = AestheticFilterStage(
        score_threshold=0.0,
        reduction="mean",
        log_stats=True,
    )
    result_tasks: list[SplitPipeTask] = run_pipeline([sample_filtering_task], [stage], runner=sequential_runner)

    # Verify there's one task returned
    assert len(result_tasks) == 1

    result_task = result_tasks[0]
    video = result_task.video
    # Verify the video has one clip (since threshold is 0.0)
    assert len(video.clips) == 1

    clip = video.clips[0]

    # Ensure aesthetic score attribute is present
    assert hasattr(clip, "aesthetic_score")

    assert clip.aesthetic_score == pytest.approx(EXPECTED_AESTHETIC_SCORE_MEAN, abs=TOLERANCE)

    # Verify stage performance stats were recorded
    assert "AestheticFilterStage" in result_task.stage_perf


@pytest.mark.env("unified")
def test_aesthetic_score_calculation_min(
    sample_filtering_task: SplitPipeTask, sequential_runner: RunnerInterface
) -> None:
    """Test that aesthetic scores are calculated correctly with min reduction.

    Args:
        sample_filtering_task: Sample task with video data
        sequential_runner: Runner for sequential test execution

    """
    stage = AestheticFilterStage(
        score_threshold=0.0,
        reduction="min",
        log_stats=True,
    )
    result_tasks: list[SplitPipeTask] = run_pipeline([sample_filtering_task], [stage], runner=sequential_runner)

    # Verify there's one task returned
    assert len(result_tasks) == 1

    result_task = result_tasks[0]
    video = result_task.video
    # Verify the video has one clip (since threshold is 0.0)
    assert len(video.clips) == 1

    clip = video.clips[0]

    # Ensure aesthetic score attribute is present
    assert hasattr(clip, "aesthetic_score")

    assert clip.aesthetic_score == pytest.approx(EXPECTED_AESTHETIC_SCORE_MIN, abs=TOLERANCE)

    # Verify stage performance stats were recorded
    assert "AestheticFilterStage" in result_task.stage_perf


@pytest.mark.env("unified")
@pytest.mark.parametrize(
    ("score_threshold", "should_be_filtered"),
    [
        # Threshold higher than expected score - clip should be filtered
        (9.0, True),
        # Threshold lower than expected score - clip should NOT be filtered
        (1.0, False),
    ],
)
def test_end_to_end_aesthetic_processing(
    sample_filtering_task: SplitPipeTask,
    sequential_runner: RunnerInterface,
    score_threshold: float,
    *,
    should_be_filtered: bool,
) -> None:
    """Test the complete aesthetic processing pipeline end-to-end with different thresholds.

    This parameterized test verifies the filtering behavior with various thresholds:
    - When actual aesthetic scores are below the threshold, the clip should be filtered out
    - When actual aesthetic scores are above the threshold, the clip should be kept

    Args:
        sample_filtering_task: The sample task fixture
        sequential_runner: Runner for sequential test execution
        score_threshold: The aesthetic score threshold to test
        should_be_filtered: Whether the clip should be filtered given the threshold

    """
    stage = AestheticFilterStage(
        score_threshold=score_threshold,
        reduction="mean",
        target_fps=1.0,
        verbose=True,
        log_stats=True,
    )
    result_tasks: list[SplitPipeTask] = run_pipeline([sample_filtering_task], [stage], runner=sequential_runner)

    # Verify the result
    video = result_tasks[0].video

    # Check that we have clips in either the main list or filtered list
    total_clips: int = len(video.clips) + len(video.filtered_clips)
    assert total_clips == 1  # We started with 1 clip

    if should_be_filtered:
        assert len(video.filtered_clips) == 1
        assert len(video.clips) == 0
    else:
        assert len(video.filtered_clips) == 0
        assert len(video.clips) == 1
