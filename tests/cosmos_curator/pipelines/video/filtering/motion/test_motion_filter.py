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
"""Functional test for motion vector decoding and motion filtering stages.

This test verifies the motion vector decoding and filtering stages using a sample video.
The expected motion score values were obtained by running the motion filter pipeline
on the sample video (ForBiggerBlazes.mp4) and capturing the actual values produced.
These values serve as a regression test to ensure the motion detection algorithm
maintains consistency across code changes.
"""

import pytest

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.runner_interface import RunnerInterface
from cosmos_curator.pipelines.video.filtering.motion.motion_filter_stages import (
    MotionFilterStage,
    MotionVectorDecodeStage,
)
from cosmos_curator.pipelines.video.utils.data_model import SplitPipeTask

# Golden values for motion scores
EXPECTED_MOTION_GLOBAL_MEAN: float = 0.002402
EXPECTED_MOTION_PER_PATCH_MIN_256: float = 0.001553
TOLERANCE: float = 0.000001


@pytest.fixture
def motion_decode_stage() -> MotionVectorDecodeStage:
    """Fixture to create a motion vector decode stage.

    Returns:
        MotionVectorDecodeStage: Configured instance of the decode stage

    """
    return MotionVectorDecodeStage(num_cpus_per_worker=1.0, log_stats=True)


@pytest.fixture
def motion_filter_stage() -> MotionFilterStage:
    """Fixture to create a motion filter stage.

    Returns:
        MotionFilterStage: Configured instance of the filter stage

    """
    return MotionFilterStage(
        score_only=True,
        global_mean_threshold=-999,
        per_patch_min_256_threshold=-999,
        log_stats=True,
    )


@pytest.mark.env("cosmos-curator")
def test_motion_vector_decode(
    motion_decode_stage: MotionVectorDecodeStage,
    sample_filtering_task: SplitPipeTask,
    sequential_runner: RunnerInterface,
) -> None:
    """Test that motion vectors can be decoded from the sample video.

    Args:
        motion_decode_stage: The decode stage to test
        sample_filtering_task: Sample task with video data
        sequential_runner: Sequential runner fixture

    """
    result_tasks: list[SplitPipeTask] = run_pipeline(
        [sample_filtering_task], [motion_decode_stage], runner=sequential_runner
    )

    # Verify there's one task returned
    assert len(result_tasks) == 1

    result_task = result_tasks[0]
    video = result_task.video
    # Verify the video has one clip
    assert len(video.clips) == 1

    clip = video.clips[0]
    # Check that motion data was decoded
    assert clip.decoded_motion_data is not None
    # Check that we have frames
    assert len(clip.decoded_motion_data.frames) > 0

    # Verify stage performance stats were recorded
    assert "MotionVectorDecodeStage" in result_task.stage_perf


@pytest.mark.env("cosmos-curator")
def test_motion_filter_calculation(
    motion_decode_stage: MotionVectorDecodeStage,
    motion_filter_stage: MotionFilterStage,
    sample_filtering_task: SplitPipeTask,
    sequential_runner: RunnerInterface,
) -> None:
    """Test that motion scores are calculated correctly and filtering works as expected.

    Args:
        motion_decode_stage: The decode stage to use
        motion_filter_stage: The filter stage to test
        sample_filtering_task: Sample task with video data
        sequential_runner: Sequential runner fixture

    """
    stages = [motion_decode_stage, motion_filter_stage]
    tasks_after_filter: list[SplitPipeTask] = run_pipeline([sample_filtering_task], stages, runner=sequential_runner)

    # Verify there's one task returned
    assert len(tasks_after_filter) == 1

    result_task = tasks_after_filter[0]
    video = result_task.video
    # Verify the video has one clip (since we're using score_only=True)
    assert len(video.clips) == 1

    clip = video.clips[0]

    # Ensure motion score attributes are present
    assert hasattr(clip, "motion_score_global_mean")
    assert hasattr(clip, "motion_score_per_patch_min_256")

    # Check that the scores are within expected range
    assert clip.motion_score_global_mean == pytest.approx(EXPECTED_MOTION_GLOBAL_MEAN, abs=TOLERANCE)
    assert clip.motion_score_per_patch_min_256 == pytest.approx(EXPECTED_MOTION_PER_PATCH_MIN_256, abs=TOLERANCE)

    # Verify that decoded_motion_data was cleared to save memory
    assert clip.decoded_motion_data is None

    # Verify stage performance stats were recorded
    assert "MotionFilterStage" in result_task.stage_perf


@pytest.mark.env("cosmos-curator")
@pytest.mark.parametrize(
    ("global_threshold", "patch_threshold", "should_be_filtered"),
    [
        # Both thresholds higher than actual values - clip should be filtered
        (0.003, 0.002, True),
        # Global threshold higher, patch threshold lower - clip should be filtered
        (0.003, 0.001, True),
        # Global threshold lower, patch threshold higher - clip should be filtered
        (0.002, 0.002, True),
        # Both thresholds lower than actual values - clip should NOT be filtered
        (0.002, 0.001, False),
    ],
)
def test_end_to_end_motion_processing(  # noqa: PLR0913 - parametrized test with multiple fixtures
    motion_decode_stage: MotionVectorDecodeStage,
    sample_filtering_task: SplitPipeTask,
    sequential_runner: RunnerInterface,
    global_threshold: float,
    patch_threshold: float,
    *,
    should_be_filtered: bool,
) -> None:
    """Test the complete motion processing pipeline end-to-end with different thresholds.

    This parameterized test verifies the filtering behavior with various threshold combinations:
    - When actual motion values are below the thresholds, the clip should be filtered out
    - When actual motion values are above the thresholds, the clip should be kept

    Args:
        motion_decode_stage: The decode stage fixture
        sample_filtering_task: The sample task fixture
        sequential_runner: Sequential runner fixture
        global_threshold: The global mean threshold to test
        patch_threshold: The per-patch min 256 threshold to test
        should_be_filtered: Whether the clip should be filtered given the thresholds

    """
    # Instantiate a fresh filter stage with the desired thresholds
    stages = [
        motion_decode_stage,
        MotionFilterStage(
            score_only=False,
            global_mean_threshold=global_threshold,
            per_patch_min_256_threshold=patch_threshold,
            log_stats=True,
        ),
    ]

    # Run through decode and filter stages
    tasks: list[SplitPipeTask] = run_pipeline([sample_filtering_task], stages, runner=sequential_runner)

    # Verify the result
    result_task = tasks[0]
    video = result_task.video

    # Check that we have clips in either the main list or filtered list
    total_clips: int = len(video.clips) + len(video.filtered_clips)
    assert total_clips == 1  # We started with 1 clip

    if should_be_filtered:
        assert len(video.filtered_clips) == 1
        assert len(video.clips) == 0
    else:
        assert len(video.filtered_clips) == 0
        assert len(video.clips) == 1
