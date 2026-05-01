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
"""Functional test for fixed stride clip extraction stage.

This test verifies the fixed stride clip extraction stage using a sample video.
The expected results were obtained by running the fixed stride extraction pipeline
on the sample video and capturing the actual values produced.
These values serve as a regression test to ensure the clip extraction algorithm
maintains consistency across code changes.
"""

import copy
import pathlib
import uuid
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import numpy as np
import pytest

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.runner_interface import RunnerInterface
from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.pipelines.video.clipping.clip_extraction_stages import (
    FixedStrideExtractorStage,
    _get_videos_durations,
    _get_videos_timestamps,
    _make_clip_uuids,
    _make_spans_fixed_stride,
    _populate_clips_fixed_stride,
    _validate_video_timestamps,
    slice_video_clips,
)
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video, VideoMetadata


def test_fixed_stride_extractor_setup() -> None:
    """Test that the fixed stride extractor stage can be set up properly."""
    # Default parameter values for FixedStrideExtractorStage
    default_clip_length = 10
    default_clip_stride = 10
    default_min_clip_length = 10
    default_limit_clips = 0
    default_cpu_count = 1.0
    default_gpu_count = 0

    stage = FixedStrideExtractorStage()

    # Verify basic properties
    assert stage.clip_len_s == default_clip_length
    assert stage.clip_stride_s == default_clip_stride
    assert stage.min_clip_length_s == default_min_clip_length
    assert stage._limit_clips == default_limit_clips
    assert stage._verbose is False
    assert stage._log_stats is False

    # Verify resource requirements (CPU-only stage)
    assert stage.resources.gpus == default_gpu_count
    assert stage.resources.cpus == default_cpu_count  # Default to 1 CPU per worker


def test_fixed_stride_extractor_custom_parameters() -> None:
    """Test that the fixed stride extractor stage accepts custom parameters."""
    # Custom parameter values for testing
    custom_clip_length = 5.0
    custom_clip_stride = 3.0
    custom_min_clip_length = 2.0
    custom_limit_clips = 10

    stage = FixedStrideExtractorStage(
        clip_len_s=custom_clip_length,
        clip_stride_s=custom_clip_stride,
        min_clip_length_s=custom_min_clip_length,
        limit_clips=custom_limit_clips,
        verbose=True,
        log_stats=True,
    )

    # Verify custom parameters are set
    assert stage.clip_len_s == custom_clip_length
    assert stage.clip_stride_s == custom_clip_stride
    assert stage.min_clip_length_s == custom_min_clip_length
    assert stage._limit_clips == custom_limit_clips
    assert stage._verbose is True
    assert stage._log_stats is True


@pytest.mark.env("cosmos-curator")
def test_fixed_stride_extraction_default_parameters(
    sample_splitting_task: SplitPipeTask, sequential_runner: RunnerInterface
) -> None:
    """Test fixed stride extraction with default parameters.

    Args:
        sample_splitting_task: Sample task with video data
        sequential_runner: Runner for sequential test execution

    """
    # Expected clip extraction results for regression testing
    # These values are based on the sample video metadata:
    # - Duration: 30 seconds (test_video_30s.mp4)
    # - With default parameters (10s clips, 10s stride, min_clip_length_s=10),
    # - we expect clips: 0-10s, 10-20s, 20-30s
    expected_clips_default = 3
    expected_clip_spans_default = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]

    stage = FixedStrideExtractorStage(log_stats=True)

    # Process the task
    result_tasks: list[SplitPipeTask] = run_pipeline([sample_splitting_task], [stage], runner=sequential_runner)

    # Verify there's one task returned
    assert len(result_tasks) == 1

    result_task = result_tasks[0]
    # Verify the task has one video
    video = result_task.video

    # Verify clips were extracted
    assert hasattr(video, "clips")
    assert len(video.clips) == expected_clips_default

    # Verify clip spans match expected values
    for i, expected_span in enumerate(expected_clip_spans_default):
        if i < len(video.clips):
            clip_span = video.clips[i].span
            assert clip_span[0] == pytest.approx(expected_span[0], abs=0.01)
            assert clip_span[1] == pytest.approx(expected_span[1], abs=0.01)

    # Verify stage performance stats were recorded
    assert "FixedStrideExtractorStage" in result_task.stage_perf


@pytest.mark.env("cosmos-curator")
def test_fixed_stride_extraction_5s_stride(
    sample_splitting_task: SplitPipeTask, sequential_runner: RunnerInterface
) -> None:
    """Test fixed stride extraction with 5-second clips and stride.

    Args:
        sample_splitting_task: Sample task with video data
        sequential_runner: Runner for sequential test execution

    """
    # With 5s clips and 5s stride, we expect clips every 5 seconds: 0-5s, 5-10s, ..., 25-30s
    expected_clips_5s_stride = 6
    expected_clip_spans_5s_stride = [
        (0.0, 5.0),
        (5.0, 10.0),
        (10.0, 15.0),
        (15.0, 20.0),
        (20.0, 25.0),
        (25.0, 30.0),
    ]

    stage = FixedStrideExtractorStage(
        clip_len_s=5.0,
        clip_stride_s=5.0,
        min_clip_length_s=5.0,
        log_stats=True,
    )

    # Process the task
    result_tasks = run_pipeline([sample_splitting_task], [stage], runner=sequential_runner)
    video = result_tasks[0].video

    # Verify clips were extracted
    assert len(video.clips) == expected_clips_5s_stride

    # Verify clip spans match expected values
    for i, expected_span in enumerate(expected_clip_spans_5s_stride):
        if i < len(video.clips):
            clip_span = video.clips[i].span
            assert clip_span[0] == pytest.approx(expected_span[0], abs=0.01)
            assert clip_span[1] == pytest.approx(expected_span[1], abs=0.01)


@pytest.mark.env("cosmos-curator")
def test_fixed_stride_extraction_overlapping_clips(
    sample_splitting_task: SplitPipeTask, sequential_runner: RunnerInterface
) -> None:
    """Test fixed stride extraction with overlapping clips (stride < clip_len).

    Args:
        sample_splitting_task: Sample task with video data
        sequential_runner: Runner for sequential test execution

    """
    # With 3s clips and 2s stride, we expect overlapping clips
    # Last clip starting at 28s would be 28-30s = 2s < min 3s, so filtered out
    expected_clips_3s_2s_stride = 14
    expected_clip_spans_3s_2s_stride = [
        (0.0, 3.0),
        (2.0, 5.0),
        (4.0, 7.0),
        (6.0, 9.0),
        (8.0, 11.0),
        (10.0, 13.0),
        (12.0, 15.0),
        (14.0, 17.0),
        (16.0, 19.0),
        (18.0, 21.0),
        (20.0, 23.0),
        (22.0, 25.0),
        (24.0, 27.0),
        (26.0, 29.0),
    ]

    stage = FixedStrideExtractorStage(
        clip_len_s=3.0,
        clip_stride_s=2.0,
        min_clip_length_s=3.0,
        log_stats=True,
    )

    # Process the task
    result_tasks = run_pipeline([sample_splitting_task], [stage], runner=sequential_runner)
    result_task = result_tasks[0]
    video = result_task.video

    # Verify clips were extracted
    assert len(video.clips) == expected_clips_3s_2s_stride

    # Verify clip spans match expected values
    for i, expected_span in enumerate(expected_clip_spans_3s_2s_stride):
        if i < len(video.clips):
            clip_span = video.clips[i].span
            assert clip_span[0] == pytest.approx(expected_span[0], abs=0.01)
            assert clip_span[1] == pytest.approx(expected_span[1], abs=0.01)


@pytest.mark.env("cosmos-curator")
def test_fixed_stride_extraction_with_limit(
    sample_splitting_task: SplitPipeTask, sequential_runner: RunnerInterface
) -> None:
    """Test fixed stride extraction with clip limit.

    Args:
        sample_splitting_task: Sample task with video data
        sequential_runner: Runner for sequential test execution

    """
    # With 3s clips and 2s stride, we expect many overlapping clips, only list the first 2
    expected_clip_spans_3s_2s_stride = [
        (0.0, 3.0),
        (2.0, 5.0),
    ]

    limit = 2
    stage = FixedStrideExtractorStage(
        clip_len_s=3.0,
        clip_stride_s=2.0,
        min_clip_length_s=3.0,
        limit_clips=limit,
        log_stats=True,
    )

    # Process the task
    result_tasks = run_pipeline([sample_splitting_task], [stage], runner=sequential_runner)
    result_task = result_tasks[0]
    video = result_task.video

    # Verify only the limited number of clips were extracted
    assert len(video.clips) == limit

    # Verify the first clips match expected values
    expected_limited_spans = expected_clip_spans_3s_2s_stride
    for i, expected_span in enumerate(expected_limited_spans):
        clip_span = video.clips[i].span
        assert clip_span[0] == pytest.approx(expected_span[0], abs=0.01)
        assert clip_span[1] == pytest.approx(expected_span[1], abs=0.01)


@pytest.mark.env("cosmos-curator")
def test_fixed_stride_extraction_min_clip_length(
    sample_splitting_task: SplitPipeTask, sequential_runner: RunnerInterface
) -> None:
    """Test fixed stride extraction with minimum clip length filtering.

    Args:
        sample_splitting_task: Sample task with video data
        sequential_runner: Runner for sequential test execution

    """
    # Should extract clips: 0-2s, 10-12s, 20-22s (3 clips total for 30s video)
    expected_clips_with_large_stride = 3

    stage = FixedStrideExtractorStage(
        clip_len_s=2.0,
        clip_stride_s=10.0,  # Large stride to get clips at end of video
        min_clip_length_s=1.5,  # Allow shorter clips
        log_stats=True,
    )

    # Process the task
    result_tasks = run_pipeline([sample_splitting_task], [stage], runner=sequential_runner)
    result_task = result_tasks[0]
    video = result_task.video

    # Should extract clips: 0-2s, 10-12s, 20-22s (3 clips total for 30s video)
    assert len(video.clips) == expected_clips_with_large_stride

    # Verify all clips meet minimum length requirement
    for clip in video.clips:
        clip_duration = clip.span[1] - clip.span[0]
        assert clip_duration >= stage.min_clip_length_s - 0.01  # Small tolerance


def test_fixed_stride_extraction_no_clips_short_video(sequential_runner: RunnerInterface) -> None:
    """Test fixed stride extraction with a video shorter than minimum clip length."""
    # Create a task with a very short video duration
    video = Video(
        input_video=pathlib.Path("short_video.mp4"),
        encoded_data=bytes_to_numpy(b"dummy_bytes"),  # We'll mock the metadata
    )

    # Mock metadata for a very short video
    video.metadata = type(
        "MockMetadata",
        (),
        {
            "num_frames": 30,
            "framerate": 30.0,
            "duration": 1.0,  # 1 second video
            "bit_rate_k": 1000,
            "height": 720,
            "width": 1280,
            "video_codec": "h264",  # Added required attribute
            "pixel_format": "yuv420p",  # Added for potential is_10_bit_color() check
            "audio_codec": None,  # Added for completeness
            "size": None,  # Added for completeness
        },
    )()

    task = SplitPipeTask(session_id="test-session", video=video, stage_perf={})

    stage = FixedStrideExtractorStage(
        clip_len_s=5.0,
        min_clip_length_s=5.0,  # Longer than video duration
        log_stats=True,
    )

    # Process the task
    result_tasks = run_pipeline([task], [stage], runner=sequential_runner)
    video = result_tasks[0].video

    # Should extract no clips since video is too short
    assert len(video.clips) == 0


def test_error_handling_no_timestamps(sequential_runner: RunnerInterface) -> None:
    """Test error handling when video has no timestamps (strict mode)."""
    video = Video(
        input_video=pathlib.Path("no_bytes_video.mp4"),
        metadata=VideoMetadata(
            height=720, width=1280, duration=30.0, framerate=30.0, num_frames=900, video_codec="h264"
        ),
        # timestamps defaults to None — strict getter will mark and raise
    )
    task = SplitPipeTask(session_id="test-session", video=video, stage_perf={})

    stage = FixedStrideExtractorStage(log_stats=True)

    run_pipeline([task], [stage], runner=sequential_runner)

    assert "FixedStrideExtractorStage" in task.errors
    assert video.errors["timestamps"] == "missing"


def test_error_handling_incomplete_metadata(sequential_runner: RunnerInterface) -> None:
    """Test error handling when video has incomplete metadata."""
    video = Video(
        input_video=pathlib.Path("incomplete_metadata_video.mp4"),
        encoded_data=bytes_to_numpy(b"dummy_bytes"),
    )

    # Don't extract metadata, leaving it incomplete
    task = SplitPipeTask(session_id="test-session", video=video, stage_perf={})

    stage = FixedStrideExtractorStage(log_stats=True)

    # Process the task
    result_tasks = run_pipeline([task], [stage], runner=sequential_runner)
    result_task = result_tasks[0]
    video = result_task.video

    # Should record metadata error and skip processing
    assert "metadata" in video.errors
    assert video.errors["metadata"] == "incomplete"
    assert len(video.clips) == 0


@pytest.mark.env("cosmos-curator")
def test_clip_uuid_generation(sample_splitting_task: SplitPipeTask, sequential_runner: RunnerInterface) -> None:
    """Test that clip UUIDs are generated consistently and uniquely.

    Args:
        sample_splitting_task: Sample task with video data
        sequential_runner: Runner for sequential test execution

    """
    stage = FixedStrideExtractorStage(
        clip_len_s=5.0,
        clip_stride_s=5.0,
        min_clip_length_s=5.0,
    )

    # Process the task twice with separate copies to avoid accumulating clips
    task_copy_1 = copy.deepcopy(sample_splitting_task)
    task_copy_2 = copy.deepcopy(sample_splitting_task)

    result_tasks_1 = run_pipeline([task_copy_1], [stage], runner=sequential_runner)
    result_tasks_2 = run_pipeline([task_copy_2], [stage], runner=sequential_runner)

    video_1 = result_tasks_1[0].video
    video_2 = result_tasks_2[0].video

    # UUIDs should be consistent across runs
    assert len(video_1.clips) == len(video_2.clips)
    for clip_1, clip_2 in zip(video_1.clips, video_2.clips, strict=False):
        assert clip_1.uuid == clip_2.uuid

    # UUIDs should be unique within a video
    uuids = [clip.uuid for clip in video_1.clips]
    assert len(uuids) == len(set(uuids))  # All UUIDs are unique


@pytest.mark.env("cosmos-curator")
def test_clip_properties(sample_splitting_task: SplitPipeTask, sequential_runner: RunnerInterface) -> None:
    """Test that extracted clips have correct properties.

    Args:
        sample_splitting_task: Sample task with video data
        sequential_runner: Runner for sequential test execution

    """
    # Span tuple contains start and end times
    expected_span_tuple_length = 2

    stage = FixedStrideExtractorStage(
        clip_len_s=5.0,
        clip_stride_s=5.0,
        min_clip_length_s=5.0,
    )

    # Process the task
    result_tasks = run_pipeline([sample_splitting_task], [stage], runner=sequential_runner)
    video = result_tasks[0].video

    # Verify clip properties
    for clip in video.clips:
        # Check that clip has required attributes
        assert hasattr(clip, "uuid")
        assert hasattr(clip, "encoded_data")
        assert hasattr(clip, "span")

        # Check that source_video matches input
        assert clip.source_video == str(video.input_video)

        # Check that span is valid
        assert len(clip.span) == expected_span_tuple_length
        assert clip.span[0] >= 0
        assert clip.span[1] > clip.span[0]

        # Check that clip duration matches expected length (within tolerance)
        clip_duration = clip.span[1] - clip.span[0]
        assert clip_duration <= stage.clip_len_s + 0.01  # Small tolerance


@pytest.mark.env("cosmos-curator")
def test_verbose_logging(
    sample_splitting_task: SplitPipeTask, sequential_runner: RunnerInterface, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that verbose logging works correctly.

    Args:
        sample_splitting_task: Sample task with video data
        sequential_runner: Runner for sequential test execution
        caplog: Pytest fixture for capturing log output

    """
    stage = FixedStrideExtractorStage(
        clip_len_s=5.0,
        clip_stride_s=5.0,
        verbose=True,
        log_stats=True,
    )

    # Process the task
    with caplog.at_level("INFO"):
        result_tasks = run_pipeline([sample_splitting_task], [stage], runner=sequential_runner)

    # Verify that some logging occurred (exact messages may vary)
    # The stage itself doesn't have explicit verbose logging, but this tests the parameter
    assert len(result_tasks) == 1


_FAKE_TIMESTAMPS = np.array([0.0, 0.033, 0.066, 0.1], dtype=np.float32)


@pytest.mark.parametrize(
    ("videos", "raises"),
    [
        pytest.param(
            [Video(input_video=pathlib.Path("video1.mp4"), timestamps=_FAKE_TIMESTAMPS)],
            nullcontext(),
            id="single_video_valid",
        ),
        pytest.param(
            [
                Video(input_video=pathlib.Path("video1.mp4"), timestamps=_FAKE_TIMESTAMPS),
                Video(input_video=pathlib.Path("video2.mp4"), timestamps=_FAKE_TIMESTAMPS),
            ],
            nullcontext(),
            id="multicam_two_videos_valid",
        ),
        pytest.param(
            [
                Video(input_video=pathlib.Path("video1.mp4"), timestamps=_FAKE_TIMESTAMPS),
                Video(input_video=pathlib.Path("video2.mp4"), timestamps=_FAKE_TIMESTAMPS),
                Video(input_video=pathlib.Path("video3.mp4"), timestamps=_FAKE_TIMESTAMPS),
            ],
            nullcontext(),
            id="multicam_three_videos_valid",
        ),
        pytest.param(
            [Video(input_video=pathlib.Path("video1.mp4"), timestamps=None)],
            pytest.raises(ValueError, match=r"Videos missing timestamps"),
            id="single_video_timestamps_none",
        ),
        pytest.param(
            [Video(input_video=pathlib.Path("video1.mp4"), timestamps=np.array([], dtype=np.float32))],
            pytest.raises(ValueError, match=r"Videos missing timestamps"),
            id="single_video_timestamps_empty",
        ),
        pytest.param(
            [
                Video(input_video=pathlib.Path("video1.mp4"), timestamps=None),
                Video(input_video=pathlib.Path("video2.mp4"), timestamps=_FAKE_TIMESTAMPS),
            ],
            pytest.raises(ValueError, match=r"Videos missing timestamps"),
            id="multicam_first_video_timestamps_none",
        ),
        pytest.param(
            [
                Video(input_video=pathlib.Path("video1.mp4"), timestamps=_FAKE_TIMESTAMPS),
                Video(input_video=pathlib.Path("video2.mp4"), timestamps=None),
            ],
            pytest.raises(ValueError, match=r"Videos missing timestamps"),
            id="multicam_second_video_timestamps_none",
        ),
    ],
)
def test_get_videos_timestamps(
    videos: list[Video],
    raises: AbstractContextManager[Any],
) -> None:
    """Test _get_videos_timestamps reads video.timestamps directly and raises on missing."""
    with raises:
        result = _get_videos_timestamps(videos)
        assert len(result) == len(videos)
        for timestamps in result:
            assert isinstance(timestamps, np.ndarray)
            assert np.array_equal(timestamps, _FAKE_TIMESTAMPS)


def test_validate_video_timestamps_empty_list() -> None:
    """Test _validate_video_timestamps raises when video_timestamps is empty."""
    with pytest.raises(ValueError, match=r"No timestamps found for videos"):
        _validate_video_timestamps([])


def test_validate_video_timestamps_some_empty_arrays() -> None:
    """Test _validate_video_timestamps raises when any video has no timestamps."""
    ts = np.array([0.0, 1.0], dtype=np.float32)
    empty = np.array([], dtype=np.float32)
    with pytest.raises(ValueError, match=r"Some videos have no timestamps"):
        _validate_video_timestamps([ts, empty])
    with pytest.raises(ValueError, match=r"Some videos have no timestamps"):
        _validate_video_timestamps([empty, ts])
    with pytest.raises(ValueError, match=r"Some videos have no timestamps"):
        _validate_video_timestamps([empty])


def test_validate_video_timestamps_valid() -> None:
    """Test _validate_video_timestamps passes when all videos have timestamps."""
    ts1 = np.array([0.0, 0.033, 0.1], dtype=np.float32)
    ts2 = np.array([0.0, 1.0], dtype=np.float32)
    _validate_video_timestamps([ts1])
    _validate_video_timestamps([ts1, ts2])


def test_get_videos_durations_empty_list() -> None:
    """Test _get_videos_durations returns empty list for no videos."""
    assert _get_videos_durations([]) == []


def test_get_videos_durations_single_video() -> None:
    """Test _get_videos_durations with one video: duration = num_frames / framerate."""
    video = Video(
        input_video=pathlib.Path("v.mp4"),
        encoded_data=bytes_to_numpy(b"x"),
        metadata=VideoMetadata(num_frames=900, framerate=30.0),
    )
    assert _get_videos_durations([video]) == [30.0]


def test_get_videos_durations_zero_framerate_returns_minus_one() -> None:
    """Test _get_videos_durations returns -1 when framerate is 0."""
    video = Video(
        input_video=pathlib.Path("v.mp4"),
        encoded_data=bytes_to_numpy(b"x"),
        metadata=VideoMetadata(num_frames=100, framerate=0.0),
    )
    assert _get_videos_durations([video]) == [-1]


def test_get_videos_durations_multiple_videos() -> None:
    """Test _get_videos_durations with multiple videos returns one duration per video."""
    v1 = Video(
        input_video=pathlib.Path("a.mp4"),
        encoded_data=bytes_to_numpy(b"x"),
        metadata=VideoMetadata(num_frames=300, framerate=30.0),
    )
    v2 = Video(
        input_video=pathlib.Path("b.mp4"),
        encoded_data=bytes_to_numpy(b"y"),
        metadata=VideoMetadata(num_frames=60, framerate=24.0),
    )
    durations = _get_videos_durations([v1, v2])
    assert durations == [10.0, 2.5]


def test_get_videos_durations_mixed_valid_and_zero_framerate() -> None:
    """Test _get_videos_durations when some videos have zero framerate."""
    v1 = Video(
        input_video=pathlib.Path("a.mp4"),
        encoded_data=bytes_to_numpy(b"x"),
        metadata=VideoMetadata(num_frames=90, framerate=30.0),
    )
    v2 = Video(
        input_video=pathlib.Path("b.mp4"),
        encoded_data=bytes_to_numpy(b"y"),
        metadata=VideoMetadata(num_frames=100, framerate=0.0),
    )
    durations = _get_videos_durations([v1, v2])
    assert durations == [3.0, -1]


def test_make_spans_fixed_stride_basic() -> None:
    """Test _make_spans_fixed_stride with basic scenarios."""
    # Create fake timestamps for 2 videos: 30 seconds at 30fps
    timestamps1 = np.linspace(0.0, 30.0, 900, dtype=np.float32)
    timestamps2 = np.linspace(0.0, 30.0, 900, dtype=np.float32)
    video_timestamps = [timestamps1, timestamps2]
    start_s = float(np.max([ts[0] for ts in video_timestamps]))
    end_s = float(np.min([ts[-1] for ts in video_timestamps]))
    duration_s = end_s - start_s

    # Test with 10s clips, 10s stride (non-overlapping)
    spans = _make_spans_fixed_stride(
        start_s=start_s,
        end_s=duration_s,
        clip_len_s=10.0,
        clip_stride_s=10.0,
        min_clip_length_s=5.0,
    )

    # Should get 3 clips: 0-10, 10-20, 20-30
    assert len(spans) == 3
    assert spans[0] == (0.0, 10.0)
    assert spans[1] == (10.0, 20.0)
    assert spans[2] == (20.0, 30.0)


def test_make_spans_fixed_stride_overlapping() -> None:
    """Test _make_spans_fixed_stride with overlapping clips."""
    # Create fake timestamps for 1 video: 30 seconds
    timestamps = np.linspace(0.0, 30.0, 900, dtype=np.float32)
    video_timestamps = [timestamps]
    start_s = float(np.max([ts[0] for ts in video_timestamps]))
    end_s = float(np.min([ts[-1] for ts in video_timestamps]))
    duration_s = end_s - start_s

    # Test with 10s clips, 5s stride (50% overlap)
    spans = _make_spans_fixed_stride(
        start_s=start_s,
        end_s=duration_s,
        clip_len_s=10.0,
        clip_stride_s=5.0,
        min_clip_length_s=5.0,
    )

    # Should get 6 clips: 0-10, 5-15, 10-20, 15-25, 20-30, 25-30 (last one is 5s)
    assert len(spans) == 6
    assert spans[0] == (0.0, 10.0)
    assert spans[1] == (5.0, 15.0)
    assert spans[2] == (10.0, 20.0)
    assert spans[3] == (15.0, 25.0)
    assert spans[4] == (20.0, 30.0)
    assert spans[5] == (25.0, 30.0)  # Last clip is only 5s but meets min_clip_length


def test_make_spans_fixed_stride_min_clip_length() -> None:
    """Test _make_spans_fixed_stride respects minimum clip length."""
    # Create fake timestamps: 12 seconds
    timestamps = np.linspace(0.0, 12.0, 360, dtype=np.float32)
    video_timestamps = [timestamps]
    start_s = float(np.max([ts[0] for ts in video_timestamps]))
    end_s = float(np.min([ts[-1] for ts in video_timestamps]))
    duration_s = end_s - start_s

    # Test with 10s clips, 10s stride, min 8s
    # Should get: 0-10 (10s ✓), 10-12 (2s ✗ too short)
    spans = _make_spans_fixed_stride(
        start_s=start_s,
        end_s=duration_s,
        clip_len_s=10.0,
        clip_stride_s=10.0,
        min_clip_length_s=8.0,
    )

    # Last clip should be filtered out
    assert len(spans) == 1
    assert spans[0] == (0.0, 10.0)


def test_make_spans_fixed_stride_shared_overlap() -> None:
    """Test _make_spans_fixed_stride finds shared temporal overlap across cameras."""
    # Camera 1: starts at 0s, ends at 30s
    timestamps1 = np.linspace(0.0, 30.0, 900, dtype=np.float32)
    # Camera 2: starts at 5s, ends at 25s (different range!)
    timestamps2 = np.linspace(5.0, 25.0, 600, dtype=np.float32)
    video_timestamps = [timestamps1, timestamps2]
    start_s = float(np.max([ts[0] for ts in video_timestamps]))
    end_s = float(np.min([ts[-1] for ts in video_timestamps]))

    # Should only create clips in the shared range: 5-25s
    spans = _make_spans_fixed_stride(
        start_s=start_s,
        end_s=end_s,
        clip_len_s=10.0,
        clip_stride_s=10.0,
        min_clip_length_s=10.0,
    )

    # Clips: 5-15, 15-25 (only 2 clips in shared overlap)
    assert len(spans) == 2
    assert spans[0] == (5.0, 15.0)
    assert spans[1] == (15.0, 25.0)


def test_make_clip_uuids_deterministic() -> None:
    """Test _make_clip_uuids creates deterministic UUIDs."""
    session_id = "test_session_123"
    spans = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]

    # Generate UUIDs twice
    uuids1 = _make_clip_uuids(session_id, spans)
    uuids2 = _make_clip_uuids(session_id, spans)

    # Should be identical (deterministic)
    assert len(uuids1) == 3
    assert len(uuids2) == 3
    assert uuids1 == uuids2

    # Each UUID should be unique
    assert len(set(uuids1)) == 3


def test_make_clip_uuids_different_sessions() -> None:
    """Test _make_clip_uuids creates different UUIDs for different sessions."""
    spans = [(0.0, 10.0), (10.0, 20.0)]

    uuids_session1 = _make_clip_uuids("session_1", spans)
    uuids_session2 = _make_clip_uuids("session_2", spans)

    # Different sessions should produce different UUIDs
    assert uuids_session1 != uuids_session2


def test_make_clip_uuids_different_spans() -> None:
    """Test _make_clip_uuids creates different UUIDs for different spans."""
    session_id = "test_session"
    spans1 = [(0.0, 10.0), (10.0, 20.0)]
    spans2 = [(5.0, 15.0), (15.0, 25.0)]

    uuids1 = _make_clip_uuids(session_id, spans1)
    uuids2 = _make_clip_uuids(session_id, spans2)

    # Different spans should produce different UUIDs
    assert uuids1 != uuids2


def test_populate_clips_fixed_stride() -> None:
    """Test _populate_clips_fixed_stride populates clips correctly."""
    # Create test videos with encoded data and timestamps
    # 30 second video at 30fps = 900 frames
    timestamps = np.linspace(0.0, 30.0, 900, dtype=np.float32)

    video1 = Video(
        input_video=pathlib.Path("cam_front.mp4"),
        encoded_data=bytes_to_numpy(b"dummy_data"),
        metadata=VideoMetadata(
            height=720,
            width=1280,
            framerate=30.0,
            num_frames=900,
            duration=30.0,
            video_codec="h264",
            pixel_format="yuv420p",
        ),
    )
    video2 = Video(
        input_video=pathlib.Path("cam_rear.mp4"),
        encoded_data=bytes_to_numpy(b"dummy_data_2"),
        metadata=VideoMetadata(
            height=720,
            width=1280,
            framerate=30.0,
            num_frames=900,
            duration=30.0,
            video_codec="h264",
            pixel_format="yuv420p",
        ),
    )

    video1.timestamps = timestamps
    video2.timestamps = timestamps
    videos = [video1, video2]

    # Populate clips: 10s clips, 10s stride (non-overlapping)
    _populate_clips_fixed_stride(
        videos=videos,
        session_id="test_session_123",
        clip_len_s=10.0,
        clip_stride_s=10.0,
        min_clip_length_s=5.0,
    )

    # Both videos should have 3 clips (0-10, 10-20, 20-30)
    assert len(video1.clips) == 3
    assert len(video2.clips) == 3

    # Verify clips have correct spans
    assert video1.clips[0].span == (0.0, 10.0)
    assert video1.clips[1].span == (10.0, 20.0)
    assert video1.clips[2].span == (20.0, 30.0)

    # Verify both videos have matching UUIDs for corresponding clips
    for i in range(3):
        assert video1.clips[i].uuid == video2.clips[i].uuid
        assert video1.clips[i].span == video2.clips[i].span

    # Verify source_video is set correctly
    assert video1.clips[0].source_video == "cam_front.mp4"
    assert video2.clips[0].source_video == "cam_rear.mp4"

    # Test limit_clips: clear and repopulate with limit
    video1.clips.clear()
    video2.clips.clear()
    _populate_clips_fixed_stride(
        videos=videos,
        session_id="test_session_123",
        clip_len_s=10.0,
        clip_stride_s=10.0,
        min_clip_length_s=5.0,
        limit_clips=2,
    )

    # Both videos should have only 2 clips (first two spans)
    assert len(video1.clips) == 2
    assert len(video2.clips) == 2
    assert video1.clips[0].span == (0.0, 10.0)
    assert video1.clips[1].span == (10.0, 20.0)
    assert video2.clips[0].span == (0.0, 10.0)
    assert video2.clips[1].span == (10.0, 20.0)
    for i in range(2):
        assert video1.clips[i].uuid == video2.clips[i].uuid


def _make_multicam_task_two_cameras() -> SplitPipeTask:
    """Build a multicam task with two videos (synthetic metadata, no network)."""
    meta = VideoMetadata(
        height=720,
        width=1280,
        framerate=30.0,
        num_frames=900,
        duration=30.0,
        video_codec="h264",
        pixel_format="yuv420p",
        audio_codec=None,
        size=1000,
    )
    timestamps = np.linspace(0.0, 30.0, 900, dtype=np.float32)
    v1 = Video(
        input_video=pathlib.Path("cam_front.mp4"),
        encoded_data=bytes_to_numpy(b"dummy"),
        metadata=meta,
        timestamps=timestamps,
    )
    v2 = Video(
        input_video=pathlib.Path("cam_rear.mp4"),
        encoded_data=bytes_to_numpy(b"dummy"),
        metadata=VideoMetadata(
            height=meta.height,
            width=meta.width,
            framerate=meta.framerate,
            num_frames=meta.num_frames,
            duration=meta.duration,
            video_codec=meta.video_codec,
            pixel_format=meta.pixel_format,
            audio_codec=meta.audio_codec,
            size=meta.size,
        ),
        timestamps=timestamps,
    )
    return SplitPipeTask(session_id="test-multicam-session", videos=[v1, v2])


def test_fixed_stride_multicam_aligned_clips(
    sequential_runner: RunnerInterface,
) -> None:
    """Multi-cam task gets aligned clips: same count, uuid, and span per column across cameras."""
    # _make_multicam_task_two_cameras() sets timestamps=np.linspace(0,30,900) on each video
    task = _make_multicam_task_two_cameras()

    stage = FixedStrideExtractorStage(
        clip_len_s=5.0,
        clip_stride_s=5.0,
        min_clip_length_s=5.0,
    )
    result_tasks = run_pipeline([task], [stage], runner=sequential_runner)
    assert len(result_tasks) == 1
    result = result_tasks[0]
    # Multi-cam task should have multiple videos
    assert len(result.videos) == 2

    clips0 = result.videos[0].clips
    clips1 = result.videos[1].clips
    assert len(clips0) == len(clips1), "Both cameras must have same number of clips"
    for j in range(len(clips0)):
        assert clips0[j].uuid == clips1[j].uuid, f"Clip index {j}: uuid must match across cameras"
        assert clips0[j].span == clips1[j].span, f"Clip index {j}: span must match across cameras"


def test_make_spans() -> None:
    """Test that the spans cover the entire duration of the video."""
    expected_clip_spans_5s_stride = [
        (0.0, 5.0),
        (5.0, 10.0),
        (10.0, 15.0),
        (15.0, 20.0),
        (20.0, 25.0),
        (25.0, 30.0),
    ]

    start_s = expected_clip_spans_5s_stride[0][0]
    end_s = expected_clip_spans_5s_stride[-1][1]

    spans = _make_spans_fixed_stride(
        start_s=start_s,
        end_s=end_s,
        clip_len_s=5.0,
        clip_stride_s=5.0,
        min_clip_length_s=5.0,
    )

    assert len(spans) == len(expected_clip_spans_5s_stride)
    for (start, end), (exp_start, exp_end) in zip(spans, expected_clip_spans_5s_stride, strict=True):
        assert start == pytest.approx(exp_start, abs=0.01)
        assert end == pytest.approx(exp_end, abs=0.01)


# ---------------------------------------------------------------------------
# New tests for CVC-694 strict timestamp consumption
# ---------------------------------------------------------------------------


def test_fixed_stride_extractor_uses_timestamps(sequential_runner: RunnerInterface) -> None:
    """FixedStrideExtractorStage uses video.timestamps; no real encoded_data needed."""
    video = Video(
        input_video=pathlib.Path("fake.mp4"),
        encoded_data=bytes_to_numpy(b"not real"),
        metadata=VideoMetadata(
            num_frames=900,
            framerate=30.0,
            duration=30.0,
            height=720,
            width=1280,
            video_codec="h264",
            pixel_format="yuv420p",
            size=8,
        ),
        timestamps=np.linspace(0.0, 30.0, 900, dtype=np.float32),
    )
    task = SplitPipeTask(session_id="ts-test", video=video)
    stage = FixedStrideExtractorStage(clip_len_s=10.0, clip_stride_s=10.0, min_clip_length_s=10.0)

    result_tasks = run_pipeline([task], [stage], runner=sequential_runner)

    assert len(result_tasks[0].video.clips) == 3
    assert result_tasks[0].video.clips[0].span == (0.0, 10.0)
    assert result_tasks[0].video.clips[2].span == (20.0, 30.0)


def test_fixed_stride_extractor_no_encoded_data_needed(sequential_runner: RunnerInterface) -> None:
    """FixedStrideExtractorStage succeeds with encoded_data absent when timestamps are set.

    Guards removal of _require_video_bytes: the stage must not touch encoded_data.
    """
    video = Video(
        input_video=pathlib.Path("fake.mp4"),
        metadata=VideoMetadata(
            num_frames=900,
            framerate=30.0,
            duration=30.0,
            height=720,
            width=1280,
            video_codec="h264",
            pixel_format="yuv420p",
            size=0,
        ),
        timestamps=np.linspace(0.0, 30.0, 900, dtype=np.float32),
    )
    task = SplitPipeTask(session_id="ts-test-no-bytes", video=video)
    stage = FixedStrideExtractorStage(clip_len_s=10.0, clip_stride_s=10.0, min_clip_length_s=10.0)

    result_tasks = run_pipeline([task], [stage], runner=sequential_runner)

    assert len(result_tasks[0].video.clips) == 3
    assert "encoded_data" not in result_tasks[0].video.errors


def test_get_videos_timestamps_marks_all_missing_multicam() -> None:
    """Both cameras get errors['timestamps']='missing' before raise (full-pass marking)."""
    v1 = Video(input_video=pathlib.Path("cam1.mp4"), timestamps=None)
    v2 = Video(input_video=pathlib.Path("cam2.mp4"), timestamps=None)

    with pytest.raises(ValueError, match=r"Videos missing timestamps"):
        _get_videos_timestamps([v1, v2])

    assert v1.errors.get("timestamps") == "missing"
    assert v2.errors.get("timestamps") == "missing"


def test_slice_video_clips_preserves_timestamps() -> None:
    """slice_video_clips() propagates video.timestamps to the chunked subtask Video."""
    timestamps = np.linspace(0.0, 30.0, 900, dtype=np.float32)
    video = Video(
        input_video=pathlib.Path("v.mp4"),
        timestamps=timestamps,
        clips=[
            Clip(uuid=uuid.uuid4(), source_video="v.mp4", span=(float(i * 10), float(i * 10 + 10))) for i in range(3)
        ],
    )

    sliced = slice_video_clips(video, 0, 2, chunk_index=0, num_chunks=2)

    assert sliced.timestamps is timestamps  # same reference


def test_get_videos_timestamps_preserves_existing_error() -> None:
    """set-if-absent: pre-set errors['timestamps'] is not overwritten with 'missing'."""
    video = Video(input_video=pathlib.Path("v.mp4"), timestamps=None)
    video.errors["timestamps"] = "decoder failure"

    with pytest.raises(ValueError, match=r"Videos missing timestamps"):
        _get_videos_timestamps([video])

    assert video.errors["timestamps"] == "decoder failure"


def test_timestamp_failure_nonfatal_at_download_fatal_at_fixed_stride(
    sequential_runner: RunnerInterface,
) -> None:
    """Downloader error detail is preserved; stage fails with that detail in errors."""
    video = Video(
        input_video=pathlib.Path("v.mp4"),
        encoded_data=bytes_to_numpy(b"x"),
        metadata=VideoMetadata(
            num_frames=900,
            framerate=30.0,
            duration=30.0,
            height=720,
            width=1280,
            video_codec="h264",
            pixel_format="yuv420p",
            size=8,
        ),
        timestamps=None,
    )
    video.errors["timestamps"] = "decoder failure"  # simulates downloader recording the error
    task = SplitPipeTask(session_id="ts-fail", video=video)
    stage = FixedStrideExtractorStage(clip_len_s=10.0, clip_stride_s=10.0, min_clip_length_s=10.0)

    result_tasks = run_pipeline([task], [stage], runner=sequential_runner)

    assert result_tasks[0].errors.get("FixedStrideExtractorStage") is not None
    # Downloader error detail preserved; not overwritten with "missing"
    assert result_tasks[0].videos[0].errors["timestamps"] == "decoder failure"
