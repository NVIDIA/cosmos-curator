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
"""Integration tests for timestamp population using real video samples.

These tests are env-gated (cosmos-curator environment) and require ffprobe/ffmpeg
on PATH. They are excluded from the default CPU test selection and must be run
explicitly or via the env-marked suite:

    pytest -m env tests/cosmos_curator/pipelines/video/test_timestamps_integration.py -v
"""

import pathlib
from unittest.mock import Mock

import numpy as np
import pytest

from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.pipelines.video.clipping.clip_extraction_stages import (
    FixedStrideExtractorStage,
    chunk_tasks,
)
from cosmos_curator.pipelines.video.read_write.download_stages import VideoDownloader
from cosmos_curator.pipelines.video.utils.data_model import SplitPipeTask, Video

_FIXTURES_DIR = pathlib.Path(__file__).parent / "data"
_SAMPLE_VIDEO_PATH = _FIXTURES_DIR / "test_video_30s.mp4"

_DURATION_TOLERANCE_S = 0.1

pytestmark = pytest.mark.env("cosmos-curator")


def _make_downloader() -> VideoDownloader:
    """VideoDownloader wired for local-path mode (no S3 client needed)."""
    downloader = VideoDownloader(input_path="/fake", input_s3_profile_name="default")
    downloader._client = None
    return downloader


def _make_mock_task(video: Video) -> Mock:
    task = Mock(spec=SplitPipeTask)
    task.videos = [video]
    task.get_major_size.return_value = 0
    task.stage_perf = {}
    return task


def test_timestamps_basic_sanity_real_mp4(sample_video_data: bytes) -> None:
    """Timestamps from a real MP4 pass all basic sanity checks.

    Non-empty, float32, monotonic, non-negative, finite, and last value within
    duration + tolerance.
    """
    video = Video(
        input_video=pathlib.Path("test.mp4"),
        encoded_data=bytes_to_numpy(sample_video_data),
    )
    video.populate_metadata()
    video.populate_timestamps()

    ts = video.timestamps
    assert ts is not None
    assert len(ts) > 0
    assert ts.dtype == np.float32
    assert np.all(ts >= 0.0), "negative timestamps"
    assert np.all(np.isfinite(ts)), "non-finite timestamps"
    assert np.all(np.diff(ts) >= 0), "timestamps not monotonically increasing"
    assert video.metadata.duration is not None
    assert float(ts[-1]) <= video.metadata.duration + _DURATION_TOLERANCE_S, (
        f"last timestamp {ts[-1]:.3f}s exceeds duration {video.metadata.duration:.3f}s"
    )


def test_major_size_delta_equals_timestamps_nbytes(sample_video_data: bytes) -> None:
    """get_major_size() increases by exactly timestamps.nbytes after populate_timestamps()."""
    video = Video(
        input_video=pathlib.Path("test.mp4"),
        encoded_data=bytes_to_numpy(sample_video_data),
    )
    video.populate_metadata()
    size_before = video.get_major_size()

    video.populate_timestamps()
    assert video.timestamps is not None
    assert video.get_major_size() == size_before + video.timestamps.nbytes


def test_downloader_populates_timestamps_real_mp4() -> None:
    """VideoDownloader sets valid timestamps on a real local MP4 file."""
    video = Video(input_video=_SAMPLE_VIDEO_PATH)
    _make_downloader().process_data([_make_mock_task(video)])

    assert "timestamps" not in video.errors
    ts = video.timestamps
    assert ts is not None
    assert len(ts) > 0
    assert ts.dtype == np.float32
    assert np.all(ts >= 0.0)
    assert np.all(np.diff(ts) >= 0)


def test_downloader_then_fixed_stride_real_mp4_produces_expected_clip_count() -> None:
    """Downloader → FixedStrideExtractorStage chain on the 30s sample produces 3 clips."""
    video = Video(input_video=_SAMPLE_VIDEO_PATH)
    _make_downloader().process_data([_make_mock_task(video)])

    assert "timestamps" not in video.errors
    assert video.timestamps is not None

    task = SplitPipeTask(session_id="test-session", video=video, stage_perf={})
    stage = FixedStrideExtractorStage(clip_len_s=10, clip_stride_s=10, min_clip_length_s=10)
    result = stage.process_data([task])

    assert result is not None
    assert "FixedStrideExtractorStage" not in result[0].errors
    assert len(result[0].videos[0].clips) == 3


def test_multicam_fixed_stride_alignment(sample_multicam_task: SplitPipeTask) -> None:
    """Both cameras in a multi-camera task produce identical clip spans after extraction."""
    stage = FixedStrideExtractorStage(clip_len_s=10, clip_stride_s=10, min_clip_length_s=10)
    result = stage.process_data([sample_multicam_task])

    assert result is not None
    task = result[0]
    assert "FixedStrideExtractorStage" not in task.errors
    assert len(task.videos) == 2

    clips_cam0 = task.videos[0].clips
    clips_cam1 = task.videos[1].clips
    assert len(clips_cam0) == len(clips_cam1) == 3

    for i, (c0, c1) in enumerate(zip(clips_cam0, clips_cam1, strict=True)):
        assert c0.span == c1.span, f"clip {i} spans misaligned: cam0={c0.span} cam1={c1.span}"


def test_chunked_tasks_preserve_timestamps_reference(
    sample_splitting_task: SplitPipeTask,
) -> None:
    """chunk_tasks via slice_video_clips carries the same timestamps array (no copy).

    Verifies clip distribution across chunks and that the timestamps reference is
    preserved in each sliced Video, satisfying the encoded_data mutation contract.
    """
    stage = FixedStrideExtractorStage(clip_len_s=10, clip_stride_s=10, min_clip_length_s=10)
    result = stage.process_data([sample_splitting_task])
    assert result is not None

    original_ts = result[0].videos[0].timestamps
    assert original_ts is not None

    # Chunk into subtasks of 2 clips each: [2, 1] from the 3-clip video
    subtasks = chunk_tasks(result, num_clips_per_chunk=2)
    assert len(subtasks) == 2
    assert len(subtasks[0].videos[0].clips) == 2
    assert len(subtasks[1].videos[0].clips) == 1

    for subtask in subtasks:
        assert subtask.videos[0].timestamps is original_ts
