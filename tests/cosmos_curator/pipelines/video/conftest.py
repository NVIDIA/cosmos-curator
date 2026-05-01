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
"""Shared fixtures for video model tests."""

import pathlib
import uuid
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.core.utils.data.lazy_data import LazyData
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video
from cosmos_curator.pipelines.video.utils.decoder_utils import (
    FrameExtractionPolicy,
    FrameExtractionSignature,
    extract_frames,
)

# Local test video fixtures (Sintel trailer segments, Creative Commons Attribution 3.0)
_FIXTURES_DIR = Path(__file__).parent / "data"
_SAMPLE_CLIP_PATH = _FIXTURES_DIR / "test_clip_10s.mp4"
_SAMPLE_VIDEO_PATH = _FIXTURES_DIR / "test_video_30s.mp4"


@pytest.fixture(scope="session")
def sample_clip_data() -> bytes:
    """Provide the shorter sample video data (10s clip).

    Returns:
        bytes: The video file content

    """
    return _SAMPLE_CLIP_PATH.read_bytes()


@pytest.fixture(scope="session")
def sample_video_data() -> bytes:
    """Provide the longer sample video data (30s video).

    Returns:
        bytes: The video file content

    """
    return _SAMPLE_VIDEO_PATH.read_bytes()


@pytest.fixture
def sample_splitting_task(sample_video_data: bytes) -> SplitPipeTask:
    """Fixture to create a sample pipeline task with a video for splitting/extraction tests.

    This fixture creates a SplitPipeTask with a video that has extracted metadata
    and initialized cutscenes list. This is suitable for both panda extraction
    and fixed stride extraction tests.

    Args:
        sample_video_data: The video file content

    Returns:
        SplitPipeTask: Task with a video for testing

    """
    video = Video(
        input_video=pathlib.Path("sample_video.mp4"),
        encoded_data=bytes_to_numpy(sample_video_data),
    )
    video.populate_metadata()
    video.populate_timestamps()

    return SplitPipeTask(
        session_id="test-session",
        video=video,
        stage_perf={},
    )


@pytest.fixture(autouse=True)
def mock_get_tmp_dir(tmp_path: Path) -> Generator[None, None, None]:
    """Automatically mock get_tmp_dir to use pytest's tmp_path for all tests.

    This prevents permission errors when tests try to create temp files in
    system directories that may not be writable in CI/CD environments.

    By default, get_tmp_dir will return /config/tmp in CI/CD environments,
    which is not writable. This fixture will mock get_tmp_dir to return the
    pytest tmp_path, which is writable, is guaranteed to be unique for each
    test, and will be cleaned up automatically.

    Args:
        tmp_path: pytest's temporary directory fixture

    """
    with patch("cosmos_curator.core.utils.config.operation_context.get_tmp_dir", return_value=tmp_path):
        yield


@pytest.fixture
def sample_filtering_task(sample_clip_data: bytes) -> SplitPipeTask:
    """Fixture to create a sample pipeline task with a video clip for filtering tests.

    This fixture creates a SplitPipeTask with a single clip that has pre-extracted frames
    at 1.0 fps. This works for both aesthetic filtering (which requires frames) and
    motion filtering (which ignores frames and works with the video buffer directly).

    Args:
        sample_clip_data: The video file content

    Returns:
        SplitPipeTask: Task with a video and clip for testing

    """
    clip = Clip(
        uuid=uuid.UUID("12345678-1234-5678-1234-567812345678"),
        source_video="sample_video.mp4",
        span=(0.0, 10.0),
        encoded_data=bytes_to_numpy(sample_clip_data),
    )

    # Always extract frames at 1.0 fps for simplicity in tests
    frame_extraction_sig = FrameExtractionSignature(
        extraction_policy=FrameExtractionPolicy.sequence, target_fps=1.0
    ).to_str()

    # Extract real frames from the video buffer
    frames = extract_frames(
        video=sample_clip_data, extraction_policy=FrameExtractionPolicy.sequence, sample_rate_fps=1.0
    )

    # Add the extracted frames to the clip (wrapped in LazyData for inter-stage transport)
    clip.extracted_frames = LazyData(value={frame_extraction_sig: frames}, nbytes=frames.nbytes)

    video = Video(
        input_video=pathlib.Path("sample_video.mp4"),
        clips=[clip],
        filtered_clips=[],
    )

    return SplitPipeTask(
        session_id="test-session",
        video=video,
        stage_perf={},
    )


@pytest.fixture
def sample_multicam_task(sample_video_data: bytes) -> SplitPipeTask:
    """Fixture to create a sample multi-camera pipeline task for testing.

    This fixture creates a SplitPipeTask with multiple videos representing different camera angles.
    Each video has extracted metadata. This is suitable for testing multi-camera pipeline features.

    Args:
        sample_video_data: The video file content (used for both cameras)

    Returns:
        SplitPipeTask: Multi-camera task with multiple videos for testing

    """
    # Create two videos representing different camera angles
    video1 = Video(
        input_video=pathlib.Path("camera_1.mp4"),
        encoded_data=bytes_to_numpy(sample_video_data),
    )
    video1.populate_metadata()
    video1.populate_timestamps()

    video2 = Video(
        input_video=pathlib.Path("camera_2.mp4"),
        encoded_data=bytes_to_numpy(sample_video_data),
    )
    video2.populate_metadata()
    video2.populate_timestamps()

    return SplitPipeTask(
        session_id="test-session",
        videos=[video1, video2],
        stage_perf={},
    )
