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
"""Test remux helpers for video pipelines."""

import io
from fractions import Fraction
from typing import cast
from unittest.mock import MagicMock

import av
import numpy as np
import pytest

from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.pipelines.video.read_write.remux_stages import (
    remux_if_needed,
    remux_to_mp4,
)
from cosmos_curator.pipelines.video.utils.data_model import Video


def _make_synthetic_video(format_name: str) -> io.BytesIO:
    """Create a synthetic video in the specified format.

    Args:
        format_name: Container format (e.g., "mp4", "mpegts", "avi")

    Returns:
        BytesIO buffer containing the synthetic video

    """
    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format=format_name)
    fps = 30

    # Use mpeg4 encoder (built-in to FFmpeg, no external codec dependency).
    # These tests exercise remuxing (container format changes), not encoding.
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width = 32
    stream.height = 32
    stream.pix_fmt = "yuv420p"
    stream.time_base = Fraction(1, fps)

    for i in range(30):
        array = np.full((stream.height, stream.width, 3), (i * 8) % 256, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush stream
    for packet in stream.encode():
        container.mux(packet)

    container.close()
    buffer.seek(0)
    return buffer


@pytest.fixture
def synthetic_mp4_video() -> io.BytesIO:
    """Create a synthetic MP4 video in memory."""
    return _make_synthetic_video("mp4")


@pytest.fixture
def synthetic_mpegts_video() -> io.BytesIO:
    """Create a synthetic MPEG-TS video in memory."""
    return _make_synthetic_video("mpegts")


@pytest.fixture
def synthetic_avi_video() -> io.BytesIO:
    """Create a synthetic AVI video in memory (unsupported format for remuxing)."""
    return _make_synthetic_video("avi")


@pytest.fixture
def synthetic_mkv_video() -> io.BytesIO:
    """Create a synthetic MKV video in memory."""
    return _make_synthetic_video("matroska")


@pytest.mark.env("cosmos-curator")
def test_remux_bad_encoded_data() -> None:
    """Test that remuxing fails if encoded_data is not set."""
    with pytest.raises(RuntimeError):
        remux_to_mp4(b"", threads=1)


@pytest.mark.env("cosmos-curator")
def test_remux_if_needed_mp4_no_change(synthetic_mp4_video: io.BytesIO) -> None:
    """Test that MP4 videos are not remuxed."""
    # Arrange
    original_data = bytes_to_numpy(synthetic_mp4_video.getvalue())
    video = Video(encoded_data=original_data, input_video="test.mp4")
    video.populate_metadata()

    # Act
    result = remux_if_needed(video, threads=1)

    # Assert
    assert result is False  # mp4 does not need remux
    assert np.array_equal(video.encoded_data.resolve(), original_data)  # Should be unchanged


@pytest.mark.env("cosmos-curator")
def test_remux_if_needed_mpegts_to_mp4(synthetic_mpegts_video: io.BytesIO) -> None:
    """Test that MPEG-TS videos are remuxed to MP4."""
    # Arrange
    original_data = bytes_to_numpy(synthetic_mpegts_video.getvalue())
    video = Video(encoded_data=original_data, input_video="test.ts")
    video.populate_metadata()

    # Verify it starts as MPEG-TS
    assert video.metadata is not None
    assert video.metadata.format_name is not None
    assert video.metadata.format_name == "mpegts"

    # Act
    result = remux_if_needed(video, threads=1)

    # Assert
    assert result is True  # mpegts was remuxed
    assert not np.array_equal(video.encoded_data.resolve(), original_data)  # Should be changed
    assert video.encoded_data.resolve() is not None

    # Verify the result is valid MP4
    resolved = video.encoded_data.resolve()
    assert resolved is not None
    result_stream = io.BytesIO(resolved.tobytes())
    with av.open(result_stream) as container:
        assert "mp4" in video.metadata.format_name.lower()
        assert "mp4" in container.format.name.lower()
        assert len(container.streams.video) > 0


@pytest.mark.env("cosmos-curator")
def test_remux_if_needed_avi_to_mp4(synthetic_avi_video: io.BytesIO) -> None:
    """Test that AVI videos are unchanged."""
    # Arrange
    original_data = bytes_to_numpy(synthetic_avi_video.getvalue())
    video = Video(encoded_data=original_data, input_video="test.avi")
    video.populate_metadata()

    # Verify it starts as AVI
    assert video.metadata is not None
    assert video.metadata.format_name is not None
    assert "avi" in video.metadata.format_name.lower()

    # Act
    result = remux_if_needed(video, threads=1)

    # Assert
    assert result is False  # avi is not in REMUX_FORMATS
    assert np.array_equal(video.encoded_data.resolve(), original_data)  # Should be unchanged

    # Verify the result is valid AVI
    resolved = video.encoded_data.resolve()
    assert resolved is not None
    result_stream = io.BytesIO(resolved.tobytes())
    with av.open(result_stream) as container:
        assert "avi" in container.format.name.lower()
        assert len(container.streams.video) > 0


@pytest.mark.env("cosmos-curator")
def test_remux_if_needed_mkv_to_mp4(synthetic_mkv_video: io.BytesIO) -> None:
    """Test that MKV videos are unchanged."""
    # Arrange
    encoded_data = bytes_to_numpy(synthetic_mkv_video.getvalue())
    video = Video(encoded_data=encoded_data, input_video="test.mkv")
    video.populate_metadata()

    # Verify it starts as MKV
    assert video.metadata is not None
    assert video.metadata.format_name is not None
    assert "matroska,webm" in video.metadata.format_name.lower()

    # Act
    result = remux_if_needed(video, threads=1)
    assert result is False  # mkv is not in REMUX_FORMATS

    # Assert
    assert np.array_equal(video.encoded_data.resolve(), encoded_data)  # Should be unchanged
    assert video.encoded_data.resolve() is not None

    # Verify the result remains MKV (unchanged)
    resolved = video.encoded_data.resolve()
    assert resolved is not None
    result_stream = io.BytesIO(resolved.tobytes())
    with av.open(result_stream) as container:
        assert "matroska,webm" in container.format.name.lower()
        assert len(container.streams.video) > 0


@pytest.mark.env("cosmos-curator")
def test_remux_if_needed_preserves_video_content(synthetic_mpegts_video: io.BytesIO) -> None:
    """Test that remuxing preserves the actual video content."""

    def _frame_count(data: bytes) -> int:
        with av.open(io.BytesIO(data)) as container:
            in_container = cast("av.container.InputContainer", container)
            return sum(1 for _ in in_container.decode(video=0))
        msg = "Shouldn't get here, adding to make mypy happy"
        raise RuntimeError(msg)

    # Arrange
    original_bytes = synthetic_mpegts_video.getvalue()
    video = Video(encoded_data=bytes_to_numpy(original_bytes), input_video="test.ts")
    video.populate_metadata()

    # Get original frame count
    original_stream = io.BytesIO(original_bytes)
    with av.open(original_stream) as container:
        original_width = container.streams.video[0].width
        original_height = container.streams.video[0].height

    original_frame_count = _frame_count(original_bytes)

    # Act
    result = remux_if_needed(video, threads=1)

    # Assert
    assert result is True  # mpegts was remuxed
    resolved = video.encoded_data.resolve()
    assert resolved is not None

    result_frame_count = _frame_count(resolved.tobytes())
    assert result_frame_count == original_frame_count

    # Verify the result is valid MP4
    result_stream = io.BytesIO(resolved.tobytes())
    with av.open(result_stream) as container:
        result_width = container.streams.video[0].width
        result_height = container.streams.video[0].height

        # Video properties should be preserved
        assert result_frame_count == original_frame_count
        assert result_width == original_width
        assert result_height == original_height


def test_remux_if_needed_invalidates_timestamps(monkeypatch: pytest.MonkeyPatch) -> None:
    """remux_if_needed() sets video.timestamps = None after replacing encoded_data."""
    remuxed_data = bytes_to_numpy(b"remuxed")
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.read_write.remux_stages.remux_to_mp4",
        lambda _data, **_kw: remuxed_data,
    )

    video = Video(
        input_video="test.ts",
        encoded_data=bytes_to_numpy(b"original"),
        timestamps=np.array([0.0, 0.033, 0.066], dtype=np.float32),
    )
    monkeypatch.setattr(Video, "populate_metadata", lambda _: None)
    video.metadata = MagicMock()
    video.metadata.format_name = "mpegts"
    video.metadata.__bool__ = lambda _: True

    result = remux_if_needed(video, threads=1)

    assert result is True
    assert video.timestamps is None
