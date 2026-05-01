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

"""Test utilities for video processing functionality."""

import io
from contextlib import AbstractContextManager, nullcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest
from numpy.typing import NDArray

from cosmos_curator.pipelines.video.utils.decoder_utils import (
    _make_video_stream,
    decode_video_cpu,
    find_closest_indices,
    get_avg_frame_rate,
    get_frame_count,
    sample_closest,
    save_stream_position,
)


@pytest.mark.parametrize(
    ("src", "dst", "expected"),
    [
        (
            np.array([0, 1, 2, 3, 4], dtype=np.float32),
            np.array([0, 1, 2, 3, 4], dtype=np.float32),
            np.array([0, 1, 2, 3, 4], dtype=np.int32),
        ),
        (
            np.array([0, 1, 2, 3, 4], dtype=np.float32),
            np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float32),
            np.array([0, 1, 2, 3, 4], dtype=np.int32),
        ),
        (
            np.array([0, 1, 2, 3, 4], dtype=np.float32),
            np.array([-0.5, 0.5, 1.5, 2.5, 3.5], dtype=np.float32),
            np.array([0, 0, 1, 2, 3], dtype=np.int32),
        ),
        (
            np.array([0, 1, 2, 3, 4], dtype=np.float32),
            np.array([0.6, 2.6, 5.6], dtype=np.float32),
            np.array([1, 3, 4], dtype=np.int32),
        ),
    ],
)
def test_find_closest_indices(src: NDArray[np.float32], dst: NDArray[np.float32], expected: NDArray[np.int32]) -> None:
    """Test that find_closest_indices correctly identifies the closest indices between source and destination arrays."""
    result = find_closest_indices(src, dst)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    (
        "src",
        "sample_rate",
        "start",
        "stop",
        "endpoint",
        "expected_indices",
        "expected_counts",
        "dedup",
    ),
    [
        # 1 Hz source sampled at 1 Hz
        (
            np.array(list(range(5)), dtype=np.float32),
            1.0,  # 1 Hz
            None,  # start at beginning
            None,  # end at last frame
            True,  # endpoint=True
            np.array([0, 1, 2, 3, 4], dtype=np.int64),  # frames at 0.0, 0.1, 0.2, 0.3, 0.4s
            np.array([1, 1, 1, 1, 1], dtype=np.int64),  # each frame sampled once
            True,  # dedup=True
        ),
        # 30 hz source sampled at 10Hz
        (
            np.array([i / 30.0 for i in range(10)], dtype=np.float32),  # 0.0 to 0.3s
            10.0,  # 10Hz
            None,  # start at beginning
            None,  # end at last frame
            True,  # endpoint=True
            np.array([0, 3, 6, 9], dtype=np.int64),  # frames at 0.0, 0.1, 0.2, 0.3s
            np.array([1, 1, 1, 1], dtype=np.int64),  # each frame sampled once
            True,  # dedup=True
        ),
        # 30 hz source sampled at 10Hz with explicit start/end times
        # ensure that the last timestamp is included
        (
            np.array([i / 30.0 for i in range(10)], dtype=np.float32),
            10.0,
            0.1,  # start at 0.1s, frame_id 3
            0.2,  # end at 0.2s, frame_id 6
            True,  # endpoint=True
            np.array([3, 6], dtype=np.int64),  # frames at 0.1, 0.2s
            np.array([1, 1], dtype=np.int64),
            True,  # dedup=True
        ),
        # 30 hz source sampled at 10Hz with explicit start/end times
        # ensure that the last timestamp is not included
        (
            np.array([i / 30.0 for i in range(10)], dtype=np.float32),
            10.0,
            0.1,  # start at 0.1s, frame_id 3
            0.2,  # end at 0.2s, frame_id 6
            False,  # endpoint=False
            np.array([3], dtype=np.int64),  # frames at 0.1, 0.2s
            np.array([1], dtype=np.int64),
            True,  # dedup=True
        ),
        # Source with missing timestamps, sampled at 5Hz
        (
            np.array([0.0, 0.1, 0.2, 0.4, 0.5, 0.6], dtype=np.float32),  # missing ts at 0.3s
            5.0,  # 5Hz
            None,
            None,
            True,  # endpoint=True
            np.array([0, 2, 3, 5], dtype=np.int64),  # frames at 0.0, 0.2, 0.4s
            np.array([1, 1, 1, 1], dtype=np.int64),
            True,  # dedup=True
        ),
        # Supersample test
        (
            np.array(list(range(10)), dtype=np.float32),
            2.0,
            None,
            None,
            False,  # endpoint=False
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64),
            np.array([2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.int64),
            True,  # dedup=True
        ),
        # Supersample, no dedup
        (
            np.array(list(range(10)), dtype=np.float32),
            2.0,
            None,
            None,
            False,  # endpoint=False
            np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8], dtype=np.int64),
            np.array([1 for _ in range(18)], dtype=np.int64),
            False,  # dedup=False
        ),
        # Supersample, no dedup, endpoint=True
        (
            np.array(list(range(10)), dtype=np.float32),
            2.0,
            None,
            None,
            True,  # endpoint=True
            np.array(
                [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9],
                dtype=np.int64,
            ),
            np.array([1 for _ in range(19)], dtype=np.int64),
            False,  # dedup=False
        ),
    ],
)
def test_sample_closest(  # noqa: PLR0913
    src: NDArray[np.float32],
    sample_rate: float,
    start: float | None,
    stop: float | None,
    endpoint: bool,  # noqa: FBT001
    expected_indices: NDArray[np.int64],
    expected_counts: NDArray[np.int64],
    dedup: bool,  # noqa: FBT001
) -> None:
    """Test successful cases of sample_timestamps."""
    indices, counts, _ = sample_closest(
        src=src,
        sample_rate=sample_rate,
        start=start,
        stop=stop,
        endpoint=endpoint,
        dedup=dedup,
    )

    assert isinstance(indices, np.ndarray)
    assert isinstance(counts, np.ndarray)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(counts, expected_counts)


@pytest.fixture
def synthetic_video() -> io.BytesIO:
    """Create a synthetic test video in memory."""
    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    fps = 30
    # Set up video stream
    stream = container.add_stream(
        "h264",
        rate=fps,
        options={
            "crf": "0",  # Lossless quality
            "preset": "veryslow",  # Best compression
        },
    )
    stream.width = 4
    stream.height = 4
    stream.pix_fmt = "yuv420p"
    stream.time_base = Fraction(1, fps)

    # Create 10 frames
    for i in range(10):
        array = np.full((stream.width, stream.height, 3), i * 10, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        packet = stream.encode(frame)
        if packet:
            container.mux(packet)

    # Flush stream
    packet = stream.encode(None)
    if packet:
        container.mux(packet)

    container.close()
    buffer.seek(0)
    return buffer


@pytest.mark.parametrize(("sample_rate_divisor", "endpoint"), [(1, True), (2, False)])
def test_decode_cpu(synthetic_video: io.BytesIO, sample_rate_divisor: int, endpoint: bool) -> None:  # noqa: FBT001
    """Test video decoding with different sample rates and endpoint configurations."""
    with save_stream_position(synthetic_video), av.open(synthetic_video) as container:
        average_rate = container.streams.video[0].average_rate
        assert average_rate is not None
        if average_rate is None:
            pytest.fail("average_rate is None")
        sample_rate_fps = float(average_rate) / sample_rate_divisor
        num_frames = container.streams.video[0].frames // sample_rate_divisor

    frames = decode_video_cpu(synthetic_video, sample_rate_fps=sample_rate_fps, endpoint=endpoint)

    assert len(frames) == num_frames

    expected = np.array([i * 10 * sample_rate_divisor for i in range(num_frames)], dtype=np.uint8)

    values = (
        frames.sum(axis=3).sum(axis=2).sum(axis=1) / (frames.shape[1] * frames.shape[2] * frames.shape[3])
    ).astype(np.uint8)

    assert np.isclose(values, expected, atol=2).all()


@pytest.mark.parametrize(
    ("input_data", "expected_type", "raises"),
    [
        # Happy path cases - Path and str now return str (path passthrough to av.open)
        (
            Path("dummy.mp4"),
            str,
            nullcontext(),
        ),  # Path input -> str
        (
            "dummy.mp4",
            str,
            nullcontext(),
        ),  # string input -> str
        (b"video data", io.BytesIO, nullcontext()),  # bytes input
        # Error cases
        (
            123,
            None,
            pytest.raises(ValueError),  # noqa: PT011, Integer input should raise ValueError
        ),
        (
            [],
            None,
            pytest.raises(ValueError),  # noqa: PT011, List input should raise ValueError
        ),
    ],
)
def test_make_video_stream(
    input_data: Path | str | bytes | io.BytesIO | io.BufferedReader | int | list[Any],
    expected_type: type | tuple[type, ...] | None,
    raises: AbstractContextManager[Any],
    tmp_path: Path,
) -> None:
    """Test the _make_video_stream function with various input types.

    Args:
        input_data: The input data to test
        expected_type: The expected type of the returned stream
        raises: Either nullcontext() for success cases or pytest.raises() for error cases
        tmp_path: Pytest fixture providing a temporary directory

    """
    if isinstance(input_data, (Path, str)):
        # Create a temporary file for Path test case
        test_file = tmp_path / "dummy.mp4"
        test_file.write_bytes(b"test data")
        # Cast input data back to the the original type so that
        # make_video_stream is properly exercised
        input_data = input_data.__class__(test_file)

    with raises:
        result = _make_video_stream(input_data)
        if expected_type is not None:
            assert isinstance(result, expected_type)
        if isinstance(result, str):
            assert Path(result).exists()
        else:
            assert result.readable()
            data = result.read(1)
            assert len(data) == 1
            result.seek(0)
            result.close()


@pytest.mark.parametrize(
    ("initial_pos", "read_bytes"),
    [
        (0, 5),  # Read from start
        (10, 5),  # Read from middle
        (0, 0),  # No read
        (5, 10),  # Read past end
    ],
)
def test_save_stream_position(initial_pos: int, read_bytes: int) -> None:
    """Test that save_stream_position correctly saves and restores stream position.

    Args:
        initial_pos: Initial position to seek to
        read_bytes: Number of bytes to read inside context

    """
    stream = io.BytesIO(b"0123456789" * 10)
    stream.seek(initial_pos)

    with save_stream_position(stream):
        stream.read(read_bytes)
        assert stream.tell() == initial_pos + read_bytes

    assert stream.tell() == initial_pos


def test_get_avg_frame_rate(synthetic_video: io.BytesIO) -> None:
    """Test that get_avg_frame_rate correctly returns the frame rate of a video.

    Args:
        synthetic_video: Fixture providing a test video with 30 fps

    """
    EXPECTED_FPS = 30.0
    fps = get_avg_frame_rate(synthetic_video)
    assert fps == EXPECTED_FPS


@pytest.mark.parametrize(
    ("stream_idx", "video_format"),
    [(0, None), (0, "mp4")],
)
def test_get_frame_count(synthetic_video: io.BytesIO, stream_idx: int, video_format: str | None) -> None:
    """Test that get_frame_count correctly returns the number of frames in a video.

    Args:
        synthetic_video: Fixture providing a test video with 10 frames
        stream_idx: Index of the video stream to read from
        video_format: Format of the video stream, like "mp4", "mkv", etc.

    """
    EXPECTED_FRAME_COUNT = 10
    frame_count = get_frame_count(synthetic_video, stream_idx=stream_idx, video_format=video_format)
    assert frame_count == EXPECTED_FRAME_COUNT
