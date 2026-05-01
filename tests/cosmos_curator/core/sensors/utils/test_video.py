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
"""Test video utilities for the sensor library."""

import io
from contextlib import AbstractContextManager, nullcontext
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import av
import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.sensors.data.video import VIDEO_METADATA_VERSION, VideoIndex, VideoMetadata
from cosmos_curator.core.sensors.types.types import VideoIndexCreationMethod
from cosmos_curator.core.sensors.utils.io import open_file
from cosmos_curator.core.sensors.utils.video import (
    CpuVideoDecodeConfig,
    CpuVideoDecoder,
    GpuVideoDecodeConfig,
    GpuVideoDecoder,
    _get_video_index_from_header,
    _HeaderIndexUnavailableError,
    make_decode_plan,
    make_index_and_metadata,
    open_video_container,
    pts_to_ns,
)

# synthetic_video fixture configuration
SYNTHETIC_VIDEO_NUM_FRAMES = 10
SYNTHETIC_VIDEO_FPS = 30


@pytest.fixture
def synthetic_video() -> io.BytesIO:
    """Create a synthetic test video in memory.

    Creates a video with SYNTHETIC_VIDEO_NUM_FRAMES frames at SYNTHETIC_VIDEO_FPS fps.
    """
    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    stream = container.add_stream("h264", rate=SYNTHETIC_VIDEO_FPS)
    stream.width = 16
    stream.height = 16
    stream.pix_fmt = "yuv420p"

    for i in range(SYNTHETIC_VIDEO_NUM_FRAMES):
        array = np.full((stream.height, stream.width, 3), i * 10, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode(None):
        container.mux(packet)

    container.close()
    buffer.seek(0)
    return buffer


def test_video_metadata_string_dict_round_trip() -> None:
    """VideoMetadata should round-trip through the string-only wire format."""
    metadata = VideoMetadata(
        codec_name="h264",
        codec_max_bframes=2,
        codec_profile="High",
        container_format="mp4",
        height=1080,
        width=1920,
        avg_frame_rate=Fraction(30000, 1001),
        pix_fmt="yuv420p",
        bit_rate_bps=4_500_000,
    )

    payload = metadata.to_string_dict()

    assert payload["avg_frame_rate_numerator"] == "30000"
    assert payload["avg_frame_rate_denominator"] == "1001"
    assert payload["version"] == VIDEO_METADATA_VERSION
    assert VideoMetadata.from_string_dict(payload) == metadata


def test_video_metadata_from_string_dict_raises_on_missing_keys() -> None:
    """VideoMetadata deserialization should fail if required keys are absent."""
    with pytest.raises(ValueError, match="missing required keys"):
        VideoMetadata.from_string_dict({"version": VIDEO_METADATA_VERSION, "codec_name": "h264"})


def test_video_metadata_from_string_dict_raises_on_version_mismatch() -> None:
    """VideoMetadata deserialization should reject unsupported payload versions."""
    metadata = VideoMetadata(
        codec_name="h264",
        codec_max_bframes=0,
        codec_profile="Main",
        container_format="mp4",
        height=16,
        width=16,
        avg_frame_rate=Fraction(30, 1),
        pix_fmt="yuv420p",
        bit_rate_bps=1234,
    )

    payload = metadata.to_string_dict()
    payload["version"] = "2"

    with pytest.raises(ValueError, match="unsupported VideoMetadata payload version"):
        VideoMetadata.from_string_dict(payload)


def test_video_metadata_from_string_dict_raises_on_invalid_fraction() -> None:
    """VideoMetadata deserialization should reject invalid frame-rate fractions."""
    metadata = VideoMetadata(
        codec_name="h264",
        codec_max_bframes=0,
        codec_profile="Main",
        container_format="mp4",
        height=16,
        width=16,
        avg_frame_rate=Fraction(30, 1),
        pix_fmt="yuv420p",
        bit_rate_bps=1234,
    )
    payload = metadata.to_string_dict()
    payload["avg_frame_rate_denominator"] = "0"

    with pytest.raises(ValueError, match="invalid VideoMetadata payload"):
        VideoMetadata.from_string_dict(payload)


def test_make_index_and_metadata_zero_denominator_average_rate_defaults_to_zero() -> None:
    """A truthy average_rate with zero denominator should not raise during metadata construction."""

    class _TruthyAverageRate:
        numerator = 30
        denominator = 0

        def __bool__(self) -> bool:
            return True

    average_rate = _TruthyAverageRate()
    video_stream = SimpleNamespace(
        time_base=Fraction(1, 30),
        average_rate=average_rate,
        width=16,
        height=16,
        codec_context=SimpleNamespace(
            name="h264",
            max_b_frames=0,
            profile="Main",
            pix_fmt="yuv420p",
        ),
    )
    container = SimpleNamespace(format=SimpleNamespace(name="mp4"))
    demux_result = (
        [0, 10],
        [100, 100],
        [0, 1],
        [True, False],
        [False, False],
    )

    with (
        patch(
            "cosmos_curator.core.sensors.utils.video.open_data_source",
            return_value=nullcontext(io.BytesIO(b"")),
        ),
        patch(
            "cosmos_curator.core.sensors.utils.video.open_video_container",
            return_value=nullcontext((container, video_stream)),
        ),
        patch(
            "cosmos_curator.core.sensors.utils.video._get_video_index_full_demux",
            return_value=demux_result,
        ),
    ):
        _index, metadata = make_index_and_metadata(b"", index_method=VideoIndexCreationMethod.FULL_DEMUX)

    assert metadata.avg_frame_rate == Fraction(0)


def test_make_index_and_metadata_raises_when_time_base_is_none() -> None:
    """make_index_and_metadata should reject streams without a time base."""
    video_stream = SimpleNamespace(time_base=None)
    container = SimpleNamespace(format=SimpleNamespace(name="mp4"))

    with (
        patch(
            "cosmos_curator.core.sensors.utils.video.open_data_source",
            return_value=nullcontext(io.BytesIO(b"")),
        ),
        patch(
            "cosmos_curator.core.sensors.utils.video.open_video_container",
            return_value=nullcontext((container, video_stream)),
        ),
        pytest.raises(ValueError, match="Time base is None for video stream 0"),
    ):
        make_index_and_metadata(b"", index_method=VideoIndexCreationMethod.FULL_DEMUX)


def test_make_index_and_metadata_raises_when_no_packets_exist() -> None:
    """make_index_and_metadata should reject streams with no valid packets."""
    video_stream = SimpleNamespace(
        time_base=Fraction(1, 30),
        average_rate=Fraction(30, 1),
        width=16,
        height=16,
        codec_context=SimpleNamespace(name="h264", max_b_frames=0, profile="Main", pix_fmt="yuv420p"),
    )
    container = SimpleNamespace(format=SimpleNamespace(name="mp4"))

    with (
        patch(
            "cosmos_curator.core.sensors.utils.video.open_data_source",
            return_value=nullcontext(io.BytesIO(b"")),
        ),
        patch(
            "cosmos_curator.core.sensors.utils.video.open_video_container",
            return_value=nullcontext((container, video_stream)),
        ),
        patch(
            "cosmos_curator.core.sensors.utils.video._get_video_index_full_demux",
            return_value=([], [], [], [], []),
        ),
        pytest.raises(ValueError, match="contains no packets with valid PTS"),
    ):
        make_index_and_metadata(b"", index_method=VideoIndexCreationMethod.FULL_DEMUX)


def test_make_index_and_metadata_raises_when_no_keyframes_exist() -> None:
    """make_index_and_metadata should reject streams without keyframes."""
    video_stream = SimpleNamespace(
        time_base=Fraction(1, 30),
        average_rate=Fraction(30, 1),
        width=16,
        height=16,
        codec_context=SimpleNamespace(name="h264", max_b_frames=0, profile="Main", pix_fmt="yuv420p"),
    )
    container = SimpleNamespace(format=SimpleNamespace(name="mp4"))
    demux_result = (
        [0, 10],
        [100, 100],
        [0, 1],
        [False, False],
        [False, False],
    )

    with (
        patch(
            "cosmos_curator.core.sensors.utils.video.open_data_source",
            return_value=nullcontext(io.BytesIO(b"")),
        ),
        patch(
            "cosmos_curator.core.sensors.utils.video.open_video_container",
            return_value=nullcontext((container, video_stream)),
        ),
        patch(
            "cosmos_curator.core.sensors.utils.video._get_video_index_full_demux",
            return_value=demux_result,
        ),
        pytest.raises(ValueError, match="contains no keyframes"),
    ):
        make_index_and_metadata(b"", index_method=VideoIndexCreationMethod.FULL_DEMUX)


def test_make_index_and_metadata_from_header_uses_header_entries() -> None:
    """make_index_and_metadata should use header index entries when available."""
    entries = [
        SimpleNamespace(timestamp=30, pos=200, size=20, is_keyframe=False, is_discard=True),
        SimpleNamespace(timestamp=0, pos=100, size=10, is_keyframe=True, is_discard=False),
    ]
    video_stream = SimpleNamespace(
        time_base=Fraction(1, 30),
        average_rate=Fraction(30, 1),
        width=16,
        height=16,
        index_entries=entries,
        codec_context=SimpleNamespace(name="h264", max_b_frames=0, profile="Main", pix_fmt="yuv420p"),
    )
    container = SimpleNamespace(format=SimpleNamespace(name="mp4"))

    with (
        patch(
            "cosmos_curator.core.sensors.utils.video.open_data_source",
            return_value=nullcontext(io.BytesIO(b"")),
        ),
        patch(
            "cosmos_curator.core.sensors.utils.video.open_video_container",
            return_value=nullcontext((container, video_stream)),
        ),
        patch("cosmos_curator.core.sensors.utils.video._get_video_index_full_demux") as full_demux,
    ):
        index, metadata = make_index_and_metadata(b"", index_method=VideoIndexCreationMethod.FROM_HEADER)

    full_demux.assert_not_called()
    np.testing.assert_array_equal(index.pts_stream, np.array([0, 30], dtype=np.int64))
    np.testing.assert_array_equal(index.offset, np.array([100, 200], dtype=np.int64))
    np.testing.assert_array_equal(index.is_keyframe, np.array([True, False], dtype=np.bool_))
    np.testing.assert_array_equal(index.is_discard, np.array([False, True], dtype=np.bool_))
    assert metadata.container_format == "mp4"


def test_make_index_and_metadata_from_header_falls_back_to_full_demux() -> None:
    """make_index_and_metadata should fall back to full demux when header indexing is unavailable."""
    video_stream = SimpleNamespace(
        time_base=Fraction(1, 30),
        average_rate=Fraction(30, 1),
        width=16,
        height=16,
        codec_context=SimpleNamespace(name="h264", max_b_frames=0, profile="Main", pix_fmt="yuv420p"),
    )
    container = SimpleNamespace(format=SimpleNamespace(name="mp4"))
    demux_result = (
        [0, 10],
        [100, 100],
        [0, 1],
        [True, False],
        [False, False],
    )

    with (
        patch(
            "cosmos_curator.core.sensors.utils.video.open_data_source",
            return_value=nullcontext(io.BytesIO(b"")),
        ),
        patch(
            "cosmos_curator.core.sensors.utils.video.open_video_container",
            return_value=nullcontext((container, video_stream)),
        ),
        patch(
            "cosmos_curator.core.sensors.utils.video._get_video_index_from_header",
            side_effect=_HeaderIndexUnavailableError("retry with FULL_DEMUX"),
        ),
        patch(
            "cosmos_curator.core.sensors.utils.video._get_video_index_full_demux",
            return_value=demux_result,
        ) as full_demux,
    ):
        index, _metadata = make_index_and_metadata(b"", index_method=VideoIndexCreationMethod.FROM_HEADER)

    full_demux.assert_called_once()
    np.testing.assert_array_equal(index.pts_stream, np.array([0, 1], dtype=np.int64))


def test_make_index_and_metadata_from_header_can_disable_fallback() -> None:
    """Diagnostics should be able to fail if the header index is unavailable."""
    video_stream = SimpleNamespace(
        time_base=Fraction(1, 30),
        average_rate=Fraction(30, 1),
        width=16,
        height=16,
        codec_context=SimpleNamespace(name="h264", max_b_frames=0, profile="Main", pix_fmt="yuv420p"),
    )
    container = SimpleNamespace(format=SimpleNamespace(name="mp4"))

    with (
        patch(
            "cosmos_curator.core.sensors.utils.video.open_data_source",
            return_value=nullcontext(io.BytesIO(b"")),
        ),
        patch(
            "cosmos_curator.core.sensors.utils.video.open_video_container",
            return_value=nullcontext((container, video_stream)),
        ),
        patch(
            "cosmos_curator.core.sensors.utils.video._get_video_index_from_header",
            side_effect=_HeaderIndexUnavailableError("retry with FULL_DEMUX"),
        ),
        patch("cosmos_curator.core.sensors.utils.video._get_video_index_full_demux") as full_demux,
        pytest.raises(_HeaderIndexUnavailableError, match="retry with FULL_DEMUX"),
    ):
        make_index_and_metadata(
            b"",
            index_method=VideoIndexCreationMethod.FROM_HEADER,
            allow_header_fallback=False,
        )

    full_demux.assert_not_called()


def test_make_index_and_metadata_forwards_client_params_to_open_data_source() -> None:
    """Video indexing should follow the sensor package client_params pattern for remote sources."""
    expected_client_params = {"transport_params": {"client": object()}}
    captured: dict[str, object] = {}
    video_stream = SimpleNamespace(
        time_base=Fraction(1, 30),
        average_rate=Fraction(30, 1),
        width=16,
        height=16,
        codec_context=SimpleNamespace(name="h264", max_b_frames=0, profile="Main", pix_fmt="yuv420p"),
    )
    container = SimpleNamespace(format=SimpleNamespace(name="mp4"))
    demux_result = (
        [0, 10],
        [100, 100],
        [0, 1],
        [True, False],
        [False, False],
    )

    def fake_open_data_source(
        data: object,
        mode: str = "rb",
        client_params: dict[str, object] | None = None,
    ) -> object:
        captured["data"] = data
        captured["mode"] = mode
        captured["client_params"] = client_params
        return nullcontext(io.BytesIO(b""))

    with (
        patch(
            "cosmos_curator.core.sensors.utils.video.open_data_source",
            side_effect=fake_open_data_source,
        ),
        patch(
            "cosmos_curator.core.sensors.utils.video.open_video_container",
            return_value=nullcontext((container, video_stream)),
        ),
        patch(
            "cosmos_curator.core.sensors.utils.video._get_video_index_full_demux",
            return_value=demux_result,
        ),
    ):
        make_index_and_metadata(
            "s3://bucket/video.mp4",
            index_method=VideoIndexCreationMethod.FULL_DEMUX,
            client_params=expected_client_params,
        )

    assert captured["data"] == "s3://bucket/video.mp4"
    assert captured["mode"] == "rb"
    assert captured["client_params"] is expected_client_params


def test_make_index_and_metadata_raises_on_unsupported_index_method() -> None:
    """make_index_and_metadata should reject unsupported index methods explicitly."""
    video_stream = SimpleNamespace(time_base=Fraction(1, 30))
    container = SimpleNamespace(format=SimpleNamespace(name="mp4"))

    with (
        patch(
            "cosmos_curator.core.sensors.utils.video.open_data_source",
            return_value=nullcontext(io.BytesIO(b"")),
        ),
        patch(
            "cosmos_curator.core.sensors.utils.video.open_video_container",
            return_value=nullcontext((container, video_stream)),
        ),
        pytest.raises(ValueError, match="unsupported index_method"),
    ):
        make_index_and_metadata(b"", index_method="bogus")  # type: ignore[arg-type]


def test_video_index_eq_returns_false_for_different_lengths() -> None:
    """VideoIndex equality should return False, not raise, for mismatched array lengths."""
    shorter = VideoIndex(
        offset=np.array([0, 10], dtype=np.int64),
        size=np.array([100, 100], dtype=np.int64),
        pts_ns=np.array([0, 1], dtype=np.int64),
        pts_stream=np.array([0, 1], dtype=np.int64),
        is_keyframe=np.array([True, False], dtype=np.bool_),
        is_discard=np.array([False, False], dtype=np.bool_),
        kf_pts_ns=np.array([0], dtype=np.int64),
        kf_pts_stream=np.array([0], dtype=np.int64),
        time_base=Fraction(1, 30),
    )
    longer = VideoIndex(
        offset=np.array([0, 10, 20], dtype=np.int64),
        size=np.array([100, 100, 100], dtype=np.int64),
        pts_ns=np.array([0, 1, 2], dtype=np.int64),
        pts_stream=np.array([0, 1, 2], dtype=np.int64),
        is_keyframe=np.array([True, False, False], dtype=np.bool_),
        is_discard=np.array([False, False, False], dtype=np.bool_),
        kf_pts_ns=np.array([0], dtype=np.int64),
        kf_pts_stream=np.array([0], dtype=np.int64),
        time_base=Fraction(1, 30),
    )

    assert (shorter == longer) is False


def test_video_index_eq_returns_false_for_non_video_index() -> None:
    """VideoIndex equality should return False for unrelated objects."""
    index = VideoIndex(
        offset=np.array([0, 10], dtype=np.int64),
        size=np.array([100, 100], dtype=np.int64),
        pts_ns=np.array([0, 1], dtype=np.int64),
        pts_stream=np.array([0, 1], dtype=np.int64),
        is_keyframe=np.array([True, False], dtype=np.bool_),
        is_discard=np.array([False, False], dtype=np.bool_),
        kf_pts_ns=np.array([0], dtype=np.int64),
        kf_pts_stream=np.array([0], dtype=np.int64),
        time_base=Fraction(1, 30),
    )

    assert (index == object()) is False


def test_video_index_eq_returns_true_for_identical_indexes() -> None:
    """VideoIndex equality should return True when all stored fields match exactly."""
    lhs = VideoIndex(
        offset=np.array([0, 10], dtype=np.int64),
        size=np.array([100, 100], dtype=np.int64),
        pts_ns=np.array([0, 1], dtype=np.int64),
        pts_stream=np.array([0, 1], dtype=np.int64),
        is_keyframe=np.array([True, False], dtype=np.bool_),
        is_discard=np.array([False, False], dtype=np.bool_),
        kf_pts_ns=np.array([0], dtype=np.int64),
        kf_pts_stream=np.array([0], dtype=np.int64),
        time_base=Fraction(1, 30),
    )
    rhs = VideoIndex(
        offset=np.array([0, 10], dtype=np.int64),
        size=np.array([100, 100], dtype=np.int64),
        pts_ns=np.array([0, 1], dtype=np.int64),
        pts_stream=np.array([0, 1], dtype=np.int64),
        is_keyframe=np.array([True, False], dtype=np.bool_),
        is_discard=np.array([False, False], dtype=np.bool_),
        kf_pts_ns=np.array([0], dtype=np.int64),
        kf_pts_stream=np.array([0], dtype=np.int64),
        time_base=Fraction(1, 30),
    )

    assert (lhs == rhs) is True


def test_video_index_eq_returns_false_for_different_keyframe_lengths() -> None:
    """VideoIndex equality should return False when packet arrays match but keyframe arrays differ in length."""
    one_keyframe = VideoIndex(
        offset=np.array([0, 10, 20], dtype=np.int64),
        size=np.array([100, 100, 100], dtype=np.int64),
        pts_ns=np.array([0, 1, 2], dtype=np.int64),
        pts_stream=np.array([0, 1, 2], dtype=np.int64),
        is_keyframe=np.array([True, False, False], dtype=np.bool_),
        is_discard=np.array([False, False, False], dtype=np.bool_),
        kf_pts_ns=np.array([0], dtype=np.int64),
        kf_pts_stream=np.array([0], dtype=np.int64),
        time_base=Fraction(1, 30),
    )
    two_keyframes = VideoIndex(
        offset=np.array([0, 10, 20], dtype=np.int64),
        size=np.array([100, 100, 100], dtype=np.int64),
        pts_ns=np.array([0, 1, 2], dtype=np.int64),
        pts_stream=np.array([0, 1, 2], dtype=np.int64),
        is_keyframe=np.array([True, False, True], dtype=np.bool_),
        is_discard=np.array([False, False, False], dtype=np.bool_),
        kf_pts_ns=np.array([0, 2], dtype=np.int64),
        kf_pts_stream=np.array([0, 2], dtype=np.int64),
        time_base=Fraction(1, 30),
    )

    assert (one_keyframe == two_keyframes) is False


def test_video_index_raises_on_mismatched_main_array_lengths() -> None:
    """VideoIndex should reject mismatched packet-array lengths before any deeper validation."""
    with pytest.raises(ValueError, match="All arrays must be the same length"):
        VideoIndex(
            offset=np.array([0, 10], dtype=np.int64),
            size=np.array([100], dtype=np.int64),
            pts_ns=np.array([0, 1], dtype=np.int64),
            pts_stream=np.array([0, 1], dtype=np.int64),
            is_keyframe=np.array([True, False], dtype=np.bool_),
            is_discard=np.array([False, False], dtype=np.bool_),
            kf_pts_ns=np.array([0], dtype=np.int64),
            kf_pts_stream=np.array([0], dtype=np.int64),
            time_base=Fraction(1, 30),
        )


def test_video_index_raises_on_mismatched_keyframe_array_lengths() -> None:
    """VideoIndex should reject kf_pts_ns / kf_pts_stream arrays with different lengths."""
    with pytest.raises(ValueError, match="kf_pts_ns and kf_pts_stream must have equal length"):
        VideoIndex(
            offset=np.array([0, 10], dtype=np.int64),
            size=np.array([100, 100], dtype=np.int64),
            pts_ns=np.array([0, 1], dtype=np.int64),
            pts_stream=np.array([0, 1], dtype=np.int64),
            is_keyframe=np.array([True, False], dtype=np.bool_),
            is_discard=np.array([False, False], dtype=np.bool_),
            kf_pts_ns=np.array([0], dtype=np.int64),
            kf_pts_stream=np.array([0, 1], dtype=np.int64),
            time_base=Fraction(1, 30),
        )


def test_video_index_raises_on_keyframe_count_mismatch() -> None:
    """VideoIndex should reject kf arrays whose length does not match is_keyframe count."""
    with pytest.raises(ValueError, match="kf_pts_ns length must equal number of keyframes in is_keyframe"):
        VideoIndex(
            offset=np.array([0, 10, 20], dtype=np.int64),
            size=np.array([100, 100, 100], dtype=np.int64),
            pts_ns=np.array([0, 1, 2], dtype=np.int64),
            pts_stream=np.array([0, 1, 2], dtype=np.int64),
            is_keyframe=np.array([True, False, False], dtype=np.bool_),
            is_discard=np.array([False, False, False], dtype=np.bool_),
            kf_pts_ns=np.array([0, 2], dtype=np.int64),
            kf_pts_stream=np.array([0, 2], dtype=np.int64),
            time_base=Fraction(1, 30),
        )


def test_video_index_raises_on_kf_pts_ns_value_mismatch() -> None:
    """VideoIndex should reject kf_pts_ns values that do not match pts_ns at keyframe positions."""
    with pytest.raises(ValueError, match=r"kf_pts_ns must equal pts_ns\[is_keyframe\]"):
        VideoIndex(
            offset=np.array([0, 10, 20], dtype=np.int64),
            size=np.array([100, 100, 100], dtype=np.int64),
            pts_ns=np.array([100, 200, 300], dtype=np.int64),
            pts_stream=np.array([10, 20, 30], dtype=np.int64),
            is_keyframe=np.array([True, False, True], dtype=np.bool_),
            is_discard=np.array([False, False, False], dtype=np.bool_),
            kf_pts_ns=np.array([101, 301], dtype=np.int64),
            kf_pts_stream=np.array([10, 30], dtype=np.int64),
            time_base=Fraction(1, 30),
        )


def test_video_index_raises_on_kf_pts_stream_value_mismatch() -> None:
    """VideoIndex should reject kf_pts_stream values that do not match pts_stream at keyframe positions."""
    with pytest.raises(ValueError, match=r"kf_pts_stream must equal pts_stream\[is_keyframe\]"):
        VideoIndex(
            offset=np.array([0, 10, 20], dtype=np.int64),
            size=np.array([100, 100, 100], dtype=np.int64),
            pts_ns=np.array([100, 200, 300], dtype=np.int64),
            pts_stream=np.array([10, 20, 30], dtype=np.int64),
            is_keyframe=np.array([True, False, True], dtype=np.bool_),
            is_discard=np.array([False, False, False], dtype=np.bool_),
            kf_pts_ns=np.array([100, 300], dtype=np.int64),
            kf_pts_stream=np.array([11, 31], dtype=np.int64),
            time_base=Fraction(1, 30),
        )


def test_video_index_raises_on_unsorted_pts_ns() -> None:
    """VideoIndex should reject non-increasing pts_ns arrays."""
    with pytest.raises(ValueError, match="pts_ns must be strictly sorted in ascending order with no duplicates"):
        VideoIndex(
            offset=np.array([0, 10, 20], dtype=np.int64),
            size=np.array([100, 100, 100], dtype=np.int64),
            pts_ns=np.array([0, 2, 1], dtype=np.int64),
            pts_stream=np.array([0, 1, 2], dtype=np.int64),
            is_keyframe=np.array([True, False, False], dtype=np.bool_),
            is_discard=np.array([False, False, False], dtype=np.bool_),
            kf_pts_ns=np.array([0], dtype=np.int64),
            kf_pts_stream=np.array([0], dtype=np.int64),
            time_base=Fraction(1, 30),
        )


def test_video_index_raises_on_unsorted_pts_stream() -> None:
    """VideoIndex should reject non-increasing pts_stream arrays."""
    with pytest.raises(ValueError, match="pts_stream must be strictly sorted in ascending order with no duplicates"):
        VideoIndex(
            offset=np.array([0, 10, 20], dtype=np.int64),
            size=np.array([100, 100, 100], dtype=np.int64),
            pts_ns=np.array([0, 1, 2], dtype=np.int64),
            pts_stream=np.array([0, 2, 1], dtype=np.int64),
            is_keyframe=np.array([True, False, False], dtype=np.bool_),
            is_discard=np.array([False, False, False], dtype=np.bool_),
            kf_pts_ns=np.array([0], dtype=np.int64),
            kf_pts_stream=np.array([0], dtype=np.int64),
            time_base=Fraction(1, 30),
        )


def test_video_index_raises_on_unsorted_keyframe_timestamps() -> None:
    """VideoIndex should reject non-increasing keyframe timestamp arrays."""
    with pytest.raises(ValueError, match="kf_pts_ns must be strictly sorted in ascending order with no duplicates"):
        VideoIndex(
            offset=np.array([0, 10, 20], dtype=np.int64),
            size=np.array([100, 100, 100], dtype=np.int64),
            pts_ns=np.array([0, 1, 2], dtype=np.int64),
            pts_stream=np.array([0, 1, 2], dtype=np.int64),
            is_keyframe=np.array([True, False, True], dtype=np.bool_),
            is_discard=np.array([False, False, False], dtype=np.bool_),
            kf_pts_ns=np.array([2, 0], dtype=np.int64),
            kf_pts_stream=np.array([0, 2], dtype=np.int64),
            time_base=Fraction(1, 30),
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("offset", np.array([[0, 10]], dtype=np.int64), r"offset must be 1-D"),
        ("pts_ns", np.array([0, 1], dtype=np.int32), r"pts_ns must have dtype int64"),
        ("pts_stream", np.array([0, 1], dtype=np.int32), r"pts_stream must have dtype int64"),
        ("is_keyframe", np.array([1, 0], dtype=np.int64), r"is_keyframe must have dtype bool"),
        ("is_discard", np.array([0, 1], dtype=np.int64), r"is_discard must have dtype bool"),
        ("pts_ns", np.array([[0, 1]], dtype=np.int64), r"pts_ns must be 1-D"),
    ],
)
def test_video_index_rejects_invalid_array_dtype_or_ndim(
    field: str,
    value: npt.NDArray[Any],
    match: str,
) -> None:
    """VideoIndex should validate array dtypes and dimensionality at the API boundary."""
    kwargs: dict[str, Any] = {
        "offset": np.array([0, 10], dtype=np.int64),
        "size": np.array([100, 100], dtype=np.int64),
        "pts_ns": np.array([0, 1], dtype=np.int64),
        "pts_stream": np.array([0, 1], dtype=np.int64),
        "is_keyframe": np.array([True, False], dtype=np.bool_),
        "is_discard": np.array([False, False], dtype=np.bool_),
        "kf_pts_ns": np.array([0], dtype=np.int64),
        "kf_pts_stream": np.array([0], dtype=np.int64),
        "time_base": Fraction(1, 30),
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=match):
        VideoIndex(**kwargs)


def test_video_index_display_view_filters_discard_packets() -> None:
    """VideoIndex should expose a reusable display-frame view."""
    index = VideoIndex(
        offset=np.array([0, 10, 20], dtype=np.int64),
        size=np.array([100, 100, 100], dtype=np.int64),
        pts_ns=np.array([100, 200, 300], dtype=np.int64),
        pts_stream=np.array([10, 20, 30], dtype=np.int64),
        is_keyframe=np.array([True, False, False], dtype=np.bool_),
        is_discard=np.array([False, True, False], dtype=np.bool_),
        kf_pts_ns=np.array([100], dtype=np.int64),
        kf_pts_stream=np.array([10], dtype=np.int64),
        time_base=Fraction(1, 30),
    )

    display_mask_0 = index.display_mask
    display_pts_ns_0 = index.display_pts_ns
    display_pts_stream_0 = index.display_pts_stream

    np.testing.assert_array_equal(display_mask_0, np.array([True, False, True], dtype=np.bool_))
    np.testing.assert_array_equal(display_pts_ns_0, np.array([100, 300], dtype=np.int64))
    np.testing.assert_array_equal(display_pts_stream_0, np.array([10, 30], dtype=np.int64))
    assert not display_mask_0.flags.writeable
    assert not display_pts_ns_0.flags.writeable
    assert not display_pts_stream_0.flags.writeable

    assert index.display_mask is display_mask_0
    assert index.display_pts_ns is display_pts_ns_0
    assert index.display_pts_stream is display_pts_stream_0


def test_video_index_does_not_mutate_caller_owned_arrays() -> None:
    """VideoIndex should keep caller-owned arrays writeable while exposing read-only views."""
    offset = np.array([0, 10, 20], dtype=np.int64)
    size = np.array([100, 100, 100], dtype=np.int64)
    pts_ns = np.array([100, 200, 300], dtype=np.int64)
    pts_stream = np.array([10, 20, 30], dtype=np.int64)
    is_keyframe = np.array([True, False, False], dtype=np.bool_)
    is_discard = np.array([False, True, False], dtype=np.bool_)
    kf_pts_ns = np.array([100], dtype=np.int64)
    kf_pts_stream = np.array([10], dtype=np.int64)

    index = VideoIndex(
        offset=offset,
        size=size,
        pts_ns=pts_ns,
        pts_stream=pts_stream,
        is_keyframe=is_keyframe,
        is_discard=is_discard,
        kf_pts_ns=kf_pts_ns,
        kf_pts_stream=kf_pts_stream,
        time_base=Fraction(1, 30),
    )

    assert offset.flags.writeable is True
    assert size.flags.writeable is True
    assert pts_ns.flags.writeable is True
    assert pts_stream.flags.writeable is True
    assert is_keyframe.flags.writeable is True
    assert is_discard.flags.writeable is True
    assert kf_pts_ns.flags.writeable is True
    assert kf_pts_stream.flags.writeable is True
    assert index.offset.flags.writeable is False
    assert index.size.flags.writeable is False
    assert index.pts_ns.flags.writeable is False
    assert index.pts_stream.flags.writeable is False
    assert index.is_keyframe.flags.writeable is False
    assert index.is_discard.flags.writeable is False
    assert index.kf_pts_ns.flags.writeable is False
    assert index.kf_pts_stream.flags.writeable is False
    assert index.offset is not offset
    assert index.size is not size
    assert index.pts_ns is not pts_ns
    assert index.pts_stream is not pts_stream
    assert index.is_keyframe is not is_keyframe
    assert index.is_discard is not is_discard
    assert index.kf_pts_ns is not kf_pts_ns
    assert index.kf_pts_stream is not kf_pts_stream
    assert np.shares_memory(index.offset, offset)
    assert np.shares_memory(index.size, size)
    assert np.shares_memory(index.pts_ns, pts_ns)
    assert np.shares_memory(index.pts_stream, pts_stream)
    assert np.shares_memory(index.is_keyframe, is_keyframe)
    assert np.shares_memory(index.is_discard, is_discard)
    assert np.shares_memory(index.kf_pts_ns, kf_pts_ns)
    assert np.shares_memory(index.kf_pts_stream, kf_pts_stream)


@pytest.mark.parametrize(("height", "width"), [(0, 1), (1, 0), (-1, 1), (1, -1)])
def test_video_metadata_raises_on_non_positive_dimensions(height: int, width: int) -> None:
    """VideoMetadata should reject zero or negative frame dimensions."""
    with pytest.raises(ValueError, match="must be > 0"):
        VideoMetadata(
            codec_name="h264",
            codec_max_bframes=0,
            codec_profile="Main",
            container_format="mp4",
            height=height,
            width=width,
            avg_frame_rate=Fraction(30, 1),
            pix_fmt="yuv420p",
            bit_rate_bps=1_000,
        )


@pytest.mark.parametrize(
    ("pts", "time_base", "expected"),
    [
        (0, Fraction(1, 1000), 0),
        (1000, Fraction(1, 1000), 1_000_000_000),
        (90, Fraction(1, 90), 1_000_000_000),
        (1, Fraction(1, 1_000_000), 1_000),
        (
            np.array([90, 180], dtype=np.int64),
            Fraction(1, 90),
            np.array([1_000_000_000, 2_000_000_000], dtype=np.int64),
        ),
    ],
)
def test_pts_to_ns(
    pts: int | npt.NDArray[np.int64],
    time_base: Fraction | SimpleNamespace,
    expected: int | npt.NDArray[np.int64] | None,
) -> None:
    """``pts_to_ns`` converts packet PTS to nanoseconds for scalars and arrays.

    A time base with zero denominator raises ``ValueError`` (not representable as ``Fraction``).
    """
    result = pts_to_ns(pts, time_base)  # type: ignore[arg-type]
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(result, expected)
        assert isinstance(pts, np.ndarray)
        assert result.dtype == pts.dtype  # type: ignore[attr-defined]
    else:
        assert result == expected


def test_pts_to_ns_array_no_int64_overflow() -> None:
    """pts_to_ns must not silently overflow int64 for large PTS values.

    Regression test ``pts * 1_000_000_000 * time_base.numerator`` is
    computed in numpy int64 arithmetic, which silently wraps for
    ``pts > int64_max // 1_000_000_000 ≈ 9.22e9`` when ``time_base.numerator == 1``.

    Scenario: MCAP nanosecond time base (``Fraction(1, 1_000_000_000)``).  A
    timestamp 9.3 seconds past the recording start is stored as
    ``pts = 9_300_000_000`` (ns units).  ``pts_to_ns`` must return
    ``9_300_000_000`` unchanged — the conversion is the identity function for a
    1 ns/unit time base.

    Without the fix, ``pts * 1_000_000_000 = 9.3e18`` overflows int64 (max
    ≈ 9.22e18), wrapping to the signed value ``-9_146_744_073_709_551_616``.
    After ``// 1_000_000_000`` the result is ``-9_146_744_074`` — a large
    negative number instead of the correct ``9_300_000_000``.
    """
    time_base = Fraction(1, 1_000_000_000)  # 1 nanosecond per unit (MCAP-style)
    pts = np.array([9_300_000_000], dtype=np.int64)  # 9.3 seconds in ns units
    result = pts_to_ns(pts, time_base)
    # Identity: pts_to_ns with 1 ns/unit time base must return pts unchanged.
    np.testing.assert_array_equal(result, pts)


@pytest.mark.parametrize(
    ("stream_idx", "video_format", "should_raise"),
    [
        (0, None, nullcontext()),
        (0, "mp4", nullcontext()),
        (99, None, pytest.raises(IndexError, match=r".*")),
    ],
)
def test_open_video_container(
    synthetic_video: io.BytesIO,
    stream_idx: int,
    video_format: str | None,
    should_raise: AbstractContextManager[Any],
) -> None:
    """Test open_video_container with an already-open stream."""
    with (
        should_raise,
        open_video_container(synthetic_video, stream_idx=stream_idx, video_format=video_format) as (
            container,
            video_stream,
        ),
    ):
        assert container is not None
        assert video_stream is not None

        packet_count = 0
        for _packet in container.demux(video=stream_idx):
            packet_count += 1
            if packet_count >= 3:
                break

        assert packet_count > 0, "Should have demuxed at least one packet"


def test_cpu_video_decoder_open_raises_when_time_base_is_none(synthetic_video: io.BytesIO) -> None:
    """CpuVideoDecoder.open should reject streams without a time base."""
    mock_stream = MagicMock()
    type(mock_stream).time_base = PropertyMock(return_value=None)
    mock_stream.width = 16
    mock_stream.height = 16

    mock_container = MagicMock()
    mock_container.__enter__ = MagicMock(return_value=mock_container)
    mock_container.__exit__ = MagicMock(return_value=False)
    mock_container.streams.video = [mock_stream]

    with (
        patch("av.open", return_value=mock_container),
        pytest.raises(ValueError, match=r".*Time base is None.*"),
        CpuVideoDecoder.open(synthetic_video, stream_idx=0),
    ):
        pass


def test_cpu_video_decoder_open_owns_stream_but_open_video_container_does_not(
    synthetic_video: io.BytesIO,
    tmp_path: Path,
) -> None:
    """Decoder sessions close what they open; open_video_container does not own caller streams."""
    video_path = tmp_path / "owned_stream.mp4"
    video_path.write_bytes(synthetic_video.getvalue())

    captured_stream: io.BufferedReader | None = None
    real_open_file = open_file

    def _capturing_open_file(src: object, mode: str = "rb", client_params: dict[str, object] | None = None) -> object:
        nonlocal captured_stream
        stream = real_open_file(src, mode=mode, client_params=client_params)
        if hasattr(stream, "closed"):
            captured_stream = stream  # type: ignore[assignment]
        return stream

    with (
        patch("cosmos_curator.core.sensors.utils.io.open_file", side_effect=_capturing_open_file),
        CpuVideoDecoder.open(video_path),
    ):
        assert captured_stream is not None
        assert not captured_stream.closed

    assert captured_stream is not None
    assert captured_stream.closed

    caller_owned = io.BytesIO(synthetic_video.getvalue())
    with open_video_container(caller_owned) as (_container, _video_stream):
        assert not caller_owned.closed
    assert not caller_owned.closed


def test_cpu_video_decoder_applies_thread_config(synthetic_video: io.BytesIO) -> None:
    """CpuVideoDecoder should apply configured thread settings to the opened stream."""
    video_bytes = synthetic_video.getvalue()

    with CpuVideoDecoder.open(video_bytes, config=CpuVideoDecodeConfig(thread_type="SLICE", thread_count=2)) as decoder:
        assert decoder.stream.thread_count == 2
        assert decoder.stream.thread_type.name == "SLICE"


def test_cpu_video_decode_config_defaults() -> None:
    """CpuVideoDecodeConfig should default to rgb24 output."""
    config = CpuVideoDecodeConfig()

    assert config.dest_format == "rgb24"
    assert config.dest_dtype == np.dtype(np.uint8)
    assert config.channels == 3


def test_cpu_video_decode_config_raises_on_unsupported_dest_format() -> None:
    """CpuVideoDecodeConfig should reject unsupported destination formats."""
    with pytest.raises(ValueError, match="unsupported dest_format"):
        CpuVideoDecodeConfig(dest_format="gray8")


def test_cpu_video_decode_config_raises_on_invalid_thread_type() -> None:
    """CpuVideoDecodeConfig should reject unsupported thread types."""
    with pytest.raises(ValueError, match="thread_type must be one of"):
        CpuVideoDecodeConfig(thread_type="BOGUS")


def test_cpu_video_decode_config_raises_on_negative_thread_count() -> None:
    """CpuVideoDecodeConfig should reject negative thread counts."""
    with pytest.raises(ValueError, match="thread_count must be non-negative"):
        CpuVideoDecodeConfig(thread_count=-1)


def test_cpu_video_decoder_uses_configured_output_format(synthetic_video: io.BytesIO) -> None:
    """CpuVideoDecoder should derive output tensor layout from the decode config."""
    video_bytes = synthetic_video.getvalue()
    config = CpuVideoDecodeConfig(dest_format="rgb24")

    with CpuVideoDecoder.open(video_bytes, config=config) as decoder:
        index, _metadata = make_index_and_metadata(video_bytes, index_method=VideoIndexCreationMethod.FULL_DEMUX)
        target_pts_stream = index.pts_stream[:1]
        counts = np.ones(len(target_pts_stream), dtype=np.int64)
        plan = make_decode_plan(index.kf_pts_stream, target_pts_stream, counts)
        frames = decoder.decode(plan)

    assert frames.dtype == config.dest_dtype
    assert frames.shape == (1, 16, 16, config.channels)


def test_cpu_video_decoder_raises_without_time_base() -> None:
    """CpuVideoDecoder should reject streams without a time base."""
    stream = SimpleNamespace(time_base=None)
    with pytest.raises(ValueError, match="Time base is None"):
        CpuVideoDecoder(SimpleNamespace(), stream)  # type: ignore[arg-type]


def test_cpu_video_decoder_decode_updates_stats_and_broadcasts_duplicates(synthetic_video: io.BytesIO) -> None:
    """CpuVideoDecoder should update stats and broadcast repeated targets into the destination buffer."""
    video_bytes = synthetic_video.getvalue()
    stats: dict[str, float] = {}

    with CpuVideoDecoder.open(video_bytes, stats=stats) as decoder:
        index, _metadata = make_index_and_metadata(video_bytes, index_method=VideoIndexCreationMethod.FULL_DEMUX)
        target_pts_stream = index.pts_stream[:1]
        counts = np.array([2], dtype=np.int64)
        plan = make_decode_plan(index.kf_pts_stream, target_pts_stream, counts)
        frames = decoder.decode(plan)

    assert frames.shape == (2, 16, 16, 3)
    np.testing.assert_array_equal(frames[0], frames[1])
    assert stats["frames_decoded"] >= 1
    assert stats["t_seek"] >= 0.0
    assert stats["t_convert"] >= 0.0
    assert stats["t_copy"] >= 0.0


def test_cpu_video_decoder_validate_monotonic_frame_pts_raises() -> None:
    """CpuVideoDecoder should raise when decoded PTS moves backward or repeats."""
    stream = SimpleNamespace(
        time_base=Fraction(1, 30),
        width=16,
        height=16,
        thread_type=None,
        thread_count=0,
        codec_context=SimpleNamespace(flush_buffers=lambda: None),
    )
    decoder = CpuVideoDecoder(SimpleNamespace(), stream)  # type: ignore[arg-type]
    decoder.last_decoded_pts = 5

    with pytest.raises(RuntimeError, match="Non-monotonic frame pts=5"):
        decoder._validate_monotonic_frame_pts(5, 0)


def test_cpu_video_decoder_decode_handles_eof_and_missing_targets() -> None:
    """CpuVideoDecoder should surface missing targets after an EOF decode error."""
    stream = SimpleNamespace(
        time_base=Fraction(1, 30),
        width=16,
        height=16,
        thread_type=None,
        thread_count=0,
        codec_context=SimpleNamespace(flush_buffers=lambda: None),
    )
    packet = SimpleNamespace(size=1, decode=MagicMock(side_effect=av.error.EOFError(0, "eof")))
    container = SimpleNamespace(
        seek=lambda *_args, **_kwargs: None,
        demux=lambda _stream: [packet],
    )
    decoder = CpuVideoDecoder(container, stream)  # type: ignore[arg-type]
    dest = np.empty((1, 16, 16, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="target\\(s\\) not found"):
        decoder._decode_group(dest, 0, 0, [(10, 1)])


def test_cpu_video_decoder_decode_skips_none_pts_and_non_targets() -> None:
    """CpuVideoDecoder should skip frames with missing or non-target PTS values."""
    frames = [
        SimpleNamespace(pts=None),
        SimpleNamespace(pts=5),
        SimpleNamespace(
            pts=10,
            to_ndarray=lambda **_kwargs: np.full((16, 16, 3), 7, dtype=np.uint8),
        ),
    ]
    packet = SimpleNamespace(size=0, decode=lambda: frames)
    stream = SimpleNamespace(
        time_base=Fraction(1, 30),
        width=16,
        height=16,
        thread_type=None,
        thread_count=0,
        codec_context=SimpleNamespace(flush_buffers=lambda: None),
    )
    container = SimpleNamespace(
        seek=lambda *_args, **_kwargs: None,
        demux=lambda _stream: [packet],
    )
    decoder = CpuVideoDecoder(container, stream)  # type: ignore[arg-type]
    dest = np.empty((1, 16, 16, 3), dtype=np.uint8)

    dest_idx = decoder._decode_group(dest, 0, 0, [(10, 1)])

    assert dest_idx == 1
    np.testing.assert_array_equal(dest[0], np.full((16, 16, 3), 7, dtype=np.uint8))


def test_cpu_video_decoder_decode_group_returns_when_empty_targets_and_empty_packet() -> None:
    """_decode_group should return the current write index when no targets are requested."""
    packet = SimpleNamespace(size=0, decode=list)
    stream = SimpleNamespace(
        time_base=Fraction(1, 30),
        width=16,
        height=16,
        thread_type=None,
        thread_count=0,
        codec_context=SimpleNamespace(flush_buffers=lambda: None),
    )
    container = SimpleNamespace(
        seek=lambda *_args, **_kwargs: None,
        demux=lambda _stream: [packet],
    )
    decoder = CpuVideoDecoder(container, stream)  # type: ignore[arg-type]
    dest = np.empty((0, 16, 16, 3), dtype=np.uint8)

    assert decoder._decode_group(dest, 0, 0, []) == 0


def test_cpu_video_decoder_decode_group_returns_when_empty_targets_and_no_packets() -> None:
    """_decode_group should return immediately when demux yields no packets and there are no targets."""
    stream = SimpleNamespace(
        time_base=Fraction(1, 30),
        width=16,
        height=16,
        thread_type=None,
        thread_count=0,
        codec_context=SimpleNamespace(flush_buffers=lambda: None),
    )
    container = SimpleNamespace(
        seek=lambda *_args, **_kwargs: None,
        demux=lambda _stream: [],
    )
    decoder = CpuVideoDecoder(container, stream)  # type: ignore[arg-type]
    dest = np.empty((0, 16, 16, 3), dtype=np.uint8)

    assert decoder._decode_group(dest, 0, 0, []) == 0


def test_cpu_video_decoder_decode_raises_on_internal_accounting_error(synthetic_video: io.BytesIO) -> None:
    """CpuVideoDecoder should raise if internal write accounting does not match the requested count."""
    video_bytes = synthetic_video.getvalue()

    with CpuVideoDecoder.open(video_bytes) as decoder:
        plan = [(0, [(0, 1)])]
        with (
            patch.object(decoder, "_decode_group", return_value=0),
            pytest.raises(RuntimeError, match="internal accounting error"),
        ):
            decoder.decode(plan)


def test_gpu_video_decoder_raises_without_time_base() -> None:
    """GpuVideoDecoder should reject streams without a time base."""
    stream = SimpleNamespace(time_base=None)
    with pytest.raises(ValueError, match="Time base is None"):
        GpuVideoDecoder(SimpleNamespace(), stream)  # type: ignore[arg-type]


def test_gpu_video_decoder_open_and_decode_not_implemented(synthetic_video: io.BytesIO) -> None:
    """GpuVideoDecoder should open successfully on a real stream and reject decode requests."""
    with GpuVideoDecoder.open(synthetic_video.getvalue(), config=GpuVideoDecodeConfig()) as decoder:
        assert decoder.time_base > 0
        with pytest.raises(NotImplementedError, match="GPU decode mode not implemented"):
            decoder.decode([])


def test_get_video_index_from_header_raises_without_index_entries() -> None:
    """Header indexing should raise a specific error when index_entries are unavailable."""
    stream = SimpleNamespace()

    with pytest.raises(ValueError, match="retry with FULL_DEMUX"):
        _get_video_index_from_header(stream)  # type: ignore[arg-type]


def test_get_video_index_from_header_reads_entries() -> None:
    """Header indexing should extract packet metadata from index entries."""
    entries = [
        SimpleNamespace(timestamp=1, pos=100, size=10, is_keyframe=True, is_discard=False),
        SimpleNamespace(timestamp=2, pos=200, size=20, is_keyframe=False, is_discard=True),
    ]
    stream = SimpleNamespace(index_entries=entries)

    result = _get_video_index_from_header(stream)  # type: ignore[arg-type]

    assert result == ([100, 200], [10, 20], [1, 2], [True, False], [False, True])


def test_get_video_index_from_header_raises_when_index_is_empty() -> None:
    """Header indexing should reject an empty header index."""
    stream = SimpleNamespace(index_entries=[])

    with pytest.raises(ValueError, match="stream header index is empty"):
        _get_video_index_from_header(stream)  # type: ignore[arg-type]


# ===========================================================================
# make_decode_plan
# ===========================================================================


@pytest.mark.parametrize(
    ("pts_us", "counts"),
    [
        pytest.param(
            np.array([0, 1, 2], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            id="pts_longer_than_counts",
        ),
        pytest.param(
            np.array([0, 1], dtype=np.int64),
            np.array([1, 1, 1], dtype=np.int64),
            id="counts_longer_than_pts",
        ),
    ],
)
def test_make_decode_plan_length_mismatch(
    pts_us: npt.NDArray[np.int64],
    counts: npt.NDArray[np.int64],
) -> None:
    """make_decode_plan raises ValueError when pts_us and counts differ in length."""
    kf_pts_us = np.array([0], dtype=np.int64)
    with pytest.raises(ValueError, match="same length"):
        make_decode_plan(kf_pts_us, pts_us, counts)


@pytest.mark.parametrize(
    ("kf_pts_us", "pts_us", "counts", "match"),
    [
        (
            np.array([0], dtype=np.int64),
            np.array([10], dtype=np.int64),
            np.array([0], dtype=np.int64),
            r"strictly positive",
        ),
        (
            np.array([0], dtype=np.int64),
            np.array([10], dtype=np.int64),
            np.array([-1], dtype=np.int64),
            r"strictly positive",
        ),
        (
            np.array([0], dtype=np.int64),
            np.array([10], dtype=np.int64),
            np.array([[1]], dtype=np.int64),
            r"counts must be 1-D",
        ),
        (
            np.array([[0]], dtype=np.int64),
            np.array([10], dtype=np.int64),
            np.array([1], dtype=np.int64),
            r"kf_pts_stream must be 1-D",
        ),
        (
            np.array([0], dtype=np.int64),
            np.array([[10]], dtype=np.int64),
            np.array([1], dtype=np.int64),
            r"pts_stream must be 1-D",
        ),
    ],
)
def test_make_decode_plan_rejects_invalid_counts(
    kf_pts_us: npt.NDArray[np.int64],
    pts_us: npt.NDArray[np.int64],
    counts: npt.NDArray[np.int64],
    match: str,
) -> None:
    """make_decode_plan should reject malformed public array inputs at the API boundary."""
    with pytest.raises(ValueError, match=match):
        make_decode_plan(kf_pts_us, pts_us, counts)


def test_make_decode_plan_empty_pts_us() -> None:
    """make_decode_plan returns an empty list when pts_us is empty."""
    kf_pts_us = np.array([0, 10, 20], dtype=np.int64)
    assert make_decode_plan(kf_pts_us, np.array([], dtype=np.int64), np.array([], dtype=np.int64)) == []


def test_make_decode_plan_empty_kf_pts_us() -> None:
    """make_decode_plan raises ValueError when kf_pts_us is empty."""
    with pytest.raises(ValueError, match="no keyframes"):
        make_decode_plan(
            np.array([], dtype=np.int64),
            np.array([10], dtype=np.int64),
            np.array([1], dtype=np.int64),
        )


def test_make_decode_plan_unsorted_kf_pts_us() -> None:
    """make_decode_plan raises ValueError when kf_pts_us is not sorted."""
    kf_pts_us = np.array([0, 20, 10], dtype=np.int64)
    with pytest.raises(ValueError, match="kf_pts_stream must be sorted"):
        make_decode_plan(kf_pts_us, np.array([10], dtype=np.int64), np.array([1], dtype=np.int64))


def test_make_decode_plan_unsorted_pts_us() -> None:
    """make_decode_plan raises ValueError when pts_us is not sorted."""
    kf_pts_us = np.array([0], dtype=np.int64)
    with pytest.raises(ValueError, match="pts_stream must be sorted"):
        make_decode_plan(kf_pts_us, np.array([20, 10], dtype=np.int64), np.array([1, 1], dtype=np.int64))


@pytest.mark.parametrize(
    ("kf_pts_us", "pts_us", "counts", "expected_plan"),
    [
        pytest.param(
            # All targets fall in the single GOP.
            np.array([0], dtype=np.int64),
            np.array([3, 8, 13], dtype=np.int64),
            np.array([1, 1, 1], dtype=np.int64),
            [(0, [(3, 1), (8, 1), (13, 1)])],
            id="single_gop",
        ),
        pytest.param(
            # One target per GOP — two seeks.
            np.array([0, 50], dtype=np.int64),
            np.array([20, 70], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            [(0, [(20, 1)]), (50, [(70, 1)])],
            id="two_gops_one_target_each",
        ),
        pytest.param(
            # Multiple targets per GOP — still only two seeks.
            np.array([0, 50], dtype=np.int64),
            np.array([10, 20, 30, 60, 70, 80], dtype=np.int64),
            np.array([1, 1, 1, 1, 1, 1], dtype=np.int64),
            [(0, [(10, 1), (20, 1), (30, 1)]), (50, [(60, 1), (70, 1), (80, 1)])],
            id="two_gops_multi_target",
        ),
        pytest.param(
            # 5 keyframes, only 2 GOPs targeted — 3 keyframes absent from output.
            # Mirrors sparse 2Hz sampling of a 30Hz/GOP5 stream.
            np.array([0, 100, 200, 300, 400], dtype=np.int64),
            np.array([50, 350], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            [(0, [(50, 1)]), (300, [(350, 1)])],
            id="gops_skipped",
        ),
        pytest.param(
            # Every frame is a keyframe; targets are a subset of keyframe timestamps.
            # Each target governs itself — no forward decoding needed.
            np.array([0, 10, 20, 30], dtype=np.int64),
            np.array([0, 10, 20], dtype=np.int64),
            np.array([1, 1, 1], dtype=np.int64),
            [(0, [(0, 1)]), (10, [(10, 1)]), (20, [(20, 1)])],
            id="all_keyframes",
        ),
        pytest.param(
            # Target PTS equals a keyframe PTS — maps to that keyframe, not the preceding one.
            np.array([0, 100], dtype=np.int64),
            np.array([100], dtype=np.int64),
            np.array([1], dtype=np.int64),
            [(100, [(100, 1)])],
            id="target_exactly_at_keyframe",
        ),
        pytest.param(
            # Counts > 1: a frame that maps to multiple grid points.
            np.array([0, 50], dtype=np.int64),
            np.array([20, 70], dtype=np.int64),
            np.array([3, 2], dtype=np.int64),
            [(0, [(20, 3)]), (50, [(70, 2)])],
            id="multi_count",
        ),
    ],
)
def test_make_decode_plan(
    kf_pts_us: npt.NDArray[np.int64],
    pts_us: npt.NDArray[np.int64],
    counts: npt.NDArray[np.int64],
    expected_plan: list[tuple[int, list[tuple[int, int]]]],
) -> None:
    """make_decode_plan groups targets by their governing keyframe.

    Contract verified:
    - Plan length ≤ total keyframe count (skipped GOPs absent).
    - Plan length equals number of unique governing keyframes.
    - Each entry is (int, list[tuple[int, int]]).
    - Entries are in strictly ascending keyframe order.
    - Every (pts, count) pair appears in exactly one group under the correct keyframe.
    """
    plan = make_decode_plan(kf_pts_us, pts_us, counts)

    assert len(plan) <= len(kf_pts_us)
    assert len(plan) == len(expected_plan)

    actual_kf_values = [kf for kf, _ in plan]
    assert actual_kf_values == sorted(actual_kf_values), "keyframe entries must be in ascending order"
    assert len(actual_kf_values) == len(set(actual_kf_values)), "no duplicate keyframe entries"

    for (actual_kf, actual_group), (expected_kf, expected_group) in zip(plan, expected_plan, strict=True):
        assert isinstance(actual_kf, int)
        assert isinstance(actual_group, list)
        assert actual_kf == expected_kf
        assert actual_group == expected_group

    all_plan_pts = [pts for _, group in plan for pts, _ in group]
    all_plan_counts = [cnt for _, group in plan for _, cnt in group]
    assert all_plan_pts == pts_us.tolist()
    assert all_plan_counts == counts.tolist()


@pytest.mark.parametrize(
    ("kf_pts_us", "pts_us", "counts"),
    [
        pytest.param(
            # Target before the first keyframe.
            np.array([100, 200], dtype=np.int64),
            np.array([50], dtype=np.int64),
            np.array([1], dtype=np.int64),
            id="target_before_first_keyframe",
        ),
        pytest.param(
            # Mix of valid and before-first-keyframe targets.
            np.array([100], dtype=np.int64),
            np.array([50, 150], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            id="mixed_in_and_out_of_range",
        ),
    ],
)
def test_make_decode_plan_before_first_keyframe(
    kf_pts_us: npt.NDArray[np.int64],
    pts_us: npt.NDArray[np.int64],
    counts: npt.NDArray[np.int64],
) -> None:
    """make_decode_plan raises ValueError for targets before the first keyframe."""
    with pytest.raises(ValueError, match="before the first keyframe"):
        make_decode_plan(kf_pts_us, pts_us, counts)


@pytest.mark.parametrize(
    ("kf_pts_us", "pts_us", "counts", "expected_kf"),
    [
        pytest.param(
            # Target after the last keyframe but within the last GOP — valid.
            np.array([0, 100], dtype=np.int64),
            np.array([115], dtype=np.int64),
            np.array([1], dtype=np.int64),
            100,
            id="target_in_last_gop",
        ),
        pytest.param(
            # Only one keyframe — all targets map to it.
            np.array([0], dtype=np.int64),
            np.array([5, 15, 35], dtype=np.int64),
            np.array([1, 1, 1], dtype=np.int64),
            0,
            id="single_keyframe_all_targets",
        ),
    ],
)
def test_make_decode_plan_boundary(
    kf_pts_us: npt.NDArray[np.int64],
    pts_us: npt.NDArray[np.int64],
    counts: npt.NDArray[np.int64],
    expected_kf: int,
) -> None:
    """make_decode_plan correctly governs targets at the edges of the valid range."""
    plan = make_decode_plan(kf_pts_us, pts_us, counts)
    assert len(plan) == 1
    actual_kf, actual_group = plan[0]
    assert actual_kf == expected_kf
    assert [pts for pts, _ in actual_group] == pts_us.tolist()
    assert [cnt for _, cnt in actual_group] == counts.tolist()


def test_make_decode_plan_output_ordering() -> None:
    """Keyframe entries in the plan are strictly ascending — seeks are always forward."""
    kf_pts_us = np.array([0, 50, 100, 150, 200, 250], dtype=np.int64)
    pts_us = np.array([15, 75, 125, 225], dtype=np.int64)  # one per targeted GOP, skipping 150
    counts = np.ones(len(pts_us), dtype=np.int64)

    plan = make_decode_plan(kf_pts_us, pts_us, counts)

    actual_kf_values = [kf for kf, _ in plan]
    assert actual_kf_values == sorted(actual_kf_values), "keyframe seek order must be strictly ascending"
    assert len(actual_kf_values) == len(set(actual_kf_values)), "no duplicate keyframe entries"
    assert len(plan) < len(kf_pts_us), "skipped GOPs must be absent"


def test_make_decode_plan_epoch_timestamps() -> None:
    """make_decode_plan handles epoch-scale nanosecond timestamps without overflow."""
    base = 1_700_000_000_000_000_000  # realistic epoch timestamp in ns
    step = 33_333_000  # ~30Hz frame interval in ns
    kf_interval = 5  # GOP size 5
    n_keyframes = 3

    kf_pts_us = np.array([base + i * kf_interval * step for i in range(n_keyframes)], dtype=np.int64)
    pts_us = np.array([base + 50_000_000, base + 350_000_000], dtype=np.int64)
    counts = np.array([1, 1], dtype=np.int64)

    plan = make_decode_plan(kf_pts_us, pts_us, counts)

    assert len(plan) == 2
    assert plan[0][0] == base  # first keyframe
    assert plan[1][0] == base + 2 * kf_interval * step  # third keyframe (GOP 2)
    assert plan[0][0] < plan[1][0]  # ascending

    all_plan_pts = [pts for _, group in plan for pts, _ in group]
    assert all_plan_pts == pts_us.tolist()


# ===========================================================================
# make_index_and_metadata — B-frame PTS sort
# ===========================================================================


def test_make_index_and_metadata_sorts_pts_by_presentation_order(synthetic_video: io.BytesIO) -> None:
    """make_index_and_metadata returns pts_ns sorted in ascending order.

    Patches _get_video_index_from_header to inject a non-monotonic PTS sequence
    that mirrors a real B-frame GOP (decode order: I P B B), then asserts that
    all returned arrays are reordered into ascending PTS order and that offset
    is non-monotonic (reflecting non-sequential file positions after the sort).
    """
    # B-frame decode order: I(pts=0) P(pts=3) B(pts=1) B(pts=2)
    # File offsets are monotonically increasing in decode/file order.
    # argsort([0,3,1,2]) produces [0,2,3,1], so after the PTS sort pts_ns
    # becomes ascending in time-base units and offset becomes [100,300,400,200] (non-monotonic).
    mock_pts = [0, 3, 1, 2]
    mock_offsets = [100, 200, 300, 400]
    mock_sizes = [500, 600, 700, 800]
    mock_keyframes = [True, False, False, False]
    mock_discards = [False, False, False, False]

    with patch(
        "cosmos_curator.core.sensors.utils.video._get_video_index_from_header",
        return_value=(mock_offsets, mock_sizes, mock_pts, mock_keyframes, mock_discards),
    ):
        synthetic_video.seek(0)
        index, _ = make_index_and_metadata(synthetic_video)

    # pts_ns must be monotonically non-decreasing.
    assert np.all(index.pts_ns[:-1] <= index.pts_ns[1:]), "pts_ns must be sorted ascending"

    # The I-frame (pts=0) must still be flagged as a keyframe after reordering.
    assert index.is_keyframe[0], "first frame in PTS order must be the keyframe"
    assert not index.is_keyframe[1:].any(), "only the I-frame should be a keyframe"

    # offset reflects the B-frame file layout: after sorting by PTS the file
    # positions are no longer sequential, so offset is not monotonically increasing.
    assert not np.all(index.offset[:-1] <= index.offset[1:]), "offset must be non-monotonic for B-frame video"

    # Each parallel array must be consistently reordered: size[i] belongs to pts_ns[i].
    # sort_idx for mock_pts=[0,3,1,2] is [0,2,3,1]:
    #   size:   [500,600,700,800][[0,2,3,1]] → [500,700,800,600]
    #   offset: [100,200,300,400][[0,2,3,1]] → [100,300,400,200]
    np.testing.assert_array_equal(index.size, np.array([500, 700, 800, 600], dtype=np.int64))
    np.testing.assert_array_equal(index.offset, np.array([100, 300, 400, 200], dtype=np.int64))


def test_make_index_and_metadata_falls_back_to_full_demux_when_header_index_unavailable(
    synthetic_video: io.BytesIO,
) -> None:
    """FROM_HEADER should transparently fall back to FULL_DEMUX when header indexing is unavailable."""
    mock_offsets = [100, 200]
    mock_sizes = [500, 600]
    mock_pts = [0, 1]
    mock_keyframes = [True, False]
    mock_discards = [False, False]

    with (
        patch(
            "cosmos_curator.core.sensors.utils.video._get_video_index_from_header",
            side_effect=_HeaderIndexUnavailableError(
                "stream does not expose header index entries; retry with FULL_DEMUX"
            ),
        ),
        patch(
            "cosmos_curator.core.sensors.utils.video._get_video_index_full_demux",
            return_value=(mock_offsets, mock_sizes, mock_pts, mock_keyframes, mock_discards),
        ) as mock_full_demux,
    ):
        synthetic_video.seek(0)
        index, _ = make_index_and_metadata(synthetic_video)

    mock_full_demux.assert_called_once()
    np.testing.assert_array_equal(index.offset, np.array(mock_offsets, dtype=np.int64))
    np.testing.assert_array_equal(index.size, np.array(mock_sizes, dtype=np.int64))


# ===========================================================================
# make_index_and_metadata — stream pts fields
# ===========================================================================


@pytest.fixture
def synthetic_video_fps_rate_timebase() -> io.BytesIO:
    """Synthetic video with a short GOP size to guarantee multiple keyframes.

    Used to verify that VideoIndex exposes pts_stream / kf_pts_stream fields and
    that the decode path produces no duplicate frames across GOP boundaries.
    """
    fps = 30
    gop_size = 5
    n_frames = 10  # 2 GOPs of 5

    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    stream = container.add_stream("h264", rate=fps)
    stream.width = 16
    stream.height = 16
    stream.pix_fmt = "yuv420p"
    stream.gop_size = gop_size

    for i in range(n_frames):
        array = np.full((stream.height, stream.width, 3), i * 20, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode(None):
        container.mux(packet)

    container.close()
    buffer.seek(0)
    return buffer


@pytest.fixture
def synthetic_mjpeg_all_keyframes_video() -> io.BytesIO:
    """Synthetic MJPEG video where every frame is a keyframe."""
    fps = 30
    n_frames = 6

    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="avi")
    stream = container.add_stream("mjpeg", rate=fps)
    stream.width = 16
    stream.height = 16
    stream.pix_fmt = "yuvj420p"

    for i in range(n_frames):
        array = np.full((stream.height, stream.width, 3), i * 30, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode(None):
        container.mux(packet)

    container.close()
    buffer.seek(0)
    return buffer


def test_make_index_and_metadata_pts_stream_fields(synthetic_video_fps_rate_timebase: io.BytesIO) -> None:
    """make_index_and_metadata returns pts_stream and kf_pts_stream alongside pts_ns and kf_pts_ns."""
    synthetic_video_fps_rate_timebase.seek(0)
    index, _ = make_index_and_metadata(
        synthetic_video_fps_rate_timebase.read(), index_method=VideoIndexCreationMethod.FULL_DEMUX
    )

    assert len(index.pts_stream) == len(index.pts_ns), "pts_stream and pts_ns must have the same length"
    assert len(index.kf_pts_stream) == len(index.kf_pts_ns), "kf_pts_stream and kf_pts_ns must have the same length"
    assert len(index.kf_pts_stream) == int(index.is_keyframe.sum()), "kf_pts_stream length must equal keyframe count"

    # pts_stream and kf_pts_stream must be sorted ascending (same invariant as pts_ns).
    assert np.all(index.pts_stream[:-1] <= index.pts_stream[1:]), "pts_stream must be sorted ascending"
    assert np.all(index.kf_pts_stream[:-1] <= index.kf_pts_stream[1:]), "kf_pts_stream must be sorted ascending"

    # pts_to_ns(pts_stream, time_base) must reproduce pts_ns exactly.
    np.testing.assert_array_equal(
        pts_to_ns(index.pts_stream, index.time_base),
        index.pts_ns,
        err_msg="pts_to_ns(pts_stream, time_base) must equal pts_ns",
    )


def test_cpu_video_decoder_no_duplicate_frames_fps_rate_timebase(synthetic_video_fps_rate_timebase: io.BytesIO) -> None:
    """CpuVideoDecoder decodes every frame exactly once on an fps-rate time_base stream.

    Regression test: the old floor-division seek caused FFmpeg to seek before the
    keyframe and re-decode GOP-0 frames when seeking into GOP-1 on a
    Fraction(1,30) stream.  With exact-match decoding the output frame count must
    equal the number of targets, and canonical timestamps derived from pts_stream
    must be unique (no frame decoded twice).
    """
    synthetic_video_fps_rate_timebase.seek(0)
    video_bytes = synthetic_video_fps_rate_timebase.read()

    index, _ = make_index_and_metadata(video_bytes, index_method=VideoIndexCreationMethod.FULL_DEMUX)

    counts = np.ones(len(index.pts_stream), dtype=np.int64)
    plan = make_decode_plan(index.kf_pts_stream, index.pts_stream, counts)

    with CpuVideoDecoder.open(video_bytes) as decoder:
        time_base = decoder.time_base
        frames = decoder.decode(plan)

    assert frames.shape[0] == len(index.pts_stream), f"expected {len(index.pts_stream)} frames, got {frames.shape[0]}"
    canonical_ts = pts_to_ns(index.pts_stream, time_base)
    assert len(canonical_ts) == len(np.unique(canonical_ts)), f"duplicate canonical timestamps: {canonical_ts.tolist()}"


def test_cpu_video_decoder_all_keyframes_exact_targets(
    synthetic_mjpeg_all_keyframes_video: io.BytesIO,
) -> None:
    """CpuVideoDecoder should decode exact target frames on an all-keyframe MJPEG stream."""
    synthetic_mjpeg_all_keyframes_video.seek(0)
    video_bytes = synthetic_mjpeg_all_keyframes_video.read()

    index, _ = make_index_and_metadata(video_bytes, index_method=VideoIndexCreationMethod.FULL_DEMUX)
    assert np.all(index.is_keyframe), "MJPEG fixture should mark every frame as a keyframe"

    target_indices = np.array([0, 2, 4], dtype=np.int64)
    target_pts_stream = index.pts_stream[target_indices]
    counts = np.ones(len(target_pts_stream), dtype=np.int64)
    plan = make_decode_plan(index.kf_pts_stream, target_pts_stream, counts)

    expected_by_pts: dict[int, npt.NDArray[np.uint8]] = {}
    with io.BytesIO(video_bytes) as raw_stream, open_video_container(raw_stream) as (container, stream):
        for packet in container.demux(stream):
            try:
                frames_in_packet = list(packet.decode())
            except av.error.EOFError:
                break
            for frame in frames_in_packet:
                if frame.pts is None:
                    continue
                frame_rgb = frame.to_ndarray(format="rgb24")
                expected_by_pts[int(frame.pts)] = np.asarray(frame_rgb, dtype=np.uint8)

    with CpuVideoDecoder.open(video_bytes) as decoder:
        frames = decoder.decode(plan)

    assert frames.shape[0] == len(target_pts_stream)
    for i, pts in enumerate(target_pts_stream):
        np.testing.assert_array_equal(frames[i], expected_by_pts[int(pts)])
