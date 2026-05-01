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
"""Tests for AlignedFrame."""

from fractions import Fraction
from typing import Any, cast

import attrs
import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.sensors.data.aligned_frame import AlignedFrame
from cosmos_curator.core.sensors.data.camera_data import CameraData
from cosmos_curator.core.sensors.data.sensor_data import SensorData
from cosmos_curator.core.sensors.data.video import VideoMetadata


@attrs.define
class _FakeSensorData:
    """Minimal SensorData implementation for AlignedFrame invariant tests."""

    align_timestamps_ns: npt.NDArray[np.int64]
    sensor_timestamps_ns: npt.NDArray[np.int64]


def _make_camera_data() -> CameraData:
    """Build a minimal CameraData instance for AlignedFrame tests."""
    metadata = VideoMetadata(
        codec_name="h264",
        codec_max_bframes=0,
        codec_profile="Main",
        container_format="mp4",
        height=1,
        width=1,
        avg_frame_rate=Fraction(30, 1),
        pix_fmt="yuv420p",
        bit_rate_bps=1,
    )
    timestamps = np.array([1], dtype=np.int64)
    frames = np.zeros((1, 1, 1, 3), dtype=np.uint8)
    return CameraData(
        align_timestamps_ns=timestamps,
        sensor_timestamps_ns=timestamps.copy(),
        pts_stream=timestamps.copy(),
        frames=frames,
        metadata=metadata,
    )


def _make_fake_sensor_data(*, timestamps_len: int, sensor_timestamps_len: int) -> SensorData:
    """Build a minimal SensorData with independently controlled lengths."""
    return _FakeSensorData(
        align_timestamps_ns=np.arange(timestamps_len, dtype=np.int64),
        sensor_timestamps_ns=np.arange(sensor_timestamps_len, dtype=np.int64),
    )


def test_aligned_frame_sensor_data_is_immutable() -> None:
    """AlignedFrame should not expose a mutable sensor_data mapping."""
    frame = AlignedFrame(
        align_timestamps_ns=np.array([1], dtype=np.int64),
        sensor_data={"cam0": cast("SensorData", _make_camera_data())},
    )

    with pytest.raises(TypeError):
        cast("Any", frame.sensor_data)["cam1"] = _make_camera_data()


def test_aligned_frame_contains_and_getitem() -> None:
    """AlignedFrame should support membership checks and keyed access."""
    camera_data = cast("SensorData", _make_camera_data())
    frame = AlignedFrame(
        align_timestamps_ns=np.array([1], dtype=np.int64),
        sensor_data={"cam0": camera_data},
    )

    assert "cam0" in frame
    assert "missing" not in frame
    assert frame["cam0"] is camera_data


def test_aligned_frame_getitem_missing_key_raises_keyerror() -> None:
    """AlignedFrame should raise KeyError for missing sensor ids."""
    frame = AlignedFrame(
        align_timestamps_ns=np.array([1], dtype=np.int64),
        sensor_data={"cam0": cast("SensorData", _make_camera_data())},
    )

    with pytest.raises(KeyError, match="missing"):
        _ = frame["missing"]


def test_aligned_frame_raises_on_align_timestamps_length_mismatch() -> None:
    """AlignedFrame should reject sensor payloads with the wrong batch length."""
    with pytest.raises(ValueError, match=r"sensor 'cam0' align_timestamps_ns length 1 != aligned frame length 2"):
        AlignedFrame(
            align_timestamps_ns=np.array([1, 2], dtype=np.int64),
            sensor_data={"cam0": _make_fake_sensor_data(timestamps_len=1, sensor_timestamps_len=1)},
        )


def test_aligned_frame_raises_on_sensor_timestamp_length_mismatch() -> None:
    """AlignedFrame should reject sensor payloads whose sensor timestamp length mismatches."""
    sensor_data = _FakeSensorData(
        align_timestamps_ns=np.array([1, 2], dtype=np.int64),
        sensor_timestamps_ns=np.array([0], dtype=np.int64),
    )

    with pytest.raises(
        ValueError,
        match=r"sensor 'cam0' sensor_timestamps_ns length 1 != aligned frame length 2",
    ):
        AlignedFrame(
            align_timestamps_ns=np.array([1, 2], dtype=np.int64),
            sensor_data={"cam0": cast("SensorData", sensor_data)},
        )


def test_aligned_frame_raises_when_sensor_reference_timestamps_do_not_match() -> None:
    """AlignedFrame should reject sensor payloads sampled on the wrong reference timestamps."""
    sensor_data = _FakeSensorData(
        align_timestamps_ns=np.array([10, 30], dtype=np.int64),
        sensor_timestamps_ns=np.array([11, 31], dtype=np.int64),
    )

    with pytest.raises(
        ValueError,
        match=r"sensor 'cam0' align_timestamps_ns must exactly match aligned frame align_timestamps_ns",
    ):
        AlignedFrame(
            align_timestamps_ns=np.array([10, 20], dtype=np.int64),
            sensor_data={"cam0": cast("SensorData", sensor_data)},
        )


def test_aligned_frame_timestamps_are_readonly() -> None:
    """AlignedFrame should expose a read-only reference timestamp array."""
    frame = AlignedFrame(
        align_timestamps_ns=np.array([1], dtype=np.int64),
        sensor_data={"cam0": cast("SensorData", _make_camera_data())},
    )

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        frame.align_timestamps_ns[0] = 2


def test_aligned_frame_does_not_mutate_caller_owned_timestamps() -> None:
    """AlignedFrame should wrap the caller's array without changing its writeability."""
    align_timestamps_ns = np.array([1], dtype=np.int64)

    frame = AlignedFrame(
        align_timestamps_ns=align_timestamps_ns,
        sensor_data={"cam0": cast("SensorData", _make_camera_data())},
    )

    assert align_timestamps_ns.flags.writeable is True
    assert frame.align_timestamps_ns.flags.writeable is False
    assert frame.align_timestamps_ns is not align_timestamps_ns
    assert np.shares_memory(frame.align_timestamps_ns, align_timestamps_ns)


@pytest.mark.parametrize(
    ("align_timestamps_ns", "match"),
    [
        (np.array([[1]], dtype=np.int64), r"align_timestamps_ns must be 1-D"),
        (
            np.array([2, 1], dtype=np.int64),
            r"align_timestamps_ns must be strictly sorted in ascending order with no duplicates",
        ),
        (
            np.array([1, 1], dtype=np.int64),
            r"align_timestamps_ns must be strictly sorted in ascending order with no duplicates",
        ),
    ],
)
def test_aligned_frame_rejects_invalid_reference_timeline(
    align_timestamps_ns: npt.NDArray[np.int64],
    match: str,
) -> None:
    """AlignedFrame should require a 1-D strictly increasing reference timeline."""
    with pytest.raises(ValueError, match=match):
        AlignedFrame(
            align_timestamps_ns=align_timestamps_ns,
            sensor_data={"cam0": cast("SensorData", _make_camera_data())},
        )
