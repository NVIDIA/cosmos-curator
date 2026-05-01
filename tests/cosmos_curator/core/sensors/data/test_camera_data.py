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
"""Tests for CameraData and MotionVectorData."""

from fractions import Fraction
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.sensors.data.camera_data import CameraData, MotionVectorData
from cosmos_curator.core.sensors.data.extrinsics import SensorExtrinsics
from cosmos_curator.core.sensors.data.intrinsics import CameraIntrinsics
from cosmos_curator.core.sensors.data.video import VideoMetadata


def _make_metadata() -> VideoMetadata:
    """Build minimal VideoMetadata for CameraData tests."""
    return VideoMetadata(
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


def _make_camera_data() -> CameraData:
    """Build a minimal CameraData instance."""
    timestamps = np.array([1], dtype=np.int64)
    frames = np.zeros((1, 1, 1, 3), dtype=np.uint8)
    return CameraData(
        align_timestamps_ns=timestamps,
        sensor_timestamps_ns=timestamps.copy(),
        pts_stream=timestamps.copy(),
        frames=frames,
        metadata=_make_metadata(),
    )


def _make_extrinsics() -> SensorExtrinsics:
    """Build a minimal SensorExtrinsics instance."""
    return SensorExtrinsics(matrix=np.eye(4, dtype=np.float64))


def _make_intrinsics(*, width: int = 1, height: int = 1) -> CameraIntrinsics:
    """Build a minimal CameraIntrinsics instance."""
    return CameraIntrinsics(
        camera_matrix=np.eye(3, dtype=np.float64),
        distortion_coefficients=np.zeros(5, dtype=np.float64),
        distortion_model="brown_conrady",
        width=width,
        height=height,
    )


def test_camera_data_raises_on_motion_vector_length_mismatch() -> None:
    """CameraData should reject motion vector lists whose length differs from the frame count."""
    timestamps = np.array([1], dtype=np.int64)
    frames = np.zeros((1, 1, 1, 3), dtype=np.uint8)
    motion_vectors = MotionVectorData(
        frames=(
            np.zeros((0, 10), dtype=np.float64),
            np.zeros((0, 10), dtype=np.float64),
        ),
    )

    with pytest.raises(ValueError, match=r"motion_vectors\.frames length 2 != frames length 1"):
        CameraData(
            align_timestamps_ns=timestamps,
            sensor_timestamps_ns=timestamps.copy(),
            pts_stream=timestamps.copy(),
            frames=frames,
            metadata=_make_metadata(),
            motion_vectors=motion_vectors,
        )


def test_camera_data_raises_on_array_length_mismatch() -> None:
    """CameraData should reject payloads whose arrays disagree on batch length."""
    with pytest.raises(ValueError, match=r"All arrays must be the same length"):
        CameraData(
            align_timestamps_ns=np.array([1, 2], dtype=np.int64),
            sensor_timestamps_ns=np.array([1], dtype=np.int64),
            pts_stream=np.array([1], dtype=np.int64),
            frames=np.zeros((1, 1, 1, 3), dtype=np.uint8),
            metadata=_make_metadata(),
        )


def test_camera_data_arrays_are_readonly() -> None:
    """CameraData should expose read-only top-level numpy arrays."""
    camera_data = _make_camera_data()

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        camera_data.align_timestamps_ns[0] = 2

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        camera_data.sensor_timestamps_ns[0] = 2

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        camera_data.pts_stream[0] = 2

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        camera_data.frames[0, 0, 0, 0] = 1


def test_camera_data_allows_repeated_canonical_and_pts_stream_values() -> None:
    """Repeated sensor timestamps and pts_stream values should still form a valid CameraData batch."""
    camera_data = CameraData(
        align_timestamps_ns=np.array([100, 200, 300], dtype=np.int64),
        sensor_timestamps_ns=np.array([110, 110, 310], dtype=np.int64),
        pts_stream=np.array([10, 10, 30], dtype=np.int64),
        frames=np.zeros((3, 1, 1, 3), dtype=np.uint8),
        metadata=_make_metadata(),
    )

    np.testing.assert_array_equal(camera_data.sensor_timestamps_ns, np.array([110, 110, 310], dtype=np.int64))
    np.testing.assert_array_equal(camera_data.pts_stream, np.array([10, 10, 30], dtype=np.int64))


def test_motion_vector_data_frames_are_immutable() -> None:
    """MotionVectorData should not expose a mutable top-level frame collection."""
    motion_vectors = MotionVectorData(
        frames=(np.zeros((0, 10), dtype=np.float64),),
    )

    with pytest.raises(AttributeError):
        cast("Any", motion_vectors.frames).append(np.zeros((0, 10), dtype=np.float64))


def test_motion_vector_data_accepts_valid_frames_and_marks_them_readonly() -> None:
    """MotionVectorData should accept valid block tables and freeze each frame array."""
    frame0 = np.zeros((1, 10), dtype=np.float64)
    frame1 = np.ones((2, 10), dtype=np.float64)

    motion_vectors = MotionVectorData(frames=(frame0, frame1))

    assert len(motion_vectors.frames) == 2
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        motion_vectors.frames[0][0, 0] = 1.0


def test_motion_vector_data_does_not_mutate_caller_owned_frames() -> None:
    """MotionVectorData should keep caller-owned frame tables writeable."""
    frame0 = np.zeros((1, 10), dtype=np.float64)
    frame1 = np.ones((2, 10), dtype=np.float64)

    motion_vectors = MotionVectorData(frames=(frame0, frame1))

    assert frame0.flags.writeable is True
    assert frame1.flags.writeable is True
    assert motion_vectors.frames[0].flags.writeable is False
    assert motion_vectors.frames[1].flags.writeable is False
    assert motion_vectors.frames[0] is not frame0
    assert motion_vectors.frames[1] is not frame1
    assert np.shares_memory(motion_vectors.frames[0], frame0)
    assert np.shares_memory(motion_vectors.frames[1], frame1)


def test_camera_data_accepts_matching_motion_vectors() -> None:
    """CameraData should accept motion vectors whose frame count matches the RGB frame count."""
    motion_vectors = MotionVectorData(
        frames=(
            np.zeros((1, 10), dtype=np.float64),
            np.ones((2, 10), dtype=np.float64),
        ),
    )

    camera_data = CameraData(
        align_timestamps_ns=np.array([1, 2], dtype=np.int64),
        sensor_timestamps_ns=np.array([1, 2], dtype=np.int64),
        pts_stream=np.array([1, 2], dtype=np.int64),
        frames=np.zeros((2, 1, 1, 3), dtype=np.uint8),
        metadata=_make_metadata(),
        motion_vectors=motion_vectors,
    )

    assert camera_data.motion_vectors is motion_vectors


def test_camera_data_does_not_mutate_caller_owned_arrays() -> None:
    """CameraData should keep caller-owned arrays writeable while exposing read-only views."""
    align_timestamps_ns = np.array([1, 2], dtype=np.int64)
    sensor_timestamps_ns = np.array([1, 2], dtype=np.int64)
    pts_stream = np.array([1, 2], dtype=np.int64)
    frames = np.zeros((2, 1, 1, 3), dtype=np.uint8)

    camera_data = CameraData(
        align_timestamps_ns=align_timestamps_ns,
        sensor_timestamps_ns=sensor_timestamps_ns,
        pts_stream=pts_stream,
        frames=frames,
        metadata=_make_metadata(),
    )

    assert align_timestamps_ns.flags.writeable is True
    assert sensor_timestamps_ns.flags.writeable is True
    assert pts_stream.flags.writeable is True
    assert frames.flags.writeable is True
    assert camera_data.align_timestamps_ns.flags.writeable is False
    assert camera_data.sensor_timestamps_ns.flags.writeable is False
    assert camera_data.pts_stream.flags.writeable is False
    assert camera_data.frames.flags.writeable is False
    assert camera_data.align_timestamps_ns is not align_timestamps_ns
    assert camera_data.sensor_timestamps_ns is not sensor_timestamps_ns
    assert camera_data.pts_stream is not pts_stream
    assert camera_data.frames is not frames
    assert np.shares_memory(camera_data.align_timestamps_ns, align_timestamps_ns)
    assert np.shares_memory(camera_data.sensor_timestamps_ns, sensor_timestamps_ns)
    assert np.shares_memory(camera_data.pts_stream, pts_stream)
    assert np.shares_memory(camera_data.frames, frames)


def test_camera_data_defaults_extrinsics_to_none() -> None:
    """CameraData should keep extrinsics optional for existing callers."""
    camera_data = _make_camera_data()

    assert camera_data.extrinsics is None


def test_camera_data_preserves_provided_extrinsics() -> None:
    """CameraData should store an explicitly provided SensorExtrinsics object unchanged."""
    extrinsics = _make_extrinsics()

    camera_data = CameraData(
        align_timestamps_ns=np.array([1], dtype=np.int64),
        sensor_timestamps_ns=np.array([1], dtype=np.int64),
        pts_stream=np.array([1], dtype=np.int64),
        frames=np.zeros((1, 1, 1, 3), dtype=np.uint8),
        metadata=_make_metadata(),
        extrinsics=extrinsics,
    )

    assert camera_data.extrinsics is extrinsics


def test_camera_data_defaults_intrinsics_to_none() -> None:
    """CameraData should keep intrinsics optional for existing callers."""
    camera_data = _make_camera_data()

    assert camera_data.intrinsics is None


def test_camera_data_preserves_provided_intrinsics() -> None:
    """CameraData should store an explicitly provided CameraIntrinsics object unchanged."""
    intrinsics = _make_intrinsics()

    camera_data = CameraData(
        align_timestamps_ns=np.array([1], dtype=np.int64),
        sensor_timestamps_ns=np.array([1], dtype=np.int64),
        pts_stream=np.array([1], dtype=np.int64),
        frames=np.zeros((1, 1, 1, 3), dtype=np.uint8),
        metadata=_make_metadata(),
        intrinsics=intrinsics,
    )

    assert camera_data.intrinsics is intrinsics


def test_camera_data_raises_when_intrinsics_dimensions_do_not_match_metadata() -> None:
    """CameraData should reject intrinsics whose image dimensions disagree with metadata."""
    intrinsics = _make_intrinsics(width=2, height=3)

    with pytest.raises(
        ValueError,
        match=(
            r"CameraIntrinsics dimensions \(2x3\) do not match VideoMetadata dimensions \(1x1\)\. "
            r"This may indicate either rig calibration errors or video encoding errors\."
        ),
    ):
        CameraData(
            align_timestamps_ns=np.array([1], dtype=np.int64),
            sensor_timestamps_ns=np.array([1], dtype=np.int64),
            pts_stream=np.array([1], dtype=np.int64),
            frames=np.zeros((1, 1, 1, 3), dtype=np.uint8),
            metadata=_make_metadata(),
            intrinsics=intrinsics,
        )


@pytest.mark.parametrize(
    ("align_timestamps_ns", "sensor_timestamps_ns", "pts_stream", "match"),
    [
        (
            np.zeros((1, 1), dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            r"align_timestamps_ns",
        ),
        (
            np.zeros(1, dtype=np.int64),
            np.zeros((1, 1), dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            r"sensor_timestamps_ns",
        ),
        (np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64), np.zeros((1, 1), dtype=np.int64), r"pts_stream"),
        (
            np.zeros(1, dtype=np.int32),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            r"align_timestamps_ns must have dtype int64",
        ),
        (
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int32),
            np.zeros(1, dtype=np.int64),
            r"sensor_timestamps_ns must have dtype int64",
        ),
        (
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int32),
            r"pts_stream must have dtype int64",
        ),
    ],
)
def test_camera_data_rejects_non_1d_timestamp_arrays(
    align_timestamps_ns: npt.NDArray[np.int64],
    sensor_timestamps_ns: npt.NDArray[np.int64],
    pts_stream: npt.NDArray[np.int64],
    match: str,
) -> None:
    """CameraData should reject timestamp arrays with invalid shape or dtype."""
    with pytest.raises(ValueError, match=match):
        CameraData(
            align_timestamps_ns=align_timestamps_ns,
            sensor_timestamps_ns=sensor_timestamps_ns,
            pts_stream=pts_stream,
            frames=np.zeros((1, 1, 1, 3), dtype=np.uint8),
            metadata=_make_metadata(),
        )


@pytest.mark.parametrize(
    ("align_timestamps_ns", "sensor_timestamps_ns", "pts_stream", "match"),
    [
        (
            np.array([2, 1], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            r"align_timestamps_ns must be strictly sorted in ascending order with no duplicates",
        ),
        (
            np.array([1, 2], dtype=np.int64),
            np.array([2, 1], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            r"sensor_timestamps_ns must be sorted in ascending order",
        ),
        (
            np.array([1, 2], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            np.array([2, 1], dtype=np.int64),
            r"pts_stream must be sorted in ascending order",
        ),
    ],
)
def test_camera_data_rejects_nonmonotonic_timestamp_fields(
    align_timestamps_ns: npt.NDArray[np.int64],
    sensor_timestamps_ns: npt.NDArray[np.int64],
    pts_stream: npt.NDArray[np.int64],
    match: str,
) -> None:
    """CameraData should enforce its temporal ordering contract."""
    with pytest.raises(ValueError, match=match):
        CameraData(
            align_timestamps_ns=align_timestamps_ns,
            sensor_timestamps_ns=sensor_timestamps_ns,
            pts_stream=pts_stream,
            frames=np.zeros((2, 1, 1, 3), dtype=np.uint8),
            metadata=_make_metadata(),
        )


@pytest.mark.parametrize(
    ("frames", "match"),
    [
        (np.zeros((1, 3), dtype=np.uint8), r"frames must be 4-D"),
        (np.zeros((1, 1, 1, 1), dtype=np.uint8), r"frames must have shape"),
        (np.zeros((1, 1, 2, 3), dtype=np.uint8), r"frames must have shape"),
        (np.zeros((1, 1, 1, 3), dtype=np.float32), r"frames must have dtype uint8"),
    ],
)
def test_camera_data_rejects_invalid_frame_tensor(
    frames: npt.NDArray[Any],
    match: str,
) -> None:
    """CameraData should validate frame rank, shape, and dtype."""
    with pytest.raises(ValueError, match=match):
        CameraData(
            align_timestamps_ns=np.array([1], dtype=np.int64),
            sensor_timestamps_ns=np.array([1], dtype=np.int64),
            pts_stream=np.array([1], dtype=np.int64),
            frames=frames,
            metadata=_make_metadata(),
        )


@pytest.mark.parametrize(
    ("frames", "match"),
    [
        ((np.zeros(10, dtype=np.float64),), r"must be 2-D"),
        ((np.zeros((1, 9), dtype=np.float64),), r"shape=\(1, 9\)"),
    ],
)
def test_motion_vector_data_rejects_invalid_frame_shape(
    frames: tuple[npt.NDArray[np.float64], ...],
    match: str,
) -> None:
    """MotionVectorData should validate each per-frame block table."""
    with pytest.raises(ValueError, match=match):
        MotionVectorData(frames=frames)
