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
"""Tests for sensor-library ``LidarData`` and ``LidarMetadata``."""

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.sensors.data.aligned_frame import AlignedFrame
from cosmos_curator.core.sensors.data.egotrajectory_data import EgoTrajectory
from cosmos_curator.core.sensors.data.extrinsics import SensorExtrinsics
from cosmos_curator.core.sensors.data.lidar_data import LidarData, LidarMetadata
from cosmos_curator.core.sensors.data.sensor_data import SensorData


def _make_metadata(**overrides: object) -> LidarMetadata:
    values: dict[str, object] = {
        "motion_compensated": False,
        "reference_frame": "sensor",
    }
    values.update(overrides)
    return LidarMetadata(**values)


def _make_lidar_data(**overrides: object) -> LidarData:
    """Build a minimal valid LidarData batch (N=2 rows, P_total=3 points)."""
    values: dict[str, object] = {
        "align_timestamps_ns": np.array([100, 200], dtype=np.int64),
        "sensor_timestamps_ns": np.array([90, 210], dtype=np.int64),
        "points_xyz": np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float32,
        ),
        "points_timestamps_ns": np.array([100, 150, 200], dtype=np.int64),
        "metadata": _make_metadata(),
    }
    values.update(overrides)
    return LidarData(**values)


def _make_ego_trajectory(**overrides: object) -> EgoTrajectory:
    """Build a minimal valid EgoTrajectory (N=2)."""
    values: dict[str, object] = {
        "align_timestamps_ns": np.array([100, 200], dtype=np.int64),
        "sensor_timestamps_ns": np.array([90, 210], dtype=np.int64),
        "positions_m": np.zeros((2, 3), dtype=np.float64),
        "orientations_xyzw": np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), (2, 1)),
        "pose_valid": np.array([True, True], dtype=np.bool_),
        "frame": "world",
    }
    values.update(overrides)
    return EgoTrajectory(**values)


# --------------------------------------------------------------------------- #
# Required-field acceptance and protocol conformance
# --------------------------------------------------------------------------- #


def test_lidar_data_accepts_required_fields() -> None:
    """LidarData should accept a minimal required-field batch."""
    lidar_data = _make_lidar_data()

    assert len(lidar_data.align_timestamps_ns) == 2
    assert lidar_data.points_xyz.shape == (3, 3)
    assert lidar_data.metadata.reference_frame == "sensor"


def test_lidar_data_satisfies_sensor_data_protocol() -> None:
    """LidarData should be structurally usable anywhere SensorData is expected."""
    sensor_data: SensorData = _make_lidar_data()

    np.testing.assert_array_equal(sensor_data.align_timestamps_ns, np.array([100, 200], dtype=np.int64))
    np.testing.assert_array_equal(sensor_data.sensor_timestamps_ns, np.array([90, 210], dtype=np.int64))


# --------------------------------------------------------------------------- #
# AlignedFrame integration
# --------------------------------------------------------------------------- #


def test_aligned_frame_accepts_matching_lidar_data() -> None:
    """AlignedFrame should accept LidarData sampled on the same reference timeline."""
    lidar_data = _make_lidar_data()

    frame = AlignedFrame(
        align_timestamps_ns=np.array([100, 200], dtype=np.int64),
        sensor_data={"lidar0": cast("SensorData", lidar_data)},
    )

    assert frame["lidar0"] is lidar_data


def test_aligned_frame_accepts_matching_ego_trajectory() -> None:
    """AlignedFrame should accept EgoTrajectory sampled on the same reference timeline."""
    ego = _make_ego_trajectory()
    lidar_data = _make_lidar_data()

    frame = AlignedFrame(
        align_timestamps_ns=np.array([100, 200], dtype=np.int64),
        sensor_data={
            "lidar0": cast("SensorData", lidar_data),
            "ego": cast("SensorData", ego),
        },
    )

    assert frame["ego"] is ego


def test_aligned_frame_rejects_mismatched_lidar_data_reference_timeline() -> None:
    """AlignedFrame should reject LidarData sampled on a different reference timeline."""
    lidar_data = _make_lidar_data(align_timestamps_ns=np.array([100, 300], dtype=np.int64))

    with pytest.raises(ValueError, match="align_timestamps_ns must exactly match"):
        AlignedFrame(
            align_timestamps_ns=np.array([100, 200], dtype=np.int64),
            sensor_data={"lidar0": cast("SensorData", lidar_data)},
        )


# --------------------------------------------------------------------------- #
# Read-only views and caller-array preservation
# --------------------------------------------------------------------------- #


def test_lidar_data_arrays_are_readonly() -> None:
    """LidarData should expose read-only NumPy arrays for every array field, including optionals."""
    lidar_data = _make_lidar_data(
        points_intensity=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        points_ring=np.array([0, 1, 2], dtype=np.uint16),
        points_return_index=np.array([0, 1, 0], dtype=np.uint8),
        points_reflectivity=np.array([10, 20, 30], dtype=np.uint16),
        points_ambient=np.array([100, 200, 150], dtype=np.uint16),
        points_validity=np.array([True, True, False], dtype=np.bool_),
        points_radial_velocity=np.array([-1.5, 0.0, 1.5], dtype=np.float32),
        points_sweep_index=np.array([0, 0, 1], dtype=np.uint16),
        points_align_index=np.array([0, 0, 1], dtype=np.uint16),
    )

    arrays: list[npt.NDArray[Any] | None] = [
        lidar_data.align_timestamps_ns,
        lidar_data.sensor_timestamps_ns,
        lidar_data.points_xyz,
        lidar_data.points_timestamps_ns,
        lidar_data.points_intensity,
        lidar_data.points_ring,
        lidar_data.points_return_index,
        lidar_data.points_reflectivity,
        lidar_data.points_ambient,
        lidar_data.points_validity,
        lidar_data.points_radial_velocity,
        lidar_data.points_sweep_index,
        lidar_data.points_align_index,
    ]
    for array in arrays:
        assert array is not None
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = array.flat[0]


def test_lidar_data_does_not_mutate_caller_owned_arrays() -> None:
    """LidarData should expose read-only views without changing caller-owned arrays."""
    points_xyz = np.zeros((3, 3), dtype=np.float32)
    points_intensity = np.ones(3, dtype=np.float32)

    lidar_data = _make_lidar_data(
        points_xyz=points_xyz,
        points_intensity=points_intensity,
    )

    assert points_xyz.flags.writeable is True
    assert points_intensity.flags.writeable is True
    assert lidar_data.points_xyz.flags.writeable is False
    assert lidar_data.points_intensity is not None
    assert lidar_data.points_intensity.flags.writeable is False
    assert lidar_data.points_xyz is not points_xyz
    assert lidar_data.points_intensity is not points_intensity
    assert np.shares_memory(lidar_data.points_xyz, points_xyz)
    assert np.shares_memory(lidar_data.points_intensity, points_intensity)


# --------------------------------------------------------------------------- #
# Required-field validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("align_timestamps_ns", np.array([2, 1], dtype=np.int64), "strictly sorted"),
        ("align_timestamps_ns", np.array([100, 100], dtype=np.int64), "strictly sorted"),
        ("sensor_timestamps_ns", np.array([200, 100], dtype=np.int64), "sorted in ascending order"),
        ("points_xyz", np.zeros((3,), dtype=np.float32), r"shape \(N, 3\)"),
        ("points_xyz", np.zeros((3, 2), dtype=np.float32), r"shape \(N, 3\)"),
        ("points_xyz", np.zeros((3, 3), dtype=np.float64), "dtype float32"),
        (
            "points_xyz",
            np.array([[np.nan, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
            "finite",
        ),
        ("points_timestamps_ns", np.array([200, 150, 100], dtype=np.int64), "sorted in ascending order"),
        ("points_timestamps_ns", np.ones(3, dtype=np.int32), "dtype int64"),
    ],
)
def test_lidar_data_rejects_invalid_required_fields(
    field_name: str,
    value: npt.NDArray[Any],
    match: str,
) -> None:
    """LidarData should validate required timestamp and point fields."""
    with pytest.raises(ValueError, match=match):
        _make_lidar_data(**{field_name: value})


# --------------------------------------------------------------------------- #
# Optional-field validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("points_intensity", np.ones((3, 1), dtype=np.float32), r"must be 1-D"),
        ("points_intensity", np.ones(3, dtype=np.float64), "dtype float32"),
        ("points_intensity", np.array([0.0, np.inf, 0.0], dtype=np.float32), "finite"),
        ("points_ring", np.ones((3, 1), dtype=np.uint16), r"must be 1-D"),
        ("points_ring", np.ones(3, dtype=np.uint8), "dtype uint16"),
        ("points_return_index", np.ones((3, 1), dtype=np.uint8), r"must be 1-D"),
        ("points_return_index", np.ones(3, dtype=np.uint16), "dtype uint8"),
        ("points_reflectivity", np.ones(3, dtype=np.int16), "dtype uint16"),
        ("points_ambient", np.ones(3, dtype=np.uint32), "dtype uint16"),
        ("points_validity", np.ones((3, 1), dtype=np.bool_), r"must be 1-D"),
        ("points_validity", np.ones(3, dtype=np.int8), "dtype bool"),
        ("points_radial_velocity", np.array([0.0, np.nan, 0.0], dtype=np.float32), "finite"),
        ("points_radial_velocity", np.ones(3, dtype=np.float64), "dtype float32"),
        ("points_sweep_index", np.ones(3, dtype=np.uint8), "dtype uint16"),
        ("points_align_index", np.ones((3, 1), dtype=np.uint16), r"must be 1-D"),
        ("points_align_index", np.ones(3, dtype=np.int32), "dtype uint16"),
    ],
)
def test_lidar_data_rejects_invalid_optional_fields(
    field_name: str,
    value: npt.NDArray[Any],
    match: str,
) -> None:
    """LidarData should validate optional field dtype, shape, and finite-value constraints."""
    with pytest.raises(ValueError, match=match):
        _make_lidar_data(**{field_name: value})


# --------------------------------------------------------------------------- #
# Batch-length invariants (N and P_total)
# --------------------------------------------------------------------------- #


def test_lidar_data_rejects_row_length_mismatch() -> None:
    """align_timestamps_ns and sensor_timestamps_ns must share the row-count ``N``."""
    with pytest.raises(ValueError, match="row-level arrays must be the same length"):
        _make_lidar_data(sensor_timestamps_ns=np.array([90, 210, 300], dtype=np.int64))


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("points_timestamps_ns", np.array([100, 150], dtype=np.int64)),
        ("points_intensity", np.ones(2, dtype=np.float32)),
        ("points_ring", np.ones(2, dtype=np.uint16)),
        ("points_return_index", np.ones(2, dtype=np.uint8)),
        ("points_reflectivity", np.ones(2, dtype=np.uint16)),
        ("points_ambient", np.ones(2, dtype=np.uint16)),
        ("points_validity", np.ones(2, dtype=np.bool_)),
        ("points_radial_velocity", np.ones(2, dtype=np.float32)),
        ("points_sweep_index", np.ones(2, dtype=np.uint16)),
        ("points_align_index", np.zeros(2, dtype=np.uint16)),
    ],
)
def test_lidar_data_rejects_point_length_mismatches(
    field_name: str,
    value: npt.NDArray[Any],
) -> None:
    """All per-point arrays must share the point-count ``P_total``."""
    with pytest.raises(ValueError, match="per-point arrays must be the same length"):
        _make_lidar_data(**{field_name: value})


@pytest.mark.parametrize(
    "value",
    [
        np.array([0, 1, 2], dtype=np.uint16),
        np.array([2, 0, 1], dtype=np.uint16),
    ],
)
def test_lidar_data_rejects_out_of_range_points_align_index(
    value: npt.NDArray[np.uint16],
) -> None:
    """points_align_index entries must satisfy ``v < N``."""
    with pytest.raises(ValueError, match=r"points_align_index entries must satisfy v < N"):
        _make_lidar_data(points_align_index=value)


def test_lidar_data_accepts_in_range_points_align_index() -> None:
    """A points_align_index whose values fall in ``[0, N)`` should be accepted."""
    lidar_data = _make_lidar_data(points_align_index=np.array([0, 0, 1], dtype=np.uint16))

    assert lidar_data.points_align_index is not None
    np.testing.assert_array_equal(
        lidar_data.points_align_index,
        np.array([0, 0, 1], dtype=np.uint16),
    )


# --------------------------------------------------------------------------- #
# LidarMetadata cross-field validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("motion_compensated", "reference_frame"),
    [
        (False, "sensor"),
        (False, "rig"),
        (True, "rig"),
        (True, "world"),
        (True, "map"),
        (True, "odom"),
    ],
)
def test_lidar_metadata_accepts_valid_frame_combinations(
    motion_compensated: bool,  # noqa: FBT001
    reference_frame: str,
) -> None:
    """Valid ``(motion_compensated, reference_frame)`` combinations should build."""
    metadata = _make_metadata(motion_compensated=motion_compensated, reference_frame=reference_frame)

    assert metadata.motion_compensated is motion_compensated
    assert metadata.reference_frame == reference_frame


@pytest.mark.parametrize(
    ("motion_compensated", "reference_frame", "match"),
    [
        (True, "sensor", r"reference_frame must be one of \['map', 'odom', 'rig', 'world'\]"),
        (True, "foo", r"reference_frame must be one of \['map', 'odom', 'rig', 'world'\]"),
        (False, "world", r"reference_frame must be one of \['rig', 'sensor'\]"),
        (False, "map", r"reference_frame must be one of \['rig', 'sensor'\]"),
        (False, "odom", r"reference_frame must be one of \['rig', 'sensor'\]"),
    ],
)
def test_lidar_metadata_rejects_invalid_frame_combinations(
    motion_compensated: bool,  # noqa: FBT001
    reference_frame: str,
    match: str,
) -> None:
    """Invalid ``(motion_compensated, reference_frame)`` combinations should raise."""
    with pytest.raises(ValueError, match=match):
        _make_metadata(motion_compensated=motion_compensated, reference_frame=reference_frame)


def test_lidar_metadata_rejects_empty_reference_frame() -> None:
    """reference_frame must be a non-empty string."""
    with pytest.raises(ValueError, match="reference_frame must be a non-empty string"):
        _make_metadata(reference_frame="")


def test_lidar_metadata_accepts_optional_extrinsics_and_model() -> None:
    """LidarMetadata should carry optional extrinsics and sensor_model."""
    extrinsics = SensorExtrinsics(matrix=np.eye(4, dtype=np.float64))
    metadata = _make_metadata(extrinsics=extrinsics, sensor_model="ouster-os1-64")

    assert metadata.extrinsics is extrinsics
    assert metadata.sensor_model == "ouster-os1-64"


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("extrinsics", np.eye(4, dtype=np.float64), "extrinsics must be a SensorExtrinsics"),
        ("sensor_model", "", "sensor_model must be a non-empty string"),
    ],
)
def test_lidar_metadata_rejects_invalid_optional_fields(
    field_name: str,
    value: object,
    match: str,
) -> None:
    """LidarMetadata should validate its optional fields."""
    with pytest.raises(ValueError, match=match):
        _make_metadata(**{field_name: value})
