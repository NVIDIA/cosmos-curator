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
"""Tests for sensor-library ``EgoTrajectory``."""

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.sensors.data.aligned_frame import AlignedFrame
from cosmos_curator.core.sensors.data.egotrajectory_data import EgoTrajectory
from cosmos_curator.core.sensors.data.sensor_data import SensorData


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


def test_ego_trajectory_satisfies_sensor_data_protocol() -> None:
    """EgoTrajectory should be structurally usable anywhere SensorData is expected."""
    sensor_data: SensorData = _make_ego_trajectory()

    np.testing.assert_array_equal(sensor_data.align_timestamps_ns, np.array([100, 200], dtype=np.int64))
    np.testing.assert_array_equal(sensor_data.sensor_timestamps_ns, np.array([90, 210], dtype=np.int64))


def test_aligned_frame_rejects_mismatched_ego_trajectory_reference_timeline() -> None:
    """AlignedFrame should reject EgoTrajectory sampled on a different reference timeline."""
    ego = _make_ego_trajectory(align_timestamps_ns=np.array([100, 300], dtype=np.int64))

    with pytest.raises(ValueError, match="align_timestamps_ns must exactly match"):
        AlignedFrame(
            align_timestamps_ns=np.array([100, 200], dtype=np.int64),
            sensor_data={"ego": cast("SensorData", ego)},
        )


def test_ego_trajectory_arrays_are_readonly() -> None:
    """EgoTrajectory should expose read-only NumPy arrays for every field."""
    ego = _make_ego_trajectory(
        host_timestamps_ns=np.array([95, 215], dtype=np.int64),
        sequence_counter=np.array([1, 2], dtype=np.uint64),
    )

    for array in (
        ego.align_timestamps_ns,
        ego.sensor_timestamps_ns,
        ego.positions_m,
        ego.orientations_xyzw,
        ego.pose_valid,
        ego.host_timestamps_ns,
        ego.sequence_counter,
    ):
        assert array is not None
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = array.flat[0]


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        (
            "align_timestamps_ns",
            np.array([100, 100], dtype=np.int64),
            "strictly sorted in ascending order with no duplicates",
        ),
        ("positions_m", np.zeros((2, 3), dtype=np.float32), "dtype float64"),
        ("positions_m", np.zeros((2, 4), dtype=np.float64), r"shape \(N, 3\)"),
        ("positions_m", np.full((2, 3), np.nan, dtype=np.float64), "finite"),
        ("orientations_xyzw", np.zeros((2, 4), dtype=np.float64), "unit norm"),
        ("orientations_xyzw", np.tile(np.array([0.0, 0.0, 0.0, 2.0], dtype=np.float64), (2, 1)), "unit norm"),
        ("orientations_xyzw", np.zeros((2, 3), dtype=np.float64), r"shape \(N, 4\)"),
        ("pose_valid", np.array([True, False], dtype=np.int8), "dtype bool"),
        ("frame", "", "frame must be a non-empty string"),
    ],
)
def test_ego_trajectory_rejects_invalid_required_fields(
    field_name: str,
    value: object,
    match: str,
) -> None:
    """EgoTrajectory should validate position, orientation, pose_valid, and frame."""
    with pytest.raises(ValueError, match=match):
        _make_ego_trajectory(**{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("sensor_timestamps_ns", np.array([90, 210, 300], dtype=np.int64)),
        ("positions_m", np.zeros((3, 3), dtype=np.float64)),
        ("orientations_xyzw", np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), (3, 1))),
        ("pose_valid", np.array([True, True, False], dtype=np.bool_)),
        ("host_timestamps_ns", np.array([1, 2, 3], dtype=np.int64)),
        ("sequence_counter", np.array([1, 2, 3], dtype=np.uint64)),
    ],
)
def test_ego_trajectory_rejects_length_mismatch(
    field_name: str,
    value: npt.NDArray[Any],
) -> None:
    """All EgoTrajectory arrays must share the row-count ``N``."""
    with pytest.raises(ValueError, match="same length"):
        _make_ego_trajectory(**{field_name: value})


def test_ego_trajectory_accepts_keep_and_mask_identity_pose() -> None:
    """Invalid rows may keep a zero-position, identity-quaternion placeholder."""
    positions = np.zeros((2, 3), dtype=np.float64)
    positions[0] = (1.0, 2.0, 3.0)
    ego = _make_ego_trajectory(
        positions_m=positions,
        pose_valid=np.array([True, False], dtype=np.bool_),
    )

    assert ego.pose_valid.tolist() == [True, False]
    np.testing.assert_array_equal(ego.positions_m[1], np.zeros(3, dtype=np.float64))
    np.testing.assert_array_equal(ego.orientations_xyzw[1], np.array([0.0, 0.0, 0.0, 1.0]))


def test_ego_trajectory_rejects_non_identity_invalid_pose() -> None:
    """pose_valid=false rows cannot retain another finite pose."""
    positions = np.zeros((2, 3), dtype=np.float64)
    positions[1] = (1.0, 2.0, 3.0)

    with pytest.raises(ValueError, match="zero position and identity quaternion"):
        _make_ego_trajectory(
            positions_m=positions,
            pose_valid=np.array([True, False], dtype=np.bool_),
        )


def test_ego_trajectory_rejects_nonfinite_invalid_pose() -> None:
    """Invalid rows still require finite identity placeholders, not NaN."""
    positions = np.zeros((2, 3), dtype=np.float64)
    positions[1] = np.nan

    with pytest.raises(ValueError, match="finite"):
        _make_ego_trajectory(
            positions_m=positions,
            pose_valid=np.array([True, False], dtype=np.bool_),
        )


def test_ego_trajectory_adapts_quaternions_to_homogeneous_matrices() -> None:
    """Matrix consumers should receive the equivalent world-from-body transforms."""
    half_sqrt = np.sqrt(0.5)
    ego = _make_ego_trajectory(
        positions_m=np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float64),
        orientations_xyzw=np.array([[0.0, 0.0, half_sqrt, half_sqrt], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    )

    matrices = ego.as_homogeneous_matrices()

    np.testing.assert_allclose(
        matrices[0],
        np.array(
            [[0.0, -1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        atol=1e-12,
    )
    np.testing.assert_allclose(matrices[1], np.eye(4, dtype=np.float64), atol=1e-12)
    assert matrices.flags.writeable is False
