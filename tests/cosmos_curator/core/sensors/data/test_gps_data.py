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
"""Tests for sensor-library ``GpsData``."""

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.sensors.data.aligned_frame import AlignedFrame
from cosmos_curator.core.sensors.data.gps_data import GpsData, GpsFixType
from cosmos_curator.core.sensors.data.sensor_data import SensorData


def _make_gps_data(**overrides: object) -> GpsData:
    """Build a minimal valid GpsData batch."""
    values: dict[str, object] = {
        "align_timestamps_ns": np.array([100, 200], dtype=np.int64),
        "sensor_timestamps_ns": np.array([90, 210], dtype=np.int64),
        "latitude_deg": np.array([37.4, 37.5], dtype=np.float64),
        "longitude_deg": np.array([-122.1, -122.2], dtype=np.float64),
        "altitude_m": np.array([10.0, 11.0], dtype=np.float64),
        "position_valid": np.ones((2, 3), dtype=np.bool_),
    }
    values.update(overrides)
    return GpsData(**values)


def test_gps_data_accepts_required_fields() -> None:
    """GpsData should accept a minimal required-field batch."""
    gps_data = _make_gps_data()

    assert len(gps_data.align_timestamps_ns) == 2
    np.testing.assert_array_equal(gps_data.latitude_deg, np.array([37.4, 37.5], dtype=np.float64))


def test_gps_data_accepts_all_optional_fields() -> None:
    """GpsData should accept the planned generic optional GPS/GNSS fields."""
    gps_data = _make_gps_data(
        position_covariance_enu_m2=np.tile(np.eye(3, dtype=np.float64), (2, 1, 1)),
        velocity_enu_m_s=np.array([[1.0, 2.0, 0.1], [1.1, 2.1, 0.0]], dtype=np.float64),
        velocity_valid=np.ones((2, 3), dtype=np.bool_),
        fix_type=np.array([GpsFixType.FIX_3D, GpsFixType.RTK_FIXED], dtype=np.uint8),
        satellites_used=np.array([12, 14], dtype=np.uint16),
        horizontal_accuracy_m=np.array([0.5, 0.6], dtype=np.float64),
        vertical_accuracy_m=np.array([0.8, 0.9], dtype=np.float64),
        hdop=np.array([0.7, 0.8], dtype=np.float64),
        vdop=np.array([0.9, 1.0], dtype=np.float64),
        pdop=np.array([1.2, 1.3], dtype=np.float64),
        host_timestamps_ns=np.array([95, 215], dtype=np.int64),
        utc_timestamps_ns=np.array([1_700_000_000_000_000_000, 1_700_000_000_100_000_000], dtype=np.int64),
        sequence_counter=np.array([7, 8], dtype=np.uint64),
    )

    assert gps_data.position_covariance_enu_m2 is not None
    assert gps_data.velocity_enu_m_s is not None
    np.testing.assert_array_equal(gps_data.fix_type, np.array([3, 6], dtype=np.uint8))


def test_gps_fix_type_values_are_normalized_uint8_values() -> None:
    """GpsFixType should name the normalized values accepted by GpsData."""
    assert {fix_type.value for fix_type in GpsFixType} == {0, 2, 3, 4, 5, 6, 8}
    assert all(0 <= fix_type.value <= np.iinfo(np.uint8).max for fix_type in GpsFixType)


def test_gps_data_satisfies_sensor_data_protocol() -> None:
    """GpsData should be structurally usable anywhere SensorData is expected."""
    sensor_data: SensorData = _make_gps_data()

    np.testing.assert_array_equal(sensor_data.align_timestamps_ns, np.array([100, 200], dtype=np.int64))
    np.testing.assert_array_equal(sensor_data.sensor_timestamps_ns, np.array([90, 210], dtype=np.int64))


def test_aligned_frame_accepts_matching_gps_data() -> None:
    """AlignedFrame should accept GpsData sampled on the same reference timeline."""
    gps_data = _make_gps_data()

    frame = AlignedFrame(
        align_timestamps_ns=np.array([100, 200], dtype=np.int64),
        sensor_data={"gps0": cast("SensorData", gps_data)},
    )

    assert frame["gps0"] is gps_data


def test_aligned_frame_rejects_mismatched_gps_data_reference_timeline() -> None:
    """AlignedFrame should reject GpsData sampled on a different reference timeline."""
    gps_data = _make_gps_data(align_timestamps_ns=np.array([100, 300], dtype=np.int64))

    with pytest.raises(ValueError, match="align_timestamps_ns must exactly match"):
        AlignedFrame(
            align_timestamps_ns=np.array([100, 200], dtype=np.int64),
            sensor_data={"gps0": cast("SensorData", gps_data)},
        )


def test_gps_data_arrays_are_readonly() -> None:
    """GpsData should expose read-only NumPy arrays."""
    gps_data = _make_gps_data(
        position_covariance_enu_m2=np.tile(np.eye(3, dtype=np.float64), (2, 1, 1)),
        velocity_enu_m_s=np.array([[1.0, 2.0, 0.1], [1.1, 2.1, 0.0]], dtype=np.float64),
        velocity_valid=np.ones((2, 3), dtype=np.bool_),
        fix_type=np.array([3, 6], dtype=np.uint8),
        satellites_used=np.array([12, 14], dtype=np.uint16),
        horizontal_accuracy_m=np.array([0.5, 0.6], dtype=np.float64),
        vertical_accuracy_m=np.array([0.8, 0.9], dtype=np.float64),
        hdop=np.array([0.7, 0.8], dtype=np.float64),
        vdop=np.array([0.9, 1.0], dtype=np.float64),
        pdop=np.array([1.2, 1.3], dtype=np.float64),
        host_timestamps_ns=np.array([95, 215], dtype=np.int64),
        utc_timestamps_ns=np.array([1_700_000_000_000_000_000, 1_700_000_000_100_000_000], dtype=np.int64),
        sequence_counter=np.array([7, 8], dtype=np.uint64),
    )

    arrays = [
        gps_data.align_timestamps_ns,
        gps_data.sensor_timestamps_ns,
        gps_data.latitude_deg,
        gps_data.longitude_deg,
        gps_data.altitude_m,
        gps_data.position_valid,
        gps_data.position_covariance_enu_m2,
        gps_data.velocity_enu_m_s,
        gps_data.velocity_valid,
        gps_data.fix_type,
        gps_data.satellites_used,
        gps_data.horizontal_accuracy_m,
        gps_data.vertical_accuracy_m,
        gps_data.hdop,
        gps_data.vdop,
        gps_data.pdop,
        gps_data.host_timestamps_ns,
        gps_data.utc_timestamps_ns,
        gps_data.sequence_counter,
    ]

    for array in arrays:
        assert array is not None
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = array.flat[0]


def test_gps_data_creates_readonly_views_without_mutating_inputs() -> None:
    """GpsData should create read-only views without changing caller's writeable flag."""
    latitude_deg = np.array([37.4, 37.5], dtype=np.float64)
    position_valid = np.ones((2, 3), dtype=np.bool_)
    velocity_enu_m_s = np.zeros((2, 3), dtype=np.float64)

    gps_data = _make_gps_data(
        latitude_deg=latitude_deg,
        position_valid=position_valid,
        velocity_enu_m_s=velocity_enu_m_s,
    )

    assert latitude_deg.flags.writeable is True
    assert position_valid.flags.writeable is True
    assert velocity_enu_m_s.flags.writeable is True
    assert gps_data.latitude_deg.flags.writeable is False
    assert gps_data.position_valid.flags.writeable is False
    assert gps_data.velocity_enu_m_s is not None
    assert gps_data.velocity_enu_m_s.flags.writeable is False
    assert gps_data.latitude_deg is not latitude_deg
    assert gps_data.position_valid is not position_valid
    assert gps_data.velocity_enu_m_s is not velocity_enu_m_s
    assert np.shares_memory(gps_data.latitude_deg, latitude_deg)
    assert np.shares_memory(gps_data.position_valid, position_valid)
    assert np.shares_memory(gps_data.velocity_enu_m_s, velocity_enu_m_s)


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("align_timestamps_ns", np.array([2, 1], dtype=np.int64), "strictly sorted"),
        ("sensor_timestamps_ns", np.array([2, 1], dtype=np.int64), "sorted in ascending order"),
        ("latitude_deg", np.array([37.4, 37.5], dtype=np.float32), "dtype float64"),
        ("latitude_deg", np.array([37.4, 91.0], dtype=np.float64), "latitude"),
        ("longitude_deg", np.array([-122.1, 181.0], dtype=np.float64), "longitude"),
        ("altitude_m", np.array([10.0, np.nan], dtype=np.float64), "finite"),
        ("position_valid", np.ones((2,), dtype=np.bool_), r"shape \(N, 3\)"),
        ("position_valid", np.ones((2, 3), dtype=np.int8), "dtype bool"),
    ],
)
def test_gps_data_rejects_invalid_required_fields(
    field_name: str,
    value: npt.NDArray[Any],
    match: str,
) -> None:
    """GpsData should validate required timestamp, position, and validity fields."""
    with pytest.raises(ValueError, match=match):
        _make_gps_data(**{field_name: value})


def test_gps_data_rejects_required_batch_length_mismatches() -> None:
    """GpsData should require required arrays to share the same batch length."""
    with pytest.raises(ValueError, match="same length"):
        _make_gps_data(latitude_deg=np.array([37.4], dtype=np.float64))


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("position_covariance_enu_m2", np.zeros((2, 3), dtype=np.float64), r"shape \(N, 3, 3\)"),
        ("position_covariance_enu_m2", np.full((2, 3, 3), np.nan, dtype=np.float64), "finite"),
        ("position_covariance_enu_m2", np.tile(np.diag([1.0, -1.0, 1.0]), (2, 1, 1)), "positive"),
        (
            "position_covariance_enu_m2",
            np.tile(np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), (2, 1, 1)),
            "symmetric",
        ),
        ("velocity_enu_m_s", np.zeros((2,), dtype=np.float64), r"shape \(N, 3\)"),
        ("velocity_enu_m_s", np.zeros((2, 3), dtype=np.float32), "dtype float64"),
        ("velocity_valid", np.ones((2,), dtype=np.bool_), r"shape \(N, 3\)"),
        ("velocity_valid", np.ones((2, 3), dtype=np.int8), "dtype bool"),
        ("fix_type", np.array([3, 6], dtype=np.int64), "dtype uint8"),
        ("fix_type", np.array([3, 7], dtype=np.uint8), "valid fix type"),
        ("satellites_used", np.ones((2, 1), dtype=np.uint16), r"shape \(N,\)"),
        ("satellites_used", np.ones(2, dtype=np.uint8), "dtype uint16"),
        ("horizontal_accuracy_m", np.array([0.5, -0.1], dtype=np.float64), "nonnegative"),
        ("vertical_accuracy_m", np.array([0.5, np.inf], dtype=np.float64), "finite"),
        ("hdop", np.array([0.7, -0.1], dtype=np.float64), "nonnegative"),
        ("vdop", np.array([0.7, np.nan], dtype=np.float64), "finite"),
        ("pdop", np.ones((2, 1), dtype=np.float64), r"shape \(N,\)"),
        ("host_timestamps_ns", np.ones((2, 1), dtype=np.int64), r"shape \(N,\)"),
        ("utc_timestamps_ns", np.ones(2, dtype=np.uint64), "dtype int64"),
        ("sequence_counter", np.ones(2, dtype=np.int64), "dtype uint64"),
    ],
)
def test_gps_data_rejects_invalid_optional_fields(
    field_name: str,
    value: npt.NDArray[Any],
    match: str,
) -> None:
    """GpsData should validate optional field dtype, shape, finite, and value constraints."""
    with pytest.raises(ValueError, match=match):
        _make_gps_data(**{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("position_covariance_enu_m2", np.eye(3, dtype=np.float64).reshape(1, 3, 3)),
        ("velocity_enu_m_s", np.ones((1, 3), dtype=np.float64)),
        ("velocity_valid", np.ones((1, 3), dtype=np.bool_)),
        ("fix_type", np.array([3], dtype=np.uint8)),
        ("satellites_used", np.array([12], dtype=np.uint16)),
        ("horizontal_accuracy_m", np.array([0.5], dtype=np.float64)),
        ("vertical_accuracy_m", np.array([0.8], dtype=np.float64)),
        ("hdop", np.array([0.7], dtype=np.float64)),
        ("vdop", np.array([0.9], dtype=np.float64)),
        ("pdop", np.array([1.2], dtype=np.float64)),
        ("host_timestamps_ns", np.array([95], dtype=np.int64)),
        ("utc_timestamps_ns", np.array([1_700_000_000_000_000_000], dtype=np.int64)),
        ("sequence_counter", np.array([7], dtype=np.uint64)),
    ],
)
def test_gps_data_rejects_optional_batch_length_mismatches(
    field_name: str,
    value: npt.NDArray[Any],
) -> None:
    """GpsData should require optional arrays to share the required batch length."""
    with pytest.raises(ValueError, match="same length"):
        _make_gps_data(**{field_name: value})
