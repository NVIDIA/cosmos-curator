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

from cosmos_curator.core.sensors.data import gps_data as gps_data_module
from cosmos_curator.core.sensors.data.aligned_frame import AlignedFrame
from cosmos_curator.core.sensors.data.gps_data import GpsData, GpsFixType
from cosmos_curator.core.sensors.data.sensor_data import SensorData

_SCALAR_VALIDITY_BY_VALUE_FIELD = dict(gps_data_module._SCALAR_VALIDITY_PAIRS)


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


def test_gps_data_accepts_source_bracket_timestamps() -> None:
    """Derived GPS rows may record the inclusive pair of source support times."""
    brackets = np.array([[100, 100], [100, 200]], dtype=np.int64)

    gps_data = _make_gps_data(source_bracket_timestamps_ns=brackets)

    assert gps_data.source_bracket_timestamps_ns is not brackets
    assert gps_data.source_bracket_timestamps_ns is not None
    assert gps_data.source_bracket_timestamps_ns.flags.writeable is False
    np.testing.assert_array_equal(gps_data.source_bracket_timestamps_ns, brackets)


def test_gps_data_accepts_all_optional_fields() -> None:
    """GpsData should accept the planned generic optional GPS/GNSS fields."""
    gps_data = _make_gps_data(
        position_covariance_enu_m2=np.tile(np.eye(3, dtype=np.float64), (2, 1, 1)),
        velocity_enu_m_s=np.array([[1.0, 2.0, 0.1], [1.1, 2.1, 0.0]], dtype=np.float64),
        velocity_valid=np.ones((2, 3), dtype=np.bool_),
        fix_type=np.array([GpsFixType.FIX_3D, GpsFixType.RTK_FIXED], dtype=np.uint8),
        satellites_used=np.array([12, 14], dtype=np.uint32),
        satellites_used_valid=np.ones(2, dtype=np.bool_),
        horizontal_accuracy_m=np.array([0.5, 0.6], dtype=np.float64),
        horizontal_accuracy_m_valid=np.ones(2, dtype=np.bool_),
        vertical_accuracy_m=np.array([0.8, 0.9], dtype=np.float64),
        vertical_accuracy_m_valid=np.ones(2, dtype=np.bool_),
        hdop=np.array([0.7, 0.8], dtype=np.float64),
        hdop_valid=np.ones(2, dtype=np.bool_),
        vdop=np.array([0.9, 1.0], dtype=np.float64),
        vdop_valid=np.ones(2, dtype=np.bool_),
        pdop=np.array([1.2, 1.3], dtype=np.float64),
        pdop_valid=np.ones(2, dtype=np.bool_),
        host_timestamps_ns=np.array([95, 215], dtype=np.int64),
        utc_timestamps_ns=np.array([1_700_000_000_000_000_000, 1_700_000_000_100_000_000], dtype=np.int64),
        sequence_counter=np.array([7, 8], dtype=np.uint64),
    )

    assert gps_data.position_covariance_enu_m2 is not None
    assert gps_data.velocity_enu_m_s is not None
    np.testing.assert_array_equal(gps_data.fix_type, np.array([3, 6], dtype=np.uint8))


def test_gps_data_accepts_scalar_validity_presence_modes() -> None:
    """Scalar optionals should model absent, present, and partial per-sample presence."""
    absent = _make_gps_data()
    assert absent.hdop is None
    assert absent.hdop_valid is None
    assert absent.satellites_used is None
    assert absent.satellites_used_valid is None

    present = _make_gps_data(
        hdop=np.array([0.7, 0.8], dtype=np.float64),
        hdop_valid=np.array([True, True], dtype=np.bool_),
        satellites_used=np.array([12, 14], dtype=np.uint32),
        satellites_used_valid=np.array([True, True], dtype=np.bool_),
    )
    np.testing.assert_allclose(present.hdop, np.array([0.7, 0.8], dtype=np.float64))
    np.testing.assert_array_equal(present.hdop_valid, np.array([True, True], dtype=np.bool_))
    np.testing.assert_array_equal(present.satellites_used, np.array([12, 14], dtype=np.uint32))
    np.testing.assert_array_equal(present.satellites_used_valid, np.array([True, True], dtype=np.bool_))

    partial = _make_gps_data(
        hdop=np.array([0.7, 99.0], dtype=np.float64),
        hdop_valid=np.array([True, False], dtype=np.bool_),
        satellites_used=np.array([12, 99], dtype=np.uint32),
        satellites_used_valid=np.array([True, False], dtype=np.bool_),
    )
    np.testing.assert_allclose(partial.hdop, np.array([0.7, 99.0], dtype=np.float64))
    np.testing.assert_array_equal(partial.hdop_valid, np.array([True, False], dtype=np.bool_))
    np.testing.assert_array_equal(partial.satellites_used, np.array([12, 99], dtype=np.uint32))
    np.testing.assert_array_equal(partial.satellites_used_valid, np.array([True, False], dtype=np.bool_))


def test_gps_data_accepts_invalid_raw_values_when_validity_false() -> None:
    """Raw GPS values may violate numeric constraints only when masked invalid."""
    gps_data = _make_gps_data(
        latitude_deg=np.array([np.nan, 91.0], dtype=np.float64),
        longitude_deg=np.array([np.inf, -181.0], dtype=np.float64),
        altitude_m=np.array([np.nan, np.inf], dtype=np.float64),
        position_valid=np.zeros((2, 3), dtype=np.bool_),
        horizontal_accuracy_m=np.array([np.nan, -1.0], dtype=np.float64),
        horizontal_accuracy_m_valid=np.array([False, False], dtype=np.bool_),
        vertical_accuracy_m=np.array([np.inf, -1.0], dtype=np.float64),
        vertical_accuracy_m_valid=np.array([False, False], dtype=np.bool_),
        hdop=np.array([np.nan, -1.0], dtype=np.float64),
        hdop_valid=np.array([False, False], dtype=np.bool_),
        vdop=np.array([np.inf, -1.0], dtype=np.float64),
        vdop_valid=np.array([False, False], dtype=np.bool_),
        pdop=np.array([np.nan, -1.0], dtype=np.float64),
        pdop_valid=np.array([False, False], dtype=np.bool_),
    )

    assert np.isnan(gps_data.latitude_deg[0])
    np.testing.assert_allclose(gps_data.latitude_deg[1:], np.array([91.0]))
    assert np.isinf(gps_data.longitude_deg[0])
    np.testing.assert_allclose(gps_data.longitude_deg[1:], np.array([-181.0]))
    assert np.isnan(gps_data.altitude_m[0])
    assert np.isinf(gps_data.altitude_m[1])
    assert gps_data.hdop is not None
    assert np.isnan(gps_data.hdop[0])
    np.testing.assert_allclose(gps_data.hdop[1:], np.array([-1.0]))


def test_gps_fix_type_values_match_normalized_contract() -> None:
    """GpsFixType should retain the documented normalized GPS/GNSS status codes."""
    assert {fix_type.value for fix_type in GpsFixType} == {0, 2, 3, 4, 5, 6, 8}


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
        satellites_used=np.array([12, 14], dtype=np.uint32),
        satellites_used_valid=np.ones(2, dtype=np.bool_),
        horizontal_accuracy_m=np.array([0.5, 0.6], dtype=np.float64),
        horizontal_accuracy_m_valid=np.ones(2, dtype=np.bool_),
        vertical_accuracy_m=np.array([0.8, 0.9], dtype=np.float64),
        vertical_accuracy_m_valid=np.ones(2, dtype=np.bool_),
        hdop=np.array([0.7, 0.8], dtype=np.float64),
        hdop_valid=np.ones(2, dtype=np.bool_),
        vdop=np.array([0.9, 1.0], dtype=np.float64),
        vdop_valid=np.ones(2, dtype=np.bool_),
        pdop=np.array([1.2, 1.3], dtype=np.float64),
        pdop_valid=np.ones(2, dtype=np.bool_),
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
        gps_data.satellites_used_valid,
        gps_data.horizontal_accuracy_m,
        gps_data.horizontal_accuracy_m_valid,
        gps_data.vertical_accuracy_m,
        gps_data.vertical_accuracy_m_valid,
        gps_data.hdop,
        gps_data.hdop_valid,
        gps_data.vdop,
        gps_data.vdop_valid,
        gps_data.pdop,
        gps_data.pdop_valid,
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
        ("source_bracket_timestamps_ns", np.array([100, 200], dtype=np.int64), r"shape \(N, 2\)"),
        ("source_bracket_timestamps_ns", np.array([[100, 200], [200, 300]], dtype=np.int32), "dtype int64"),
        ("source_bracket_timestamps_ns", np.array([[200, 100], [200, 300]], dtype=np.int64), "left bracket"),
        (
            "source_bracket_timestamps_ns",
            np.array([[101, 200], [200, 300]], dtype=np.int64),
            "contain the matching align_timestamps_ns",
        ),
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
        ("horizontal_accuracy_m", np.array([0.5, -0.1], dtype=np.float64), "nonnegative"),
        ("vertical_accuracy_m", np.array([0.5, np.inf], dtype=np.float64), "finite"),
        ("hdop", np.array([0.7, -0.1], dtype=np.float64), "nonnegative"),
        ("vdop", np.array([0.7, np.nan], dtype=np.float64), "finite"),
    ],
)
def test_gps_data_rejects_invalid_optional_fields(
    field_name: str,
    value: npt.NDArray[Any],
    match: str,
) -> None:
    """GpsData should validate optional field dtype, shape, finite, and value constraints."""
    overrides: dict[str, npt.NDArray[Any]] = {field_name: value}
    validity_name = _SCALAR_VALIDITY_BY_VALUE_FIELD.get(field_name)
    if validity_name is not None:
        overrides[validity_name] = np.ones(len(value), dtype=np.bool_)
    with pytest.raises(ValueError, match=match):
        _make_gps_data(**overrides)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("position_covariance_enu_m2", np.eye(3, dtype=np.float64).reshape(1, 3, 3)),
        ("velocity_enu_m_s", np.ones((1, 3), dtype=np.float64)),
        ("velocity_valid", np.ones((1, 3), dtype=np.bool_)),
        ("fix_type", np.array([3], dtype=np.uint8)),
        ("satellites_used", np.array([12], dtype=np.uint32)),
        ("horizontal_accuracy_m", np.array([0.5], dtype=np.float64)),
        ("vertical_accuracy_m", np.array([0.8], dtype=np.float64)),
        ("hdop", np.array([0.7], dtype=np.float64)),
        ("vdop", np.array([0.9], dtype=np.float64)),
        ("pdop", np.array([1.2], dtype=np.float64)),
        ("host_timestamps_ns", np.array([95], dtype=np.int64)),
        ("utc_timestamps_ns", np.array([1_700_000_000_000_000_000], dtype=np.int64)),
        ("sequence_counter", np.array([7], dtype=np.uint64)),
        ("source_bracket_timestamps_ns", np.array([[100, 100]], dtype=np.int64)),
    ],
)
def test_gps_data_rejects_optional_batch_length_mismatches(
    field_name: str,
    value: npt.NDArray[Any],
) -> None:
    """GpsData should require optional arrays to share the required batch length."""
    overrides: dict[str, npt.NDArray[Any]] = {field_name: value}
    validity_name = _SCALAR_VALIDITY_BY_VALUE_FIELD.get(field_name)
    if validity_name is not None:
        overrides[validity_name] = np.ones(len(value), dtype=np.bool_)

    with pytest.raises(ValueError, match="same length"):
        _make_gps_data(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"hdop": np.array([0.7, 0.8], dtype=np.float64)},
        {"hdop_valid": np.array([True, True], dtype=np.bool_)},
        {"satellites_used": np.array([12, 14], dtype=np.uint32)},
        {"satellites_used_valid": np.array([True, True], dtype=np.bool_)},
    ],
)
def test_gps_data_rejects_unpaired_scalar_values_and_validity_masks(overrides: dict[str, npt.NDArray[Any]]) -> None:
    """Scalar optionals should be provided as both value and validity arrays, or neither."""
    with pytest.raises(ValueError, match="provided together"):
        _make_gps_data(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "hdop": np.array([0.7, 0.8], dtype=np.float64),
            "hdop_valid": np.array([True], dtype=np.bool_),
        },
        {
            "satellites_used": np.array([12, 14], dtype=np.uint32),
            "satellites_used_valid": np.array([True], dtype=np.bool_),
        },
    ],
)
def test_gps_data_rejects_scalar_validity_mask_batch_length_mismatches(
    overrides: dict[str, npt.NDArray[Any]],
) -> None:
    """Scalar validity masks should match the GPS batch length."""
    with pytest.raises(ValueError, match="same length"):
        _make_gps_data(**overrides)
