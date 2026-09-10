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

"""Tests for pure GPS alignment operations."""

import numpy as np
import pytest

from cosmos_curator.core.sensors.alignment.gps import (
    _ecef_to_wgs84,
    _wgs84_to_ecef,
    align_gps,
    align_gps_interpolated,
    align_gps_nearest,
)
from cosmos_curator.core.sensors.data.gps_data import GpsData
from cosmos_curator.core.sensors.exceptions import AlignmentError, AlignmentFailureReason
from cosmos_curator.core.sensors.sampling.policy import GpsInterpolationPolicy, NearestTimestampPolicy


def _source_gps_data() -> GpsData:
    """Build source fixes with distinguishable timestamps, positions, and metadata."""
    return GpsData(
        align_timestamps_ns=np.array([100, 200, 400], dtype=np.int64),
        sensor_timestamps_ns=np.array([1_100, 1_200, 1_400], dtype=np.int64),
        latitude_deg=np.array([10.0, 20.0, 40.0], dtype=np.float64),
        longitude_deg=np.array([-10.0, -20.0, -40.0], dtype=np.float64),
        altitude_m=np.array([1.0, 2.0, 4.0], dtype=np.float64),
        position_valid=np.array([[True, True, True], [False, False, False], [True, True, True]], dtype=np.bool_),
        hdop=np.array([0.1, 0.2, 0.4], dtype=np.float64),
        hdop_valid=np.array([True, True, True], dtype=np.bool_),
        host_timestamps_ns=np.array([2_100, 2_200, 2_400], dtype=np.int64),
    )


def _uniform_gps_data(align_timestamps_ns: np.ndarray) -> GpsData:
    """Build valid equatorial source fixes at the supplied alignment timestamps."""
    count = len(align_timestamps_ns)
    return GpsData(
        align_timestamps_ns=align_timestamps_ns,
        sensor_timestamps_ns=align_timestamps_ns,
        latitude_deg=np.zeros(count, dtype=np.float64),
        longitude_deg=align_timestamps_ns.astype(np.float64) / 1_000.0,
        altitude_m=np.zeros(count, dtype=np.float64),
        position_valid=np.ones((count, 3), dtype=np.bool_),
    )


def test_nearest_on_a_clip_with_halo_fixes_matches_the_unsplit_session() -> None:
    """Two halo fixes make clip-local nearest selection independent of where the cut fell.

    Any fix beyond a halo is farther from an in-span request than the halo
    itself, so a clip carrying one predecessor and one successor resolves every
    request exactly as the whole session would. This is why nearest is allowed
    to select a halo fix, and the test also pins the consequence: some selected
    provenance timestamps fall outside the clip span.
    """
    session_timestamps_ns = np.arange(0, 10_001, 1_000, dtype=np.int64)
    clip_start_ns, clip_exclusive_end_ns = 3_200, 6_900
    requested_timestamps_ns = np.arange(clip_start_ns, clip_exclusive_end_ns, 250, dtype=np.int64)

    in_span = session_timestamps_ns[
        (session_timestamps_ns >= clip_start_ns) & (session_timestamps_ns < clip_exclusive_end_ns)
    ]
    predecessor = session_timestamps_ns[session_timestamps_ns < clip_start_ns][-1:]
    successor = session_timestamps_ns[session_timestamps_ns >= clip_exclusive_end_ns][:1]

    policy = NearestTimestampPolicy()
    session = align_gps_nearest(_uniform_gps_data(session_timestamps_ns), requested_timestamps_ns, policy=policy)
    with_halo = align_gps_nearest(
        _uniform_gps_data(np.concatenate([predecessor, in_span, successor])),
        requested_timestamps_ns,
        policy=policy,
    )
    without_halo = align_gps_nearest(_uniform_gps_data(in_span), requested_timestamps_ns, policy=policy)

    np.testing.assert_array_equal(with_halo.sensor_timestamps_ns, session.sensor_timestamps_ns)
    np.testing.assert_array_equal(with_halo.longitude_deg, session.longitude_deg)
    # The halo is what buys that: span-restricted eligibility answers differently.
    assert not np.array_equal(without_halo.sensor_timestamps_ns, session.sensor_timestamps_ns)
    selected = with_halo.sensor_timestamps_ns
    assert np.any(selected < clip_start_ns)
    assert np.any(selected >= clip_exclusive_end_ns)


def test_nearest_selects_exact_earlier_later_and_reused_source_rows() -> None:
    """Nearest output has one row per request, preserving ties and source reuse."""
    result = align_gps_nearest(
        _source_gps_data(),
        np.array([100, 150, 190, 205, 350], dtype=np.int64),
        policy=NearestTimestampPolicy(),
    )

    np.testing.assert_array_equal(result.align_timestamps_ns, [100, 150, 190, 205, 350])
    np.testing.assert_array_equal(result.sensor_timestamps_ns, [1_100, 1_100, 1_200, 1_200, 1_400])
    np.testing.assert_allclose(result.latitude_deg, [10.0, 10.0, 20.0, 20.0, 40.0])
    assert result.hdop is not None
    np.testing.assert_allclose(result.hdop, [0.1, 0.1, 0.2, 0.2, 0.4])
    assert result.host_timestamps_ns is not None
    np.testing.assert_array_equal(result.host_timestamps_ns, [2_100, 2_100, 2_200, 2_200, 2_400])
    assert result.source_bracket_timestamps_ns is None


def test_align_gps_routes_by_explicit_policy() -> None:
    """The public entry point dispatches to the matching pure alignment operation."""
    nearest = align_gps(
        _source_gps_data(),
        np.array([100], dtype=np.int64),
        policy=NearestTimestampPolicy(),
    )
    interpolated = align_gps(
        _source_gps_data(),
        np.array([100], dtype=np.int64),
        policy=GpsInterpolationPolicy(),
    )

    assert nearest.source_bracket_timestamps_ns is None
    np.testing.assert_array_equal(interpolated.source_bracket_timestamps_ns, [[100, 100]])


def test_align_gps_rejects_a_bare_policy() -> None:
    """The public dispatcher requires one of the two concrete GPS alignment policies."""
    with pytest.raises(TypeError, match="NearestTimestampPolicy or GpsInterpolationPolicy"):
        align_gps(
            _source_gps_data(),
            np.array([100], dtype=np.int64),
            policy=None,  # type: ignore[arg-type]
        )


def test_nearest_preserves_invalid_selected_positions() -> None:
    """Nearest selection is provenance-preserving and does not repair source validity."""
    result = align_gps_nearest(
        _source_gps_data(),
        np.array([200], dtype=np.int64),
        policy=NearestTimestampPolicy(),
    )

    np.testing.assert_array_equal(result.position_valid, [[False, False, False]])


def test_nearest_rejects_a_selection_outside_maximum_delta() -> None:
    """Tolerance failures retain both request and selected source diagnostics."""
    with pytest.raises(AlignmentError) as error_info:
        align_gps_nearest(
            _source_gps_data(),
            np.array([100, 260], dtype=np.int64),
            policy=NearestTimestampPolicy(max_delta_ns=50),
        )

    error = error_info.value
    assert error.reason is AlignmentFailureReason.GPS_MAX_DELTA_EXCEEDED
    assert error.delta_ns == 60
    assert error.max_delta_ns == 50
    np.testing.assert_array_equal(error.align_timestamps_ns, [260])
    np.testing.assert_array_equal(error.sensor_timestamps_ns, [1_200])
    np.testing.assert_array_equal(error.known_source_timestamps_ns, [200])


def test_nearest_zero_maximum_delta_requires_an_exact_match() -> None:
    """A zero nearest tolerance permits only exact source alignment timestamps."""
    exact = align_gps_nearest(
        _source_gps_data(),
        np.array([100], dtype=np.int64),
        policy=NearestTimestampPolicy(max_delta_ns=0),
    )
    np.testing.assert_array_equal(exact.sensor_timestamps_ns, [1_100])

    with pytest.raises(AlignmentError) as error_info:
        align_gps_nearest(
            _source_gps_data(),
            np.array([101], dtype=np.int64),
            policy=NearestTimestampPolicy(max_delta_ns=0),
        )

    assert error_info.value.reason is AlignmentFailureReason.GPS_MAX_DELTA_EXCEEDED
    assert error_info.value.delta_ns == 1


def test_nearest_rejects_nonempty_request_without_source_fixes() -> None:
    """An empty source is an expected data-alignment failure, not a programming error."""
    source = GpsData(
        align_timestamps_ns=np.array([], dtype=np.int64),
        sensor_timestamps_ns=np.array([], dtype=np.int64),
        latitude_deg=np.array([], dtype=np.float64),
        longitude_deg=np.array([], dtype=np.float64),
        altitude_m=np.array([], dtype=np.float64),
        position_valid=np.empty((0, 3), dtype=np.bool_),
    )

    with pytest.raises(AlignmentError, match="no GPS source fix") as error_info:
        align_gps_nearest(source, np.array([100], dtype=np.int64), policy=NearestTimestampPolicy())

    assert error_info.value.reason is AlignmentFailureReason.GPS_NO_ELIGIBLE_FIX


def test_nearest_none_maximum_delta_leaves_selection_unbounded() -> None:
    """None disables only the nearest-distance limit."""
    result = align_gps_nearest(
        _source_gps_data(),
        np.array([1_000], dtype=np.int64),
        policy=NearestTimestampPolicy(max_delta_ns=None),
    )

    np.testing.assert_array_equal(result.sensor_timestamps_ns, [1_400])


def test_interpolation_derives_ecef_position_and_bracket_provenance() -> None:
    """ECEF interpolation differs from direct WGS-84 coordinate interpolation."""
    source = GpsData(
        align_timestamps_ns=np.array([100, 200], dtype=np.int64),
        sensor_timestamps_ns=np.array([1_100, 1_200], dtype=np.int64),
        latitude_deg=np.array([0.0, 0.0], dtype=np.float64),
        longitude_deg=np.array([0.0, 90.0], dtype=np.float64),
        altitude_m=np.array([0.0, 0.0], dtype=np.float64),
        position_valid=np.ones((2, 3), dtype=np.bool_),
        hdop=np.array([0.1, 0.2], dtype=np.float64),
        hdop_valid=np.array([True, True], dtype=np.bool_),
    )

    result = align_gps_interpolated(
        source,
        np.array([150], dtype=np.int64),
        policy=GpsInterpolationPolicy(max_gap_ns=100),
    )

    np.testing.assert_allclose(result.latitude_deg, [0.0], atol=1e-10)
    np.testing.assert_allclose(result.longitude_deg, [45.0], atol=1e-10)
    np.testing.assert_allclose(result.altitude_m, [-1_868_113.0759631777], atol=1e-6)
    np.testing.assert_array_equal(result.align_timestamps_ns, [150])
    np.testing.assert_array_equal(result.sensor_timestamps_ns, [150])
    np.testing.assert_array_equal(result.position_valid, [[True, True, True]])
    np.testing.assert_array_equal(result.source_bracket_timestamps_ns, [[100, 200]])
    assert result.hdop is None
    assert result.hdop_valid is None


def test_interpolation_handles_wgs84_poles_without_a_projection_library() -> None:
    """WGS-84 pole positions round-trip through ECEF interpolation exactly."""
    source = GpsData(
        align_timestamps_ns=np.array([100, 200], dtype=np.int64),
        sensor_timestamps_ns=np.array([1_100, 1_200], dtype=np.int64),
        latitude_deg=np.array([90.0, 90.0], dtype=np.float64),
        longitude_deg=np.array([0.0, 0.0], dtype=np.float64),
        altitude_m=np.array([0.0, 0.0], dtype=np.float64),
        position_valid=np.ones((2, 3), dtype=np.bool_),
    )

    result = align_gps_interpolated(source, np.array([150], dtype=np.int64), policy=GpsInterpolationPolicy())

    np.testing.assert_allclose(result.latitude_deg, [90.0], atol=1e-10)
    np.testing.assert_allclose(result.longitude_deg, [0.0], atol=1e-10)
    np.testing.assert_allclose(result.altitude_m, [0.0], atol=1e-6)


def test_ecef_inverse_recovers_altitude_near_but_not_on_the_polar_axis() -> None:
    """The inverse conversion avoids a near-axis altitude cancellation error."""
    latitude_deg = np.array([90.0 - np.rad2deg(1e-10)], dtype=np.float64)
    longitude_deg = np.array([0.0], dtype=np.float64)
    altitude_m = np.array([123.0], dtype=np.float64)
    ecef_x_m, ecef_y_m, ecef_z_m = _wgs84_to_ecef(longitude_deg, latitude_deg, altitude_m)

    recovered_longitude_deg, recovered_latitude_deg, recovered_altitude_m = _ecef_to_wgs84(
        ecef_x_m,
        ecef_y_m,
        ecef_z_m,
    )

    np.testing.assert_allclose(recovered_longitude_deg, longitude_deg, atol=1e-10)
    np.testing.assert_allclose(recovered_latitude_deg, latitude_deg, atol=1e-10)
    np.testing.assert_allclose(recovered_altitude_m, altitude_m, atol=1e-6)


def test_interpolation_exact_valid_fix_uses_degenerate_bracket() -> None:
    """Exact source fixes remain exact while retaining derived-row schema."""
    result = align_gps_interpolated(
        _source_gps_data(),
        np.array([100], dtype=np.int64),
        policy=GpsInterpolationPolicy(),
    )

    np.testing.assert_allclose(result.latitude_deg, [10.0])
    np.testing.assert_allclose(result.longitude_deg, [-10.0])
    np.testing.assert_allclose(result.altitude_m, [1.0])
    np.testing.assert_array_equal(result.source_bracket_timestamps_ns, [[100, 100]])
    assert result.host_timestamps_ns is None


def test_interpolation_uses_clip_halo_fixes_at_both_boundaries() -> None:
    """A clip's adjacent source fixes support interpolation inside its half-open span.

    The fixes sit on the equator one degree apart, so an ECEF chord midpoint has
    an analytic answer: latitude stays exactly zero and longitude is the exact
    bracket midpoint, while altitude drops by the chord's sagitta below the
    ellipsoid. Requests 120 and 180 are those midpoints, and each one can only
    be answered using a halo fix.
    """
    source = GpsData(
        # The planned clip is [100, 200); 90 and 210 are its copied halos.
        align_timestamps_ns=np.array([90, 150, 210], dtype=np.int64),
        sensor_timestamps_ns=np.array([90, 150, 210], dtype=np.int64),
        latitude_deg=np.zeros(3, dtype=np.float64),
        longitude_deg=np.array([0.0, 1.0, 2.0], dtype=np.float64),
        altitude_m=np.zeros(3, dtype=np.float64),
        position_valid=np.ones((3, 3), dtype=np.bool_),
    )

    result = align_gps_interpolated(
        source,
        np.array([100, 120, 180, 199], dtype=np.int64),
        policy=GpsInterpolationPolicy(),
    )

    np.testing.assert_array_equal(
        result.source_bracket_timestamps_ns,
        [[90, 150], [90, 150], [150, 210], [150, 210]],
    )
    np.testing.assert_array_equal(result.latitude_deg, np.zeros(4))
    # One degree of equatorial chord sags 6378137 * (1 - cos(0.5 deg)) below the ellipsoid.
    expected_sagitta_m = -242.859_913_255_088
    np.testing.assert_allclose(result.longitude_deg[[1, 2]], [0.5, 1.5], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(result.altitude_m[[1, 2]], [expected_sagitta_m] * 2, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize(
    ("source_timestamps_ns", "requested_timestamp_ns"),
    [
        (np.array([150, 210], dtype=np.int64), 100),
        (np.array([90, 150], dtype=np.int64), 199),
    ],
)
def test_interpolation_rejects_a_clip_boundary_that_would_need_extrapolation(
    source_timestamps_ns: np.ndarray,
    requested_timestamp_ns: int,
) -> None:
    """A missing predecessor or successor halo is an alignment failure, never extrapolation."""
    source = GpsData(
        align_timestamps_ns=source_timestamps_ns,
        sensor_timestamps_ns=source_timestamps_ns,
        latitude_deg=np.linspace(0.0, 10.0, len(source_timestamps_ns)),
        longitude_deg=np.linspace(0.0, 10.0, len(source_timestamps_ns)),
        altitude_m=np.linspace(0.0, 10.0, len(source_timestamps_ns)),
        position_valid=np.ones((len(source_timestamps_ns), 3), dtype=np.bool_),
    )

    with pytest.raises(AlignmentError, match="would require extrapolation") as error_info:
        align_gps_interpolated(
            source,
            np.array([requested_timestamp_ns], dtype=np.int64),
            policy=GpsInterpolationPolicy(),
        )

    error = error_info.value
    assert error.reason is AlignmentFailureReason.GPS_MISSING_BRACKET
    np.testing.assert_array_equal(error.align_timestamps_ns, [requested_timestamp_ns])
    np.testing.assert_array_equal(error.known_source_timestamps_ns, source_timestamps_ns)


def test_extrapolation_failure_reports_only_the_source_bounds() -> None:
    """The failure payload stays bounded rather than copying a session-scale timeline."""
    source_timestamps_ns = np.arange(0, 10_000, dtype=np.int64)

    with pytest.raises(AlignmentError) as error_info:
        align_gps_interpolated(
            _uniform_gps_data(source_timestamps_ns),
            np.array([20_000], dtype=np.int64),
            policy=GpsInterpolationPolicy(),
        )

    assert error_info.value.reason is AlignmentFailureReason.GPS_MISSING_BRACKET
    np.testing.assert_array_equal(error_info.value.known_source_timestamps_ns, [0, 9_999])


def test_interpolation_rejects_missing_or_invalid_support() -> None:
    """Interpolation neither extrapolates nor uses an invalid source position."""
    with pytest.raises(AlignmentError) as missing_error:
        align_gps_interpolated(
            _source_gps_data(),
            np.array([50], dtype=np.int64),
            policy=GpsInterpolationPolicy(),
        )

    assert missing_error.value.reason is AlignmentFailureReason.GPS_MISSING_BRACKET

    with pytest.raises(AlignmentError) as above_range_error:
        align_gps_interpolated(
            _source_gps_data(),
            np.array([500], dtype=np.int64),
            policy=GpsInterpolationPolicy(),
        )

    assert above_range_error.value.reason is AlignmentFailureReason.GPS_MISSING_BRACKET

    with pytest.raises(AlignmentError) as invalid_error:
        align_gps_interpolated(
            _source_gps_data(),
            np.array([300], dtype=np.int64),
            policy=GpsInterpolationPolicy(),
        )

    assert invalid_error.value.reason is AlignmentFailureReason.GPS_INVALID_SUPPORT_POSITION
    np.testing.assert_array_equal(invalid_error.value.align_timestamps_ns, [300])
    np.testing.assert_array_equal(invalid_error.value.known_source_timestamps_ns, [200, 400])


def test_interpolation_rejects_nonfinite_support() -> None:
    """Interpolation rejects nonfinite source positions marked invalid by ``GpsData``."""
    source = GpsData(
        align_timestamps_ns=np.array([100, 200], dtype=np.int64),
        sensor_timestamps_ns=np.array([1_100, 1_200], dtype=np.int64),
        latitude_deg=np.array([0.0, np.nan], dtype=np.float64),
        longitude_deg=np.array([0.0, 90.0], dtype=np.float64),
        altitude_m=np.array([0.0, 0.0], dtype=np.float64),
        position_valid=np.array([[True, True, True], [False, True, True]], dtype=np.bool_),
    )

    with pytest.raises(AlignmentError) as error_info:
        align_gps_interpolated(source, np.array([150], dtype=np.int64), policy=GpsInterpolationPolicy())

    assert error_info.value.reason is AlignmentFailureReason.GPS_INVALID_SUPPORT_POSITION
    np.testing.assert_array_equal(error_info.value.align_timestamps_ns, [150])
    np.testing.assert_array_equal(error_info.value.known_source_timestamps_ns, [100, 200])


def test_interpolation_checks_gap_before_invalid_support() -> None:
    """An oversized bracket is reported before inspecting unusable support data."""
    with pytest.raises(AlignmentError) as error_info:
        align_gps_interpolated(
            _source_gps_data(),
            np.array([300], dtype=np.int64),
            policy=GpsInterpolationPolicy(max_gap_ns=100),
        )

    error = error_info.value
    assert error.reason is AlignmentFailureReason.GPS_MAX_GAP_EXCEEDED
    assert error.gap_ns == 200
    assert error.max_gap_ns == 100
    np.testing.assert_array_equal(error.align_timestamps_ns, [300])
    np.testing.assert_array_equal(error.known_source_timestamps_ns, [200, 400])


def test_interpolation_none_maximum_gap_leaves_valid_brackets_unbounded() -> None:
    """None disables only the interpolation support-gap limit."""
    source = GpsData(
        align_timestamps_ns=np.array([100, 1_000], dtype=np.int64),
        sensor_timestamps_ns=np.array([1_100, 2_000], dtype=np.int64),
        latitude_deg=np.array([0.0, 0.0], dtype=np.float64),
        longitude_deg=np.array([0.0, 90.0], dtype=np.float64),
        altitude_m=np.array([0.0, 0.0], dtype=np.float64),
        position_valid=np.ones((2, 3), dtype=np.bool_),
    )

    result = align_gps_interpolated(
        source,
        np.array([550], dtype=np.int64),
        policy=GpsInterpolationPolicy(max_gap_ns=None),
    )

    np.testing.assert_array_equal(result.source_bracket_timestamps_ns, [[100, 1_000]])


def test_interpolation_empty_request_has_derived_schema() -> None:
    """Empty derived output retains its fixed provenance schema without source metadata."""
    result = align_gps_interpolated(
        _source_gps_data(),
        np.empty(0, dtype=np.int64),
        policy=GpsInterpolationPolicy(),
    )

    assert len(result.align_timestamps_ns) == 0
    assert result.source_bracket_timestamps_ns is not None
    assert result.source_bracket_timestamps_ns.shape == (0, 2)
    assert result.hdop is None
