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

"""Pure GPS alignment operations over decoded GPS batches."""

import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.alignment.nearest import nearest_timestamp_indices, timestamp_distance_ns
from cosmos_curator.core.sensors.data.gps_data import GpsData
from cosmos_curator.core.sensors.exceptions import AlignmentError, AlignmentFailureReason
from cosmos_curator.core.sensors.sampling.policy import (
    GpsInterpolationPolicy,
    GpsSamplingPolicy,
    NearestTimestampPolicy,
)
from cosmos_curator.core.sensors.utils.validation import require_1d, require_strictly_increasing

_WGS84_SEMI_MAJOR_AXIS_M = np.float64(6_378_137.0)
_WGS84_FLATTENING = np.float64(1.0 / 298.257_223_563)
_WGS84_SEMI_MINOR_AXIS_M = _WGS84_SEMI_MAJOR_AXIS_M * (1.0 - _WGS84_FLATTENING)
_WGS84_ECCENTRICITY_SQUARED = _WGS84_FLATTENING * (2.0 - _WGS84_FLATTENING)
_WGS84_SECOND_ECCENTRICITY_SQUARED = (
    _WGS84_SEMI_MAJOR_AXIS_M**2 - _WGS84_SEMI_MINOR_AXIS_M**2
) / _WGS84_SEMI_MINOR_AXIS_M**2
_POLE_ECEF_RADIUS_M = np.float64(1e-9)


def _wgs84_to_ecef(
    longitude_deg: npt.NDArray[np.float64],
    latitude_deg: npt.NDArray[np.float64],
    altitude_m: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Convert WGS-84 geodetic coordinates to Earth-centered, Earth-fixed coordinates."""
    longitude_rad = np.deg2rad(longitude_deg)
    latitude_rad = np.deg2rad(latitude_deg)
    sin_latitude = np.sin(latitude_rad)
    normal_radius_m = _WGS84_SEMI_MAJOR_AXIS_M / np.sqrt(1.0 - _WGS84_ECCENTRICITY_SQUARED * sin_latitude**2)
    ecef_x_m = (normal_radius_m + altitude_m) * np.cos(latitude_rad) * np.cos(longitude_rad)
    ecef_y_m = (normal_radius_m + altitude_m) * np.cos(latitude_rad) * np.sin(longitude_rad)
    ecef_z_m = (normal_radius_m * (1.0 - _WGS84_ECCENTRICITY_SQUARED) + altitude_m) * sin_latitude
    return ecef_x_m, ecef_y_m, ecef_z_m


def _ecef_to_wgs84(
    ecef_x_m: npt.NDArray[np.float64],
    ecef_y_m: npt.NDArray[np.float64],
    ecef_z_m: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Convert Earth-centered, Earth-fixed coordinates to WGS-84 geodetic coordinates."""
    horizontal_radius_m = np.hypot(ecef_x_m, ecef_y_m)
    auxiliary_angle = np.arctan2(
        _WGS84_SEMI_MAJOR_AXIS_M * ecef_z_m,
        _WGS84_SEMI_MINOR_AXIS_M * horizontal_radius_m,
    )
    latitude_rad = np.arctan2(
        ecef_z_m + _WGS84_SECOND_ECCENTRICITY_SQUARED * _WGS84_SEMI_MINOR_AXIS_M * np.sin(auxiliary_angle) ** 3,
        horizontal_radius_m - _WGS84_ECCENTRICITY_SQUARED * _WGS84_SEMI_MAJOR_AXIS_M * np.cos(auxiliary_angle) ** 3,
    )
    longitude_rad = np.arctan2(ecef_y_m, ecef_x_m)
    polar = horizontal_radius_m <= _POLE_ECEF_RADIUS_M
    latitude_rad[polar] = np.copysign(np.pi / 2.0, ecef_z_m[polar])
    longitude_rad[polar] = 0.0
    sin_latitude = np.sin(latitude_rad)
    altitude_m = (
        horizontal_radius_m * np.cos(latitude_rad)
        + ecef_z_m * sin_latitude
        - _WGS84_SEMI_MAJOR_AXIS_M * np.sqrt(1.0 - _WGS84_ECCENTRICITY_SQUARED * sin_latitude**2)
    )
    return np.rad2deg(longitude_rad), np.rad2deg(latitude_rad), altitude_m


def _select(
    values: npt.NDArray[np.generic] | None,
    indices: npt.NDArray[np.int64],
) -> npt.NDArray[np.generic] | None:
    """Select rows from an optional GPS column."""
    return None if values is None else values[indices]


def align_gps(
    source: GpsData,
    requested_timestamps_ns: npt.NDArray[np.int64],
    *,
    policy: GpsSamplingPolicy,
) -> GpsData:
    """Align GPS source data with the caller's explicit selection policy."""
    runtime_policy: object = policy
    if isinstance(runtime_policy, NearestTimestampPolicy):
        return align_gps_nearest(source, requested_timestamps_ns, policy=runtime_policy)
    if isinstance(runtime_policy, GpsInterpolationPolicy):
        return align_gps_interpolated(source, requested_timestamps_ns, policy=runtime_policy)
    msg = f"policy must be NearestTimestampPolicy or GpsInterpolationPolicy, got {type(runtime_policy).__name__}"
    raise TypeError(msg)


def align_gps_nearest(
    source: GpsData,
    requested_timestamps_ns: npt.NDArray[np.int64],
    *,
    policy: NearestTimestampPolicy,
) -> GpsData:
    """Align GPS fixes to requested canonical timestamps using nearest selection.

    Callers supply decoded source data whose ``align_timestamps_ns`` use the
    mapper-produced alignment clock, and requested timestamps must use that same
    clock. ``GpsSensor`` currently populates this field from
    ``align_timestamp_ns``. This function neither reads MCAP nor knows about
    clips or splitting.

    A source fix may serve any number of requested rows. Equal-distance ties use
    the earlier fix. The returned rows retain the selected source metadata and
    device timestamps; only ``align_timestamps_ns`` changes to the requested
    canonical timestamps.

    Every source row is eligible for every request, so a selected
    ``sensor_timestamps_ns`` may lie outside the requested range: nearest
    provenance is not bounded by it. The splitter aligns against a whole
    session, so the nearest fix to a clip's first or last row is available
    whether or not it falls inside that clip.
    """
    if not isinstance(policy, NearestTimestampPolicy):
        msg = f"policy must be NearestTimestampPolicy, got {type(policy).__name__}"  # type: ignore[unreachable]
        raise TypeError(msg)

    require_1d("requested_timestamps_ns", requested_timestamps_ns, np.int64)
    require_strictly_increasing("requested_timestamps_ns", requested_timestamps_ns)

    if len(requested_timestamps_ns) and not len(source.align_timestamps_ns):
        msg = "no GPS source fix is available for the requested timestamps"
        raise AlignmentError(
            AlignmentFailureReason.GPS_NO_ELIGIBLE_FIX,
            msg,
            align_timestamps_ns=requested_timestamps_ns,
            known_source_timestamps_ns=source.align_timestamps_ns,
        )

    indices: npt.NDArray[np.int64]
    if not len(requested_timestamps_ns):
        indices = np.empty_like(requested_timestamps_ns, dtype=np.int64)
    else:
        indices = nearest_timestamp_indices(source.align_timestamps_ns, requested_timestamps_ns)

    selected_timestamps_ns = source.align_timestamps_ns[indices]
    deltas_ns = timestamp_distance_ns(requested_timestamps_ns, selected_timestamps_ns)
    if policy.max_delta_ns is not None and len(deltas_ns):
        worst_index = int(deltas_ns.argmax())
        worst_delta_ns = int(deltas_ns[worst_index])
        if worst_delta_ns > policy.max_delta_ns:
            msg = (
                f"max_delta_ns={policy.max_delta_ns} exceeded: max delta was {worst_delta_ns} ns "
                f"for requested={int(requested_timestamps_ns[worst_index])}, "
                f"source={int(selected_timestamps_ns[worst_index])}"
            )
            raise AlignmentError(
                AlignmentFailureReason.GPS_MAX_DELTA_EXCEEDED,
                msg,
                align_timestamps_ns=requested_timestamps_ns[[worst_index]],
                sensor_timestamps_ns=source.sensor_timestamps_ns[indices[[worst_index]]],
                known_source_timestamps_ns=selected_timestamps_ns[[worst_index]],
                delta_ns=worst_delta_ns,
                max_delta_ns=policy.max_delta_ns,
            )

    return GpsData(
        align_timestamps_ns=np.array(requested_timestamps_ns, copy=True),
        sensor_timestamps_ns=source.sensor_timestamps_ns[indices],
        latitude_deg=source.latitude_deg[indices],
        longitude_deg=source.longitude_deg[indices],
        altitude_m=source.altitude_m[indices],
        position_valid=source.position_valid[indices],
        position_covariance_enu_m2=_select(source.position_covariance_enu_m2, indices),
        velocity_enu_m_s=_select(source.velocity_enu_m_s, indices),
        velocity_valid=_select(source.velocity_valid, indices),
        fix_type=_select(source.fix_type, indices),
        satellites_used=_select(source.satellites_used, indices),
        satellites_used_valid=_select(source.satellites_used_valid, indices),
        horizontal_accuracy_m=_select(source.horizontal_accuracy_m, indices),
        horizontal_accuracy_m_valid=_select(source.horizontal_accuracy_m_valid, indices),
        vertical_accuracy_m=_select(source.vertical_accuracy_m, indices),
        vertical_accuracy_m_valid=_select(source.vertical_accuracy_m_valid, indices),
        hdop=_select(source.hdop, indices),
        hdop_valid=_select(source.hdop_valid, indices),
        vdop=_select(source.vdop, indices),
        vdop_valid=_select(source.vdop_valid, indices),
        pdop=_select(source.pdop, indices),
        pdop_valid=_select(source.pdop_valid, indices),
        host_timestamps_ns=_select(source.host_timestamps_ns, indices),
        utc_timestamps_ns=_select(source.utc_timestamps_ns, indices),
        sequence_counter=_select(source.sequence_counter, indices),
    )


def _empty_interpolated_gps_data() -> GpsData:
    """Build an empty derived GPS batch with interpolation's fixed schema."""
    return GpsData(
        align_timestamps_ns=np.empty(0, dtype=np.int64),
        sensor_timestamps_ns=np.empty(0, dtype=np.int64),
        latitude_deg=np.empty(0, dtype=np.float64),
        longitude_deg=np.empty(0, dtype=np.float64),
        altitude_m=np.empty(0, dtype=np.float64),
        position_valid=np.empty((0, 3), dtype=np.bool_),
        source_bracket_timestamps_ns=np.empty((0, 2), dtype=np.int64),
    )


def align_gps_interpolated(
    source: GpsData,
    requested_timestamps_ns: npt.NDArray[np.int64],
    *,
    policy: GpsInterpolationPolicy,
) -> GpsData:
    """Derive WGS-84 GPS positions for requested timestamps by ECEF interpolation.

    Every requested timestamp needs either an exact valid source fix or valid
    lower and upper source brackets. The splitter aligns against a whole session
    rather than a clip, so a request at a clip boundary has both. The function
    never extrapolates: a missing required support fix raises
    :attr:`~cosmos_curator.core.sensors.exceptions.AlignmentFailureReason.GPS_MISSING_BRACKET`.
    Derived rows intentionally contain no source-only optional metadata; their
    bracket provenance records the canonical source times used for the position.
    """
    if not isinstance(policy, GpsInterpolationPolicy):
        msg = f"policy must be GpsInterpolationPolicy, got {type(policy).__name__}"  # type: ignore[unreachable]
        raise TypeError(msg)

    require_1d("requested_timestamps_ns", requested_timestamps_ns, np.int64)
    require_strictly_increasing("requested_timestamps_ns", requested_timestamps_ns)
    if not len(requested_timestamps_ns):
        return _empty_interpolated_gps_data()

    source_timestamps_ns = source.align_timestamps_ns
    right_indices = np.searchsorted(source_timestamps_ns, requested_timestamps_ns, side="left")
    right_in_bounds = right_indices < len(source_timestamps_ns)
    exact_matches = np.zeros(len(requested_timestamps_ns), dtype=np.bool_)
    exact_matches[right_in_bounds] = (
        source_timestamps_ns[right_indices[right_in_bounds]] == requested_timestamps_ns[right_in_bounds]
    )
    missing_bracket = ~exact_matches & ((right_indices == 0) | ~right_in_bounds)
    if np.any(missing_bracket):
        failing_index = int(np.flatnonzero(missing_bracket)[0])
        requested_timestamp_ns = int(requested_timestamps_ns[failing_index])
        # Report the source bounds rather than the whole timeline, which a
        # session-scale source makes far too large to copy into an exception.
        source_bounds_ns = source_timestamps_ns[[0, -1]] if len(source_timestamps_ns) else source_timestamps_ns
        msg = f"GPS interpolation would require extrapolation for requested={requested_timestamp_ns}"
        raise AlignmentError(
            AlignmentFailureReason.GPS_MISSING_BRACKET,
            msg,
            align_timestamps_ns=requested_timestamps_ns[[failing_index]],
            known_source_timestamps_ns=source_bounds_ns,
        )

    lower_indices = right_indices - 1
    lower_indices[exact_matches] = right_indices[exact_matches]
    upper_indices = right_indices
    bracket_durations_ns = timestamp_distance_ns(
        source_timestamps_ns[upper_indices], source_timestamps_ns[lower_indices]
    )

    if policy.max_gap_ns is not None and len(bracket_durations_ns):
        worst_index = int(bracket_durations_ns.argmax())
        worst_gap_ns = int(bracket_durations_ns[worst_index])
        if worst_gap_ns > policy.max_gap_ns:
            msg = (
                f"max_gap_ns={policy.max_gap_ns} exceeded: max gap was {worst_gap_ns} ns "
                f"for requested={int(requested_timestamps_ns[worst_index])}"
            )
            raise AlignmentError(
                AlignmentFailureReason.GPS_MAX_GAP_EXCEEDED,
                msg,
                align_timestamps_ns=requested_timestamps_ns[[worst_index]],
                known_source_timestamps_ns=source_timestamps_ns[
                    [lower_indices[worst_index], upper_indices[worst_index]]
                ],
                gap_ns=worst_gap_ns,
                max_gap_ns=policy.max_gap_ns,
            )

    support_valid = np.all(source.position_valid[lower_indices], axis=1) & np.all(
        source.position_valid[upper_indices], axis=1
    )
    support_valid &= np.isfinite(source.latitude_deg[lower_indices])
    support_valid &= np.isfinite(source.longitude_deg[lower_indices])
    support_valid &= np.isfinite(source.altitude_m[lower_indices])
    support_valid &= np.isfinite(source.latitude_deg[upper_indices])
    support_valid &= np.isfinite(source.longitude_deg[upper_indices])
    support_valid &= np.isfinite(source.altitude_m[upper_indices])
    if not np.all(support_valid):
        failing_index = int(np.flatnonzero(~support_valid)[0])
        msg = (
            "GPS interpolation support has an invalid position for "
            f"requested={int(requested_timestamps_ns[failing_index])}"
        )
        raise AlignmentError(
            AlignmentFailureReason.GPS_INVALID_SUPPORT_POSITION,
            msg,
            align_timestamps_ns=requested_timestamps_ns[[failing_index]],
            known_source_timestamps_ns=source_timestamps_ns[
                [lower_indices[failing_index], upper_indices[failing_index]]
            ],
        )

    lower_x_m, lower_y_m, lower_z_m = _wgs84_to_ecef(
        source.longitude_deg[lower_indices],
        source.latitude_deg[lower_indices],
        source.altitude_m[lower_indices],
    )
    upper_x_m, upper_y_m, upper_z_m = _wgs84_to_ecef(
        source.longitude_deg[upper_indices],
        source.latitude_deg[upper_indices],
        source.altitude_m[upper_indices],
    )
    fractions = np.zeros(len(requested_timestamps_ns), dtype=np.float64)
    interpolated = ~exact_matches
    fractions[interpolated] = timestamp_distance_ns(
        requested_timestamps_ns[interpolated], source_timestamps_ns[lower_indices[interpolated]]
    ).astype(np.float64) / bracket_durations_ns[interpolated].astype(np.float64)
    longitude_deg, latitude_deg, altitude_m = _ecef_to_wgs84(
        lower_x_m + fractions * (upper_x_m - lower_x_m),
        lower_y_m + fractions * (upper_y_m - lower_y_m),
        lower_z_m + fractions * (upper_z_m - lower_z_m),
    )
    latitude_deg[exact_matches] = source.latitude_deg[lower_indices[exact_matches]]
    longitude_deg[exact_matches] = source.longitude_deg[lower_indices[exact_matches]]
    altitude_m[exact_matches] = source.altitude_m[lower_indices[exact_matches]]

    return GpsData(
        align_timestamps_ns=np.array(requested_timestamps_ns, copy=True),
        sensor_timestamps_ns=np.array(requested_timestamps_ns, copy=True),
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        altitude_m=altitude_m,
        position_valid=np.ones((len(requested_timestamps_ns), 3), dtype=np.bool_),
        source_bracket_timestamps_ns=np.column_stack(
            (source_timestamps_ns[lower_indices], source_timestamps_ns[upper_indices])
        ).astype(np.int64),
    )
