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
"""GPS/GNSS data structures for the Sensor Library."""

import enum
from typing import TYPE_CHECKING, Any, Protocol

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.utils.helpers import as_readonly_view
from cosmos_curator.core.sensors.utils.validation import (
    int64_array,
    nondecreasing_int64_array,
    require_finite_float64_array,
    strictly_increasing_int64_array,
    uint8_array,
    uint16_array,
    uint64_array,
)

if TYPE_CHECKING:
    AttrsAttribute = attrs.Attribute[Any]
else:
    AttrsAttribute = attrs.Attribute

_VECTOR_BATCH_NDIM = 2
_VECTOR_COLUMNS = 3
_COVARIANCE_BATCH_NDIM = 3
_COVARIANCE_COLUMNS = 3
_COVARIANCE_SYMMETRY_ATOL = 1e-9
_COVARIANCE_PSD_ATOL = 1e-9
_MIN_LATITUDE_DEG = -90.0
_MAX_LATITUDE_DEG = 90.0
_MIN_LONGITUDE_DEG = -180.0
_MAX_LONGITUDE_DEG = 180.0


class GpsFixType(enum.IntEnum):
    """Normalized GPS/GNSS fix type values stored in ``GpsData.fix_type``."""

    NO_FIX_OR_UNKNOWN = 0
    FIX_2D = 2
    FIX_3D = 3
    DIFFERENTIAL = 4
    RTK_FLOAT = 5
    RTK_FIXED = 6
    EXTRAPOLATED = 8


_VALID_FIX_TYPES = frozenset(fix_type.value for fix_type in GpsFixType)
_VALID_FIX_TYPES_ARRAY = np.asarray(sorted(_VALID_FIX_TYPES), dtype=np.uint8)


class _HasGpsBatchFields(Protocol):
    align_timestamps_ns: npt.NDArray[np.int64]
    sensor_timestamps_ns: npt.NDArray[np.int64]
    latitude_deg: npt.NDArray[np.float64]
    longitude_deg: npt.NDArray[np.float64]
    altitude_m: npt.NDArray[np.float64]
    position_valid: npt.NDArray[np.bool_]
    position_covariance_enu_m2: npt.NDArray[np.float64] | None
    velocity_enu_m_s: npt.NDArray[np.float64] | None
    velocity_valid: npt.NDArray[np.bool_] | None
    fix_type: npt.NDArray[np.uint8] | None
    satellites_used: npt.NDArray[np.uint16] | None
    horizontal_accuracy_m: npt.NDArray[np.float64] | None
    vertical_accuracy_m: npt.NDArray[np.float64] | None
    hdop: npt.NDArray[np.float64] | None
    vdop: npt.NDArray[np.float64] | None
    pdop: npt.NDArray[np.float64] | None
    host_timestamps_ns: npt.NDArray[np.int64] | None
    utc_timestamps_ns: npt.NDArray[np.int64] | None
    sequence_counter: npt.NDArray[np.uint64] | None


def _as_optional_readonly_view(array: npt.NDArray[Any] | None) -> npt.NDArray[Any] | None:
    """Return a read-only view of ``array`` while preserving ``None``."""
    if array is None:
        return None
    return as_readonly_view(array)


def _require_float64_vector(name: str, value: npt.NDArray[np.float64]) -> None:
    """Raise if ``value`` is not a finite 1-D ``float64`` vector."""
    if value.ndim != 1:
        msg = f"{name} must have shape (N,), got shape={value.shape}"
        raise ValueError(msg)
    require_finite_float64_array(name, value)


def _require_nonnegative_float64_vector(name: str, value: npt.NDArray[np.float64]) -> None:
    """Raise if ``value`` is not a finite nonnegative 1-D ``float64`` vector."""
    _require_float64_vector(name, value)
    if np.any(value < 0):
        msg = f"{name} must contain only nonnegative values"
        raise ValueError(msg)


def _latitude_vector(
    _instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.float64],
) -> None:
    """Validate WGS-84 latitude values in degrees."""
    _require_float64_vector(attribute.name, value)
    if np.any((value < _MIN_LATITUDE_DEG) | (value > _MAX_LATITUDE_DEG)):
        msg = f"{attribute.name} latitude values must be in [-90.0, 90.0]"
        raise ValueError(msg)


def _longitude_vector(
    _instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.float64],
) -> None:
    """Validate WGS-84 longitude values in degrees."""
    _require_float64_vector(attribute.name, value)
    if np.any((value < _MIN_LONGITUDE_DEG) | (value > _MAX_LONGITUDE_DEG)):
        msg = f"{attribute.name} longitude values must be in [-180.0, 180.0]"
        raise ValueError(msg)


def _float64_vector(
    _instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.float64],
) -> None:
    """Validate a finite 1-D ``float64`` array."""
    _require_float64_vector(attribute.name, value)


def _float64_vector_batch(
    _instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.float64],
) -> None:
    """Validate a finite ``float64`` vector batch with shape ``(N, 3)``."""
    if value.ndim != _VECTOR_BATCH_NDIM or value.shape[1:] != (_VECTOR_COLUMNS,):
        msg = f"{attribute.name} must have shape (N, 3), got shape={value.shape}"
        raise ValueError(msg)
    require_finite_float64_array(attribute.name, value)


def _per_axis_validity_mask(
    _instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.bool_],
) -> None:
    """Validate a per-axis validity mask with shape ``(N, 3)``."""
    if value.ndim != _VECTOR_BATCH_NDIM or value.shape[1:] != (_VECTOR_COLUMNS,):
        msg = f"{attribute.name} must have shape (N, 3), got shape={value.shape}"
        raise ValueError(msg)
    if value.dtype != np.bool_:
        msg = f"{attribute.name} must have dtype bool, got {value.dtype}"
        raise ValueError(msg)


def _optional_covariance_batch(
    _instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.float64] | None,
) -> None:
    """Validate optional covariance matrices with shape ``(N, 3, 3)``."""
    if value is None:
        return
    if value.ndim != _COVARIANCE_BATCH_NDIM or value.shape[1:] != (_COVARIANCE_COLUMNS, _COVARIANCE_COLUMNS):
        msg = f"{attribute.name} must have shape (N, 3, 3), got shape={value.shape}"
        raise ValueError(msg)
    require_finite_float64_array(attribute.name, value)
    if not np.allclose(value, np.swapaxes(value, 1, 2), atol=_COVARIANCE_SYMMETRY_ATOL, rtol=0.0):
        msg = f"{attribute.name} covariance matrices must be symmetric"
        raise ValueError(msg)
    min_eigenvalue = np.min(np.linalg.eigvalsh(value)) if len(value) > 0 else 0.0
    if min_eigenvalue < -_COVARIANCE_PSD_ATOL:
        msg = f"{attribute.name} covariance matrices must be positive semidefinite"
        raise ValueError(msg)


def _optional_vector_batch(
    instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.float64] | None,
) -> None:
    """Validate optional finite ``float64`` vector batches."""
    if value is None:
        return
    _float64_vector_batch(instance, attribute, value)


def _optional_per_axis_validity_mask(
    instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.bool_] | None,
) -> None:
    """Validate optional per-axis validity masks."""
    if value is None:
        return
    _per_axis_validity_mask(instance, attribute, value)


def _optional_uint8_vector(
    instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.uint8] | None,
) -> None:
    """Validate optional 1-D ``uint8`` arrays."""
    if value is None:
        return
    if value.ndim != 1:
        msg = f"{attribute.name} must have shape (N,), got shape={value.shape}"
        raise ValueError(msg)
    uint8_array(instance, attribute, value)


def _optional_fix_type(
    instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.uint8] | None,
) -> None:
    """Validate optional normalized GPS/GNSS fix type arrays."""
    if value is None:
        return
    _optional_uint8_vector(instance, attribute, value)
    if not np.all(np.isin(value, _VALID_FIX_TYPES_ARRAY)):
        msg = f"{attribute.name} must contain only valid fix type values: {sorted(_VALID_FIX_TYPES)}"
        raise ValueError(msg)


def _optional_uint16_vector(
    instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.uint16] | None,
) -> None:
    """Validate optional 1-D ``uint16`` arrays."""
    if value is None:
        return
    if value.ndim != 1:
        msg = f"{attribute.name} must have shape (N,), got shape={value.shape}"
        raise ValueError(msg)
    uint16_array(instance, attribute, value)


def _optional_int64_vector(
    instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.int64] | None,
) -> None:
    """Validate optional 1-D ``int64`` arrays."""
    if value is None:
        return
    if value.ndim != 1:
        msg = f"{attribute.name} must have shape (N,), got shape={value.shape}"
        raise ValueError(msg)
    int64_array(instance, attribute, value)


def _optional_uint64_vector(
    instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.uint64] | None,
) -> None:
    """Validate optional 1-D ``uint64`` arrays."""
    if value is None:
        return
    if value.ndim != 1:
        msg = f"{attribute.name} must have shape (N,), got shape={value.shape}"
        raise ValueError(msg)
    uint64_array(instance, attribute, value)


def _optional_float64_vector(
    _instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.float64] | None,
) -> None:
    """Validate optional finite 1-D ``float64`` arrays."""
    if value is None:
        return
    _require_float64_vector(attribute.name, value)


def _optional_nonnegative_float64_vector(
    _instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.float64] | None,
) -> None:
    """Validate optional finite nonnegative 1-D ``float64`` arrays."""
    if value is None:
        return
    _require_nonnegative_float64_vector(attribute.name, value)


def _batch_lengths(
    instance: _HasGpsBatchFields,
    _attribute: object,
    _value: object,
) -> None:
    """Validate shared row-count invariants across GPS/GNSS batch arrays."""
    expected_len = len(instance.align_timestamps_ns)
    lengths = {
        "align_timestamps_ns": len(instance.align_timestamps_ns),
        "sensor_timestamps_ns": len(instance.sensor_timestamps_ns),
        "latitude_deg": len(instance.latitude_deg),
        "longitude_deg": len(instance.longitude_deg),
        "altitude_m": len(instance.altitude_m),
        "position_valid": len(instance.position_valid),
    }
    optional_fields = (
        "position_covariance_enu_m2",
        "velocity_enu_m_s",
        "velocity_valid",
        "fix_type",
        "satellites_used",
        "horizontal_accuracy_m",
        "vertical_accuracy_m",
        "hdop",
        "vdop",
        "pdop",
        "host_timestamps_ns",
        "utc_timestamps_ns",
        "sequence_counter",
    )
    for field_name in optional_fields:
        field_value = getattr(instance, field_name)
        if field_value is not None:
            lengths[field_name] = len(field_value)
    if any(length != expected_len for length in lengths.values()):
        length_summary = " ".join(f"{name}={length}" for name, length in lengths.items())
        msg = f"All arrays must be the same length: {length_summary}"
        raise ValueError(msg)


@attrs.define(hash=False, frozen=True)
class GpsData:
    """GPS/GNSS fix samples stored as structure-of-arrays batches.

    Satisfies ``SensorData`` (``cosmos_curator.core.sensors.data.sensor_data``).
    Required position fields use WGS-84 geodetic coordinates.
    """

    __hash__ = None  # type: ignore[assignment]

    align_timestamps_ns: npt.NDArray[np.int64] = attrs.field(
        converter=as_readonly_view,
        validator=strictly_increasing_int64_array,
    )
    sensor_timestamps_ns: npt.NDArray[np.int64] = attrs.field(
        converter=as_readonly_view,
        validator=nondecreasing_int64_array,
    )
    latitude_deg: npt.NDArray[np.float64] = attrs.field(
        converter=as_readonly_view,
        validator=_latitude_vector,
    )
    longitude_deg: npt.NDArray[np.float64] = attrs.field(
        converter=as_readonly_view,
        validator=_longitude_vector,
    )
    altitude_m: npt.NDArray[np.float64] = attrs.field(
        converter=as_readonly_view,
        validator=_float64_vector,
    )
    position_valid: npt.NDArray[np.bool_] = attrs.field(
        converter=as_readonly_view,
        validator=_per_axis_validity_mask,
    )

    position_covariance_enu_m2: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=_optional_covariance_batch,
    )
    velocity_enu_m_s: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=_optional_vector_batch,
    )
    velocity_valid: npt.NDArray[np.bool_] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=_optional_per_axis_validity_mask,
    )
    fix_type: npt.NDArray[np.uint8] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=_optional_fix_type,
    )
    satellites_used: npt.NDArray[np.uint16] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=_optional_uint16_vector,
    )
    horizontal_accuracy_m: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=_optional_nonnegative_float64_vector,
    )
    vertical_accuracy_m: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=_optional_nonnegative_float64_vector,
    )
    hdop: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=_optional_nonnegative_float64_vector,
    )
    vdop: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=_optional_nonnegative_float64_vector,
    )
    pdop: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=_optional_nonnegative_float64_vector,
    )
    host_timestamps_ns: npt.NDArray[np.int64] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=_optional_int64_vector,
    )
    utc_timestamps_ns: npt.NDArray[np.int64] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=_optional_int64_vector,
    )
    # sequence_counter is the final field so _batch_lengths runs after attrs has
    # set and validated every required and optional GPS/GNSS array.
    sequence_counter: npt.NDArray[np.uint64] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=attrs.validators.and_(
            _optional_uint64_vector,
            _batch_lengths,
        ),
    )
