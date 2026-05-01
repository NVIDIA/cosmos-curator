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
"""IMU data structures for cosmos_curator.core.sensors package."""

from typing import TYPE_CHECKING, Any, Protocol

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.utils.helpers import as_readonly_view
from cosmos_curator.core.sensors.utils.validation import (
    nondecreasing_int64_array,
    require_finite_float64_array,
    strictly_increasing_int64_array,
    uint64_array,
)

if TYPE_CHECKING:
    AttrsAttribute = attrs.Attribute[Any]
else:
    AttrsAttribute = attrs.Attribute

_VECTOR_COLUMNS = 3
_VECTOR_BATCH_NDIM = 2
_QUATERNION_COLUMNS = 4
_COVARIANCE_NDIM = 3
_QUATERNION_NORM_TOLERANCE = 1e-6
_COVARIANCE_TOLERANCE = 1e-9


class _HasImuBatchFields(Protocol):
    align_timestamps_ns: npt.NDArray[np.int64]
    sensor_timestamps_ns: npt.NDArray[np.int64]
    angular_velocity_rad_s: npt.NDArray[np.float64]
    linear_acceleration_m_s2: npt.NDArray[np.float64]
    orientation_quat_xyzw: npt.NDArray[np.float64] | None
    angular_velocity_covariance: npt.NDArray[np.float64] | None
    linear_acceleration_covariance: npt.NDArray[np.float64] | None
    orientation_covariance: npt.NDArray[np.float64] | None
    angular_velocity_valid: npt.NDArray[np.bool_] | None
    linear_acceleration_valid: npt.NDArray[np.bool_] | None
    orientation_valid: npt.NDArray[np.bool_] | None
    host_timestamps_ns: npt.NDArray[np.int64] | None
    sequence_counter: npt.NDArray[np.uint64] | None
    temperature_c: npt.NDArray[np.float64] | None


def _as_optional_readonly_view(array: npt.NDArray[Any] | None) -> npt.NDArray[Any] | None:
    """Return a read-only view of ``array`` while preserving ``None``."""
    if array is None:
        return None
    return as_readonly_view(array)


def _require_bool_array(name: str, value: npt.NDArray[np.bool_]) -> None:
    """Raise if ``value`` is not a ``bool`` array."""
    if value.dtype != np.bool_:
        msg = f"{name} must have dtype bool, got {value.dtype}"
        raise ValueError(msg)


def _require_int64_vector(name: str, value: npt.NDArray[np.int64]) -> None:
    """Raise if ``value`` is not a 1-D ``int64`` vector."""
    if value.ndim != 1:
        msg = f"{name} must have shape (N,), got shape={value.shape}"
        raise ValueError(msg)
    if value.dtype != np.int64:
        msg = f"{name} must have dtype int64, got {value.dtype}"
        raise ValueError(msg)


def _require_float64_vector(name: str, value: npt.NDArray[np.float64]) -> None:
    """Raise if ``value`` is not a finite 1-D ``float64`` vector."""
    if value.ndim != 1:
        msg = f"{name} must have shape (N,), got shape={value.shape}"
        raise ValueError(msg)
    require_finite_float64_array(name, value)


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


def _optional_quaternion_batch(
    _instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.float64] | None,
) -> None:
    """Validate optional orientation quaternions with shape ``(N, 4)``."""
    if value is None:
        return
    if value.ndim != _VECTOR_BATCH_NDIM or value.shape[1:] != (_QUATERNION_COLUMNS,):
        msg = f"{attribute.name} must have shape (N, 4), got shape={value.shape}"
        raise ValueError(msg)
    require_finite_float64_array(attribute.name, value)
    norms = np.linalg.norm(value, axis=1)
    if not np.all(np.isclose(norms, 1.0, rtol=0.0, atol=_QUATERNION_NORM_TOLERANCE)):
        msg = f"{attribute.name} quaternion rows must have unit norm within tolerance"
        raise ValueError(msg)


def _optional_covariance_batch(
    _instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.float64] | None,
) -> None:
    """Validate optional covariance matrices with shape ``(N, 3, 3)``."""
    if value is None:
        return
    if value.ndim != _COVARIANCE_NDIM or value.shape[1:] != (_VECTOR_COLUMNS, _VECTOR_COLUMNS):
        msg = f"{attribute.name} must have shape (N, 3, 3), got shape={value.shape}"
        raise ValueError(msg)
    require_finite_float64_array(attribute.name, value)
    if not np.allclose(value, np.swapaxes(value, 1, 2), rtol=_COVARIANCE_TOLERANCE, atol=_COVARIANCE_TOLERANCE):
        msg = f"{attribute.name} covariance matrices must be symmetric"
        raise ValueError(msg)
    if value.shape[0] and np.min(np.linalg.eigvalsh(value)) < -_COVARIANCE_TOLERANCE:
        msg = f"{attribute.name} covariance matrices must be positive semidefinite"
        raise ValueError(msg)


def _optional_per_axis_validity_mask(
    _instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.bool_] | None,
) -> None:
    """Validate optional per-axis validity masks with shape ``(N, 3)``."""
    if value is None:
        return
    if value.ndim != _VECTOR_BATCH_NDIM or value.shape[1:] != (_VECTOR_COLUMNS,):
        msg = f"{attribute.name} must have shape (N, 3), got shape={value.shape}"
        raise ValueError(msg)
    _require_bool_array(attribute.name, value)


def _optional_row_validity_mask(
    _instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.bool_] | None,
) -> None:
    """Validate optional row-level validity masks with shape ``(N,)``."""
    if value is None:
        return
    if value.ndim != 1:
        msg = f"{attribute.name} must have shape (N,), got shape={value.shape}"
        raise ValueError(msg)
    _require_bool_array(attribute.name, value)


def _optional_int64_vector(
    _instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.int64] | None,
) -> None:
    """Validate optional 1-D ``int64`` arrays."""
    if value is None:
        return
    _require_int64_vector(attribute.name, value)


def _optional_uint64_vector(
    instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.uint64] | None,
) -> None:
    """Validate optional 1-D ``uint64`` arrays."""
    if value is None:
        return
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


def _batch_lengths(
    instance: _HasImuBatchFields,
    _attribute: object,
    _value: npt.NDArray[np.float64] | None,
) -> None:
    """Validate shared row-count invariants across IMU batch arrays."""
    expected_len = len(instance.align_timestamps_ns)
    lengths = {
        "align_timestamps_ns": len(instance.align_timestamps_ns),
        "sensor_timestamps_ns": len(instance.sensor_timestamps_ns),
        "angular_velocity_rad_s": len(instance.angular_velocity_rad_s),
        "linear_acceleration_m_s2": len(instance.linear_acceleration_m_s2),
    }
    optional_fields = (
        "orientation_quat_xyzw",
        "angular_velocity_covariance",
        "linear_acceleration_covariance",
        "orientation_covariance",
        "angular_velocity_valid",
        "linear_acceleration_valid",
        "orientation_valid",
        "host_timestamps_ns",
        "sequence_counter",
        "temperature_c",
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
class ImuData:
    """IMU point samples stored as structure-of-arrays batches.

    Satisfies ``SensorData`` (``cosmos_curator.core.sensors.data.sensor_data``).
    Required vector fields use SI units in the IMU sensor frame.
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
    angular_velocity_rad_s: npt.NDArray[np.float64] = attrs.field(
        converter=as_readonly_view,
        validator=_float64_vector_batch,
    )
    linear_acceleration_m_s2: npt.NDArray[np.float64] = attrs.field(
        converter=as_readonly_view,
        validator=_float64_vector_batch,
    )

    orientation_quat_xyzw: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=_optional_quaternion_batch,
    )
    angular_velocity_covariance: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=_optional_covariance_batch,
    )
    linear_acceleration_covariance: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=_optional_covariance_batch,
    )
    orientation_covariance: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=_optional_covariance_batch,
    )

    angular_velocity_valid: npt.NDArray[np.bool_] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=_optional_per_axis_validity_mask,
    )
    linear_acceleration_valid: npt.NDArray[np.bool_] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=_optional_per_axis_validity_mask,
    )
    orientation_valid: npt.NDArray[np.bool_] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=_optional_row_validity_mask,
    )

    host_timestamps_ns: npt.NDArray[np.int64] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=_optional_int64_vector,
    )
    sequence_counter: npt.NDArray[np.uint64] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=_optional_uint64_vector,
    )
    # temperature_c is the last optional field; its _as_optional_readonly_view converter runs before validators.
    # attrs.validators.and_ then runs _optional_float64_vector and _batch_lengths, even when temperature_c is None.
    temperature_c: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=attrs.validators.and_(
            _optional_float64_vector,
            _batch_lengths,
        ),
    )
