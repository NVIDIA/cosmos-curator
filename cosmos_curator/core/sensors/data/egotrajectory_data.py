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
"""Quaternion-based ego-trajectory data structures."""

from typing import Protocol

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.utils.helpers import as_optional_readonly_view, as_readonly_view
from cosmos_curator.core.sensors.utils.validation import (
    bool_batch,
    float64_batch,
    nondecreasing_int64_array,
    nonempty_str,
    optional_int64_array,
    optional_uint64_array,
    strictly_increasing_int64_array,
    unit_quaternion_batch,
)

_POSITION_TAIL_SHAPE = (3,)
_POSITION_ZERO_TOLERANCE = 1e-6
_QUATERNION_TOLERANCE = 1e-6
_IDENTITY_QUATERNION_XYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
_POSITION_BATCH_VALIDATOR = float64_batch(_POSITION_TAIL_SHAPE)
_POSE_VALID_VALIDATOR = bool_batch(())


class _HasEgoBatchFields(Protocol):
    align_timestamps_ns: npt.NDArray[np.int64]
    sensor_timestamps_ns: npt.NDArray[np.int64]
    positions_m: npt.NDArray[np.float64]
    orientations_xyzw: npt.NDArray[np.float64]
    pose_valid: npt.NDArray[np.bool_]
    host_timestamps_ns: npt.NDArray[np.int64] | None
    sequence_counter: npt.NDArray[np.uint64] | None
    frame: str


def _ego_batch_lengths(
    instance: _HasEgoBatchFields,
    _attribute: object,
    _value: object,
) -> None:
    """Validate the shared row-count ``N`` invariant for :class:`EgoTrajectory`."""
    expected_len = len(instance.align_timestamps_ns)
    lengths = {
        "align_timestamps_ns": len(instance.align_timestamps_ns),
        "sensor_timestamps_ns": len(instance.sensor_timestamps_ns),
        "positions_m": len(instance.positions_m),
        "orientations_xyzw": len(instance.orientations_xyzw),
        "pose_valid": len(instance.pose_valid),
    }
    if instance.host_timestamps_ns is not None:
        lengths["host_timestamps_ns"] = len(instance.host_timestamps_ns)
    if instance.sequence_counter is not None:
        lengths["sequence_counter"] = len(instance.sequence_counter)
    if any(length != expected_len for length in lengths.values()):
        length_summary = " ".join(f"{name}={length}" for name, length in lengths.items())
        msg = f"All arrays must be the same length: {length_summary}"
        raise ValueError(msg)


def _invalid_poses_are_identity(
    instance: _HasEgoBatchFields,
    _attribute: object,
    _value: object,
) -> None:
    """Require zero-position, identity-quaternion placeholders for invalid rows."""
    invalid = ~instance.pose_valid
    if not np.any(invalid):
        return
    if not np.allclose(instance.positions_m[invalid], 0.0, rtol=0.0, atol=_POSITION_ZERO_TOLERANCE) or not np.allclose(
        instance.orientations_xyzw[invalid],
        _IDENTITY_QUATERNION_XYZW,
        rtol=0.0,
        atol=_QUATERNION_TOLERANCE,
    ):
        msg = "pose_valid=false rows must use zero position and identity quaternion"
        raise ValueError(msg)


@attrs.define(hash=False, frozen=True)
class EgoTrajectory:
    """Sensor-origin trajectory rows stored as position plus XYZW quaternion.

    Each row describes ``T_worldENU_from_sensorBody`` in a clip-local ENU
    ``frame``. ``as_homogeneous_matrices`` provides the equivalent row-major
    ``(N, 4, 4)`` transforms for matrix-based consumers.

    Quaternion signs are preserved from the source and are not guaranteed to
    be hemisphere-aligned across rows: ``q`` and ``-q`` represent the same
    rotation. Consumers that interpolate, differentiate, or otherwise operate
    directly in quaternion space must align consecutive valid rows themselves.

    Satisfies ``SensorData`` (``cosmos_curator.core.sensors.data.sensor_data``).
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
    positions_m: npt.NDArray[np.float64] = attrs.field(
        converter=as_readonly_view,
        validator=_POSITION_BATCH_VALIDATOR,
    )
    orientations_xyzw: npt.NDArray[np.float64] = attrs.field(
        converter=as_readonly_view,
        validator=unit_quaternion_batch,
    )
    pose_valid: npt.NDArray[np.bool_] = attrs.field(
        converter=as_readonly_view,
        validator=_POSE_VALID_VALIDATOR,
    )
    frame: str = attrs.field(validator=nonempty_str)
    host_timestamps_ns: npt.NDArray[np.int64] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=optional_int64_array,
    )
    sequence_counter: npt.NDArray[np.uint64] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=attrs.validators.and_(
            optional_uint64_array,
            _ego_batch_lengths,
            _invalid_poses_are_identity,
        ),
    )

    def as_homogeneous_matrices(self) -> npt.NDArray[np.float64]:
        """Return row-major ``T_worldENU_from_sensorBody`` matrices."""
        x, y, z, w = self.orientations_xyzw.T
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        matrices = np.zeros((len(self.positions_m), 4, 4), dtype=np.float64)
        matrices[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
        matrices[:, 0, 1] = 2.0 * (xy - wz)
        matrices[:, 0, 2] = 2.0 * (xz + wy)
        matrices[:, 1, 0] = 2.0 * (xy + wz)
        matrices[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
        matrices[:, 1, 2] = 2.0 * (yz - wx)
        matrices[:, 2, 0] = 2.0 * (xz - wy)
        matrices[:, 2, 1] = 2.0 * (yz + wx)
        matrices[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
        matrices[:, :3, 3] = self.positions_m
        matrices[:, 3, 3] = 1.0
        return as_readonly_view(matrices)
