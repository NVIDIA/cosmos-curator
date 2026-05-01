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

"""Extrinsics data structures for cosmos_curator.core.sensors package."""

from typing import TYPE_CHECKING, Any

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.utils.helpers import as_readonly_view

if TYPE_CHECKING:
    AttrsAttribute = attrs.Attribute[Any]
else:
    AttrsAttribute = attrs.Attribute

_EXTRINSICS_SHAPE = (4, 4)


def _float64_matrix_4x4(
    _instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.float64],
) -> None:
    """Validate a homogeneous transform matrix."""
    if value.dtype != np.float64:
        msg = f"{attribute.name} must have dtype float64, got {value.dtype}"
        raise ValueError(msg)
    if value.shape != _EXTRINSICS_SHAPE:
        msg = f"{attribute.name} must have shape {_EXTRINSICS_SHAPE}, got {value.shape}"
        raise ValueError(msg)


@attrs.define(hash=False, frozen=True)
class SensorExtrinsics:
    """Rigid transform from a sensor frame to a caller-defined reference frame.

    The matrix is a row-major homogeneous transform with the expected structure::

        [R | t]
        [0 | 1]

    where ``R`` is a ``3x3`` rotation matrix and ``t`` is a ``3x1`` translation
    vector. The reference frame is caller-defined, for example a vehicle body or
    rig frame.
    """

    __hash__ = None  # type: ignore[assignment]

    matrix: npt.NDArray[np.float64] = attrs.field(
        converter=as_readonly_view,
        validator=_float64_matrix_4x4,
    )
