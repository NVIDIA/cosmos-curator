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

"""Intrinsics data structures for cosmos_curator.core.sensors package."""

from typing import TYPE_CHECKING, Any, Final

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.utils.helpers import as_readonly_view

if TYPE_CHECKING:
    AttrsAttribute = attrs.Attribute[Any]
else:
    AttrsAttribute = attrs.Attribute

_CAMERA_MATRIX_SHAPE: Final[tuple[int, int]] = (3, 3)
_SUPPORTED_DISTORTION_MODELS: Final[frozenset[str]] = frozenset({"brown_conrady", "fisheye", "none"})


def _float64_camera_matrix(
    _instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.float64],
) -> None:
    """Validate a row-major ``3x3`` intrinsic matrix."""
    if value.dtype != np.float64:
        msg = f"{attribute.name} must have dtype float64, got {value.dtype}"
        raise ValueError(msg)
    if value.shape != _CAMERA_MATRIX_SHAPE:
        msg = f"{attribute.name} must have shape {_CAMERA_MATRIX_SHAPE}, got {value.shape}"
        raise ValueError(msg)


def _float64_distortion_coefficients(
    _instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.float64],
) -> None:
    """Validate a 1-D ``float64`` distortion vector."""
    if value.dtype != np.float64:
        msg = f"{attribute.name} must have dtype float64, got {value.dtype}"
        raise ValueError(msg)
    if value.ndim != 1:
        msg = f"{attribute.name} must be 1-D, got ndim={value.ndim}"
        raise ValueError(msg)


def _distortion_model(
    _instance: object,
    attribute: AttrsAttribute,
    value: str,
) -> None:
    """Validate the named distortion model."""
    if value not in _SUPPORTED_DISTORTION_MODELS:
        msg = f"{attribute.name} must be one of {sorted(_SUPPORTED_DISTORTION_MODELS)}, got {value!r}"
        raise ValueError(msg)


def _positive_int(
    _instance: object,
    attribute: AttrsAttribute,
    value: object,
) -> None:
    """Validate a strictly positive integer scalar."""
    if not isinstance(value, int):
        msg = f"{attribute.name} must be an int, got {type(value).__name__}"
        raise TypeError(msg)
    if value <= 0:
        msg = f"{attribute.name} must be positive, got value={value}"
        raise ValueError(msg)


@attrs.define(hash=False, frozen=True)
class CameraIntrinsics:
    """Typed camera calibration intrinsics for a specific image geometry."""

    __hash__ = None  # type: ignore[assignment]

    camera_matrix: npt.NDArray[np.float64] = attrs.field(
        converter=as_readonly_view,
        validator=_float64_camera_matrix,
    )
    distortion_coefficients: npt.NDArray[np.float64] = attrs.field(
        converter=as_readonly_view,
        validator=_float64_distortion_coefficients,
    )
    distortion_model: str = attrs.field(validator=_distortion_model)
    width: int = attrs.field(validator=_positive_int)
    height: int = attrs.field(validator=_positive_int)
