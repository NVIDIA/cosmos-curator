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
"""Validation helpers for sensor-library data structures and algorithms."""

from typing import TYPE_CHECKING, Any

import attrs
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    AttrsAttribute = attrs.Attribute[Any]
else:
    AttrsAttribute = attrs.Attribute


def _require_1d_int64(name: str, values: npt.NDArray[np.int64]) -> None:
    """Raise if *values* is not a 1-D ``int64`` array."""
    if values.ndim != 1:
        msg = f"{name} must be 1-D, got ndim={values.ndim}"
        raise ValueError(msg)
    if values.dtype != np.int64:
        msg = f"{name} must have dtype int64, got {values.dtype}"
        raise ValueError(msg)


def _require_1d_bool(name: str, values: npt.NDArray[np.bool_]) -> None:
    """Raise if *values* is not a 1-D ``bool`` array."""
    if values.ndim != 1:
        msg = f"{name} must be 1-D, got ndim={values.ndim}"
        raise ValueError(msg)
    if values.dtype != np.bool_:
        msg = f"{name} must have dtype bool, got {values.dtype}"
        raise ValueError(msg)


def _require_1d_uint8(name: str, values: npt.NDArray[np.uint8]) -> None:
    """Raise if *values* is not a 1-D ``uint8`` array."""
    if values.ndim != 1:
        msg = f"{name} must be 1-D, got ndim={values.ndim}"
        raise ValueError(msg)
    if values.dtype != np.uint8:
        msg = f"{name} must have dtype uint8, got {values.dtype}"
        raise ValueError(msg)


def _require_1d_uint16(name: str, values: npt.NDArray[np.uint16]) -> None:
    """Raise if *values* is not a 1-D ``uint16`` array."""
    if values.ndim != 1:
        msg = f"{name} must be 1-D, got ndim={values.ndim}"
        raise ValueError(msg)
    if values.dtype != np.uint16:
        msg = f"{name} must have dtype uint16, got {values.dtype}"
        raise ValueError(msg)


def _require_1d_uint64(name: str, values: npt.NDArray[np.uint64]) -> None:
    """Raise if *values* is not a 1-D ``uint64`` array."""
    if values.ndim != 1:
        msg = f"{name} must be 1-D, got ndim={values.ndim}"
        raise ValueError(msg)
    if values.dtype != np.uint64:
        msg = f"{name} must have dtype uint64, got {values.dtype}"
        raise ValueError(msg)


def require_finite_float64_array(name: str, values: npt.NDArray[np.float64]) -> None:
    """Raise if *values* is not a finite ``float64`` array."""
    if values.dtype != np.float64:
        msg = f"{name} must have dtype float64, got {values.dtype}"
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)


def require_strictly_increasing(name: str, values: npt.NDArray[np.int64]) -> None:
    """Raise if *values* is not strictly sorted in ascending order."""
    if len(values) > 1 and not np.all(values[:-1] < values[1:]):
        msg = f"{name} must be strictly sorted in ascending order with no duplicates"
        raise ValueError(msg)


def require_nondecreasing(name: str, values: npt.NDArray[np.int64]) -> None:
    """Raise if *values* is not sorted in ascending order allowing duplicates."""
    if len(values) > 1 and not np.all(values[:-1] <= values[1:]):
        msg = f"{name} must be sorted in ascending order"
        raise ValueError(msg)


def strictly_increasing_int64_array(
    instance: object,  # noqa: ARG001
    attribute: AttrsAttribute,
    value: npt.NDArray[np.int64],
) -> None:
    """Attrs validator for a 1-D strictly increasing ``int64`` array."""
    _require_1d_int64(attribute.name, value)
    require_strictly_increasing(attribute.name, value)


def nondecreasing_int64_array(
    instance: object,  # noqa: ARG001
    attribute: AttrsAttribute,
    value: npt.NDArray[np.int64],
) -> None:
    """Attrs validator for a 1-D nondecreasing ``int64`` array."""
    _require_1d_int64(attribute.name, value)
    require_nondecreasing(attribute.name, value)


def int64_array(
    instance: object,  # noqa: ARG001
    attribute: AttrsAttribute,
    value: npt.NDArray[np.int64],
) -> None:
    """Attrs validator for a 1-D ``int64`` array."""
    _require_1d_int64(attribute.name, value)


def bool_array(
    instance: object,  # noqa: ARG001
    attribute: AttrsAttribute,
    value: npt.NDArray[np.bool_],
) -> None:
    """Attrs validator for a 1-D ``bool`` array."""
    _require_1d_bool(attribute.name, value)


def uint8_array(
    instance: object,  # noqa: ARG001
    attribute: AttrsAttribute,
    value: npt.NDArray[np.uint8],
) -> None:
    """Attrs validator for a 1-D ``uint8`` array."""
    _require_1d_uint8(attribute.name, value)


def uint16_array(
    instance: object,  # noqa: ARG001
    attribute: AttrsAttribute,
    value: npt.NDArray[np.uint16],
) -> None:
    """Attrs validator for a 1-D ``uint16`` array."""
    _require_1d_uint16(attribute.name, value)


def uint64_array(
    instance: object,  # noqa: ARG001
    attribute: AttrsAttribute,
    value: npt.NDArray[np.uint64],
) -> None:
    """Attrs validator for a 1-D ``uint64`` array."""
    _require_1d_uint64(attribute.name, value)


def finite_float64_array(
    instance: object,  # noqa: ARG001
    attribute: AttrsAttribute,
    value: npt.NDArray[np.float64],
) -> None:
    """Attrs validator for a finite ``float64`` array."""
    require_finite_float64_array(attribute.name, value)


def uint8_frame_batch(
    instance: object,  # noqa: ARG001
    attribute: AttrsAttribute,
    value: npt.NDArray[np.uint8],
) -> None:
    """Attrs validator for a 4-D ``uint8`` frame batch."""
    frame_ndim = 4
    if value.ndim != frame_ndim:
        msg = f"{attribute.name} must be 4-D with shape (N, H, W, 3), got ndim={value.ndim}"
        raise ValueError(msg)
    if value.dtype != np.uint8:
        msg = f"{attribute.name} must have dtype uint8, got {value.dtype}"
        raise ValueError(msg)


def positive_value(
    instance: object,  # noqa: ARG001
    attribute: AttrsAttribute,
    value: int,
) -> None:
    """Attrs validator ensuring a value is positive."""
    if value <= 0:
        msg = f"{attribute.name} must be positive, got {value=}"
        raise ValueError(msg)
