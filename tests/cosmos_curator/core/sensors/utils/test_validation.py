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
"""Unit tests for sensor validation helpers."""

import attrs
import numpy as np
import pytest

from cosmos_curator.core.sensors.utils.validation import (
    _require_1d_bool,
    _require_1d_int64,
    _require_1d_uint64,
    bool_array,
    finite_float64_array,
    int64_array,
    nondecreasing_int64_array,
    require_finite_float64_array,
    require_strictly_increasing,
    strictly_increasing_int64_array,
    uint8_array,
    uint8_frame_batch,
    uint16_array,
    uint64_array,
)


def test_require_1d_int64_accepts_int64_vector() -> None:
    """A 1-D int64 array should pass base timestamp validation."""
    values = np.array([0, 10, 20], dtype=np.int64)

    _require_1d_int64("values", values)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([[0, 10, 20]], dtype=np.int64), r"values must be 1-D, got ndim=2"),
        (np.array([0, 10, 20], dtype=np.int32), r"values must have dtype int64, got int32"),
    ],
)
def test_require_1d_int64_rejects_invalid_shape_or_dtype(values: np.ndarray, match: str) -> None:
    """Non-vector or non-int64 arrays should fail base timestamp validation."""
    with pytest.raises(ValueError, match=match):
        _require_1d_int64("values", values)


def test_require_1d_bool_accepts_bool_vector() -> None:
    """A 1-D bool array should pass base bool validation."""
    values = np.array([True, False, True], dtype=np.bool_)

    _require_1d_bool("values", values)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([[True, False]], dtype=np.bool_), r"values must be 1-D, got ndim=2"),
        (np.array([1, 0], dtype=np.int64), r"values must have dtype bool, got int64"),
    ],
)
def test_require_1d_bool_rejects_invalid_shape_or_dtype(values: np.ndarray, match: str) -> None:
    """Non-vector or non-bool arrays should fail base bool validation."""
    with pytest.raises(ValueError, match=match):
        _require_1d_bool("values", values)


def test_require_1d_uint64_accepts_uint64_vector() -> None:
    """A 1-D uint64 array should pass base unsigned-integer validation."""
    values = np.array([0, 10, 20], dtype=np.uint64)

    _require_1d_uint64("values", values)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([[0, 10, 20]], dtype=np.uint64), r"values must be 1-D, got ndim=2"),
        (np.array([0, 10, 20], dtype=np.int64), r"values must have dtype uint64, got int64"),
    ],
)
def test_require_1d_uint64_rejects_invalid_shape_or_dtype(values: np.ndarray, match: str) -> None:
    """Non-vector or non-uint64 arrays should fail base unsigned-integer validation."""
    with pytest.raises(ValueError, match=match):
        _require_1d_uint64("values", values)


def test_require_finite_float64_array_accepts_finite_float64_array() -> None:
    """A finite float64 array of any rank should pass finite-float validation."""
    values = np.zeros((2, 3), dtype=np.float64)

    require_finite_float64_array("values", values)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([0.0, 1.0], dtype=np.float32), r"values must have dtype float64, got float32"),
        (np.array([0.0, np.inf], dtype=np.float64), r"values must contain only finite values"),
        (np.array([0.0, np.nan], dtype=np.float64), r"values must contain only finite values"),
    ],
)
def test_require_finite_float64_array_rejects_invalid_values(values: np.ndarray, match: str) -> None:
    """Non-float64 or non-finite arrays should fail finite-float validation."""
    with pytest.raises(ValueError, match=match):
        require_finite_float64_array("values", values)


def test_require_strictly_increasing_accepts_sorted_values() -> None:
    """Strictly ascending arrays should pass validation."""
    values = np.array([0, 10, 20], dtype=np.int64)

    require_strictly_increasing("values", values)


@pytest.mark.parametrize(
    "values",
    [
        np.array([0, 10, 10], dtype=np.int64),
        np.array([0, 20, 10], dtype=np.int64),
    ],
)
def test_require_strictly_increasing_rejects_non_increasing_values(values: np.ndarray) -> None:
    """Duplicate or descending values should raise ValueError."""
    with pytest.raises(ValueError, match="values must be strictly sorted in ascending order with no duplicates"):
        require_strictly_increasing("values", values)


@attrs.define
class _StrictlyIncreasingArrayHolder:
    """Test fixture for the strictly increasing int64 array attrs validator."""

    values: np.ndarray = attrs.field(validator=strictly_increasing_int64_array)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([100, 200, 200], dtype=np.int64), "values must be strictly sorted in ascending order"),
        (np.array([[100, 200, 300]], dtype=np.int64), r"values must be 1-D, got ndim=2"),
        (np.array([100, 200, 300], dtype=np.int32), r"values must have dtype int64, got int32"),
    ],
)
def test_strictly_increasing_int64_array_rejects_invalid_inputs(values: np.ndarray, match: str) -> None:
    """Strict validator should reject invalid ordering, rank, or dtype."""
    with pytest.raises(ValueError, match=match):
        _StrictlyIncreasingArrayHolder(values=values)


def test_strictly_increasing_int64_array_accepts_strictly_increasing_int64_vector() -> None:
    """Strict validator should accept 1-D int64 arrays with strictly increasing values."""
    values = np.array([100, 200, 300], dtype=np.int64)
    holder = _StrictlyIncreasingArrayHolder(values=values)
    np.testing.assert_array_equal(holder.values, values)


@attrs.define
class _NondecreasingArrayHolder:
    """Test fixture for the nondecreasing int64 array attrs validator."""

    values: np.ndarray = attrs.field(validator=nondecreasing_int64_array)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([100, 300, 200], dtype=np.int64), "values must be sorted in ascending order"),
        (np.array([[100, 200, 300]], dtype=np.int64), r"values must be 1-D, got ndim=2"),
        (np.array([100, 200, 300], dtype=np.int32), r"values must have dtype int64, got int32"),
    ],
)
def test_nondecreasing_int64_array_rejects_invalid_inputs(values: np.ndarray, match: str) -> None:
    """Nondecreasing validator should reject descending values, rank mismatches, and dtype mismatches."""
    with pytest.raises(ValueError, match=match):
        _NondecreasingArrayHolder(values=values)


def test_nondecreasing_int64_array_accepts_nondecreasing_int64_vector() -> None:
    """Nondecreasing validator should allow duplicate timestamps."""
    values = np.array([100, 200, 200, 300], dtype=np.int64)
    holder = _NondecreasingArrayHolder(values=values)
    np.testing.assert_array_equal(holder.values, values)


@attrs.define
class _BoolArrayHolder:
    """Test fixture for the bool array attrs validator."""

    values: np.ndarray = attrs.field(validator=bool_array)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([[True, False]], dtype=np.bool_), r"values must be 1-D, got ndim=2"),
        (np.array([1, 0], dtype=np.int64), r"values must have dtype bool, got int64"),
    ],
)
def test_bool_array_rejects_invalid_inputs(values: np.ndarray, match: str) -> None:
    """Bool-array validator should reject non-vector or non-bool arrays."""
    with pytest.raises(ValueError, match=match):
        _BoolArrayHolder(values=values)


def test_bool_array_accepts_bool_vector() -> None:
    """Bool-array validator should accept a 1-D bool array."""
    values = np.array([True, False], dtype=np.bool_)
    holder = _BoolArrayHolder(values=values)
    np.testing.assert_array_equal(holder.values, values)


@attrs.define
class _Int64ArrayHolder:
    """Test fixture for the int64 array attrs validator."""

    values: np.ndarray = attrs.field(validator=int64_array)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([[1, 2]], dtype=np.int64), r"values must be 1-D, got ndim=2"),
        (np.array([1, 2], dtype=np.int32), r"values must have dtype int64, got int32"),
    ],
)
def test_int64_array_rejects_invalid_inputs(values: np.ndarray, match: str) -> None:
    """Int64-array validator should reject non-vector or non-int64 arrays."""
    with pytest.raises(ValueError, match=match):
        _Int64ArrayHolder(values=values)


def test_int64_array_accepts_int64_vector() -> None:
    """Int64-array validator should accept a 1-D int64 array."""
    values = np.array([1, 2], dtype=np.int64)
    holder = _Int64ArrayHolder(values=values)
    np.testing.assert_array_equal(holder.values, values)


@attrs.define
class _Uint8ArrayHolder:
    """Test fixture for the uint8 array attrs validator."""

    values: np.ndarray = attrs.field(validator=uint8_array)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([[1, 2]], dtype=np.uint8), r"values must be 1-D, got ndim=2"),
        (np.array([1, 2], dtype=np.uint16), r"values must have dtype uint8, got uint16"),
    ],
)
def test_uint8_array_rejects_invalid_inputs(values: np.ndarray, match: str) -> None:
    """Uint8-array validator should reject non-vector or non-uint8 arrays."""
    with pytest.raises(ValueError, match=match):
        _Uint8ArrayHolder(values=values)


def test_uint8_array_accepts_uint8_vector() -> None:
    """Uint8-array validator should accept a 1-D uint8 array."""
    values = np.array([1, 2], dtype=np.uint8)
    holder = _Uint8ArrayHolder(values=values)
    np.testing.assert_array_equal(holder.values, values)


@attrs.define
class _Uint16ArrayHolder:
    """Test fixture for the uint16 array attrs validator."""

    values: np.ndarray = attrs.field(validator=uint16_array)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([[1, 2]], dtype=np.uint16), r"values must be 1-D, got ndim=2"),
        (np.array([1, 2], dtype=np.uint8), r"values must have dtype uint16, got uint8"),
    ],
)
def test_uint16_array_rejects_invalid_inputs(values: np.ndarray, match: str) -> None:
    """Uint16-array validator should reject non-vector or non-uint16 arrays."""
    with pytest.raises(ValueError, match=match):
        _Uint16ArrayHolder(values=values)


def test_uint16_array_accepts_uint16_vector() -> None:
    """Uint16-array validator should accept a 1-D uint16 array."""
    values = np.array([1, 2], dtype=np.uint16)
    holder = _Uint16ArrayHolder(values=values)
    np.testing.assert_array_equal(holder.values, values)


@attrs.define
class _Uint64ArrayHolder:
    """Test fixture for the uint64 array attrs validator."""

    values: np.ndarray = attrs.field(validator=uint64_array)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([[1, 2]], dtype=np.uint64), r"values must be 1-D, got ndim=2"),
        (np.array([1, 2], dtype=np.int64), r"values must have dtype uint64, got int64"),
    ],
)
def test_uint64_array_rejects_invalid_inputs(values: np.ndarray, match: str) -> None:
    """Uint64-array validator should reject non-vector or non-uint64 arrays."""
    with pytest.raises(ValueError, match=match):
        _Uint64ArrayHolder(values=values)


def test_uint64_array_accepts_uint64_vector() -> None:
    """Uint64-array validator should accept a 1-D uint64 array."""
    values = np.array([1, 2], dtype=np.uint64)
    holder = _Uint64ArrayHolder(values=values)
    np.testing.assert_array_equal(holder.values, values)


@attrs.define
class _FiniteFloat64ArrayHolder:
    """Test fixture for the finite float64 attrs validator."""

    values: np.ndarray = attrs.field(validator=finite_float64_array)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([0.0, 1.0], dtype=np.float32), r"values must have dtype float64, got float32"),
        (np.array([0.0, np.inf], dtype=np.float64), r"values must contain only finite values"),
        (np.array([0.0, np.nan], dtype=np.float64), r"values must contain only finite values"),
    ],
)
def test_finite_float64_array_rejects_invalid_inputs(values: np.ndarray, match: str) -> None:
    """Finite-float validator should reject non-float64 or non-finite arrays."""
    with pytest.raises(ValueError, match=match):
        _FiniteFloat64ArrayHolder(values=values)


def test_finite_float64_array_accepts_finite_float64_array() -> None:
    """Finite-float validator should accept finite float64 arrays of any rank."""
    values = np.ones((2, 3), dtype=np.float64)
    holder = _FiniteFloat64ArrayHolder(values=values)
    np.testing.assert_array_equal(holder.values, values)


@attrs.define
class _FrameBatchHolder:
    """Test fixture for the uint8 frame batch attrs validator."""

    frames: np.ndarray = attrs.field(validator=uint8_frame_batch)


@pytest.mark.parametrize(
    ("frames", "match"),
    [
        (np.zeros((1, 3), dtype=np.uint8), r"frames must be 4-D with shape \(N, H, W, 3\), got ndim=2"),
        (np.zeros((1, 1, 1, 3), dtype=np.float32), r"frames must have dtype uint8, got float32"),
    ],
)
def test_uint8_frame_batch_rejects_invalid_inputs(frames: np.ndarray, match: str) -> None:
    """Frame-batch validator should reject wrong rank or dtype."""
    with pytest.raises(ValueError, match=match):
        _FrameBatchHolder(frames=frames)


def test_uint8_frame_batch_accepts_uint8_4d_array() -> None:
    """Frame-batch validator should accept a 4-D uint8 array."""
    frames = np.zeros((1, 2, 3, 3), dtype=np.uint8)
    holder = _FrameBatchHolder(frames=frames)
    np.testing.assert_array_equal(holder.frames, frames)
