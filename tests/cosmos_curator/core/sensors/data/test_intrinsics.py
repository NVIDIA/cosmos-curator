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
"""Tests for CameraIntrinsics."""

import numpy as np
import pytest

from cosmos_curator.core.sensors.data.intrinsics import CameraIntrinsics


def _make_intrinsics() -> CameraIntrinsics:
    """Build a minimal valid CameraIntrinsics instance."""
    return CameraIntrinsics(
        camera_matrix=np.eye(3, dtype=np.float64),
        distortion_coefficients=np.zeros(5, dtype=np.float64),
        distortion_model="brown_conrady",
        width=2,
        height=1,
    )


def test_camera_intrinsics_rejects_wrong_camera_matrix_shape() -> None:
    """CameraIntrinsics should require a ``3x3`` intrinsic matrix."""
    with pytest.raises(ValueError, match=r"camera_matrix must have shape \(3, 3\), got \(2, 2\)"):
        CameraIntrinsics(
            camera_matrix=np.eye(2, dtype=np.float64),
            distortion_coefficients=np.zeros(5, dtype=np.float64),
            distortion_model="brown_conrady",
            width=2,
            height=1,
        )


def test_camera_intrinsics_rejects_non_float64_array_dtypes() -> None:
    """CameraIntrinsics should require float64 array inputs for calibration arrays."""
    with pytest.raises(ValueError, match=r"camera_matrix must have dtype float64"):
        CameraIntrinsics(
            camera_matrix=np.eye(3, dtype=np.float32),
            distortion_coefficients=np.zeros(5, dtype=np.float64),
            distortion_model="brown_conrady",
            width=2,
            height=1,
        )

    with pytest.raises(ValueError, match=r"distortion_coefficients must have dtype float64"):
        CameraIntrinsics(
            camera_matrix=np.eye(3, dtype=np.float64),
            distortion_coefficients=np.zeros(5, dtype=np.float32),
            distortion_model="brown_conrady",
            width=2,
            height=1,
        )


def test_camera_intrinsics_rejects_unknown_distortion_model() -> None:
    """CameraIntrinsics should allow only the supported distortion model names."""
    with pytest.raises(ValueError, match=r"distortion_model must be one of"):
        CameraIntrinsics(
            camera_matrix=np.eye(3, dtype=np.float64),
            distortion_coefficients=np.zeros(5, dtype=np.float64),
            distortion_model="plumb_bob",
            width=2,
            height=1,
        )


@pytest.mark.parametrize(("width", "height"), [(0, 1), (2, 0), (-1, 1), (2, -1)])
def test_camera_intrinsics_rejects_non_positive_dimensions(width: int, height: int) -> None:
    """CameraIntrinsics should require positive image dimensions."""
    with pytest.raises(ValueError, match=r"must be positive"):
        CameraIntrinsics(
            camera_matrix=np.eye(3, dtype=np.float64),
            distortion_coefficients=np.zeros(5, dtype=np.float64),
            distortion_model="none",
            width=width,
            height=height,
        )


@pytest.mark.parametrize(
    ("width", "height", "match"), [(1.5, 1, r"width must be an int"), (2, 1.5, r"height must be an int")]
)
def test_camera_intrinsics_rejects_non_integer_dimensions(width: object, height: object, match: str) -> None:
    """CameraIntrinsics should require integer image dimensions, not just positive scalars."""
    with pytest.raises(TypeError, match=match):
        CameraIntrinsics(
            camera_matrix=np.eye(3, dtype=np.float64),
            distortion_coefficients=np.zeros(5, dtype=np.float64),
            distortion_model="none",
            width=width,  # type: ignore[arg-type]
            height=height,  # type: ignore[arg-type]
        )


def test_camera_intrinsics_arrays_are_readonly_without_mutating_inputs() -> None:
    """CameraIntrinsics should expose read-only views while leaving caller arrays mutable."""
    camera_matrix = np.eye(3, dtype=np.float64)
    distortion_coefficients = np.zeros(5, dtype=np.float64)

    intrinsics = CameraIntrinsics(
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients,
        distortion_model="fisheye",
        width=2,
        height=1,
    )

    assert intrinsics.camera_matrix.flags.writeable is False
    assert intrinsics.distortion_coefficients.flags.writeable is False
    assert intrinsics.camera_matrix is not camera_matrix
    assert intrinsics.distortion_coefficients is not distortion_coefficients
    assert np.shares_memory(intrinsics.camera_matrix, camera_matrix)
    assert np.shares_memory(intrinsics.distortion_coefficients, distortion_coefficients)
    assert camera_matrix.flags.writeable is True
    assert distortion_coefficients.flags.writeable is True

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        intrinsics.camera_matrix[0, 0] = 2.0

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        intrinsics.distortion_coefficients[0] = 1.0


def test_camera_intrinsics_rejects_non_1d_distortion_coefficients() -> None:
    """CameraIntrinsics should require a 1-D distortion vector."""
    with pytest.raises(ValueError, match=r"distortion_coefficients must be 1-D, got ndim=2"):
        CameraIntrinsics(
            camera_matrix=np.eye(3, dtype=np.float64),
            distortion_coefficients=np.zeros((1, 5), dtype=np.float64),
            distortion_model="none",
            width=2,
            height=1,
        )


def test_camera_intrinsics_accepts_valid_payload() -> None:
    """CameraIntrinsics should preserve valid calibration values."""
    intrinsics = _make_intrinsics()

    assert intrinsics.distortion_model == "brown_conrady"
    assert intrinsics.width == 2
    assert intrinsics.height == 1
