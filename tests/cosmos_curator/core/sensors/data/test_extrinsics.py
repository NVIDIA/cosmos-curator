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
"""Tests for SensorExtrinsics."""

import numpy as np
import pytest

from cosmos_curator.core.sensors.data.extrinsics import SensorExtrinsics


def test_sensor_extrinsics_rejects_wrong_matrix_shape() -> None:
    """SensorExtrinsics should reject non-4x4 matrices."""
    with pytest.raises(ValueError, match=r"matrix must have shape \(4, 4\)"):
        SensorExtrinsics(matrix=np.eye(3, dtype=np.float64))


def test_sensor_extrinsics_rejects_non_float64_dtype() -> None:
    """SensorExtrinsics should require float64 matrices."""
    with pytest.raises(ValueError, match="matrix must have dtype float64"):
        SensorExtrinsics(matrix=np.eye(4, dtype=np.float32))


def test_sensor_extrinsics_exposes_readonly_view_without_mutating_caller_array() -> None:
    """SensorExtrinsics should keep the caller-owned matrix writeable."""
    matrix = np.eye(4, dtype=np.float64)

    extrinsics = SensorExtrinsics(matrix=matrix)

    assert matrix.flags.writeable is True
    assert extrinsics.matrix.flags.writeable is False
    assert extrinsics.matrix is not matrix
    assert np.shares_memory(extrinsics.matrix, matrix)
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        extrinsics.matrix[0, 0] = 2.0
