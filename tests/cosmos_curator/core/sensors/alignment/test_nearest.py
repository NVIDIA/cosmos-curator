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

"""Tests for generic timestamp nearest-neighbour alignment."""

import numpy as np
import pytest

from cosmos_curator.core.sensors.alignment.nearest import nearest_timestamp_indices


def test_nearest_timestamp_indices_selects_nearest_and_earlier_midpoint() -> None:
    """The generic operation gives GPS and future pose alignment identical tie semantics."""
    indices = nearest_timestamp_indices(
        np.array([100, 200, 400], dtype=np.int64),
        np.array([100, 150, 190, 350], dtype=np.int64),
    )

    np.testing.assert_array_equal(indices, [0, 0, 1, 2])


@pytest.mark.parametrize(
    ("source_timestamps_ns", "requested_timestamps_ns", "match"),
    [
        (np.empty(0, dtype=np.int64), np.array([100], dtype=np.int64), "source_timestamps_ns must be non-empty"),
        (np.array([100], dtype=np.int64), np.empty(0, dtype=np.int64), "requested_timestamps_ns must be non-empty"),
    ],
)
def test_nearest_timestamp_indices_rejects_empty_inputs(
    source_timestamps_ns: np.ndarray,
    requested_timestamps_ns: np.ndarray,
    match: str,
) -> None:
    """The generic helper rejects empty input before attempting a nearest search."""
    with pytest.raises(ValueError, match=match):
        nearest_timestamp_indices(source_timestamps_ns, requested_timestamps_ns)


def test_nearest_timestamp_indices_reuses_the_only_source_timestamp() -> None:
    """One source timestamp serves every requested row without a search boundary."""
    indices = nearest_timestamp_indices(
        np.array([100], dtype=np.int64),
        np.array([50, 150], dtype=np.int64),
    )

    np.testing.assert_array_equal(indices, [0, 0])
