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
"""Unit tests for SamplingSpec."""

import numpy as np

from cosmos_curator.core.sensors.sampling.grid import SamplingGrid
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec


def test_sampling_spec_instantiation() -> None:
    """SamplingSpec can be constructed with only a grid, defaulting policy to None."""
    grid = SamplingGrid(
        timestamps_ns=np.array([0], dtype=np.int64),
        start_ns=0,
        exclusive_end_ns=1,
        stride_ns=1,
        duration_ns=1,
    )
    spec = SamplingSpec(grid=grid)
    assert spec.grid is grid
    assert spec.policy is None
