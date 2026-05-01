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
"""Shared helpers for sensor tests."""

import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.sampling.grid import SamplingGrid


def make_sampling_grid(
    timestamps_ns: npt.NDArray[np.int64],
    stride_ns: int,
    duration_ns: int,
) -> SamplingGrid:
    """Build a SamplingGrid from a boundary-inclusive timestamp series.

    Previously, SamplingGrid took the same args as this function, but
    the signature was expanded.

    This helper uses the old semantics and translates to the expanded
    signature.
    """
    if len(timestamps_ns) == 0:
        start_ns = 0
        exclusive_end_ns = 0
    elif len(timestamps_ns) == 1:
        start_ns = int(timestamps_ns[0])
        exclusive_end_ns = int(timestamps_ns[0]) + duration_ns
    else:
        start_ns = int(timestamps_ns[0])
        exclusive_end_ns = int(timestamps_ns[-1])

    return SamplingGrid(
        start_ns=start_ns,
        exclusive_end_ns=exclusive_end_ns,
        timestamps_ns=timestamps_ns[:-1],
        stride_ns=stride_ns,
        duration_ns=duration_ns,
    )
