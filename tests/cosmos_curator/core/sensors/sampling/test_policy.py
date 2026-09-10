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
"""Unit tests for sampling policies."""

import pytest

from cosmos_curator.core.sensors.sampling.policy import GpsInterpolationPolicy, NearestTimestampPolicy, NoSamplingPolicy


def test_nearest_timestamp_policy_instantiation() -> None:
    """NearestTimestampPolicy can be constructed with defaults or an explicit maximum delta."""
    default = NearestTimestampPolicy()
    assert default.max_delta_ns is None

    explicit = NearestTimestampPolicy(max_delta_ns=5_000_000)
    assert explicit.max_delta_ns == 5_000_000


def test_nearest_timestamp_policy_rejects_negative_max_delta() -> None:
    """Negative maximum deltas are rejected at construction time."""
    msg = r"'max_delta_ns' must be >= 0: -1"
    with pytest.raises(ValueError, match=msg):
        NearestTimestampPolicy(max_delta_ns=-1)


def test_gps_interpolation_policy_instantiation() -> None:
    """GpsInterpolationPolicy can omit or explicitly bound its support gap."""
    assert GpsInterpolationPolicy().max_gap_ns is None
    assert GpsInterpolationPolicy(max_gap_ns=5_000_000).max_gap_ns == 5_000_000


def test_gps_interpolation_policy_rejects_negative_max_gap() -> None:
    """Negative interpolation support gaps are rejected at construction time."""
    with pytest.raises(ValueError, match=r"'max_gap_ns' must be >= 0: -1"):
        GpsInterpolationPolicy(max_gap_ns=-1)


def test_no_sampling_policy_instantiation() -> None:
    """NoSamplingPolicy is an explicit concrete no-op policy."""
    assert isinstance(NoSamplingPolicy(), NoSamplingPolicy)
