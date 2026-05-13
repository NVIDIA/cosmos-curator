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

from cosmos_curator.core.sensors.sampling.policy import SamplingPolicy


def test_sampling_policy_instantiation() -> None:
    """SamplingPolicy can be constructed with defaults or explicit tolerance."""
    default = SamplingPolicy()
    assert default.tolerance_ns == 0
    assert default.sensor_overlap == 0.0

    explicit = SamplingPolicy(tolerance_ns=5_000_000, sensor_overlap=0.5)
    assert explicit.tolerance_ns == 5_000_000
    assert explicit.sensor_overlap == 0.5


def test_sampling_policy_rejects_negative_tolerance() -> None:
    """Negative tolerances are rejected at construction time."""
    msg = r"'tolerance_ns' must be >= 0: -1"
    with pytest.raises(ValueError, match=msg):
        SamplingPolicy(tolerance_ns=-1)


@pytest.mark.parametrize("sensor_overlap", [0.0, 0.5, 1.0])
def test_sampling_policy_accepts_valid_sensor_overlap(sensor_overlap: float) -> None:
    """Sensor overlap thresholds in [0.0, 1.0] are accepted."""
    policy = SamplingPolicy(sensor_overlap=sensor_overlap)
    assert policy.sensor_overlap == sensor_overlap


@pytest.mark.parametrize("sensor_overlap", [-0.1, 1.1])
def test_sampling_policy_rejects_invalid_sensor_overlap(sensor_overlap: float) -> None:
    """Sensor overlap thresholds outside [0.0, 1.0] are rejected."""
    with pytest.raises(ValueError, match="sensor_overlap"):
        SamplingPolicy(sensor_overlap=sensor_overlap)
