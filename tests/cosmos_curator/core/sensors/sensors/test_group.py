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

"""Tests for SensorGroup."""

from collections.abc import Generator

import attrs
import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.sensors.data.aligned_frame import AlignedFrame
from cosmos_curator.core.sensors.sampling.grid import SamplingGrid
from cosmos_curator.core.sensors.sampling.policy import SamplingPolicy
from cosmos_curator.core.sensors.sampling.sampler import sample_window_indices
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors.group import SensorGroup


@attrs.define
class _FakeSensorData:
    align_timestamps_ns: npt.NDArray[np.int64]
    sensor_timestamps_ns: npt.NDArray[np.int64]


class _FakeSensor:
    """Minimal in-memory sensor for unit tests: nearest-neighbour sampling, no I/O."""

    def __init__(self, sensor_timestamps_ns: npt.NDArray[np.int64]) -> None:
        self._ts = np.array(sensor_timestamps_ns, dtype=np.int64, copy=True)

    @property
    def start_ns(self) -> int:
        return int(self._ts[0])

    @property
    def end_ns(self) -> int:
        return int(self._ts[-1])

    def sample(self, spec: SamplingSpec) -> Generator[_FakeSensorData, None, None]:
        empty = np.empty(0, dtype=np.int64)
        for window in spec.grid:
            if len(window) == 0:
                yield _FakeSensorData(align_timestamps_ns=empty, sensor_timestamps_ns=empty)
                continue
            indices, _counts = sample_window_indices(self._ts, window, policy=spec.policy, dedup=False)
            if len(indices) == 0:
                yield _FakeSensorData(align_timestamps_ns=empty, sensor_timestamps_ns=empty)
                continue
            yield _FakeSensorData(
                align_timestamps_ns=np.array(window.timestamps_ns, dtype=np.int64),
                sensor_timestamps_ns=self._ts[indices],
            )


def _make_grid(timestamps_ns: npt.NDArray[np.int64], stride_ns: int, duration_ns: int) -> SamplingGrid:
    return SamplingGrid(
        start_ns=int(timestamps_ns[0]),
        exclusive_end_ns=int(timestamps_ns[-1]) + stride_ns,
        timestamps_ns=timestamps_ns,
        stride_ns=stride_ns,
        duration_ns=duration_ns,
    )


_TS = np.array([0, 1_000, 2_000, 3_000, 4_000], dtype=np.int64)
_STRIDE = 1_000


def test_single_sensor_yields_one_frame_per_window() -> None:
    """SensorGroup with one sensor yields one AlignedFrame per grid window."""
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    group = SensorGroup({"a": _FakeSensor(_TS)})

    frames = list(group.sample(spec))
    windows = list(grid)

    assert len(frames) == len(windows)
    for frame, window in zip(frames, windows, strict=True):
        assert isinstance(frame, AlignedFrame)
        np.testing.assert_array_equal(frame.align_timestamps_ns, window.timestamps_ns)
        assert "a" in frame.sensor_data


def test_multi_sensor_all_present_when_coverage_complete() -> None:
    """All sensors appear in every frame when both cover all windows."""
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    group = SensorGroup({"a": _FakeSensor(_TS), "b": _FakeSensor(_TS)})

    for frame in group.sample(spec):
        assert "a" in frame.sensor_data
        assert "b" in frame.sensor_data


def test_start_ns_is_min_across_sensors() -> None:
    """start_ns is the minimum start_ns across all sensors."""
    ts_early = np.array([0, 1_000, 2_000], dtype=np.int64)
    ts_late = np.array([500, 1_500, 2_500], dtype=np.int64)
    group = SensorGroup({"early": _FakeSensor(ts_early), "late": _FakeSensor(ts_late)})
    assert group.start_ns == 0


def test_end_ns_is_max_across_sensors() -> None:
    """end_ns is the maximum end_ns across all sensors."""
    ts_short = np.array([0, 1_000, 2_000], dtype=np.int64)
    ts_long = np.array([0, 1_000, 5_000], dtype=np.int64)
    group = SensorGroup({"short": _FakeSensor(ts_short), "long": _FakeSensor(ts_long)})
    assert group.end_ns == 5_000


def test_policy_none_does_not_raise() -> None:
    """policy=None passes through without enforcement."""
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid, policy=None)
    group = SensorGroup({"a": _FakeSensor(_TS)})
    frames = list(group.sample(spec))
    assert len(frames) == len(_TS)


def test_policy_tolerance_exceeded_raises() -> None:
    """A sensor whose nearest match exceeds policy.tolerance_ns raises ValueError."""
    # Window [1000, 2000): eligible sensor ts=[1500], grid ts=[1000], delta=500 > tolerance=100
    sensor_ts = np.array([0, 1_500, 2_000, 3_000, 4_000], dtype=np.int64)
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid, policy=SamplingPolicy(tolerance_ns=100))
    group = SensorGroup({"a": _FakeSensor(sensor_ts)})

    with pytest.raises(ValueError, match="tolerance_ns"):
        list(group.sample(spec))


def test_sensor_with_no_coverage_omitted_from_frame() -> None:
    """A sensor with no eligible data for a window is excluded from that frame's sensor_data."""
    ts_short = np.array([0, 1_000], dtype=np.int64)
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    group = SensorGroup({"full": _FakeSensor(_TS), "short": _FakeSensor(ts_short)})

    frames = list(group.sample(spec))

    # ts_short covers windows [0,1000) and [1000,2000) — both sensors present
    assert "short" in frames[0].sensor_data
    assert "short" in frames[1].sensor_data

    # Windows [2000,5000) are outside ts_short's range — "short" omitted
    for frame in frames[2:]:
        assert "short" not in frame.sensor_data
        assert "full" in frame.sensor_data


def test_empty_sensors_raises() -> None:
    """Constructing SensorGroup with no sensors raises ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        SensorGroup({})
