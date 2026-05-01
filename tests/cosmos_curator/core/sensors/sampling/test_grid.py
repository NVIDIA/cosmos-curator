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
"""Unit tests for make_ts_grid and SamplingGrid."""

from contextlib import AbstractContextManager, nullcontext
from itertools import pairwise
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.sensors.sampling.grid import SamplingGrid, SamplingWindow, make_ts_grid
from tests.cosmos_curator.core.sensors.test_utils import make_sampling_grid


def _iter_window_arrays(
    ts: npt.NDArray[np.int64],
    *,
    stride_ns: int,
    duration_ns: int,
) -> list[SamplingWindow]:
    p = make_sampling_grid(
        timestamps_ns=ts,
        stride_ns=stride_ns,
        duration_ns=duration_ns,
    )
    return list(p)


def test_sampling_grid_iter() -> None:
    """SamplingGrid should yield raw window slices including the right boundary marker."""
    ts = np.array([0, 10, 20, 30, 40, 50], dtype=np.int64)
    grid = make_sampling_grid(ts, 20, 20)
    got = list(grid)
    want = [
        SamplingWindow(
            start_ns=0,
            exclusive_end_ns=20,
            timestamps_ns=np.array([0, 10], dtype=np.int64),
        ),
        SamplingWindow(
            start_ns=20,
            exclusive_end_ns=40,
            timestamps_ns=np.array([20, 30], dtype=np.int64),
        ),
        SamplingWindow(
            start_ns=40,
            exclusive_end_ns=50,
            timestamps_ns=np.array([40], dtype=np.int64),
        ),
    ]

    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_array_equal(a.timestamps_ns, b.timestamps_ns)
        assert a.start_ns == b.start_ns
        assert a.exclusive_end_ns == b.exclusive_end_ns


def test_sampling_grid_half_open_window_contract() -> None:
    """A yielded window should include the exclusive right boundary marker as its final element."""
    ts = np.array([100, 200, 300, 400], dtype=np.int64)
    grid = make_sampling_grid(ts, 200, 200)
    windows = list(grid)

    np.testing.assert_array_equal(windows[0].timestamps_ns, np.array([100, 200], dtype=np.int64))
    assert windows[0].exclusive_end_ns == 300
    np.testing.assert_array_equal(windows[1].timestamps_ns, np.array([300], dtype=np.int64))
    assert windows[1].exclusive_end_ns == 400


def test_sampling_grid_adjacent_windows_share_boundary_marker() -> None:
    """Adjacent windows should share the boundary timestamp that separates their half-open intervals."""
    ts = np.array([0, 100, 200, 300], dtype=np.int64)
    grid = make_sampling_grid(ts, 100, 100)
    windows = list(grid)

    expected_windows = [
        SamplingWindow(
            start_ns=0,
            exclusive_end_ns=100,
            timestamps_ns=np.array([0], dtype=np.int64),
        ),
        SamplingWindow(
            start_ns=100,
            exclusive_end_ns=200,
            timestamps_ns=np.array([100], dtype=np.int64),
        ),
        SamplingWindow(
            start_ns=200,
            exclusive_end_ns=300,
            timestamps_ns=np.array([200], dtype=np.int64),
        ),
    ]

    assert len(windows) == len(expected_windows)
    for window, expected_window in zip(windows, expected_windows, strict=True):
        np.testing.assert_array_equal(window.timestamps_ns, expected_window.timestamps_ns)
        assert window.start_ns == expected_window.start_ns
        assert window.exclusive_end_ns == expected_window.exclusive_end_ns


def test_make_ts_grid() -> None:
    """Test the make_ts_grid function."""
    start_s = 0.0
    end_s = 5.0
    sample_rate_hz = 30.0
    sample_interval_s = 1 / sample_rate_hz
    n_samples = int(np.floor(np.nextafter((end_s - start_s) / sample_interval_s, np.inf))) + 2
    expected_ts = start_s + np.arange(n_samples, dtype=np.float64) * sample_interval_s
    expected_grid = np.round(expected_ts * 1_000_000_000).astype(np.int64)

    start_ns = int(start_s * 1_000_000_000)
    end_ns = int(end_s * 1_000_000_000)
    start_ns, exclusive_end_ns, timestamps_ns = make_ts_grid(start_ns, end_ns, sample_rate_hz)
    expected_start_ns = int(expected_grid[0])
    expected_exclusive_end_ns = int(expected_grid[-1])
    expected_timestamps_ns = expected_grid[:-1]

    assert start_ns == expected_start_ns
    assert exclusive_end_ns == expected_exclusive_end_ns
    np.testing.assert_array_equal(timestamps_ns, expected_timestamps_ns)


@pytest.mark.parametrize(
    ("start_ns", "end_ns", "sample_rate_hz"),
    [
        (0, 1_000_000_000, 30.0),
        (123, 987_654_321, 29.97),
        (0, 5_000_000_000, 59.94),
        (42, 42 + 1_000_000, 1_000.0),
    ],
)
def test_make_ts_grid_brackets_end_ns(start_ns: int, end_ns: int, sample_rate_hz: float) -> None:
    """make_ts_grid should always produce a final pair that strictly brackets exclusive_end_ns."""
    got_start_ns, got_exclusive_end_ns, got_timestamps_ns = make_ts_grid(start_ns, end_ns, sample_rate_hz)

    assert got_start_ns == start_ns
    assert len(got_timestamps_ns) >= 2
    assert int(got_timestamps_ns[-1]) <= end_ns < got_exclusive_end_ns


@pytest.mark.parametrize(
    ("start_ns", "end_ns", "sample_rate_hz"),
    [
        (0, 1_000_000_000, 30.0),
        (123, 987_654_321, 29.97),
        (0, 5_000_000_000, 59.94),
        (42, 42 + 1_000_000, 1_000.0),
    ],
)
def test_make_ts_grid_is_strictly_increasing_and_on_grid(start_ns: int, end_ns: int, sample_rate_hz: float) -> None:
    """make_ts_grid should stay strictly increasing on the rounded sample interval."""
    _, _, grid = make_ts_grid(start_ns, end_ns, sample_rate_hz)

    deltas = np.diff(grid)
    expected_step_ns = int(np.round(1_000_000_000 / sample_rate_hz))

    assert np.all(deltas > 0)
    assert np.all(np.abs(deltas - expected_step_ns) <= 1)


def test_make_ts_grid_single_timestamp() -> None:
    """When start_ns == end_ns, make_ts_grid should add the next on-grid sample."""
    start_ns, exclusive_end_ns, timestamps_ns = make_ts_grid(42, 42, 30.0)

    assert start_ns == 42
    assert exclusive_end_ns == 33333375

    assert len(timestamps_ns) == 1
    assert int(timestamps_ns[0]) == 42
    assert start_ns <= int(timestamps_ns[0]) < exclusive_end_ns
    expected_delta_ns = int(np.round(1_000_000_000 / 30.0))
    assert (exclusive_end_ns - start_ns) == expected_delta_ns


@pytest.mark.parametrize("sample_rate_hz", [0.0, -1.0])
def test_make_ts_grid_raises_on_non_positive_sample_rate(sample_rate_hz: float) -> None:
    """make_ts_grid should reject zero or negative sampling rates with ValueError."""
    with pytest.raises(ValueError, match="sample_rate_hz must be greater than 0"):
        make_ts_grid(0, 1_000_000_000, sample_rate_hz)


def test_make_ts_grid_raises_when_end_precedes_start() -> None:
    """make_ts_grid should reject intervals whose end precedes the start."""
    with pytest.raises(ValueError, match="end_ns must be greater than or equal to start_ns"):
        make_ts_grid(10, 0, 1.0)


def test_make_ts_grid_raises_when_rounding_makes_grid_non_increasing() -> None:
    """make_ts_grid should reject sample rates that cannot produce a strictly increasing ns grid."""
    with pytest.raises(ValueError, match="does not produce a strictly increasing nanosecond grid"):
        make_ts_grid(0, 10, 1.5e9)


def test_make_ts_grid_exclusive_end_aligned_boundary() -> None:
    """Aligned exclusive_end_ns should be returned unchanged and stop strictly before the boundary."""
    start_ns = 0
    sample_rate_hz = 10.0
    exclusive_end_ns = 1_000_000_000  # 10 samples at 10 Hz lands exactly on the boundary

    got_start_ns, got_exclusive_end_ns, got_timestamps_ns = make_ts_grid(
        start_ns,
        sample_rate_hz=sample_rate_hz,
        exclusive_end_ns=exclusive_end_ns,
    )

    assert got_start_ns == start_ns
    assert got_exclusive_end_ns == exclusive_end_ns
    assert int(got_timestamps_ns[-1]) < exclusive_end_ns
    expected_step_ns = int(np.round(1_000_000_000 / sample_rate_hz))
    deltas = np.diff(got_timestamps_ns)
    assert np.all(deltas > 0)
    assert np.all(np.abs(deltas - expected_step_ns) <= 1)


def test_make_ts_grid_exclusive_end_non_aligned_boundary() -> None:
    """Non-aligned exclusive_end_ns should be returned unchanged with timestamps strictly inside it."""
    start_ns = 0
    sample_rate_hz = 30.0
    exclusive_end_ns = 5_000_000_000

    got_start_ns, got_exclusive_end_ns, got_timestamps_ns = make_ts_grid(
        start_ns,
        sample_rate_hz=sample_rate_hz,
        exclusive_end_ns=exclusive_end_ns,
    )

    assert got_start_ns == start_ns
    assert got_exclusive_end_ns == exclusive_end_ns
    assert int(got_timestamps_ns[-1]) < exclusive_end_ns
    assert exclusive_end_ns not in got_timestamps_ns


def test_make_ts_grid_exclusive_end_with_sampling_grid() -> None:
    """make_ts_grid with exclusive_end_ns should compose cleanly with SamplingGrid."""
    start_ns = 0
    sample_rate_hz = 10.0
    exclusive_end_ns = 500_000_000  # 0.5 s
    stride_ns = 200_000_000
    duration_ns = 200_000_000

    got_start_ns, got_exclusive_end_ns, got_timestamps_ns = make_ts_grid(
        start_ns,
        sample_rate_hz=sample_rate_hz,
        exclusive_end_ns=exclusive_end_ns,
    )
    grid = SamplingGrid(
        start_ns=got_start_ns,
        exclusive_end_ns=got_exclusive_end_ns,
        timestamps_ns=got_timestamps_ns,
        stride_ns=stride_ns,
        duration_ns=duration_ns,
    )
    windows = list(grid)

    assert grid.exclusive_end_ns == exclusive_end_ns
    assert all(w.exclusive_end_ns <= exclusive_end_ns for w in windows)
    assert all(int(w.timestamps_ns[-1]) < exclusive_end_ns for w in windows if len(w) > 0)


def test_make_ts_grid_raises_when_both_ends_supplied() -> None:
    """make_ts_grid should reject having both end_ns and exclusive_end_ns supplied."""
    with pytest.raises(ValueError, match="exactly one of end_ns or exclusive_end_ns"):
        make_ts_grid(0, 1_000_000_000, 30.0, exclusive_end_ns=1_000_000_000)


def test_make_ts_grid_raises_when_neither_end_supplied() -> None:
    """make_ts_grid should reject having neither end_ns nor exclusive_end_ns supplied."""
    with pytest.raises(ValueError, match="exactly one of end_ns or exclusive_end_ns"):
        make_ts_grid(0, sample_rate_hz=30.0)


def test_make_ts_grid_raises_when_exclusive_end_le_start() -> None:
    """make_ts_grid should reject exclusive_end_ns that is not strictly greater than start_ns."""
    with pytest.raises(ValueError, match="exclusive_end_ns must be greater than start_ns"):
        make_ts_grid(100, sample_rate_hz=30.0, exclusive_end_ns=100)


def _expected_window_count(first: int, last: int, stride_ns: int) -> int:
    """Windows emitted while start <= last with start = first + k * stride_ns."""
    if stride_ns <= 0 or first > last:
        return 0
    if first == last:
        return 1
    return (last - first) // stride_ns


@pytest.mark.parametrize(
    ("timestamps_ns", "stride_ns", "duration_ns", "raises"),
    [
        # valid
        (np.array([0, 1], dtype=np.int64), 1, 1, nullcontext()),
        # timestamps must be int64
        (np.array([0, 1], dtype=np.int32), 1, 1, pytest.raises(ValueError, match=r".*")),
        # zero duration
        (np.array([0], dtype=np.int64), 1, 0, pytest.raises(ValueError, match=r".*")),
        # negative duration
        (np.array([0], dtype=np.int64), 1, -1, pytest.raises(ValueError, match=r".*")),
        # zero stride
        (np.array([0], dtype=np.int64), 0, 1, pytest.raises(ValueError, match=r".*")),
        # negative stride
        (np.array([0], dtype=np.int64), -1, 1, pytest.raises(ValueError, match=r".*")),
        # not sorted
        (np.array([1, 0], dtype=np.int64), 1, 1, pytest.raises(ValueError, match=r".*")),
    ],
)
def test_initializer_asserts(
    timestamps_ns: npt.NDArray[np.int64],
    stride_ns: int,
    duration_ns: int,
    raises: AbstractContextManager[Any],
) -> None:
    """Test initializer asserts."""
    with raises:
        make_sampling_grid(timestamps_ns=timestamps_ns, stride_ns=stride_ns, duration_ns=duration_ns)


def test_initializer_state() -> None:
    """Test state after initialization."""
    ts = np.array([100, 200, 300], dtype=np.int64)
    p = make_sampling_grid(timestamps_ns=ts, stride_ns=1, duration_ns=10)
    for t, p_t in zip(ts[:-1], p.timestamps_ns, strict=True):
        assert t == p_t
    assert p.start_ns == ts[0]
    assert p.exclusive_end_ns == ts[-1]
    assert p.stride_ns == 1
    assert p.duration_ns == 10
    assert not np.shares_memory(p.timestamps_ns, ts)


def test_irregular_grid() -> None:
    """Test an irregular grid.

    * Irregular grid should produce the same number of windows as the regular grid.
    * Windows should have the same start and exclusive end times.
    """
    regular_ts_ns = np.array([0, 10, 20, 30, 40, 50], dtype=np.int64)
    irregular_ts_ns = np.array([0, 9, 21, 29, 40, 51], dtype=np.int64)
    expected_irregular_windows = [
        np.array([0, 9], dtype=np.int64),
        np.array([21, 29], dtype=np.int64),
        np.array([40, 51], dtype=np.int64),
    ]
    start_ns = regular_ts_ns[0]
    exclusive_end_ns = 60
    duration_ns = 20
    stride_ns = 20

    grid_regular = SamplingGrid(
        start_ns=start_ns,
        exclusive_end_ns=exclusive_end_ns,
        timestamps_ns=regular_ts_ns,
        stride_ns=stride_ns,
        duration_ns=duration_ns,
    )
    grid_irregular = SamplingGrid(
        start_ns=start_ns,
        exclusive_end_ns=exclusive_end_ns,
        timestamps_ns=irregular_ts_ns,
        stride_ns=stride_ns,
        duration_ns=duration_ns,
    )
    windows_regular = list(grid_regular)
    windows_irregular = list(grid_irregular)

    assert len(windows_regular) == len(windows_irregular)

    for window_regular, window_irregular, expected_irregular in zip(
        windows_regular, windows_irregular, expected_irregular_windows, strict=True
    ):
        assert len(window_regular.timestamps_ns) == len(window_irregular.timestamps_ns)
        np.testing.assert_array_equal(window_irregular.timestamps_ns, expected_irregular)
        assert window_regular.start_ns == window_irregular.start_ns
        assert window_regular.exclusive_end_ns == window_irregular.exclusive_end_ns


def test_stride_equals_duration_tiling() -> None:
    """With stride == duration, windows tile the timeline and reuse boundary markers between windows."""
    ts = np.array([0, 100, 200, 300, 400], dtype=np.int64)
    duration_ns = 200
    stride_ns = 200
    got = _iter_window_arrays(ts, stride_ns=stride_ns, duration_ns=duration_ns)

    want = [
        SamplingWindow(
            start_ns=0,
            exclusive_end_ns=200,
            timestamps_ns=np.array([0, 100], dtype=np.int64),
        ),
        SamplingWindow(
            start_ns=200,
            exclusive_end_ns=400,
            timestamps_ns=np.array([200, 300], dtype=np.int64),
        ),
    ]

    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_array_equal(a.timestamps_ns, b.timestamps_ns)
        assert a.start_ns == b.start_ns
        assert a.exclusive_end_ns == b.exclusive_end_ns


def test_iter_stride_equals_duration_last_window_full_query_one_sample() -> None:
    """The final yielded window may have no timestamps when the last sample only serves as the terminal boundary."""
    ts = np.array([0, 100, 200, 500], dtype=np.int64)
    duration_ns = 400
    stride_ns = 400
    windows = list(make_sampling_grid(ts, stride_ns, duration_ns))
    expected_windows = [
        SamplingWindow(
            start_ns=0,
            exclusive_end_ns=400,
            timestamps_ns=np.array([0, 100, 200], dtype=np.int64),
        ),
        SamplingWindow(
            start_ns=400,
            exclusive_end_ns=500,
            timestamps_ns=np.array([], dtype=np.int64),
        ),
    ]

    assert len(windows) == len(expected_windows)
    for window, expected_window in zip(windows, expected_windows, strict=True):
        np.testing.assert_array_equal(window.timestamps_ns, expected_window.timestamps_ns)
        assert window.start_ns == expected_window.start_ns
        assert window.exclusive_end_ns == expected_window.exclusive_end_ns


def test_iter_stride_less_than_duration_overlap() -> None:
    """Stride smaller than duration yields overlapping windows and predictable count."""
    ts = np.arange(0, 501, 50, dtype=np.int64)
    duration_ns = 200
    stride_ns = 100
    p = make_sampling_grid(ts, stride_ns, duration_ns)
    windows = list(p)
    n = _expected_window_count(p.start_ns, p.exclusive_end_ns, stride_ns)
    assert len(windows) == n

    # Check that windows actually overlap when stride_ns < duration_ns
    # Restrict this to the two windows that share the 150us timestamp
    t_overlap = 150
    windows_with_t = [w for w in windows if np.any(w.timestamps_ns == t_overlap)]
    assert len(windows_with_t) == 2

    # Check that timestamps in each window are within the expected range
    starts = [p.start_ns + k * stride_ns for k in range(len(windows))]
    for w, start in zip(windows, starts, strict=True):
        end = start + duration_ns
        for t in w.timestamps_ns:
            assert start <= int(t) <= end

    # Check that the indices of the windows are monotonic and non-overlapping
    # This is a thin belt & suspenders check, indices should never decrease
    # This grid has a sample on every window start, so indices strictly increase
    first_indices = [int(np.searchsorted(ts, w.timestamps_ns[0], side="left")) for w in windows]
    for prev, cur in pairwise(first_indices):
        assert cur > prev


def test_iter_irregular_sparse_grid_repeated_first_sample_across_windows() -> None:
    """Sparse, irregular timestamps can cause consecutive windows to start with the same sample.

    Here the windows starting at ``3000`` and ``5000`` both begin with ``5000`` and include the
    same sparse pair ``[5000, 5100]``. This is expected on irregular timelines even though the
    window bounds still advance by ``stride_ns``.
    """
    # Gaps: 4ms idle, ~0.1ms pair, ~3.9ms idle — not on a fixed grid.
    ts = np.array([1_000, 5_000, 5_100, 9_000], dtype=np.int64)
    duration_ns = 3_000
    stride_ns = 2_000
    got = _iter_window_arrays(ts, stride_ns=stride_ns, duration_ns=duration_ns)
    want = [
        SamplingWindow(start_ns=1000, exclusive_end_ns=4000, timestamps_ns=np.array([1000], dtype=np.int64)),
        SamplingWindow(start_ns=3000, exclusive_end_ns=6000, timestamps_ns=np.array([5000, 5100], dtype=np.int64)),
        SamplingWindow(start_ns=5000, exclusive_end_ns=8000, timestamps_ns=np.array([5000, 5100], dtype=np.int64)),
        SamplingWindow(start_ns=7000, exclusive_end_ns=9000, timestamps_ns=np.array([], dtype=np.int64)),
    ]

    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_array_equal(a.timestamps_ns, b.timestamps_ns)
        assert a.start_ns == b.start_ns
        assert a.exclusive_end_ns == b.exclusive_end_ns


def test_iter_stride_greater_than_duration_gaps() -> None:
    """Stride larger than duration leaves timestamps that fall in no yielded window.

    grid:    0    50   100    200    300    400    500
    windows: |----------|          |------|         |----|
             0         100        250    350        500  500
                                                    No later boundary marker, so no final window
    """
    ts = np.array([0, 50, 100, 200, 300, 400, 500], dtype=np.int64)
    duration_ns = 100
    stride_ns = 250
    got = _iter_window_arrays(ts, stride_ns=stride_ns, duration_ns=duration_ns)
    want = [
        SamplingWindow(timestamps_ns=np.array([0, 50], dtype=np.int64), start_ns=0, exclusive_end_ns=100),
        SamplingWindow(timestamps_ns=np.array([300], dtype=np.int64), start_ns=250, exclusive_end_ns=350),
    ]
    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_array_equal(a.timestamps_ns, b.timestamps_ns)
        assert a.start_ns == b.start_ns
        assert a.exclusive_end_ns == b.exclusive_end_ns

    expected_orphans = [200, 400, 500]
    covered = np.array(
        list({int(x) for x in np.concatenate([w.timestamps_ns for w in got])} | {w.exclusive_end_ns for w in want}),
        dtype=np.int64,
    )
    orphans = np.setdiff1d(ts, covered)
    np.testing.assert_array_equal(orphans, expected_orphans)


def test_boundaries_on_every_timestamp() -> None:
    """Every timestamp after the first can serve as a shared boundary marker between adjacent windows."""
    ts = np.array([0, 100, 200], dtype=np.int64)
    duration_ns = 100
    stride_ns = 100
    got = _iter_window_arrays(ts, stride_ns=stride_ns, duration_ns=duration_ns)
    want = [
        SamplingWindow(start_ns=0, exclusive_end_ns=100, timestamps_ns=np.array([0], dtype=np.int64)),
        SamplingWindow(start_ns=100, exclusive_end_ns=200, timestamps_ns=np.array([100], dtype=np.int64)),
    ]
    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_array_equal(a.timestamps_ns, b.timestamps_ns)
        assert a.start_ns == b.start_ns
        assert a.exclusive_end_ns == b.exclusive_end_ns


def test_initializer_defensively_copies_timestamps() -> None:
    """Mutating the caller-owned array after construction must not affect the grid."""
    ts = np.array([100, 200, 300], dtype=np.int64)
    got = _iter_window_arrays(ts, stride_ns=100, duration_ns=100)
    ts[1] = 999
    want = [
        SamplingWindow(start_ns=100, exclusive_end_ns=200, timestamps_ns=np.array([100], dtype=np.int64)),
        SamplingWindow(start_ns=200, exclusive_end_ns=300, timestamps_ns=np.array([200], dtype=np.int64)),
    ]

    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_array_equal(a.timestamps_ns, b.timestamps_ns)
        assert a.start_ns == b.start_ns
        assert a.exclusive_end_ns == b.exclusive_end_ns


def test_iter_sample_span_inside_window_may_be_shorter_than_duration() -> None:
    """Timestamps in a yielded window can span less than ``duration_ns`` when the timeline has gaps.

    grid:     0    100   200   400   700   900 <-- exclusive_end_ns, not included in the last window
    windows:  |--------------|-----|--------|
              0             300   600      900
    """
    duration_ns = 300
    stride_ns = 300
    ts = np.array([0, 100, 200, 400, 700, 900], dtype=np.int64)
    got = _iter_window_arrays(ts, stride_ns=stride_ns, duration_ns=duration_ns)

    want = [
        SamplingWindow(start_ns=0, exclusive_end_ns=300, timestamps_ns=np.array([0, 100, 200], dtype=np.int64)),
        SamplingWindow(start_ns=300, exclusive_end_ns=600, timestamps_ns=np.array([400], dtype=np.int64)),
        SamplingWindow(start_ns=600, exclusive_end_ns=900, timestamps_ns=np.array([700], dtype=np.int64)),
    ]

    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_array_equal(a.timestamps_ns, b.timestamps_ns)
        assert a.start_ns == b.start_ns
        assert a.exclusive_end_ns == b.exclusive_end_ns


def test_duration_shorter_than_stride() -> None:
    """Duration shorter than stride yields disjoint windows whose slices may end before the nominal interval end.

    grid:     0    60    120   180     240   300
    windows:  |-------|   |--------|    |--------|
              0      100 120       220  240     340

    The final yielded slice ends at 300 because no later timestamp is available.
    """
    ts = np.array([0, 60, 120, 180, 240, 300], dtype=np.int64)
    duration_ns = 100
    stride_ns = 120
    p = make_sampling_grid(ts, stride_ns, duration_ns)
    got_windows = list(p)
    expected_windows = [
        SamplingWindow(start_ns=0, exclusive_end_ns=100, timestamps_ns=np.array([0, 60], dtype=np.int64)),
        SamplingWindow(start_ns=120, exclusive_end_ns=220, timestamps_ns=np.array([120, 180], dtype=np.int64)),
        SamplingWindow(start_ns=240, exclusive_end_ns=300, timestamps_ns=np.array([240], dtype=np.int64)),
    ]

    assert len(got_windows) == len(expected_windows)
    for got_window, expected_window in zip(got_windows, expected_windows, strict=True):
        np.testing.assert_array_equal(got_window.timestamps_ns, expected_window.timestamps_ns)
        assert got_window.start_ns == expected_window.start_ns
        assert got_window.exclusive_end_ns == expected_window.exclusive_end_ns


def test_iter_stride_equals_duration_many_steps() -> None:
    """With stride == duration, a coarse rounded grid still tiles into predictable windows even when counts differ."""
    # 31 timestamps 0, 1, 2, ..., 30 each scaled by 1_000_000_000 // 3 ns (~333.3 ms steps).
    # >>> ts = np.arange(0, 31, dtype=np.int64) * (1_000_000_000 // 3)
    # This is a slightly irregular grid, the first window has more samples than the others.
    ts = np.array(
        [
            # window 0: start_ns: 0, exclusive_end_ns: 2_000_000_000
            0,
            333333333,
            666666666,
            999999999,
            1333333332,
            1666666665,
            1999999998,
            # window 1: start_ns: 2_000_000_000, exclusive_end_ns: 4_000_000_000
            2333333331,
            2666666664,
            2999999997,
            3333333330,
            3666666663,
            3999999996,
            # window 2: start_ns: 4_000_000_000, exclusive_end_ns: 6_000_000_000
            4333333329,
            4666666662,
            4999999995,
            5333333328,
            5666666661,
            5999999994,
            # window 3: start_ns: 6_000_000_000, exclusive_end_ns: 8_000_000_000
            6333333327,
            6666666660,
            6999999993,
            7333333326,
            7666666659,
            7999999992,
            # window 4: start_ns: 8_000_000_000, exclusive_end_ns: 10_000_000_000
            8333333325,
            8666666658,
            8999999991,
            9333333324,
            9666666657,
            9999999990,
        ],
        dtype=np.int64,
    )

    duration_ns = 2 * 1_000_000_000
    stride_ns = duration_ns
    windows = list(make_sampling_grid(ts, stride_ns, duration_ns))
    expected_windows = [
        SamplingWindow(
            start_ns=0,
            exclusive_end_ns=2_000_000_000,
            timestamps_ns=np.array(
                [0, 333333333, 666666666, 999999999, 1333333332, 1666666665, 1999999998], dtype=np.int64
            ),
        ),
        SamplingWindow(
            start_ns=2_000_000_000,
            exclusive_end_ns=4_000_000_000,
            timestamps_ns=np.array(
                [2333333331, 2666666664, 2999999997, 3333333330, 3666666663, 3999999996], dtype=np.int64
            ),
        ),
        SamplingWindow(
            start_ns=4_000_000_000,
            exclusive_end_ns=6_000_000_000,
            timestamps_ns=np.array(
                [4333333329, 4666666662, 4999999995, 5333333328, 5666666661, 5999999994], dtype=np.int64
            ),
        ),
        SamplingWindow(
            start_ns=6_000_000_000,
            exclusive_end_ns=8_000_000_000,
            timestamps_ns=np.array(
                [6333333327, 6666666660, 6999999993, 7333333326, 7666666659, 7999999992], dtype=np.int64
            ),
        ),
        SamplingWindow(
            start_ns=8_000_000_000,
            exclusive_end_ns=ts[-1],
            timestamps_ns=np.array([8333333325, 8666666658, 8999999991, 9333333324, 9666666657], dtype=np.int64),
        ),
    ]

    assert len(windows) == len(expected_windows)
    for w, expected_w in zip(windows, expected_windows, strict=True):
        np.testing.assert_array_equal(w.timestamps_ns, expected_w.timestamps_ns)
        assert w.start_ns == expected_w.start_ns
        assert w.exclusive_end_ns == expected_w.exclusive_end_ns


@pytest.mark.parametrize(
    ("timestamps_ns", "start_ns", "exclusive_end_ns"),
    [
        (np.array([10, 20], dtype=np.int64), 10, 30),
        (np.array([10, 20], dtype=np.int64), 5, 30),
        (np.array([], dtype=np.int64), 10, 20),
    ],
)
def test_sampling_window_accepts_valid_bounds(
    timestamps_ns: npt.NDArray[np.int64],
    start_ns: int,
    exclusive_end_ns: int,
) -> None:
    """SamplingWindow should allow start_ns to precede the first timestamp."""
    window = SamplingWindow(
        timestamps_ns=timestamps_ns,
        start_ns=start_ns,
        exclusive_end_ns=exclusive_end_ns,
    )
    np.testing.assert_array_equal(window.timestamps_ns, timestamps_ns)


@pytest.mark.parametrize(
    ("timestamps_ns", "start_ns", "exclusive_end_ns", "raises"),
    [
        (np.array([10], dtype=np.int64), 10, 9, pytest.raises(ValueError, match="end_ns must be greater than")),
        (
            np.array([9, 10], dtype=np.int64),
            10,
            20,
            pytest.raises(ValueError, match="start_ns must be <="),
        ),
        (
            np.array([10, 20], dtype=np.int64),
            10,
            20,
            pytest.raises(ValueError, match="end_ns must be < exclusive_end_ns"),
        ),
        (
            np.array([10, 10], dtype=np.int64),
            10,
            20,
            pytest.raises(ValueError, match="strictly sorted"),
        ),
        (
            np.array([10, 20], dtype=np.int32),
            10,
            30,
            pytest.raises(ValueError, match="must have dtype int64"),
        ),
        (
            np.array([[10, 20]], dtype=np.int64),
            10,
            30,
            pytest.raises(ValueError, match="must be 1-D"),
        ),
    ],
)
def test_sampling_window_rejects_invalid_inputs(
    timestamps_ns: npt.NDArray[np.int64],
    start_ns: int,
    exclusive_end_ns: int,
    raises: AbstractContextManager[Any],
) -> None:
    """SamplingWindow should reject inputs that violate its bounds or array invariants."""
    with raises:
        SamplingWindow(
            timestamps_ns=timestamps_ns,
            start_ns=start_ns,
            exclusive_end_ns=exclusive_end_ns,
        )


def test_sampling_window_timestamps_are_read_only() -> None:
    """SamplingWindow should expose a read-only timestamp array."""
    window = SamplingWindow(
        timestamps_ns=np.array([0, 100, 200], dtype=np.int64),
        start_ns=0,
        exclusive_end_ns=300,
    )

    assert not window.timestamps_ns.flags.writeable
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        window.timestamps_ns[0] = 1


def test_sampling_window_does_not_mutate_caller_owned_timestamps() -> None:
    """SamplingWindow should keep the caller's timestamp array writeable."""
    timestamps_ns = np.array([0, 100, 200], dtype=np.int64)

    window = SamplingWindow(
        timestamps_ns=timestamps_ns,
        start_ns=0,
        exclusive_end_ns=300,
    )

    assert timestamps_ns.flags.writeable is True
    assert window.timestamps_ns.flags.writeable is False
    assert window.timestamps_ns is not timestamps_ns
    assert np.shares_memory(window.timestamps_ns, timestamps_ns)


def test_sampling_window_len() -> None:
    """SamplingWindow should return the number of active timestamps in the window."""
    ts = np.array([0, 100, 200], dtype=np.int64)
    window = SamplingWindow(
        timestamps_ns=ts,
        start_ns=0,
        exclusive_end_ns=300,
    )
    assert len(window) == len(ts)
    assert len(window.timestamps_ns) == len(ts)

    ts = np.array([], dtype=np.int64)
    window = SamplingWindow(
        timestamps_ns=ts,
        start_ns=0,
        exclusive_end_ns=300,
    )
    assert len(window) == 0
    assert len(window.timestamps_ns) == 0
