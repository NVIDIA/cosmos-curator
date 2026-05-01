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
"""Test sampling utilities for the sensor library."""

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.sensors.sampling.grid import SamplingWindow
from cosmos_curator.core.sensors.sampling.policy import SamplingPolicy
from cosmos_curator.core.sensors.sampling.sampler import (
    find_closest_indices,
    sample_window_indices,
)


def _window_from_grid(grid: npt.NDArray[np.int64]) -> SamplingWindow:
    if len(grid) == 0:
        return SamplingWindow(start_ns=0, exclusive_end_ns=0, timestamps_ns=np.array([], dtype=np.int64))
    return SamplingWindow(start_ns=grid[0], exclusive_end_ns=grid[-1], timestamps_ns=grid[:-1])


@pytest.mark.parametrize(
    ("canonical", "grid", "expected_indices"),
    [
        # Exact matches.
        (
            np.array([0, 10, 20], dtype=np.int64),
            np.array([0, 10, 20], dtype=np.int64),
            np.array([0, 1, 2], dtype=np.int64),
        ),
        # Standard nearest-neighbour matching.
        (
            np.array([0, 10, 20, 30, 40], dtype=np.int64),
            np.array([6, 26], dtype=np.int64),
            np.array([1, 3], dtype=np.int64),
        ),
        # Returned indices refer to positions in the original canonical array.
        (
            np.array([100, 200, 300, 400], dtype=np.int64),
            np.array([150, 350], dtype=np.int64),
            np.array([0, 2], dtype=np.int64),
        ),
    ],
)
def test_find_closest_indices_core_contract(
    canonical: npt.NDArray[np.int64],
    grid: npt.NDArray[np.int64],
    expected_indices: npt.NDArray[np.int64],
) -> None:
    """find_closest_indices should return nearest-neighbour indices into the original canonical array."""
    result = find_closest_indices(canonical, grid)

    np.testing.assert_array_equal(result, expected_indices)
    assert np.all(result >= 0)
    assert np.all(result < len(canonical))


@pytest.mark.parametrize(
    ("canonical", "grid", "expected_indices"),
    [
        # Midpoint ties go left.
        (
            np.array([0, 10], dtype=np.int64),
            np.array([5], dtype=np.int64),
            np.array([0], dtype=np.int64),
        ),
        # Values before the first canonical snap to the first index.
        (
            np.array([10, 20], dtype=np.int64),
            np.array([0, 5], dtype=np.int64),
            np.array([0, 0], dtype=np.int64),
        ),
        # Values after the last canonical snap to the last index.
        (
            np.array([10, 20], dtype=np.int64),
            np.array([25, 30], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
        ),
        # A single-element canonical maps every grid point to index 0.
        (
            np.array([10], dtype=np.int64),
            np.array([0, 10, 20], dtype=np.int64),
            np.array([0, 0, 0], dtype=np.int64),
        ),
    ],
)
def test_find_closest_indices_boundary_and_tie_behavior(
    canonical: npt.NDArray[np.int64],
    grid: npt.NDArray[np.int64],
    expected_indices: npt.NDArray[np.int64],
) -> None:
    """find_closest_indices should handle ties and out-of-range values consistently."""
    result = find_closest_indices(canonical, grid)

    np.testing.assert_array_equal(result, expected_indices)
    assert np.all(result >= 0)
    assert np.all(result < len(canonical))


@pytest.mark.parametrize(
    ("canonical", "grid", "match"),
    [
        (
            np.array([], dtype=np.int64),
            np.array([0, 10, 20], dtype=np.int64),
            "canonical must be non-empty",
        ),
        (
            np.array([0, 10, 20], dtype=np.int64),
            np.array([], dtype=np.int64),
            "grid must be non-empty",
        ),
        (
            np.array([0, 10, 10, 20], dtype=np.int64),
            np.array([0, 5, 15], dtype=np.int64),
            "canonical must be strictly sorted in ascending order with no duplicates",
        ),
        (
            np.array([0, 10, 20], dtype=np.int64),
            np.array([0, 10, 10], dtype=np.int64),
            "grid must be strictly sorted in ascending order with no duplicates",
        ),
    ],
)
def test_find_closest_indices_input_validation(
    canonical: npt.NDArray[np.int64],
    grid: npt.NDArray[np.int64],
    match: str,
) -> None:
    """find_closest_indices should reject invalid canonical and grid inputs."""
    with pytest.raises(ValueError, match=match):
        find_closest_indices(canonical, grid)


@pytest.mark.parametrize(
    ("canonical", "grid", "expected_canonical", "expected_indices", "expected_counts", "dedup"),
    [
        # Basic in-window matching
        # Establish the normal behavior
        # Why:
        # - active reference timestamps are 150, 260
        # - eligible canonicals are 200, 300
        # - both get used
        # Basic in-window matching.
        (
            np.array([100, 200, 300, 400], dtype=np.int64),
            np.array([150, 260, 350], dtype=np.int64),
            np.array([200, 300], dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            True,
        ),
        # Boundary marker is not sampled.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([150, 250, 300], dtype=np.int64),
            np.array([200, 200], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            False,
        ),
        # Right boundary is exclusive for canonical eligibility.
        (
            np.array([150, 250, 350], dtype=np.int64),
            np.array([150, 260, 350], dtype=np.int64),
            np.array([150, 250], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            True,
        ),
        # Repeated picks collapse under dedup.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([110, 140, 240, 350], dtype=np.int64),
            np.array([200], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([3], dtype=np.int64),
            True,
        ),
        # Repeated picks are preserved when dedup is disabled.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([110, 140, 240, 350], dtype=np.int64),
            np.array([200, 200, 200], dtype=np.int64),
            np.array([1, 1, 1], dtype=np.int64),
            np.array([1, 1, 1], dtype=np.int64),
            False,
        ),
        # Returned indices refer to the original canonical array.
        (
            np.array([50, 150, 250, 350, 450], dtype=np.int64),
            np.array([140, 260, 340], dtype=np.int64),
            np.array([150, 250], dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            False,
        ),
    ],
)
def test_sample_window_indices_core_contract(  # noqa: PLR0913
    canonical: npt.NDArray[np.int64],
    grid: npt.NDArray[np.int64],
    expected_canonical: npt.NDArray[np.int64],
    expected_indices: npt.NDArray[np.int64],
    expected_counts: npt.NDArray[np.int64],
    *,
    dedup: bool,
) -> None:
    """Test the core contract of sample_window_indices."""
    window = _window_from_grid(grid)
    indices, counts = sample_window_indices(canonical=canonical, window=window, dedup=dedup)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_array_equal(canonical[indices], expected_canonical)
    assert np.all(indices >= 0)
    assert np.all(indices < len(canonical))


@pytest.mark.parametrize(
    ("canonical", "grid", "expected_canonical", "expected_indices", "expected_counts", "dedup"),
    [
        # Left boundary is inclusive: canonical == grid[0] is eligible.
        (
            np.array([100, 200], dtype=np.int64),
            np.array([100, 150, 250], dtype=np.int64),
            np.array([100, 100], dtype=np.int64),
            np.array([0, 0], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            False,
        ),
        # Right boundary is exclusive: canonical == grid[-1] is not eligible.
        (
            np.array([150, 250, 350], dtype=np.int64),
            np.array([150, 260, 350], dtype=np.int64),
            np.array([150, 250], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            True,
        ),
        # grid[-1] is a boundary marker, not an active reference timestamp.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([150, 250, 300], dtype=np.int64),
            np.array([200, 200], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            False,
        ),
        # A singleton window has no active reference timestamps.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([150], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            True,
        ),
        # A two-element window has exactly one active reference timestamp.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([150, 250], dtype=np.int64),
            np.array([200], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([1], dtype=np.int64),
            False,
        ),
    ],
)
def test_sample_window_indices_half_open_window_semantics(  # noqa: PLR0913
    canonical: npt.NDArray[np.int64],
    grid: npt.NDArray[np.int64],
    expected_canonical: npt.NDArray[np.int64],
    expected_indices: npt.NDArray[np.int64],
    expected_counts: npt.NDArray[np.int64],
    *,
    dedup: bool,
) -> None:
    """sample_window_indices should obey the half-open window contract."""
    window = _window_from_grid(grid)
    indices, counts = sample_window_indices(canonical=canonical, window=window, dedup=dedup)

    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(canonical[indices], expected_canonical)
    np.testing.assert_array_equal(counts, expected_counts)

    assert np.all(indices >= 0)
    assert np.all(indices < len(canonical))


@pytest.mark.parametrize(
    ("canonical", "grid", "expected_canonical", "expected_indices", "expected_counts"),
    [
        # Out-of-window canonical timestamps are ignored even if they would be
        # closer under a global nearest-neighbour search.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([150, 260, 350], dtype=np.int64),
            np.array([200, 300], dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
        ),
        # If there are no canonical timestamps in the current window, return
        # empty arrays rather than sampling from neighbouring windows.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([400, 500], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        ),
        # Canonical timestamps on both sides of the window do not leak in.
        (
            np.array([100, 200, 300, 400], dtype=np.int64),
            np.array([180, 220, 280], dtype=np.int64),
            np.array([200], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([2], dtype=np.int64),
        ),
    ],
)
def test_sample_window_indices_window_local_eligibility(
    canonical: npt.NDArray[np.int64],
    grid: npt.NDArray[np.int64],
    expected_canonical: npt.NDArray[np.int64],
    expected_indices: npt.NDArray[np.int64],
    expected_counts: npt.NDArray[np.int64],
) -> None:
    """sample_window_indices should use only canonical timestamps from the current window."""
    window = _window_from_grid(grid)
    indices, counts = sample_window_indices(canonical=canonical, window=window)

    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(canonical[indices], expected_canonical)
    np.testing.assert_array_equal(counts, expected_counts)

    assert np.all(indices >= 0)
    assert np.all(indices < len(canonical))


@pytest.mark.parametrize(
    ("canonical", "grid", "expected_canonical", "expected_indices", "expected_counts", "dedup"),
    [
        # Exact matches are preserved.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([100, 200, 300, 350], dtype=np.int64),
            np.array([100, 200, 300], dtype=np.int64),
            np.array([0, 1, 2], dtype=np.int64),
            np.array([1, 1, 1], dtype=np.int64),
            False,
        ),
        # Left/right nearest choice inside the eligible window-local subset.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([160, 260, 350], dtype=np.int64),
            np.array([200, 300], dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            False,
        ),
        # Midpoint ties resolve to the left eligible canonical timestamp.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([150, 250, 350], dtype=np.int64),
            np.array([200, 200], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            False,
        ),
        # Values beyond the last eligible canonical snap to the last eligible one.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([260, 340], dtype=np.int64),
            np.array([300], dtype=np.int64),
            np.array([2], dtype=np.int64),
            np.array([1], dtype=np.int64),
            False,
        ),
        # Values before the first eligible canonical snap to the first eligible one.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([150, 170, 350], dtype=np.int64),
            np.array([200, 200], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            False,
        ),
    ],
)
def test_sample_window_indices_nearest_neighbor_within_window(  # noqa: PLR0913
    canonical: npt.NDArray[np.int64],
    grid: npt.NDArray[np.int64],
    expected_canonical: npt.NDArray[np.int64],
    expected_indices: npt.NDArray[np.int64],
    expected_counts: npt.NDArray[np.int64],
    *,
    dedup: bool,
) -> None:
    """sample_window_indices should perform nearest-neighbour matching within the eligible window-local subset."""
    window = _window_from_grid(grid)
    indices, counts = sample_window_indices(canonical=canonical, window=window, dedup=dedup)

    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(canonical[indices], expected_canonical)
    np.testing.assert_array_equal(counts, expected_counts)

    assert np.all(indices >= 0)
    assert np.all(indices < len(canonical))


def test_sample_window_indices_returns_original_indices_for_sidecar_arrays() -> None:
    """Returned indices should address the original canonical-aligned sidecar arrays, not the filtered subset."""
    canonical = np.array([50, 150, 250, 350, 450], dtype=np.int64)
    pts_stream = np.array([500, 1500, 2500, 3500, 4500], dtype=np.int64)
    grid = np.array([140, 260, 340], dtype=np.int64)
    window = _window_from_grid(grid)
    indices, counts = sample_window_indices(canonical=canonical, window=window, dedup=False)

    # The eligible canonical subset is [150, 250], but the returned indices
    # must still refer to positions in the original canonical / pts_stream arrays.
    np.testing.assert_array_equal(indices, np.array([1, 2], dtype=np.int64))
    np.testing.assert_array_equal(canonical[indices], np.array([150, 250], dtype=np.int64))
    np.testing.assert_array_equal(pts_stream[indices], np.array([1500, 2500], dtype=np.int64))
    np.testing.assert_array_equal(counts, np.array([1, 1], dtype=np.int64))


def test_sample_window_indices_policy_tolerance_passes() -> None:
    """Matches within tolerance_ns should pass."""
    canonical = np.array([100, 200, 300], dtype=np.int64)
    grid = np.array([150, 205, 350], dtype=np.int64)
    window = _window_from_grid(grid)
    policy = SamplingPolicy(tolerance_ns=50)

    indices, counts = sample_window_indices(canonical=canonical, window=window, policy=policy, dedup=False)

    np.testing.assert_array_equal(indices, np.array([1, 1], dtype=np.int64))
    np.testing.assert_array_equal(canonical[indices], np.array([200, 200], dtype=np.int64))
    np.testing.assert_array_equal(counts, np.array([1, 1], dtype=np.int64))


def test_sample_window_indices_policy_tolerance_raises_with_offending_pair() -> None:
    """A tolerance failure should report the offending grid and canonical timestamps."""
    canonical = np.array([100, 200, 300], dtype=np.int64)
    grid = np.array([150, 260, 350], dtype=np.int64)
    window = _window_from_grid(grid)
    policy = SamplingPolicy(tolerance_ns=30)

    with pytest.raises(ValueError, match=r"tolerance_ns=30 exceeded: max delta was 50 ns for grid=150, canonical=200"):
        sample_window_indices(canonical=canonical, window=window, policy=policy)


def test_sample_window_indices_policy_returns_empty_when_no_canonical_timestamps_are_in_window() -> None:
    """An empty in-window result with policy should return empty arrays rather than raising."""
    canonical = np.array([100, 200, 300], dtype=np.int64)
    grid = np.array([400, 500], dtype=np.int64)
    window = _window_from_grid(grid)
    policy = SamplingPolicy(tolerance_ns=0)

    indices, counts = sample_window_indices(canonical=canonical, window=window, policy=policy)

    np.testing.assert_array_equal(indices, np.array([], dtype=np.int64))
    np.testing.assert_array_equal(counts, np.array([], dtype=np.int64))


@pytest.mark.parametrize(
    ("canonical", "grid", "match"),
    [
        (
            np.array([], dtype=np.int64),
            np.array([100, 200], dtype=np.int64),
            "canonical must be non-empty",
        ),
        (
            np.array([100, 200, 150], dtype=np.int64),
            np.array([100, 200], dtype=np.int64),
            "canonical must be strictly sorted in ascending order with no duplicates",
        ),
        (
            np.array([100, 200, 200, 300], dtype=np.int64),
            np.array([100, 200], dtype=np.int64),
            "canonical must be strictly sorted in ascending order with no duplicates",
        ),
    ],
)
def test_sample_window_indices_input_validation(
    canonical: npt.NDArray[np.int64],
    grid: npt.NDArray[np.int64],
    match: str,
) -> None:
    """sample_window_indices should reject invalid canonical and grid inputs."""
    window = _window_from_grid(grid)
    with pytest.raises(ValueError, match=match):
        sample_window_indices(canonical=canonical, window=window)


@pytest.mark.parametrize(
    ("canonical", "grid", "expected_canonical", "expected_indices", "expected_counts", "dedup"),
    [
        # Supersampling: more reference timestamps than canonical timestamps.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([110, 140, 240, 350], dtype=np.int64),
            np.array([200, 200, 200], dtype=np.int64),
            np.array([1, 1, 1], dtype=np.int64),
            np.array([1, 1, 1], dtype=np.int64),
            False,
        ),
        # Subsampling: fewer reference timestamps than canonical timestamps.
        (
            np.array([100, 150, 200, 250, 300], dtype=np.int64),
            np.array([140, 260, 350], dtype=np.int64),
            np.array([150, 250], dtype=np.int64),
            np.array([1, 3], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            True,
        ),
        # Jittery canonical timestamps sampled against a regular grid.
        (
            np.array([101, 199, 301, 399], dtype=np.int64),
            np.array([110, 210, 310, 410], dtype=np.int64),
            np.array([199, 301], dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
            np.array([2, 1], dtype=np.int64),
            True,
        ),
        # Sparse canonical timestamps inside a wider window.
        (
            np.array([100, 300], dtype=np.int64),
            np.array([110, 210, 310, 410], dtype=np.int64),
            np.array([300], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([3], dtype=np.int64),
            True,
        ),
        # Dense canonical timestamps with only a few active reference timestamps.
        (
            np.array([100, 120, 140, 160, 180, 200, 220, 240, 260], dtype=np.int64),
            np.array([115, 235, 300], dtype=np.int64),
            np.array([120, 240], dtype=np.int64),
            np.array([1, 7], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            True,
        ),
    ],
)
def test_sample_window_indices_shape_and_density_cases(  # noqa: PLR0913
    canonical: npt.NDArray[np.int64],
    grid: npt.NDArray[np.int64],
    expected_canonical: npt.NDArray[np.int64],
    expected_indices: npt.NDArray[np.int64],
    expected_counts: npt.NDArray[np.int64],
    *,
    dedup: bool,
) -> None:
    """sample_window_indices should behave predictably across realistic grid/canonical density patterns."""
    window = _window_from_grid(grid)
    indices, counts = sample_window_indices(canonical=canonical, window=window, dedup=dedup)

    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(canonical[indices], expected_canonical)
    np.testing.assert_array_equal(counts, expected_counts)

    assert np.all(indices >= 0)
    assert np.all(indices < len(canonical))


def test_sample_window_indices_differs_from_global_nearest_neighbour() -> None:
    """Window-local sampling should ignore a globally closer canonical timestamp that lies outside the window."""
    canonical = np.array([100, 200, 300], dtype=np.int64)
    grid = np.array([150, 260, 350], dtype=np.int64)
    window = _window_from_grid(grid)
    global_indices = find_closest_indices(canonical, window.timestamps_ns)
    window_indices, window_counts = sample_window_indices(canonical=canonical, window=window, dedup=False)

    # A global nearest-neighbour search would use 100 for the first reference
    # timestamp, because 100 is closer to 150 than 200 is.
    np.testing.assert_array_equal(global_indices, np.array([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(canonical[global_indices], np.array([100, 300], dtype=np.int64))

    # Window-local sampling ignores 100 because it lies outside [150, 350).
    np.testing.assert_array_equal(window_indices, np.array([1, 2], dtype=np.int64))
    np.testing.assert_array_equal(canonical[window_indices], np.array([200, 300], dtype=np.int64))
    np.testing.assert_array_equal(window_counts, np.array([1, 1], dtype=np.int64))
