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
"""Sampling functions for the sensor library."""

import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.sampling.grid import SamplingWindow
from cosmos_curator.core.sensors.sampling.policy import SamplingPolicy
from cosmos_curator.core.sensors.utils.validation import require_strictly_increasing


def find_closest_indices(canonical: npt.NDArray[np.int64], grid: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Find the closest indices to values in canonical for each element in grid.

    If an element in grid is equidistant from two elements in canonical, the
    left index in canonical is used.

    This is a low-level nearest-neighbour helper only. It does not apply any
    sampling-window semantics or restrict ``canonical`` by timestamp range.
    Callers that need window-local matching must filter ``canonical`` before
    calling this function.

    Args:
        canonical: The canonical timestamps to sample from. Must be strictly
            increasing.
        grid: The sampling grid. Must be strictly increasing.

    Returns:
        Array of closest indices in canonical for each element in grid.

    """
    if len(canonical) == 0:
        msg = "canonical must be non-empty"
        raise ValueError(msg)

    if len(grid) == 0:
        msg = "grid must be non-empty"
        raise ValueError(msg)

    require_strictly_increasing("canonical", canonical)
    require_strictly_increasing("grid", grid)

    if len(canonical) == 1:
        return np.zeros_like(grid, dtype=np.int64)

    # Rightmost indices are the insertion points into sorted array
    right_idx = np.searchsorted(canonical, grid)
    right_idx = np.clip(right_idx, 1, len(canonical) - 1)

    # leftmost elements now, becomes closest index later
    closest_idx = right_idx - 1

    # Compare distances to left and right neighbors
    left = canonical[closest_idx]
    right = canonical[right_idx]
    right_closest = np.abs(grid - right) < np.abs(grid - left)
    closest_idx[right_closest] = right_idx[right_closest]

    return closest_idx.astype(np.int64)


def sample_window_indices(
    canonical: npt.NDArray[np.int64],
    window: SamplingWindow,
    *,
    policy: SamplingPolicy | None = None,
    dedup: bool = True,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Sample ``canonical`` using one window from ``grid`` and return indices into ``canonical``.

    Window-local semantics
    ----------------------
    This function treats ``grid`` as one sampling window emitted by
    :class:`~cosmos_curator.core.sensors.sampling.grid.SamplingGrid`.

    - ``window.timestamps_ns`` are the reference timestamps that belong to the current
      half-open window.
    - ``window.exclusive_end_ns`` is the exclusive right boundary marker.
    - Eligible canonical timestamps are restricted to the same half-open
      interval ``[window.timestamps_ns[0], window.exclusive_end_ns)``.

    In other words, this function performs nearest-neighbour matching only
    within the current window. Canonical timestamps outside the window are
    ignored, even if one of them would be closer in absolute time to a
    reference timestamp in ``window.timestamps_ns``.

    Return value semantics
    ----------------------
    The returned ``indices`` always refer to the original ``canonical`` array
    passed by the caller, not to an internal filtered sub-array. This allows
    callers to reuse the indices to look up aligned sidecar arrays such as
    ``pts_stream`` or frame payloads that are stored in parallel with
    ``canonical``.

    Args:
        canonical: Full canonical timestamp timeline for one sensor. Must be
            strictly increasing.
        window: One strictly increasing sampling window. window.exclusive_end_ns
            is an exclusive right boundary marker and is not sampled.
        policy: Optional sampling policy. When provided, each matched canonical
            timestamp must be within ``policy.tolerance_ns`` of its reference
            grid timestamp.
        dedup: Whether to deduplicate repeated canonical picks. When True,
            repeated matches are collapsed and ``counts[i]`` records how many
            reference timestamps mapped to ``canonical[indices[i]]``.

    Returns:
        Tuple ``(indices, counts)``.

        - ``indices`` are indices into the original ``canonical`` array.
        - ``counts`` records multiplicity for each returned canonical index.

        If there are no eligible canonical timestamps in the current window,
        returns two empty ``int64`` arrays.

    Raises:
        ValueError: If ``canonical`` is empty.
        ValueError: If ``canonical`` is not strictly increasing.
        ValueError: If ``window.timestamps_ns`` is not strictly increasing.
        ValueError: If ``policy`` is provided and any matched canonical
            timestamp exceeds ``policy.tolerance_ns`` from its reference
            timestamp.

    """
    if len(canonical) < 1:
        msg = "canonical must be non-empty"
        raise ValueError(msg)

    require_strictly_increasing("canonical", canonical)

    if len(window) < 1:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    # `grid[-1]` is the exclusive boundary marker for the half-open interval.
    # The actual reference timestamps to sample in this window are `grid[:-1]`.
    active_grid = window.timestamps_ns

    # If the current window contains only the boundary marker, there are no
    # reference timestamps to sample.
    if len(active_grid) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    # Build a boolean mask selecting only canonical timestamps that are
    # eligible for this window.
    #
    # Window-local contract:
    #   eligible canonical timestamps are those in [grid[0], grid[-1])
    eligible_mask = (canonical >= window.timestamps_ns[0]) & (canonical < window.exclusive_end_ns)

    # Convert the mask into integer indices into the ORIGINAL canonical array.
    # We keep these indices so that after matching on the filtered subset, we
    # can map the results back to original-array coordinates for the caller.
    eligible_indices = np.nonzero(eligible_mask)[0]

    # Pull out just the in-window canonical timestamps for nearest-neighbour
    # matching.
    eligible_canonical = canonical[eligible_indices]

    # No eligible canonical timestamps means this sensor has no data in the
    # current window, so return an empty result rather than sampling from a
    # neighbouring window.
    if len(eligible_canonical) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    # Perform nearest-neighbour matching against ONLY the eligible canonical
    # timestamps from this window.
    #
    # Important:
    #   `local_indices` are indices into `eligible_canonical`, not into the
    #   caller's original `canonical` array.
    local_indices = find_closest_indices(eligible_canonical, active_grid)

    # Map the subset-local indices back to indices into the ORIGINAL canonical
    # array, so callers can use them to index sidecar arrays that share the
    # same layout as `canonical`.
    indices = eligible_indices[local_indices]

    if policy is not None:
        deltas = np.abs(canonical[indices] - active_grid)
        if np.any(deltas > policy.tolerance_ns):
            worst_idx = int(deltas.argmax())
            max_delta = int(deltas[worst_idx])
            grid_ts = int(active_grid[worst_idx])
            canonical_ts = int(canonical[indices[worst_idx]])
            msg = (
                f"tolerance_ns={policy.tolerance_ns} exceeded: "
                f"max delta was {max_delta} ns for grid={grid_ts}, canonical={canonical_ts}"
            )
            raise ValueError(msg)

    if dedup:
        # Collapse repeated matches of the same canonical timestamp. The counts
        # tell the caller how many reference timestamps in this window mapped
        # to that canonical sample.
        indices, counts = np.unique(indices, return_counts=True)
    else:
        # Keep one output row per reference timestamp.
        counts = np.ones_like(indices, dtype=np.int64)

    return indices, counts
