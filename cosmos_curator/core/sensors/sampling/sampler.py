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

from cosmos_curator.core.sensors.alignment.nearest import nearest_timestamp_indices, timestamp_distance_ns
from cosmos_curator.core.sensors.exceptions import AlignmentError, AlignmentFailureReason
from cosmos_curator.core.sensors.sampling.grid import SamplingWindow
from cosmos_curator.core.sensors.sampling.policy import NearestTimestampPolicy
from cosmos_curator.core.sensors.utils.validation import require_strictly_increasing


def find_closest_indices(canonical: npt.NDArray[np.int64], grid: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Find the closest indices to values in canonical for each element in grid.

    If an element in grid is equidistant from two elements in canonical, the
    left index in canonical is used.

    This is a low-level nearest-neighbour helper only. It does not apply any
    sampling-window semantics or restrict ``canonical`` by timestamp range.

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

    return nearest_timestamp_indices(canonical, grid, source_name="canonical", requested_name="grid")


def sample_window_indices(
    canonical: npt.NDArray[np.int64],
    window: SamplingWindow,
    *,
    policy: NearestTimestampPolicy,
    dedup: bool = True,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Sample ``canonical`` using one window from ``grid`` and return indices into ``canonical``.

    Window semantics
    ----------------
    This function treats ``window`` as one sampling window emitted by
    :class:`~cosmos_curator.core.sensors.sampling.grid.SamplingGrid`.

    - ``window.timestamps_ns`` are the reference timestamps that belong to the current
      half-open window, and therefore the rows this call produces.
    - ``window.exclusive_end_ns`` is the exclusive right boundary marker.

    The window bounds choose reference timestamps. They do **not** restrict which
    canonical timestamps may serve them: every timestamp in ``canonical`` is
    eligible for every reference timestamp, and matching is plain
    nearest-neighbour. A window is a batching choice, so letting it filter
    ``canonical`` would make the selected data depend on ``stride_ns`` and
    ``duration_ns``.

    Reach is bounded by ``policy.max_delta_ns``, not by any span. Callers that
    can only materialise part of the timeline -- a forward-only decoder, say --
    bound it further by what they pass in ``canonical``; one observation beyond
    each window edge is enough to reproduce whole-timeline selection exactly.

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
        policy: Nearest-timestamp policy. When ``policy.max_delta_ns`` is not
            ``None``, each matched canonical timestamp must be within that
            maximum delta of its reference grid timestamp.
        dedup: Whether to deduplicate repeated canonical picks. When True,
            repeated matches are collapsed and ``counts[i]`` records how many
            reference timestamps mapped to ``canonical[indices[i]]``.

    Returns:
        Tuple ``(indices, counts)``.

        - ``indices`` are indices into the original ``canonical`` array.
        - ``counts`` records multiplicity for each returned canonical index.

        With ``dedup=False`` there is one index per reference timestamp. If the
        window carries no reference timestamps, returns two empty ``int64``
        arrays.

    Raises:
        ValueError: If ``canonical`` is empty.
        ValueError: If ``canonical`` is not strictly increasing.
        ValueError: If ``window.timestamps_ns`` is not strictly increasing.
        TypeError: If ``policy`` is not a ``NearestTimestampPolicy``.
        AlignmentError: With reason ``TOLERANCE_EXCEEDED`` if ``policy.max_delta_ns``
            is not ``None`` and any matched canonical timestamp exceeds it from its
            reference timestamp.

    """
    if not isinstance(policy, NearestTimestampPolicy):
        msg = f"policy must be NearestTimestampPolicy, got {type(policy).__name__}"  # type: ignore[unreachable]
        raise TypeError(msg)

    if len(canonical) < 1:
        msg = "canonical must be non-empty"
        raise ValueError(msg)

    # A window with no reference timestamps produces no rows. `len(window)` is
    # `len(window.timestamps_ns)`, so this is the only emptiness check needed.
    if len(window) == 0:
        require_strictly_increasing("canonical", canonical)
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    active_grid = window.timestamps_ns

    # The window bounds deliberately do not filter `canonical`. They say which
    # reference timestamps belong to this batch, not which observations may serve
    # them. Filtering here would make selection depend on stride_ns/duration_ns,
    # which are a batching choice.
    indices = find_closest_indices(canonical, active_grid)

    if policy.max_delta_ns is not None:
        deltas = timestamp_distance_ns(canonical[indices], active_grid)
        if np.any(deltas > policy.max_delta_ns):
            worst_idx = int(deltas.argmax())
            max_delta = int(deltas[worst_idx])
            grid_ts = int(active_grid[worst_idx])
            canonical_ts = int(canonical[indices[worst_idx]])
            msg = (
                f"max_delta_ns={policy.max_delta_ns} exceeded: "
                f"max delta was {max_delta} ns for grid={grid_ts}, canonical={canonical_ts}"
            )
            # No sensor id is available here; SensorGroup stamps its configured one.
            raise AlignmentError(
                AlignmentFailureReason.TOLERANCE_EXCEEDED,
                msg,
                align_timestamps_ns=active_grid,
                sensor_timestamps_ns=canonical[indices],
                delta_ns=max_delta,
                max_delta_ns=policy.max_delta_ns,
            )

    if dedup:
        # Collapse repeated matches of the same canonical timestamp. The counts
        # tell the caller how many reference timestamps in this window mapped
        # to that canonical sample.
        indices, counts = np.unique(indices, return_counts=True)
    else:
        # Keep one output row per reference timestamp.
        counts = np.ones_like(indices, dtype=np.int64)

    return indices, counts
