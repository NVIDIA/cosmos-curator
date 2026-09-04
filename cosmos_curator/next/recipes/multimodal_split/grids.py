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

"""Clip geometry: how a session is tiled and how a clip's grid is built.

Shared by planning, which decides which spans exist, and splitting, which
samples them. Neither owns it: a span and the grid inside it are the same
arithmetic whichever stage is asking.
"""

from decimal import Decimal

import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.sampling.grid import SamplingGrid, make_ts_grid
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.next.media.spans import Span, fixed_stride_spans
from cosmos_curator.next.recipes.multimodal_split.config import ResolvedMultimodalSplitConfig

_NS_PER_SECOND = 1_000_000_000
# One timestamp cannot show a gap, so the period is unknowable and the exclusive
# end is the only thing that can be said.
_MIN_TIMESTAMPS_FOR_A_GAP = 2


def plan_clip_spans(*, start_ns: int, end_ns: int, duration_ns: int) -> tuple[Span, ...]:
    """Tile ``[start_ns, end_ns)`` into contiguous, full-length clip spans.

    Clips do not overlap and no trailing partial clip is kept, so the stride and
    the minimum length are both the clip duration. Spans are returned on the
    camera's own timeline rather than relative to it, because every later step --
    the sampling grid, the descriptors, the diagnostics -- speaks in source
    timestamps.
    """
    if end_ns < start_ns:
        msg = f"end_ns must be greater than or equal to start_ns, got {start_ns=} {end_ns=}"
        raise ValueError(msg)
    relative = fixed_stride_spans(
        end_ns - start_ns,
        duration_ns=duration_ns,
        stride_ns=duration_ns,
        min_duration_ns=duration_ns,
    )
    return tuple(Span(start_ns=span.start_ns + start_ns, end_ns=span.end_ns + start_ns) for span in relative)


def timeline_end_ns(timestamps_ns: npt.NDArray[np.int64]) -> int:
    """Return the exclusive end of what the camera recorded.

    The last entry is the final frame's timestamp, but a frame occupies the
    period after it, so a camera's coverage runs one period past. Tiling to the last
    timestamp instead discards a whole clip whenever that coverage lands within
    one frame period of a clip multiple, which is every fixed-length chunk.
    """
    if timestamps_ns.size < _MIN_TIMESTAMPS_FOR_A_GAP:
        return int(timestamps_ns[-1]) + 1
    # From the count over the observed rate, not by adding a period to the last
    # entry. Timestamps are whole nanoseconds, so a rounded period accumulates:
    # on a three-second 30 fps camera it lands a nanosecond short of the end,
    # which is a whole clip.
    span_ns = int(timestamps_ns[-1]) - int(timestamps_ns[0])
    period_ns = span_ns / (timestamps_ns.size - 1)
    return int(timestamps_ns[0]) + round(timestamps_ns.size * period_ns)


def sampling_spec(span: Span, config: ResolvedMultimodalSplitConfig) -> SamplingSpec:
    """Build the clip's grid as a single sampling window over the whole span.

    Window bounds decide which grid timestamps are handed over together; they do
    not restrict which source frames may serve them, so one window per clip is a
    batching choice and not a selection one.
    """
    start_ns, exclusive_end_ns, timestamps_ns = make_ts_grid(
        span.start_ns,
        exclusive_end_ns=span.end_ns,
        sample_rate_hz=config.clip.output_fps,
    )
    grid = SamplingGrid(
        start_ns,
        exclusive_end_ns,
        timestamps_ns,
        stride_ns=span.duration_ns,
        duration_ns=span.duration_ns,
    )
    return SamplingSpec(grid=grid)


def clip_duration_ns(config: ResolvedMultimodalSplitConfig) -> int:
    """Return the clip duration in the integer nanoseconds every later step uses.

    Through ``Decimal`` rather than float, matching how the config validated the
    same value: ``65.693362 * 1e9`` is ``65693361999.99999`` in binary floating
    point, which would put every span boundary one nanosecond early.
    """
    return int(Decimal(str(config.clip.duration_s)) * _NS_PER_SECOND)


def sampling_grid_ns(span: Span, config: ResolvedMultimodalSplitConfig) -> npt.NDArray[np.int64]:
    """Return the timestamps a clip's frames are resampled onto."""
    return sampling_spec(span, config).grid.timestamps_ns
