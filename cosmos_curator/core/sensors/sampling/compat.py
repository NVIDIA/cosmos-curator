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
"""Compatibility helpers for legacy sampling behavior."""

import numpy as np
import numpy.typing as npt

_NS_PER_SECOND = 1_000_000_000


def make_decoder_utils_compat_grid(
    start_ns: int,
    stop_ns: int,
    sample_rate_hz: float,
    *,
    endpoint: bool = True,
) -> tuple[int, int, npt.NDArray[np.int64]]:
    """Build a ``SamplingGrid`` timestamp tuple close to ``decoder_utils.sample_closest``.

    The returned timestamps are an int64-native approximation of the ideal
    decoder-utils sampling timeline, before nearest-neighbour snapping to
    source frames. Duplicate source-frame selections are therefore represented
    later by sampler counts, not by duplicate timestamps in this grid.

    This compatibility shim is intended to be temporary. It should be removed
    once pipelines have migrated from ``decoder_utils`` sampling semantics to
    the sensor library's native sampling semantics.

    Args:
        start_ns: First ideal sample timestamp, in nanoseconds.
        stop_ns: Legacy decoder-utils stop timestamp, in nanoseconds.
        sample_rate_hz: Sampling rate in hertz.
        endpoint: Whether to allow the stop timestamp to be sampled when it
            fits the sample cadence.

    Returns:
        A ``(start_ns, exclusive_end_ns, timestamps_ns)`` tuple suitable for a
        full-clip ``SamplingGrid``.

    Raises:
        ValueError: If the sample rate is non-positive, ``stop_ns`` precedes
            ``start_ns``, or the sample rate cannot produce a strictly
            increasing nanosecond grid.

    """
    if sample_rate_hz <= 0:
        msg = f"sample_rate_hz must be greater than 0, got {sample_rate_hz=}"
        raise ValueError(msg)
    if stop_ns < start_ns:
        msg = f"stop_ns must be greater than or equal to start_ns, got {start_ns=} {stop_ns=}"
        raise ValueError(msg)

    sample_interval_ns = round(_NS_PER_SECOND / sample_rate_hz)
    if sample_interval_ns < 1:
        msg = (
            "sample_rate_hz does not produce a strictly increasing nanosecond grid "
            f"after rounding, got {sample_rate_hz=}"
        )
        raise ValueError(msg)

    exclusive_end_ns = stop_ns
    if endpoint:
        exclusive_end_ns += max(1, sample_interval_ns // 2)

    timestamps_ns = np.arange(start_ns, exclusive_end_ns, sample_interval_ns, dtype=np.int64)
    # Note: it's not clear that this is the intended behavior, but this matches make_ts_grid.
    timestamps_ns.flags.writeable = False
    return int(start_ns), exclusive_end_ns, timestamps_ns
