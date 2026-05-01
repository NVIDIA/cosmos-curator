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
"""Timestamp sampling grid."""

import math
from collections.abc import Iterator
from typing import Any, Protocol

import attrs
import numpy as np
import numpy.typing as npt
from attrs import validators

from cosmos_curator.core.sensors.utils.helpers import as_readonly_view
from cosmos_curator.core.sensors.utils.validation import (
    positive_value,
    strictly_increasing_int64_array,
)


class _HasHalfOpenWindowBounds(Protocol):
    start_ns: int
    exclusive_end_ns: int


def _end_ns_ge_start_ns(
    instance: _HasHalfOpenWindowBounds,
    _attribute: object,
    value: int,
) -> None:
    if value < instance.start_ns:
        msg = f"end_ns must be greater than or equal to start_ns, got {instance.start_ns=} {value=}"
        raise ValueError(msg)


def make_ts_grid(
    start_ns: int,
    end_ns: int | None = None,
    sample_rate_hz: float | None = None,
    *,
    exclusive_end_ns: int | None = None,
) -> tuple[int, int, npt.NDArray[np.int64]]:
    """Make a grid of timestamps in nanoseconds.

    Samples are ``start_ns + k * (1/sample_rate_hz)`` in float space, rounded
    to int64 nanoseconds. The grid always includes ``start_ns`` and the
    returned ``timestamps_ns`` is strictly ascending and read-only.

    Exactly one of ``end_ns`` and ``exclusive_end_ns`` must be supplied.

    Inclusive end (``end_ns``):
        ``end_ns`` is the last timestamp to *include*. The grid continues
        until the final sample is strictly greater than ``end_ns``; that
        sample is retained as an exclusive boundary marker so the requested
        ``end_ns`` remains reachable under half-open sampling-window
        semantics, even when ``end_ns - start_ns`` is not evenly divisible by
        the sample interval. The returned tuple satisfies
        ``timestamps_ns[-1] <= end_ns < exclusive_end_ns``.

    Exclusive end (``exclusive_end_ns``):
        ``exclusive_end_ns`` is the half-open right boundary; the grid stops
        strictly before it. The supplied value is returned unchanged as the
        second element of the tuple, which makes this convenient for clip
        spans that already use ``[start_s, end_s)`` semantics. The returned
        tuple satisfies ``timestamps_ns[-1] < exclusive_end_ns`` and
        ``exclusive_end_ns`` equals the supplied value exactly.

    Args:
        start_ns: the start timestamp in nanoseconds.
        end_ns: optional inclusive end timestamp in nanoseconds. Mutually
            exclusive with ``exclusive_end_ns``.
        sample_rate_hz: the sample rate in Hz. Required.
        exclusive_end_ns: optional exclusive end timestamp in nanoseconds.
            Mutually exclusive with ``end_ns``.

    Returns:
        A ``(start_ns, exclusive_end_ns, timestamps_ns)`` tuple where
        ``timestamps_ns`` is a strictly ascending, read-only ``int64`` array.

    Raises:
        ValueError: if neither or both of ``end_ns`` and ``exclusive_end_ns``
            are supplied, if ``sample_rate_hz`` is missing or non-positive,
            if the supplied bound precedes ``start_ns``, or if rounding to
            nanoseconds does not produce a strictly increasing grid.

    """
    if sample_rate_hz is None or sample_rate_hz <= 0:
        msg = f"sample_rate_hz must be greater than 0, got {sample_rate_hz=}"
        raise ValueError(msg)
    if end_ns is not None and exclusive_end_ns is not None:
        msg = f"exactly one of end_ns or exclusive_end_ns must be supplied, got both {end_ns=} and {exclusive_end_ns=}"
        raise ValueError(msg)

    # Reduce the half-open form to the existing inclusive-end algorithm by
    # shifting one nanosecond inward, then override the returned boundary so
    # the supplied exclusive_end_ns is preserved exactly.
    if exclusive_end_ns is not None:
        if exclusive_end_ns <= start_ns:
            msg = f"exclusive_end_ns must be greater than start_ns, got {start_ns=} {exclusive_end_ns=}"
            raise ValueError(msg)
        inclusive_end_ns = exclusive_end_ns - 1
    elif end_ns is not None:
        if end_ns < start_ns:
            msg = f"end_ns must be greater than or equal to start_ns, got {start_ns=} {end_ns=}"
            raise ValueError(msg)
        inclusive_end_ns = end_ns
    else:
        msg = "exactly one of end_ns or exclusive_end_ns must be supplied, got neither"
        raise ValueError(msg)

    sample_interval = 1.0 / sample_rate_hz
    start = start_ns / 1_000_000_000
    end = inclusive_end_ns / 1_000_000_000

    # Calculate the number of samples needed to cover the range, guarding against
    # floating-point roundoff at exact boundaries
    intervals_to_end = np.nextafter((end - start) / sample_interval, np.inf)
    sample_intervals_to_end = max(2, math.floor(intervals_to_end) + 2)

    # Build the grid of timestamps in float space, then round to int64 nanoseconds
    timestamps_s = start + np.arange(sample_intervals_to_end, dtype=np.float64) * sample_interval
    retval = np.round(timestamps_s * 1_000_000_000).astype(np.int64)
    if np.any(np.diff(retval) <= 0):
        msg = (
            "sample_rate_hz does not produce a strictly increasing nanosecond grid after rounding, "
            f"got {sample_rate_hz=}"
        )
        raise ValueError(msg)
    retval.flags.writeable = False

    start_ns = int(retval[0])
    timestamps_ns = retval[:-1]
    out_exclusive_end_ns = exclusive_end_ns if exclusive_end_ns is not None else int(retval[-1])
    return start_ns, out_exclusive_end_ns, timestamps_ns


def _start_ns_le_first_timestamp(
    instance: _HasHalfOpenWindowBounds,
    _attribute: object,
    value: npt.NDArray[np.int64],
) -> None:
    """Validate start_ns is less than or equal to the first timestamp."""
    if len(value) == 0:
        return
    first_ts = int(value[0])
    if instance.start_ns > first_ts:
        msg = f"start_ns must be <= timestamps_ns[0], got {instance.start_ns} > {first_ts}"
        raise ValueError(msg)


def _start_ns_eq_first_timestamp(
    instance: _HasHalfOpenWindowBounds,
    _attribute: object,
    value: npt.NDArray[np.int64],
) -> None:
    """Validate start_ns is equal to the first timestamp."""
    if len(value) == 0:
        return
    first_ts = int(value[0])
    if instance.start_ns != first_ts:
        msg = f"start_ns must == timestamps_ns[0], got {instance.start_ns} != {first_ts}"
        raise ValueError(msg)


def _end_ns_lt_exclusive_end_ns(
    instance: _HasHalfOpenWindowBounds,
    _attribute: object,
    value: npt.NDArray[np.int64],
) -> None:
    if len(value) == 0:
        return
    end_ns = int(value[-1])
    if end_ns >= instance.exclusive_end_ns:
        msg = f"end_ns must be < exclusive_end_ns, got {end_ns} >= {instance.exclusive_end_ns}"
        raise ValueError(msg)


def _copy_numpy_array(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Copy a numpy array."""
    return np.array(array, copy=True)


def _copy_as_readonly_view(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Copy a numpy array and expose the copy as a read-only view."""
    return as_readonly_view(_copy_numpy_array(array))


@attrs.define(frozen=True)
class SamplingWindow:
    """One half-open sampling window `[start_ns, exclusive_end_ns)`.

    For non-empty windows, the timestamps are strictly increasing and satisfy
    ``start_ns <= timestamps_ns[0]`` and
    ``timestamps_ns[-1] < exclusive_end_ns``.

    Attributes:
        start_ns:
            Left boundary of the half-open interval, must be less than or
            equal to the first timestamp in timestamps_ns.
        exclusive_end_ns:
            Exclusive right boundary of the window, must be greater than the
            last timestamp in timestamps_ns.
        timestamps_ns:
            Strictly increasing ``int64`` timestamps. For non-empty windows,
            the first timestamp must be greater than or equal to ``start_ns``
            and the last timestamp must be strictly less than
            ``exclusive_end_ns``.

    """

    start_ns: int
    exclusive_end_ns: int = attrs.field(validator=_end_ns_ge_start_ns)
    timestamps_ns: npt.NDArray[np.int64] = attrs.field(
        converter=as_readonly_view,
        validator=validators.and_(
            strictly_increasing_int64_array,
            _start_ns_le_first_timestamp,
            _end_ns_lt_exclusive_end_ns,
        ),
    )

    def __len__(self) -> int:
        """Return the number of timestamps in this window."""
        return len(self.timestamps_ns)


@attrs.define(frozen=True, hash=False)
class SamplingGrid:
    """Iterable view over timestamped sampling windows.

    ``SamplingGrid`` turns a strictly increasing timestamp series into a
    sequence of half-open windows whose nominal starts are
    ``start_ns + k * stride_ns``. Iteration yields :class:`SamplingWindow`
    objects that preserve the exclusive right boundary needed by downstream
    samplers.

    The common use case is a regular timestamp grid that includes one extra
    boundary marker strictly after the final sample that should remain
    reachable under half-open window semantics. :func:`make_ts_grid` produces
    timestamps in that format automatically. The iterator is also intentionally
    permissive enough to support irregular or bursty sensor timelines.

    Empty windows are yielded instead of being filtered out. That preserves the
    invariant that window index ``i`` always corresponds to the nominal time
    range starting at ``start_ns + i * stride_ns``. On sparse or irregular
    grids, consecutive yielded windows may therefore look identical even though
    their nominal time bounds differ; this is expected.

    Attributes:
        start_ns:
            Left boundary of the half-open interval, must be equal to the first
            timestamp in timestamps_ns.
        exclusive_end_ns:
            Exclusive right boundary of the window, must be greater than the
            last timestamp in timestamps_ns.
        timestamps_ns:
            One-dimensional, strictly increasing ``int64`` timestamp array in
            nanoseconds.
        stride_ns:
            Distance in nanoseconds between consecutive nominal window starts.
            Must be positive.
        duration_ns:
            Width in nanoseconds of each sampling window. Must be positive.

    """

    __hash__ = None  # type: ignore[assignment]
    start_ns: int
    exclusive_end_ns: int = attrs.field(validator=_end_ns_ge_start_ns)
    timestamps_ns: npt.NDArray[np.int64] = attrs.field(
        converter=_copy_as_readonly_view,
        validator=validators.and_(
            strictly_increasing_int64_array,
            _start_ns_eq_first_timestamp,
            _end_ns_lt_exclusive_end_ns,
        ),
    )
    stride_ns: int = attrs.field(validator=positive_value)
    duration_ns: int = attrs.field(validator=positive_value)

    def __iter__(self) -> Iterator[SamplingWindow]:
        """Iterate over timestamp windows on the timeline.

        The nominal sampling-window start times are
        ``start_ns + k * stride_ns`` for ``k = 0, 1, ...``, and the nominal
        window end times are ``start_ns + k * stride_ns + duration_ns``.

        Each yielded :class:`SamplingWindow` stores active timestamps in
        ``window.timestamps_ns`` and the nominal half-open bounds in
        ``window.start_ns`` and ``window.exclusive_end_ns``. The bounds advance
        by ``stride_ns`` even when a window is empty or its timestamps begin
        later within that interval.

        Interpretation:
        - ``window.exclusive_end_ns`` is the exclusive right boundary marker.
        - If a timestamp lands exactly on a window boundary, it is sampled in
          the next window, not the current one.

        This prevents double-counting across adjacent windows.

        Empty windows:
        - A yielded window may have ``len(window) == 0`` if no active
          timestamps fall in that time range.
        - Empty windows are kept so window index ``i`` still maps to the time window
          starting at ``start_ns + i * stride_ns``.

        Boundary-only case:
        - A yielded window may have zero active timestamps while still carrying
          a real boundary marker in ``exclusive_end_ns``. This represents a
          valid half-open sampling window with no active timestamps assigned to
          it.

        Yields:
            ``SamplingWindow`` describing one half-open sampling window.

        """
        if self.start_ns == self.exclusive_end_ns:
            yield SamplingWindow(
                start_ns=self.start_ns,
                exclusive_end_ns=self.exclusive_end_ns,
                timestamps_ns=self.timestamps_ns[:-1],
            )
            return

        start_ns = self.start_ns
        ts = np.concatenate([self.timestamps_ns, [self.exclusive_end_ns]])
        while start_ns < self.exclusive_end_ns:
            window_end_ns = min(start_ns + self.duration_ns, self.exclusive_end_ns)
            i, j = np.searchsorted(ts, [start_ns, window_end_ns], side="left")
            yield SamplingWindow(start_ns=start_ns, exclusive_end_ns=window_end_ns, timestamps_ns=ts[i:j])
            start_ns += self.stride_ns
