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

"""The argparse surface, exit codes and Ctrl-C handling both data-integrity CLIs share.

Shared by the single-video ``di-check`` CLI (:mod:`.cli`) and the single-session
``di-session`` tool (:mod:`.session_cli`) so that a policy flag never means two
different things, an exit code never has two variants of the same status, and an
operator abort behaves identically whichever entry point they ran.

``Thresholds`` / ``DEFAULT_THRESHOLDS`` are re-exported from here for callers that
already import the policy alongside the flags that populate it; they are defined in
:mod:`cosmos_curator.core.sensors.data_integrity.instruments`.

The compute this wraps lives in
:mod:`cosmos_curator.core.sensors.data_integrity.engine`, and the URI-level IO in
:mod:`.sources`.
"""

import argparse
import io
import math
import os
import signal
import sys
import threading
import types
from collections.abc import Generator
from contextlib import contextmanager
from typing import BinaryIO, cast

from cosmos_curator.core.sensors.data_integrity.instruments import DEFAULT_THRESHOLDS, MAX_PERCENT, Thresholds

# Shared process exit codes for both CLIs. Keep them together so a wrapper can
# depend on one contract: PASS / FAIL / ERROR / interrupted, never two variants
# of the same status.
# INTERRUPTED = 128 + SIGINT: the shell convention for "terminated by Ctrl-C".
# Distinct from ERROR so a wrapper can tell an operator abort from a genuine failure to evaluate.
PASS_EXIT_CODE = 0
FAIL_EXIT_CODE = 1
ERROR_EXIT_CODE = 2
INTERRUPTED_EXIT_CODE = 130


def positive_finite_float(raw: str) -> float:
    """Argparse ``type=`` for flags that must be strictly positive and finite (rejects 0, NaN, inf)."""
    msg = f"expected a positive finite number, got {raw!r}"
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(msg) from None
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(msg)
    return value


def non_negative_finite_float(raw: str) -> float:
    """Argparse ``type=`` for flags that must be non-negative and finite (rejects negatives, NaN, inf).

    Distinct from :func:`positive_finite_float` because zero is meaningful for a
    tolerance: ``--max-jitter-percent 0`` asks for an exactly uniform cadence.
    """
    msg = f"expected a non-negative finite number, got {raw!r}"
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(msg) from None
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError(msg)
    return value


def share_percent(raw: str) -> float:
    """Argparse ``type=`` for a flag whose value is a share of something, in percent.

    Distinct from :func:`non_negative_finite_float` by its ceiling: a share cannot
    exceed :data:`~.instruments.MAX_PERCENT`, so a larger limit describes no session
    that exists rather than a more lenient policy. ``100`` already tolerates
    everything, which is what makes anything above it a typo rather than an intent.
    The bound comes from the registry so this and the config model cannot disagree
    about it.
    """
    msg = f"expected a number between 0 and {MAX_PERCENT:g}, got {raw!r}"
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(msg) from None
    if not math.isfinite(value) or value < 0.0 or value > MAX_PERCENT:
        raise argparse.ArgumentTypeError(msg)
    return value


def non_negative_int(raw: str) -> int:
    """Argparse ``type=`` for flags that must be a non-negative integer (rejects negatives that would wrap)."""
    msg = f"expected a non-negative integer, got {raw!r}"
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(msg) from None
    if value < 0:
        raise argparse.ArgumentTypeError(msg)
    return value


def positive_int(raw: str) -> int:
    """Argparse ``type=`` for flags that must be a positive integer (rejects 0 and negatives)."""
    msg = f"expected a positive integer, got {raw!r}"
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(msg) from None
    if value < 1:
        raise argparse.ArgumentTypeError(msg)
    return value


def non_empty_str(raw: str) -> str:
    """Argparse ``type=`` for flags whose empty value would be taken for a real one.

    A blank pattern is the case worth catching: argparse would accept it, discovery
    would match nothing against it, and the session would be reported as holding no
    streams -- a finding about the data rather than the input error it is. The config
    block rejects blanks for the same reason, so the two entrypoints agree.
    """
    value = raw.strip()
    if not value:
        msg = f"expected a non-empty value, got {raw!r}"
        raise argparse.ArgumentTypeError(msg)
    return value


@contextmanager
def interrupt_guard() -> Generator[threading.Event]:
    """Turn SIGINT into a cooperative stop request, yielding the event that carries it.

    The first Ctrl-C only sets the event; it deliberately does not raise. Python's
    default handler raises ``KeyboardInterrupt`` wherever the interpreter happens to
    be, and under this CLI that is usually inside one of libav's IO callbacks -- which
    cannot carry a Python exception back out. What the operator gets instead is a
    swallowed exception printed as a traceback, followed by libav either relabelling
    the abort as an ``InvalidDataError`` (blaming the file) or recovering and finishing
    the run as though nothing had been asked of it. Neither is an interruption.

    So the stop is cooperative: readers abandon their source at the next boundary they
    control (:func:`cancellable_reader`) and callers consult the event to decide the
    outcome. Nothing is raised through libav, so nothing is printed by it.

    A second Ctrl-C raises ``KeyboardInterrupt`` in the classic way, on the assumption
    that an operator pressing it again wants out regardless of the mess -- the escape
    hatch for anything that never reaches a read, such as a stalled bucket listing.

    An event rather than a flag because worker threads read it too: only the main
    thread receives the signal. The previous handler is restored on exit, keeping this
    usable inside a library caller's process, and installation is only valid on the
    main thread, which is where :mod:`signal` permits it.
    """
    interrupted = threading.Event()

    def _on_sigint(_signum: int, _frame: types.FrameType | None) -> None:
        if interrupted.is_set():
            raise KeyboardInterrupt
        interrupted.set()

    previous = signal.signal(signal.SIGINT, _on_sigint)
    try:
        yield interrupted
    finally:
        signal.signal(signal.SIGINT, previous)


def raise_if_interrupted(cancel: threading.Event | None) -> None:
    """Raise ``KeyboardInterrupt`` when a cooperative stop has been requested.

    Call after a unit of work so a swallowed abort is not mistaken for a finished
    run: libav can absorb the signal and return successfully on a truncated source.
    ``None`` is a no-op so library callers that never install a handler stay quiet.
    """
    if cancel is not None and cancel.is_set():
        raise KeyboardInterrupt


def report_interrupted() -> int:
    """Write the operator-abort message to stderr and return :data:`INTERRUPTED_EXIT_CODE`.

    No report body is emitted: a partial or truncated verdict would be easy to
    misread as the real answer. The leading newline clears the terminal's echoed
    ``^C`` (and any progress line left mid-redraw).
    """
    sys.stderr.write("\ninterrupted: no report written\n")
    return INTERRUPTED_EXIT_CODE


def report_error(message: str) -> int:
    """Write a one-line operator-facing error to stderr and return :data:`ERROR_EXIT_CODE`.

    Shared so the ``error: `` prefix that wrappers grep for stays identical across both
    CLIs and across every failure mode within them. Callers pass only the description;
    what could not be done belongs at the call site, which knows what it was attempting.
    """
    sys.stderr.write(f"error: {message}\n")
    return ERROR_EXIT_CODE


class _CancellableReader(io.BufferedIOBase):
    """Binary stream wrapper that reads as an exhausted source once a cancel event is set.

    A thread blocked in libav cannot be preempted, so a cancellation is only as prompt
    as the next boundary the reader itself controls. Checking one flag before each read
    turns an abandoned session's tail from "as long as the largest in-flight file still
    needs" into a single read's worth of latency. The check precedes the call so no
    further range request is issued; the read already in progress still has to return.

    Cancellation surfaces as end-of-file rather than as an exception because libav
    cannot carry a Python exception out of its read callback: raising leaves the process
    printing an unraisable-exception traceback for every abandoned stream, whereas a
    short read is something a demuxer already knows how to end on. Reporting a source
    as shorter than it is would be dangerous if anyone trusted the resulting index, so
    it is safe only because the caller re-checks the same event and discards the result
    (see :func:`~cosmos_curator.next.recipes.data_integrity.session_runner.run_session`).

    ``read1`` / ``readinto`` route through ``read`` so the check holds whichever access
    pattern PyAV uses, ``readable`` / ``seekable`` report the wrapped stream's own
    capabilities rather than asserting both, and ``close`` is left alone because the
    wrapped stream belongs to :func:`~.sources.open_source`'s context manager.
    """

    def __init__(self, raw: BinaryIO, cancel: threading.Event) -> None:
        super().__init__()
        self._raw = raw
        self._cancel = cancel

    def read(self, size: int | None = -1) -> bytes:
        if self._cancel.is_set():
            return b""
        return self._raw.read(size if size is not None else -1)

    def read1(self, size: int = -1) -> bytes:
        return self.read(size)

    def readinto(self, b: "memoryview | bytearray") -> int:  # type: ignore[override]
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if self._cancel.is_set():
            # Skipped, not forwarded: on a cloud source a seek opens a fresh ranged GET,
            # which is the very round trip the cancellation is trying to avoid. The
            # honest current position is returned so the demuxer sees the seek fail
            # rather than being told it landed somewhere it did not.
            return self._raw.tell()
        return self._raw.seek(offset, whence)

    def tell(self) -> int:
        return self._raw.tell()

    # Delegated rather than hardcoded to True: the sensor library rejects a stream that is
    # not readable and seekable, and claiming both would smuggle an unusable stream past
    # that guard into an opaque failure inside libav instead of the clear one up front.
    def seekable(self) -> bool:
        return self._raw.seekable()

    def readable(self) -> bool:
        return self._raw.readable()


def cancellable_reader(stream: BinaryIO, cancel: threading.Event) -> BinaryIO:
    """Wrap ``stream`` so it reads as an empty source once ``cancel`` is set."""
    return cast("BinaryIO", _CancellableReader(stream, cancel))


def available_cpu_count() -> int:
    """How many CPUs this process may actually run on, never below one.

    Prefers the scheduling affinity mask over the machine's core count, because
    under ``docker --cpuset-cpus`` or ``taskset`` the two disagree and only the mask
    bounds real parallelism. Affinity is Linux-only, hence the fallback. Note that a
    CFS quota (``docker --cpus``) caps CPU *time* without narrowing the mask, so it
    is not reflected here.
    """
    # Fetched dynamically because sched_getaffinity does not exist off Linux.
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        return max(1, len(sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def add_threshold_args(parser: argparse.ArgumentParser) -> None:
    """Add the pass/fail policy flags for the per-stream metrics, one per field.

    Shared by both CLIs so a policy flag never means two different things, and so
    adding a threshold is a change in one place. Defaults come from
    :data:`DEFAULT_THRESHOLDS` rather than being repeated as literals, keeping the
    help text honest if a default is ever retuned. The session-grain limits are in
    :func:`add_session_threshold_args`, which only the session CLI adds.
    """
    group = parser.add_argument_group(
        "thresholds",
        "Pass/fail policy. Defaults are neutral first-principles limits, not values tuned on a dataset.",
    )
    group.add_argument(
        "--max-strict-violations",
        type=non_negative_int,
        default=DEFAULT_THRESHOLDS.max_strict_violations,
        metavar="N",
        help=(
            "Ordering violations (backward or duplicate timestamps) tolerated "
            f"(default: {DEFAULT_THRESHOLDS.max_strict_violations}, i.e. require strictly increasing)."
        ),
    )
    group.add_argument(
        "--max-rate-deviation-percent",
        type=non_negative_finite_float,
        default=DEFAULT_THRESHOLDS.max_rate_deviation_percent,
        metavar="PCT",
        help=(
            "Mean-period deviation from the expected cadence tolerated, in percent "
            f"(default: {DEFAULT_THRESHOLDS.max_rate_deviation_percent})."
        ),
    )
    group.add_argument(
        "--max-gaps",
        type=non_negative_int,
        default=DEFAULT_THRESHOLDS.max_gaps,
        metavar="N",
        help=f"Inferred gaps (missing samples) tolerated (default: {DEFAULT_THRESHOLDS.max_gaps}).",
    )
    group.add_argument(
        "--max-jitter-percent",
        type=non_negative_finite_float,
        default=DEFAULT_THRESHOLDS.max_jitter_percent,
        metavar="PCT",
        help=(
            "Inter-sample jitter tolerated, as a percent of the expected period "
            f"(default: {DEFAULT_THRESHOLDS.max_jitter_percent})."
        ),
    )
    group.add_argument(
        "--allow-frame-reordering",
        action="store_true",
        help="Treat a frame-reordering (B-frame) stream as acceptable instead of failing that check.",
    )


def add_session_threshold_args(parser: argparse.ArgumentParser) -> None:
    """Add the policy flags for the metrics whose subject is a session.

    Separate from :func:`add_threshold_args`, and added only by the CLI that measures a
    session: a single-video run never compares two sensors, so offering it these limits
    would be offering it a knob that cannot turn anything.
    """
    group = parser.add_argument_group(
        "session thresholds",
        "Pass/fail policy for the checks that compare a session's sensors to each other.",
    )
    group.add_argument(
        "--max-sensor-spread-ns",
        type=non_negative_int,
        default=DEFAULT_THRESHOLDS.max_sensor_spread_ns,
        metavar="NS",
        help=(
            "Spread tolerated between the session's sensors starting, or stopping, whichever is worse, "
            f"in nanoseconds (default: {DEFAULT_THRESHOLDS.max_sensor_spread_ns}). Unlike the limits above "
            "this default is a placeholder rather than a first-principles limit."
        ),
    )
    group.add_argument(
        "--max-non-overlap-percent",
        type=share_percent,
        default=DEFAULT_THRESHOLDS.max_non_overlap_percent,
        metavar="PCT",
        help=(
            "Share of the session tolerated during which some sensor was not recording, in percent "
            f"(default: {DEFAULT_THRESHOLDS.max_non_overlap_percent}, max: {MAX_PERCENT:g}). Stated as the "
            "complement of the overlap because every threshold here is a ceiling."
        ),
    )


def thresholds_from_args(args: argparse.Namespace) -> Thresholds:
    """Build a :class:`Thresholds` from a namespace populated by the threshold flags.

    The session limits fall back to their defaults when the parser did not offer them,
    so a CLI that adds only :func:`add_threshold_args` still gets a whole policy.
    """
    return Thresholds(
        max_strict_violations=args.max_strict_violations,
        max_rate_deviation_percent=args.max_rate_deviation_percent,
        max_gaps=args.max_gaps,
        max_jitter_percent=args.max_jitter_percent,
        allow_frame_reordering=args.allow_frame_reordering,
        max_sensor_spread_ns=getattr(args, "max_sensor_spread_ns", DEFAULT_THRESHOLDS.max_sensor_spread_ns),
        max_non_overlap_percent=getattr(args, "max_non_overlap_percent", DEFAULT_THRESHOLDS.max_non_overlap_percent),
    )
