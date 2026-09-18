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
#

# LOCAL DEV SETUP
#
# Same environment as the single-video di-check CLI (see cli.py). Once that venv
# exists:
#
#   alias di-session="$VENV/bin/python -m cosmos_curator.next.recipes.data_integrity.session_cli"
#
# Run against one session (a clips/<uuid>/ directory or cloud prefix):
#
#   di-session --session-path /path/to/clips/<uuid>
#   di-session --session-path s3://bucket/clips/<uuid>/ --s3-profile-name my-profile --json
#   di-session --session-path s3://hyperion8/clips/<uuid>/ --endpoint-url <endpoint-url> --limit 5
#
# Where a session prefix holds copies of its cameras beside the originals, name the
# subprefixes that hold the real streams:
#
#   di-session --session-path s3://bucket/clips/<uuid>/ --stream-glob 'recorder-*/*'
#

"""Run every data-integrity metric against every stream of a single session.

A *session* is one recording -- typically a ``clips/<uuid>/`` directory (or cloud
prefix) holding one video per camera. This CLI discovers those streams, runs the
shared per-stream engine
(:mod:`cosmos_curator.core.sensors.data_integrity.engine`) on each, and prints
a per-stream + session-level verdict. It is the session-level companion to the
single-video ``di-check`` CLI (:mod:`.cli`) and reuses the same engine, cloud
credential flags, and reason strings.
"""

import argparse
import io
import sys
import threading
import time
from collections.abc import Callable
from typing import BinaryIO, cast

from cosmos_curator.core.sensors.data_integrity.results import OverallStatus, StreamResult
from cosmos_curator.core.utils.storage.s3_client import resolve_s3_endpoint_url
from cosmos_curator.core.utils.storage.storage_client import StorageStat
from cosmos_curator.core.utils.storage.storage_utils import is_remote_path
from cosmos_curator.core.utils.storage_cli import StorageCliError, add_storage_credential_args
from cosmos_curator.next.recipes.data_integrity import storage_io
from cosmos_curator.next.recipes.data_integrity.cli_support import (
    ERROR_EXIT_CODE,
    FAIL_EXIT_CODE,
    PASS_EXIT_CODE,
    add_session_threshold_args,
    add_threshold_args,
    available_cpu_count,
    interrupt_guard,
    non_empty_str,
    non_negative_int,
    positive_finite_float,
    positive_int,
    raise_if_interrupted,
    report_error,
    report_interrupted,
    thresholds_from_args,
)
from cosmos_curator.next.recipes.data_integrity.report import render_text, to_json
from cosmos_curator.next.recipes.data_integrity.session_runner import run_session
from cosmos_curator.next.recipes.data_integrity.store_cli import add_store_args, persist_run

# Session verdict -> process exit code. Uses the shared CLI exit-code contract:
# PASS = 0, FAIL = 1, and anything that could not be checked (ERROR) = 2.
_EXIT_CODES = {
    OverallStatus.PASS: PASS_EXIT_CODE,
    OverallStatus.FAIL: FAIL_EXIT_CODE,
    OverallStatus.ERROR: ERROR_EXIT_CODE,
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="di-session",
        description=(
            "Run every data-integrity metric against every stream of a single session, "
            "then the session-grain metrics across all of them."
        ),
        epilog=(
            "Exit codes: 0 = everything PASS; 1 = at least one FAIL, of a stream or of the session itself; "
            "2 = an error occurred; 130 = interrupted with Ctrl-C. "
            "A stream that could not be measured outranks a FAIL, since the session verdict stays incomplete."
        ),
    )
    parser.add_argument(
        "--session-path",
        required=True,
        metavar="PATH",
        help=(
            "Local directory or file, or an s3:// / az:// prefix for one session (typically a single clips/<uuid>/)."
        ),
    )
    parser.add_argument(
        "--expected-hz",
        type=positive_finite_float,
        default=None,
        metavar="HZ",
        help=(
            "Authoritative expected sample rate for the rate, gap, and jitter checks, applied to every "
            "stream. Defaults to each stream's declared rate, which is best-effort and will not catch a "
            "uniformly wrong sensor sampling rate; SKIPs those checks where neither is available."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=non_negative_int,
        default=0,
        metavar="N",
        help="Timestamps per metric update; 0 (default) feeds each stream's whole array at once.",
    )
    parser.add_argument(
        "--limit",
        type=non_negative_int,
        default=0,
        metavar="N",
        help="Cap the number of streams checked; 0 (default) means all. Useful for sampling a large session.",
    )
    parser.add_argument(
        "--stream-glob",
        action="append",
        type=non_empty_str,
        default=None,
        metavar="GLOB",
        help=(
            "Check only streams whose path below the session matches this glob; repeatable, and every "
            "video is checked when it is omitted. '*' does not cross '/', '**' spans subdirectories. "
            "Pass --stream-glob 'recorder-*/*' where a session prefix also holds copies of its own "
            "cameras, which would otherwise be measured twice and inflate the session's sensor count."
        ),
    )
    default_max_workers = available_cpu_count()
    parser.add_argument(
        "--max-workers",
        type=positive_int,
        default=default_max_workers,
        metavar="N",
        help=(
            f"Streams to check concurrently (default: one per usable CPU, so {default_max_workers} here). "
            "Each stream spends most of its time waiting on the source, so overlapping them helps most "
            "on cloud sessions. Use 1 for serial checking and a live byte counter."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit the JSON report to stdout.")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Print a per-stream progress line to stderr as each stream is opened and finished "
            "(default: on). stdout stays a clean report / JSON payload. Use --no-progress to silence."
        ),
    )
    add_threshold_args(parser)
    add_session_threshold_args(parser)
    add_store_args(parser)
    add_storage_credential_args(parser)
    return parser.parse_args(argv)


_PROGRESS_MIN_INTERVAL_S = 0.5  # throttle the in-place byte-counter refresh


class _CountingReader(io.BufferedIOBase):
    """Wrap a binary stream and report cumulative bytes read via a callback.

    Every ``read`` is forwarded to the wrapped stream and the running total is
    handed to ``on_bytes``. ``readinto`` / ``read1`` route through ``read`` so the
    count is captured whichever access pattern PyAV uses. ``close`` is a no-op: the
    underlying remote stream is owned by ``open_storage_source``'s context manager, so
    the sensor closing this wrapper must not close it out from under that manager.
    """

    def __init__(self, raw: BinaryIO, on_bytes: Callable[[int], None]) -> None:
        super().__init__()
        self._raw = raw
        self._on_bytes = on_bytes
        self._count = 0

    def read(self, size: int | None = -1) -> bytes:
        data = self._raw.read(size if size is not None else -1)
        if data:
            self._count += len(data)
            self._on_bytes(self._count)
        return data

    def read1(self, size: int = -1) -> bytes:
        return self.read(size)

    def readinto(self, b: "memoryview | bytearray") -> int:  # type: ignore[override]
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self._raw.seek(offset, whence)

    def tell(self) -> int:
        return self._raw.tell()

    # Delegated rather than hardcoded to True, for the reason given on the cancellable
    # reader: a wrapper that overstates its stream's capabilities defeats the sensor
    # library's own precondition check.
    def seekable(self) -> bool:
        return self._raw.seekable()

    def readable(self) -> bool:
        return self._raw.readable()


class _Progress:
    """Per-stream progress printer for :func:`run_session` (stderr only).

    Prints the source on start, the verdict on finish, and -- when checking one
    stream at a time -- an in-place refreshing ``N.N MB read`` counter in between.
    Concurrent streams cannot share that one redrawn line, so with several in flight
    the counter is dropped and only the start/finish lines are printed; the byte
    total still lands on the finish line. All output goes to stderr so the report /
    ``--json`` on stdout stays a clean single document.

    Hooks are called from :func:`run_session`'s worker threads, so writes are
    serialised on a lock to keep lines from interleaving mid-write.
    """

    def __init__(
        self,
        stat_lookup: Callable[[str], StorageStat | None] | None = None,
        *,
        live_byte_counter: bool = True,
        collect_stats: bool = False,
    ) -> None:
        self._bytes: dict[int, int] = {}
        self._last_emit: dict[int, float] = {}
        self._stat_lookup = stat_lookup
        self._live_byte_counter = live_byte_counter
        self._collect_stats = collect_stats
        self._lock = threading.Lock()
        #: Every ``HEAD`` response this run made, by source. Handed to the store so a
        #: run that persists still issues exactly one per stream.
        self.stats: dict[str, StorageStat] = {}

    def _write(self, text: str) -> None:
        with self._lock:
            sys.stderr.write(text)
            sys.stderr.flush()

    def start(self, index: int, total: int, source: str) -> None:
        self._write(f"[{index}/{total}] {source}\n")

    def make_wrapper(self, index: int, total: int, source: str) -> Callable[[BinaryIO], BinaryIO]:
        # One cheap HEAD up front so the counter can show read / total (pct%);
        # None (unknown / lookup disabled) falls back to a bare "read" figure. The lookup
        # is injected, so guard it here instead of trusting every implementation to be
        # exception-safe -- a cosmetic byte count must never abandon the streams behind it.
        #
        # Skipped outright when nothing will read the result, because the lookup is not
        # free: on a remote source it builds a fresh client per stream, and that parsing
        # holds the interpreter lock, so a dozen concurrent streams serialise behind it.
        # Two things read it -- the live counter (the finish line reports bytes actually
        # read, so it needs no total) and the store, which records the same response as
        # content identity rather than issuing a second HEAD of its own.
        stat: StorageStat | None = None
        if self._stat_lookup is not None and (self._live_byte_counter or self._collect_stats):
            try:
                stat = self._stat_lookup(source)
            except Exception:  # noqa: BLE001 - advisory only; unknown size is fine
                stat = None
            # Only a lookup that answered is worth handing on. Caching a failed one
            # would satisfy the store's "use the caller's stat" path and persist a null
            # ETag and size for an object whose HEAD merely failed here; leaving it out
            # lets the store try again for itself.
            if stat is not None:
                with self._lock:
                    self.stats[source] = stat
        total_bytes = stat.size_bytes if stat is not None else None

        def _wrap(stream: BinaryIO) -> BinaryIO:
            def _on_bytes(count: int) -> None:
                self._bytes[index] = count
                if not self._live_byte_counter:
                    return
                now = time.monotonic()
                if now - self._last_emit.get(index, 0.0) >= _PROGRESS_MIN_INTERVAL_S:
                    self._last_emit[index] = now
                    if total_bytes:
                        pct = min(100, int(count * 100 / total_bytes))
                        detail = f"{count / 1e6:.1f}/{total_bytes / 1e6:.1f} MB ({pct}%)"
                    else:
                        detail = f"{count / 1e6:.1f} MB read"
                    self._write(f"\r[{index}/{total}]   {detail}")

            return cast("BinaryIO", _CountingReader(stream, _on_bytes))

        return _wrap

    def finish(self, index: int, total: int, result: StreamResult) -> None:
        mb = self._bytes.get(index, 0) / 1e6
        tail = f" ({mb:.1f} MB read)" if mb else ""
        # Leading \r + trailing pad overwrite the in-place byte counter on this line;
        # with no counter drawn there is nothing to overwrite.
        prefix, pad = ("\r", "          ") if self._live_byte_counter else ("", "")
        self._write(f"{prefix}[{index}/{total}] -> {result.status.value}{tail}{pad}\n")


def main(argv: list[str] | None = None) -> int:
    """Discover and check every stream in the session, then print the report."""
    args = _parse_args(argv)
    endpoint_url = resolve_s3_endpoint_url(args.endpoint_url)

    def _stat_lookup(source: str) -> StorageStat | None:
        if not is_remote_path(source):
            return None
        return storage_io.object_stat(
            source,
            s3_profile_name=args.s3_profile_name,
            azure_profile_name=args.azure_profile_name,
            endpoint_url=endpoint_url,
        )

    progress = (
        _Progress(
            stat_lookup=_stat_lookup,
            live_byte_counter=args.max_workers == 1,
            collect_stats=args.store_path is not None,
        )
        if args.progress
        else None
    )
    with interrupt_guard() as interrupted:
        try:
            return _run(args, endpoint_url, progress, interrupted)
        # Caught out here rather than inside _run so that an abort recognised *while*
        # another failure is being handled still lands on it: a handler cannot be caught
        # by its own siblings.
        except KeyboardInterrupt:
            return report_interrupted()


def _run(
    args: argparse.Namespace, endpoint_url: str | None, progress: "_Progress | None", interrupted: threading.Event
) -> int:
    """Body of :func:`main`, split out so the interrupt guard wraps the whole attempt.

    Raises:
        KeyboardInterrupt: If the operator aborted, including when libav disguised the
            abort as a decode error. :func:`main` turns it into the exit code.

    """
    thresholds = thresholds_from_args(args)
    try:
        report = run_session(
            args.session_path,
            expected_hz=args.expected_hz,
            thresholds=thresholds,
            batch_size=args.batch_size,
            limit=args.limit,
            stream_globs=tuple(args.stream_glob or ()),
            s3_profile_name=args.s3_profile_name,
            azure_profile_name=args.azure_profile_name,
            endpoint_url=endpoint_url,
            max_workers=args.max_workers,
            cancel=interrupted,
            on_stream_start=progress.start if progress else None,
            on_stream_finish=progress.finish if progress else None,
            make_stream_wrapper=progress.make_wrapper if progress else None,
        )
        # Checked after a *successful* run too, and not only inside the runner: libav
        # can swallow the abort and finish off a truncated stream, while discovery
        # never reaches a read at all, so a returned report is not evidence that the
        # operator still wants it.
        raise_if_interrupted(interrupted)
        # Rendering and writing stay inside the handler: serializing a report can fail
        # (non-finite measurements) and so can the write itself (a closed pipe), and both
        # owe the caller exit code 2 rather than a traceback.
        sys.stdout.write((to_json(report) if args.json else render_text(report)) + "\n")
        if args.store_path is not None:
            # After the report, so the operator keeps the verdict even when persisting
            # fails -- but a failure is still exit code 2, because saving the results
            # was part of what was asked for.
            try:
                persist_run(
                    args.store_path,
                    report.streams,
                    session_path=report.session_path,
                    thresholds=thresholds,
                    tool="di-session",
                    # The same verdicts the report printed, so what this exited on is
                    # what reads back.
                    session_metrics=report.metrics,
                    storage_stats=progress.stats if progress is not None else None,
                    s3_profile_name=args.s3_profile_name,
                    azure_profile_name=args.azure_profile_name,
                    endpoint_url=endpoint_url,
                )
            except Exception as e:  # noqa: BLE001 - the store can fail in as many ways as its backend
                return report_error(f"checked session {args.session_path!r} but could not write the store: {e}")
        return _EXIT_CODES[report.status]
    except (StorageCliError, FileNotFoundError) as e:
        return report_error(str(e))
    except Exception as e:  # noqa: BLE001
        # A Ctrl-C inside a libav read arrives here as a decode error rather than a
        # KeyboardInterrupt, so trust the recorded signal over the exception type.
        raise_if_interrupted(interrupted)
        return report_error(f"could not check session {args.session_path!r}: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
