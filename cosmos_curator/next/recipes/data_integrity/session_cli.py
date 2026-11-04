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

# The `di` pixi environment carries this tool's whole import closure and nothing that
# pins a CUDA platform, so it installs on a laptop as well as on a node:
#
#   pixi install -e di
#   pixi run -e di di di.yaml
#
# On a node that already has `default` installed, skip the extra environment:
#
#   pixi run --as-is python -m cosmos_curator.next.recipes.data_integrity.session_cli di.yaml
#

"""Run every data-integrity metric over one session, or one video, without Ray.

The companion to the ``data-integrity`` pipeline: same config, same engine, same
report, one session at a time and no cluster. That is the whole of its reason to
exist -- it is how a scan is iterated on and profiled, and how a single file is
checked, with Ray and Lance out of the picture.

A *session* is one recording, typically a ``clips/<uuid>/`` directory or cloud prefix
holding one video per camera. A lone video file is a session of one stream, which is
why there is no separate single-video tool: :func:`~.discovery.discover_streams`
already resolves a file, a directory and a prefix the same way.

Settings come from the config file rather than from flags
(:mod:`.config`), so the two executors cannot drift apart, and so a config iterated on
here is the config a pipeline run takes. The flags left here are the ones with no
config twin: they shape what this process prints, not what gets measured. Per-session
settings the config carries for the pipeline alone -- ``execution.session_concurrency``,
``append_batch_size``, ``progress`` and ``no_progress_timeout_s`` -- have nothing to
act on here and are ignored.
"""

import argparse
import io
import sys
import threading
import time
from collections.abc import Callable
from typing import BinaryIO, cast

from cosmos_curator.core.sensors.data_integrity.results import OverallStatus, SessionReport, StreamResult
from cosmos_curator.core.utils.storage.s3_client import resolve_s3_endpoint_url
from cosmos_curator.core.utils.storage.storage_client import StorageStat
from cosmos_curator.core.utils.storage.storage_utils import is_remote_path
from cosmos_curator.core.utils.storage_cli import StorageCliError
from cosmos_curator.next.recipes.data_integrity import storage_io
from cosmos_curator.next.recipes.data_integrity.cli_support import (
    ERROR_EXIT_CODE,
    FAIL_EXIT_CODE,
    PASS_EXIT_CODE,
    interrupt_guard,
    non_negative_int,
    open_index_manifest,
    raise_if_interrupted,
    report_error,
    report_interrupted,
)
from cosmos_curator.next.recipes.data_integrity.config import ResolvedDataIntegrityConfig, resolve_config
from cosmos_curator.next.recipes.data_integrity.report import render_text, to_json
from cosmos_curator.next.recipes.data_integrity.session_runner import run_session
from cosmos_curator.next.recipes.data_integrity.sessions import single_session
from cosmos_curator.next.recipes.data_integrity.store_cli import persist_run

#: What the store records as the writer of this tool's rows. Write-only and
#: unvalidated, so rows written by another entry point keep their own value.
TOOL = "di"

# Session verdict -> process exit code. Uses the shared CLI exit-code contract:
# PASS = 0, FAIL = 1, and anything that could not be checked (ERROR) = 2.
_EXIT_CODES = {
    OverallStatus.PASS: PASS_EXIT_CODE,
    OverallStatus.FAIL: FAIL_EXIT_CODE,
    OverallStatus.ERROR: ERROR_EXIT_CODE,
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="di",
        description=(
            "Run every data-integrity metric over one session -- or one video -- from a config file, "
            "then the session-grain metrics across its streams."
        ),
        epilog=(
            "Exit codes: 0 = everything PASS; 1 = at least one FAIL, of a stream or of the session itself; "
            "2 = an error occurred; 130 = interrupted with Ctrl-C. "
            "A stream that could not be measured outranks a FAIL, since the session verdict stays incomplete."
        ),
    )
    parser.add_argument(
        "config",
        metavar="CONFIG",
        help=(
            "A data-integrity config (YAML or JSON), as `cosmos-curator pipeline template data-integrity` "
            "prints. input.sessions must name exactly one session; output.store_root may be null to "
            "report without persisting."
        ),
    )
    parser.add_argument(
        "--set",
        action="append",
        metavar="PATH=VALUE",
        help=(
            "Override one config setting, e.g. --set checks.expected_hz=30. Repeatable. "
            "VALUE is parsed as YAML, so `null`, numbers, booleans and JSON lists all work."
        ),
    )
    parser.add_argument(
        "--stream-idx",
        type=non_negative_int,
        default=0,
        metavar="N",
        help=(
            "Video stream index to open inside each container (default: 0). For reaching a secondary "
            "track of one file; a session normally holds one video per file. The store keys on it, so "
            "index 0 and index 1 of a file are two streams rather than one overwriting the other."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit the JSON report to stdout.")
    parser.add_argument(
        "--perf-profile",
        action="store_true",
        help="Emit per-phase wall-clock timings to stderr after the report.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Print a per-stream progress line to stderr as each stream is opened and finished. "
            "Default: on for a multi-stream session, off for a single video, where the report "
            "itself arrives at the same moment the progress would. stdout stays a clean report / "
            "JSON payload either way."
        ),
    )
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

    Also where the run's ``HEAD`` responses are collected for the store, which is why
    this is constructed even when nothing will be printed: silencing the display must
    not cost the store its content identity, nor make it issue a second round of
    lookups of its own.

    Hooks are called from :func:`run_session`'s worker threads, so writes are
    serialised on a lock to keep lines from interleaving mid-write.
    """

    def __init__(
        self,
        stat_lookup: Callable[[str], StorageStat | None] | None = None,
        *,
        display: bool | None = True,
        live_byte_counter: bool = True,
        collect_stats: bool = False,
    ) -> None:
        self._bytes: dict[int, int] = {}
        self._last_emit: dict[int, float] = {}
        self._stat_lookup = stat_lookup
        #: ``None`` defers to the stream count, which is only known once the session has
        #: been listed -- hence resolved per call from the ``total`` the hooks are given
        #: rather than once up front.
        self._display = display
        self._live_byte_counter = live_byte_counter
        self._collect_stats = collect_stats
        self._lock = threading.Lock()
        #: Every ``HEAD`` response this run made, by source. Handed to the store so a
        #: run that persists still issues exactly one per stream.
        self.stats: dict[str, StorageStat] = {}

    def _shows(self, total: int) -> bool:
        """Whether a session of ``total`` streams gets progress output."""
        # A single stream's start line is immediately followed by the report itself, so
        # it tells the operator nothing they are not about to be told anyway; a session
        # of many is the case where knowing which stream is being waited on matters.
        return total > 1 if self._display is None else self._display

    def _write(self, text: str) -> None:
        with self._lock:
            sys.stderr.write(text)
            sys.stderr.flush()

    def start(self, index: int, total: int, source: str) -> None:
        if not self._shows(total):
            return
        self._write(f"[{index}/{total}] {source}\n")

    def make_wrapper(self, index: int, total: int, source: str) -> Callable[[BinaryIO], BinaryIO]:
        counting = self._shows(total)
        live = counting and self._live_byte_counter

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
        if self._stat_lookup is not None and (live or self._collect_stats):
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

        if not counting:
            # Nothing reads the count, so do not pay a Python-level call per read for it.
            return lambda stream: stream

        def _wrap(stream: BinaryIO) -> BinaryIO:
            def _on_bytes(count: int) -> None:
                self._bytes[index] = count
                if not live:
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
        if not self._shows(total):
            return
        mb = self._bytes.get(index, 0) / 1e6
        tail = f" ({mb:.1f} MB read)" if mb else ""
        # Leading \r + trailing pad overwrite the in-place byte counter on this line;
        # with no counter drawn there is nothing to overwrite.
        prefix, pad = ("\r", "          ") if self._live_byte_counter else ("", "")
        self._write(f"{prefix}[{index}/{total}] -> {result.status.value}{tail}{pad}\n")


def _render_perf(  # noqa: PLR0913 -- one argument per thing measured, all independent
    stats: dict[str, dict[str, float]],
    report: SessionReport,
    *,
    batch_size: int,
    stream_concurrency: int,
    render_ms: float,
    total_ms: float,
) -> str:
    """Format per-phase wall-clock timings for stderr.

    One ``key = value`` line per field, greppable and stable for scripting. Timing keys
    are milliseconds. Emitted on stderr so ``--json`` stdout stays a clean
    single-document payload.

    The phases are per stream because that is the grain they are measured at, and a
    single sum across them would be the one number a profile must not report above
    concurrency 1: concurrent streams overlap, so their phases add up to more than the
    session took. ``total_ms`` is the session's own elapsed time instead, which is
    true at any concurrency. ``render_ms`` is session-grain for the plainer reason that
    one report is rendered, not one per stream.
    """
    lines = [
        f"  batch_size         = {batch_size}",
        f"  stream_concurrency = {stream_concurrency}",
    ]
    if stream_concurrency > 1:
        lines.append("  (streams overlap: the per-stream phases below do not sum to total_ms)")
    for index, stream in enumerate(report.streams, start=1):
        lines.append(f"  [{index}/{len(report.streams)}] {stream.source}")
        stream_stats = stats.get(stream.source)
        if stream_stats is None:
            # Only a measured stream is timed, so say so rather than printing zeros that
            # would read as a stream that was instant.
            lines.append(f"    not measured  = {stream.status.value}")
            continue
        lines.extend(
            [
                f"    sensor_init_ms = {stream_stats.get('sensor_init_ms', 0.0):.3f}",
                f"    stream_ms      = {stream_stats.get('stream_ms', 0.0):.3f}",
                f"    evaluate_ms    = {stream_stats.get('evaluate_ms', 0.0):.3f}",
                f"    num_samples    = {stream.num_samples}",
                f"    has_bframes    = {stream.has_bframes}",
                f"    codec          = {stream.codec_name}",
            ]
        )
    lines.extend([f"  render_ms          = {render_ms:.3f}", f"  total_ms           = {total_ms:.3f}"])
    # Leading newline so the block visually separates from the report above (which ends
    # with "Session overall: <status>\n" in human mode or the closing "}\n" in JSON mode).
    return "\nPer-phase performance stats:\n" + "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Resolve the config, check every stream of its one session, and print the report."""
    args = _parse_args(argv)
    try:
        config = resolve_config(args.config, overrides=tuple(args.set or ()))
        session_path = single_session(config.input)
    # Resolved before the interrupt guard because none of it reads a source: a bad
    # config is a usage error, and reporting it needs no cooperative shutdown.
    # ``OSError`` and not just its missing-file subclass, matching
    # :mod:`cosmos_curator.pipelines.pipeline_runtime`: a path can also be a directory
    # or be unreadable, and an operator who mistyped one is owed the same one-line
    # error as one who mistyped the other, not a traceback.
    except (OSError, TypeError, ValueError) as e:
        return report_error(str(e))

    endpoint_url = resolve_s3_endpoint_url(config.execution.endpoint_url)

    def _stat_lookup(source: str) -> StorageStat | None:
        if not is_remote_path(source):
            return None
        return storage_io.object_stat(
            source,
            s3_profile_name=config.execution.s3_profile_name,
            azure_profile_name=config.execution.azure_profile_name,
            endpoint_url=endpoint_url,
        )

    progress = _Progress(
        stat_lookup=_stat_lookup,
        display=args.progress,
        live_byte_counter=config.execution.stream_concurrency == 1,
        collect_stats=config.output.store_root is not None,
    )
    with interrupt_guard() as interrupted:
        try:
            return _run(args, config, session_path, endpoint_url, progress, interrupted)
        # Caught out here rather than inside _run so that an abort recognised *while*
        # another failure is being handled still lands on it: a handler cannot be caught
        # by its own siblings.
        except KeyboardInterrupt:
            return report_interrupted()


def _run(  # noqa: PLR0913 -- the resolved config plus the per-process concerns around it
    args: argparse.Namespace,
    config: ResolvedDataIntegrityConfig,
    session_path: str,
    endpoint_url: str | None,
    progress: "_Progress",
    interrupted: threading.Event,
) -> int:
    """Body of :func:`main`, split out so the interrupt guard wraps the whole attempt.

    Raises:
        KeyboardInterrupt: If the operator aborted, including when libav disguised the
            abort as a decode error. :func:`main` turns it into the exit code.

    """
    thresholds = config.checks.thresholds.to_thresholds()
    stats: dict[str, dict[str, float]] | None = {} if args.perf_profile else None
    started = time.perf_counter()
    try:
        index_lookup = open_index_manifest(
            config.input.mp4_index_uri,
            s3_profile_name=config.execution.s3_profile_name,
            endpoint_url=endpoint_url,
        )
        report = run_session(
            session_path,
            expected_hz=config.checks.expected_hz,
            thresholds=thresholds,
            batch_size=config.checks.batch_size,
            # The config spells "no cap" as null and the runner as 0.
            limit=config.input.limit or 0,
            stream_globs=config.input.stream_globs,
            s3_profile_name=config.execution.s3_profile_name,
            azure_profile_name=config.execution.azure_profile_name,
            endpoint_url=endpoint_url,
            stream_idx=args.stream_idx,
            index_lookup=index_lookup,
            alignment_fields=config.checks.camera_alignment_fields,
            max_workers=config.execution.stream_concurrency,
            max_attempts=config.execution.stream_attempts,
            cancel=interrupted,
            on_stream_start=progress.start,
            on_stream_finish=progress.finish,
            make_stream_wrapper=progress.make_wrapper,
            stats=stats,
        )
        # Checked after a *successful* run too, and not only inside the runner: libav
        # can swallow the abort and finish off a truncated stream, while discovery
        # never reaches a read at all, so a returned report is not evidence that the
        # operator still wants it.
        raise_if_interrupted(interrupted)
        # Rendering and writing stay inside the handler: serializing a report can fail
        # (non-finite measurements) and so can the write itself (a closed pipe), and both
        # owe the caller exit code 2 rather than a traceback.
        render_started = time.perf_counter()
        output = (to_json(report) if args.json else render_text(report)) + "\n"
        render_ms = (time.perf_counter() - render_started) * 1000
        sys.stdout.write(output)
        if stats is not None:
            # Flush the report first so interactive users see it above the perf block.
            sys.stdout.flush()
            sys.stderr.write(
                _render_perf(
                    stats,
                    report,
                    batch_size=config.checks.batch_size,
                    stream_concurrency=config.execution.stream_concurrency,
                    render_ms=render_ms,
                    total_ms=(time.perf_counter() - started) * 1000,
                )
            )
        if config.output.store_root is not None:
            # After the report, so the operator keeps the verdict even when persisting
            # fails -- but a failure is still exit code 2, because saving the results
            # was part of what was asked for.
            try:
                persist_run(
                    config.output.store_root,
                    report.streams,
                    session_path=report.session_path,
                    thresholds=thresholds,
                    tool=TOOL,
                    # The same verdicts the report printed, so what this exited on is
                    # what reads back.
                    session_metrics=report.metrics,
                    storage_stats=progress.stats,
                    s3_profile_name=config.execution.s3_profile_name,
                    azure_profile_name=config.execution.azure_profile_name,
                    endpoint_url=endpoint_url,
                )
            except Exception as e:  # noqa: BLE001 - the store can fail in as many ways as its backend
                return report_error(f"checked session {session_path!r} but could not write the store: {e}")
        return _EXIT_CODES[report.status]
    except (StorageCliError, FileNotFoundError) as e:
        return report_error(str(e))
    except Exception as e:  # noqa: BLE001
        # A Ctrl-C inside a libav read arrives here as a decode error rather than a
        # KeyboardInterrupt, so trust the recorded signal over the exception type.
        raise_if_interrupted(interrupted)
        return report_error(f"could not check session {session_path!r}: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
