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
# plain uv venv is the simplest cross-platform path:
#
#   VENV=~/.venvs/cosmos-curator-di
#   uv venv --python 3.13 $VENV
#   uv pip install --python $VENV/bin/python attrs numpy av 'smart_open[s3,azure]' azure-identity loguru
#   uv pip install --python $VENV/bin/python --no-deps -e .
#   alias di-check="$VENV/bin/python -m cosmos_curator.next.recipes.data_integrity.cli"
#
# Run against an S3 clip (add --json or --perf-profile; --help lists all flags):
#
#   di-check --source /path/to/local/clip.mp4
#   di-check --source s3://bucket/path/clip.mp4 --s3-profile-name my-profile
#

"""Run every data-integrity metric against a single video and report the verdict.

Thin CLI wrapper: the per-stream engine (running the metrics, judging them, and
serialising measurements) lives in
:mod:`cosmos_curator.core.sensors.data_integrity.engine`, reached through
:mod:`.sources`, which opens the URI it is pointed at. The argparse surface and
exit codes shared with the single-session tool live in :mod:`.cli_support`. This
module owns only the single-video report shapes, the argument surface, and the
exit-code contract.
"""

import argparse
import json
import sys
import threading
import time

from cosmos_curator.core.sensors.data_integrity.results import (
    CheckResult,
    CheckStatus,
    ResolvedConfig,
    VideoInfo,
    overall_status,
    stream_result,
)
from cosmos_curator.core.utils.storage.s3_client import resolve_s3_endpoint_url
from cosmos_curator.core.utils.storage_cli import (
    StorageCliError,
    add_storage_credential_args,
    validate_source,
)
from cosmos_curator.next.recipes.data_integrity.cli_support import (
    FAIL_EXIT_CODE,
    PASS_EXIT_CODE,
    add_index_args,
    add_threshold_args,
    cancellable_reader,
    interrupt_guard,
    non_negative_int,
    open_index_manifest,
    positive_finite_float,
    raise_if_interrupted,
    report_error,
    report_interrupted,
    thresholds_from_args,
)
from cosmos_curator.next.recipes.data_integrity.sources import run_checks
from cosmos_curator.next.recipes.data_integrity.store_cli import add_store_args, persist_run


def _format_expected_hz_line(resolved_cfg: ResolvedConfig) -> str:
    """Format the ``expected_hz:`` line for the human-readable report header.

    Uses the enum ``.value`` directly so the ``source:`` annotation matches the
    JSON payload's ``expected_hz_source`` field verbatim (``user`` / ``header`` /
    ``unavailable``). New variants automatically render without additional wiring.
    """
    value = f"{resolved_cfg.expected_hz:.3f}" if resolved_cfg.expected_hz is not None else "N/A"
    return f"  expected_hz: {value} (source: {resolved_cfg.expected_hz_source.value})\n"


def _render_human(
    source: str,
    video_info: VideoInfo,
    resolved_cfg: ResolvedConfig,
    results: list[CheckResult],
) -> str:
    header = (
        f"Data-integrity report for: {source}\n"
        f"  codec: {video_info.codec_name}"
        f"   has_bframes: {str(video_info.has_bframes).lower()}"
        f"   num_samples: {video_info.num_samples}"
        f"   start_ns: {video_info.start_ns}"
        f"   end_ns: {video_info.end_ns}\n"
        f"{_format_expected_hz_line(resolved_cfg)}"
        f"  timestamps: {video_info.timestamp_source.value}\n\n"
    )
    name_width = max(len(r.name) for r in results)
    lines = [f"  {r.name.ljust(name_width)}  {r.status.value:<7}  {r.reason}\n" for r in results]
    overall = overall_status(results)
    return header + "".join(lines) + f"\nOverall: {overall.value}\n"


def _render_json(
    source: str,
    effective_batch_size: int,
    video_info: VideoInfo,
    resolved_cfg: ResolvedConfig,
    results: list[CheckResult],
) -> str:
    payload: dict[str, object] = {
        "source": source,
        "expected_hz": resolved_cfg.expected_hz,
        "expected_hz_source": resolved_cfg.expected_hz_source.value,
        "batch_size": effective_batch_size,
        "video": video_info.to_dict(),
        "checks": [
            {
                "name": r.name,
                "status": r.status.value,
                "reason": r.reason,
                "measurement": r.measurement,
                "evaluation": r.evaluation,
            }
            for r in results
        ],
        "overall_status": overall_status(results).value,
    }
    return json.dumps(payload, indent=2, allow_nan=False) + "\n"


def _render_perf(stats: dict[str, float], *, video_info: VideoInfo, batch_size: int) -> str:
    """Format per-phase wall-clock timings for stderr.

    One ``key = value`` line per field, greppable and stable for scripting. Timing
    keys are milliseconds. Emitted on stderr so ``--json`` stdout stays a clean
    single-document payload.
    """
    total_ms = sum(stats.values())
    lines = [
        f"  sensor_init_ms  = {stats.get('sensor_init_ms', 0.0):.3f}",
        f"  stream_ms       = {stats.get('stream_ms', 0.0):.3f}",
        f"  evaluate_ms     = {stats.get('evaluate_ms', 0.0):.3f}",
        f"  render_ms       = {stats.get('render_ms', 0.0):.3f}",
        f"  total_ms        = {total_ms:.3f}",
        f"  num_samples     = {video_info.num_samples}",
        f"  has_bframes     = {video_info.has_bframes}",
        f"  codec           = {video_info.codec_name}",
        f"  batch_size      = {batch_size}",
    ]
    # Leading newline so the block visually separates from the report above (which
    # ends with "Overall: <status>\n" in human mode or the closing "}\n" in JSON mode).
    return "\nPer-phase performance stats:\n" + "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="di-check",
        description="Run every data-integrity metric against a single video and print the verdict.",
        epilog=(
            "Exit codes: 0 = every check PASS or SKIPPED; 1 = at least one FAIL; "
            "2 = input, credential, or runtime error; 130 = interrupted with Ctrl-C."
        ),
    )
    parser.add_argument(
        "--source",
        required=True,
        metavar="URI",
        help="Local path, s3://, or az:// URI to the video file.",
    )
    parser.add_argument(
        "--expected-hz",
        type=positive_finite_float,
        default=None,
        metavar="HZ",
        help=(
            "Authoritative expected sample rate for the rate, gap, and jitter checks. "
            "Defaults to the container's declared rate, which is best-effort and will not catch a "
            "uniformly wrong sensor sampling rate; SKIPs those checks if neither is available."
        ),
    )
    parser.add_argument(
        "--stream-idx",
        type=non_negative_int,
        default=0,
        metavar="N",
        help="Video stream index (default: 0).",
    )
    parser.add_argument(
        "--batch-size",
        type=non_negative_int,
        default=0,
        metavar="N",
        help="Timestamps per metric update; 0 (default) feeds the whole array at once.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the JSON report to stdout.")
    parser.add_argument(
        "--perf-profile",
        action="store_true",
        help="Emit per-phase wall-clock timings to stderr after the report.",
    )
    add_index_args(parser)
    add_threshold_args(parser)
    add_store_args(parser)
    add_storage_credential_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run every data-integrity metric on the supplied video and print the report."""
    args = _parse_args(argv)
    stats: dict[str, float] | None = {} if args.perf_profile else None
    with interrupt_guard() as interrupted:
        try:
            return _run(args, stats, interrupted)
        # Caught out here rather than inside _run so that an abort recognised *while*
        # another failure is being handled still lands on it: a handler cannot be caught
        # by its own siblings.
        except KeyboardInterrupt:
            return report_interrupted()


def _run(args: argparse.Namespace, stats: dict[str, float] | None, interrupted: threading.Event) -> int:
    """Body of :func:`main`, split out so the interrupt guard wraps the whole attempt.

    Raises:
        KeyboardInterrupt: If the operator aborted, including when libav disguised the
            abort as a decode error. :func:`main` turns it into the exit code.

    """
    try:
        validate_source(args.source)
        thresholds = thresholds_from_args(args)
        endpoint_url = resolve_s3_endpoint_url(args.endpoint_url)
        index_lookup = open_index_manifest(
            args.index_manifest, s3_profile_name=args.s3_profile_name, endpoint_url=endpoint_url
        )
        results, video_info, resolved_cfg = run_checks(
            args.source,
            expected_hz=args.expected_hz,
            thresholds=thresholds,
            stream_idx=args.stream_idx,
            batch_size=args.batch_size,
            s3_profile_name=args.s3_profile_name,
            azure_profile_name=args.azure_profile_name,
            endpoint_url=endpoint_url,
            stats=stats,
            stream_wrapper=lambda stream: cancellable_reader(stream, interrupted),
            index_lookup=index_lookup,
        )
        # Checked after a *successful* run too: libav can swallow the abort and finish
        # off a truncated source, and a report the operator has already walked away
        # from is worse than none.
        raise_if_interrupted(interrupted)

        # Rendering and writing stay inside the handler: serializing a report can fail
        # (non-finite measurements) and so can the write itself (a closed pipe), and both
        # owe the caller exit code 2 rather than a traceback.
        t_render_start = time.perf_counter()
        output: str = (
            _render_json(args.source, args.batch_size, video_info, resolved_cfg, results)
            if args.json
            else _render_human(args.source, video_info, resolved_cfg, results)
        )
        if stats is not None:
            stats["render_ms"] = (time.perf_counter() - t_render_start) * 1000

        sys.stdout.write(output)
        if stats is not None:
            # Flush the report first so interactive users see it above the perf block.
            sys.stdout.flush()
            sys.stderr.write(_render_perf(stats, video_info=video_info, batch_size=args.batch_size))
        if args.store_path is not None:
            # After the report, so the operator keeps the verdict even when persisting
            # fails -- but a failure is still exit code 2, because saving the results
            # was part of what was asked for.
            try:
                persist_run(
                    args.store_path,
                    # The stream index rides along: --stream-idx 0 and --stream-idx 1
                    # of one file are two streams, and the store keys on that.
                    [
                        stream_result(
                            args.source, results, video_info, resolved_cfg, selector_value=str(args.stream_idx)
                        )
                    ],
                    # A single-video run has no session; that null is the only
                    # structural difference between what the two CLIs write.
                    session_path=None,
                    thresholds=thresholds,
                    tool="di-check",
                    s3_profile_name=args.s3_profile_name,
                    azure_profile_name=args.azure_profile_name,
                    endpoint_url=endpoint_url,
                )
            except Exception as e:  # noqa: BLE001 - the store can fail in as many ways as its backend
                return report_error(f"checked {args.source!r} but could not write the store: {e}")
        return FAIL_EXIT_CODE if overall_status(results) is CheckStatus.FAIL else PASS_EXIT_CODE
    except StorageCliError as e:
        return report_error(str(e))
    except Exception as e:  # noqa: BLE001
        # A Ctrl-C inside a libav read arrives here as a decode error rather than a
        # KeyboardInterrupt, so trust the recorded signal over the exception type.
        raise_if_interrupted(interrupted)
        return report_error(f"could not evaluate data integrity for {args.source!r}: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
