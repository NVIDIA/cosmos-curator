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

r"""Ray Data entry point for the Curator Next ``multimodal-split`` recipe.

Usage::

    python -m cosmos_curator.next.recipes.multimodal_split.pipeline config.yaml
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from time import monotonic
from typing import Any, NamedTuple, cast

import pyarrow as pa
import ray
from loguru import logger
from ray.data.aggregate import Count, Sum

from cosmos_curator.next.core.ray_runtime import (
    configure_ray_data_progress,
    configure_ray_data_stability,
    ensure_ray_initialized,
)
from cosmos_curator.next.recipes.multimodal_split.config import ResolvedMultimodalSplitConfig, resolve_config
from cosmos_curator.next.recipes.multimodal_split.discovery import discover_candidate_sessions
from cosmos_curator.next.recipes.multimodal_split.plan import span_planner
from cosmos_curator.next.recipes.multimodal_split.preflight import check_readable, check_writable
from cosmos_curator.next.recipes.multimodal_split.split import split_session
from cosmos_curator.next.recipes.multimodal_split.storage import list_error_reports, open_sensor_source


def run_config(config: ResolvedMultimodalSplitConfig) -> dict[str, Any]:
    """Plan every discovered session into clips, write them, and report what happened.

    Discovery names the sessions, planning decides which spans every camera can
    serve, and splitting walks each camera once to write them. The rows are
    counted rather than collected: a driver that pulls every clip row back holds
    the whole run in one process, and nothing here needs more than the count.
    """
    started_at = monotonic()
    sessions = discover_candidate_sessions(config.input, storage_profile=config.execution.storage_profile)
    if sessions.num_rows == 0:
        msg = (
            f"Discovery realized 0 candidate sessions under {config.input.input_path_prefix}; "
            f"refusing to write to {config.output.path}"
        )
        raise ValueError(msg)
    logger.info("Discovered {} candidate session(s)", sessions.num_rows)

    # After discovery, because when discovery lists the input prefix it already
    # reports a missing one in terms of the config field it came from, and its
    # wording is better than anything this could add. It is the other mode that
    # needs this: reading session IDs from a file never touches the input
    # prefix, so a credential that cannot reach it surfaces once per session
    # inside Ray -- a thousand identical failures and a run that writes nothing
    # and exits successfully. Nothing checks the output at all until a clip is
    # ready to be written, by which point the run has decoded a session.
    #
    # Still before Ray: both of these answer in one request.
    check_readable(config.input.input_path_prefix, storage_profile=config.execution.storage_profile)
    check_writable(config.output.path, storage_profile=config.execution.storage_profile)

    ensure_ray_initialized()
    configure_ray_data_progress(progress=config.execution.progress)
    configure_ray_data_stability(disable_high_memory_detector=True)

    # Planning fans a session out to one row per camera; splitting turns each of
    # those into that camera's clips. One lineage, and the driver is not part of
    # it: the rows are counted in the same pass that sums their bytes, so a run
    # of ten thousand sessions never assembles its clips in one process.
    plans = plan_dataset(sessions, config)
    clips = clip_dataset(plans, config)
    totals = clips.aggregate(Count(), Sum("clip_bytes"))
    clips_written = int(totals["count()"] or 0)
    clip_bytes = int(totals["sum(clip_bytes)"] or 0)

    failures = read_failures(config)
    # A sample, not the set. One warning per failure buries the shape of a bad
    # run: a thousand lines say nothing about whether they are one cause or
    # twenty, and a run where every span was refused would print one line per
    # span. The counts say it; a few examples make it actionable.
    for entry in failures.examples:
        logger.warning(
            "Failed for {} sensor {} at {}: {} ({})",
            entry["source_session_id"],
            entry["sensor_key"],
            entry["start_ns"],
            entry["message"],
            entry["category"],
        )
    if failures.total > len(failures.examples):
        logger.warning("...and {} more failure(s)", failures.total - len(failures.examples))
    errors_by_category = failures.by_category
    elapsed_seconds = monotonic() - started_at
    summary: dict[str, Any] = {
        "sessions": sessions.num_rows,
        "camera_clips_written": clips_written,
        "failures": failures.total,
        "failures_by_category": errors_by_category,
        "clip_bytes": clip_bytes,
        "elapsed_seconds": elapsed_seconds,
        "output_path": config.output.path,
    }
    logger.info(
        "Wrote {} camera clip(s) ({}) from {} session(s) to {}",
        clips_written,
        _format_bytes(clip_bytes),
        sessions.num_rows,
        config.output.path,
    )
    # Every report under errors/, planning's and splitting's alike, so a run that
    # lost clips to a bad camera no longer summarises as clean.
    logger.info("Recorded {} failure(s) under {}/errors/", failures.total, config.output.path)
    if errors_by_category:
        logger.info(
            "Failures by category: {}",
            ", ".join(f"{category}={count}" for category, count in sorted(errors_by_category.items())),
        )
    logger.info("Multimodal split finished: elapsed={}", _format_elapsed(elapsed_seconds))
    return summary


def _format_elapsed(elapsed_seconds: float) -> str:
    """Format a monotonic duration with explicit, compact time units."""
    total_tenths = max(0, round(elapsed_seconds * 10))
    hours, remaining_tenths = divmod(total_tenths, 36_000)
    minutes, remaining_tenths = divmod(remaining_tenths, 600)
    seconds, tenths = divmod(remaining_tenths, 10)
    seconds_text = f"{seconds}.{tenths}s" if tenths else f"{seconds}s"
    if hours:
        return f"{hours}h {minutes}m {seconds_text}"
    if minutes:
        return f"{minutes}m {seconds_text}"
    return seconds_text


def _format_bytes(total: int) -> str:
    """Format a byte count in the largest unit that keeps it readable."""
    size = float(total)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":  # noqa: PLR2004 -- the base of the unit ladder
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GiB"


# Bytes per gibibyte, so the config's ``memory_gb`` maps to Ray's
# ``memory=`` (bytes) without a magic ``1024**3`` at the call site. This matches
# how the embeddings recipe converts its own per-worker reservation.
_BYTES_PER_GIB = 1024**3
# Enough to see whether a bad run is one cause or twenty, without printing a
# line per refused span.
_FAILURE_EXAMPLES = 5


def plan_dataset(session_table: pa.Table, config: ResolvedMultimodalSplitConfig) -> ray.data.Dataset:
    """Fan sessions out to one planning task each, emitting a row per camera.

    Takes the table discovery built rather than a list of dicts: from_arrow keeps
    the sessions in Arrow the whole way, where from_items would materialize a
    dict per session on the driver first.
    """
    sessions: ray.data.Dataset = ray.data.from_arrow(session_table, override_num_blocks=session_table.num_rows)
    return sessions.flat_map(
        cast("Any", span_planner),
        fn_kwargs={"config": config, "storage_profile": config.execution.storage_profile},
        # Measured over S3: a planning task keeps 0.13 of a core busy, spending
        # 96% of its wall clock waiting on storage -- 0.29 on a local disk, which
        # is why SpanPlannerResources makes this a setting rather than a constant. Asking for a whole CPU overstates
        # it eightfold and is what holds concurrency down -- Ray also scales its
        # default memory reservation by this, so one figure sets both.
        num_cpus=config.execution.span_planner.num_cpus,
    )


def clip_dataset(plans: ray.data.Dataset, config: ResolvedMultimodalSplitConfig) -> ray.data.Dataset:
    """Turn each planned camera into its clips, one row at a time.

    A camera, not a session: a task now holds one decode rather than a whole
    rig's twelve, which is a truer figure to declare. It is not more
    parallelism, though: ``flat_map`` dispatches one task per block, and the
    planner's per-camera rows stay inside their session's block, so a rig's
    cameras are split one after another in the same task. Concurrency is the
    session count either way.
    ``execution.split_session.memory_gb`` is what Ray schedules on; unset,
    Ray Data invents 2.4 GiB per CPU -- a number no one measured -- and caps the
    run by that instead.
    """
    memory_gb = config.execution.split_session.memory_gb
    return plans.flat_map(
        cast("Any", split_session),
        fn_kwargs={"config": config, "storage_profile": config.execution.storage_profile},
        memory=(None if memory_gb is None else int(memory_gb * _BYTES_PER_GIB)),
    )


class FailureSummary(NamedTuple):
    """What a run's planning failures amount to, without holding them all.

    A count and a histogram are what a summary needs; the examples are what
    makes them actionable. Collecting every failure row instead would put the
    whole set in the driver, and the set is bounded by spans refused, not by
    anything this process chose.
    """

    total: int
    by_category: dict[str, int]
    examples: tuple[dict[str, Any], ...]


def read_failures(config: ResolvedMultimodalSplitConfig) -> FailureSummary:
    """Total up every failure a run recorded, streaming them as they arrive.

    Read back rather than carried on the row stream: the summary wants every
    session's outcome, and a driver that drains the stream to get it is a second
    consumer of the dataset the next stage reads.

    Listed, then read. Reports are written only where something failed, so
    asking each session for one is a request per session that misses whenever
    the session was fine -- a million of them on a million-session run, to count
    something usually zero. One listing of the errors tree names every report
    there is, and only those are opened. It also means splitting's reports are
    counted with planning's, since both live in that tree: a run that lost clips
    to a bad camera no longer summarises as clean.
    """
    storage_profile = config.execution.storage_profile
    reports = list_error_reports(config.output.path, storage_profile=storage_profile)
    if not reports:
        return FailureSummary(total=0, by_category={}, examples=())
    read = ray.data.from_items([{"uri": uri} for uri in reports]).flat_map(
        cast("Any", _read_report_failures),
        fn_kwargs={"storage_profile": storage_profile},
    )
    counted: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    total = 0
    # iter_rows streams block by block; take_all would assemble the lot first.
    for row in read.iter_rows():
        total += 1
        counted[str(row["category"])] += 1
        if len(examples) < _FAILURE_EXAMPLES:
            examples.append(dict(row))
    return FailureSummary(total=total, by_category=dict(counted), examples=tuple(examples))


def _read_report_failures(row: dict[str, Any], *, storage_profile: str) -> list[dict[str, Any]]:
    """Read one report the listing found, and hand back the failures it holds.

    Listed rather than guessed, so a read that finds nothing means the report
    was removed between the listing and this call rather than that the session
    was fine.
    """
    report = _read_json(str(row["uri"]), storage_profile=storage_profile)
    if report is None:
        return []
    failures: list[dict[str, Any]] = report.get("failures", [])
    return failures


def _read_json(uri: str, *, storage_profile: str) -> dict[str, Any] | None:
    """Read one JSON artifact, or nothing when it was never written."""
    try:
        with open_sensor_source(uri, storage_profile=storage_profile) as stream:
            loaded: dict[str, Any] = json.loads(stream.read().decode("utf-8"))
    except FileNotFoundError:
        return None
    except Exception as error:  # noqa: BLE001 -- a bad read is not an absent file, and must not read as one
        # Absence is the ordinary case and says nothing. Anything else means a
        # session's recorded failures are missing from the summary while the
        # file that holds them is still on disk, so the count under-reports and
        # only this line says why.
        logger.warning("Could not read {}: {!r}; its failures are absent from this summary", uri, error)
        return None
    return loaded


def main() -> None:
    """Run ``multimodal-split`` from a config file."""
    parser = argparse.ArgumentParser(description="Split AV recording sessions into per-camera clips.")
    parser.add_argument("config", type=Path, help="Path to a multimodal-split config YAML file.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Dotted config override.")
    args = parser.parse_args()
    summary = run_config(resolve_config(args.config, overrides=args.overrides))
    print(json.dumps(summary, indent=2))  # noqa: T201 -- this is the command's output


if __name__ == "__main__":
    main()
