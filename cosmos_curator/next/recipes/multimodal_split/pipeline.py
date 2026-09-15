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
from ray.data.aggregate import Sum
from ray.data.expressions import col

from cosmos_curator.next.core.ray_runtime import (
    configure_ray_data_progress,
    configure_ray_data_stability,
    ensure_ray_initialized,
)
from cosmos_curator.next.recipes.multimodal_split.config import ResolvedMultimodalSplitConfig, resolve_config
from cosmos_curator.next.recipes.multimodal_split.discovery import discover_candidate_sessions
from cosmos_curator.next.recipes.multimodal_split.identities import make_source_id
from cosmos_curator.next.recipes.multimodal_split.lance_sink import (
    append_episode_fragment,
    open_episode_table,
    open_or_create_episode_table,
)
from cosmos_curator.next.recipes.multimodal_split.plan import (
    ALREADY_CANONICAL,
    CAMERA_PLAN,
    MCAP_PLAN,
    remaining_episode_ids_batch,
    span_planner_batch,
)
from cosmos_curator.next.recipes.multimodal_split.preflight import check_readable, check_writable
from cosmos_curator.next.recipes.multimodal_split.publication import (
    PublicationSummary,
    assemble_episode_group,
    stage_episode_batch,
)
from cosmos_curator.next.recipes.multimodal_split.recovery import ReconciliationMap, read_reconciliation
from cosmos_curator.next.recipes.multimodal_split.split import split_camera_batch, split_mcap_batch
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

    episode_dataset = open_or_create_episode_table(
        uri=config.output.episodes_lance_uri,
        storage_profile=config.execution.storage_profile,
    )
    source_ids = [make_source_id(str(location)) for location in sessions["session_uri"].to_pylist()]
    # Read existing episode facts from one driver-pinned snapshot so planner
    # workers do not open Lance.
    reconciliation = read_reconciliation(episode_dataset, source_ids)

    ensure_ray_initialized()
    configure_ray_data_progress(progress=config.execution.progress)
    configure_ray_data_stability(disable_high_memory_detector=True)

    # Planning fans a session out to one row per camera; splitting turns each of
    # those into that camera's clips. One lineage, and the driver is not part of
    # it: the rows are counted in the same pass that sums their bytes, so a run
    # of ten thousand sessions never assembles its clips in one process.
    # Planning writes manifests for absent work, so materialize it before the
    # tagged stream is consumed twice for encoding and skip accounting.
    plans = plan_dataset(sessions, config, reconciliation=reconciliation).materialize()
    already_canonical_plans = plans.filter(expr=col("result_kind") == ALREADY_CANONICAL)
    camera_plans = plans.filter(expr=col("result_kind") == CAMERA_PLAN)
    already_canonical = already_canonical_plans.count()
    # Camera rows repeat each planned episode ID. De-duplicate within each
    # block first to discard the large repeated plan payload, then globally
    # group the projected IDs so Ray block shaping cannot inflate the count.
    remaining_episode_ids = camera_plans.map_batches(
        cast("Any", remaining_episode_ids_batch),
        batch_format="pyarrow",
        batch_size=None,
    )
    episodes_remaining = _count_unique_episode_ids(remaining_episode_ids)
    episodes_planned = already_canonical + episodes_remaining
    if episodes_remaining:
        outcomes = clip_dataset(plans, config)
        # Cache assembled results so publication's later consumers cannot replay
        # splitting's media writes.
        assembled = assembly_dataset(outcomes).materialize()
        publication = publish_assembled_episodes(assembled, config=config)
    else:
        publication = PublicationSummary(
            episodes_newly_published=0,
            episodes_dropped=0,
            camera_clips_written=0,
            clip_bytes=0,
        )
    validate_episode_accounting(
        episodes_planned=episodes_planned,
        episodes_already_canonical=already_canonical,
        episodes_newly_published=publication.episodes_newly_published,
        episodes_dropped=publication.episodes_dropped,
    )
    final_episode_dataset = open_episode_table(
        uri=config.output.episodes_lance_uri,
        storage_profile=config.execution.storage_profile,
    )
    final_lance_version = int(final_episode_dataset.version)

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
        "camera_clips_written": publication.camera_clips_written,
        "sensor_clips_written": publication.sensor_clips_written,
        "sensor_clip_bytes": publication.sensor_clip_bytes,
        "episodes_planned": episodes_planned,
        "episodes_already_canonical": already_canonical,
        "episodes_newly_published": publication.episodes_newly_published,
        "episodes_dropped": publication.episodes_dropped,
        "failures": failures.total,
        "failures_by_category": errors_by_category,
        "clip_bytes": publication.clip_bytes,
        "elapsed_seconds": elapsed_seconds,
        "output_path": config.output.path,
        "episodes_lance_uri": config.output.episodes_lance_uri,
        "episodes_lance_version": final_lance_version,
    }
    logger.info(
        "Wrote {} camera clip(s) ({}) from {} session(s) to {}",
        publication.camera_clips_written,
        _format_bytes(publication.clip_bytes),
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
    logger.info(
        "Canonical episodes at {} v{}: planned={}, already_canonical={}, newly_published={}, dropped={}",
        config.output.episodes_lance_uri,
        final_lance_version,
        episodes_planned,
        already_canonical,
        publication.episodes_newly_published,
        publication.episodes_dropped,
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


def validate_episode_accounting(
    *,
    episodes_planned: int,
    episodes_already_canonical: int,
    episodes_newly_published: int,
    episodes_dropped: int,
) -> None:
    """Require every independently planned episode to reach one terminal state."""
    accounted = episodes_already_canonical + episodes_newly_published + episodes_dropped
    if episodes_planned != accounted:
        msg = (
            f"Episode accounting mismatch: planned={episodes_planned}, "
            f"already_canonical={episodes_already_canonical}, "
            f"newly_published={episodes_newly_published}, dropped={episodes_dropped}"
        )
        raise RuntimeError(msg)


def _count_unique_episode_ids(episode_ids: ray.data.Dataset) -> int:
    """Count episode IDs globally, independent of Ray block layout."""
    # Exact by construction. Use distributed key grouping rather than
    # CountDistinct's global unique-set accumulator because this value gates a
    # hard post-publication accounting failure.
    return int(episode_ids.groupby("split_clip_id").count().count())


# Bytes per gibibyte, so the config's ``memory_gb`` maps to Ray's
# ``memory=`` (bytes) without a magic ``1024**3`` at the call site. This matches
# how the embeddings recipe converts its own per-worker reservation.
_BYTES_PER_GIB = 1024**3
# Enough to see whether a bad run is one cause or twenty, without printing a
# line per refused span.
_FAILURE_EXAMPLES = 5


def plan_dataset(
    session_table: pa.Table,
    config: ResolvedMultimodalSplitConfig,
    *,
    reconciliation: ReconciliationMap | None = None,
) -> ray.data.Dataset:
    """Fan sessions out to one planning task each, emitting a row per camera.

    Takes the table discovery built rather than a list of dicts: from_arrow keeps
    the sessions in Arrow the whole way, where from_items would materialize a
    dict per session on the driver first.
    """
    sessions: ray.data.Dataset = ray.data.from_arrow(session_table, override_num_blocks=session_table.num_rows)
    return sessions.map_batches(
        cast("Any", span_planner_batch),
        batch_format="pyarrow",
        batch_size=None,
        fn_kwargs={
            "config": config,
            "storage_profile": config.execution.storage_profile,
            "reconciliation": reconciliation or {},
        },
        # Measured over S3: a planning task keeps 0.13 of a core busy, spending
        # 96% of its wall clock waiting on storage -- 0.29 on a local disk, which
        # is why SpanPlannerResources makes this a setting rather than a constant. Asking for a whole CPU overstates
        # it eightfold and is what holds concurrency down -- Ray also scales its
        # default memory reservation by this, so one figure sets both.
        num_cpus=config.execution.span_planner.num_cpus,
    )


def clip_dataset(plans: ray.data.Dataset, config: ResolvedMultimodalSplitConfig) -> ray.data.Dataset:
    """Turn each planned camera into fixed-schema terminal outcomes.

    A camera, not a session: a task now holds one decode rather than a whole
    rig's twelve, which is a truer figure to declare. It is not more
    parallelism, though: ``map_batches`` dispatches one task per block, and the
    planner's per-camera rows stay inside their session's block, so a rig's
    cameras are split one after another in the same task. Concurrency is the
    session count either way.
    ``execution.split_session.memory_gb`` is what Ray schedules on; unset,
    Ray Data invents 2.4 GiB per CPU -- a number no one measured -- and caps the
    run by that instead.
    """
    memory_gb = config.execution.split_session.memory_gb
    fn_kwargs = {"config": config, "storage_profile": config.execution.storage_profile}
    camera_clips = plans.filter(expr=col("result_kind") == CAMERA_PLAN).map_batches(
        cast("Any", split_camera_batch),
        batch_format="pyarrow",
        batch_size=None,
        fn_kwargs=fn_kwargs,
        memory=(None if memory_gb is None else int(memory_gb * _BYTES_PER_GIB)),
    )
    # Two stages rather than one branching on sensor type, because their costs
    # have nothing in common: a camera decodes and re-encodes video, while an
    # MCAP is copied. Scheduling both under the camera's memory reservation would
    # price the cheap one at the expensive one's rate.
    mcap_clips = plans.filter(expr=col("result_kind") == MCAP_PLAN).map_batches(
        cast("Any", split_mcap_batch),
        batch_format="pyarrow",
        batch_size=None,
        fn_kwargs=fn_kwargs,
    )
    # One terminal stream, because an episode is assembled from every artifact in
    # it at once: grouping by clip has to see a clip's video and its trajectory
    # in the same group or completeness cannot be decided.
    return camera_clips.union(mcap_clips)


def assembly_dataset(outcomes: ray.data.Dataset) -> ray.data.Dataset:
    """Group current-run sensor outcomes into one terminal episode result."""
    # Completeness belongs to the full planned camera set, so this episode-level
    # shuffle is required; publishing camera outcomes directly could publish partial episodes.
    return outcomes.groupby("split_clip_id").map_groups(
        cast("Any", assemble_episode_group),
        batch_format="pyarrow",
        zero_copy_batch=True,
    )


# Reuse robot_action_split's private 8,000-row publish-batch default.
# Tune independently when multimodal episode fragment measurements exist.
_EPISODES_PER_PUBLICATION_BLOCK = 8_000
_APPEND_ATTEMPTS = 3


def publish_assembled_episodes(
    assembled: ray.data.Dataset,
    *,
    config: ResolvedMultimodalSplitConfig,
) -> PublicationSummary:
    """Stage complete rows on workers, commit serially, and total terminal accounting."""
    totals = assembled.aggregate(
        Sum("camera_clips_written"),
        Sum("camera_clip_bytes"),
        Sum("sensor_clips_written"),
        Sum("sensor_clip_bytes"),
    )
    camera_clips_written = int(totals["sum(camera_clips_written)"] or 0)
    clip_bytes = int(totals["sum(camera_clip_bytes)"] or 0)
    sensor_clips_written = int(totals["sum(sensor_clips_written)"] or 0)
    sensor_clip_bytes = int(totals["sum(sensor_clip_bytes)"] or 0)
    dropped = assembled.filter(expr=col("assembly_status") == "dropped").count()
    complete = assembled.filter(expr=col("assembly_status") == "complete")
    complete_count = complete.count()
    if complete_count == 0:
        return PublicationSummary(0, dropped, camera_clips_written, clip_bytes, sensor_clips_written, sensor_clip_bytes)

    batches = complete.repartition(target_num_rows_per_block=_EPISODES_PER_PUBLICATION_BLOCK, strict=True)
    staged = batches.map_batches(
        cast("Any", stage_episode_batch),
        batch_format="pyarrow",
        batch_size=None,
        fn_kwargs={
            "uri": config.output.episodes_lance_uri,
            "storage_profile": config.execution.storage_profile,
        },
    )
    published = 0
    for batch in staged.iter_batches(prefetch_batches=0, batch_size=None, batch_format="pyarrow"):
        for result in cast("pa.Table", batch).to_pylist():
            append_episode_fragment(
                str(result["candidate_json"]),
                uri=config.output.episodes_lance_uri,
                storage_profile=config.execution.storage_profile,
                attempts=_APPEND_ATTEMPTS,
            )
            published += int(result["episode_count"])
    return PublicationSummary(
        published, dropped, camera_clips_written, clip_bytes, sensor_clips_written, sensor_clip_bytes
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
