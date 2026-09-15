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

r"""Entry point for ``robot-action-split`` pipeline runs.

Reconciles freshly discovered spans against the canonical Lance clip table,
processes only missing work, and commits successful clip rows incrementally
as bounded fragments — the same recovery shape as ``video_split``, adapted to
this recipe's cheap always-rerun parquet discovery (see
``docs/curator/design/curator-next-robot-action-split.md``, "Cross-Run Recovery").

Usage::

    python -m cosmos_curator.next.recipes.robot_action_split.pipeline \
        --config path/to/config.yaml

Example config (local paths, handful of videos)::

    schema_version: 1
    kind: robot-action-split
    input:
      uris: [/data/robot_dataset/]
      source_dataset: my_dataset
      limit: 5
    output:
      media_root: /tmp/robot_clips/
      lance_uri: /tmp/robot_clips/lance/clips.lance
    execution:
      discovery_workers: 2
"""

import argparse
import itertools
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

if TYPE_CHECKING:
    import lance
    import pyarrow as pa

from cosmos_curator.core.utils.misc.retry_utils import do_with_retries
from cosmos_curator.next.recipes.robot_action_split.config import ResolvedRobotActionSplitConfig, load_config
from cosmos_curator.next.recipes.robot_action_split.contracts import CLIP_RECORD_SCHEMA_VERSION, MEDIA_CONTRACT_VERSION
from cosmos_curator.next.recipes.robot_action_split.discovery import ChunkSpanBatch, discover_spans
from cosmos_curator.next.recipes.robot_action_split.lance_sink import (
    append_clip_fragment,
    open_or_create_clip_table,
    write_clip_fragment,
)
from cosmos_curator.next.recipes.robot_action_split.processing import process_batch
from cosmos_curator.next.recipes.robot_action_split.records import clip_table, error_records
from cosmos_curator.next.recipes.robot_action_split.recovery import reconcile_batches
from cosmos_curator.next.utils.storage import write_media

_ERROR_REPORT_BACKOFF_FACTOR = 2.0
_ERROR_REPORT_MAX_BACKOFF_S = 30.0


def _iter_sequential(
    batches: list[ChunkSpanBatch],
    config: ResolvedRobotActionSplitConfig,
) -> Iterator[dict[str, Any]]:
    """Process batches one at a time, reusing each downloaded chunk across its batches."""
    total = len(batches)
    batch_num = 0
    # Group consecutive batches by source chunk so the video file is downloaded
    # once per chunk rather than once per batch.  max_segments_per_batch splits
    # one chunk's spans across many batches; without grouping each call to
    # process_batch would re-download the same (potentially large) source file.
    for _chunk_uri, chunk_batches_iter in itertools.groupby(batches, key=lambda b: b.chunk_mp4_uri):
        with tempfile.TemporaryDirectory(dir=config.execution.tmp_dir) as chunk_tmp:
            staged_path = str(Path(chunk_tmp) / "chunk.mp4")
            for batch in chunk_batches_iter:
                batch_num += 1
                logger.info(f"  Batch {batch_num}/{total}: {batch.chunk_mp4_uri} ({len(batch.items)} span(s))")
                outcomes = process_batch(batch, config=config, staged_chunk_path=staged_path)
                n_ok = sum(1 for o in outcomes if o["status"] == "success")
                n_fail = len(outcomes) - n_ok
                logger.info(f"    -> {n_ok} succeeded, {n_fail} failed")
                for o in outcomes:
                    if o["status"] != "success":
                        logger.warning(f"      FAIL [{o.get('error_stage')}] {o.get('error_message', '')[:300]}")
                yield from outcomes


def _iter_ray_data(
    batches: list[ChunkSpanBatch],
    config: ResolvedRobotActionSplitConfig,
) -> Iterator[dict[str, Any]]:
    """Process batches in parallel using Ray Data flat_map across Ray workers.

    Each task downloads its own copy of the source chunk — the staged-path
    optimisation used by the sequential path does not apply here because tasks
    are distributed across workers that do not share a local filesystem.  The
    trade-off is acceptable: Ray Data parallelises many chunks simultaneously,
    so overall throughput improves even though individual chunks are re-fetched
    per task.
    """
    import ray  # noqa: PLC0415
    import ray.data  # noqa: PLC0415

    if not ray.is_initialized():
        # In a managed Slurm-Ray run the driver sets RAY_ADDRESS to the head's
        # address before launching this process.  For standalone sbatch jobs and
        # local runs (development, CI) RAY_ADDRESS is unset and ray.init() starts
        # a single-node cluster on the current machine.
        ray_address = os.environ.get("RAY_ADDRESS") or None
        ray.init(address=ray_address)
        logger.info(f"Ray initialised: {ray.cluster_resources()}")  # type: ignore[no-untyped-call]

    def _udf(batch_row: dict[str, Any]) -> list[dict[str, Any]]:
        """Ray Data UDF: process one ChunkSpanBatch; config is captured in the closure."""
        batch: ChunkSpanBatch = batch_row["batch"]
        return process_batch(batch, config=config)

    rows = [{"batch": b} for b in batches]
    ds = ray.data.from_items(rows, override_num_blocks=len(rows))

    # concurrency=None lets Ray choose based on available CPU resources.
    # Each task uses cut_cpus CPUs as declared in the execution config.
    result_ds = ds.flat_map(_udf, num_cpus=config.execution.cut_cpus)
    # iter_rows() hardcodes one batch of prefetch: Ray fetches the next
    # RefBundle before yielding the current one, delaying every fragment
    # commit until another block finishes (matches video_split's
    # _iter_publication_rows fix for the same underlying behavior).
    result_batches = result_ds.iter_batches(prefetch_batches=0, batch_size=None, batch_format="pyarrow")
    for result_batch in result_batches:
        for outcome in cast("pa.Table", result_batch).to_pylist():
            if outcome["status"] != "success":
                logger.warning(f"  FAIL [{outcome.get('error_stage')}] {outcome.get('error_message', '')[:300]}")
            yield outcome


def _commit_clip_rows(rows: list[dict[str, Any]], *, config: ResolvedRobotActionSplitConfig) -> int:
    """Stage and idempotently commit one Lance fragment of successful clip rows."""
    table = clip_table(
        rows,
        record_schema_version=CLIP_RECORD_SCHEMA_VERSION,
        media_contract_version=MEDIA_CONTRACT_VERSION,
    )
    candidate = write_clip_fragment(
        table, uri=config.output.lance_uri, storage_profile=config.execution.output_storage_profile
    )
    if candidate is None:
        # Not just an internal invariant: any caller that hands this an all-failure
        # batch hits this path directly, so this must survive `python -O`.
        msg = "Publication buffer flushed with no successful rows"
        raise ValueError(msg)
    return append_clip_fragment(
        candidate,
        uri=config.output.lance_uri,
        storage_profile=config.execution.output_storage_profile,
        attempts=config.execution.storage_attempts,
    )


def _publish_outcomes(
    outcomes: Iterable[dict[str, Any]],
    *,
    config: ResolvedRobotActionSplitConfig,
    initial_clips_version: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """Buffer successful rows and commit a fragment each time the buffer fills.

    Mirrors ``video_split``'s streaming publication: a long run commits every
    ``clips_per_publish_batch`` successful rows as they become available rather
    than waiting for the whole run to finish, bounding how much work a crash
    can discard. A short run still gets exactly one commit, at the end.

    Only failed outcomes are retained in memory for the whole run (needed for
    ``errors.json``); successful rows leave the buffer as soon as they're
    committed, so driver memory does not grow with total run size.
    """
    threshold = config.execution.clips_per_publish_batch
    buffer: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    succeeded_count = 0
    clips_version = initial_clips_version
    for outcome in outcomes:
        if outcome["status"] == "success":
            succeeded_count += 1
            buffer.append(outcome)
        else:
            failed.append(outcome)
        if len(buffer) >= threshold:
            clips_version = _commit_clip_rows(buffer, config=config)
            buffer = []
    if buffer:
        clips_version = _commit_clip_rows(buffer, config=config)
    return failed, succeeded_count, clips_version


def run(
    config_path: str,
    *,
    config: ResolvedRobotActionSplitConfig | None = None,
) -> dict[str, object]:
    """Discover spans, reconcile against committed clips, cut, and publish incrementally."""
    resolved = config if config is not None else load_config(config_path)
    logger.info(f"Loaded config: kind={resolved.kind} source_dataset={resolved.input.source_dataset}")

    # Phase 1: span discovery (pre-Ray, CPU-bound parquet reads). Cheap enough
    # to always rerun in full — see the recovery design doc for why this
    # recipe does not need video_split's "known source" replan-avoidance path.
    logger.info("Discovering spans...")
    batches = discover_spans(resolved)
    total_items = sum(len(b.items) for b in batches)
    logger.info(f"Discovery complete: {len(batches)} batch(es), {total_items} span(s)")

    clip_dataset: lance.LanceDataset = open_or_create_clip_table(
        uri=resolved.output.lance_uri,
        storage_profile=resolved.execution.output_storage_profile,
    )

    if not batches:
        logger.warning("No spans found; nothing to do.")
        failed: list[dict[str, Any]] = []
        succeeded_count = 0
        clips_version = int(clip_dataset.version)
    else:
        reconciliation = reconcile_batches(batches, dataset=clip_dataset)
        logger.info(
            f"Reconciled {reconciliation.complete_batches} complete, {reconciliation.partial_batches} partial, "
            f"and {reconciliation.unknown_batches} unknown batch(es) from "
            f"{reconciliation.committed_clip_rows} committed clip row(s)"
        )

        if reconciliation.batches:
            logger.info("Cutting clips...")
            remaining = list(reconciliation.batches)
            outcomes_iter = (
                _iter_ray_data(remaining, resolved)
                if resolved.execution.ray_data
                else _iter_sequential(remaining, resolved)
            )
            failed, succeeded_count, clips_version = _publish_outcomes(
                outcomes_iter,
                config=resolved,
                initial_clips_version=int(clip_dataset.version),
            )
        else:
            failed = []
            succeeded_count = 0
            clips_version = int(clip_dataset.version)

    logger.info(f"Done: {succeeded_count} clip(s) written, {len(failed)} failed")
    if failed:
        for o in failed:
            logger.warning(f"  FAILED {o['clip_id']}: [{o.get('error_stage')}] {o.get('error_message')}")
    logger.info(f"Lance dataset version after run: {clips_version}")

    # Replace the error report every run, local or S3 — the durable record of
    # this run's failures. Never delays or is delayed by the canonical clip
    # commits above, matching video_split's report-after-commits ordering.
    media_root = resolved.output.media_root
    errors_uri = f"{media_root.rstrip('/')}/errors.json"
    do_with_retries(
        partial(
            write_media,
            errors_uri,
            json.dumps(error_records(failed), indent=2).encode("utf-8"),
            storage_profile=resolved.execution.output_storage_profile,
        ),
        max_attempts=resolved.execution.storage_attempts,
        backoff_factor=_ERROR_REPORT_BACKOFF_FACTOR,
        max_wait_time_s=_ERROR_REPORT_MAX_BACKOFF_S,
        name="error-report-write",
    )
    logger.info(f"Error report written to {errors_uri}")

    total = succeeded_count + len(failed)

    # Write a run summary (local only; skipped for S3 media_root to keep the
    # driver stateless — the Lance dataset and errors.json are the durable record).
    # "outcomes" carries only the failed rows: successful rows are never retained
    # in full for the whole run (see _publish_outcomes), and the durable record of
    # a success is the committed Lance row, not this summary.
    if not media_root.startswith("s3://"):
        summary_path = Path(media_root.rstrip("/")) / "run_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "total": total,
                    "succeeded": succeeded_count,
                    "failed": len(failed),
                    "outcomes": failed,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(f"Summary written to {summary_path}")

    return {
        "total": total,
        "succeeded": succeeded_count,
        "failed": len(failed),
        "outcomes": failed,
        "clips_lance_version": clips_version,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Curator Next robot-action-split pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML or JSON config file")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
