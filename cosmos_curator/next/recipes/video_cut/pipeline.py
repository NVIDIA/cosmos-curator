# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ray Data source checkpoints followed by packed, append-only Lance publication."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any, cast

import lance
import pyarrow as pa
import ray
from loguru import logger
from ray.data.aggregate import Count
from ray.data.block import BlockMetadata
from ray.data.checkpoint import CheckpointConfig
from ray.data.context import DataContext
from ray.data.datasource import Datasource, ReadTask

from cosmos_curator.next.core.ray_runtime import (
    configure_ray_data_progress,
    configure_ray_data_stability,
    ensure_ray_initialized,
)
from cosmos_curator.next.recipes.video_cut.config import ResolvedVideoCutConfig
from cosmos_curator.next.recipes.video_cut.contracts import INPUT_SCHEMA, make_source_id, span_schema
from cosmos_curator.next.recipes.video_cut.discovery import resolve_input_uris
from cosmos_curator.next.recipes.video_cut.processing import plan_sources
from cosmos_curator.next.recipes.video_cut.publication import (
    Workspace,
    cleanup_workspace,
    commit_fragments,
    completed_publication,
    expand_sources,
    open_span_table,
    pending_source_uris,
    resolve_workspace,
    select_source_results,
    source_result_files,
    stage_fragment,
)
from cosmos_curator.next.recipes.video_cut.storage import arrow_filesystem, filesystem_path
from cosmos_curator.next.utils.identity import canonical_digest


def _source_batches(uris: tuple[str, ...]) -> Iterator[pa.Table]:
    yield pa.Table.from_pylist(
        [{"source_id": make_source_id(uri), "source_uri": uri} for uri in uris],
        schema=INPUT_SCHEMA,
    )


class SourceDatasource(Datasource):
    """Expose source identities at Ray's native recovery-filter insertion point."""

    def __init__(self, uris: tuple[str, ...]) -> None:
        """Capture normalized source URIs without reading video bytes."""
        super().__init__()  # type: ignore[no-untyped-call]
        self._uris = uris

    def estimate_inmemory_data_size(self) -> int:
        """Estimate small URI/identity records without touching media."""
        return sum(len(uri.encode()) + 80 for uri in self._uris)

    def get_read_tasks(
        self,
        parallelism: int,
        per_task_row_limit: int | None = None,
        data_context: DataContext | None = None,
    ) -> list[ReadTask]:
        """Partition the source list independently of worker membership."""
        del data_context
        count = min(max(1, parallelism), len(self._uris))
        tasks = []
        for index in range(count):
            uris = self._uris[index * len(self._uris) // count : (index + 1) * len(self._uris) // count]
            tasks.append(
                ReadTask(
                    partial(_source_batches, uris),
                    BlockMetadata(num_rows=len(uris), size_bytes=None, input_files=None, exec_stats=None),
                    schema=INPUT_SCHEMA,
                    per_task_row_limit=per_task_row_limit,
                )
            )
        return tasks


@contextmanager
def source_checkpoints(config: ResolvedVideoCutConfig, workspace: Workspace) -> Iterator[None]:
    """Install native checkpoints for exactly one read-to-Parquet execution."""
    context = DataContext.get_current()
    if context.checkpoint_config is not None:
        msg = "video-cut cannot replace an existing Ray checkpoint configuration"
        raise RuntimeError(msg)
    context.checkpoint_config = CheckpointConfig(
        id_column="source_id",
        checkpoint_path=workspace.checkpoints,
        delete_checkpoint_on_success=False,
        override_filesystem=arrow_filesystem(workspace.root, config.execution.storage_profile),
    )
    try:
        yield
    finally:
        context.checkpoint_config = None


def cut_sources(config: ResolvedVideoCutConfig, workspace: Workspace, source_uris: tuple[str, ...]) -> None:
    """Checkpoint complete source plans before any Lance publication begins."""
    with source_checkpoints(config, workspace):
        sources = ray.data.read_datasource(
            SourceDatasource(source_uris),
            override_num_blocks=min(len(source_uris), 256),
        )
        planned = sources.map_batches(
            cast("Any", plan_sources),
            batch_size=None,
            batch_format="pyarrow",
            fn_kwargs={"split": config.split, "execution": config.execution},
            num_cpus=config.execution.probe_cpus,
        )
        groups = planned.repartition(target_num_rows_per_block=config.execution.sources_per_checkpoint, strict=True)
        groups.write_parquet(
            workspace.results,
            filesystem=arrow_filesystem(workspace.root, config.execution.storage_profile),
            min_rows_per_file=config.execution.sources_per_checkpoint,
            max_rows_per_file=config.execution.sources_per_checkpoint,
        )


def _write_error_report(outcomes: ray.data.Dataset, config: ResolvedVideoCutConfig, input_snapshot_id: str) -> str:
    """Keep terminal diagnostics with the output before the commit makes them visible."""
    uri = f"{config.output.spans_lance_uri}/video-cut/{input_snapshot_id}/errors.jsonl"
    fs = arrow_filesystem(uri, config.execution.storage_profile)
    path = filesystem_path(uri)
    fs.create_dir(path.rsplit("/", 1)[0], recursive=True)
    errors = outcomes.select_columns(["source_id", "source_uri", "status", "error_stage", "error_message"]).filter(
        lambda row: row["status"] == "error"
    )
    with fs.open_output_stream(path) as report:
        for row in errors.iter_rows():
            report.write((json.dumps(row, sort_keys=True) + "\n").encode())
    return uri


def publish_spans(
    config: ResolvedVideoCutConfig, workspace: Workspace, *, input_snapshot_id: str, source_uris: tuple[str, ...]
) -> lance.LanceDataset:
    """Pack this run's new sources and append them in one atomic commit."""
    dataset = open_span_table(config, input_snapshot_id)
    if completed_publication(dataset, input_snapshot_id) is not None:
        return dataset
    files = source_result_files(workspace, config)
    if not files:
        msg = "No source results are available for the frozen input snapshot"
        raise ValueError(msg)
    expected = {make_source_id(uri) for uri in source_uris}
    outcomes = ray.data.read_parquet(
        files,
        filesystem=arrow_filesystem(workspace.results, config.execution.storage_profile),
    ).map_batches(
        cast("Any", select_source_results),
        batch_format="pyarrow",
        batch_size=None,
        fn_kwargs={"source_ids": tuple(sorted(expected))},
        num_cpus=0.25,
    )
    # A previous attempt may have checkpointed sources that another, sequential
    # cut invocation has since published. Reuse only this attempt's new sources.
    source_results = errors = 0
    counts = outcomes.select_columns(["status"]).groupby("status").aggregate(Count(alias_name="sources"))
    for outcome in counts.take_all():
        if outcome["status"] not in {"success", "error"}:
            msg = f"Invalid source outcome: {outcome['status']}"
            raise ValueError(msg)
        source_results += outcome["sources"]
        if outcome["status"] == "error":
            errors += outcome["sources"]
    source_ids = [row["source_id"] for row in outcomes.select_columns(["source_id"]).iter_rows()]
    if source_results != len(expected) or len(source_ids) != len(expected) or set(source_ids) != expected:
        msg = "Source results do not cover the frozen input snapshot"
        raise ValueError(msg)
    errors_uri = _write_error_report(outcomes, config, input_snapshot_id) if errors else None
    expanded = outcomes.map_batches(
        cast("Any", expand_sources),
        batch_format="pyarrow",
        batch_size=None,
        fn_kwargs={"contract_id": config.split.contract_id, "spans_per_fragment": config.execution.spans_per_fragment},
    )
    packed = expanded.repartition(target_num_rows_per_block=config.execution.spans_per_fragment, strict=True)
    results = packed.map_batches(
        cast("Any", stage_fragment),
        batch_format="pyarrow",
        batch_size=None,
        fn_kwargs={
            "schema": span_schema(config.split, input_snapshot_id),
            "spans_lance_uri": config.output.spans_lance_uri,
            "spans_per_fragment": config.execution.spans_per_fragment,
            "storage_profile": config.execution.storage_profile,
        },
    )
    candidates: list[str] = []
    for batch in results.iter_batches(prefetch_batches=0, batch_size=None, batch_format="pyarrow"):
        candidates.extend(cast("pa.Table", batch).column("fragment").to_pylist())
    return commit_fragments(
        candidates,
        config=config,
        input_snapshot_id=input_snapshot_id,
        publication={
            "source_results": source_results,
            "errors": errors,
            "errors_uri": errors_uri,
            "source_ids": sorted(source_ids),
        },
        read_version=int(dataset.version),
    )


def run_config(config: ResolvedVideoCutConfig) -> dict[str, Any]:
    """Append new immutable sources, preserving all existing fragments and enrichment."""
    source_uris = resolve_input_uris(config.input, storage_profile=config.execution.storage_profile)
    input_snapshot_id = canonical_digest(source_uris)
    dataset = open_span_table(config, input_snapshot_id)
    publication = completed_publication(dataset, input_snapshot_id)
    cleanup_complete: bool | None = None
    if publication is None:
        pending = pending_source_uris(dataset, source_uris)
        if not pending:
            publication = {"source_results": 0, "errors": 0, "errors_uri": None}
        else:
            dataset, publication, cleanup_complete = _run_pending(config, pending, input_snapshot_id)
    summary = {
        "selected_sources": len(source_uris),
        "source_results": publication["source_results"],
        "errors": publication["errors"],
        "errors_uri": publication["errors_uri"],
        "spans": dataset.count_rows(),
        "spans_lance_uri": config.output.spans_lance_uri,
        "spans_lance_version": int(dataset.version),
        "cleanup_complete": cleanup_complete,
    }
    logger.info(
        "video-cut: {} canonical span(s), {} source result(s), {} error(s)",
        summary["spans"],
        summary["source_results"],
        summary["errors"],
    )
    return summary


def _run_pending(
    config: ResolvedVideoCutConfig, pending: tuple[str, ...], input_snapshot_id: str
) -> tuple[lance.LanceDataset, dict[str, Any], bool]:
    """Process one finite selection of new sources and clean only its own workspace."""
    workspace = resolve_workspace(config, input_snapshot_id)
    ensure_ray_initialized()
    configure_ray_data_progress(progress=config.execution.progress)
    configure_ray_data_stability()
    cut_sources(config, workspace, pending)
    dataset = publish_spans(config, workspace, input_snapshot_id=input_snapshot_id, source_uris=pending)
    publication = completed_publication(dataset, input_snapshot_id)
    if publication is None:
        msg = "Span publication completed without its completion record"
        raise RuntimeError(msg)
    try:
        cleanup_workspace(workspace, config)
    except Exception as exc:  # noqa: BLE001 -- cleanup cannot invalidate a successful publication
        logger.warning("Spans are committed at Lance v{}, but workspace cleanup failed: {!r}", dataset.version, exc)
        return dataset, publication, False
    return dataset, publication, True
