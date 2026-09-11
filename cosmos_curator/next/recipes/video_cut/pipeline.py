# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ray Data source checkpoints followed by packed, atomic Lance span publication."""

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
    resolve_workspace,
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
        source_tasks = sources.repartition(target_num_rows_per_block=1, strict=True)
        planned = source_tasks.map_batches(
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


def _write_error_report(outcomes: ray.data.Dataset, config: ResolvedVideoCutConfig) -> str:
    """Keep terminal diagnostics with the output before the commit makes them visible."""
    uri = f"{config.output.spans_lance_uri}/video-cut/errors.jsonl"
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
    config: ResolvedVideoCutConfig, workspace: Workspace, *, input_snapshot_id: str, source_count: int
) -> lance.LanceDataset:
    """Pack the frozen source snapshot and publish it in one atomic commit."""
    dataset = open_span_table(config, input_snapshot_id)
    if completed_publication(dataset) is not None:
        return dataset
    files = source_result_files(workspace, config)
    if not files:
        msg = "No source results are available for the frozen input snapshot"
        raise ValueError(msg)
    outcomes = ray.data.read_parquet(
        files,
        filesystem=arrow_filesystem(workspace.results, config.execution.storage_profile),
    )
    source_results = errors = 0
    counts = outcomes.select_columns(["status"]).groupby("status").aggregate(Count(alias_name="sources"))
    for outcome in counts.take_all():
        if outcome["status"] not in {"success", "error"}:
            msg = f"Invalid source outcome: {outcome['status']}"
            raise ValueError(msg)
        source_results += outcome["sources"]
        if outcome["status"] == "error":
            errors += outcome["sources"]
    if source_results != source_count:
        msg = "Source results do not cover the frozen input snapshot"
        raise ValueError(msg)
    errors_uri = _write_error_report(outcomes, config) if errors else None
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
        },
        read_version=int(dataset.version),
    )


def run_config(config: ResolvedVideoCutConfig) -> dict[str, Any]:
    """Publish or reopen a frozen source snapshot, cleaning only the workspace used in this invocation."""
    source_uris = resolve_input_uris(config.input, storage_profile=config.execution.storage_profile)
    input_snapshot_id = canonical_digest(source_uris)
    dataset = open_span_table(config, input_snapshot_id)
    publication = completed_publication(dataset)
    cleanup_complete: bool | None = None
    if publication is None:
        workspace = resolve_workspace(config, input_snapshot_id)
        ensure_ray_initialized()
        configure_ray_data_progress(progress=config.execution.progress)
        configure_ray_data_stability()
        cut_sources(config, workspace, source_uris)
        dataset = publish_spans(config, workspace, input_snapshot_id=input_snapshot_id, source_count=len(source_uris))
        publication = completed_publication(dataset)
        if publication is None:
            msg = "Span publication completed without its completion record"
            raise RuntimeError(msg)
        try:
            cleanup_workspace(workspace, config)
        except Exception as exc:  # noqa: BLE001 -- cleanup cannot invalidate a successful publication
            cleanup_complete = False
            logger.warning("Spans are committed at Lance v{}, but workspace cleanup failed: {!r}", dataset.version, exc)
        else:
            cleanup_complete = True
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
