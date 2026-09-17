# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Latest Lance spans -> checkpointed clip metadata -> one Lance publication."""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any, cast

import lance
import pyarrow as pa
import pyarrow.compute as pc
import ray
from loguru import logger
from ray.data.block import BlockMetadata
from ray.data.checkpoint import CheckpointConfig
from ray.data.context import DataContext, ShuffleStrategy
from ray.data.datasource import Datasource, ReadTask

from cosmos_curator.core.utils import environment
from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.next.core.ray_runtime import (
    configure_ray_data_progress,
    configure_ray_data_stability,
    curator_io_resources,
    ensure_ray_initialized,
)
from cosmos_curator.next.recipes.video_cut.contracts import SPAN_SCHEMA
from cosmos_curator.next.recipes.video_transcode.config import ResolvedVideoTranscodeConfig
from cosmos_curator.next.recipes.video_transcode.contracts import ADDRESS_FIELDS, completed_publication
from cosmos_curator.next.recipes.video_transcode.processing import (
    SourceDownloader,
    group_source_spans,
    transcode_source,
    upload_clips,
)
from cosmos_curator.next.recipes.video_transcode.publication import (
    Workspace,
    cleanup_workspace,
    commit_fragments,
    open_input,
    pending_fragment_ids,
    prepare_fragment,
    resolve_workspace,
    verify_completed,
)
from cosmos_curator.next.utils.arrow_utils import arrow_filesystem, s3_compatibility_env_vars

_SPAN_COLUMNS = ["source_id", "source_uri", "source_size_bytes", "stream_index", "span_id", "start_ns", "end_ns"]
_INPUT_SCHEMA = pa.schema([*(SPAN_SCHEMA.field(name) for name in _SPAN_COLUMNS), *ADDRESS_FIELDS])
_ROW_OFFSET_BITS = 32


def _read_fragment(
    *, uri: str, version: int, fragment_id: int, storage_options: dict[str, str] | None, batch_size: int
) -> Iterator[pa.Table]:
    dataset = lance.dataset(uri, version=version, storage_options=storage_options)
    fragment = dataset.get_fragment(fragment_id)
    if fragment is None:
        msg = f"Selected fragment {fragment_id} is absent at Lance version {version}"
        raise ValueError(msg)
    for batch in fragment.scanner(columns=_SPAN_COLUMNS, with_row_address=True, batch_size=batch_size).to_batches():
        table = pa.Table.from_batches([batch])
        addresses = table["_rowaddr"]
        offsets = pc.bit_wise_and(addresses, pa.scalar((1 << _ROW_OFFSET_BITS) - 1, type=pa.uint64())).cast(pa.int64())
        yield (
            table.select(_SPAN_COLUMNS)
            .append_column("fragment_id", pa.array([fragment_id] * table.num_rows, type=pa.int64()))
            .append_column("row_offset", offsets)
            .cast(_INPUT_SCHEMA)
        )


class SpanDatasource(Datasource):
    """Expose span IDs at Ray's read boundary without serializing live Lance handles."""

    def __init__(self, dataset: lance.LanceDataset, config: ResolvedVideoTranscodeConfig) -> None:
        """Capture serializable metadata for the invocation's selected version."""
        super().__init__()  # type: ignore[no-untyped-call]
        self._uri = config.input.spans_lance_uri
        self._version = int(dataset.version)
        pending = set(pending_fragment_ids(dataset, config))
        self._fragments = [
            (int(fragment.fragment_id), int(fragment.count_rows()))
            for fragment in dataset.get_fragments()
            if fragment.fragment_id in pending
        ]
        self._options = get_lance_storage_options(self._uri, profile_name=config.execution.storage_profile)
        self._batch_size = config.execution.lance_read_batch_size

    def estimate_inmemory_data_size(self) -> int | None:
        """Let Ray estimate Arrow memory independently of compressed Lance sizes."""
        return None

    def get_read_tasks(
        self, parallelism: int, per_task_row_limit: int | None = None, data_context: DataContext | None = None
    ) -> list[ReadTask]:
        """Reopen each selected fragment on workers with the configured credentials."""
        del parallelism, data_context
        return [
            ReadTask(
                partial(
                    _read_fragment,
                    uri=self._uri,
                    version=self._version,
                    fragment_id=fragment_id,
                    storage_options=self._options,
                    batch_size=self._batch_size,
                ),
                BlockMetadata(num_rows=rows, size_bytes=None, input_files=None, exec_stats=None),
                schema=_INPUT_SCHEMA,
                per_task_row_limit=per_task_row_limit,
            )
            for fragment_id, rows in self._fragments
        ]


@contextmanager
def span_checkpoints(config: ResolvedVideoTranscodeConfig, workspace: Workspace) -> Iterator[None]:
    """Scope native checkpoint filtering and writes to the media execution only."""
    context = DataContext.get_current()
    if context.checkpoint_config is not None:
        msg = "video-transcode cannot replace an existing Ray checkpoint configuration"
        raise RuntimeError(msg)
    context.checkpoint_config = CheckpointConfig(
        id_column="span_id",
        checkpoint_path=workspace.checkpoints,
        delete_checkpoint_on_success=False,
        override_filesystem=arrow_filesystem(workspace.root, storage_profile=config.execution.storage_profile),
    )
    try:
        yield
    finally:
        context.checkpoint_config = None


def transcode_spans(dataset: lance.LanceDataset, config: ResolvedVideoTranscodeConfig, workspace: Workspace) -> None:
    """Execute one read-to-Parquet plan with span filtering before source grouping."""
    with span_checkpoints(config, workspace):
        # Sort shuffle uses the read block count to size its partitions.
        spans = ray.data.read_datasource(
            SpanDatasource(dataset, config), override_num_blocks=config.execution.source_partitions
        )
        # Hash-shuffle actors retain CPU/memory reservations until the whole
        # execution ends. Sort tasks release theirs before downloads begin.
        spans.context.shuffle_strategy = ShuffleStrategy.SORT_SHUFFLE_PULL_BASED
        # One in-flight task per actor cannot reach Ray's default 1.75 scaling
        # threshold. Scope this to the media plan, not subsequent publication.
        spans.context.autoscaling_config.actor_pool_util_upscaling_threshold = 1.0
        sources = spans.groupby("source_id").map_groups(
            group_source_spans,
            batch_format="pyarrow",
            num_cpus=0.25,
        )
        # Break size ties so equal-sized sources can occupy separate sort partitions.
        sources = sources.sort(["source_size_bytes", "source_id"], descending=[True, False])
        # Ray Data's task submission budget does not account for curator_io.
        # Actors acquire a slot before accepting work, bounding submission too.
        # Leave the pool uncapped so late Slurm workers can contribute.
        downloaded = sources.map(
            SourceDownloader,
            fn_constructor_kwargs={"config": config},
            compute=ray.data.ActorPoolStrategy(min_size=1, max_tasks_in_flight_per_actor=1),
            num_cpus=0.25,
            max_concurrency=1,
            resources={environment.CURATOR_IO_RESOURCE_NAME: 1.0},
            max_restarts=-1,
            max_task_retries=-1,
        )
        encoded = downloaded.flat_map(
            transcode_source,
            fn_kwargs={"config": config},
            num_cpus=config.execution.transcode_cpus,
            memory=int(config.execution.transcode_memory_gib * 1024**3),
        )
        # Clip production and Ray backpressure pace uploads without download slots.
        uploaded = encoded.map_batches(
            upload_clips,
            batch_format="pyarrow",
            batch_size=None,
            zero_copy_batch=True,
            fn_kwargs={"config": config},
            num_cpus=0.25,
        )
        groups = uploaded.repartition(target_num_rows_per_block=config.execution.spans_per_checkpoint, strict=True)
        groups.write_parquet(
            workspace.results,
            filesystem=arrow_filesystem(workspace.root, storage_profile=config.execution.storage_profile),
            min_rows_per_file=config.execution.spans_per_checkpoint,
            max_rows_per_file=config.execution.spans_per_checkpoint,
        )


def publish_results(
    dataset: lance.LanceDataset, config: ResolvedVideoTranscodeConfig, workspace: Workspace
) -> lance.LanceDataset:
    """Regroup only metadata by original fragments, then publish all descriptors."""
    fragments = pending_fragment_ids(dataset, config)

    def prepare(version: int) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        if not fragments:
            return candidates
        staged = ray.data.read_parquet(
            workspace.results,
            filesystem=arrow_filesystem(workspace.root, storage_profile=config.execution.storage_profile),
        )
        prepared = staged.groupby("fragment_id", num_partitions=len(fragments)).map_groups(
            cast("Any", prepare_fragment),
            batch_format="pyarrow",
            fn_kwargs={"config": config, "version": version},
            num_cpus=1,
        )
        for batch in prepared.iter_batches(batch_format="pyarrow", batch_size=None, prefetch_batches=0):
            candidates.extend(cast("pa.Table", batch).to_pylist())
        return candidates

    return commit_fragments(prepare(int(dataset.version)), config=config, dataset=dataset, reprepare=prepare)


def run_config(config: ResolvedVideoTranscodeConfig) -> dict[str, object]:
    """Open latest, resume durable spans, publish atomically, and clean this workspace."""
    dataset = open_input(config)
    selected = [int(fragment.fragment_id) for fragment in dataset.get_fragments()]
    pending = pending_fragment_ids(dataset, config)
    spans = dataset.count_rows()
    transcoded_spans = sum(
        fragment.count_rows() for fragment in dataset.get_fragments() if fragment.fragment_id in pending
    )
    cleanup_complete: bool | None = None
    if pending or (not selected and completed_publication(dataset) is None):
        workspace = resolve_workspace(config, dataset)
        if pending:
            ensure_ray_initialized(
                local_resources=curator_io_resources(),
                runtime_env_vars=s3_compatibility_env_vars(
                    config.input.media_root,
                    config.input.spans_lance_uri,
                    workspace.root,
                    storage_profile=config.execution.storage_profile,
                ),
            )
            configure_ray_data_progress(progress=config.execution.progress)
            # Suppress reservation-based warnings for whole-source payloads;
            # logical memory reservations and Ray's backpressure remain enabled.
            configure_ray_data_stability(disable_high_memory_detector=True)
            transcode_spans(dataset, config, workspace)
        dataset = publish_results(dataset, config, workspace)
        verify_completed(dataset, config, selected)
        try:
            cleanup_workspace(workspace, config)
        except Exception as exc:  # noqa: BLE001 -- cleanup cannot invalidate committed media
            cleanup_complete = False
            logger.warning("Clips committed at Lance v{}, but workspace cleanup failed: {!r}", dataset.version, exc)
        else:
            cleanup_complete = True
    summary: dict[str, object] = {
        "spans": spans,
        "transcoded_spans": transcoded_spans,
        "selected_fragments": len(selected),
        "pending_fragments": len(pending),
        "clips_lance_uri": config.input.spans_lance_uri,
        "clips_lance_version": int(dataset.version),
        "cleanup_complete": cleanup_complete,
    }
    logger.info("video-transcode: {} clip(s) at Lance v{}", summary["spans"], dataset.version)
    return summary
