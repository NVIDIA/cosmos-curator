# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpointed clip inference followed by one atomic Lance enrichment commit."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any, cast

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import ray
from loguru import logger
from ray.data.block import BlockMetadata
from ray.data.checkpoint import CheckpointConfig
from ray.data.context import DataContext
from ray.data.datasource import Datasource, ReadTask

from cosmos_curator.core.utils import environment
from cosmos_curator.core.utils.pixi_runtime_envs import ray_data_gpu_runtime_env
from cosmos_curator.next.core.ray_runtime import (
    configure_ray_data_progress,
    configure_ray_data_stability,
    curator_io_resources,
    ensure_ray_initialized,
)
from cosmos_curator.next.recipes.video_embed.config import Concurrency, ResolvedVideoEmbedConfig
from cosmos_curator.next.recipes.video_embed.contracts import ModelSpec, resolve_model_spec, validate_model_directory
from cosmos_curator.next.recipes.video_embed.processing import (
    INPUT_SCHEMA,
    VideoEmbedder,
    download_clip,
)
from cosmos_curator.next.recipes.video_embed.publication import commit_fragments, prepare_fragment, verify_completed
from cosmos_curator.next.recipes.video_embed.state import (
    Workspace,
    ensure_workspace,
    open_input,
    pending_fragment_ids,
    resolve_workspace,
    storage_options,
)
from cosmos_curator.next.utils.arrow_utils import arrow_filesystem, filesystem_path, s3_compatibility_env_vars
from cosmos_curator.next.utils.storage import download_bytes, remove_prefix, write_if_absent

_SOURCE_COLUMNS = ["clip_id", "clip_uri", "clip_size_bytes"]
_ROW_OFFSET_BITS = 32


def _read_fragment(*, config: ResolvedVideoEmbedConfig, version: int, fragment_id: int) -> Iterator[pa.Table]:
    dataset = lance.dataset(config.input.clips_lance_uri, version=version, storage_options=storage_options(config))
    fragment = dataset.get_fragment(fragment_id)
    if fragment is None:
        msg = f"Selected clip fragment {fragment_id} is absent at version {version}"
        raise ValueError(msg)
    for batch in fragment.scanner(
        columns=_SOURCE_COLUMNS, with_row_address=True, batch_size=config.execution.lance_read_batch_size
    ).to_batches():
        table = pa.Table.from_batches([batch])
        offsets = pc.bit_wise_and(table["_rowaddr"], pa.scalar((1 << _ROW_OFFSET_BITS) - 1, type=pa.uint64()))
        yield (
            table.select(_SOURCE_COLUMNS)
            .append_column("fragment_id", pa.array([fragment_id] * table.num_rows, type=pa.int64()))
            .append_column("row_offset", offsets.cast(pa.int64()))
            .select(INPUT_SCHEMA.names)
            .cast(INPUT_SCHEMA)
        )


class ClipDatasource(Datasource):
    """Expose stable clip IDs at Ray's checkpoint filter, before downloading media."""

    def __init__(self, dataset: lance.LanceDataset, config: ResolvedVideoEmbedConfig, pending: tuple[int, ...]) -> None:
        """Capture serializable metadata, keeping live Lance handles on the driver."""
        super().__init__()  # type: ignore[no-untyped-call]
        self._config = config
        self._version = int(dataset.version)
        selected = set(pending)
        self._fragments = [
            (int(fragment.fragment_id), int(fragment.count_rows()))
            for fragment in dataset.get_fragments()
            if fragment.fragment_id in selected
        ]

    def estimate_inmemory_data_size(self) -> int | None:
        """Let Ray estimate Arrow memory independently of Lance compression."""
        return None

    def get_read_tasks(
        self, parallelism: int, per_task_row_limit: int | None = None, data_context: DataContext | None = None
    ) -> list[ReadTask]:
        """Reopen the pinned input on workers under its runtime credentials."""
        del parallelism, data_context
        return [
            ReadTask(
                partial(_read_fragment, config=self._config, version=self._version, fragment_id=fragment_id),
                BlockMetadata(num_rows=rows, size_bytes=None, input_files=None, exec_stats=None),
                schema=INPUT_SCHEMA,
                per_task_row_limit=per_task_row_limit,
            )
            for fragment_id, rows in self._fragments
        ]


@contextmanager
def embedding_checkpoints(config: ResolvedVideoEmbedConfig, workspace: Workspace) -> Iterator[None]:
    """Limit native checkpoint filtering/writing to the read-through-Parquet plan."""
    context = DataContext.get_current()
    if context.checkpoint_config is not None:
        msg = "video-embed cannot replace an existing Ray checkpoint configuration"
        raise RuntimeError(msg)
    context.checkpoint_config = CheckpointConfig(
        id_column="clip_id",
        checkpoint_path=workspace.checkpoints,
        delete_checkpoint_on_success=False,
        override_filesystem=arrow_filesystem(workspace.root, storage_profile=config.execution.storage_profile),
    )
    try:
        yield
    finally:
        context.checkpoint_config = None


def _actor_pool(concurrency: Concurrency) -> ray.data.ActorPoolStrategy:
    # Ray's default task prefetching lets serial model actors reach its
    # autoscaling threshold without lowering that threshold for the whole plan.
    return ray.data.ActorPoolStrategy(
        min_size=1,
        max_size=None if concurrency == "auto" else concurrency,
    )


def embed_clips(
    dataset: lance.LanceDataset,
    config: ResolvedVideoEmbedConfig,
    spec: ModelSpec,
    workspace: Workspace,
    *,
    pending: tuple[int, ...],
) -> None:
    """Stream downloaded clips through GPU preprocessing and inference to durable results."""
    execution = config.execution
    model_dir = str(validate_model_directory(spec))
    with embedding_checkpoints(config, workspace):
        clips = ray.data.read_datasource(ClipDatasource(dataset, config, pending))
        # Keep loaded actors while their pool has work in flight between download bursts.
        clips.context.autoscaling_config.actor_pool_util_downscaling_threshold = 0.0
        downloaded = clips.map(
            download_clip,
            fn_kwargs={"storage_profile": execution.storage_profile, "max_attempts": execution.media_attempts},
            num_cpus=0.25,
            # Backpressured tasks retain encoded buffers on their worker heap.
            resources={environment.CURATOR_IO_RESOURCE_NAME: 1.0},
            max_retries=-1,
        )
        inferred = downloaded.map_batches(
            VideoEmbedder,
            fn_constructor_kwargs={"spec": spec, "model_dir": model_dir},
            batch_format="pyarrow",
            batch_size=execution.inference_batch_size,
            compute=_actor_pool(execution.inference_concurrency),
            num_cpus=1,
            num_gpus=1,
            runtime_env=ray_data_gpu_runtime_env("legacy-transformers"),
            max_concurrency=1,
            max_restarts=-1,
            max_task_retries=-1,
        )
        groups = inferred.repartition(target_num_rows_per_block=execution.parquet_rows_per_file, strict=True)
        groups.write_parquet(
            workspace.results,
            filesystem=arrow_filesystem(workspace.root, storage_profile=execution.storage_profile),
            max_rows_per_file=execution.parquet_rows_per_file,
        )
    # This marker only avoids model setup on a publication-only retry. The
    # publisher still proves exact coverage and validates every staged value.
    try:
        write_if_absent(
            workspace.completion,
            json.dumps({"scope": workspace.scope}).encode(),
            storage_profile=execution.storage_profile,
        )
    except Exception as exc:  # noqa: BLE001 -- the marker is only a recovery optimization
        logger.warning("Results are checkpointed, but the inference completion marker could not be written: {!r}", exc)


def publish_results(
    dataset: lance.LanceDataset,
    config: ResolvedVideoEmbedConfig,
    spec: ModelSpec,
    workspace: Workspace,
    *,
    pending: tuple[int, ...],
) -> lance.LanceDataset:
    """Group compact results by fragment, then publish and verify the Lance snapshot."""

    def prepare(version: int) -> list[dict[str, Any]]:
        staged = ray.data.read_parquet(
            workspace.results,
            filesystem=arrow_filesystem(workspace.root, storage_profile=config.execution.storage_profile),
        )
        prepared = staged.groupby("fragment_id", num_partitions=len(pending)).map_groups(
            cast("Any", prepare_fragment),
            batch_format="pyarrow",
            fn_kwargs={"config": config, "spec": spec, "version": version},
            num_cpus=1,
        )
        candidates = []
        for batch in prepared.iter_batches(batch_format="pyarrow", batch_size=None, prefetch_batches=0):
            candidates.extend(cast("pa.Table", batch).to_pylist())
        return candidates

    return commit_fragments(
        prepare(int(dataset.version)),
        config=config,
        dataset=dataset,
        spec=spec,
        workspace=workspace,
        pending=pending,
        reprepare=prepare,
    )


def _inference_complete(workspace: Workspace, config: ResolvedVideoEmbedConfig) -> bool:
    filesystem = arrow_filesystem(workspace.root, storage_profile=config.execution.storage_profile)
    if filesystem.get_file_info(filesystem_path(workspace.completion)).type != pafs.FileType.File:
        return False
    return (
        download_bytes(workspace.completion, storage_profile=config.execution.storage_profile)
        == json.dumps({"scope": workspace.scope}).encode()
    )


def run_config(config: ResolvedVideoEmbedConfig) -> dict[str, object]:
    """Resume compatible GPU work, publish once, and retain failed attempts for recovery."""
    spec = resolve_model_spec(config.model.variant)
    dataset = open_input(config, spec)
    original = dataset
    pending = pending_fragment_ids(dataset, spec)
    workspace = resolve_workspace(config, dataset, spec)
    filesystem = arrow_filesystem(workspace.root, storage_profile=config.execution.storage_profile)
    manifest_exists = (
        filesystem.get_file_info(filesystem_path(f"{workspace.root}/workspace.json")).type == pafs.FileType.File
    )
    if pending or manifest_exists:
        ensure_workspace(workspace, config)
    if pending:
        ensure_ray_initialized(
            local_resources=curator_io_resources(),
            runtime_env_vars=s3_compatibility_env_vars(
                config.input.media_root,
                config.input.clips_lance_uri,
                workspace.root,
                storage_profile=config.execution.storage_profile,
            ),
        )
        configure_ray_data_progress(progress=config.execution.progress)
        configure_ray_data_stability(disable_high_memory_detector=True)
        if not _inference_complete(workspace, config):
            embed_clips(dataset, config, spec, workspace, pending=pending)
        dataset = publish_results(dataset, config, spec, workspace, pending=pending)
    else:
        verify_completed(dataset, original, spec)
    cleanup_complete = True
    try:
        remove_prefix(workspace.root, storage_profile=config.execution.storage_profile)
    except Exception as exc:  # noqa: BLE001 -- cleanup cannot invalidate canonical vectors
        cleanup_complete = False
        logger.warning("Embeddings committed at Lance v{}, but workspace cleanup failed: {!r}", dataset.version, exc)
    summary: dict[str, object] = {
        "clips": original.count_rows(),
        "pending_fragments": len(pending),
        "clips_lance_uri": config.input.clips_lance_uri,
        "clips_lance_version": int(dataset.version),
        "embedding_field": spec.vector_field,
        "contract_digest": spec.digest,
        "cleanup_complete": cleanup_complete,
    }
    logger.info("video-embed: {}", json.dumps(summary, sort_keys=True))
    return summary
