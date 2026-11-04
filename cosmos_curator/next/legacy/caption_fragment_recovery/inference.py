# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase A: checkpoint-filtered caption inference staged as durable Parquet.

Reads an attempt's pending fragments, runs video-caption's shared caption plan
on the projected rows, and writes terminal rows inside one Ray checkpoint keyed
on the recipe's row ID.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import lance
import pyarrow as pa
import ray
from loguru import logger
from ray.data.block import BlockMetadata
from ray.data.context import DataContext
from ray.data.datasource import Datasource, ReadTask

from cosmos_curator.next.core.ray_checkpoints import installed_checkpoint
from cosmos_curator.next.legacy.caption_fragment_recovery.bindings import CaptionRunBindings
from cosmos_curator.next.legacy.caption_fragment_recovery.lance_state import CaptionAttempt
from cosmos_curator.next.legacy.caption_fragment_recovery.workspace import (
    CaptionWorkspace,
    record_phase_a_completion,
)
from cosmos_curator.next.recipes.video_caption.inference import CaptionRequest, caption_rows


@dataclass(frozen=True)
class CaptionExecutionConfig:
    """Caption plan settings plus Phase A read and staging sizes."""

    storage_profile: str
    inference_concurrency: int | Literal["auto"]
    inference_batch_size: int
    max_concurrent_batches: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    media_concurrency: int | Literal["auto"]
    media_batch_size: int
    media_cpus: float
    media_attempts: int
    lance_read_batch_size: int
    parquet_rows_per_file: int
    max_model_len: int | None = None


@dataclass(frozen=True)
class _CaptionFragmentRead:
    """Picklable driver-side metadata for one pinned Lance fragment."""

    fragment_id: int
    num_rows: int
    input_files: tuple[str, ...]
    schema: pa.Schema


class _CaptionLanceDatasource(Datasource):
    """Read selected fragments without putting live ``LanceFragment`` objects in Ray tasks.

    ``LanceFragment.__reduce__`` reopens its dataset without storage options. Ray's
    built-in Lance datasource captures explicitly supplied fragments in each read
    task, so pickling an S3 fragment can lose its configured credentials before the
    task even starts. This datasource plans from inert metadata and has each worker
    reopen the pinned dataset with the original storage options and fragment IDs.
    """

    def __init__(  # noqa: PLR0913 -- serialized datasource connection inputs are independently meaningful
        self,
        *,
        uri: str,
        version: int,
        fragments: tuple[_CaptionFragmentRead, ...],
        storage_options: dict[str, str] | None,
        batch_size: int,
        source_columns: tuple[str, ...],
        skip_null_column: str | None = None,
    ) -> None:
        super().__init__()  # type: ignore[no-untyped-call]  # Ray's public Datasource constructor lacks typing.
        self._uri = uri
        self._version = version
        self._fragments = fragments
        self._storage_options = dict(storage_options) if storage_options is not None else None
        self._batch_size = batch_size
        self._source_columns = source_columns
        self._skip_null_column = skip_null_column

    @property
    def num_rows(self) -> int:
        """Return the pre-checkpoint-filter work ceiling without scanning rows."""
        return sum(fragment.num_rows for fragment in self._fragments)

    def estimate_inmemory_data_size(self) -> int | None:
        """Let Ray choose memory targets because Lance file size is not an Arrow size estimate."""
        return None

    def get_read_tasks(
        self,
        parallelism: int,
        per_task_row_limit: int | None = None,
        data_context: DataContext | None = None,
    ) -> list[ReadTask]:
        """Build tasks containing only fragment IDs and serializable connection data."""
        del data_context
        if parallelism < 1:
            msg = f"Caption Lance read parallelism must be positive, got {parallelism}"
            raise ValueError(msg)
        task_count = min(parallelism, len(self._fragments))
        tasks = []
        for task_index in range(task_count):
            start = task_index * len(self._fragments) // task_count
            stop = (task_index + 1) * len(self._fragments) // task_count
            fragments = self._fragments[start:stop]
            fragment_ids = tuple(fragment.fragment_id for fragment in fragments)
            metadata = BlockMetadata(
                num_rows=None if self._skip_null_column else sum(fragment.num_rows for fragment in fragments),
                size_bytes=None,
                input_files=tuple(path for fragment in fragments for path in fragment.input_files),
                exec_stats=None,
            )
            tasks.append(
                ReadTask(
                    partial(
                        _read_caption_fragments,
                        uri=self._uri,
                        version=self._version,
                        fragment_ids=fragment_ids,
                        storage_options=self._storage_options,
                        batch_size=self._batch_size,
                        source_columns=self._source_columns,
                        skip_null_column=self._skip_null_column,
                    ),
                    metadata,
                    schema=fragments[0].schema,
                    per_task_row_limit=per_task_row_limit,
                )
            )
        return tasks


def caption_lance_datasource(  # noqa: PLR0913 -- datasource identity and schema inputs are independently meaningful
    dataset: lance.LanceDataset,
    attempt: CaptionAttempt,
    *,
    storage_options: dict[str, str] | None,
    batch_size: int,
    source_columns: tuple[str, ...],
    source_schema: pa.Schema,
    skip_null_column: str | None = None,
) -> _CaptionLanceDatasource:
    """Snapshot selected-fragment metadata while keeping live handles on the driver."""
    fragments = []
    for fragment_id in attempt.pending_fragment_ids:
        fragment = dataset.get_fragment(fragment_id)
        if fragment is None:
            msg = f"Pending fragment {fragment_id} disappeared from attempt version {attempt.version}"
            raise ValueError(msg)
        fragments.append(
            _CaptionFragmentRead(
                fragment_id=fragment_id,
                num_rows=int(fragment.count_rows()),
                input_files=tuple(
                    str(data_file.path)
                    for data_file in fragment.data_files()  # type: ignore[no-untyped-call]  # Missing in Lance stubs.
                ),
                schema=source_schema,
            )
        )
    return _CaptionLanceDatasource(
        uri=dataset.uri,
        version=attempt.version,
        fragments=tuple(fragments),
        storage_options=storage_options,
        batch_size=batch_size,
        source_columns=source_columns,
        skip_null_column=skip_null_column,
    )


def _read_caption_fragments(  # noqa: PLR0913 -- serialized read-task inputs are independently meaningful
    *,
    uri: str,
    version: int,
    fragment_ids: tuple[int, ...],
    storage_options: dict[str, str] | None,
    batch_size: int,
    source_columns: tuple[str, ...],
    skip_null_column: str | None = None,
) -> Iterator[pa.Table]:
    """Reopen a pinned Lance snapshot with credentials and stream selected fragments."""
    dataset = lance.dataset(uri, version=version, storage_options=storage_options)
    fragments = []
    for fragment_id in fragment_ids:
        fragment = dataset.get_fragment(fragment_id)
        if fragment is None:
            msg = f"Pending fragment {fragment_id} disappeared from attempt version {version}"
            raise ValueError(msg)
        fragments.append(fragment)
    scanner = dataset.scanner(
        columns=list(source_columns),
        fragments=fragments,
        with_row_address=True,
        batch_size=batch_size,
    )
    for batch in scanner.to_reader():
        table = pa.Table.from_batches([batch])
        if skip_null_column is not None:
            table = table.filter(table[skip_null_column].is_valid())
        if table.num_rows:
            yield table


def caption_request(bindings: CaptionRunBindings) -> CaptionRequest:
    """Describe the bindings' media, request, and terminal rows to the shared caption plan."""
    return CaptionRequest(
        media=bindings.input.media,
        identity_columns=bindings.input.staged_binding_columns,
        prompt_template=bindings.prompt_template,
        build_terminal_row=bindings.result.build_terminal_row,
        result_schema=bindings.result.staged_schema,
        inference=bindings.inference,
        context_stage=bindings.context_stage,
        carried_result_columns=bindings.result.carried_result_columns,
    )


def run_inference_phase(  # noqa: PLR0913 -- explicit phase boundary inputs are independently meaningful
    dataset: lance.LanceDataset,
    attempt: CaptionAttempt,
    config: CaptionExecutionConfig,
    bindings: CaptionRunBindings,
    model_dir: Path,
    workspace: CaptionWorkspace,
    *,
    storage_options: dict[str, str] | None,
) -> None:
    """Execute one checkpointed read-to-Parquet plan for the attempt's pending fragments."""
    if not attempt.pending_fragment_ids:
        return
    caption_source = caption_lance_datasource(
        dataset,
        attempt,
        storage_options=storage_options,
        batch_size=config.lance_read_batch_size,
        source_columns=bindings.input.source_columns,
        source_schema=bindings.input.source_schema,
        skip_null_column=bindings.input.row_id_column if bindings.input.skip_null_ids else None,
    )

    with installed_checkpoint(
        id_column=bindings.input.row_id_column,
        checkpoint_path=workspace.checkpoints_uri,
        filesystem=workspace.filesystem,
    ):
        source = ray.data.read_datasource(
            caption_source,
            override_num_blocks=len(attempt.pending_fragment_ids),
        )
        mapped = source.map_batches(
            bindings.input.project_batch,
            batch_format="pyarrow",
            batch_size=None,
            zero_copy_batch=True,
        )
        terminal = caption_rows(
            mapped,
            caption_request(bindings),
            config,
            model_dir,
            max_rows=caption_source.num_rows,
        )
        checkpoint_groups = terminal.repartition(
            target_num_rows_per_block=config.parquet_rows_per_file,
            strict=True,
        )
        checkpoint_groups.write_parquet(
            workspace.results_uri,
            filesystem=workspace.filesystem,
            max_rows_per_file=config.parquet_rows_per_file,
        )
    try:
        record_phase_a_completion(workspace, attempt, bindings.result.digest)
    except Exception as exc:  # noqa: BLE001 -- this marker is only a restart optimization
        logger.warning(
            "Phase A results are durable, but writing completion marker {} failed; a restart will let Ray reload "
            "checkpoints instead: {!r}",
            workspace.phase_a_completion_uri,
            exc,
        )
    else:
        logger.info(
            "Recorded Phase A completion for {} pending fragment(s) at {}",
            len(attempt.pending_fragment_ids),
            workspace.phase_a_completion_uri,
        )
