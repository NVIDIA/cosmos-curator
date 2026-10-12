# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read parallelism is independent of Lance fragments and preserves physical addresses."""

from pathlib import Path
from typing import ClassVar

import lance
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import ray
from ray.data._internal.execution.execution_callback import ExecutionCallback
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.streaming_executor import StreamingExecutor
from ray.data.context import ShuffleStrategy

from cosmos_curator.next.core import lance_datasource
from cosmos_curator.next.core.lance_datasource import LanceFragmentDatasource
from cosmos_curator.next.core.ray_checkpoints import installed_checkpoint
from cosmos_curator.next.enrichment.workspace import RecoveryWorkspace
from cosmos_curator.next.utils.arrow_utils import arrow_filesystem

_SCHEMA = pa.schema(
    [
        pa.field("item_id", pa.int64(), nullable=False),
        pa.field("fragment_id", pa.int64(), nullable=False),
        pa.field("row_offset", pa.int64(), nullable=False),
    ]
)


@pytest.fixture
def dataset(tmp_path: Path) -> lance.LanceDataset:
    """Keep enough rows to satisfy 256 tasks across only six physical fragments."""
    return lance.write_dataset(
        pa.table({"item_id": range(600), "unused": ["payload"] * 600}),
        tmp_path / "input.lance",
        max_rows_per_file=100,
        data_storage_version="2.2",
    )


def _source(dataset: lance.LanceDataset, fragment_ids: list[int] | None = None) -> LanceFragmentDatasource:
    return LanceFragmentDatasource(
        dataset,
        fragment_ids=[int(f.fragment_id) for f in dataset.get_fragments()] if fragment_ids is None else fragment_ids,
        schema=_SCHEMA,
        storage_profile="default",
        batch_size=7,
    )


@pytest.mark.parametrize("parallelism", [1, 4, 6, 7, 256, 1000])
def test_balanced_tasks_cover_rows_and_addresses_once(dataset: lance.LanceDataset, parallelism: int) -> None:
    """Split large fragments, combine small ranges, and avoid empty tasks."""
    tasks = _source(dataset).get_read_tasks(parallelism)
    assert len(tasks) == min(parallelism, 600)
    tables = [pa.concat_tables(list(task())) for task in tasks]
    counts = [table.num_rows for table in tables]
    assert max(counts) - min(counts) <= 1
    assert counts == [task.metadata.num_rows for task in tasks]
    combined = pa.concat_tables(tables)
    assert combined.schema.equals(_SCHEMA, check_metadata=True)
    assert combined.to_pylist() == [
        {"item_id": index, "fragment_id": index // 100, "row_offset": index % 100} for index in range(600)
    ]


def test_selected_fragments_keep_physical_offsets_after_deletions(dataset: lance.LanceDataset) -> None:
    """Offsets come from Lance row addresses, never the new task's local row numbering."""
    dataset.delete("item_id IN (101, 104, 402, 499)")
    tasks = _source(dataset, [4, 1]).get_read_tasks(7)
    rows = [row for task in tasks for batch in task() for row in batch.to_pylist()]
    assert rows == [
        {"item_id": index, "fragment_id": index // 100, "row_offset": index % 100}
        for index in [*range(100, 200), *range(400, 500)]
        if index not in (101, 104, 402, 499)
    ]
    counts = [task.metadata.num_rows for task in tasks]
    assert counts == [28] * 7


def test_read_tasks_reopen_the_pinned_version(dataset: lance.LanceDataset) -> None:
    """Changes made after planning cannot move a read range or change its contents."""
    source = _source(dataset)
    tasks = source.get_read_tasks(7)
    dataset.delete("item_id < 150")
    rows = [row for task in tasks for batch in task() for row in batch.to_pylist()]
    assert [row["item_id"] for row in rows] == list(range(600))
    assert dataset.count_rows() == 450


def test_row_limit_applies_across_fragment_ranges(dataset: lance.LanceDataset) -> None:
    """A single task's limit covers its combined ranges, including a partial final fragment."""
    tasks = _source(dataset).get_read_tasks(1, per_task_row_limit=150)
    assert len(tasks) == 1
    rows = [row for batch in tasks[0]() for row in batch.to_pylist()]
    assert [row["item_id"] for row in rows] == list(range(150))
    assert rows[-1] == {"item_id": 149, "fragment_id": 1, "row_offset": 49}


def test_empty_selection_creates_no_tasks(dataset: lance.LanceDataset) -> None:
    """No selected work should produce neither read tasks nor a division by zero."""
    assert _source(dataset, []).get_read_tasks(256) == []


def test_missing_fragment_is_rejected(dataset: lance.LanceDataset) -> None:
    """An invalid selection cannot silently omit required rows."""
    with pytest.raises(ValueError, match="Selected Lance fragments are absent"):
        _source(dataset, [6])


def test_read_schema_requires_row_addresses(dataset: lance.LanceDataset) -> None:
    """Without both addresses, staged results could not be aligned with their fragments."""
    with pytest.raises(ValueError, match="requires non-null int64 fragment_id and row_offset"):
        LanceFragmentDatasource(
            dataset,
            fragment_ids=[0],
            schema=pa.schema([_SCHEMA.field("item_id"), _SCHEMA.field("fragment_id")]),
            storage_profile="default",
            batch_size=7,
        )


def test_workers_resolve_credentials_from_the_profile(
    dataset: lance.LanceDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read tasks carry only the profile name and resolve credentials when they run."""
    resolved = []

    def storage_options(uri: str, *, profile_name: str) -> None:
        resolved.append((uri, profile_name))

    monkeypatch.setattr(lance_datasource, "get_lance_storage_options", storage_options)
    source = LanceFragmentDatasource(
        dataset, fragment_ids=[0], schema=_SCHEMA, storage_profile="worker-profile", batch_size=7
    )
    (task,) = source.get_read_tasks(1)
    assert resolved == []
    assert pa.concat_tables(list(task())).num_rows == 100
    assert resolved == [(dataset.uri, "worker-profile")]


@pytest.mark.parametrize("parallelism", [0, -1])
def test_invalid_parallelism_is_rejected(dataset: lance.LanceDataset, parallelism: int) -> None:
    """Ray must resolve automatic parallelism before asking the datasource for tasks."""
    with pytest.raises(ValueError, match="Read parallelism must be positive"):
        _source(dataset).get_read_tasks(parallelism)


class _ObserveReadPartitions(ExecutionCallback):
    """Fail before a regressed read/filter plan can create a quadratic shuffle."""

    observations: ClassVar[list[tuple[bool, int]]] = []

    def before_execution_starts(self, executor: StreamingExecutor) -> None:
        """Check the actual native checkpoint plan without triggering a large sort first."""
        for op in executor._topology or ():
            if isinstance(op, MapOperator) and op.name.startswith(("ReadLanceFragment", "CheckpointFilter")):
                assert op.get_additional_split_factor() == 1, op.name

    def after_execution_succeeds(self, executor: StreamingExecutor) -> None:
        """Record read output counts in both the initial and resumed plans."""
        operators = list(executor._topology or ())
        for op in operators:
            if op.name.startswith("ReadLanceFragment"):
                self.observations.append(
                    (
                        any(other.name.startswith("CheckpointFilter") for other in operators),
                        op.metrics.num_task_outputs_generated,
                    )
                )


@pytest.mark.integration
def test_checkpoint_resume_keeps_requested_read_partition_count(
    dataset: lance.LanceDataset, tmp_path: Path, curator_io_cluster: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Six fragments and 256 partitions stay bounded with Ray's native checkpoint filter."""
    del curator_io_cluster
    monkeypatch.setattr(_ObserveReadPartitions, "observations", [])
    workspace = RecoveryWorkspace(str(tmp_path / "recovery"), {"test": "read-partitioning"})
    workspace.ensure(storage_profile="default")
    filesystem = arrow_filesystem(workspace.root, storage_profile="default")
    for fragment_ids in ([0, 1, 2], list(range(6))):
        with installed_checkpoint(id_column="item_id", checkpoint_path=workspace.checkpoints, filesystem=filesystem):
            rows = ray.data.read_datasource(_source(dataset, fragment_ids), override_num_blocks=256)
            rows.context.enable_progress_bars = False
            rows.context.shuffle_strategy = ShuffleStrategy.SORT_SHUFFLE_PULL_BASED
            rows.context.custom_execution_callback_classes.append(_ObserveReadPartitions)
            rows.sort("item_id").write_parquet(
                workspace.results, filesystem=filesystem, min_rows_per_file=50, max_rows_per_file=50
            )
    assert _ObserveReadPartitions.observations == [(False, 256), (True, 256)]
    results = pq.read_table(workspace.results).sort_by("item_id")
    assert results.to_pylist() == [
        {"item_id": index, "fragment_id": index // 100, "row_offset": index % 100} for index in range(600)
    ]
