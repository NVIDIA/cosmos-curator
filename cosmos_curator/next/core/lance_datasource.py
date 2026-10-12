# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read pinned Lance fragments at Ray's native checkpoint-filter boundary."""

from collections.abc import Iterator, Sequence
from functools import partial

import attrs
import lance
import pyarrow as pa
import pyarrow.compute as pc
from ray.data.block import BlockMetadata
from ray.data.context import DataContext
from ray.data.datasource import Datasource, ReadTask

from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options

# Original physical row addresses, carried from the read through staged results.
ROW_ADDRESS_FIELDS = (
    pa.field("fragment_id", pa.int64(), nullable=False),
    pa.field("row_offset", pa.int64(), nullable=False),
)
_ROW_OFFSET_BITS = 32


@attrs.frozen
class _FragmentRange:
    fragment_id: int
    offset: int
    num_rows: int


def _read_fragments(  # noqa: PLR0913 -- serializable worker inputs
    *,
    uri: str,
    version: int,
    ranges: tuple[_FragmentRange, ...],
    storage_profile: str,
    schema: pa.Schema,
    batch_size: int,
) -> Iterator[pa.Table]:
    # Resolve credentials on the worker so long runs pick up refreshed profiles.
    storage_options = get_lance_storage_options(uri, profile_name=storage_profile)
    dataset = lance.dataset(uri, version=version, storage_options=storage_options)
    columns = [name for name in schema.names if name not in {field.name for field in ROW_ADDRESS_FIELDS}]
    for selected in ranges:
        fragment = dataset.get_fragment(selected.fragment_id)
        if fragment is None:
            msg = f"Selected fragment {selected.fragment_id} is absent at Lance version {version}"
            raise ValueError(msg)
        scanner = fragment.scanner(
            columns=columns,
            with_row_address=True,
            batch_size=batch_size,
            offset=selected.offset,
            limit=selected.num_rows,
        )
        for batch in scanner.to_batches():
            table = pa.Table.from_batches([batch])
            offsets = pc.bit_wise_and(table["_rowaddr"], pa.scalar((1 << _ROW_OFFSET_BITS) - 1, type=pa.uint64()))
            yield (
                table.select(columns)
                .append_column(
                    ROW_ADDRESS_FIELDS[0], pa.array([selected.fragment_id] * table.num_rows, type=pa.int64())
                )
                .append_column(ROW_ADDRESS_FIELDS[1], offsets.cast(pa.int64()))
                .select(schema.names)
                .cast(schema)
            )


class LanceFragmentDatasource(Datasource):
    """Capture serializable metadata, retaining live Lance handles on the driver.

    ``schema`` lists the projected input columns plus ``ROW_ADDRESS_FIELDS``.
    Workers reopen the pinned version under ``storage_profile``.
    """

    def __init__(
        self,
        dataset: lance.LanceDataset,
        *,
        fragment_ids: Sequence[int],
        schema: pa.Schema,
        storage_profile: str,
        batch_size: int,
    ) -> None:
        """Pin the input version and selected physical fragments before execution."""
        super().__init__()  # type: ignore[no-untyped-call]
        if any(
            field.name not in schema.names or not field.equals(schema.field(field.name), check_metadata=True)
            for field in ROW_ADDRESS_FIELDS
        ):
            msg = "The read schema requires non-null int64 fragment_id and row_offset addresses"
            raise ValueError(msg)
        self._uri = dataset.uri
        self._version = int(dataset.version)
        selected = set(fragment_ids)
        self._fragments = [
            (int(fragment.fragment_id), int(fragment.count_rows()))
            for fragment in dataset.get_fragments()
            if fragment.fragment_id in selected
        ]
        if {fragment_id for fragment_id, _ in self._fragments} != selected:
            msg = "Selected Lance fragments are absent from the pinned input"
            raise ValueError(msg)
        self._schema = schema
        self._storage_profile = storage_profile
        self._batch_size = batch_size

    def estimate_inmemory_data_size(self) -> int | None:
        """Let Ray estimate Arrow memory independently of compressed Lance sizes."""
        return None

    def get_read_tasks(
        self, parallelism: int, per_task_row_limit: int | None = None, data_context: DataContext | None = None
    ) -> list[ReadTask]:
        """Balance rows across the requested tasks, independently of physical fragments.

        Ray may split outputs when a datasource supplies too few read tasks.
        Its checkpoint planner can apply that split again to the filter, so
        create the requested parallelism here instead of relying on splitting.
        """
        del data_context
        if parallelism < 1:
            msg = "Read parallelism must be positive"
            raise ValueError(msg)
        total_rows = sum(rows for _, rows in self._fragments)
        if total_rows == 0:
            return []
        num_tasks = min(parallelism, total_rows)
        rows_per_task, extra_rows = divmod(total_rows, num_tasks)
        tasks: list[ReadTask] = []
        ranges: list[_FragmentRange] = []
        task_rows = 0
        for fragment_id, rows in self._fragments:
            offset = 0
            while offset < rows:
                target_rows = rows_per_task + (len(tasks) < extra_rows)
                count = min(target_rows - task_rows, rows - offset)
                ranges.append(_FragmentRange(fragment_id, offset, count))
                offset += count
                task_rows += count
                if task_rows == target_rows:
                    tasks.append(
                        ReadTask(
                            partial(
                                _read_fragments,
                                uri=self._uri,
                                version=self._version,
                                ranges=tuple(ranges),
                                storage_profile=self._storage_profile,
                                schema=self._schema,
                                batch_size=self._batch_size,
                            ),
                            BlockMetadata(num_rows=task_rows, size_bytes=None, input_files=None, exec_stats=None),
                            schema=self._schema,
                            per_task_row_limit=per_task_row_limit,
                        )
                    )
                    ranges = []
                    task_rows = 0
        return tasks
