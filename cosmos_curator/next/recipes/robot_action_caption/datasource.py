# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scan owed clips with prompt text and a durable per-row checkpoint id.

The checkpoint id must exist on the Read output and survive to staged Parquet.
Prompt rendering also happens at this boundary so optional context fields
collapse into one fixed downstream schema.

Planning narrows the fragment universe in a fixed order, and the order is
load-bearing rather than incidental::

    fragment universe   every fragment the pinned manifest holds
          |
          v
    shard ownership     this job's scope, decided on the manifest id alone
          |
          v
    pending count       one filtered count per OWNED fragment
          |
          v
    read tasks          one per owed fragment

Ownership precedes the count for two reasons, and only these two: a job pays
driver cost for its own scope rather than the whole table, and every owed figure
it then reports is scoped to what it owns, so a sharded run's diagnostics do not
send an operator after a backlog another job already holds. Both are properties
of the ORDER - the plan itself would be identical either way.

See ``docs/curator/design/curator-next-robot-action-caption.md`` for checkpoint
identity, index-memory, partitioning, and prompt-ownership rationale, and
``curator-next-sharding.md`` for the shared ownership mechanism.
"""

from collections.abc import Iterator
from functools import partial
from typing import Any, Final

import attrs
import lance
import pyarrow as pa
from ray.data.block import BlockMetadata
from ray.data.context import DataContext
from ray.data.datasource import Datasource, ReadTask

from cosmos_curator.next.core.sharding import ShardConfig
from cosmos_curator.next.recipes.robot_action_caption.prompt import PromptContextField, render_prompt
from cosmos_curator.next.recipes.robot_action_caption.table import (
    CLIP_ID_COLUMN,
    CLIP_URI_COLUMN,
    REQUIRED_SOURCE_COLUMNS,
    require_clip_id,
)
from cosmos_curator.next.utils.lance_utils import ROW_ADDRESS_COLUMN, ROW_ADDRESS_FRAGMENT_SHIFT

FRAGMENT_ID_COLUMN: Final = "fragment_id"
PROMPT_COLUMN: Final = "prompt_text"

CHECKPOINT_ID_COLUMN: Final = "__checkpoint_id"
"""Column Ray's row checkpoint keys on. Defined HERE, beside the scan that fills it.

The value is the row's Lance address, unique per row within the pinned snapshot
by construction - so recovery relies on no uniqueness contract of its own.
Keying on ``clip_id`` alone would not do. Publication scopes staging, coverage,
and ``update_columns`` by fragment, so two fragments holding one clip id would
see the resume filter mark the second complete and no attempt would ever caption
it, a gap with no error.

Stability is scoped to a FRAGMENT REWRITE, not to a table version. Measured at
pylance 10.0.0 and 11.0.0, every commit this recipe issues leaves an address
untouched - the metadata-only widening in ``table.ensure_caption_fields`` and the
per-fragment ``update_columns`` in ``publication.prepare_fragment``, a sibling
shard's included - which is why the recovery namespace need not separate runs by
read version. Only compaction moves an address, and that is excluded from
recovery for every candidate key: compaction never reuses a fragment id, so a
``(fragment_id, clip_id)`` pair moves with it while costing roughly nine times
this id's width in the driver's filter index.

The checkpoint filter and this scan are two halves of one mechanism: the Read
output placement makes resume filtering possible, while the current Ray sink
checks the terminal output and fails loudly if the column was dropped. The
string therefore has exactly one definition, in the module that produces its
values. A generic home would make it look like an independent contract that a
leg merely opts into.
"""

# One scanned address serves as both the checkpoint id and the source of the
# staged partition key. Taking the fragment id from the same read as the row is
# what makes the two unable to disagree; a join back to the table could resolve
# differently. The packing itself is a property of the Lance format rather than
# of this table, so both facts come from ``lance_utils``.


SCAN_SCHEMA: Final = pa.schema(
    [
        pa.field(CLIP_ID_COLUMN, pa.string(), nullable=False),
        pa.field(FRAGMENT_ID_COLUMN, pa.int64(), nullable=False),
        # Nullable, unlike its siblings, because it is the one field whose value
        # is copied straight from the source table rather than derived here, and
        # ``pa.Table.from_pylist`` does not validate nullability - so declaring
        # it non-null would read as a pinned contract that nothing enforces. A
        # null reaches the media stage, which refuses that row alone; see
        # ``media.ClipMediaFetch``.
        pa.field(CLIP_URI_COLUMN, pa.large_string(), nullable=True),
        pa.field(PROMPT_COLUMN, pa.large_string(), nullable=False),
        pa.field(CHECKPOINT_ID_COLUMN, pa.uint64(), nullable=False),
    ]
)


def _copy_storage_options(value: dict[str, str] | None) -> dict[str, str] | None:
    """Return a caller-independent copy of Lance storage options."""
    return dict(value) if value is not None else None


@attrs.frozen
class FragmentRead:
    """Inert per-fragment plan data, safe to pickle into a Ray read task.

    Attributes:
        fragment_id: Fragment to scan at the pinned version.
        num_rows: Owed rows it holds, for Ray's block metadata only.
        input_files: Data file paths, for Ray's locality hints.

    """

    fragment_id: int
    num_rows: int
    input_files: tuple[str, ...]


@attrs.frozen
class CaptionScanPlan:
    """What this attempt owes, discovered in ONE pass over the manifest.

    Planning and counting are one step because they read the same per-fragment
    filtered count. Split in two, the driver walked every fragment twice and the
    two walks could disagree - the caller would then decide there was work from
    one count and size the engine pool from the other.

    Both halves of the scope are recorded for the same reason. Every count a
    caller reports must be relative to what this job owns, and deriving the
    owned set a second time would put the ownership decision in two places,
    free to disagree.

    Attributes:
        fragments: Fragments holding at least one owed row, in manifest order.
            A subset of ``owned_fragment_ids``. Empty means THIS JOB owes
            nothing, which on a sharded run says nothing about the table.
        owned_fragment_ids: Every fragment in this job's execution scope,
            whether or not it owes a row, in manifest order. The whole manifest
            on an unsharded run.

    """

    fragments: tuple[FragmentRead, ...]
    owned_fragment_ids: tuple[int, ...]

    @property
    def num_rows(self) -> int:
        """Return the owed row count before the checkpoint filter runs."""
        return sum(fragment.num_rows for fragment in self.fragments)

    @property
    def fragment_ids(self) -> tuple[int, ...]:
        """Return the ids of the owed fragments, in manifest order."""
        return tuple(fragment.fragment_id for fragment in self.fragments)

    @property
    def owned_fragments(self) -> int:
        """Return how many fragments this job's scope covers."""
        return len(self.owned_fragment_ids)


def plan_caption_scan(
    dataset: lance.LanceDataset,
    row_filter: str,
    *,
    shard: ShardConfig | None,
) -> CaptionScanPlan:
    """Find the fragments this run owes captions for, at the pinned version.

    Args:
        dataset: The clip table, open at the run's pinned version.
        row_filter: Predicate defining what this attempt owes.
        shard: This job's execution scope over the fragment universe, or ``None``
            when one job owns every fragment.

    Returns:
        The plan. A fragment with no row matching the attempt's combined scope
        and pending predicate is excluded, so a resume plans no read task for it.

    """
    owned_fragment_ids = []
    fragments = []
    for fragment in dataset.get_fragments():
        fragment_id = int(fragment.fragment_id)
        # Ownership BEFORE the count, which is what keeps a job's driver cost
        # proportional to its own scope rather than to the whole table: eight
        # jobs each pay one eighth of the filtered counts, not all of them. It
        # is also what makes every owed figure below scoped to this job.
        #
        # The manifest id, never the owed count: a partition decided on how much
        # work remains would move under a later job as rows get captioned, so a
        # re-launch of this index would own a different set than it owns now.
        if shard is not None and not shard.owns(fragment_id):
            continue
        owned_fragment_ids.append(fragment_id)
        # One filtered count per fragment on one column. O(fragments) on the
        # driver, never O(rows): the count is pushed into the fragment's own
        # statistics rather than materializing the rows here.
        num_rows = int(fragment.count_rows(filter=row_filter))
        if not num_rows:
            continue
        fragments.append(
            FragmentRead(
                fragment_id=fragment_id,
                num_rows=num_rows,
                input_files=tuple(
                    str(data_file.path)
                    for data_file in fragment.data_files()  # type: ignore[no-untyped-call]  # Missing in Lance stubs.
                ),
            )
        )
    return CaptionScanPlan(
        fragments=tuple(fragments),
        owned_fragment_ids=tuple(owned_fragment_ids),
    )


@attrs.frozen
class PinnedScan:
    """Everything a read task needs to reproduce this run's scan on its own worker.

    One object rather than seven arguments because these values travel together
    from the driver into every read task and are meaningless apart: the version
    pins the snapshot, the row filter selects rows within it, and the
    context fields decide the projection AND the prompt. Splitting them across a
    signature is how one task comes to scan a different subset than the contract
    describes.

    Attributes:
        uri: Clip table URI.
        read_version: The run's pinned Lance version.
        storage_options: Lance credentials copied at construction and re-applied
            on each worker's re-open.
        batch_size: Rows per scanned Arrow batch. Must be at least 1; the attrs
            field validator enforces this on every construction path.
        context_fields: The context subset resolved for this run, in render order.
        row_filter: Predicate selecting the rows this attempt owes.

    """

    uri: str
    read_version: int
    storage_options: dict[str, str] | None = attrs.field(converter=_copy_storage_options)
    batch_size: int = attrs.field(validator=attrs.validators.ge(1))
    context_fields: tuple[PromptContextField, ...]
    row_filter: str

    @classmethod
    def for_dataset(  # Snapshot, row scope, and read settings are independently meaningful.
        cls,
        dataset: lance.LanceDataset,
        *,
        storage_options: dict[str, str] | None,
        batch_size: int,
        context_fields: tuple[PromptContextField, ...],
        row_filter: str,
    ) -> "PinnedScan":
        """Bind a scan to the version an already-open handle sits at.

        Raises:
            ValueError: If ``batch_size`` is less than 1, propagated from the
                :class:`PinnedScan` field validator.

        """
        return cls(
            uri=dataset.uri,
            read_version=int(dataset.version),
            storage_options=storage_options,
            batch_size=batch_size,
            context_fields=context_fields,
            row_filter=row_filter,
        )

    def projection(self) -> list[str]:
        """Return the columns to scan: the structural pair plus resolved context.

        Only the RESOLVED context columns are projected. Projecting the full
        supported set instead would fail the scan on any table that has not
        landed all of them yet, which is the normal case upstream.
        """
        return [*REQUIRED_SOURCE_COLUMNS, *(field.column for field in self.context_fields)]


class CaptionScanDatasource(Datasource):
    """Scan owed clips from named fragments of one pinned Lance snapshot.

    Plans from inert metadata and has each worker re-open the dataset itself.
    A live ``LanceFragment`` cannot be the payload: its ``__reduce__`` reopens
    the dataset WITHOUT storage options, so pickling an S3 fragment loses its
    credentials before the read task even starts.
    """

    def __init__(self, scan: PinnedScan, fragments: tuple[FragmentRead, ...]) -> None:
        """Store the pinned scan and the per-fragment plan.

        Args:
            scan: The snapshot, contract, and projection every read task shares.
            fragments: Fragments to scan, with their planning metadata.

        """
        super().__init__()  # type: ignore[no-untyped-call]  # Ray's public Datasource constructor lacks typing.
        self._scan = scan
        self._fragments = fragments

    def estimate_inmemory_data_size(self) -> int | None:
        """Let Ray pick memory targets; a Lance file size is not an Arrow size."""

    def get_read_tasks(
        self,
        parallelism: int,
        per_task_row_limit: int | None = None,
        data_context: DataContext | None = None,
    ) -> list[ReadTask]:
        """Split the fragments across at most ``parallelism`` read tasks.

        Args:
            parallelism: Ray's requested task count.
            per_task_row_limit: Ray's optional per-task row cap.
            data_context: Unused.

        Returns:
            One task per fragment group. Empty when no fragment is owed.

        Raises:
            ValueError: If ``parallelism`` is not positive.

        """
        del data_context
        if parallelism < 1:
            msg = f"read parallelism must be positive, got {parallelism}"
            raise ValueError(msg)
        if not self._fragments:
            return []
        task_count = min(parallelism, len(self._fragments))
        tasks = []
        for index in range(task_count):
            group = self._fragments[
                index * len(self._fragments) // task_count : (index + 1) * len(self._fragments) // task_count
            ]
            metadata = BlockMetadata(
                num_rows=sum(fragment.num_rows for fragment in group),
                size_bytes=None,
                input_files=tuple(path for fragment in group for path in fragment.input_files),
                exec_stats=None,
            )
            tasks.append(
                ReadTask(
                    partial(
                        read_pending_clips,
                        self._scan,
                        tuple(fragment.fragment_id for fragment in group),
                    ),
                    metadata,
                    schema=SCAN_SCHEMA,
                    per_task_row_limit=per_task_row_limit,
                )
            )
        return tasks


def read_pending_clips(scan: PinnedScan, fragment_ids: tuple[int, ...]) -> Iterator[pa.Table]:
    """Stream one read task's owed clips as :data:`SCAN_SCHEMA` tables.

    Args:
        scan: The pinned snapshot and row scope this task reproduces.
        fragment_ids: Fragments this task owns.

    Yields:
        Non-empty tables of owed clip rows.

    Raises:
        ValueError: If a fragment is absent at the pinned version.

    """
    dataset = lance.dataset(scan.uri, version=scan.read_version, storage_options=scan.storage_options)
    fragments = []
    for fragment_id in fragment_ids:
        fragment = dataset.get_fragment(fragment_id)
        if fragment is None:
            msg = f"fragment {fragment_id} is not present in {scan.uri} at v{scan.read_version}"
            raise ValueError(msg)
        fragments.append(fragment)
    scanner = dataset.scanner(
        columns=scan.projection(),
        filter=scan.row_filter,
        fragments=fragments,
        with_row_address=True,
        batch_size=scan.batch_size,
    )
    for batch in scanner.to_reader():
        if batch.num_rows:
            yield scanned_rows(pa.Table.from_batches([batch]), scan.context_fields)


def scanned_rows(scanned: pa.Table, context_fields: tuple[PromptContextField, ...]) -> pa.Table:
    """Project one scanned Lance batch onto :data:`SCAN_SCHEMA`.

    Args:
        scanned: A batch carrying ``_rowaddr`` plus :data:`REQUIRED_SOURCE_COLUMNS`,
            and the resolved context columns.
        context_fields: The context subset to render, in render order.

    Returns:
        The batch's rows with the prompt rendered, and the row address carried
        through as the checkpoint id with its fragment half split out.

    Raises:
        ValueError: If a required column, or a resolved context column, is
            missing from the scan, or if a row has an invalid ``clip_id``.

    """
    # The resolved context columns are required HERE even though they are optional
    # on the source table: resolution already decided this snapshot supplies them,
    # so one absent now means the projection and the render disagree, and the
    # prompt would silently lose a line the digest says it carries.
    required = {ROW_ADDRESS_COLUMN, *REQUIRED_SOURCE_COLUMNS, *(field.column for field in context_fields)}
    missing = sorted(required - set(scanned.schema.names))
    if missing:
        msg = f"caption scan is missing required column(s): {', '.join(missing)}"
        raise ValueError(msg)
    rows = [_scanned_row(row, context_fields) for row in scanned.select(sorted(required)).to_pylist()]
    return pa.Table.from_pylist(rows, schema=SCAN_SCHEMA)


def _scanned_row(row: dict[str, Any], context_fields: tuple[PromptContextField, ...]) -> dict[str, Any]:
    """Project one scanned row onto :data:`SCAN_SCHEMA`."""
    clip_id = require_clip_id(row[CLIP_ID_COLUMN])
    row_address = int(row[ROW_ADDRESS_COLUMN])
    return {
        CLIP_ID_COLUMN: clip_id,
        # The address's high bits, so the staged partition key and the checkpoint
        # id come from the same read rather than from a join back to the table.
        FRAGMENT_ID_COLUMN: row_address >> ROW_ADDRESS_FRAGMENT_SHIFT,
        # Carried through as it arrived, null included: refusing a null here
        # would fail the read task and so the whole attempt, whereas the
        # media stage can refuse the one row that names no location.
        CLIP_URI_COLUMN: row[CLIP_URI_COLUMN],
        PROMPT_COLUMN: render_prompt(row, context_fields),
        CHECKPOINT_ID_COLUMN: row_address,
    }
