# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the rows one modality still owes from a pinned clips snapshot.

The plan is made once on the driver from inert per-fragment metadata; each read
task then reopens the table itself and scans only the fragments it owns. The
``uint64`` Lance row address must reach the Read output for Ray's row checkpoint
to have anything to key on, so every batch leaves here carrying it.

::

    plan_embedding_scan --> EmbeddingScanDatasource --> read_owed_rows
    (driver, per fragment    (splits fragments into      (worker, reopens
     filtered counts)         read tasks)                 at the pinned version)

Nothing here knows what a modality computes: the owed-row predicate arrives
already built (``columns.pending_filter``) and the emitted schema is whatever
``staged.scan_schema`` says the projection is.
"""

from collections.abc import Iterator
from functools import partial

import attrs
import lance
import pyarrow as pa
from ray.data.block import BlockMetadata
from ray.data.context import DataContext
from ray.data.datasource import Datasource, ReadTask

from cosmos_curator.next.recipes.embeddings.staged import attach_row_identity, scan_schema


def _copy_storage_options(value: dict[str, str] | None) -> dict[str, str] | None:
    """Return a caller-independent copy of Lance storage options."""
    return dict(value) if value is not None else None


@attrs.frozen
class FragmentRead:
    """Inert per-fragment plan data, safe to pickle into a Ray read task.

    Attributes:
        fragment_id: Fragment to scan at the pinned version.
        num_rows: Owed rows it holds. Ray reads it as block metadata, and
            publication reads it as the exact row count this fragment's staged
            partition must yield, so it is a contract and not an estimate.
        input_files: Data file paths, for Ray's locality hints.

    """

    fragment_id: int
    num_rows: int
    input_files: tuple[str, ...]


@attrs.frozen
class EmbeddingScanPlan:
    """What this attempt owes one modality, discovered in ONE pass over the manifest.

    Planning and counting are one step because they read the same per-fragment
    filtered count. Split in two, the driver walks every fragment twice and the
    two walks can disagree - the caller would then decide there was work from one
    count and size its actor pool from the other.

    Attributes:
        fragments: Fragments holding at least one owed row, in manifest order,
            after any cap. Empty means the modality owes nothing.

    """

    fragments: tuple[FragmentRead, ...]

    @property
    def num_rows(self) -> int:
        """Return the owed row count before the checkpoint filter runs."""
        return sum(fragment.num_rows for fragment in self.fragments)

    @property
    def fragment_ids(self) -> tuple[int, ...]:
        """Return the ids of the owed fragments, in manifest order."""
        return tuple(fragment.fragment_id for fragment in self.fragments)


def plan_embedding_scan(
    dataset: lance.LanceDataset, *, row_filter: str, max_fragments: int | None
) -> EmbeddingScanPlan:
    """Find the fragments this attempt owes work in, at the pinned version.

    Args:
        dataset: The clips table, open at the run's pinned version.
        row_filter: Predicate defining what this attempt owes.
        max_fragments: Optional cap on how many owed fragments to take on. It
            bounds this walk as well as the plan, so a capped run does not pay
            the uncapped one's corpus-wide count.

    Returns:
        The plan. A fragment holding no owed row is excluded, so a resume plans
        no read task for it.

    """
    owed: list[FragmentRead] = []
    for fragment in dataset.get_fragments():
        # Stop AT the cap rather than counting the rest and slicing to it. Both
        # keep this same manifest-order prefix, so the plan is identical either
        # way; the difference is that the discarded counts are reads, and not
        # taking them is what makes a capped run cost its own footprint rather
        # than the whole corpus. Tested BEFORE the count and not after the
        # append, so that a cap of zero plans nothing - after the append it
        # could never match, having skipped from zero owed straight to one.
        if max_fragments is not None and len(owed) >= max_fragments:
            break
        # A filtered count READS the predicate's columns; only an unfiltered one
        # answers from fragment metadata. Measured at pylance 11.0.0 with the
        # table's ``data/`` moved aside: ``count_rows()`` still returns, while
        # ``count_rows(filter=...)`` raises ``OSError``. So every turn of this
        # loop is a projected driver-side read of the vector and identity
        # columns, single-threaded; an uncapped run pays it corpus-wide once per
        # modality, and a run that turns out to owe nothing pays it in full.
        num_rows = int(fragment.count_rows(filter=row_filter))
        if not num_rows:
            continue
        owed.append(
            FragmentRead(
                fragment_id=int(fragment.fragment_id),
                num_rows=num_rows,
                input_files=tuple(
                    str(data_file.path)
                    for data_file in fragment.data_files()  # type: ignore[no-untyped-call]  # Missing in Lance stubs.
                ),
            )
        )
    # Capped AFTER dropping the fully-current fragments, never before. Capping the
    # raw manifest prefix would hand back the same already-current fragments on
    # every attempt, so a capped run would make no progress at all.
    return EmbeddingScanPlan(fragments=tuple(owed))


@attrs.frozen
class PinnedScan:
    """Everything a read task needs to reproduce this run's scan on its own worker.

    One object rather than six arguments because these values travel together
    from the driver into every read task and are meaningless apart: the version
    pins the snapshot, the row filter selects rows within it, and the source
    columns decide the projection AND the emitted schema. Splitting them across a
    signature is how one task comes to scan a different subset than the contract
    describes.

    Attributes:
        uri: Clips table URI.
        read_version: The run's pinned Lance version.
        storage_options: Lance credentials copied at construction and re-applied
            on each worker's re-open.
        source_columns: The modality's projection, in scan order.
        row_filter: Predicate selecting the rows this attempt owes.
        batch_size: Rows per scanned Arrow batch. Must be at least 1; the attrs
            field validator enforces this on every construction path.

    """

    uri: str
    read_version: int
    storage_options: dict[str, str] | None = attrs.field(converter=_copy_storage_options)
    source_columns: tuple[str, ...]
    row_filter: str
    batch_size: int = attrs.field(validator=attrs.validators.ge(1))

    @classmethod
    def for_dataset(
        cls,
        dataset: lance.LanceDataset,
        *,
        storage_options: dict[str, str] | None,
        source_columns: tuple[str, ...],
        row_filter: str,
        batch_size: int,
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
            source_columns=source_columns,
            row_filter=row_filter,
            batch_size=batch_size,
        )

    def projection(self) -> list[str]:
        """Return the columns to scan, which are exactly the modality's source columns."""
        return list(self.source_columns)


class EmbeddingScanDatasource(Datasource):
    """Scan owed rows from named fragments of one pinned Lance snapshot.

    Plans from inert metadata and has each worker re-open the dataset itself.
    A live ``LanceFragment`` cannot be the payload: its ``__reduce__`` reopens
    the dataset WITHOUT storage options, so pickling a cloud fragment loses its
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
        schema = scan_schema(self._scan.source_columns)
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
                        read_owed_rows,
                        self._scan,
                        tuple(fragment.fragment_id for fragment in group),
                    ),
                    metadata,
                    schema=schema,
                    per_task_row_limit=per_task_row_limit,
                )
            )
        return tasks


def read_owed_rows(scan: PinnedScan, fragment_ids: tuple[int, ...]) -> Iterator[pa.Table]:
    """Stream one read task's owed rows as :func:`scan_schema` tables.

    Args:
        scan: The pinned snapshot and row scope this task reproduces.
        fragment_ids: Fragments this task owns.

    Yields:
        Non-empty tables of owed rows, each carrying its row identity.

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
    schema = scan_schema(scan.source_columns)
    scanner = dataset.scanner(
        columns=scan.projection(),
        filter=scan.row_filter,
        fragments=fragments,
        with_row_address=True,
        batch_size=scan.batch_size,
    )
    for batch in scanner.to_reader():
        if batch.num_rows:
            yield attach_row_identity(batch, schema)
