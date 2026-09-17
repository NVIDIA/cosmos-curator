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

"""Driver-side schema, state, and narrow reads over the embedding groups of ``clips.lance``.

Everything about the table that is neither model computation nor the physical
column write lives here, as functions over a dataset the CALLER opened. There is
no bound URI and no private opener, so which version a call reads is visible at
the call site: a caller that must see its own commit re-opens and passes the new
handle, rather than trusting shared state to have refreshed itself.

::

    dataset (opened by the caller - its version is the version read)
        |
        +--> SCHEMA   widen one group in; a single metadata commit, and groups
        |             stay independent, so a text-only run never creates the
        |             image or action columns
        |
        +--> READ     the owed-work predicate, filled counts, and a one-column
                      stream

There is no state VALIDATION here, and its absence is a decision rather than a
gap: a group whose stored rows name a different computation than the current one
is owed WORK, not a reason to refuse. A corpus part-way through a rebuild is a
normal transient that an ordinary re-run finishes, so the comparison that used to
refuse now lives inside ``pending_filter`` and selects instead.

The reads are deliberately cheap. The owed-work predicate is only BUILT here; it
is evaluated per fragment, first by the driver's plan and then by the read task
that owns that fragment - one string, two evaluators; the filled count is
resolved inside Lance and returns a bounded result whatever the table's length;
and the one function that streams rows projects a single column.

The physical write and its commit are NOT here; they belong to the publication
path. Splitting them lets schema and selection be tested with no dependency on
the distributed write.

All failures raise a plain, informative ``ValueError`` (the CLI's
``except ValueError -> exit(1)`` handler catches them); no custom exception
hierarchy is introduced.
"""

from collections.abc import Iterator, Mapping, Sequence

import lance
import pyarrow as pa

from cosmos_curator.next.embeddings.schemas import EmbeddingColumnGroup
from cosmos_curator.next.utils.lance_fragment_recovery import sql_string_literal


def _assert_present_group_matches(schema: pa.Schema, group: EmbeddingColumnGroup) -> None:
    """Assert every field of an already-present group matches the group's schema.

    Compares type (which pins the fixed-size-list width) and nullability, so a
    table carrying a differently-shaped column under one of the group's names is
    rejected before any compute runs rather than failing inside a worker's cast.

    Raises:
        ValueError: On any type or nullability mismatch.

    """
    for expected in group.schema:
        stored = schema.field(expected.name)
        if not stored.type.equals(expected.type):
            msg = (
                f"embedding column group {group.name!r}: column {expected.name!r} has type {stored.type} "
                f"but the group requires {expected.type}"
            )
            raise ValueError(msg)
        if stored.nullable != expected.nullable:
            msg = f"embedding column group {group.name!r}: column {expected.name!r} must be nullable"
            raise ValueError(msg)


def _group_presence(schema_names: set[str], group: EmbeddingColumnGroup) -> str:
    """Classify a group as ``absent`` (no fields), ``present`` (all fields), or ``partial``."""
    present = [name for name in group.field_names if name in schema_names]
    if not present:
        return "absent"
    if len(present) == len(group.field_names):
        return "present"
    return "partial"


def ensure_embedding_columns(
    dataset: lance.LanceDataset, groups: Sequence[EmbeddingColumnGroup]
) -> tuple[int, int | None]:
    """Add the missing fields of only the given groups in one metadata commit.

    ``add_columns`` with an all-nullable schema is a METADATA-ONLY operation:
    Lance records the new fields in the manifest and reads them as NULL for
    every existing row. No data file is rewritten and no row is replaced, so
    this cannot create a tombstone.

    Each group must be absent as a whole (all its fields are added) or present
    and exactly matching its schema (nothing added). A partially present group
    is a corrupt schema and raises. Groups not passed are left untouched, so a
    text-only run never creates the image or action columns.

    Args:
        dataset: The clips table to widen.
        groups: The enabled modalities' column groups.

    Returns:
        ``(fields_added, commit_version)``. ``commit_version`` is the widening
        metadata commit's version, captured on the handle immediately after
        ``add_columns``; ``None`` when every group was already present.

    Raises:
        ValueError: If a group is partially present, or a present group does not
            match its schema exactly.

    """
    schema = dataset.schema
    names = set(schema.names)
    fields_to_add: list[pa.Field] = []
    for group in groups:
        presence = _group_presence(names, group)
        if presence == "absent":
            fields_to_add.extend(group.schema)
        elif presence == "present":
            _assert_present_group_matches(schema, group)
        else:
            missing = [name for name in group.field_names if name not in names]
            found = [name for name in group.field_names if name in names]
            msg = (
                f"embedding column group {group.name!r} is partially present on {dataset.uri}: "
                f"has {found}, missing {missing}; a group must be absent as a whole or present and complete"
            )
            raise ValueError(msg)
    if fields_to_add:
        dataset.add_columns(pa.schema(fields_to_add))
        return len(fields_to_add), int(dataset.version)
    return 0, None


def pending_filter(
    group: EmbeddingColumnGroup,
    applicability_filter: str | None,
    result_identity: Mapping[str, str],
) -> str:
    """Return the predicate selecting rows this modality still owes work on.

    The single definition of owed work: an applicable row is owed when its primary
    vector is missing, when any other group field is missing (companion vectors
    are not provenance but still required for a complete row), OR when what it
    already holds came from a different computation than the one this run
    performs. Staleness is therefore SELECTED WORK rather than a refusal - a
    corpus part-way through a rebuild is a normal transient, and an ordinary
    re-run finishes it.

    Args:
        group: The column group whose rows are being selected.
        applicability_filter: Predicate restricting the rows this modality can
            embed at all, or ``None`` when every row is applicable.
        result_identity: ``{provenance_column: value}`` naming this run's
            computation. Every one of ``group.provenance_columns`` must appear;
            ``ModalityFill`` enforces that, and a missing key here would silently
            stop selecting rows that are stale in the omitted column.

    Returns:
        A predicate for Lance's scanner-filter surface (``count_rows(filter=...)``,
        ``scanner(filter=...)``), NOT for ``dataset.sql``.

    Raises:
        ValueError: If ``result_identity`` does not cover every provenance column.

    """
    missing = [column for column in group.provenance_columns if column not in result_identity]
    if missing:
        msg = (
            f"embedding column group {group.name!r}: result_identity is missing {missing}; "
            f"it must cover every provenance column {list(group.provenance_columns)}"
        )
        raise ValueError(msg)
    # Identifiers stay BARE. On Lance's scanner-filter surface a double-quoted
    # name parses as a STRING LITERAL, so `"col" IS NOT NULL` compares two
    # constants and matches every row instead of raising. Values go through the
    # shared escaper. `!=` rather than `<>` is the spelling already proven against
    # this surface in robot_action_caption/table.py.
    #
    # The IS NULL arm is not redundant with `!=`: under SQL three-valued logic
    # `NULL != 'x'` is unknown rather than true, so a row whose identity column is
    # NULL while its vector is present would drop out of the selection entirely.
    owed = [f"{group.primary_vector} IS NULL"]
    for column in group.provenance_columns:
        owed.append(f"{column} IS NULL")
        owed.append(f"{column} != {sql_string_literal(result_identity[column])}")
    for column in group.field_names:
        if column == group.primary_vector or column in group.provenance_columns:
            continue
        owed.append(f"{column} IS NULL")
    disjunction = " OR ".join(owed)
    if applicability_filter:
        return f"({applicability_filter}) AND ({disjunction})"
    return f"({disjunction})"


def count_filled(dataset: lance.LanceDataset, group: EmbeddingColumnGroup) -> int:
    """Return the number of rows whose primary vector is non-NULL."""
    # Zero, not an error, when the group's columns have not been added yet: a
    # caller asking "how much of this group is filled" before the widening commit
    # must read nothing rather than reference a column the SQL planner cannot
    # resolve.
    if group.primary_vector not in dataset.schema.names:
        return 0
    return int(dataset.count_rows(filter=f"{group.primary_vector} IS NOT NULL"))


def scan_column(
    dataset: lance.LanceDataset, column: str, *, row_filter: str | None, batch_size: int
) -> Iterator[pa.Table]:
    """Stream one narrow column of the table, batch by batch.

    The driver-side equivalent of a worker's fragment scan: it projects a single
    column and pushes the predicate into Lance, so a caller that needs a
    corpus-wide view of one field (the action leg's PCA candidate sampling) never
    materializes the table. Yields nothing when the column has no matching rows.

    Args:
        dataset: The clips table to read.
        column: The single column to project.
        row_filter: SQL predicate pushed into the scan, or ``None`` for all rows.
        batch_size: Rows per yielded table.

    """
    scanner = dataset.scanner(columns=[column], filter=row_filter, batch_size=batch_size)
    for record_batch in scanner.to_batches():
        if record_batch.num_rows:
            yield pa.Table.from_batches([record_batch])
