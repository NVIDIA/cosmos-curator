# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Arrow contract for one modality's scanned and staged embedding rows.

Two schemas, one per side of the compute phase, plus the row identity that
carries a Lance row address from the scan all the way into staged Parquet. The
module is deliberately pure - pyarrow, attrs, and this package's schema
declarations only - so the workspace can digest a staged fingerprint without
importing Ray.

::

    clips.lance --> scan_schema --> [ embed ] --> staged_schema --> Parquet
                     + row identity                                 partitioned
                                                                    by fragment_id

Both schemas are PER MODALITY, unlike robot_action_caption's single fixed staged
schema, because each modality stages its own group's columns and nothing else.
The fingerprint is therefore per modality too, which is what stops an image
attempt resuming from a text attempt's staged files.
"""

from typing import Final

import pyarrow as pa
import pyarrow.compute as pc

from cosmos_curator.next.embeddings.schemas import EMBED_SOURCE_ROW, KEY_COLUMN, EmbeddingColumnGroup
from cosmos_curator.next.utils.identity import canonical_digest
from cosmos_curator.next.utils.lance_utils import ROW_ADDRESS_COLUMN, ROW_ADDRESS_FRAGMENT_SHIFT

FRAGMENT_ID_COLUMN: Final = "fragment_id"

CHECKPOINT_ID_COLUMN: Final = "__checkpoint_id"
"""Column Ray's row checkpoint keys on. Defined HERE, beside the scan that fills it.

The checkpoint filter and the scan are two halves of one mechanism: the column
must be on the Read output for resume filtering to be possible, and Ray reads it
back off the written output to mark a row complete. Dropping it would checkpoint
nothing while the run appeared to work, so the string has exactly one definition
- in the module that produces its values.
"""

STAGING_PROTOCOL_VERSION: Final = 1
# Bumping this invalidates every staged workspace, so the trigger set is exhaustive
# rather than illustrative. Bump when changing: the meaning of __checkpoint_id or
# its derivation from _rowaddr; whether checkpoints survive a phase that succeeded;
# the `results` / `checkpoints` segment names; the Hive partition scheme; the file
# format; the write mode; or any assumption made about Ray's file-sink transaction
# behaviour. NOT triggers, because they change only how fast a workspace fills and
# never how a later run reads it: batch sizes, actor counts, parquet_rows_per_file.

_IDENTITY_FIELDS: Final = (
    pa.field(FRAGMENT_ID_COLUMN, pa.int64(), nullable=False),
    pa.field(CHECKPOINT_ID_COLUMN, pa.uint64(), nullable=False),
)


def _staged_field(field: pa.Field) -> pa.Field:
    """Return a group field in the shape Parquet can round-trip with NULL rows.

    A ``fixed_size_list`` column holding any NULL is UNREADABLE once written to
    Parquet (measured, pyarrow 25.0.1: ``Expected all lists to be of size=N but
    index i had size=0``, on every read path, with an explicit schema supplied and
    without). A per-row compute failure stages exactly that - a NULL vector beside
    its surviving siblings - so the width is dropped here and re-asserted by the
    cast at publication, which is also where a wrong-width row is caught.
    """
    if pa.types.is_fixed_size_list(field.type):
        return pa.field(field.name, pa.list_(field.type.value_type), nullable=True)
    return field


def scan_schema(source_columns: tuple[str, ...]) -> pa.Schema:
    """Return the schema the datasource emits: the named source columns plus row identity.

    Args:
        source_columns: Columns this modality reads, each named in
            ``EMBED_SOURCE_ROW``, in projection order.

    Returns:
        Those columns with their declared source types, then ``fragment_id`` and
        ``__checkpoint_id``.

    Raises:
        ValueError: If a name is not an ``EMBED_SOURCE_ROW`` column.

    """
    unknown = [column for column in source_columns if column not in EMBED_SOURCE_ROW.names]
    if unknown:
        msg = f"scan columns {unknown} are not embedding source columns {list(EMBED_SOURCE_ROW.names)}"
        raise ValueError(msg)
    return pa.schema([*(EMBED_SOURCE_ROW.field(column) for column in source_columns), *_IDENTITY_FIELDS])


def staged_schema(group: EmbeddingColumnGroup) -> pa.Schema:
    """Return the schema one modality's staged Parquet rows carry.

    The key and the row identity travel with every staged row: the key is what
    publication joins on, and the checkpoint id is what lets a later attempt skip
    a row already staged.
    """
    return pa.schema(
        [
            EMBED_SOURCE_ROW.field(KEY_COLUMN),
            *_IDENTITY_FIELDS,
            *(_staged_field(field) for field in group.schema),
        ]
    )


def staged_columns(group: EmbeddingColumnGroup) -> tuple[str, ...]:
    """Return the columns publication reads back out of a staged file.

    ``fragment_id`` is deliberately absent: it becomes a Hive partition key at
    write time, so it is read from the directory name rather than from inside the
    file. ``__checkpoint_id`` is absent because publication joins on the key.
    """
    return (KEY_COLUMN, *group.field_names)


def staged_schema_fingerprint(group: EmbeddingColumnGroup) -> str:
    """Hash one modality's ordered staged field contract for recovery identity."""
    return canonical_digest(
        [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
            }
            for field in staged_schema(group)
        ]
    )


def attach_row_identity(batch: pa.RecordBatch, schema: pa.Schema) -> pa.Table:
    """Split a scanned batch's row address into the fragment id and the checkpoint id.

    Columnar throughout. robot_action_caption rebuilds its rows through
    ``pa.Table.from_pylist`` because it renders a prompt per row; embeddings does
    no per-row work at this boundary, so the rows must never be materialized as
    Python objects here.

    Args:
        batch: One scanned batch carrying ``_rowaddr`` plus the projected columns.
        schema: The :func:`scan_schema` to emit.

    Returns:
        The batch's rows, in the order scanned, under ``schema``.

    Raises:
        ValueError: If the batch carries no ``_rowaddr``, which means the scanner
            was built without ``with_row_address=True``.

    """
    table = pa.Table.from_batches([batch])
    if ROW_ADDRESS_COLUMN not in table.schema.names:
        msg = f"scanned batch carries no {ROW_ADDRESS_COLUMN!r} column; the scanner must set with_row_address=True"
        raise ValueError(msg)
    row_address = table.column(ROW_ADDRESS_COLUMN)
    # The high bits of the address, so the staged partition key is derived from the
    # same read as the id rather than joined back to it later.
    fragment_id = pc.shift_right(row_address, ROW_ADDRESS_FRAGMENT_SHIFT).cast(pa.int64())
    identified = (
        table.drop_columns([ROW_ADDRESS_COLUMN])
        .append_column(_IDENTITY_FIELDS[0], fragment_id)
        .append_column(_IDENTITY_FIELDS[1], row_address)
    )
    return identified.select(schema.names).cast(schema)
