# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for the staged Arrow contract and the row identity it carries.

The module under test is pure pyarrow, so nothing here writes a table or starts
Ray. Two concerns:

- FINGERPRINT: which changes to a modality's staged contract a recovery
  namespace must notice, and which it deliberately does not;
- ROW IDENTITY: that a Lance row address is split into the documented halves
  without reordering the batch.

Field NULLABILITY is absent from the fingerprint cases on purpose: every staged
group field comes from an ``EmbeddingColumnGroup``, whose own validator rejects a
non-nullable field, and the remaining fields are fixed by ``staged_schema``. No
caller can vary it, so a test that appeared to cover it would have to fabricate a
group the production type refuses to construct.
"""

import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cosmos_curator.next.embeddings.schemas import (
    IMAGE_COLUMN_GROUP,
    IMAGE_DIM,
    TEXT_COLUMN_GROUP,
    EmbeddingColumnGroup,
)
from cosmos_curator.next.recipes.embeddings.staged import (
    CHECKPOINT_ID_COLUMN,
    FRAGMENT_ID_COLUMN,
    attach_row_identity,
    scan_schema,
    staged_columns,
    staged_schema,
    staged_schema_fingerprint,
)
from cosmos_curator.next.utils.lance_utils import ROW_ADDRESS_COLUMN

_FRAGMENT = 3


def _image_group(fields: list[pa.Field], *, provenance: str = "embedding_image_model_id") -> EmbeddingColumnGroup:
    """Build an image-shaped group from ``fields`` so one contract detail can be varied at a time."""
    return EmbeddingColumnGroup(
        name="image",
        schema=pa.schema(fields),
        primary_vector="embedding_image",
        provenance_columns=(provenance,),
    )


# The packing Lance uses for a row address: (fragment_id << 32) | offset. Spelled
# out here rather than imported from the module under test, and deliberately not
# taken from the shared ``row_address`` helper, because two tests below assert the
# SPLIT: an input packed with the same constant the code unpacks with recovers the
# fragment id whatever that constant holds, so those assertions would be satisfied
# by the constant instead of by the behaviour. Measured by mutating the production
# shift to 16: with the shared constant both tests stay green, with this literal
# they fail.
_ROW_ADDRESS_FRAGMENT_SHIFT = 32


def _row_address(fragment_id: int, offset: int) -> int:
    """Pack a Lance row address the way the format documents."""
    return (fragment_id << _ROW_ADDRESS_FRAGMENT_SHIFT) | offset


def test_the_fingerprint_is_stable_across_calls() -> None:
    """Two reads of one unchanged contract agree, so a resume is not refused at random."""
    assert staged_schema_fingerprint(IMAGE_COLUMN_GROUP) == staged_schema_fingerprint(IMAGE_COLUMN_GROUP)


def test_two_modalities_have_different_fingerprints() -> None:
    """Each modality stages its own columns, so one cannot resume from another's files."""
    assert staged_schema_fingerprint(IMAGE_COLUMN_GROUP) != staged_schema_fingerprint(TEXT_COLUMN_GROUP)


def test_renaming_a_staged_field_changes_the_fingerprint() -> None:
    """A renamed column makes already-staged files unreadable under the new contract."""
    renamed = _image_group(
        [
            pa.field("embedding_image", pa.list_(pa.float32(), IMAGE_DIM)),
            pa.field("embedding_image_producer", pa.string()),
        ],
        provenance="embedding_image_producer",
    )
    assert staged_schema_fingerprint(renamed) != staged_schema_fingerprint(IMAGE_COLUMN_GROUP)


def test_changing_a_staged_field_type_changes_the_fingerprint() -> None:
    """A retyped column would be read back as the wrong type, so the namespace must differ."""
    retyped = _image_group(
        [
            pa.field("embedding_image", pa.list_(pa.float32(), IMAGE_DIM)),
            pa.field("embedding_image_model_id", pa.large_string()),
        ]
    )
    assert staged_schema_fingerprint(retyped) != staged_schema_fingerprint(IMAGE_COLUMN_GROUP)


def test_reordering_staged_fields_changes_the_fingerprint() -> None:
    """Order is part of the contract because the digest is taken over an ordered list.

    Arrow itself tolerates a reorder, so nothing downstream would fail; the
    fingerprint is the only thing that can notice, and it should, because a
    reader written against one order reads the other's columns by position.
    """
    reordered = _image_group(
        [
            pa.field("embedding_image_model_id", pa.string()),
            pa.field("embedding_image", pa.list_(pa.float32(), IMAGE_DIM)),
        ]
    )
    assert staged_schema_fingerprint(reordered) != staged_schema_fingerprint(IMAGE_COLUMN_GROUP)


def test_a_vector_width_change_does_not_change_the_fingerprint() -> None:
    """The staged vector is width-free, so the width is not what the namespace separates on.

    Pinning the consequence of staging ``list<float32>`` rather than
    ``fixed_size_list<float32, N>`` - forced because a fixed-size column holding
    any NULL cannot be read back out of Parquet. The width is still enforced,
    twice and elsewhere: the cast at publication refuses a wrong-width row, and
    the workspace namespace digests the run's result identity, which for
    text/image changes with the model and for action names ``action_dim``
    outright. Removing the widening should redden this test and send the reader
    back to that reasoning.
    """
    narrower = _image_group(
        [
            pa.field("embedding_image", pa.list_(pa.float32(), IMAGE_DIM // 2)),
            pa.field("embedding_image_model_id", pa.string()),
        ]
    )
    assert staged_schema_fingerprint(narrower) == staged_schema_fingerprint(IMAGE_COLUMN_GROUP)


def test_the_publication_projection_omits_the_partition_key() -> None:
    """``fragment_id`` lives in the directory name, not in the file, so reading it back would fail."""
    columns = staged_columns(IMAGE_COLUMN_GROUP)
    assert columns == ("clip_id", *IMAGE_COLUMN_GROUP.field_names)
    assert FRAGMENT_ID_COLUMN not in columns


def test_the_row_address_splits_into_the_documented_halves() -> None:
    """The fragment id is the address's high bits and the checkpoint id is the whole address."""
    schema = scan_schema(("clip_id",))
    batch = pa.RecordBatch.from_pydict(
        {
            "clip_id": ["a"],
            ROW_ADDRESS_COLUMN: pa.array([_row_address(_FRAGMENT, 7)], pa.uint64()),
        }
    )

    identified = attach_row_identity(batch, schema)

    assert identified.column(FRAGMENT_ID_COLUMN).to_pylist() == [_FRAGMENT]
    assert identified.column(CHECKPOINT_ID_COLUMN).to_pylist() == [_row_address(_FRAGMENT, 7)]


def test_row_order_survives_the_split() -> None:
    """Rows keep the order they were scanned in, so the id attached to a row stays its own.

    The addresses descend while the ids ascend, so a sort introduced anywhere in
    the columnar rewrite pairs every clip with the wrong address.
    """
    schema = scan_schema(("clip_id",))
    batch = pa.RecordBatch.from_pydict(
        {
            "clip_id": ["a", "b", "c"],
            ROW_ADDRESS_COLUMN: pa.array(
                [_row_address(_FRAGMENT, 9), _row_address(_FRAGMENT, 5), _row_address(1, 2)], pa.uint64()
            ),
        }
    )

    identified = attach_row_identity(batch, schema)

    assert identified.column("clip_id").to_pylist() == ["a", "b", "c"]
    assert identified.column(FRAGMENT_ID_COLUMN).to_pylist() == [_FRAGMENT, _FRAGMENT, 1]


def test_a_batch_scanned_without_row_addresses_is_refused() -> None:
    """Forgetting ``with_row_address=True`` fails here rather than staging unkeyed rows.

    Staged rows with no checkpoint id would be written successfully and then
    checkpoint nothing, so the whole phase would silently lose its recovery.
    """
    batch = pa.RecordBatch.from_pydict({"clip_id": ["a"]})
    with pytest.raises(ValueError, match="with_row_address"):
        attach_row_identity(batch, scan_schema(("clip_id",)))


def test_a_scan_column_the_source_contract_lacks_is_refused() -> None:
    """A projection naming an unknown column fails here, not as a Lance planner error."""
    with pytest.raises(ValueError, match="not embedding source columns"):
        scan_schema(("clip_id", "not_a_source_column"))


def test_the_scanned_schema_keeps_each_column_its_declared_source_type() -> None:
    """Types come from the one source contract, so a scan cannot re-declare a column's type."""
    schema = scan_schema(("clip_id", "clip_uri"))
    assert schema.field("clip_uri").type == pa.large_string()
    assert schema.field(CHECKPOINT_ID_COLUMN).type == pa.uint64()
    assert not schema.field(FRAGMENT_ID_COLUMN).nullable


def test_a_staged_vector_column_round_trips_through_parquet_with_a_null_row(tmp_path: pathlib.Path) -> None:
    """The reason the staged vector is width-free: a fixed-size column holding a NULL cannot be read back.

    Executed rather than asserted from the schema, because the defect is in the
    Parquet reader (pyarrow 25.0.1) and not in anything the schema shows. A
    per-row compute failure stages exactly this shape - a NULL vector beside its
    surviving siblings - so a regression to ``fixed_size_list`` makes every
    staged file unreadable at publication and only a round trip catches it.
    """
    schema = staged_schema(IMAGE_COLUMN_GROUP)
    table = pa.table(
        {
            "clip_id": ["a", "b"],
            FRAGMENT_ID_COLUMN: pa.array([_FRAGMENT, _FRAGMENT], pa.int64()),
            CHECKPOINT_ID_COLUMN: pa.array([_row_address(_FRAGMENT, 0), _row_address(_FRAGMENT, 1)], pa.uint64()),
            "embedding_image": pa.array([[1.0] * IMAGE_DIM, None], pa.list_(pa.float32())),
            "embedding_image_model_id": ["fp", None],
        },
        schema=schema,
    )
    path = tmp_path / "staged.parquet"
    pq.write_table(table, path)

    back = pq.read_table(path)

    assert back.column("embedding_image").to_pylist() == [[1.0] * IMAGE_DIM, None]
    # The width the staged schema dropped, restored exactly as publication restores it.
    restored = back.column("embedding_image").cast(pa.list_(pa.float32(), IMAGE_DIM))
    assert restored.to_pylist() == [[1.0] * IMAGE_DIM, None]
