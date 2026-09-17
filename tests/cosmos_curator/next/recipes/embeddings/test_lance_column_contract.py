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

"""Pinned Lance column-update contract (hard gate for the wide-table design).

Embeddings live as nullable column groups on the ``clips.lance`` table that
``robot_action_split`` owns and other legs read. That is only safe if the storage
engine actually behaves as the design assumes, so this file pins the guarantees
against the installed Lance rather than trusting API names:

1. ``add_columns(pa.Schema)`` installs a nullable fixed-size-list column group in
   a single commit, and every row reads NULL until a value lands;
2. the producer can still append its narrow base rows after the table has been
   widened, and those rows read every embedding field as NULL;
3. a per-fragment ``update_columns`` keyed on the ordinary persisted ``clip_id``
   column targets exactly the rows present in the update table (left-outer: an
   absent row stays NULL), creates no tombstones, preserves fragment ids and row
   count, and makes a group's vector and provenance visible together;
4. a column ``Update`` committed at an older ``read_version`` rebases cleanly
   over a concurrent base-row ``Append``;
5. ``drop_columns`` is all-or-nothing over the names it is given, and a dropped
   group can be re-added and refilled by the ordinary write path - the two facts
   a WIDTH migration rests on. A model whose vector width changes cannot be
   overwritten in place (``ensure_embedding_columns`` rejects a present group
   whose type differs), so the group is dropped and re-added at the new width and
   an ordinary run refills it.

These assertions are deliberately narrower than the storage engine's behaviour:
they cover only what the pipeline depends on, keyed the way the pipeline keys it.
The rejected alternatives are not re-tested here because each fails structurally
rather than intermittently - ``merge`` and ``merge_columns`` are add-only, a
``Merge`` commit demands the complete fragment list rather than the touched
subset, and ``merge_insert`` replaces whole rows, so it writes tombstones and
reorders them. See ``docs/curator/design/curator-next-embeddings.md`` for that
comparison and the design it justifies.
"""

import lance
import numpy as np
import pyarrow as pa
import pytest

from .conftest import CLIPS_BASE_SCHEMA, ClipsTableFactory, add_group_columns

# Synthetic fixed-size-list groups for the contract proof. These are deliberately
# NOT the production schemas: the storage contract is independent of what a vector
# means. Width 4 keeps the fixtures tiny; the list child is left nullable
# (pyarrow default) so an all-NULL column round-trips.
_DIM = 4


def _fsl() -> pa.DataType:
    return pa.list_(pa.float32(), _DIM)


_TEXT_GROUP = pa.schema(
    [
        pa.field("embedding_text_subtask", _fsl(), nullable=True),
        pa.field("embedding_text_task", _fsl(), nullable=True),
        pa.field("embedding_text_model_id", pa.string(), nullable=True),
    ]
)
_IMAGE_GROUP = pa.schema(
    [
        pa.field("embedding_image", _fsl(), nullable=True),
        pa.field("embedding_image_model_id", pa.string(), nullable=True),
    ]
)
_ACTION_GROUP = pa.schema(
    [
        pa.field("embedding_action", _fsl(), nullable=True),
        pa.field("embedding_action_descriptor_version", pa.string(), nullable=True),
        pa.field("embedding_action_pca_fingerprint", pa.string(), nullable=True),
    ]
)
_ALL_GROUPS = pa.schema(list(_TEXT_GROUP) + list(_IMAGE_GROUP) + list(_ACTION_GROUP))


def _group_value_table(dataset: lance.LanceDataset, group_schema: pa.Schema, clip_ids: list[str]) -> pa.Table:
    """Build a ``clip_id`` + group-field update table with deterministic non-null values.

    Mirrors what a fill worker hands to ``update_columns``: the join key plus
    every field of one group. Vectors are cast to the stored fixed-size-list type
    and provenance strings encode the field name so a test can read them back.
    """
    key_type = dataset.schema.field("clip_id").type
    columns: dict[str, pa.Array] = {"clip_id": pa.array(clip_ids, key_type)}
    rows = len(clip_ids)
    for field in group_schema:
        stored = dataset.schema.field(field.name).type
        if pa.types.is_fixed_size_list(stored):
            matrix = np.arange(rows * _DIM, dtype=np.float32).reshape(rows, _DIM)
            values = pa.array(matrix.reshape(-1), pa.float32())
            columns[field.name] = pa.FixedSizeListArray.from_arrays(values, _DIM).cast(stored)
        else:
            columns[field.name] = pa.array([f"{field.name}:v" for _ in range(rows)], pa.string())
    return pa.table(columns)


def _publish_group(
    uri: str,
    group_schema: pa.Schema,
    *,
    read_version: int | None = None,
    fragment_ids: set[int] | None = None,
    keep_clip: object = None,
) -> tuple[lance.LanceDataset, list[int]]:
    """Fill ``group_schema`` via per-fragment ``update_columns`` + one ``Update`` commit.

    This is the production write shape in miniature: each fragment is updated
    independently against its own ``clip_id`` values, and the resulting fragment
    metadata is committed as a single ``Update`` naming only the touched fragments.

    Args:
        uri: Target table.
        group_schema: The column group to fill (all fields written together).
        read_version: Commit against this pinned version (defaults to current);
            an older value is used to model a concurrent writer.
        fragment_ids: Restrict the fill to these fragments (default: all).
        keep_clip: Optional ``clip_id -> bool`` predicate selecting which rows of
            a fragment to fill; unselected rows must stay NULL (proves the join
            key targets exactly the rows the update table carries).

    Returns:
        The committed dataset handle and the sorted list of modified field ids.

    """
    dataset = lance.dataset(uri)
    version = dataset.version if read_version is None else read_version
    updated: list[object] = []
    modified: set[int] = set()
    for fragment in dataset.get_fragments():
        if fragment_ids is not None and fragment.fragment_id not in fragment_ids:
            continue
        clip_ids = fragment.scanner(columns=["clip_id"]).to_table().column("clip_id").to_pylist()
        if keep_clip is not None:
            clip_ids = [clip_id for clip_id in clip_ids if keep_clip(clip_id)]  # type: ignore[operator]
        if not clip_ids:
            continue
        batch = _group_value_table(dataset, group_schema, clip_ids)
        new_meta, fields_modified = fragment.update_columns(batch, left_on="clip_id", right_on="clip_id")
        updated.append(new_meta)
        modified.update(fields_modified)
    operation = lance.LanceOperation.Update(updated_fragments=updated, fields_modified=sorted(modified))
    committed = lance.LanceDataset.commit(uri, operation, read_version=version)
    return committed, sorted(modified)


def test_add_columns_installs_all_nullable_fixed_size_list_groups_atomically(
    make_clips_table: ClipsTableFactory,
) -> None:
    """One ``add_columns`` adds every group's fields in a single version, all rows NULL.

    Widening is metadata-only: the new fields exist and read NULL for every row
    without any data file being written, which is why installing a group can
    neither rewrite nor tombstone a base row.
    """
    uri = make_clips_table(rows=6, rows_per_file=3)
    before_version = lance.dataset(uri).version

    add_group_columns(uri, _ALL_GROUPS)

    after = lance.dataset(uri)
    assert after.version == before_version + 1  # exactly one commit for the whole widening
    for field in _ALL_GROUPS:
        stored = after.schema.field(field.name)
        assert stored.type == field.type
        assert stored.nullable
    table = after.to_table(columns=[field.name for field in _ALL_GROUPS])
    for column in table.itercolumns():
        assert column.null_count == after.count_rows()


def test_base_schema_append_after_widening_reads_embedding_groups_as_null(
    make_clips_table: ClipsTableFactory,
) -> None:
    """A narrow base-row append into a widened table null-fills the embedding fields."""
    uri = make_clips_table(rows=4, rows_per_file=2)
    add_group_columns(uri, _ALL_GROUPS)

    appended = pa.table(
        {
            "clip_id": pa.array(["c4", "c5"], pa.string()),
            "task_name": pa.array(["task4", "task5"], pa.string()),
            "subtask_name": pa.array(["subtask4", "subtask5"], pa.string()),
            "clip_uri": pa.array(["clip4.bin", "clip5.bin"], pa.large_string()),
            "action_data_uri": pa.array(["act4.bin", "act5.bin"], pa.large_string()),
            "source_dataset": pa.array(["ds_under_test", "ds_under_test"], pa.string()),
        },
        schema=CLIPS_BASE_SCHEMA,
    )
    # The producer writes only its narrow OUTCOME_SCHEMA columns; Lance schema
    # evolution supplies typed NULLs for the omitted embedding fields.
    lance.write_dataset(appended, uri, mode="append", data_storage_version="2.2")

    result = lance.dataset(uri).to_table(columns=["clip_id", *[f.name for f in _ALL_GROUPS]])
    rows = result.to_pydict()
    for index, clip_id in enumerate(rows["clip_id"]):
        if clip_id in {"c4", "c5"}:
            for field in _ALL_GROUPS:
                assert rows[field.name][index] is None


def test_per_fragment_update_columns_joins_on_the_clip_id_key(make_clips_table: ClipsTableFactory) -> None:
    """``update_columns`` fills exactly the ``clip_id`` rows the update table carries.

    Left-outer semantics are what make a partial fill safe: a fragment row absent
    from the update table keeps its previous value (here NULL), so a run that
    embeds only some rows cannot blank the rest.
    """
    uri = make_clips_table(rows=6, rows_per_file=3)
    add_group_columns(uri, _IMAGE_GROUP)

    # Fragment 1 holds c3, c4, c5; fill only c3 and c5.
    _publish_group(uri, _IMAGE_GROUP, fragment_ids={1}, keep_clip=lambda clip_id: clip_id in {"c3", "c5"})

    result = lance.dataset(uri).to_table(columns=["clip_id", "embedding_image"]).to_pydict()
    non_null = {clip for clip, vec in zip(result["clip_id"], result["embedding_image"], strict=True) if vec is not None}
    assert non_null == {"c3", "c5"}


def test_column_update_preserves_fragment_ids_and_row_count(make_clips_table: ClipsTableFactory) -> None:
    """A column update changes neither the fragment id set nor the row count."""
    uri = make_clips_table(rows=6, rows_per_file=3)
    add_group_columns(uri, _IMAGE_GROUP)
    before = lance.dataset(uri)
    before_ids = sorted(fragment.fragment_id for fragment in before.get_fragments())
    before_rows = before.count_rows()

    _publish_group(uri, _IMAGE_GROUP)

    after = lance.dataset(uri)
    assert sorted(fragment.fragment_id for fragment in after.get_fragments()) == before_ids
    assert after.count_rows() == before_rows


def test_column_update_creates_no_deleted_rows(make_clips_table: ClipsTableFactory) -> None:
    """A column update adds no tombstones (unlike a row-level ``merge_insert``)."""
    uri = make_clips_table(rows=6, rows_per_file=3)
    add_group_columns(uri, _IMAGE_GROUP)

    _publish_group(uri, _IMAGE_GROUP)

    after = lance.dataset(uri)
    for fragment in after.get_fragments():
        assert fragment.metadata.num_deletions == 0


def test_append_rebases_with_an_uncommitted_column_update(make_clips_table: ClipsTableFactory) -> None:
    """An Update at an older read_version rebases over a concurrent base-row Append."""
    uri = make_clips_table(rows=6, rows_per_file=3)
    add_group_columns(uri, _IMAGE_GROUP)
    read_version = lance.dataset(uri).version

    # A concurrent producer appends two narrow base rows, advancing the version.
    appended = pa.table(
        {
            "clip_id": pa.array(["c6", "c7"], pa.string()),
            "task_name": pa.array(["task6", "task7"], pa.string()),
            "subtask_name": pa.array(["subtask6", "subtask7"], pa.string()),
            "clip_uri": pa.array(["clip6.bin", "clip7.bin"], pa.large_string()),
            "action_data_uri": pa.array(["act6.bin", "act7.bin"], pa.large_string()),
            "source_dataset": pa.array(["ds_under_test", "ds_under_test"], pa.string()),
        },
        schema=CLIPS_BASE_SCHEMA,
    )
    lance.write_dataset(appended, uri, mode="append", data_storage_version="2.2")

    # Commit the image fill against the pre-append snapshot: Lance must rebase it.
    _publish_group(uri, _IMAGE_GROUP, read_version=read_version, fragment_ids={0})

    result = lance.dataset(uri).to_table(columns=["clip_id", "embedding_image"]).to_pydict()
    per_clip = dict(zip(result["clip_id"], result["embedding_image"], strict=True))
    assert set(per_clip) == {f"c{i}" for i in range(8)}  # both changes preserved
    assert per_clip["c0"] is not None  # fragment 0 image filled
    assert per_clip["c6"] is None  # appended row stays NULL
    assert per_clip["c7"] is None  # appended row stays NULL


def test_fixed_size_list_vector_and_provenance_publish_together(make_clips_table: ClipsTableFactory) -> None:
    """A single Update commit makes a group's vector and provenance visible atomically."""
    uri = make_clips_table(rows=6, rows_per_file=3)
    add_group_columns(uri, _IMAGE_GROUP)

    _publish_group(uri, _IMAGE_GROUP, fragment_ids={0})

    result = lance.dataset(uri).to_table(columns=["embedding_image", "embedding_image_model_id"]).to_pydict()
    for vector, model_id in zip(result["embedding_image"], result["embedding_image_model_id"], strict=True):
        assert (vector is None) == (model_id is None)  # never a vector without its provenance


def test_dropping_one_absent_name_drops_none_of_the_present_ones(make_clips_table: ClipsTableFactory) -> None:
    """``drop_columns`` is all-or-nothing: one unknown name aborts the whole drop.

    This is why a width migration narrows to the fields the schema really carries.
    A half-installed group passed whole would raise on its missing field and leave
    the present one behind, so the corrupt group could never be dropped at all.
    """
    uri = make_clips_table(rows=2, rows_per_file=2)
    add_group_columns(uri, _IMAGE_GROUP)
    dataset = lance.dataset(uri)
    before = dataset.version

    with pytest.raises(ValueError, match="does not exist"):
        dataset.drop_columns(["embedding_image", "embedding_image_absent"])

    after = lance.dataset(uri)
    assert set(_IMAGE_GROUP.names) <= set(after.schema.names)
    assert after.version == before


def test_a_dropped_group_can_be_re_added_and_refilled_by_the_ordinary_write_path(
    make_clips_table: ClipsTableFactory,
) -> None:
    """Drop then refill: a re-added group takes new values through the same per-fragment update.

    The drop detaches the group's data file from every fragment, so this proves the
    refill writes a fresh one rather than needing the fragments to be rewritten -
    which is what keeps a width migration to two metadata commits plus an ordinary
    run, instead of a table rewrite.
    """
    uri = make_clips_table(rows=4, rows_per_file=2)
    add_group_columns(uri, _IMAGE_GROUP)
    _publish_group(uri, _IMAGE_GROUP)
    lance.dataset(uri).drop_columns(list(_IMAGE_GROUP.names))
    add_group_columns(uri, _IMAGE_GROUP)

    _publish_group(uri, _IMAGE_GROUP)

    refilled = lance.dataset(uri).to_table(columns=list(_IMAGE_GROUP.names)).to_pydict()
    assert all(vector is not None for vector in refilled["embedding_image"])


def test_duplicate_clip_id_receives_the_same_vector_on_both_rows(make_clips_table: ClipsTableFactory) -> None:
    """Two rows sharing one ``clip_id`` both receive that key's value, so no dedup layer is needed.

    ``clip_id`` uniqueness is not enforced by the producer, and Lance resolves a
    duplicate key by taking the value of one matching row. The write path assumes
    rows sharing a key agree on their source columns, so every duplicate computes the
    same value and both rows end up correct - which is why no de-duplicating shuffle
    is paid on every run.
    """
    uri = make_clips_table(rows=2, rows_per_file=2, clip_ids=["dup", "dup"])
    add_group_columns(uri, _IMAGE_GROUP)

    _publish_group(uri, _IMAGE_GROUP)

    result = lance.dataset(uri).to_table(columns=["embedding_image", "embedding_image_model_id"]).to_pydict()
    assert all(vector is not None for vector in result["embedding_image"])
    assert result["embedding_image_model_id"] == ["embedding_image_model_id:v"] * 2
    # The update table gives the two "dup" rows DIFFERENT vectors on purpose; the
    # join collapses the duplicate key to ONE matching value, so both stored rows
    # must be EQUAL. A row-wise (positional) join bug would leave them different.
    assert result["embedding_image"][0] == result["embedding_image"][1]
