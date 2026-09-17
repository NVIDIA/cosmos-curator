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

"""Behavioural tests for the driver-side schema and read functions over a clips table.

Every function under test takes a dataset the CALLER opened, so these tests are
pure Lance and run everywhere - nothing here touches Ray. Two concerns share the
file because they share the same fixture shape:

- SCHEMA: widening one group in (``ensure_embedding_columns``), a single metadata
  commit;
- READ: the owed-work predicate (``pending_filter``), the filled count, and the
  one-column stream.

Most of the predicate is asserted by COUNTING the rows it matches against a real
table rather than by comparing SQL text, so a rewrite preserving the meaning does
not redden the file. ONE test pins the emitted text instead, and the exception is
deliberate: identifier quoting is the failure Lance accepts SILENTLY - a
double-quoted name parses as a string literal and matches every row rather than
raising - so the emitted form is a contract at the boundary and not an
implementation detail behind it.
"""

import pathlib

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.embeddings.schemas import (
    ACTION_COLUMN_GROUP,
    IMAGE_COLUMN_GROUP,
    IMAGE_DIM,
    TEXT_COLUMN_GROUP,
    TEXT_DIM,
    TEXT_GROUP_SCHEMA,
)
from cosmos_curator.next.recipes.embeddings.columns import (
    count_filled,
    ensure_embedding_columns,
    pending_filter,
    scan_column,
)
from cosmos_curator.next.recipes.embeddings.modalities import _IMAGE_APPLICABILITY_FILTER
from cosmos_curator.next.utils.lance_utils import LANCE_DATA_STORAGE_VERSION

from .conftest import (
    CLIPS_BASE_SCHEMA,
    ClipsTableFactory,
    add_group_columns,
    fixed_size_list,
    vector_of,
    write_clips_dataset,
)

_CURRENT_IDENTITY = "the-identity-this-run-would-produce"
_IMAGE_VECTOR = vector_of(1, IMAGE_DIM)


def test_ensure_adds_only_enabled_group_in_one_version(make_clips_table: ClipsTableFactory) -> None:
    """Ensuring one group adds exactly that group's fields in a single metadata version."""
    uri = make_clips_table()
    before = lance.dataset(uri).version

    added, commit_version = ensure_embedding_columns(lance.dataset(uri), [TEXT_COLUMN_GROUP])

    after = lance.dataset(uri)
    assert added == len(TEXT_COLUMN_GROUP.field_names)
    assert commit_version == before + 1
    assert after.version == before + 1
    names = set(after.schema.names)
    assert set(TEXT_COLUMN_GROUP.field_names) <= names
    assert not (set(IMAGE_COLUMN_GROUP.field_names) & names)


def test_text_only_ensure_leaves_other_groups_absent(make_clips_table: ClipsTableFactory) -> None:
    """A text-only ensure creates no image or action columns."""
    uri = make_clips_table()
    ensure_embedding_columns(lance.dataset(uri), [TEXT_COLUMN_GROUP])
    names = set(lance.dataset(uri).schema.names)
    assert not any(name.startswith("embedding_image") for name in names)
    assert not any(name.startswith("embedding_action") for name in names)


def test_ensure_present_group_is_idempotent_noop(make_clips_table: ClipsTableFactory) -> None:
    """Ensuring an already-present, exactly-matching group adds nothing and no new version."""
    uri = make_clips_table()
    add_group_columns(uri, TEXT_GROUP_SCHEMA)
    before = lance.dataset(uri).version

    added, commit_version = ensure_embedding_columns(lance.dataset(uri), [TEXT_COLUMN_GROUP])

    assert added == 0
    assert commit_version is None
    assert lance.dataset(uri).version == before


def test_ensure_one_group_present_another_absent_is_valid(make_clips_table: ClipsTableFactory) -> None:
    """Ensuring [text, image] with text already present adds only the image fields."""
    uri = make_clips_table()
    add_group_columns(uri, TEXT_GROUP_SCHEMA)

    added, _ = ensure_embedding_columns(lance.dataset(uri), [TEXT_COLUMN_GROUP, IMAGE_COLUMN_GROUP])

    assert added == len(IMAGE_COLUMN_GROUP.field_names)
    names = set(lance.dataset(uri).schema.names)
    assert set(IMAGE_COLUMN_GROUP.field_names) <= names


def test_ensure_partial_group_fails(make_clips_table: ClipsTableFactory) -> None:
    """A group with only some of its fields present is a corrupt schema and fails."""
    uri = make_clips_table()
    add_group_columns(uri, pa.schema([TEXT_GROUP_SCHEMA.field("embedding_text_subtask")]))
    with pytest.raises(ValueError, match="partially present"):
        ensure_embedding_columns(lance.dataset(uri), [TEXT_COLUMN_GROUP])


def test_ensure_wrong_dimension_fails(make_clips_table: ClipsTableFactory) -> None:
    """A present image vector of the wrong width fails the exact-match check."""
    uri = make_clips_table()
    wrong = pa.schema(
        [
            pa.field("embedding_image", pa.list_(pa.float32(), IMAGE_DIM + 1), nullable=True),
            pa.field("embedding_image_model_id", pa.string(), nullable=True),
        ]
    )
    add_group_columns(uri, wrong)
    with pytest.raises(ValueError, match="type"):
        ensure_embedding_columns(lance.dataset(uri), [IMAGE_COLUMN_GROUP])


def test_ensure_wrong_type_fails(make_clips_table: ClipsTableFactory) -> None:
    """A present image vector stored as a variable list (not fixed-size) fails."""
    uri = make_clips_table()
    wrong = pa.schema(
        [
            pa.field("embedding_image", pa.list_(pa.float32()), nullable=True),
            pa.field("embedding_image_model_id", pa.string(), nullable=True),
        ]
    )
    add_group_columns(uri, wrong)
    with pytest.raises(ValueError, match="type"):
        ensure_embedding_columns(lance.dataset(uri), [IMAGE_COLUMN_GROUP])


def test_ensure_wrong_nullability_fails(tmp_path: pathlib.Path) -> None:
    """A present image vector declared non-nullable fails the exact-match check."""
    rows = 2
    schema = pa.schema(
        [
            *list(CLIPS_BASE_SCHEMA),
            pa.field("embedding_image", pa.list_(pa.float32(), IMAGE_DIM), nullable=False),
            pa.field("embedding_image_model_id", pa.string(), nullable=True),
        ]
    )
    table = pa.table(
        {
            "clip_id": pa.array([f"c{i}" for i in range(rows)], pa.string()),
            "task_name": pa.array([f"t{i}" for i in range(rows)], pa.string()),
            "subtask_name": pa.array([f"s{i}" for i in range(rows)], pa.string()),
            "clip_uri": pa.array([f"u{i}" for i in range(rows)], pa.large_string()),
            "action_data_uri": pa.array([f"a{i}" for i in range(rows)], pa.large_string()),
            "source_dataset": pa.array(["ds"] * rows, pa.string()),
            "embedding_image": fixed_size_list([vector_of(i, IMAGE_DIM) for i in range(rows)], IMAGE_DIM),
            "embedding_image_model_id": pa.array(["m"] * rows, pa.string()),
        },
        schema=schema,
    )
    uri = str(tmp_path / "clips.lance")
    lance.write_dataset(table, uri, data_storage_version=LANCE_DATA_STORAGE_VERSION)
    with pytest.raises(ValueError, match="nullable"):
        ensure_embedding_columns(lance.dataset(uri), [IMAGE_COLUMN_GROUP])


def test_ensure_tolerates_unknown_future_group(make_clips_table: ClipsTableFactory) -> None:
    """An unrelated future embedding group on the table does not block ensuring a known one."""
    uri = make_clips_table()
    future = pa.schema(
        [
            pa.field("embedding_future_x", pa.list_(pa.float32(), 8), nullable=True),
            pa.field("embedding_future_model_id", pa.string(), nullable=True),
        ]
    )
    add_group_columns(uri, future)

    added, _ = ensure_embedding_columns(lance.dataset(uri), [TEXT_COLUMN_GROUP])

    names = set(lance.dataset(uri).schema.names)
    assert added == len(TEXT_COLUMN_GROUP.field_names)
    assert {"embedding_future_x", "embedding_future_model_id"} <= names


def test_count_filled_counts_non_null_primary(tmp_path: pathlib.Path) -> None:
    """count_filled returns the number of rows whose primary vector is non-NULL."""
    uri = write_clips_dataset(
        tmp_path,
        rows=3,
        image={"image": [vector_of(0, IMAGE_DIM), None, vector_of(2, IMAGE_DIM)], "model_id": ["m", None, "m"]},
    )
    assert count_filled(lance.dataset(uri), IMAGE_COLUMN_GROUP) == 2


def test_count_filled_is_zero_for_a_group_whose_columns_are_absent(make_clips_table: ClipsTableFactory) -> None:
    """Asking how much of a not-yet-added group is filled reads zero rather than failing.

    The action leg's outcome check counts a group that a failed run may never have
    widened, so an unresolvable column must answer "nothing is filled" instead of
    raising out of the SQL planner.
    """
    uri = make_clips_table()
    assert count_filled(lance.dataset(uri), IMAGE_COLUMN_GROUP) == 0


@pytest.mark.parametrize(
    ("clip_uri", "vector", "identity", "owed_rows"),
    [
        pytest.param("clip.bin", None, None, 1, id="no vector yet"),
        pytest.param("clip.bin", _IMAGE_VECTOR, None, 1, id="vector with no recorded identity"),
        pytest.param("clip.bin", _IMAGE_VECTOR, "an-older-computation", 1, id="another computation's vector"),
        pytest.param("clip.bin", _IMAGE_VECTOR, _CURRENT_IDENTITY, 0, id="this computation's vector"),
        pytest.param("", None, None, 0, id="row this modality cannot embed"),
    ],
)
def test_which_rows_the_predicate_says_are_owed(
    clip_uri: str, vector: list[float] | None, identity: str | None, owed_rows: int, tmp_path: pathlib.Path
) -> None:
    """One row per state, so a wrong verdict names the state that produced it.

    Each case is a separate one-row table rather than one table holding all five,
    because a combined assertion on a clip_id set cannot say WHICH state the
    predicate got wrong. The "vector with no recorded identity" case is the one
    three-valued logic loses: ``NULL != 'x'`` is unknown rather than true, so
    without the explicit ``IS NULL`` arm that row silently stops being owed.
    """
    uri = write_clips_dataset(
        tmp_path,
        rows=1,
        clip_uris=[clip_uri],
        image={"image": [vector], "model_id": [identity]},
    )
    row_filter = pending_filter(
        IMAGE_COLUMN_GROUP, _IMAGE_APPLICABILITY_FILTER, {"embedding_image_model_id": _CURRENT_IDENTITY}
    )

    assert lance.dataset(uri).count_rows(filter=row_filter) == owed_rows


def test_a_row_with_a_matching_identity_but_a_missing_companion_vector_is_still_owed(
    tmp_path: pathlib.Path,
) -> None:
    """The primary vector and model id alone do not make a text row complete.

    A row whose subtask vector and provenance match this run but whose task
    companion is NULL must still be selected; otherwise a partial publish would
    freeze the row as current under the identity check.
    """
    uri = write_clips_dataset(
        tmp_path,
        rows=1,
        text={
            "subtask": [vector_of(1, TEXT_DIM)],
            "task": [None],
            "model_id": [_CURRENT_IDENTITY],
        },
    )
    row_filter = pending_filter(TEXT_COLUMN_GROUP, None, {"embedding_text_model_id": _CURRENT_IDENTITY})

    assert lance.dataset(uri).count_rows(filter=row_filter) == 1


def test_a_modality_with_no_applicability_predicate_still_owes_every_unembedded_row(
    tmp_path: pathlib.Path,
) -> None:
    """Text applies to every row, so a row with no media is still owed text work.

    Pinned separately from the parameterized cases because ``applicability_filter
    is None`` takes the other branch of the emitted shape, and a regression there
    would silently make the text leg inherit the image leg's media requirement.
    """
    rows = 3
    uri = write_clips_dataset(
        tmp_path,
        rows=rows,
        clip_uris=[""] * rows,
        text={"subtask": [None] * rows, "task": [None] * rows, "model_id": [None] * rows},
    )
    row_filter = pending_filter(TEXT_COLUMN_GROUP, None, {"embedding_text_model_id": _CURRENT_IDENTITY})

    assert lance.dataset(uri).count_rows(filter=row_filter) == rows


def test_an_identity_value_containing_a_quote_still_selects_correctly(tmp_path: pathlib.Path) -> None:
    """A single quote inside the identity value is escaped rather than closing the literal.

    Interpolating it raw produces SQL Lance cannot parse, so this asserts the
    verdict rather than merely that nothing raised: the row holding exactly that
    value is current, and the row holding a different one is owed.
    """
    quoted = "o'brien"
    uri = write_clips_dataset(
        tmp_path,
        rows=2,
        image={"image": [_IMAGE_VECTOR, _IMAGE_VECTOR], "model_id": [quoted, "other"]},
    )
    row_filter = pending_filter(IMAGE_COLUMN_GROUP, None, {"embedding_image_model_id": quoted})

    matched = lance.dataset(uri).to_table(columns=["clip_id"], filter=row_filter)

    assert matched.column("clip_id").to_pylist() == ["c1"]


def test_the_predicate_leaves_identifiers_bare_and_quotes_only_the_value() -> None:
    """The emitted text names columns bare and the identity value as a string literal.

    The one text assertion in this file. Lance ACCEPTS a double-quoted identifier
    on this surface and reads it as a string literal, so `"embedding_image" IS
    NULL` compares two constants and the predicate silently matches the wrong
    rows instead of raising - a failure a row count catches only on a table that
    happens to hold a distinguishing row. Pinning the whole string makes the
    quoting itself the thing under test.
    """
    assert pending_filter(IMAGE_COLUMN_GROUP, None, {"embedding_image_model_id": "fp"}) == (
        "(embedding_image IS NULL OR embedding_image_model_id IS NULL OR embedding_image_model_id != 'fp')"
    )


def test_an_identity_missing_a_provenance_column_is_refused() -> None:
    """Building the predicate without a value for every provenance column raises.

    Uses the action group, the only one with two provenance columns, because a
    partial identity is not expressible for a single-column group. A narrowed
    predicate would stop selecting rows stale in the omitted column, and nothing
    downstream could detect that they were skipped.
    """
    with pytest.raises(ValueError, match="missing"):
        pending_filter(ACTION_COLUMN_GROUP, None, {"embedding_action_descriptor_version": "dual-wrist-v3"})


def test_scan_column_streams_only_the_requested_column_of_matching_rows(
    make_clips_table: ClipsTableFactory,
) -> None:
    """``scan_column`` projects one column and pushes the predicate into the scan."""
    uri = make_clips_table(rows=4, rows_per_file=2, action_uris=["a0.bin", "", "a2.bin", None])

    batches = list(
        scan_column(
            lance.dataset(uri),
            "action_data_uri",
            row_filter="action_data_uri IS NOT NULL AND action_data_uri != ''",
            batch_size=1,
        )
    )

    assert [batch.schema.names for batch in batches] == [["action_data_uri"]] * len(batches)
    values = [value for batch in batches for value in batch.column("action_data_uri").to_pylist()]
    assert values == ["a0.bin", "a2.bin"]


def test_scan_column_yields_nothing_when_no_row_matches(make_clips_table: ClipsTableFactory) -> None:
    """An empty match yields no batches at all, so a caller needs no zero-row special case."""
    uri = make_clips_table(rows=2, rows_per_file=2)

    batches = list(
        scan_column(lance.dataset(uri), "action_data_uri", row_filter="action_data_uri = 'absent'", batch_size=8)
    )

    assert batches == []
