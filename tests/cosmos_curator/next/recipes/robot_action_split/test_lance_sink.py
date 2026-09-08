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

"""Append-after-evolution regression tests for the robot-action-split Lance sink.

The embeddings recipe widens the shared ``clips.lance`` table with nullable
``embedding_*`` column groups. The producer's append path must stay narrow: it
writes only its ``CLIP_SCHEMA`` columns and never imports the embeddings
package, and Lance schema evolution must supply typed NULLs for the omitted
embedding fields on the newly appended rows. These tests pin that contract
against the *production* writer (``open_or_create_clip_table`` +
``write_clip_fragment`` + ``append_clip_fragment``, not a raw
``lance.write_dataset``), so a regression in the sink - or in the Lance
version's narrow-append-into-widened-table behaviour - is caught here.

``CLIP_SCHEMA`` is successes-only (see
``docs/curator/design/curator-next-robot-action-split.md``, "Cross-Run Recovery"): unlike
the former ``OUTCOME_SCHEMA``, ``clip_uri``/``action_data_uri`` are non-null,
so every row built here needs real values. The 6-column read contract embed
depends on (``EMBED_SOURCE_ROW`` in ``cosmos_curator/next/embeddings/schemas.py``)
is unaffected: ``clip_id``/``task_name``/``subtask_name``/``source_dataset``
were already non-null and stay that way, and ``clip_uri``/``action_data_uri``
were never actually written as null in practice even before this change (the
former writer already filtered to ``status == "success"`` rows only).

All tests are pure Lance (no Ray): the production writer uses only
``write_fragments`` + ``LanceDataset.commit``. The schema widening and the
one-off fill of an existing row use the embeddings package the way the recipe
would, which is allowed in a test even though the sink itself must not.
"""

import pathlib
from typing import Any

import lance
import numpy as np
import pyarrow as pa

from cosmos_curator.next.embeddings.schemas import IMAGE_COLUMN_GROUP, IMAGE_DIM, TEXT_COLUMN_GROUP
from cosmos_curator.next.recipes.embeddings.columns import ensure_embedding_columns
from cosmos_curator.next.recipes.robot_action_split.contracts import CLIP_RECORD_SCHEMA_VERSION, MEDIA_CONTRACT_VERSION
from cosmos_curator.next.recipes.robot_action_split.lance_sink import (
    append_clip_fragment,
    open_or_create_clip_table,
    write_clip_fragment,
)
from cosmos_curator.next.recipes.robot_action_split.records import CLIP_SCHEMA, clip_table, error_records


def _make_outcome(
    clip_id: str,
    *,
    task_description: str | None = "A person prepares coffee.",
    clip_uri: str | None = None,
    action_data_uri: str | None = None,
) -> dict[str, Any]:
    """Build one minimal successful outcome dict conforming to ``CLIP_SCHEMA``.

    Only ``clip_id``, ``task_description``, and the two optional URIs vary
    between rows; every other required field gets a deterministic placeholder.
    ``status`` is ``success`` so ``clip_table`` writes the row (it drops
    non-success rows) — every clip row is successful in practice, so
    ``clip_uri``/``action_data_uri`` default to deterministic placeholders
    rather than ``None``.
    """
    return {
        "clip_id": clip_id,
        "span_group_id": f"span_{clip_id}",
        "view_name": "main",
        "source_id": f"src_{clip_id}",
        "source_dataset": "ds_under_test",
        "episode_id": "ep_000",
        "episode_index": 0,
        "subtask_index": 0,
        "subtask_name": "pick up coffee pod",
        "task_index": 0,
        "task_name": "make coffee",
        "task_description": task_description,
        "frame_start": 0,
        "frame_end": 100,
        "start_ns": 0,
        "end_ns": 1_000_000,
        "native_fps": 24.0,
        "episode_from_timestamp": 0.0,
        "clip_uri": clip_uri if clip_uri is not None else f"{clip_id}.mp4",
        "action_data_uri": action_data_uri if action_data_uri is not None else f"{clip_id}.bin",
        "camera_motion_annotation": None,
        "status": "success",
    }


def _write_outcomes(outcomes: list[dict[str, Any]], *, lance_uri: str) -> int:
    """Bootstrap, stage, and commit one fragment of successful outcome rows."""
    open_or_create_clip_table(uri=lance_uri, storage_profile="default")
    table = clip_table(
        outcomes,
        record_schema_version=CLIP_RECORD_SCHEMA_VERSION,
        media_contract_version=MEDIA_CONTRACT_VERSION,
    )
    candidate = write_clip_fragment(table, uri=lance_uri, storage_profile="default")
    assert candidate is not None
    return append_clip_fragment(candidate, uri=lance_uri, storage_profile="default", attempts=1)


def _fill_image_group(uri: str) -> dict[str, list[float]]:
    """Fill the image group for every current row via one ``Update`` commit.

    Mirrors the publisher's per-fragment ``update_columns`` write pinned in
    ``tests/cosmos_curator/next/recipes/embeddings/test_lance_column_contract.py``,
    but minimal: one deterministic vector per row keyed on the global ``_rowaddr``.
    Returns the per-clip vector written so a caller can assert a later base-row
    append left these existing values byte-identical.
    """
    dataset = lance.dataset(uri)
    stored_type = dataset.schema.field(IMAGE_COLUMN_GROUP.primary_vector).type
    written: dict[str, list[float]] = {}
    updated: list[Any] = []
    modified: set[int] = set()
    for fragment in dataset.get_fragments():
        table = fragment.scanner(columns=["clip_id"], with_row_address=True).to_table()
        addrs = table.column("_rowaddr").to_pylist()
        clip_ids = table.column("clip_id").to_pylist()
        rows = len(addrs)
        matrix = np.arange(rows * IMAGE_DIM, dtype=np.float32).reshape(rows, IMAGE_DIM)
        for index, clip_id in enumerate(clip_ids):
            written[clip_id] = matrix[index].tolist()
        vectors = pa.FixedSizeListArray.from_arrays(pa.array(matrix.reshape(-1), pa.float32()), IMAGE_DIM)
        batch = pa.table(
            {
                "_rowaddr": pa.array(addrs, pa.uint64()),
                "embedding_image": vectors.cast(stored_type),
                "embedding_image_model_id": pa.array([f"model:{clip_id}" for clip_id in clip_ids], pa.string()),
            }
        )
        new_meta, fields_modified = fragment.update_columns(batch, left_on="_rowaddr", right_on="_rowaddr")
        updated.append(new_meta)
        modified.update(fields_modified)
    operation = lance.LanceOperation.Update(updated_fragments=updated, fields_modified=sorted(modified))
    lance.LanceDataset.commit(uri, operation, read_version=dataset.version)
    return written


def _image_data_file_paths(uri: str) -> set[str]:
    """Return the set of data-file paths that hold the image vector column today.

    A base-row append adds a new fragment; it must not rewrite the data files of
    the fragments that already carry image values, so a pre-append snapshot of
    this set must remain a subset after the append.
    """
    dataset = lance.dataset(uri)
    image_field_id = {field.name(): field.id() for field in dataset.lance_schema.fields()}["embedding_image"]
    paths: set[str] = set()
    for fragment in dataset.get_fragments():
        for data_file in fragment.metadata.data_files():
            if image_field_id in data_file.field_ids():
                paths.add(data_file.path())
    return paths


def test_clip_schema_declares_nullable_large_string_task_description() -> None:
    """Verify the published description field's type, position, and nullability."""
    field = CLIP_SCHEMA.field("task_description")

    assert field.type == pa.large_string()
    assert field.nullable
    assert CLIP_SCHEMA.names.index("task_description") == CLIP_SCHEMA.names.index("task_name") + 1


def test_clip_record_schema_version_is_published_literal_four() -> None:
    """Pin the description schema migration to literal version 4."""
    assert CLIP_RECORD_SCHEMA_VERSION == 4


def test_clip_table_preserves_task_description() -> None:
    """Verify the Lance projection carries the exact episode description value."""
    description = "  Prepare coffee without spilling.\n"

    table = clip_table(
        [_make_outcome("c0", task_description=description)],
        record_schema_version=CLIP_RECORD_SCHEMA_VERSION,
        media_contract_version=MEDIA_CONTRACT_VERSION,
    )

    assert table.column("task_description").to_pylist() == [description]


def test_error_records_preserves_task_description() -> None:
    """Verify failure reports retain pre-cut episode descriptions."""
    description = "  Prepare coffee without spilling.\n"
    failed = _make_outcome("c0", task_description=description) | {
        "status": "failed",
        "clip_uri": None,
        "action_data_uri": None,
        "error_stage": "cut",
        "error_message": "synthetic cut failure",
    }

    assert error_records([failed])[0]["task_description"] == description


def test_append_before_widening_grows_clip_rows(tmp_path: pathlib.Path) -> None:
    """Appending base rows before any embedding group exists simply grows the table."""
    uri = str(tmp_path / "clips.lance")
    create_version = _write_outcomes([_make_outcome("c0"), _make_outcome("c1")], lance_uri=uri)
    append_version = _write_outcomes([_make_outcome("c2"), _make_outcome("c3")], lance_uri=uri)

    assert append_version > create_version
    table = lance.dataset(uri).to_table()
    assert sorted(table.column("clip_id").to_pylist()) == ["c0", "c1", "c2", "c3"]


def test_append_after_widening_reads_present_embedding_fields_as_null(tmp_path: pathlib.Path) -> None:
    """A narrow base-row append into a widened table null-fills every present embedding field."""
    uri = str(tmp_path / "clips.lance")
    _write_outcomes([_make_outcome("c0"), _make_outcome("c1")], lance_uri=uri)
    # Widen with the text and image groups (action deliberately left absent).
    ensure_embedding_columns(lance.dataset(uri), [TEXT_COLUMN_GROUP, IMAGE_COLUMN_GROUP])

    _write_outcomes([_make_outcome("c2"), _make_outcome("c3")], lance_uri=uri)

    present_fields = [*TEXT_COLUMN_GROUP.field_names, *IMAGE_COLUMN_GROUP.field_names]
    result = lance.dataset(uri).to_table(columns=["clip_id", *present_fields]).to_pydict()
    appended = {"c2", "c3"}
    for index, clip_id in enumerate(result["clip_id"]):
        if clip_id in appended:
            for field_name in present_fields:
                assert result[field_name][index] is None, f"{clip_id}.{field_name} must read as NULL"
    # The action group was never enabled, so its columns must not exist at all.
    schema_names = set(lance.dataset(uri).schema.names)
    assert "embedding_action" not in schema_names


def test_append_does_not_alter_existing_embedding_values_or_files(tmp_path: pathlib.Path) -> None:
    """Appending new base rows leaves already-filled embedding values and files untouched."""
    uri = str(tmp_path / "clips.lance")
    _write_outcomes(
        [_make_outcome("c0", clip_uri="c0.mp4"), _make_outcome("c1", clip_uri="c1.mp4")],
        lance_uri=uri,
    )
    ensure_embedding_columns(lance.dataset(uri), [IMAGE_COLUMN_GROUP])
    written = _fill_image_group(uri)
    files_before = _image_data_file_paths(uri)

    _write_outcomes([_make_outcome("c2", clip_uri="c2.mp4")], lance_uri=uri)

    result = lance.dataset(uri).to_table(columns=["clip_id", "embedding_image"]).to_pydict()
    per_clip = dict(zip(result["clip_id"], result["embedding_image"], strict=True))
    assert per_clip["c0"] == written["c0"]  # existing value byte-identical
    assert per_clip["c1"] == written["c1"]
    assert per_clip["c2"] is None  # appended row reads NULL
    # The append added a new fragment; it did not rewrite the existing image files.
    assert files_before <= _image_data_file_paths(uri)
