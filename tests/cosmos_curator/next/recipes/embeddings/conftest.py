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

"""Fixtures for the direct-``clips.lance`` embedding recipe tests.

Builds a tiny base-schema ``clips.lance`` (the columns the embedding legs read
from the source: key, media/action URIs, and the text label columns) so a test
can exercise schema widening, selection, and column publication against a real
Lance table without the robot-action-split producer. The base schema is a
faithful subset of that producer's ``OUTCOME_SCHEMA`` (``clip_uri`` /
``action_data_uri`` are ``large_string``); embedding columns are never seeded
here - a test adds them with :func:`add_group_columns` to model the
"column absent until its modality runs" contract.
"""

import pathlib
from collections.abc import Callable, Sequence

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.embeddings.schemas import (
    ACTION_DIM,
    ACTION_GROUP_SCHEMA,
    IMAGE_DIM,
    IMAGE_GROUP_SCHEMA,
    TEXT_DIM,
    TEXT_GROUP_SCHEMA,
)
from cosmos_curator.next.recipes.embeddings.config import EmbeddingPipelineConfig
from cosmos_curator.next.utils.lance_utils import LANCE_DATA_STORAGE_VERSION

# Placeholder clips URI for tests that never open a table. Tests that do open one
# pass their own ``tmp_path``-based URI through the factory.
DEFAULT_CLIPS_URI = "s3://bucket/clips.lance"

# Factory that returns an EmbeddingPipelineConfig. Keyword-only; see ``make_embeddings_config``.
EmbeddingsConfigFactory = Callable[..., EmbeddingPipelineConfig]

# Base ``clips.lance`` schema used by the recipe tests: the exact columns the
# embedding source projection reads (see EMBED_SOURCE_ROW). Mirrors the producer
# OUTCOME_SCHEMA field types so the append-after-widening behaviour a test models
# here matches the real narrow-append path (clip_uri / action_data_uri are
# large_string; the label columns are non-null string).
CLIPS_BASE_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("clip_id", pa.string(), nullable=False),
        pa.field("task_name", pa.string(), nullable=False),
        pa.field("subtask_name", pa.string(), nullable=False),
        pa.field("clip_uri", pa.large_string()),
        pa.field("action_data_uri", pa.large_string()),
        pa.field("source_dataset", pa.string(), nullable=False),
    ]
)

# Factory that returns a ``clips.lance`` URI. Keyword-only; see ``make_clips_table``.
ClipsTableFactory = Callable[..., str]


def _sentinel_uri(row_index: int, kind: str) -> str:
    """Return a deterministic non-empty media/action URI for a row."""
    return f"{kind}{row_index}.bin"


def vector_of(seed: int, dim: int) -> list[float]:
    """Return a deterministic non-null vector of width ``dim``."""
    return [float(seed)] * dim


def fixed_size_list(values: Sequence[list[float] | None], dim: int) -> pa.Array:
    """Build a ``fixed_size_list<float32, dim>`` array; a ``None`` entry is a null row."""
    return pa.array(values, type=pa.list_(pa.float32(), dim))


def write_clips_dataset(
    tmp_path: pathlib.Path,
    *,
    rows: int,
    clip_uris: Sequence[str | None] | None = None,
    action_uris: Sequence[str | None] | None = None,
    text: dict[str, list] | None = None,
    image: dict[str, list] | None = None,
    action: dict[str, list] | None = None,
    rows_per_file: int = 3,
    name: str = "clips.lance",
) -> str:
    """Write a ``clips.lance`` with base columns and optionally populated embedding groups.

    Each of ``text`` / ``image`` / ``action`` (when given) carries per-row column
    value lists (a vector list or ``None`` per row, and the provenance strings),
    so a test can model empty, complete, corrupt, or stale group rows directly.
    Unlike :func:`add_group_columns` (metadata-only, all NULL), this writes the
    values into the base table so selection and basis binding have data to read.

    ``clip_uris`` / ``action_uris`` override the per-row media and action artifact
    URIs; an entry may be ``""`` or ``None`` to model a clip a modality cannot
    embed. Both default to a distinct non-empty URI per row.
    """
    clip_ids = [f"c{i}" for i in range(rows)]
    uris = list(clip_uris) if clip_uris is not None else [f"clip{i}.bin" for i in range(rows)]
    actions = list(action_uris) if action_uris is not None else [_sentinel_uri(i, "act") for i in range(rows)]
    arrays: dict[str, pa.Array] = {
        "clip_id": pa.array(clip_ids, pa.string()),
        "task_name": pa.array([f"task{i}" for i in range(rows)], pa.string()),
        "subtask_name": pa.array([f"subtask{i}" for i in range(rows)], pa.string()),
        "clip_uri": pa.array(uris, pa.large_string()),
        "action_data_uri": pa.array(actions, pa.large_string()),
        "source_dataset": pa.array(["ds_under_test"] * rows, pa.string()),
    }
    fields = list(CLIPS_BASE_SCHEMA)
    if text is not None:
        arrays["embedding_text_subtask"] = fixed_size_list(text["subtask"], TEXT_DIM)
        arrays["embedding_text_task"] = fixed_size_list(text["task"], TEXT_DIM)
        arrays["embedding_text_model_id"] = pa.array(text["model_id"], pa.string())
        fields += list(TEXT_GROUP_SCHEMA)
    if image is not None:
        arrays["embedding_image"] = fixed_size_list(image["image"], IMAGE_DIM)
        arrays["embedding_image_model_id"] = pa.array(image["model_id"], pa.string())
        fields += list(IMAGE_GROUP_SCHEMA)
    if action is not None:
        arrays["embedding_action"] = fixed_size_list(action["action"], ACTION_DIM)
        arrays["embedding_action_descriptor_version"] = pa.array(action["descriptor_version"], pa.string())
        arrays["embedding_action_pca_fingerprint"] = pa.array(action["fingerprint"], pa.string())
        fields += list(ACTION_GROUP_SCHEMA)
    schema = pa.schema(fields)
    table = pa.table({column: arrays[column] for column in schema.names}, schema=schema)
    uri = str(tmp_path / name)
    lance.write_dataset(table, uri, max_rows_per_file=rows_per_file, data_storage_version=LANCE_DATA_STORAGE_VERSION)
    return uri


@pytest.fixture
def make_embeddings_config() -> EmbeddingsConfigFactory:
    """Return a builder for an ``EmbeddingPipelineConfig`` with the boilerplate filled in.

    ``schema_version``, ``kind``, and ``clips_lance_uri`` are required by the
    model but incidental to almost every test, so they are supplied here and any
    of them can still be overridden by keyword. Every keyword is forwarded
    unchanged, so a test can also pass an unknown key to exercise
    ``extra="forbid"``.
    """

    def build(**overrides: object) -> EmbeddingPipelineConfig:
        fields: dict[str, object] = {
            "schema_version": 1,
            "kind": "embeddings",
            "clips_lance_uri": DEFAULT_CLIPS_URI,
        }
        fields.update(overrides)
        return EmbeddingPipelineConfig(**fields)

    return build


def add_group_columns(uri: str, group_schema: pa.Schema) -> None:
    """Add an all-nullable column group to ``uri`` as one metadata-only version."""
    dataset = lance.dataset(uri)
    dataset.add_columns(group_schema)


@pytest.fixture
def make_clips_table(tmp_path: pathlib.Path) -> ClipsTableFactory:
    """Return a builder for a base-schema ``clips.lance`` under ``tmp_path``.

    Keyword arguments (all optional):

    - ``rows``: number of clip rows (default 6);
    - ``rows_per_file``: ``max_rows_per_file`` so a test can force multiple
      fragments (default 3, i.e. two fragments at the default row count);
    - ``clip_ids``: per-row key override, so a test can model the duplicate
      ``clip_id`` the producer does not rule out. Defaults to ``c0..c{rows-1}``;
    - ``clip_uris`` / ``action_uris``: per-row override lists; an entry may be
      ``None`` or ``""`` to model a clip with no media / no action artifact.
      Defaults populate every row with a distinct non-empty URI;
    - ``name``: dataset directory name (default ``clips.lance``).

    Returns the dataset URI (a local path string). Cleanup is automatic through
    ``tmp_path``.
    """

    def build(
        *,
        rows: int = 6,
        rows_per_file: int = 3,
        clip_ids: Sequence[str] | None = None,
        clip_uris: Sequence[str | None] | None = None,
        action_uris: Sequence[str | None] | None = None,
        name: str = "clips.lance",
    ) -> str:
        keys = list(clip_ids) if clip_ids is not None else [f"c{i}" for i in range(rows)]
        uris = list(clip_uris) if clip_uris is not None else [_sentinel_uri(i, "clip") for i in range(rows)]
        actions = list(action_uris) if action_uris is not None else [_sentinel_uri(i, "act") for i in range(rows)]
        table = pa.table(
            {
                "clip_id": pa.array(keys, pa.string()),
                "task_name": pa.array([f"task{i}" for i in range(rows)], pa.string()),
                "subtask_name": pa.array([f"subtask{i}" for i in range(rows)], pa.string()),
                "clip_uri": pa.array(uris, pa.large_string()),
                "action_data_uri": pa.array(actions, pa.large_string()),
                "source_dataset": pa.array(["ds_under_test"] * rows, pa.string()),
            },
            schema=CLIPS_BASE_SCHEMA,
        )
        uri = str(tmp_path / name)
        lance.write_dataset(
            table,
            uri,
            max_rows_per_file=rows_per_file,
            data_storage_version=LANCE_DATA_STORAGE_VERSION,
        )
        return uri

    return build
