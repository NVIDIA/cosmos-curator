# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small canonical clip tables and typed visual results, without a Ray runtime."""

import json
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.enrichment import inputs
from cosmos_curator.next.enrichment.execution import selection_datasource
from cosmos_curator.next.recipes.video_cut.contracts import make_source_id, make_span_id
from cosmos_curator.next.recipes.video_embed import pipeline
from cosmos_curator.next.recipes.video_embed.bindings import resolve_bindings
from cosmos_curator.next.recipes.video_embed.config import ResolvedVideoEmbedConfig, resolve_config_data
from cosmos_curator.next.recipes.video_embed.contracts import READ_COLUMNS, ModelSpec, resolve_model_spec, terminal_row
from cosmos_curator.next.recipes.video_transcode.contracts import (
    CLIP_SCHEMA,
    PUBLICATION_KEY,
    PUBLICATION_VERSION,
    make_clip_id,
)
from tests.cosmos_curator.next.enrichment.conformance import RecipeUnderTest
from tests.cosmos_curator.next.enrichment.helpers import attempt_all


@pytest.fixture
def config(tmp_path: Path) -> ResolvedVideoEmbedConfig:
    """Use local media and compact recovery groups."""
    return resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-embed",
            "input": {"media_root": str(tmp_path / "media")},
            "execution": {
                "parquet_rows_per_file": 1,
                "lance_read_batch_size": 2,
                "media_attempts": 1,
                "publication_memory_gib": 0.25,
            },
        }
    )


@pytest.fixture
def spec(config: ResolvedVideoEmbedConfig) -> ModelSpec:
    """Resolve the same immutable model used by the recipe."""
    return resolve_model_spec(config.model.variant)


def write_clips(config: ResolvedVideoEmbedConfig, *, payload: bytes = b"mp4") -> lance.LanceDataset:
    """Write four immutable clip rows across two fragments with transcode provenance."""
    root = Path(config.input.media_root)
    (root / "clips").mkdir(parents=True, exist_ok=True)
    source_uri = str(root / "source.mp4")
    source_id = make_source_id(source_uri)
    rows = []
    for index in range(4):
        start = index * 1_000_000_000
        span_id = make_span_id(source_id, start, start + 1_000_000_000)
        clip_id = make_clip_id(span_id, "transcode-contract")
        clip_path = root / "clips" / f"{clip_id}.mp4"
        clip_path.write_bytes(payload)
        rows.append(
            {
                "record_schema_version": 1,
                "cut_contract_id": "cut-contract",
                "source_id": source_id,
                "source_uri": source_uri,
                "source_size_bytes": 100,
                "source_duration_ns": 4_000_000_000,
                "source_width": 64,
                "source_height": 48,
                "source_frame_rate": 8.0,
                "source_frame_count": 32,
                "source_video_codec": "h264",
                "stream_index": 0,
                "span_id": span_id,
                "start_ns": start,
                "end_ns": start + 1_000_000_000,
                "transcode_contract_id": "transcode-contract",
                "clip_id": clip_id,
                "clip_uri": str(clip_path),
                "clip_size_bytes": len(payload),
                "clip_duration_ns": 1_000_000_000,
                "clip_width": 64,
                "clip_height": 48,
                "clip_frame_rate": 8.0,
                "clip_frame_count": 8,
                "clip_video_codec": "h264",
            }
        )
    return lance.write_dataset(
        pa.Table.from_pylist(rows, schema=CLIP_SCHEMA),
        config.input.clips_lance_uri,
        max_rows_per_file=2,
        data_storage_version="2.2",
        transaction_properties={
            PUBLICATION_KEY: json.dumps({"publication_version": PUBLICATION_VERSION, "spans": len(rows)})
        },
    )


@pytest.fixture
def enrichment_input(config: ResolvedVideoEmbedConfig) -> inputs.EnrichmentInput:
    """Validate input and register embedding columns exactly as a new invocation does."""
    write_clips(config)
    return inputs.open_enrichment_input(resolve_bindings(config))


@pytest.fixture
def dataset(enrichment_input: inputs.EnrichmentInput) -> lance.LanceDataset:
    """Expose the pinned handle for tests of the lower-level Lance primitives."""
    return enrichment_input.dataset


@pytest.fixture
def work_rows(enrichment_input: inputs.EnrichmentInput) -> list[dict[str, Any]]:
    """Read the pending selection with the real datasource, without starting Ray."""
    datasource = selection_datasource(attempt_all(enrichment_input), columns=READ_COLUMNS)
    return [row for task in datasource.get_read_tasks(2) for batch in task() for row in batch.to_pylist()]


@pytest.fixture
def results(enrichment_input: inputs.EnrichmentInput, work_rows: list[dict[str, Any]], spec: ModelSpec) -> pa.Table:
    """Supply valid normalized output, including one terminal media failure."""
    rows = [terminal_row(row, spec, [1.0, *([0.0] * (spec.dimension - 1))]) for row in work_rows[:-1]]
    rows.append(
        terminal_row(work_rows[-1] | {"error_type": "MediaObjectError", "error_message": "missing"}, spec, None)
    )
    return pa.Table.from_pylist(rows, schema=enrichment_input.bindings.spec.staged_schema)


def _retune(config: ResolvedVideoEmbedConfig) -> ResolvedVideoEmbedConfig:
    execution = config.execution.model_copy(update={"inference_batch_size": 8, "parquet_rows_per_file": 7})
    return config.model_copy(update={"execution": execution})


def _recontract(config: ResolvedVideoEmbedConfig) -> ResolvedVideoEmbedConfig:
    return config.model_copy(update={"model": config.model.model_copy(update={"variant": "448p"})})


# How the shared conformance suite and fresh-cluster scenario drive this recipe.
RECIPE = RecipeUnderTest(pipeline=pipeline, resolve_bindings=resolve_bindings, retune=_retune, recontract=_recontract)


@pytest.fixture
def recipe() -> RecipeUnderTest:
    """Drive the embedding pipeline with its model's bindings; shared publication records provenance."""
    return RECIPE
