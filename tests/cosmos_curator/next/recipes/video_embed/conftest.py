# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small canonical clip tables and typed visual results, without a Ray runtime."""

import json
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.video_cut.contracts import make_source_id, make_span_id
from cosmos_curator.next.recipes.video_embed.config import ResolvedVideoEmbedConfig, resolve_config_data
from cosmos_curator.next.recipes.video_embed.contracts import ModelSpec, resolve_model_spec, terminal_row
from cosmos_curator.next.recipes.video_embed.pipeline import ClipDatasource
from cosmos_curator.next.recipes.video_embed.state import open_input, pending_fragment_ids
from cosmos_curator.next.recipes.video_transcode.contracts import CLIP_SCHEMA, make_clip_id


@pytest.fixture
def config(tmp_path: Path) -> ResolvedVideoEmbedConfig:
    """Use local media and compact recovery groups."""
    return resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-embed",
            "input": {"media_root": str(tmp_path / "media")},
            "execution": {"parquet_rows_per_file": 1, "lance_read_batch_size": 2, "media_attempts": 1},
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
    schema = CLIP_SCHEMA.with_metadata(
        {b"video_transcode.publication": json.dumps({"publication_version": 1, "spans": len(rows)}).encode()}
    )
    return lance.write_dataset(
        pa.Table.from_pylist(rows, schema=schema),
        config.input.clips_lance_uri,
        max_rows_per_file=2,
        data_storage_version="2.2",
    )


@pytest.fixture
def dataset(config: ResolvedVideoEmbedConfig, spec: ModelSpec) -> lance.LanceDataset:
    """Register embedding columns exactly as a new invocation does."""
    write_clips(config)
    return open_input(config, spec)


@pytest.fixture
def work_rows(dataset: lance.LanceDataset, config: ResolvedVideoEmbedConfig, spec: ModelSpec) -> list[dict[str, Any]]:
    """Read the real datasource directly without starting Ray."""
    return [
        row
        for task in ClipDatasource(dataset, config, pending_fragment_ids(dataset, spec)).get_read_tasks(2)
        for batch in task()
        for row in batch.to_pylist()
    ]


@pytest.fixture
def results(work_rows: list[dict[str, Any]], spec: ModelSpec) -> pa.Table:
    """Supply valid normalized output, including one terminal media failure."""
    rows = [terminal_row(row, spec, [1.0, *([0.0] * (spec.dimension - 1))]) for row in work_rows[:-1]]
    rows.append(
        terminal_row(work_rows[-1] | {"error_type": "MediaObjectError", "error_message": "missing"}, spec, None)
    )
    return pa.Table.from_pylist(rows, schema=spec.result_schema)
