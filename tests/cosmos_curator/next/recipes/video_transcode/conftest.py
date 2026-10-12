# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small canonical span tables and deterministic transcode results."""

from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.enrichment.contracts import EnrichmentInput, open_enrichment_input
from cosmos_curator.next.recipes.video_cut.config import FixedStrideConfig
from cosmos_curator.next.recipes.video_cut.contracts import make_source_id, make_span_id, span_schema
from cosmos_curator.next.recipes.video_transcode.bindings import resolve_bindings
from cosmos_curator.next.recipes.video_transcode.config import ResolvedVideoTranscodeConfig, resolve_config_data
from cosmos_curator.next.recipes.video_transcode.contracts import make_clip_id


@pytest.fixture
def config(tmp_path: Path) -> ResolvedVideoTranscodeConfig:
    """Use local media and one-span durability for recovery tests."""
    return resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-transcode",
            "input": {"media_root": str(tmp_path / "media")},
            "execution": {
                "transcode_cpus": 1.0,
                "transcode_memory_gib": 0.5,
                "publication_memory_gib": 0.25,
                "source_partitions": 2,
                "parquet_rows_per_file": 1,
                "ffmpeg_batch_size": 2,
                "transcode_attempts": 1,
                "probe_attempts": 1,
                "storage_attempts": 1,
            },
        }
    )


@pytest.fixture
def source_path(tmp_path: Path) -> Path:
    """Provide immutable bytes; media unit tests replace FFmpeg at its boundary."""
    path = tmp_path / "source.mp4"
    path.write_bytes(b"source-video")
    return path


@pytest.fixture
def span_rows(source_path: Path) -> list[dict[str, Any]]:
    """Put four spans of the same source across two physical fragments."""
    source_id = make_source_id(str(source_path))
    return [
        {
            "record_schema_version": 1,
            "cut_contract_id": FixedStrideConfig().contract_id,
            "source_id": source_id,
            "source_uri": str(source_path),
            "stream_index": 0,
            "source_size_bytes": source_path.stat().st_size,
            "source_duration_ns": 4_000_000_000,
            "source_width": 64,
            "source_height": 64,
            "source_frame_rate": 10.0,
            "source_frame_count": 40,
            "source_video_codec": "h264",
            "span_id": make_span_id(source_id, index * 1_000_000_000, (index + 1) * 1_000_000_000),
            "start_ns": index * 1_000_000_000,
            "end_ns": (index + 1) * 1_000_000_000,
        }
        for index in range(4)
    ]


@pytest.fixture
def enrichment_input(config: ResolvedVideoTranscodeConfig, span_rows: list[dict[str, Any]]) -> EnrichmentInput:
    """Create canonical span fragments; transcode does not require cut receipts."""
    schema = span_schema(FixedStrideConfig())
    lance.write_dataset(
        pa.Table.from_pylist(span_rows, schema=schema),
        config.input.spans_lance_uri,
        max_rows_per_file=2,
        data_storage_version="2.2",
    )
    return open_enrichment_input(resolve_bindings(config))


@pytest.fixture
def dataset(enrichment_input: EnrichmentInput) -> lance.LanceDataset:
    """Expose the pinned handle for tests of the lower-level Lance primitives."""
    return enrichment_input.dataset


@pytest.fixture
def staged_schema(config: ResolvedVideoTranscodeConfig) -> pa.Schema:
    """Use the staged clip rows the recipe's enrichment spec defines, as processing receives them."""
    return resolve_bindings(config).spec.staged_schema


@pytest.fixture
def results(config: ResolvedVideoTranscodeConfig, dataset: lance.LanceDataset, staged_schema: pa.Schema) -> pa.Table:
    """Build completed results with exact original span addresses."""
    rows = []
    for fragment in dataset.get_fragments():
        for offset, span_id in enumerate(fragment.to_table(columns=["span_id"])["span_id"].to_pylist()):
            clip_id = make_clip_id(span_id, config.transcode.contract_id)
            rows.append(
                {
                    "fragment_id": fragment.fragment_id,
                    "row_offset": offset,
                    "span_id": span_id,
                    "transcode_contract_id": config.transcode.contract_id,
                    "clip_id": clip_id,
                    "clip_uri": f"{config.input.media_root}/clips/{clip_id}.mp4",
                    "clip_size_bytes": 50,
                    "clip_duration_ns": 1_000_000_000,
                    "clip_width": 64,
                    "clip_height": 64,
                    "clip_frame_rate": 10.0,
                    "clip_frame_count": 10,
                    "clip_video_codec": "h264",
                }
            )
    return pa.Table.from_pylist(rows, schema=staged_schema)
