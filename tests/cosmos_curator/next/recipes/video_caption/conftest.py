# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for video-caption contract and Lance integration tests."""

import json
from collections.abc import Callable
from pathlib import Path

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.video_caption.bindings import CaptionRunBindings
from cosmos_curator.next.recipes.video_caption.contracts import (
    CaptionModelSpec,
    caption_contract_digest,
    resolve_model_spec,
)
from cosmos_curator.next.recipes.video_transcode.contracts import CLIP_SCHEMA, PUBLICATION_KEY, PUBLICATION_VERSION
from tests.cosmos_curator.next.recipes.video_caption.helpers import caption_run_bindings


def clip_row(index: int, *, clip_id: str | None = None) -> dict[str, object]:
    """Return one valid canonical video-transcode row."""
    return {
        "record_schema_version": 1,
        "cut_contract_id": "cut-contract",
        "span_id": f"span-{index}",
        "stream_index": 0,
        "transcode_contract_id": "transcode-contract",
        "source_id": f"source-{index}",
        "source_uri": f"s3://input/source-{index}.mp4",
        "source_size_bytes": 1_000 + index,
        "source_duration_ns": 10_000_000_000,
        "source_width": 1_920,
        "source_height": 1_080,
        "source_frame_rate": 30.0,
        "source_frame_count": 300,
        "source_video_codec": "h264",
        "start_ns": 0,
        "end_ns": 10_000_000_000,
        "clip_id": clip_id or f"clip-{index}",
        "clip_uri": f"s3://output/clips/clip-{index}.mp4",
        "clip_size_bytes": 500 + index,
        "clip_duration_ns": 10_000_000_000,
        "clip_width": 1_920,
        "clip_height": 1_080,
        "clip_frame_rate": 30.0,
        "clip_frame_count": 300,
        "clip_video_codec": "h264",
    }


@pytest.fixture
def caption_spec() -> CaptionModelSpec:
    """Return the default pinned caption model contract."""
    return resolve_model_spec("qwen3_8_27b_fp8")


@pytest.fixture
def caption_digest(caption_spec: CaptionModelSpec) -> str:
    """Return the default contract digest."""
    return caption_contract_digest(caption_spec)


@pytest.fixture
def caption_bindings(caption_spec: CaptionModelSpec, caption_digest: str) -> CaptionRunBindings:
    """Return the clip recipe's explicit private run bindings."""
    return caption_run_bindings(caption_spec, caption_digest)


@pytest.fixture
def clip_row_factory() -> Callable[..., dict[str, object]]:
    """Expose the canonical row builder without importing pytest's conftest module."""
    return clip_row


@pytest.fixture
def clip_dataset_factory(
    tmp_path: Path,
) -> Callable[..., tuple[str, lance.LanceDataset]]:
    """Create valid v2.2 clip tables under unique test paths."""
    created = 0

    def create(
        *,
        rows: list[dict[str, object]] | None = None,
        clip_ids: list[str] | None = None,
        count: int = 2,
        rows_per_fragment: int = 2,
    ) -> tuple[str, lance.LanceDataset]:
        nonlocal created
        created += 1
        values = (
            rows
            if rows is not None
            else [clip_row(index, clip_id=clip_ids[index] if clip_ids is not None else None) for index in range(count)]
        )
        uri = str(tmp_path / f"clips-{created}.lance")
        dataset = lance.write_dataset(
            pa.Table.from_pylist(values, schema=CLIP_SCHEMA),
            uri,
            max_rows_per_file=rows_per_fragment,
            data_storage_version="2.2",
            transaction_properties={
                PUBLICATION_KEY: json.dumps({"publication_version": PUBLICATION_VERSION, "spans": len(values)})
            },
        )
        return uri, dataset

    return create
