# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for video-caption contract and Lance integration tests."""

from collections.abc import Callable
from functools import partial
from pathlib import Path

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.video_caption.bindings import (
    CaptionInputBindings,
    CaptionPublicationBindings,
    CaptionResultBindings,
    CaptionRunBindings,
)
from cosmos_curator.next.recipes.video_caption.contracts import (
    CaptionModelSpec,
    build_terminal_result_row,
    caption_contract_digest,
    resolve_model_spec,
    validate_terminal_row,
    validate_terminal_value,
)
from cosmos_curator.next.recipes.video_caption.inference import (
    MAPPED_INPUT_SCHEMA,
    decompose_row_addresses,
    result_schema,
)
from cosmos_curator.next.recipes.video_split.lance_sink import validate_clip_table
from cosmos_curator.next.recipes.video_split.records import CLIP_SCHEMA


def clip_row(index: int, *, clip_id: str | None = None) -> dict[str, object]:
    """Return one valid canonical video-split row."""
    return {
        "record_schema_version": 1,
        "media_contract_version": 1,
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
    return CaptionRunBindings(
        input=CaptionInputBindings(
            source_columns=("clip_id", "clip_uri", "clip_size_bytes"),
            source_schema=pa.schema([CLIP_SCHEMA.field(name) for name in ("clip_id", "clip_uri", "clip_size_bytes")]),
            mapped_schema=MAPPED_INPUT_SCHEMA,
            project_batch=decompose_row_addresses,
            row_id_column="clip_id",
            media_uri_column="clip_uri",
            media_size_column="clip_size_bytes",
            media_sha256_column=None,
            staged_binding_columns=("clip_id",),
        ),
        result=CaptionResultBindings(
            caption_field_name=caption_spec.caption_field_name,
            metadata_field_name=caption_spec.metadata_field_name,
            fields=caption_spec.fields,
            staged_schema=result_schema(caption_spec),
            digest=caption_digest,
            build_terminal_row=partial(build_terminal_result_row, spec=caption_spec, digest=caption_digest),
            validate_terminal_value=partial(
                validate_terminal_value,
                spec=caption_spec,
                digest=caption_digest,
            ),
            validate_staged_row=partial(
                validate_terminal_row,
                spec=caption_spec,
                digest=caption_digest,
            ),
        ),
        publication=CaptionPublicationBindings(
            validate_base_schema=partial(validate_clip_table, uri="test-clips.lance"),
            protected_field_names=tuple(CLIP_SCHEMA.names),
            base_field_owner="video-split",
            transaction_properties={"kind": "video-caption"},
        ),
    )


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
        )
        return uri, dataset

    return create
