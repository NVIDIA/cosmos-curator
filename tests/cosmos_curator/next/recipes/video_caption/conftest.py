# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small canonical clip tables and typed caption results, without a Ray runtime."""

from functools import partial
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.enrichment import inputs
from cosmos_curator.next.enrichment.execution import selection_datasource
from cosmos_curator.next.recipes.video_caption import pipeline
from cosmos_curator.next.recipes.video_caption.bindings import resolve_bindings
from cosmos_curator.next.recipes.video_caption.config import ResolvedVideoCaptionConfig, resolve_config_data
from cosmos_curator.next.recipes.video_caption.contracts import (
    DEFAULT_PROMPT,
    READ_COLUMNS,
    CaptionModelSpec,
    build_terminal_result_row,
    caption_contract_digest,
    resolve_model_spec,
)
from cosmos_curator.next.recipes.video_caption.inference import CaptionRequest, MediaColumns
from cosmos_curator.next.recipes.video_cut.contracts import make_source_id, make_span_id
from cosmos_curator.next.recipes.video_transcode.contracts import CLIP_SCHEMA, make_clip_id
from tests.cosmos_curator.next.enrichment.conformance import RecipeUnderTest
from tests.cosmos_curator.next.enrichment.helpers import attempt_all


@pytest.fixture
def config(tmp_path: Path) -> ResolvedVideoCaptionConfig:
    """Use local media and compact recovery groups."""
    return resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-caption",
            "input": {"media_root": str(tmp_path / "media")},
            "model": {"variant": "qwen3_8_27b_fp8"},
            "execution": {
                "rows_per_checkpoint": 1,
                "lance_read_batch_size": 2,
                "media_batch_size": 1,
                "media_attempts": 1,
                "publication_memory_gib": 0.25,
            },
        }
    )


@pytest.fixture
def caption_spec(config: ResolvedVideoCaptionConfig) -> CaptionModelSpec:
    """Resolve the same pinned model used by the recipe."""
    return resolve_model_spec(config.model.variant)


@pytest.fixture
def caption_digest(caption_spec: CaptionModelSpec) -> str:
    """Return the default contract digest."""
    return caption_contract_digest(caption_spec)


@pytest.fixture
def caption_request(
    config: ResolvedVideoCaptionConfig, caption_spec: CaptionModelSpec, caption_digest: str
) -> CaptionRequest:
    """Describe clip captioning to the shared plan as the recipe does."""
    return CaptionRequest(
        media=MediaColumns(uri="clip_uri", size="clip_size_bytes"),
        identity_columns=("clip_id",),
        prompt_template=DEFAULT_PROMPT,
        build_terminal_row=partial(build_terminal_result_row, spec=caption_spec, digest=caption_digest),
        result_schema=resolve_bindings(config).spec.result_schema,
    )


def clip_rows(config: ResolvedVideoCaptionConfig, *, payload: bytes = b"mp4", count: int = 4) -> list[dict[str, Any]]:
    """Return completed clip rows whose media files exist with their recorded sizes."""
    root = Path(config.input.media_root)
    (root / "clips").mkdir(parents=True, exist_ok=True)
    source_uri = str(root / "source.mp4")
    source_id = make_source_id(source_uri)
    rows = []
    for index in range(count):
        start = index * 1_000_000_000
        span_id = make_span_id(source_id, start, start + 1_000_000_000)
        clip_id = make_clip_id(span_id, "transcode-contract")
        clip_path = root / "clips" / f"{clip_id}.mp4"
        clip_path.write_bytes(payload)
        rows.append(
            {
                "source_id": source_id,
                "source_uri": source_uri,
                "source_size_bytes": 100,
                "source_duration_ns": count * 1_000_000_000,
                "source_width": 64,
                "source_height": 48,
                "source_frame_rate": 8.0,
                "source_frame_count": count * 8,
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
    return rows


def write_clips(config: ResolvedVideoCaptionConfig, *, payload: bytes = b"mp4") -> lance.LanceDataset:
    """Write four immutable clip rows across two fragments."""
    return lance.write_dataset(
        pa.Table.from_pylist(clip_rows(config, payload=payload), schema=CLIP_SCHEMA),
        config.input.clips_lance_uri,
        max_rows_per_file=2,
        data_storage_version="2.2",
    )


@pytest.fixture
def enrichment_input(config: ResolvedVideoCaptionConfig) -> inputs.EnrichmentInput:
    """Validate input and register caption columns exactly as a new invocation does."""
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
def results(
    enrichment_input: inputs.EnrichmentInput,
    work_rows: list[dict[str, Any]],
    caption_spec: CaptionModelSpec,
    caption_digest: str,
) -> pa.Table:
    """Supply valid captions, including one truncated caption and one terminal media failure."""
    terminal = partial(build_terminal_result_row, spec=caption_spec, digest=caption_digest)
    rows = [
        terminal(
            row,
            caption=f"caption for {row['clip_id']}",
            status="success",
            prompt_token_count=4,
            generated_token_count=8,
            error_type=None,
            error_message=None,
        )
        for row in work_rows[:-2]
    ]
    rows.append(
        terminal(
            work_rows[-2],
            caption="long caption",
            status="truncated",
            prompt_token_count=4,
            generated_token_count=None,
            error_type=None,
            error_message=None,
        )
    )
    rows.append(
        terminal(
            work_rows[-1],
            caption=None,
            status="error",
            prompt_token_count=None,
            generated_token_count=None,
            error_type="ItemMediaError",
            error_message="MP4 object is unavailable",
        )
    )
    return pa.Table.from_pylist(rows, schema=enrichment_input.bindings.spec.result_schema)


def _retune(config: ResolvedVideoCaptionConfig) -> ResolvedVideoCaptionConfig:
    execution = config.execution.model_copy(update={"inference_batch_size": 8, "rows_per_checkpoint": 7})
    return config.model_copy(update={"execution": execution})


def _recontract(config: ResolvedVideoCaptionConfig) -> ResolvedVideoCaptionConfig:
    return config.model_copy(update={"model": config.model.model_copy(update={"variant": "qwen3_8_27b"})})


# How the shared conformance suite and fresh-cluster scenario drive this recipe.
RECIPE = RecipeUnderTest(pipeline=pipeline, resolve_bindings=resolve_bindings, retune=_retune, recontract=_recontract)


@pytest.fixture
def recipe() -> RecipeUnderTest:
    """Drive the caption pipeline with its model's bindings; shared publication records provenance."""
    return RECIPE
