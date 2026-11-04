# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clip-shaped two-phase bindings and competing table writes for the engine's tests.

The multimodal recipes are the engine's production users; these tests exercise it
with video-caption's clip table and caption contract, which the engine served
before video-caption moved to shared enrichment.
"""

from functools import partial

import pyarrow as pa

from cosmos_curator.next.legacy.caption_fragment_recovery.bindings import (
    CaptionInputBindings,
    CaptionPublicationBindings,
    CaptionResultBindings,
    CaptionRunBindings,
)
from cosmos_curator.next.recipes.video_caption.contracts import (
    DEFAULT_PROMPT,
    CaptionModelSpec,
    build_terminal_result_row,
    validate_terminal_row,
    validate_terminal_value,
)
from cosmos_curator.next.recipes.video_transcode.contracts import CLIP_SCHEMA, validate_clip_table
from cosmos_curator.next.utils import lance_fragment_recovery

_ROW_ADDRESS_FRAGMENT_SHIFT = 32
_ROW_ADDRESS_OFFSET_MASK = (1 << _ROW_ADDRESS_FRAGMENT_SHIFT) - 1
MAPPED_INPUT_SCHEMA = pa.schema(
    [
        pa.field("fragment_id", pa.int64(), nullable=False),
        pa.field("row_offset", pa.int64(), nullable=False),
        pa.field("clip_id", pa.string(), nullable=False),
        pa.field("clip_uri", pa.large_string(), nullable=False),
        pa.field("clip_size_bytes", pa.int64(), nullable=False),
    ]
)


def result_schema(spec: CaptionModelSpec) -> pa.Schema:
    """Return the staged clip result schema."""
    return pa.schema(
        [
            pa.field("fragment_id", pa.int64(), nullable=False),
            pa.field("row_offset", pa.int64(), nullable=False),
            pa.field("clip_id", pa.string(), nullable=False),
            *spec.fields,
        ]
    )


def decompose_row_addresses(batch: pa.Table) -> pa.Table:
    """Turn Lance's global row address into physical fragment and local offset."""
    required = {"_rowaddr", "clip_id", "clip_uri", "clip_size_bytes"}
    missing = sorted(required - set(batch.schema.names))
    if missing:
        msg = f"Lance caption read is missing required column(s): {', '.join(missing)}"
        raise ValueError(msg)
    rows = []
    for row in batch.select(["_rowaddr", "clip_id", "clip_uri", "clip_size_bytes"]).to_pylist():
        row_address = int(row["_rowaddr"])
        rows.append(
            {
                "fragment_id": row_address >> _ROW_ADDRESS_FRAGMENT_SHIFT,
                "row_offset": row_address & _ROW_ADDRESS_OFFSET_MASK,
                "clip_id": row["clip_id"],
                "clip_uri": row["clip_uri"],
                "clip_size_bytes": row["clip_size_bytes"],
            }
        )
    return pa.Table.from_pylist(rows, schema=MAPPED_INPUT_SCHEMA)


def caption_run_bindings(spec: CaptionModelSpec, digest: str) -> CaptionRunBindings:
    """Build clip bindings for the engine's tests."""
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
            skip_null_ids=True,
        ),
        result=CaptionResultBindings(
            caption_field_name=spec.caption_field_name,
            metadata_field_name=spec.metadata_field_name,
            fields=spec.fields,
            staged_schema=result_schema(spec),
            digest=digest,
            build_terminal_row=partial(build_terminal_result_row, spec=spec, digest=digest),
            validate_terminal_value=partial(validate_terminal_value, spec=spec, digest=digest),
            validate_staged_row=partial(validate_terminal_row, spec=spec, digest=digest),
        ),
        publication=CaptionPublicationBindings(
            validate_base_schema=partial(validate_clip_table, uri="test-clips.lance"),
            protected_field_names=tuple(CLIP_SCHEMA.names),
            base_field_owner="video-transcode",
            transaction_properties={"kind": "video-caption"},
        ),
        prompt_template=DEFAULT_PROMPT,
    )


def write_clip_fragment(clips: pa.Table, *, uri: str, storage_profile: str) -> str | None:
    """Stage a test writer's additional rows."""
    return lance_fragment_recovery.write_row_fragment(
        clips,
        uri=uri,
        storage_profile=storage_profile,
        schema=CLIP_SCHEMA,
        id_column="clip_id",
        data_storage_version="2.2",
    )


def append_clip_fragment(candidate_json: str, *, uri: str, storage_profile: str, attempts: int) -> int:
    """Publish a competing append for reconciliation tests."""
    return lance_fragment_recovery.append_row_fragment(
        candidate_json,
        uri=uri,
        storage_profile=storage_profile,
        schema=CLIP_SCHEMA,
        id_column="clip_id",
        kind="test-writer",
        operation="append-clips",
        data_storage_version="2.2",
        attempts=attempts,
    )
