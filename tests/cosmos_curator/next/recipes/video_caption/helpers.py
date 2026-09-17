# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared caption bindings and competing table writes for recipe tests."""

from functools import partial

import pyarrow as pa

from cosmos_curator.next.recipes.video_caption.bindings import (
    CaptionInputBindings,
    CaptionPublicationBindings,
    CaptionResultBindings,
    CaptionRunBindings,
)
from cosmos_curator.next.recipes.video_caption.contracts import (
    CaptionModelSpec,
    build_terminal_result_row,
    validate_terminal_row,
    validate_terminal_value,
)
from cosmos_curator.next.recipes.video_caption.inference import (
    MAPPED_INPUT_SCHEMA,
    decompose_row_addresses,
    result_schema,
)
from cosmos_curator.next.recipes.video_transcode.contracts import CLIP_SCHEMA, validate_clip_table
from cosmos_curator.next.utils import lance_fragment_recovery


def caption_run_bindings(spec: CaptionModelSpec, digest: str) -> CaptionRunBindings:
    """Build clip bindings for caption tests and upstream compatibility tests."""
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
