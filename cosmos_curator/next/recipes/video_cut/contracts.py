# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source outcomes and canonical spans, with independent source/span identities."""

import json

import pyarrow as pa

from cosmos_curator.next.recipes.video_cut.config import FixedStrideConfig
from cosmos_curator.next.utils.identity import canonical_digest

INPUT_SCHEMA = pa.schema(
    [
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_uri", pa.string(), nullable=False),
    ]
)
_PROVENANCE_FIELDS = [
    pa.field("record_schema_version", pa.int32(), nullable=False),
    pa.field("cut_contract_id", pa.string(), nullable=False),
    *INPUT_SCHEMA,
    pa.field("stream_index", pa.int32(), nullable=False),
]
MEDIA_FIELDS = [
    pa.field("source_size_bytes", pa.int64()),
    pa.field("source_duration_ns", pa.int64()),
    pa.field("source_width", pa.int32()),
    pa.field("source_height", pa.int32()),
    pa.field("source_frame_rate", pa.float64()),
    pa.field("source_frame_count", pa.int64()),
    pa.field("source_video_codec", pa.string()),
]
SPAN_FIELDS = [
    pa.field("span_id", pa.string(), nullable=False),
    pa.field("start_ns", pa.int64(), nullable=False),
    pa.field("end_ns", pa.int64(), nullable=False),
]
SOURCE_SCHEMA = pa.schema(
    [
        *_PROVENANCE_FIELDS,
        *MEDIA_FIELDS,
        pa.field("spans", pa.list_(pa.struct(SPAN_FIELDS)), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("error_stage", pa.string()),
        pa.field("error_message", pa.string()),
    ]
)
SPAN_SCHEMA = pa.schema([*_PROVENANCE_FIELDS, *MEDIA_FIELDS, *SPAN_FIELDS])


def make_source_id(uri: str) -> str:
    """Identify immutable content by normalized URI."""
    return canonical_digest(uri)


def make_span_id(source_id: str, start_ns: int, end_ns: int) -> str:
    """Identify an interval in the first video stream, regardless of cutter settings."""
    return canonical_digest({"source_id": source_id, "stream_index": 0, "start_ns": start_ns, "end_ns": end_ns})


def span_schema(split: FixedStrideConfig, input_snapshot_id: str) -> pa.Schema:
    """Bind the canonical table to its cutting contract and frozen source set."""
    return SPAN_SCHEMA.with_metadata(
        {
            "video_cut.contract": json.dumps(split.cutting_contract, sort_keys=True),
            "video_cut.input_snapshot": input_snapshot_id,
        }
    )
