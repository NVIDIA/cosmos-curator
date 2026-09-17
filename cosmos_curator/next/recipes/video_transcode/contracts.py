# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Span-addressed result rows and the canonical transcoded clip schema."""

import json
from typing import Any

import lance
import pyarrow as pa

from cosmos_curator.next.recipes.video_cut.contracts import SPAN_FIELDS, SPAN_SCHEMA
from cosmos_curator.next.utils.identity import canonical_digest

PUBLICATION_KEY = b"video_transcode.publication"
PUBLICATION_VERSION = 2
CLIP_FIELDS = pa.schema(
    [
        pa.field("transcode_contract_id", pa.string(), nullable=False),
        pa.field("clip_id", pa.string(), nullable=False),
        pa.field("clip_uri", pa.large_string(), nullable=False),
        pa.field("clip_size_bytes", pa.int64(), nullable=False),
        pa.field("clip_duration_ns", pa.int64(), nullable=False),
        pa.field("clip_width", pa.int32(), nullable=False),
        pa.field("clip_height", pa.int32(), nullable=False),
        pa.field("clip_frame_rate", pa.float64(), nullable=False),
        pa.field("clip_frame_count", pa.int64()),
        pa.field("clip_video_codec", pa.string(), nullable=False),
    ]
)
# Enrichment fields remain nullable at the schema level, as required by
# video-cut's producer validation. Publication requires complete values.
CLIP_FIELDS = pa.schema(
    [
        field.with_nullable(nullable=True).with_metadata({b"cosmos_curator.owner": b"video-transcode"})
        for field in CLIP_FIELDS
    ]
)
CLIP_SCHEMA = pa.schema([*SPAN_SCHEMA, *CLIP_FIELDS])
ADDRESS_FIELDS = [
    pa.field("fragment_id", pa.int64(), nullable=False),
    pa.field("row_offset", pa.int64(), nullable=False),
]
SPAN_WORK_SCHEMA = pa.schema([*SPAN_FIELDS, *ADDRESS_FIELDS])
SOURCE_WORK_SCHEMA = pa.schema(
    [
        SPAN_SCHEMA.field("source_id"),
        SPAN_SCHEMA.field("source_uri"),
        SPAN_SCHEMA.field("source_size_bytes"),
        pa.field("spans", pa.large_list(pa.struct(SPAN_WORK_SCHEMA)), nullable=False),
    ]
)
RESULT_SCHEMA = pa.schema([*ADDRESS_FIELDS, SPAN_SCHEMA.field("span_id"), *CLIP_FIELDS])
PAYLOAD_SCHEMA = pa.schema([*RESULT_SCHEMA, pa.field("clip_bytes", pa.large_binary(), nullable=False)])


def make_clip_id(span_id: str, contract_id: str) -> str:
    """Identify one encoded rendition independently of batching and placement."""
    return canonical_digest({"span_id": span_id, "transcode_contract_id": contract_id})


def completed_publication(dataset: lance.LanceDataset) -> dict[str, Any] | None:
    """Read the latest finite-run receipt, falling back to the legacy schema record.

    This describes that run's input snapshot, not necessarily every row in the
    current table. Transcode scheduling uses fragment values instead.
    """
    for version in range(int(dataset.version), 0, -1):
        transaction = dataset.read_transaction(version)
        if transaction is None:
            continue
        encoded = (transaction.transaction_properties or {}).get(PUBLICATION_KEY.decode())
        if encoded is not None:
            record = json.loads(encoded)
            if not isinstance(record, dict) or record.get("publication_version") != PUBLICATION_VERSION:
                msg = "Unsupported video-transcode publication record"
                raise ValueError(msg)
            return record
    encoded = (dataset.schema.metadata or {}).get(PUBLICATION_KEY)
    if encoded is None:
        return None
    record = json.loads(encoded)
    if not isinstance(record, dict) or record.get("publication_version") != 1:
        msg = "Unsupported video-transcode publication record"
        raise ValueError(msg)
    return record


def validate_clip_table(dataset: lance.LanceDataset, *, uri: str) -> None:
    """Require completed video-transcode output while preserving enrichment fields."""
    for field in CLIP_SCHEMA:
        if field.name not in dataset.schema.names or not dataset.schema.field(field.name).equals(
            field, check_metadata=True
        ):
            msg = f"Lance table {uri} does not have the video-transcode schema: {field.name}"
            raise ValueError(msg)
    record = completed_publication(dataset)
    if record is None:
        msg = f"Lance table {uri} has no completed video-transcode publication"
        raise ValueError(msg)
    if record.get("spans") != dataset.count_rows():
        msg = f"Lance table {uri} changed after video-transcode publication"
        raise ValueError(msg)
