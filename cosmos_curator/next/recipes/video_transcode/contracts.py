# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Span-addressed result rows, the canonical transcoded clip schema, and the receipt video-caption reads."""

import json
import math
from typing import Any

import lance
import pyarrow as pa

from cosmos_curator.next.core.lance_datasource import ROW_ADDRESS_FIELDS
from cosmos_curator.next.enrichment.contracts import owned_fields
from cosmos_curator.next.enrichment.transactions import find_transaction
from cosmos_curator.next.recipes.video_cut.contracts import SPAN_FIELDS, SPAN_SCHEMA
from cosmos_curator.next.utils.identity import canonical_digest

PUBLICATION_KEY = "video_transcode.publication"
PUBLICATION_VERSION = 1
# Enrichment fields remain nullable at the schema level, as required by
# video-cut's producer validation. Publication requires complete values.
CLIP_FIELDS = owned_fields(
    pa.schema(
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
    ),
    owner="video-transcode",
    field_set="clip",
)
CLIP_SCHEMA = pa.schema([*SPAN_SCHEMA, *CLIP_FIELDS])
# Span columns processing reads besides span_id and row addresses.
READ_COLUMNS = ("source_id", "source_uri", "source_size_bytes", "stream_index", "start_ns", "end_ns")
SPAN_WORK_SCHEMA = pa.schema([*SPAN_FIELDS, *ROW_ADDRESS_FIELDS])
SOURCE_WORK_SCHEMA = pa.schema(
    [
        SPAN_SCHEMA.field("source_id"),
        SPAN_SCHEMA.field("source_uri"),
        SPAN_SCHEMA.field("source_size_bytes"),
        pa.field("spans", pa.large_list(pa.struct(SPAN_WORK_SCHEMA)), nullable=False),
    ]
)


def make_clip_id(span_id: str, contract_id: str) -> str:
    """Identify one encoded rendition independently of batching and placement."""
    return canonical_digest({"span_id": span_id, "transcode_contract_id": contract_id})


def completed_publication(dataset: lance.LanceDataset) -> dict[str, Any] | None:
    """Read the latest finite-run receipt from Lance transaction properties.

    This describes that run's input snapshot, not necessarily every row in the
    current table. Transcode scheduling uses fragment values instead.
    """
    transaction = find_transaction(dataset, property_key=PUBLICATION_KEY)
    if transaction is None:
        return None
    record = json.loads((transaction.transaction_properties or {})[PUBLICATION_KEY])
    if not isinstance(record, dict) or record.get("publication_version") != PUBLICATION_VERSION:
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


def _validate_result(row: dict[str, Any], *, contract_id: str, media_root: str) -> None:
    clip_id = make_clip_id(row["span_id"], contract_id)
    if (
        row["transcode_contract_id"] != contract_id
        or row["clip_id"] != clip_id
        or row["clip_uri"] != f"{media_root}/clips/{clip_id}.mp4"
        or any(
            row[name] is None or row[name] <= 0
            for name in ("clip_size_bytes", "clip_duration_ns", "clip_width", "clip_height", "clip_frame_rate")
        )
        or not math.isfinite(row["clip_frame_rate"])
        or row["clip_video_codec"] != "h264"
        or (row["clip_frame_count"] is not None and row["clip_frame_count"] < 0)
    ):
        msg = f"Invalid completed transcode result for {row['span_id']} (contract, destination, or clip metadata)"
        raise ValueError(msg)


def validate_terminal_batch(batch: pa.RecordBatch, *, contract_id: str, media_root: str) -> None:
    """Validate compact clip metadata against the result contract in the publisher's bounded batches."""
    for row in batch.to_pylist():
        _validate_result(row, contract_id=contract_id, media_root=media_root)
