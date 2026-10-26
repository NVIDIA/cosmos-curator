# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source outcomes and canonical spans, with independent source/span identities.

Each discovered source stages one terminal outcome holding its complete span plan,
so a checkpointed source never has partial spans. Both cutting methods build an
outcome from a probed source with ``source_result`` or finish it with
``error_result``; a non-null ``error_stage`` marks a terminal error.
``expand_sources`` produces flat span rows and one placeholder row for each
empty/error outcome. Every source remains present in the canonical table.
"""

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from cosmos_curator.next.media.ffmpeg import VideoMetadata
from cosmos_curator.next.media.spans import Span
from cosmos_curator.next.utils.identity import canonical_digest

# Bounds the span rows built at once from one block of source outcomes.
EXPANSION_ROWS = 100_000
# Bounds one terminal error's message as it is staged and reported.
_ERROR_MESSAGE_CHARS = 8192

INPUT_SCHEMA = pa.schema(
    [
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_uri", pa.string(), nullable=False),
    ]
)
_SOURCE_FIELDS = [*INPUT_SCHEMA, pa.field("stream_index", pa.int32(), nullable=False)]
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
        *_SOURCE_FIELDS,
        *MEDIA_FIELDS,
        pa.field("spans", pa.list_(pa.struct(SPAN_FIELDS)), nullable=False),
        pa.field("error_stage", pa.string()),
        pa.field("error_message", pa.string()),
    ]
)
SPAN_SCHEMA = pa.schema(
    [
        *_SOURCE_FIELDS,
        *MEDIA_FIELDS,
        *(field.with_nullable(nullable=True) for field in SPAN_FIELDS),
        SOURCE_SCHEMA.field("error_stage"),
        SOURCE_SCHEMA.field("error_message"),
    ]
)


def make_source_id(uri: str) -> str:
    """Identify immutable content by normalized URI."""
    return canonical_digest(uri)


def make_span_id(source_id: str, start_ns: int, end_ns: int) -> str:
    """Identify an interval in the first video stream, regardless of cutter settings."""
    return canonical_digest({"source_id": source_id, "stream_index": 0, "start_ns": start_ns, "end_ns": end_ns})


def media_fields(metadata: VideoMetadata, size_bytes: int) -> dict[str, Any]:
    """Describe a probed source in the ``MEDIA_FIELDS`` its outcome and spans carry."""
    return {
        "source_size_bytes": size_bytes,
        "source_duration_ns": metadata.duration_ns,
        "source_width": metadata.width,
        "source_height": metadata.height,
        "source_frame_rate": metadata.frame_rate,
        "source_frame_count": metadata.frame_count,
        "source_video_codec": metadata.video_codec,
    }


def error_details(error_stage: str, error_message: str) -> dict[str, str]:
    """Bound one terminal error's description as it is staged and reported."""
    return {"error_stage": error_stage, "error_message": error_message[:_ERROR_MESSAGE_CHARS]}


def source_result(source: Mapping[str, Any], spans: Sequence[Span]) -> dict[str, Any]:
    """Build a probed source's complete, successful plan, possibly without spans.

    ``source`` carries the work item's fields and its probed ``MEDIA_FIELDS``.
    """
    return {
        "source_id": source["source_id"],
        "source_uri": source["source_uri"],
        "stream_index": 0,
        **{field.name: source[field.name] for field in MEDIA_FIELDS},
        "spans": [
            {
                "span_id": make_span_id(source["source_id"], span.start_ns, span.end_ns),
                "start_ns": span.start_ns,
                "end_ns": span.end_ns,
            }
            for span in spans
        ],
        "error_stage": None,
        "error_message": None,
    }


def error_result(source: Mapping[str, Any], *, error_stage: str, error_message: str) -> dict[str, Any]:
    """Finish a source with a terminal error and no spans, for either cutting method."""
    return {
        "source_id": source["source_id"],
        "source_uri": source["source_uri"],
        "stream_index": 0,
        "spans": [],
        **error_details(error_stage, error_message),
    }


def validate_source_results(batch: pa.RecordBatch) -> None:
    """Accept only terminal outcomes for the first video stream.

    A successful plan, possibly empty, carries its probed duration and no error;
    a terminal error names its stage and message and plans no spans. Operational
    failures never reach a result.
    """
    if not batch.num_rows:
        return
    if not pc.all(pc.equal(batch.column("stream_index"), 0)).as_py():
        msg = "Source results must plan the first video stream"
        raise ValueError(msg)
    has_error = pc.is_valid(batch.column("error_stage"))
    described = pc.equal(has_error, pc.is_valid(batch.column("error_message")))
    planned = pc.greater(pc.list_value_length(batch.column("spans")), 0)
    probed = pc.is_valid(batch.column("source_duration_ns"))
    terminal = pc.if_else(has_error, pc.invert(planned), probed)
    if not pc.all(pc.and_(described, terminal)).as_py():
        msg = "Source results must be probed plans or described terminal errors without spans"
        raise ValueError(msg)


def expand_sources(sources: pa.Table, *, max_rows: int = EXPANSION_ROWS) -> Iterator[pa.Table]:
    """Keep flat spans and one durable outcome for each source without spans.

    A null span ID identifies an empty or error outcome; ``error_stage`` marks
    the errors. Every source remains discoverable in current Lance rows after
    history and checkpoints are pruned.
    """
    outcomes = sources.filter(pc.equal(pc.list_value_length(sources["spans"]), 0))
    span_names = {field.name for field in SPAN_FIELDS}
    for offset in range(0, outcomes.num_rows, max_rows):
        chunk = outcomes.slice(offset, max_rows)
        yield pa.Table.from_arrays(
            [
                pa.nulls(chunk.num_rows, type=field.type) if field.name in span_names else chunk[field.name]
                for field in SPAN_SCHEMA
            ],
            schema=SPAN_SCHEMA,
        )
    spans = pc.list_flatten(sources["spans"])
    parents = pc.list_parent_indices(sources["spans"])
    for offset in range(0, len(spans), max_rows):
        items = spans.slice(offset, max_rows)
        indices = parents.slice(offset, max_rows)
        yield pa.Table.from_arrays(
            [
                pc.struct_field(items, field.name)
                if field.name in span_names
                else pc.take(sources[field.name], indices)
                for field in SPAN_SCHEMA
            ],
            schema=SPAN_SCHEMA,
        )
