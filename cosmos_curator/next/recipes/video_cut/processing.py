# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-local probing and fixed-stride planning; no recovery decisions."""

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
from botocore.exceptions import ClientError

from cosmos_curator.core.utils.storage.s3_client import S3Prefix
from cosmos_curator.next.media.ffmpeg import MediaError, ProbeError, VideoMetadata, probe_video_source
from cosmos_curator.next.media.spans import Span, fixed_stride_spans, seconds_to_nanoseconds
from cosmos_curator.next.recipes.video_cut.config import ExecutionConfig, FixedStrideConfig, SplitConfig
from cosmos_curator.next.recipes.video_cut.contracts import SOURCE_SCHEMA, make_source_id, make_span_id
from cosmos_curator.next.recipes.video_cut.storage import s3_client


def source_result(uri: str, metadata: VideoMetadata, size_bytes: int, split: FixedStrideConfig) -> dict[str, Any]:
    """Build one complete source plan, including a successful empty span list."""
    spans = fixed_stride_spans(
        metadata.duration_ns,
        duration_ns=seconds_to_nanoseconds(split.duration_s),
        stride_ns=seconds_to_nanoseconds(split.stride_s),
        min_duration_ns=seconds_to_nanoseconds(split.min_duration_s),
    )
    return result_from_spans(uri, metadata, size_bytes, split, spans)


def result_from_spans(
    uri: str, metadata: VideoMetadata, size_bytes: int, split: SplitConfig, spans: Sequence[Span]
) -> dict[str, Any]:
    """Build the common source result without retaining decoded or encoded media."""
    result = source_envelope(uri, split)
    result.update(
        source_size_bytes=size_bytes,
        source_duration_ns=metadata.duration_ns,
        source_width=metadata.width,
        source_height=metadata.height,
        source_frame_rate=metadata.frame_rate,
        source_frame_count=metadata.frame_count,
        source_video_codec=metadata.video_codec,
        spans=[
            {
                "span_id": make_span_id(result["source_id"], span.start_ns, span.end_ns),
                "start_ns": span.start_ns,
                "end_ns": span.end_ns,
            }
            for span in spans
        ],
    )
    return result


def source_envelope(uri: str, split: SplitConfig) -> dict[str, Any]:
    """Create a successful empty source plan, also used to record terminal errors."""
    return {
        "record_schema_version": 1,
        "cut_contract_id": split.contract_id,
        "source_id": make_source_id(uri),
        "source_uri": uri,
        "stream_index": 0,
        "status": "success",
        "spans": [],
        "error_stage": None,
        "error_message": None,
    }


def _probe(uri: str, execution: ExecutionConfig) -> tuple[VideoMetadata, int]:
    client = s3_client(execution.storage_profile)
    prefix = S3Prefix(uri)
    timeout = execution.probe_timeout_s
    if execution.probe_mode == "auto":
        size = int(client.s3.head_object(Bucket=prefix.bucket, Key=prefix.prefix)["ContentLength"])
        signed = client.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": prefix.bucket, "Key": prefix.prefix},
            ExpiresIn=max(900, timeout * 2),
        )
        try:
            return probe_video_source(signed, timeout_s=timeout), size
        except (MediaError, ProbeError):
            # Confirm remote failures against a complete download before deciding
            # whether the source is unusable or probing must be retried.
            pass
    with tempfile.TemporaryDirectory(prefix="curator_video_cut_") as temporary:
        source = Path(temporary) / "source.mp4"
        client.download_to_path(prefix, source)
        return probe_video_source(source, timeout_s=timeout), source.stat().st_size


def plan_sources(batch: pa.Table, *, split: FixedStrideConfig, execution: ExecutionConfig) -> pa.Table:
    """Produce exactly one terminal row per source; operational failures propagate."""
    results = []
    for row in batch.to_pylist():
        uri = str(row["source_uri"])
        if row["source_id"] != make_source_id(uri):
            msg = f"Source identity mismatch for {uri}"
            raise ValueError(msg)
        try:
            metadata, size = _probe(uri, execution)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code"))
            if code not in {"404", "NoSuchKey", "NotFound"}:
                raise
            results.append(
                source_envelope(uri, split)
                | {
                    "status": "error",
                    "error_stage": "source-read",
                    "error_message": f"S3 source unavailable: {code}",
                }
            )
        except MediaError as exc:
            results.append(
                source_envelope(uri, split)
                | {
                    "status": "error",
                    "error_stage": "source-probe",
                    "error_message": str(exc)[:8192],
                }
            )
        else:
            results.append(source_result(uri, metadata, size, split))
    return pa.Table.from_pylist(results, schema=SOURCE_SCHEMA)
