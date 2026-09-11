# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Streaming download, transcode, and upload operations for ``video-split``."""

import json
import tempfile
from collections.abc import Callable, Iterator
from itertools import batched
from pathlib import Path
from typing import Any

from loguru import logger

from cosmos_curator.core.utils.misc.retry_utils import do_with_retries
from cosmos_curator.next.media.ffmpeg import (
    MediaError,
    ProbeError,
    TranscodeError,
    VideoMetadata,
    assert_video_encoder_available,
    probe_video_path,
    transcode_span_to_path,
    transcode_spans_to_paths,
)
from cosmos_curator.next.media.spans import Span, fixed_stride_spans, seconds_to_nanoseconds
from cosmos_curator.next.recipes.video_split.config import ResolvedVideoSplitConfig
from cosmos_curator.next.recipes.video_split.contracts import (
    CLIP_RECORD_SCHEMA_VERSION,
    ERROR_RECORD_SCHEMA_VERSION,
    MEDIA_CONTRACT_VERSION,
    UNKNOWN_FRAME_COUNT,
)
from cosmos_curator.next.recipes.video_split.identities import make_clip_id, make_source_id
from cosmos_curator.next.recipes.video_split.records import SOURCE_MEDIA_FIELDS
from cosmos_curator.next.recipes.video_split.storage import (
    RETRYABLE_STORAGE_ERRORS,
    STORAGE_ERRORS,
    download_bytes,
    upload_bytes,
)
from cosmos_curator.next.recipes.video_split.uris import join_s3_uri

# Inconclusive probe failures propagate after retries; they are not media outcomes.
_RETRYABLE_MEDIA_ERRORS = (ProbeError, TranscodeError)

_BACKOFF_FACTOR = 2.0
_MAX_BACKOFF_S = 30.0

_SOURCE_READY = "ready"
_SOURCE_ERROR = "error"
_SPAN_ITEM_LENGTH = 2

type _PreparedClip = tuple[dict[str, Any], Path | None]


def download_and_plan_source(row: dict[str, Any], *, config: ResolvedVideoSplitConfig) -> dict[str, Any]:
    """Download one source, probing unknown sources and reusing known metadata."""
    source_uri = str(row["source_uri"])
    source_id = str(row.get("source_id") or make_source_id(source_uri))
    execution = config.execution
    envelope = _source_envelope(source_uri, source_id)

    try:
        source_bytes = _retry(
            lambda: download_bytes(source_uri, storage_profile=execution.storage_profile),
            RETRYABLE_STORAGE_ERRORS,
            attempts=execution.storage_attempts,
            name="source-read",
        )
    except STORAGE_ERRORS as exc:
        return envelope | {
            "source_state": _SOURCE_ERROR,
            "error_stage": "source-read",
            "error_message": _error_message(exc),
        }

    if bool(row.get("source_known", False)):
        source_fields = _known_source_fields(row, source_uri=source_uri, source_id=source_id)
        clip_work = _plan_missing_clips(source_fields, str(row["missing_spans_json"]), config=config)
    else:
        with tempfile.TemporaryDirectory(prefix="curator_next_video_split_probe_") as tmp_dir:
            source_path = Path(tmp_dir) / "source.mp4"
            source_path.write_bytes(source_bytes)
            try:
                metadata = _retry(
                    lambda: probe_video_path(source_path, timeout_s=execution.probe_timeout_s),
                    _RETRYABLE_MEDIA_ERRORS,
                    attempts=execution.probe_attempts,
                    name="source-probe",
                )
            except MediaError as exc:
                return envelope | {
                    "source_state": _SOURCE_ERROR,
                    "error_stage": "source-probe",
                    "error_message": _error_message(exc),
                }
        source_fields = _source_fields(source_uri, source_id, len(source_bytes), metadata)
        clip_work = plan_clip_work(source_fields, config=config)

    return (
        envelope
        | source_fields
        | {
            "source_state": _SOURCE_READY,
            "source_bytes": source_bytes,
            "clip_work": clip_work,
        }
    )


def transcode_source(row: dict[str, Any], *, config: ResolvedVideoSplitConfig) -> Iterator[dict[str, Any]]:
    """Yield bounded clip payloads and sparse errors from one source."""
    state = str(row["source_state"])
    if state == _SOURCE_ERROR:
        yield _source_failure(
            str(row["source_uri"]),
            str(row["source_id"]),
            stage=str(row["error_stage"]),
            message=str(row["error_message"]),
        )
        return
    if state != _SOURCE_READY:
        msg = f"Cannot transcode source {row['source_uri']} from state {state!r}"
        raise ValueError(msg)

    clip_work = [dict(record) for record in row["clip_work"]]
    if not clip_work:
        return

    # A worker whose FFmpeg build lacks the encoder is a broken environment,
    # not a bad source, so it fails the task.
    assert_video_encoder_available(config.transcode.video_encoder)

    with tempfile.TemporaryDirectory(prefix="curator_next_video_split_transcode_") as tmp_dir:
        root = Path(tmp_dir)
        source_path = root / "source.mp4"
        source_path.write_bytes(bytes(row["source_bytes"]))
        for clip_batch in batched(clip_work, config.execution.ffmpeg_batch_size, strict=False):
            prepared = _transcode_clip_batch(
                clip_batch,
                source_path=source_path,
                clips_dir=root / "clips",
                config=config,
            )
            for outcome, clip_path in prepared:
                if clip_path is None:
                    yield outcome
                    continue
                try:
                    # Ray's flat-map block builder releases a block around the
                    # configured byte target. Downstream upload tasks consume
                    # those blocks while this generator continues transcoding.
                    yield dict(outcome) | {"clip_bytes": clip_path.read_bytes()}
                finally:
                    clip_path.unlink(missing_ok=True)


def upload_clip(row: dict[str, Any], *, config: ResolvedVideoSplitConfig) -> dict[str, Any]:
    """Upload one clip payload, dropping its bytes before publication."""
    record_type = str(row["record_type"])
    payload = bytes(row["clip_bytes"])
    if record_type == "error":
        if payload:
            msg = f"Error record for {row['source_uri']} unexpectedly carries clip media"
            raise ValueError(msg)
        return dict(row)
    if record_type != "clip":
        msg = f"Upload received unexpected record type {record_type!r}"
        raise ValueError(msg)
    if not payload:
        msg = f"Clip {row['clip_id']} reached upload without media bytes"
        raise ValueError(msg)

    execution = config.execution
    _retry(
        lambda: upload_bytes(
            payload,
            str(row["clip_uri"]),
            storage_profile=execution.storage_profile,
        ),
        RETRYABLE_STORAGE_ERRORS,
        attempts=execution.storage_attempts,
        name="media-write",
    )
    return dict(row) | {"clip_bytes": b""}


def _transcode_clip_batch(
    rows: tuple[dict[str, Any], ...],
    *,
    source_path: Path,
    clips_dir: Path,
    config: ResolvedVideoSplitConfig,
) -> list[_PreparedClip]:
    """Transcode one bounded group, isolating a failed group clip by clip."""
    paths = tuple(clips_dir / f"{row['clip_id']}.mp4" for row in rows)
    if len(rows) == 1:
        return [_transcode_clip(rows[0], source_path=source_path, clip_path=paths[0], config=config)]

    execution = config.execution
    outputs = tuple(
        (Span(start_ns=int(row["start_ns"]), end_ns=int(row["end_ns"])), path)
        for row, path in zip(rows, paths, strict=True)
    )
    try:
        _retry(
            lambda: transcode_spans_to_paths(
                source_path,
                outputs,
                config.transcode,
                encoder_threads=execution.encoder_threads,
                timeout_s=execution.transcode_timeout_s,
            ),
            _RETRYABLE_MEDIA_ERRORS,
            attempts=execution.transcode_attempts,
            name="transcode-batch",
        )
    except MediaError as exc:
        logger.warning(
            "Batched FFmpeg transcode failed for {} clips; retrying each clip independently: {}",
            len(rows),
            exc,
        )
        for path in paths:
            path.unlink(missing_ok=True)
        return [
            _transcode_clip(row, source_path=source_path, clip_path=path, config=config)
            for row, path in zip(rows, paths, strict=True)
        ]

    return [_inspect_transcoded_clip(row, clip_path=path, config=config) for row, path in zip(rows, paths, strict=True)]


def _transcode_clip(
    row: dict[str, Any],
    *,
    source_path: Path,
    clip_path: Path,
    config: ResolvedVideoSplitConfig,
) -> _PreparedClip:
    span = Span(start_ns=int(row["start_ns"]), end_ns=int(row["end_ns"]))
    execution = config.execution
    try:
        _retry(
            lambda: transcode_span_to_path(
                source_path,
                clip_path,
                span,
                config.transcode,
                encoder_threads=execution.encoder_threads,
                timeout_s=execution.transcode_timeout_s,
            ),
            _RETRYABLE_MEDIA_ERRORS,
            attempts=execution.transcode_attempts,
            name="transcode",
        )
    except MediaError as exc:
        clip_path.unlink(missing_ok=True)
        return _clip_failure(row, stage="transcode", error=exc), None

    return _inspect_transcoded_clip(row, clip_path=clip_path, config=config)


def _inspect_transcoded_clip(
    row: dict[str, Any],
    *,
    clip_path: Path,
    config: ResolvedVideoSplitConfig,
) -> _PreparedClip:
    """Probe an FFmpeg output and return metadata plus its worker-local path."""
    execution = config.execution
    try:
        clip_metadata = _retry(
            lambda: probe_video_path(clip_path, timeout_s=execution.probe_timeout_s),
            _RETRYABLE_MEDIA_ERRORS,
            attempts=execution.probe_attempts,
            name="clip-probe",
        )
        clip_size_bytes = clip_path.stat().st_size
    except MediaError as exc:
        clip_path.unlink(missing_ok=True)
        return _clip_failure(row, stage="clip-probe", error=exc), None

    return (
        dict(row)
        | {
            "record_type": "clip",
            "clip_size_bytes": clip_size_bytes,
            "clip_duration_ns": clip_metadata.duration_ns,
            "clip_width": clip_metadata.width,
            "clip_height": clip_metadata.height,
            "clip_frame_rate": clip_metadata.frame_rate,
            "clip_frame_count": _frame_count(clip_metadata),
            "clip_video_codec": clip_metadata.video_codec,
            "error_stage": "",
            "error_message": "",
        },
        clip_path,
    )


def plan_clip_work(source_fields: dict[str, Any], *, config: ResolvedVideoSplitConfig) -> list[dict[str, Any]]:
    """Build deterministic clip work after the source has been probed."""
    split = config.split
    spans = fixed_stride_spans(
        int(source_fields["source_duration_ns"]),
        duration_ns=seconds_to_nanoseconds(split.duration_s),
        stride_ns=seconds_to_nanoseconds(split.stride_s),
        min_duration_ns=seconds_to_nanoseconds(split.min_duration_s),
    )
    records: list[dict[str, Any]] = []
    for span in spans:
        clip_id = make_clip_id(str(source_fields["source_id"]), span, config.transcode)
        records.append(
            _clip_work_record(
                source_fields,
                span=span,
                clip_id=clip_id,
                clip_uri=join_s3_uri(config.output.media_root, "clips", f"{clip_id}.mp4"),
            )
        )
    return records


def _plan_missing_clips(
    source_fields: dict[str, Any],
    missing_spans_json: str,
    *,
    config: ResolvedVideoSplitConfig,
) -> list[dict[str, Any]]:
    raw_spans = json.loads(missing_spans_json)
    if not isinstance(raw_spans, list):
        msg = "Reconciled missing spans must be a JSON array"
        raise TypeError(msg)

    records: list[dict[str, Any]] = []
    for raw_span in raw_spans:
        if (
            not isinstance(raw_span, list)
            or len(raw_span) != _SPAN_ITEM_LENGTH
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in raw_span)
        ):
            msg = f"Invalid reconciled span: {raw_span!r}"
            raise TypeError(msg)
        span = Span(start_ns=raw_span[0], end_ns=raw_span[1])
        clip_id = make_clip_id(str(source_fields["source_id"]), span, config.transcode)
        records.append(
            _clip_work_record(
                source_fields,
                span=span,
                clip_id=clip_id,
                clip_uri=join_s3_uri(config.output.media_root, "clips", f"{clip_id}.mp4"),
            )
        )
    return records


def _source_envelope(source_uri: str, source_id: str) -> dict[str, Any]:
    """Return a schema-stable carrier used between download and transcode."""
    return {
        "source_uri": source_uri,
        "source_id": source_id,
        "source_state": _SOURCE_READY,
        "source_bytes": b"",
        "source_size_bytes": 0,
        "source_duration_ns": 0,
        "source_width": 0,
        "source_height": 0,
        "source_frame_rate": 0.0,
        "source_frame_count": UNKNOWN_FRAME_COUNT,
        "source_video_codec": "",
        "clip_work": [],
        "error_stage": "",
        "error_message": "",
    }


def _source_fields(source_uri: str, source_id: str, size_bytes: int, metadata: VideoMetadata) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "source_uri": source_uri,
        "source_size_bytes": size_bytes,
        "source_duration_ns": metadata.duration_ns,
        "source_width": metadata.width,
        "source_height": metadata.height,
        "source_frame_rate": metadata.frame_rate,
        "source_frame_count": _frame_count(metadata),
        "source_video_codec": metadata.video_codec,
    }


def _known_source_fields(row: dict[str, Any], *, source_uri: str, source_id: str) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "source_uri": source_uri,
        **{field: row[field] for field in SOURCE_MEDIA_FIELDS},
    }


def _source_failure(source_uri: str, source_id: str, *, stage: str, message: str) -> dict[str, Any]:
    """Build an error record for a source that could not be planned."""
    source_fields = _source_envelope(source_uri, source_id)
    return _base_record(source_fields, record_type="error") | {
        "record_schema_version": ERROR_RECORD_SCHEMA_VERSION,
        "error_stage": stage,
        "error_message": message,
    }


def _base_record(source_fields: dict[str, Any], *, record_type: str) -> dict[str, Any]:
    """Build the uniform row shape carried from transcode through upload."""
    return {
        "record_type": record_type,
        "record_schema_version": CLIP_RECORD_SCHEMA_VERSION,
        "media_contract_version": MEDIA_CONTRACT_VERSION,
        "source_id": source_fields["source_id"],
        "source_uri": source_fields["source_uri"],
        **{field: source_fields[field] for field in SOURCE_MEDIA_FIELDS},
        "start_ns": 0,
        "end_ns": 0,
        "clip_id": "",
        "clip_uri": "",
        "clip_bytes": b"",
        "clip_size_bytes": 0,
        "clip_duration_ns": 0,
        "clip_width": 0,
        "clip_height": 0,
        "clip_frame_rate": 0.0,
        "clip_frame_count": UNKNOWN_FRAME_COUNT,
        "clip_video_codec": "",
        "error_stage": "",
        "error_message": "",
    }


def _clip_work_record(
    source_fields: dict[str, Any],
    *,
    span: Span,
    clip_id: str,
    clip_uri: str,
) -> dict[str, Any]:
    """Build one pending clip row."""
    return _base_record(source_fields, record_type="clip") | {
        "start_ns": span.start_ns,
        "end_ns": span.end_ns,
        "clip_id": clip_id,
        "clip_uri": clip_uri,
    }


def _clip_failure(row: dict[str, Any], *, stage: str, error: Exception) -> dict[str, Any]:
    return dict(row) | {
        "record_type": "error",
        "record_schema_version": ERROR_RECORD_SCHEMA_VERSION,
        "clip_bytes": b"",
        "error_stage": stage,
        "error_message": _error_message(error),
    }


def _retry[T](
    operation: Callable[[], T],
    exceptions: tuple[type[Exception], ...],
    *,
    attempts: int,
    name: str,
) -> T:
    """Retry one worker-local operation with the house backoff policy."""
    return do_with_retries(
        operation,
        exceptions,
        max_attempts=attempts,
        backoff_factor=_BACKOFF_FACTOR,
        max_wait_time_s=_MAX_BACKOFF_S,
        name=name,
    )


def _frame_count(metadata: VideoMetadata) -> int:
    return metadata.frame_count if metadata.frame_count is not None else UNKNOWN_FRAME_COUNT


def _error_message(error: Exception) -> str:
    message = str(error).strip() or type(error).__name__
    return message[:8192]
