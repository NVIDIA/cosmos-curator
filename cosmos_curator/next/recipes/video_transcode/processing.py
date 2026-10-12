# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-grouped downloads, bounded FFmpeg output, and independent uploads."""

import tempfile
from collections.abc import Iterator
from functools import partial
from itertools import batched
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from botocore.exceptions import BotoCoreError

from cosmos_curator.core.utils.misc.retry_utils import do_with_retries
from cosmos_curator.next.media.ffmpeg import (
    ProbeError,
    TranscodeError,
    assert_video_encoder_available,
    probe_video_path,
    transcode_spans_to_paths,
)
from cosmos_curator.next.media.spans import Span
from cosmos_curator.next.recipes.video_cut.contracts import make_source_id, make_span_id
from cosmos_curator.next.recipes.video_transcode.config import ResolvedVideoTranscodeConfig
from cosmos_curator.next.recipes.video_transcode.contracts import (
    SOURCE_WORK_SCHEMA,
    SPAN_WORK_SCHEMA,
    make_clip_id,
)
from cosmos_curator.next.utils.storage import download_bytes, upload_bytes


def group_source_spans(batch: pa.Table) -> pa.Table:
    """Package one source's remaining spans into a download row.

    Grouping follows Ray's span checkpoint filter, so completed spans do not
    trigger downloads. Each group becomes one row, consumed by a separate
    download call within a Ray task.
    """
    if not batch.num_rows:
        return pa.Table.from_batches([], schema=SOURCE_WORK_SCHEMA)
    source_ids = pc.unique(batch["source_id"]).to_pylist()
    source_uris = pc.unique(batch["source_uri"]).to_pylist()
    sizes = pc.unique(batch["source_size_bytes"]).to_pylist()
    if len(source_ids) != 1 or len(source_uris) != 1 or len(sizes) != 1:
        msg = "A source group must have consistent source identity, URI, and size"
        raise ValueError(msg)
    source_id = str(source_ids[0])
    source_uri = str(source_uris[0])
    if source_id != make_source_id(source_uri):
        msg = "Source identity does not match its immutable URI"
        raise ValueError(msg)
    ordered = batch.take(pc.sort_indices(batch, sort_keys=[("start_ns", "ascending")]))
    for row in ordered.to_pylist():
        if (
            row["stream_index"] != 0
            or row["start_ns"] < 0
            or row["end_ns"] <= row["start_ns"]
            or row["span_id"] != make_span_id(source_id, row["start_ns"], row["end_ns"])
        ):
            msg = "Span does not match its source, stream, or interval identity"
            raise ValueError(msg)
    return pa.Table.from_pylist(
        [
            {
                "source_id": source_id,
                "source_uri": source_uri,
                "source_size_bytes": sizes[0],
                "spans": ordered.select(SPAN_WORK_SCHEMA.names).to_pylist(),
            }
        ],
        schema=SOURCE_WORK_SCHEMA,
    )


def download_source(source: dict[str, Any], *, config: ResolvedVideoTranscodeConfig) -> dict[str, Any]:
    """Download source bytes for handoff to a transcode task on any worker node."""
    source_uri = source["source_uri"]
    expected_size = source["source_size_bytes"]
    payload = do_with_retries(
        partial(download_bytes, source_uri, storage_profile=config.execution.storage_profile),
        exceptions_to_retry=(BotoCoreError, OSError),
        max_attempts=config.execution.storage_attempts,
        name="source-download",
    )
    if expected_size is not None and len(payload) != expected_size:
        msg = f"Source size changed since video-cut: {source_uri}"
        raise ValueError(msg)
    return source | {"source_bytes": payload}


class SourceDownloader:
    """Download rows through a Ray Data actor that owns an IO slot."""

    def __init__(self, *, config: ResolvedVideoTranscodeConfig) -> None:
        """Keep only configuration so a replacement actor can replay any input."""
        self._config = config

    def __call__(self, source: dict[str, Any]) -> dict[str, Any]:
        """Read one source without retaining its payload between calls."""
        return download_source(source, config=self._config)


def transcode_source(source: dict[str, Any], *, config: ResolvedVideoTranscodeConfig) -> Iterator[dict[str, Any]]:
    """Stream encoded clips from downloaded sources without fetching remote media.

    Ray supplies one source row. The generator holds its source file and
    at most one FFmpeg batch of clip files, yielding one payload at a time. Only
    successfully uploaded spans are eligible for durable checkpoints.
    """
    execution = config.execution
    assert_video_encoder_available(config.transcode.video_encoder)
    with tempfile.TemporaryDirectory(prefix="curator_video_transcode_") as directory:
        root = Path(directory)
        source_path = root / "source.mp4"
        # Release the Python copy after staging; Ray still owns the input block.
        source_path.write_bytes(source.pop("source_bytes"))
        for group in batched(source["spans"], execution.ffmpeg_batch_size, strict=False):
            outputs = tuple(
                (Span(start_ns=row["start_ns"], end_ns=row["end_ns"]), root / f"{row['span_id']}.mp4") for row in group
            )
            try:
                _encode(source_path, outputs, config)
            except TranscodeError:
                if len(outputs) == 1:
                    raise
                # Retry each output of a failed multi-output invocation;
                # only successfully uploaded spans can be checkpointed.
                for output in outputs:
                    _encode(source_path, (output,), config)
            for row, (_, path) in zip(group, outputs, strict=True):
                try:
                    metadata = do_with_retries(
                        partial(probe_video_path, path, timeout_s=execution.probe_timeout_s),
                        exceptions_to_retry=(ProbeError,),
                        max_attempts=execution.probe_attempts,
                        name="clip-probe",
                    )
                    clip_id = make_clip_id(row["span_id"], config.transcode.contract_id)
                    payload = path.read_bytes()
                    if not payload:
                        msg = f"Transcode produced an empty clip for {row['span_id']}"
                        raise ValueError(msg)
                    yield {
                        "span_id": row["span_id"],
                        "fragment_id": row["fragment_id"],
                        "row_offset": row["row_offset"],
                        "transcode_contract_id": config.transcode.contract_id,
                        "clip_id": clip_id,
                        "clip_uri": f"{config.input.media_root}/clips/{clip_id}.mp4",
                        "clip_size_bytes": len(payload),
                        "clip_duration_ns": metadata.duration_ns,
                        "clip_width": metadata.width,
                        "clip_height": metadata.height,
                        "clip_frame_rate": metadata.frame_rate,
                        "clip_frame_count": metadata.frame_count,
                        "clip_video_codec": metadata.video_codec,
                        "clip_bytes": payload,
                    }
                finally:
                    path.unlink(missing_ok=True)


def _encode(source: Path, outputs: tuple[tuple[Span, Path], ...], config: ResolvedVideoTranscodeConfig) -> None:
    do_with_retries(
        lambda: transcode_spans_to_paths(
            source,
            outputs,
            config.transcode,
            encoder_threads=config.execution.encoder_threads,
            timeout_s=config.execution.transcode_timeout_s,
        ),
        exceptions_to_retry=(TranscodeError,),
        max_attempts=config.execution.transcode_attempts,
        name="transcode",
    )


def upload_clips(batch: pa.Table, *, config: ResolvedVideoTranscodeConfig, staged_schema: pa.Schema) -> pa.Table:
    """Upload each clip before returning its metadata as ``staged_schema`` rows for checkpointing."""
    for index in range(batch.num_rows):
        payload = batch["clip_bytes"][index].as_py()
        if not payload or len(payload) != batch["clip_size_bytes"][index].as_py():
            msg = f"Clip payload does not match its metadata for {batch['span_id'][index].as_py()}"
            raise ValueError(msg)
        do_with_retries(
            partial(
                upload_bytes,
                payload,
                batch["clip_uri"][index].as_py(),
                storage_profile=config.execution.storage_profile,
            ),
            exceptions_to_retry=(BotoCoreError, OSError),
            max_attempts=config.execution.storage_attempts,
            name="clip-upload",
        )
    return batch.select(staged_schema.names).cast(staged_schema)
