# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-local TransNetV2 decoding, inference, and scene planning.

Download actors transfer encoded bytes to CPU decode tasks. Small RGB frames and
timestamps then flow through Ray to model actors that emit the common source-result
schema. With the NVDEC backend, decoding and inference share a GPU actor instead.
All persistent progress belongs to video-cut's existing source checkpoint and Lance
publication protocol.
"""

from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import av
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from av.video.reformatter import VideoReformatter
from botocore.exceptions import ClientError

from cosmos_curator.models.registry import resolve_registered_model
from cosmos_curator.models.transnetv2 import predict_transitions
from cosmos_curator.next.media.ffmpeg import InvalidMediaError, MediaError, VideoMetadata, probe_video_path
from cosmos_curator.next.media.spans import Span, fixed_stride_spans, seconds_to_nanoseconds
from cosmos_curator.next.recipes.video_cut.config import ExecutionConfig, TransNetV2Config
from cosmos_curator.next.recipes.video_cut.contracts import INPUT_SCHEMA, SOURCE_SCHEMA, make_source_id
from cosmos_curator.next.recipes.video_cut.processing import result_from_spans, source_envelope
from cosmos_curator.next.utils.storage import download_bytes

_HEIGHT = 27
_WIDTH = 48

# Transient, Arrow-native transport between CPU tasks and GPU actors. Packed RGB
# bytes avoid nested Python pixel lists and support different source frame counts.
_DECODED_SOURCE_SCHEMA = pa.schema(
    [
        *INPUT_SCHEMA,
        pa.field("source_size_bytes", pa.int64()),
        pa.field(
            "metadata",
            pa.struct(
                [
                    pa.field("duration_ns", pa.int64()),
                    pa.field("width", pa.int32()),
                    pa.field("height", pa.int32()),
                    pa.field("frame_rate", pa.float64()),
                    pa.field("frame_count", pa.int64()),
                    pa.field("video_codec", pa.string()),
                ]
            ),
        ),
        pa.field("frame_bytes", pa.large_binary(), nullable=False),
        pa.field("timestamps_ns", pa.large_list(pa.int64()), nullable=False),
        pa.field("error_stage", pa.string()),
        pa.field("error_message", pa.string()),
    ]
)


def validate_weights() -> Path:
    """Require the registered runtime checkpoint; staging owns its revision and integrity."""
    path = resolve_registered_model("transnetv2").runtime_model_dir / "transnetv2-pytorch-weights.pth"
    if not path.is_file():
        msg = f"TransNetV2 checkpoint file is not staged at {path}"
        raise FileNotFoundError(msg)
    return path


class SourceDownloader:
    """One sequential source download per actor, admitted by a curator_io slot."""

    def __init__(self, *, storage_profile: str) -> None:
        """Retain credentials configuration without retaining downloaded sources."""
        self._storage_profile = storage_profile

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        """Carry missing-object outcomes to the same source checkpoint as successes."""
        uri = str(row["source_uri"])
        if row["source_id"] != make_source_id(uri):
            msg = "Source identity does not match its immutable URI"
            raise ValueError(msg)
        try:
            payload = download_bytes(uri, storage_profile=self._storage_profile)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code"))
            if code not in {"404", "NoSuchKey", "NotFound"}:
                raise
            return {**row, "video_bytes": b"", "download_error": f"S3 source unavailable: {code}"}
        return {**row, "video_bytes": payload, "download_error": ""}


def normalize_frame_timestamps(timestamps_ns: list[int]) -> npt.NDArray[np.int64]:
    """Validate decoded timestamps and normalize them to the first frame."""
    if not timestamps_ns:
        msg = "Source contains no decodable video frames"
        raise InvalidMediaError(msg)
    times = np.asarray(timestamps_ns, dtype=np.int64)
    times -= times[0]
    if np.any(np.diff(times) <= 0):
        msg = "Decoded video timestamps must be strictly increasing"
        raise InvalidMediaError(msg)
    return times


def decode_frames(path: Path, *, cpus: int) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int64]]:
    """Decode the first video stream at native cadence, retaining relative frame PTS."""
    frames = []
    timestamps = []
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                msg = "Source has no video stream"
                raise InvalidMediaError(msg)
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            stream.codec_context.thread_count = cpus
            # Reuse scaling state and avoid automatic threading for tiny output frames.
            reformatter = VideoReformatter()
            for frame in container.decode(stream):
                if frame.pts is None or frame.time_base is None:
                    msg = "Decoded video frame has no presentation timestamp"
                    raise InvalidMediaError(msg)
                timestamps.append(round(frame.pts * frame.time_base * 1_000_000_000))
                frames.append(
                    reformatter.reformat(frame, width=_WIDTH, height=_HEIGHT, format="rgb24", threads=1).to_ndarray()
                )
    except av.error.InvalidDataError as exc:
        msg = "Source contains invalid video data"
        raise InvalidMediaError(msg) from exc
    times = normalize_frame_timestamps(timestamps)
    return np.stack(frames), times


def _decode_source(row: dict[str, Any], execution: ExecutionConfig, *, nvdec: bool = False) -> dict[str, Any]:
    uri = str(row["source_uri"])
    if row["source_id"] != make_source_id(uri):
        msg = "Source identity does not match its immutable URI"
        raise ValueError(msg)
    result = {
        "source_id": row["source_id"],
        "source_uri": uri,
        "source_size_bytes": None,
        "metadata": None,
        "frame_bytes": b"",
        "timestamps_ns": [],
        "error_stage": None,
        "error_message": None,
    }
    if row["download_error"]:
        return result | {"error_stage": "source-read", "error_message": row["download_error"]}
    error_stage = "source-probe"
    try:
        with TemporaryDirectory(prefix="curator_transnetv2_") as directory:
            path = Path(directory) / "source.mp4"
            path.write_bytes(row.pop("video_bytes"))
            size_bytes = path.stat().st_size
            metadata = probe_video_path(path, timeout_s=execution.probe_timeout_s)
            error_stage = "source-decode"
            if nvdec:
                from cosmos_curator.next.recipes.video_cut.transnetv2_nvdec import decode_frames_nvdec  # noqa: PLC0415

                frames, timestamps = decode_frames_nvdec(path)
            else:
                frames, timestamps = decode_frames(path, cpus=execution.transnetv2.cpus)
    except MediaError as exc:
        return result | {"error_stage": error_stage, "error_message": str(exc)[:8192]}
    return result | {
        "source_size_bytes": size_bytes,
        "metadata": asdict(metadata),
        "frame_bytes": frames.tobytes(),
        "timestamps_ns": timestamps.tolist(),
    }


def decode_sources(batch: pa.Table, *, execution: ExecutionConfig) -> pa.Table:
    """Probe and decode on CPUs, dropping encoded payloads before the GPU boundary."""
    return pa.Table.from_pylist(
        [_decode_source(row, execution) for row in batch.to_pylist()], schema=_DECODED_SOURCE_SCHEMA
    )


def scene_spans(
    transitions: npt.NDArray[np.bool_],
    timestamps_ns: npt.NDArray[np.int64],
    *,
    duration_ns: int,
    split: TransNetV2Config,
) -> tuple[Span, ...]:
    """Exclude transition runs, tile long scenes, and discard scenes/tails below the minimum."""
    if (
        not len(timestamps_ns)
        or len(transitions) != len(timestamps_ns)
        or timestamps_ns[0] != 0
        or np.any(np.diff(timestamps_ns) <= 0)
        or timestamps_ns[-1] >= duration_ns
    ):
        msg = "Decoded frame timestamps do not fit the source video timeline"
        raise InvalidMediaError(msg)
    # False sentinels close the first and final non-transition runs, including
    # an entirely transition-free source. An all-transition source has no spans.
    scenes = np.concatenate(([False], ~transitions, [False]))
    starts = np.flatnonzero(~scenes[:-1] & scenes[1:])
    ends = np.flatnonzero(scenes[:-1] & ~scenes[1:])
    maximum = seconds_to_nanoseconds(split.max_duration_s)
    minimum = seconds_to_nanoseconds(split.min_duration_s)
    spans: list[Span] = []
    for start, end in zip(starts, ends, strict=True):
        start_ns = int(timestamps_ns[start])
        end_ns = duration_ns if end == len(timestamps_ns) else int(timestamps_ns[end])
        spans.extend(
            Span(start_ns + span.start_ns, start_ns + span.end_ns)
            for span in fixed_stride_spans(
                end_ns - start_ns, duration_ns=maximum, stride_ns=maximum, min_duration_ns=minimum
            )
        )
    return tuple(spans)


class TransNetV2Planner:
    """Keep one model resident; NVDEC also decodes sources on the actor's GPU."""

    def __init__(self, *, split: TransNetV2Config, execution: ExecutionConfig) -> None:
        """Validate staged weights and initialize each new GPU actor independently."""
        from cosmos_curator.models.transnetv2 import load_transnetv2_model  # noqa: PLC0415

        self._split = split
        self._execution = execution
        self._model = load_transnetv2_model(validate_weights())

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Return exactly one typed terminal result per source, without media payloads."""
        return pa.Table.from_pylist([self._plan(row) for row in batch.to_pylist()], schema=SOURCE_SCHEMA)

    def _plan(self, row: dict[str, Any]) -> dict[str, Any]:
        if self._split.decoder == "nvdec":
            row = _decode_source(row, self._execution, nvdec=True)
        uri = str(row["source_uri"])
        if row["source_id"] != make_source_id(uri):
            msg = "Source identity does not match its immutable URI"
            raise ValueError(msg)
        result = source_envelope(uri, self._split)
        if row["error_stage"] is not None:
            return result | {
                "status": "error",
                "error_stage": row["error_stage"],
                "error_message": row["error_message"],
            }
        metadata = VideoMetadata(**row["metadata"])
        frames = np.frombuffer(row.pop("frame_bytes"), dtype=np.uint8).reshape(-1, _HEIGHT, _WIDTH, 3)
        timestamps = np.asarray(row["timestamps_ns"], dtype=np.int64)
        try:
            transitions = predict_transitions(
                self._model,
                frames,
                threshold=self._split.threshold,
                window_batch_size=self._execution.transnetv2.window_batch_size,
            )
            spans = scene_spans(transitions, timestamps, duration_ns=metadata.duration_ns, split=self._split)
        except MediaError as exc:
            return result | {"status": "error", "error_stage": "source-decode", "error_message": str(exc)[:8192]}
        return result_from_spans(uri, metadata, row["source_size_bytes"], self._split, spans)
