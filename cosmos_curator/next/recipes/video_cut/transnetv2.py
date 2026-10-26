# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TransNetV2 cutting: admitted downloads, source-local decoding, inference, and scene planning.

Download actors, each holding a ``curator_io`` slot, transfer encoded bytes to
CPU decode tasks. Small RGB frames, timestamps, and probed media fields then
flow through Ray to model actors that emit the common source-result schema.
With the NVDEC backend, decoding and inference share a GPU actor instead. All
persistent progress belongs to the shared source checkpoints and append
publication.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import av
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import ray
from av.video.reformatter import VideoReformatter
from botocore.exceptions import ClientError

from cosmos_curator.core.utils import environment
from cosmos_curator.core.utils.pixi_runtime_envs import ray_data_gpu_runtime_env
from cosmos_curator.models.registry import resolve_registered_model
from cosmos_curator.models.transnetv2 import predict_transitions
from cosmos_curator.next.core.ray_autoscaling import configure_download_actor_autoscaling
from cosmos_curator.next.media.ffmpeg import InvalidMediaError, MediaError, probe_video_path
from cosmos_curator.next.media.spans import Span, fixed_stride_spans, seconds_to_nanoseconds
from cosmos_curator.next.recipes.video_cut.config import ExecutionConfig, ResolvedVideoCutConfig, TransNetV2Config
from cosmos_curator.next.recipes.video_cut.contracts import (
    INPUT_SCHEMA,
    MEDIA_FIELDS,
    SOURCE_SCHEMA,
    error_details,
    error_result,
    media_fields,
    source_result,
)
from cosmos_curator.next.recipes.video_cut.sources import unreadable_source_error
from cosmos_curator.next.utils.storage import download_bytes

_HEIGHT = 27
_WIDTH = 48

# Transient, Arrow-native transport between CPU tasks and GPU actors. Packed RGB
# bytes avoid nested Python pixel lists and support different source frame counts.
_DECODED_SOURCE_SCHEMA = pa.schema(
    [
        *INPUT_SCHEMA,
        *MEDIA_FIELDS,
        pa.field("frame_bytes", pa.large_binary(), nullable=False),
        pa.field("timestamps_ns", pa.large_list(pa.int64()), nullable=False),
        SOURCE_SCHEMA.field("error_stage"),
        SOURCE_SCHEMA.field("error_message"),
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
    """One sequential source download per actor, passing the encoded bytes on for decoding."""

    def __init__(self, *, storage_profile: str) -> None:
        """Retain credentials configuration without retaining downloaded sources."""
        self._storage_profile = storage_profile

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        """Carry missing-object outcomes to the same source checkpoint as successes."""
        try:
            payload = download_bytes(row["source_uri"], storage_profile=self._storage_profile)
        except ClientError as exc:
            error = unreadable_source_error(exc, row["source_uri"], storage_profile=self._storage_profile)
            if error is None:
                raise
            return {**row, "video_bytes": b"", **error}
        return {**row, "video_bytes": payload, "error_stage": None, "error_message": None}


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
    result = {
        "source_id": row["source_id"],
        "source_uri": row["source_uri"],
        "frame_bytes": b"",
        "timestamps_ns": [],
        "error_stage": None,
        "error_message": None,
    }
    if row["error_stage"] is not None:
        return result | error_details(row["error_stage"], row["error_message"])
    error_stage = "source-probe"
    try:
        with TemporaryDirectory(prefix="curator_transnetv2_") as directory:
            path = Path(directory) / "source.mp4"
            path.write_bytes(row.pop("video_bytes"))
            media = media_fields(probe_video_path(path, timeout_s=execution.probe_timeout_s), path.stat().st_size)
            error_stage = "source-decode"
            if nvdec:
                from cosmos_curator.next.recipes.video_cut.transnetv2_nvdec import decode_frames_nvdec  # noqa: PLC0415

                frames, timestamps = decode_frames_nvdec(path)
            else:
                frames, timestamps = decode_frames(path, cpus=execution.transnetv2.cpus)
    except MediaError as exc:
        return result | error_details(error_stage, str(exc))
    return result | media | {"frame_bytes": frames.tobytes(), "timestamps_ns": timestamps.tolist()}


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
        if row["error_stage"] is not None:
            return error_result(row, error_stage=row["error_stage"], error_message=row["error_message"])
        frames = np.frombuffer(row.pop("frame_bytes"), dtype=np.uint8).reshape(-1, _HEIGHT, _WIDTH, 3)
        timestamps = np.asarray(row["timestamps_ns"], dtype=np.int64)
        try:
            transitions = predict_transitions(
                self._model,
                frames,
                threshold=self._split.threshold,
                window_batch_size=self._execution.transnetv2.window_batch_size,
            )
            spans = scene_spans(transitions, timestamps, duration_ns=row["source_duration_ns"], split=self._split)
        except MediaError as exc:
            return error_result(row, error_stage="source-decode", error_message=str(exc))
        return source_result(row, spans)


def plan(sources: ray.data.Dataset, config: ResolvedVideoCutConfig) -> ray.data.Dataset:
    """Download sources, decode them on CPUs unless NVDEC shares the GPU actor, and plan scenes on GPUs."""
    split = config.split
    assert isinstance(split, TransNetV2Config)
    validate_weights()
    configure_download_actor_autoscaling(sources.context)
    downloaded = sources.map(
        cast("Any", SourceDownloader),
        fn_constructor_kwargs={"storage_profile": config.execution.storage_profile},
        compute=ray.data.ActorPoolStrategy(min_size=1, max_tasks_in_flight_per_actor=1),
        num_cpus=0.25,
        resources={environment.CURATOR_IO_RESOURCE_NAME: 1.0},
        max_concurrency=1,
        max_restarts=-1,
        max_task_retries=3,
    )
    execution = config.execution.transnetv2
    decoded = downloaded
    inference_memory_gib = execution.inference_memory_gib
    if split.decoder == "pyav":
        decoded = downloaded.map_batches(
            cast("Any", decode_sources),
            fn_kwargs={"execution": config.execution},
            batch_size=1,
            batch_format="pyarrow",
            num_cpus=execution.cpus,
            num_gpus=0,
            memory=int(execution.memory_gib * 1024**3),
            max_retries=3,
        )
    else:
        # NVDEC and inference share a GPU actor; account for both host-memory needs.
        inference_memory_gib += execution.memory_gib
    return decoded.map_batches(
        TransNetV2Planner,
        fn_constructor_kwargs={"split": split, "execution": config.execution},
        batch_size=1,
        batch_format="pyarrow",
        compute=ray.data.ActorPoolStrategy(min_size=1, max_tasks_in_flight_per_actor=1),
        num_cpus=execution.inference_cpus,
        num_gpus=execution.gpus,
        memory=int(inference_memory_gib * 1024**3),
        max_concurrency=1,
        max_restarts=-1,
        max_task_retries=3,
        runtime_env=ray_data_gpu_runtime_env("default"),
    )
