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

"""The split stage: one session in, its clips out.

The first stage of ``multimodal-split``, and the one every later stage reads
from. A clip is written only if every one of its grid timestamps was served, so
a partially decodable session yields whole clips and named failures rather than
short clips.
"""

import hashlib
import io
import itertools
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa

import cosmos_curator.core.sensors as cosmos_sensors
from cosmos_curator.core.sensors.alignment.gps import align_gps
from cosmos_curator.core.sensors.data.gps_data import GpsData
from cosmos_curator.core.sensors.sampling.grid import SamplingGrid
from cosmos_curator.core.sensors.sampling.policy import (
    GpsInterpolationPolicy,
    GpsSamplingPolicy,
    NearestTimestampPolicy,
)
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors.gps_sensor import GpsSensor
from cosmos_curator.core.sensors.sensors.streaming_camera_sensor import StreamingCameraSensor
from cosmos_curator.core.sensors.utils.timestamps import write_timestamps_ns
from cosmos_curator.core.sensors.utils.validation import require_nondecreasing
from cosmos_curator.next.media.spans import Span
from cosmos_curator.next.recipes.multimodal_split.config import (
    ALIGNED_SENSOR_TYPES,
    COPIED_SENSOR_TYPES,
    MultimodalSplitSensorConfig,
    ResolvedMultimodalSplitConfig,
)
from cosmos_curator.next.recipes.multimodal_split.cpu_video_encode import encode_frames_to_path
from cosmos_curator.next.recipes.multimodal_split.gps_streams import GpsClipWriter
from cosmos_curator.next.recipes.multimodal_split.grids import clip_duration_ns, sampling_grid_ns, sampling_spec
from cosmos_curator.next.recipes.multimodal_split.mcap_streams import McapSpanWriter, SpanCut
from cosmos_curator.next.recipes.multimodal_split.reports import failure, write_sensor_errors
from cosmos_curator.next.recipes.multimodal_split.storage import (
    open_sensor_source,
)
from cosmos_curator.next.recipes.multimodal_split.uris import (
    clip_dir_uri,
    clip_media_uri,
    sensor_timestamps_uri,
    timestamps_sidecar_uri,
)
from cosmos_curator.next.utils.storage import write_media

# One clip's frames. The span and the row count come from planning, so a
# provider decides only where pixels come from -- never which span is written
# or how many rows it holds.
FrameProvider = Callable[[Span, int, list[int]], Iterator[npt.NDArray[np.uint8]]]

SENSOR_OUTCOME_SCHEMA = pa.schema(
    [
        pa.field("status", pa.string(), nullable=False),
        pa.field("record_schema_version", pa.int32(), nullable=False),
        pa.field("media_contract_version", pa.int32(), nullable=False),
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("split_clip_id", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_session_id", pa.large_string(), nullable=False),
        pa.field("source_session_location", pa.large_string(), nullable=False),
        pa.field("start_ns", pa.int64(), nullable=False),
        pa.field("exclusive_end_ns", pa.int64(), nullable=False),
        pa.field("sampling_fps", pa.int32(), nullable=False),
        pa.field("expected_row_count", pa.int64(), nullable=False),
        pa.field("timestamp_source", pa.string(), nullable=False),
        pa.field("split_contract_digest", pa.string(), nullable=False),
        pa.field(
            "planned_camera_keys",
            pa.large_list(pa.field("item", pa.string(), nullable=False)),
            nullable=False,
        ),
        pa.field(
            "planned_sensor_keys",
            pa.large_list(pa.field("item", pa.string(), nullable=False)),
            nullable=False,
        ),
        pa.field("sensor_key", pa.string(), nullable=False),
        pa.field("sensor_type", pa.string(), nullable=False),
        pa.field("source_media_location", pa.large_string(), nullable=False),
        pa.field("media_location", pa.large_string()),
        pa.field("media_size_bytes", pa.int64()),
        pa.field("media_sha256", pa.string()),
        pa.field("timeline_location", pa.large_string()),
        pa.field("timeline_sha256", pa.string()),
        pa.field("sensor_timestamps_location", pa.large_string()),
        pa.field("sensor_timestamps_sha256", pa.string()),
    ]
)


_MCAP_SUFFIX = ".mcap"


def split_session(
    item: dict[str, Any],
    *,
    config: ResolvedMultimodalSplitConfig,
    storage_profile: str = "default",
) -> list[dict[str, Any]]:
    """Write every planned span of one camera, in a single forward pass.

    One camera, not one span. The stream is opened once and walked forward
    through every span the planner allowed: bounds are what an index is for, and
    a pass that cuts a new clip every N frames never needs to seek. Taking a span
    at a time would reopen a remote artifact per clip and decode from the start
    each time, which is the cost a single forward pass exists to avoid.

    The source timeline arrives on the row rather than being read again. Planning
    already paid for it, and a second fetch per camera is a high-latency round trip
    for bytes this run holds.
    """
    session_id = str(item["source_session_id"])
    sensor_key = str(item["sensor_key"])
    sensor_type = str(item["sensor_type"])
    sensor_uri = str(item["sensor_uri"])
    timeline_ns = np.asarray(item["timeline_ns"], dtype=np.int64)
    spans = tuple(
        Span(start_ns=int(entry["start_ns"]), end_ns=int(entry["exclusive_end_ns"])) for entry in item["spans"]
    )
    clip_ids = [str(entry["clip_id"]) for entry in item["spans"]]

    clips: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        # The stream is caller-owned and stays open for every span: the library
        # neither closes it nor restores its position.
        with open_sensor_source(
            sensor_uri, storage_profile=storage_profile, read_buffer_bytes=config.execution.read_buffer_bytes
        ) as source:
            clips, walk_failures = _stream_clips(
                source,
                spans,
                clip_ids,
                timeline_ns,
                session_id,
                sensor_key,
                sensor_type,
                config,
                storage_profile,
            )
            failures.extend(walk_failures)
    except Exception as error:  # noqa: BLE001 -- one unreadable view is one failed sensor
        failures.append(failure(session_id, sensor_key, None, "open_sensor", error))

    write_sensor_errors(
        session_id, sensor_key, failures, output_path=config.output.path, storage_profile=storage_profile
    )
    # Join successes back onto every planned span. errors/ is diagnostic history,
    # so this terminal stream must also represent failures and all-failed episodes.
    clips_by_id = {str(clip["clip_id"]): clip for clip in clips}
    if len(clips_by_id) != len(clips):
        msg = f"Camera {sensor_key!r} emitted duplicate successful clip IDs"
        raise ValueError(msg)
    return [_sensor_outcome(item, entry, clips_by_id.get(str(entry["clip_id"]))) for entry in item["spans"]]


def split_camera_batch(
    plans: pa.Table,
    *,
    config: ResolvedMultimodalSplitConfig,
    storage_profile: str = "default",
) -> pa.Table:
    """Split one Arrow block while preserving the fixed terminal outcome schema."""
    rows = [
        outcome
        for plan in plans.to_pylist()
        for outcome in split_session(plan, config=config, storage_profile=storage_profile)
    ]
    return pa.Table.from_pylist(rows, schema=SENSOR_OUTCOME_SCHEMA)


#: The schema and mapping every recorder in this fleet writes GPS with. Config
#: does not expose them yet because nothing has needed a second producer; the
#: sensor library takes both, so widening this is adding two fields, not a path.
_GPS_SCHEMA_NAME = "foxglove.LocationFix"
_GPS_PROTOBUF_MAPPING = (
    Path(cosmos_sensors.__file__).parent / "examples" / "foxglove_location_fix_protobuf_mapping.yaml"
)


def _gps_policy(sensor: MultimodalSplitSensorConfig) -> GpsSamplingPolicy:
    """Build the alignment policy the config asked for, in the units it takes."""
    if sensor.sampling == "nearest_neighbor":
        return NearestTimestampPolicy(max_delta_ns=sensor.max_delta_ns)
    return GpsInterpolationPolicy(max_gap_ns=sensor.max_gap_ns)


def split_aligned_session(
    item: dict[str, Any],
    *,
    config: ResolvedMultimodalSplitConfig,
    storage_profile: str = "default",
) -> list[dict[str, Any]]:
    """Place one sensor's samples on every planned clip's grid.

    The same grid the cameras are resampled onto, so row *i* of this and frame
    *i* of the video are one instant. That correspondence is the only reason an
    episode has a single timeline, and a sensor-specific rate would hand a
    consumer back the alignment problem the episode exists to remove.

    Decoded once for the session and aligned per span. Every source fix is
    eligible for every grid point, so the windows batch output rather than
    restricting selection, and re-decoding per clip would reparse the whole
    topic once per clip for nothing.
    """
    session_id = str(item["source_session_id"])
    sensor_key = str(item["sensor_key"])
    sensor_uri = str(item["sensor_uri"])
    sensor = config.sensors[sensor_key]
    entries = list(item["spans"])
    clip_ids = [str(entry["clip_id"]) for entry in entries]
    if len(set(clip_ids)) != len(clip_ids):
        msg = f"Sensor {sensor_key!r} was planned with duplicate clip IDs"
        raise ValueError(msg)

    clips: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    try:
        with open_sensor_source(
            sensor_uri, storage_profile=storage_profile, read_buffer_bytes=config.execution.read_buffer_bytes
        ) as source:
            payload = source.read()
        writer = GpsClipWriter(payload, sensor.topic)
        # Before anything is cut: the window came from log_time and the alignment
        # runs on the payload, so a source where those disagree cannot produce a
        # correct clip and should say so once, here, rather than once per clip.
        writer.verify_log_time_matches_payload()
        decoded = GpsSensor(
            io.BytesIO(payload),
            topic=writer.topic,
            schema_name=_GPS_SCHEMA_NAME,
            protobuf_mapping=_GPS_PROTOBUF_MAPPING,
        ).read_all()
    except Exception as error:  # noqa: BLE001 -- one unreadable sensor is one failed sensor
        failures.append(failure(session_id, sensor_key, None, "open_sensor", error))
    else:
        policy = _gps_policy(sensor)
        for clip_id, entry in zip(clip_ids, entries, strict=True):
            span = Span(start_ns=int(entry["start_ns"]), end_ns=int(entry["exclusive_end_ns"]))
            clip, clip_failure = _write_aligned_clip(
                writer, decoded, policy, span, clip_id, session_id, sensor_key, config, storage_profile
            )
            if clip is not None:
                clips[clip_id] = clip
            if clip_failure is not None:
                failures.append(clip_failure)

    write_sensor_errors(
        session_id, sensor_key, failures, output_path=config.output.path, storage_profile=storage_profile
    )
    return [_sensor_outcome(item, entry, clips.get(str(entry["clip_id"]))) for entry in entries]


def _write_aligned_clip(  # noqa: PLR0913 -- every argument names a different part of one clip's identity
    writer: GpsClipWriter,
    decoded: GpsData,
    policy: GpsSamplingPolicy,
    span: Span,
    clip_id: str,
    session_id: str,
    sensor_key: str,
    config: ResolvedMultimodalSplitConfig,
    storage_profile: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Align one clip's grid and write it, or describe why it could not be."""
    destination = clip_media_uri(clip_dir_uri(config.output.path, session_id, clip_id), sensor_key, _MCAP_SUFFIX)
    sampling = str(config.sensors[sensor_key].sampling)
    try:
        grid_ns = sampling_grid_ns(span, config)
        aligned = align_gps(decoded, grid_ns, policy=policy)
        with tempfile.TemporaryDirectory() as staging:
            local = Path(staging) / f"{clip_id}{_MCAP_SUFFIX}"
            written = writer.write_rows(aligned, sampling, local)
            if written != len(grid_ns):
                # A row per grid point or the clip is not on the grid it claims.
                msg = f"aligned {written} rows for a clip whose grid has {len(grid_ns)}"
                raise ValueError(msg)  # noqa: TRY301 -- reported as this clip's failure, like any other
            payload = local.read_bytes()
            write_media(destination, payload, storage_profile=storage_profile)
    except Exception as error:  # noqa: BLE001 -- one clip's failure never ends the session
        return None, failure(session_id, sensor_key, span, "align_sensor", error, clip_id)

    return {
        "clip_id": clip_id,
        "clip_location": destination,
        "clip_bytes": len(payload),
        "media_sha256": hashlib.sha256(payload).hexdigest(),
        # Placed on the episode's grid rather than resampled from a stream, so
        # there is no separate timeline to record: the grid is the episode's.
        "timeline_location": None,
        "timeline_sha256": None,
        "sensor_timestamps_location": None,
        "sensor_timestamps_sha256": None,
    }, None


def split_mcap_session(
    item: dict[str, Any],
    *,
    config: ResolvedMultimodalSplitConfig,
    storage_profile: str = "default",
) -> list[dict[str, Any]]:
    """Cut every planned span of one MCAP sensor into a clip's own MCAP.

    Copied, not resampled. A camera's clip is decoded and rebuilt onto the
    episode's grid; a trajectory's is the session's messages at the rate they
    were recorded, on the bounds the cameras were cut on. Nothing here decodes a
    payload, so the clip carries the schema it arrived with and opens with
    whatever code already reads the session.

    Each cut carries the message immediately before its span. Aligning a clip's
    first grid row asks for a sample at exactly that instant, and both nearest
    selection and linear interpolation need one sample on the earlier side --
    which a strict half-open cut does not leave. Measured across a real session,
    every clip but the session's first had nothing at or before its leading grid
    point without it.

    Whole rather than windowed, unlike a camera. These files are small next to
    video, and a remote read per span would re-fetch the chunks that adjacent
    spans share.
    """
    session_id = str(item["source_session_id"])
    sensor_key = str(item["sensor_key"])
    sensor_uri = str(item["sensor_uri"])
    entries = list(item["spans"])
    spans = [Span(start_ns=int(entry["start_ns"]), end_ns=int(entry["exclusive_end_ns"])) for entry in entries]
    clip_ids = [str(entry["clip_id"]) for entry in entries]
    if len(set(clip_ids)) != len(clip_ids):
        # Parity with the camera path, which raises on the same collision. Two
        # spans sharing a clip id would write one file and then describe it with
        # two rows carrying different bounds.
        msg = f"Sensor {sensor_key!r} was planned with duplicate clip IDs"
        raise ValueError(msg)

    clips: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    try:
        with open_sensor_source(
            sensor_uri, storage_profile=storage_profile, read_buffer_bytes=config.execution.read_buffer_bytes
        ) as source:
            writer = McapSpanWriter(source.read(), config.sensors[sensor_key].topic)
        for clip_id, cut in zip(clip_ids, writer.iter_spans(spans), strict=True):
            clip, clip_failure = _write_mcap_clip(writer, cut, clip_id, session_id, sensor_key, config, storage_profile)
            if clip is not None:
                clips[clip_id] = clip
            if clip_failure is not None:
                failures.append(clip_failure)
    except Exception as error:  # noqa: BLE001 -- one unreadable sensor is one failed sensor
        failures.append(failure(session_id, sensor_key, None, "open_sensor", error))

    write_sensor_errors(
        session_id, sensor_key, failures, output_path=config.output.path, storage_profile=storage_profile
    )
    return [_sensor_outcome(item, entry, clips.get(str(entry["clip_id"]))) for entry in entries]


def _write_mcap_clip(  # noqa: PLR0913 -- every argument names a different part of one clip's identity
    writer: McapSpanWriter,
    cut: SpanCut,
    clip_id: str,
    session_id: str,
    sensor_key: str,
    config: ResolvedMultimodalSplitConfig,
    storage_profile: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Write one span's messages, or describe why they could not be written."""
    destination = clip_media_uri(clip_dir_uri(config.output.path, session_id, clip_id), sensor_key, _MCAP_SUFFIX)
    try:
        with tempfile.TemporaryDirectory() as staging:
            local = Path(staging) / f"{clip_id}{_MCAP_SUFFIX}"
            _total, own = writer.write_cut(cut, local)
            if own == 0:
                # The recording has a gap covering this whole clip. Planning
                # cannot see one -- it reads the topic's first and last time and
                # nothing in between -- so this is a legitimate span, not a file
                # whose index disagrees with itself.
                #
                # Refused rather than written, because a clip holding no poses of
                # its own cannot motion-compensate anything, which is what the
                # trajectory is carried for. It takes the clip's video with it,
                # the same way a missing camera does.
                msg = (
                    f"the recording has no messages between {cut.span.start_ns} and {cut.span.end_ns}, "
                    f"so this clip holds none of its own"
                )
                raise ValueError(msg)  # noqa: TRY301 -- reported as this clip's failure, like any other
            payload = local.read_bytes()
            write_media(destination, payload, storage_profile=storage_profile)
    except Exception as error:  # noqa: BLE001 -- one clip's failure never ends the session
        return None, failure(session_id, sensor_key, cut.span, "cut_mcap", error, clip_id)

    clip = {
        "clip_id": clip_id,
        "clip_location": destination,
        "clip_bytes": len(payload),
        "media_sha256": hashlib.sha256(payload).hexdigest(),
        # A trajectory is not resampled, so there is no grid it was put onto and
        # no per-row record of which sample served which row.
        "timeline_location": None,
        "timeline_sha256": None,
        "sensor_timestamps_location": None,
        "sensor_timestamps_sha256": None,
    }
    return clip, None


def split_mcap_batch(
    plans: pa.Table,
    *,
    config: ResolvedMultimodalSplitConfig,
    storage_profile: str = "default",
) -> pa.Table:
    """Cut one Arrow block of MCAP plans while preserving the terminal outcome schema."""
    rows = []
    for plan in plans.to_pylist():
        # Both kinds come from an MCAP and both are cheap next to video, so they
        # share a stage. How the clip is written is the only difference -- and a
        # kind in neither set is refused rather than falling through to a copy
        # nobody chose for it.
        sensor_type = str(plan["sensor_type"])
        if sensor_type in ALIGNED_SENSOR_TYPES:
            cut = split_aligned_session
        elif sensor_type in COPIED_SENSOR_TYPES:
            cut = split_mcap_session
        else:
            msg = f"sensor_type {sensor_type!r} comes from an MCAP but names no way to cut it into a clip"
            raise ValueError(msg)
        rows.extend(cut(plan, config=config, storage_profile=storage_profile))
    return pa.Table.from_pylist(rows, schema=SENSOR_OUTCOME_SCHEMA)


def _optional_str(value: object) -> str | None:
    """Return *value* as a string, or None where a sensor has no such artifact."""
    return None if value is None else str(value)


def _sensor_outcome(
    plan: dict[str, Any],
    span: dict[str, Any],
    clip: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build one terminal outcome for every planned span of one sensor."""
    if clip is None:
        success_fields: dict[str, Any] = {
            "media_location": None,
            "media_size_bytes": None,
            "media_sha256": None,
            "timeline_location": None,
            "timeline_sha256": None,
            "sensor_timestamps_location": None,
            "sensor_timestamps_sha256": None,
        }
    else:
        # The sidecars are a camera's. A trajectory is copied rather than
        # resampled, so it has no grid to record and no per-row account of which
        # sample served which row, and those fields stay null for it.
        success_fields = {
            "media_location": str(clip["clip_location"]),
            "media_size_bytes": int(clip["clip_bytes"]),
            "media_sha256": str(clip["media_sha256"]),
            "timeline_location": _optional_str(clip["timeline_location"]),
            "timeline_sha256": _optional_str(clip["timeline_sha256"]),
            "sensor_timestamps_location": _optional_str(clip["sensor_timestamps_location"]),
            "sensor_timestamps_sha256": _optional_str(clip["sensor_timestamps_sha256"]),
        }
    return {
        "status": "success" if clip is not None else "failure",
        "record_schema_version": int(span["record_schema_version"]),
        "media_contract_version": int(span["media_contract_version"]),
        "episode_id": str(span["episode_id"]),
        "split_clip_id": str(span["clip_id"]),
        "source_id": str(span["source_id"]),
        "source_session_id": str(span["source_session_id"]),
        "source_session_location": str(span["source_session_location"]),
        "start_ns": int(span["start_ns"]),
        "exclusive_end_ns": int(span["exclusive_end_ns"]),
        "sampling_fps": int(span["sampling_fps"]),
        "expected_row_count": int(span["expected_row_count"]),
        "timestamp_source": str(span["timestamp_source"]),
        "split_contract_digest": str(span["split_contract_digest"]),
        "planned_camera_keys": [str(key) for key in span["planned_camera_keys"]],
        "planned_sensor_keys": [str(key) for key in span["planned_sensor_keys"]],
        "sensor_key": str(plan["sensor_key"]),
        "sensor_type": str(plan["sensor_type"]),
        "source_media_location": str(plan["sensor_uri"]),
        **success_fields,
    }


def _process_one_clip(  # noqa: PLR0913 -- every argument names a different part of one clip's identity
    provide_frames: FrameProvider,
    span: Span,
    clip_id: str,
    session_id: str,
    sensor_key: str,
    sensor_type: str,
    config: ResolvedMultimodalSplitConfig,
    storage_profile: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Sample, encode, and publish one clip, or describe why it could not be.

    Exactly one of the pair is set. A clip and a failure are different things
    with different fields, and a stage that hands both down one stream forces
    whoever reads it to ask every row which kind it is.
    """
    destination = clip_media_uri(clip_dir_uri(config.output.path, session_id, clip_id), sensor_key)
    sensor_timestamps_ns: list[int] = []
    try:
        timeline_ns = sampling_grid_ns(span, config)
        row_count = len(timeline_ns)
        with tempfile.TemporaryDirectory() as staging:
            local = Path(staging) / f"{clip_id}.mp4"
            written = encode_frames_to_path(
                provide_frames(span, row_count, sensor_timestamps_ns),
                local,
                fps=config.clip.sampling_fps,
                encoder=config.video.encoder.video_encoder,
                bitrate=config.video.encoder.video_bitrate,
                gop_size=config.video.encoder.gop_size,
                faststart=config.video.encoder.faststart,
            )
            if written != row_count:
                msg = f"encoded {written} frames for a clip whose grid has {row_count}"
                raise ValueError(msg)  # noqa: TRY301 -- reported as this clip's failure, like any other
            video = local.read_bytes()
            # Sidecars first, media last. A consumer walking the tree looks for
            # the MP4, so publishing it before its timeline and sensor timestamps
            # would leave a window where a clip exists and cannot say when its
            # frames happened -- and a failure between the two writes would
            # leave that state permanently, under a stable clip id. Written this
            # way the MP4 is the commit point: sidecars without media are
            # invisible, media without sidecars cannot happen.
            timeline_location, timeline_sha256 = _write_clip_timeline(destination, timeline_ns, storage_profile)
            sensor_timestamps_location, sensor_timestamps_sha256 = _write_clip_sensor_timestamps(
                destination, sensor_timestamps_ns, storage_profile
            )
            write_media(destination, video, storage_profile=storage_profile)
            clip_bytes = len(video)
            media_sha256 = hashlib.sha256(video).hexdigest()
    except _StreamEndedError:
        # The tail was shorter than a whole clip. encode_frames_to_path has
        # already removed the partial file.
        raise
    except Exception as error:  # noqa: BLE001 -- one clip's failure never ends the session
        return None, failure(session_id, sensor_key, span, "sample_and_encode", error, clip_id)

    clip = {
        "clip_id": clip_id,
        "source_session_id": session_id,
        "sensor_key": sensor_key,
        "sensor_type": sensor_type,
        "start_ns": span.start_ns,
        "exclusive_end_ns": span.end_ns,
        # The string the storage layer accepts, not a normalized URI. A location
        # that has to be un-normalized before it can be used is not a location:
        # the file:// form of this field once sent captions to a directory named
        # for the scheme.
        "clip_location": destination,
        # Measured from the bytes that were written rather than derived from the
        # bitrate, so a run's reported total cannot drift from what is on disk.
        "clip_bytes": clip_bytes,
        "media_sha256": media_sha256,
        "timeline_location": timeline_location,
        "timeline_sha256": timeline_sha256,
        "sensor_timestamps_location": sensor_timestamps_location,
        "sensor_timestamps_sha256": sensor_timestamps_sha256,
    }
    return clip, None


def _write_clip_sensor_timestamps(
    destination: str, sensor_timestamps_ns: list[int], storage_profile: str
) -> tuple[str, str]:
    """Write, for each of the clip's frames, when the camera actually saw it.

    The timeline beside this says where the clip's frames sit: the grid it was
    resampled onto, evenly spaced by construction. The session was not, and
    once the clip is written that difference is unrecoverable -- a later stage
    splitting this clip again would build on an idealized clock with no way back
    to the instants a camera observed.

    Per camera, because twelve views of one moment are served by twelve
    different sensor samples. Non-decreasing rather than ascending, because one
    sensor sample serves several frames wherever the grid asks for more than the
    sensor produced, which is what a missed exposure looks like.
    """
    sensor_timestamps = np.asarray(sensor_timestamps_ns, dtype=np.int64)
    require_nondecreasing("sensor_timestamps_ns", sensor_timestamps)
    buffer = io.BytesIO()
    np.save(buffer, sensor_timestamps, allow_pickle=False)
    location = sensor_timestamps_uri(destination)
    payload = buffer.getvalue()
    write_media(location, payload, storage_profile=storage_profile)
    return location, hashlib.sha256(payload).hexdigest()


def _write_clip_timeline(destination: str, timeline_ns: npt.NDArray[np.int64], storage_profile: str) -> tuple[str, str]:
    """Write the clip's alignment timeline beside its media.

    A clip is read the same way a session is, and an MP4 starts at zero however it
    was cut, so without this a later stage cannot say when a clip's frames
    happened -- the problem this stage solved for its input, handed straight on.
    Written to the same name the recorder's converter uses, so the next stage
    reads a clip exactly as this one read a session.

    The grid rather than the frames that served it: row k of the clip is at
    reference k, which is what the clip resampled onto. Which source observation
    supplied the pixels is provenance and travels on the diagnostics row, where
    it may repeat -- one source frame can serve several rows, and a timeline with
    duplicates is not a timeline.
    """
    buffer = io.BytesIO()
    write_timestamps_ns(timeline_ns.astype(np.int64), buffer)
    location = timestamps_sidecar_uri(destination)
    payload = buffer.getvalue()
    write_media(location, payload, storage_profile=storage_profile)
    return location, hashlib.sha256(payload).hexdigest()


def _stream_clips(  # noqa: PLR0913 -- one argument per part of a session's identity
    source: io.BufferedIOBase,
    spans: tuple[Span, ...],
    clip_ids: list[str],
    timeline_ns: npt.NDArray[np.int64],
    session_id: str,
    sensor_key: str,
    sensor_type: str,
    config: ResolvedMultimodalSplitConfig,
    storage_profile: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Cut one forward pass over a camera into a clip per planned span.

    The spans come from planning; the end of the video does not. Bounds are what
    an index is for, and a pass that cuts a new clip every N frames never needs
    to know where the feed stops, only that it did -- which is what lets this
    run without one.

    The feed running out is not how a session ends -- it is the media and its
    source timeline disagreeing. Planning tiles that timeline and drops the tail,
    so the last planned span always ends inside it, and a container with one frame
    per timestamp cannot run short. When it does, every span still to
    come is unproducible for the same reason, and each is recorded: a clip that
    quietly never appears is worse than one that failed, because nothing in the
    output says to go looking.
    """
    feed = _open_frame_feed(source, spans, timeline_ns, config)
    provider = _feed_provider(feed)

    clips: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    expected = 0
    planned = list(zip(spans, clip_ids, strict=True))
    for index, (span, clip_id) in enumerate(planned):
        expected += len(sampling_grid_ns(span, config))
        try:
            clip, walk_failure = _process_one_clip(
                provider, span, clip_id, session_id, sensor_key, sensor_type, config, storage_profile
            )
        except _StreamEndedError as error:
            # This span was never produced either, so it is reported with the
            # rest rather than being the one the reader has to infer.
            failures.extend(_unproducible(planned[index:], session_id, sensor_key, error))
            break
        # Recorded before realigning, never after. A clip that failed partway
        # through a span the stream cannot finish leaves too few frames to catch
        # up on, so realigning ends the pass -- and the failure that caused it
        # would go unreported if its record depended on that catching up.
        _collect(clip, walk_failure, clips, failures)
        try:
            # Back to the span boundary before the next one starts. A failure
            # above stopped reading partway, and this pass cannot seek back.
            feed.realign(expected)
        except Exception as error:  # noqa: BLE001 -- whatever ended the pass, the spans after it are lost
            # Any fault here ends the pass, not only the feed running dry. A
            # container that stops short raises on the way out rather than
            # simply stopping, and left to propagate that would blame this
            # camera for failing to open -- one session-level record where there
            # should be one per span it cost.
            #
            # This span already has its outcome; the ones after it never will.
            failures.extend(_unproducible(planned[index + 1 :], session_id, sensor_key, error))
            break
    return clips, failures


def _unproducible(
    remaining: list[tuple[Span, str]],
    session_id: str,
    sensor_key: str,
    error: BaseException,
) -> list[dict[str, Any]]:
    """Record every planned span the exhausted feed can no longer reach.

    One per span rather than one for the camera, because the clips already
    written are good and a reader needs to know exactly which moments are
    missing -- the same reason an encode failure names its own span.
    """
    detail = str(error) or "its media holds fewer frames than its sensor timestamps describe"
    reason = _StreamEndedError(f"the camera ran out of frames before this span: {detail}")
    return [failure(session_id, sensor_key, span, "stream_ended", reason, clip_id) for span, clip_id in remaining]


def _collect(
    clip: dict[str, Any] | None,
    failure: dict[str, Any] | None,
    clips: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    """File one span's outcome on the side it belongs to."""
    if clip is not None:
        clips.append(clip)
    if failure is not None:
        failures.append(failure)


class _StreamEndedError(Exception):
    """The frame feed ran out partway through a clip.

    Not a failure: it means the tail of the session was shorter than a whole
    clip, which is exactly what planning drops.
    """


class _FrameFeed:
    """One forward pass over a camera, handed out a clip's worth of frames at a time.

    Opened once and consumed across every clip, because a forward-only sensor
    cannot go back: reopening per clip would re-decode the camera from the start
    each time.
    """

    def __init__(self, frames: Iterator[tuple[int, int, npt.NDArray[np.uint8]]]) -> None:
        first = next(frames, None)
        if first is None:
            msg = "video stream contains no displayable frames"
            raise ValueError(msg)
        self._frames = itertools.chain([first], frames)
        self.taken = 0

    def take(self, count: int, sensor_timestamps_ns: list[int]) -> Iterator[npt.NDArray[np.uint8]]:
        """Yield the next ``count`` frames, recording which source frame served each."""
        for _ in range(count):
            row = next(self._frames, None)
            if row is None:
                raise _StreamEndedError
            _align_ns, pts_ns, frame = row
            sensor_timestamps_ns.append(pts_ns)
            self.taken += 1
            yield frame

    def realign(self, expected: int) -> None:
        """Drop frames until the pass has consumed *expected* of them.

        A clip that fails partway through leaves this cursor inside its span:
        the encoder stopped reading, but the frames it did read are gone. The
        next span would start mid-clip and every clip after it would hold frames
        from the wrong moment, written without an error because nothing compares
        a clip's frames against the span it claims.

        Nothing to do when a clip succeeded, which is the common case: it
        consumed exactly what it asked for.

        Each dropped frame arrives already converted to RGB, and that cost is
        accepted rather than overlooked. The walk exists to feed a dense
        transcode, so the library converts on the way past; skipping the
        conversion for a frame nobody reads would mean a second entry point
        there, to save at most one span of conversions per failed clip.
        """
        while self.taken < expected:
            if next(self._frames, None) is None:
                raise _StreamEndedError
            self.taken += 1


def _spec_for_spans(spans: tuple[Span, ...], config: ResolvedMultimodalSplitConfig) -> SamplingSpec:
    """Build one grid whose windows are the session's clip spans, in order.

    The streaming sensor serves a grid rather than inventing a timeline, so the
    spans a session was planned into become its windows. One window per clip
    means one payload per clip, which is what the provider hands out.
    """
    timestamps = np.concatenate([sampling_spec(span, config).grid.timestamps_ns for span in spans]).astype(np.int64)
    duration_ns = clip_duration_ns(config)
    return SamplingSpec(
        grid=SamplingGrid(
            spans[0].start_ns,
            spans[-1].end_ns,
            timestamps,
            stride_ns=duration_ns,
            duration_ns=duration_ns,
        )
    )


def _open_frame_feed(
    source: io.BufferedIOBase,
    spans: tuple[Span, ...],
    timeline_ns: npt.NDArray[np.int64],
    config: ResolvedMultimodalSplitConfig,
) -> _FrameFeed:
    """Open one forward pass over a camera, serving the planned spans in order."""
    sensor = StreamingCameraSensor(source, timestamps_ns=timeline_ns)
    policy = NearestTimestampPolicy(max_delta_ns=config.video.max_delta_ns)
    payloads = sensor.sample(_spec_for_spans(spans, config), policy=policy)
    return _FrameFeed(row for payload in payloads for row in payload.frames())


def _feed_provider(feed: _FrameFeed) -> FrameProvider:
    """Serve each clip from the next run of frames in one continuous pass."""

    def provide(_span: Span, row_count: int, sensor_timestamps_ns: list[int]) -> Iterator[npt.NDArray[np.uint8]]:
        return feed.take(row_count, sensor_timestamps_ns)

    return provide
