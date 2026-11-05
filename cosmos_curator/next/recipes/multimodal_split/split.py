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
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa

import cosmos_curator.core.sensors as cosmos_sensors
from cosmos_curator.core.sensors.alignment.gps import align_gps
from cosmos_curator.core.sensors.data.gps_data import GpsData
from cosmos_curator.core.sensors.data.streaming_camera_data import StreamingCameraData
from cosmos_curator.core.sensors.sampling.grid import SamplingGrid
from cosmos_curator.core.sensors.sampling.policy import (
    GpsInterpolationPolicy,
    GpsSamplingPolicy,
    NearestTimestampPolicy,
)
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors.gps_sensor import GpsSensor
from cosmos_curator.core.sensors.sensors.streaming_camera_sensor import StreamingCameraSensor, stream_camera_windows
from cosmos_curator.core.sensors.types.types import VideoFrameAlignmentField
from cosmos_curator.core.sensors.utils.video import iter_video_frames
from cosmos_curator.next.media.spans import Span
from cosmos_curator.next.recipes.mp4_index.index_lookup import Mp4IndexLookup
from cosmos_curator.next.recipes.multimodal_split.config import (
    ALIGNED_SENSOR_TYPES,
    COPIED_SENSOR_TYPES,
    MultimodalSplitSensorConfig,
    ResolvedMultimodalSplitConfig,
)
from cosmos_curator.next.recipes.multimodal_split.cpu_video_encode import encode_frames_to_path
from cosmos_curator.next.recipes.multimodal_split.gps_streams import GpsClipWriter
from cosmos_curator.next.recipes.multimodal_split.grids import sampling_grid_ns, sampling_spec
from cosmos_curator.next.recipes.multimodal_split.identities import make_split_contract_digest
from cosmos_curator.next.recipes.multimodal_split.mcap_streams import McapSpanWriter, SpanCut
from cosmos_curator.next.recipes.multimodal_split.output_index import (
    find_output_index,
    index_descriptor,
    output_index_uri,
    publish_output_index,
    validate_output_index,
)
from cosmos_curator.next.recipes.multimodal_split.reports import failure, write_sensor_errors
from cosmos_curator.next.recipes.multimodal_split.storage import (
    open_sensor_source,
)
from cosmos_curator.next.recipes.multimodal_split.uris import (
    clip_dir_uri,
    clip_media_uri,
)
from cosmos_curator.next.utils.storage import write_media

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
        pa.field("mp4_index_uri", pa.large_string()),
        pa.field("index_sha256", pa.string()),
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
            windows = _open_camera_windows(
                source,
                sensor_uri,
                spans,
                timeline_ns,
                config,
                storage_profile,
                camera_alignment_field=item.get("camera_alignment_field"),
            )
            clips, walk_failures = _stream_clips(
                windows,
                spans,
                clip_ids,
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
        "mp4_index_uri": None,
        "index_sha256": None,
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
        "mp4_index_uri": None,
        "index_sha256": None,
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
            "mp4_index_uri": None,
            "index_sha256": None,
        }
    else:
        # Only cameras carry output indexes; MCAPs are self-contained.
        success_fields = {
            "media_location": str(clip["clip_location"]),
            "media_size_bytes": int(clip["clip_bytes"]),
            "media_sha256": str(clip["media_sha256"]),
            "mp4_index_uri": _optional_str(clip["mp4_index_uri"]),
            "index_sha256": _optional_str(clip["index_sha256"]),
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
    window: StreamingCameraData,
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
    try:
        timeline_ns = sampling_grid_ns(span, config)
        if not np.array_equal(window.align_timestamps_ns, timeline_ns):
            msg = "camera window timestamps disagree with the planned clip grid"
            raise ValueError(msg)  # noqa: TRY301 -- a mismatched window cannot be published
        row_count = len(timeline_ns)
        uri = output_index_uri(destination)
        existing = find_output_index(destination, uri=uri, storage_profile=storage_profile)
        if existing is not None:
            row = existing.to_pylist()[0]
            if (
                row["split_contract_digest"] != make_split_contract_digest(config)
                or row["alignment_timestamps_ns"] != timeline_ns.tolist()
                or row["source_timestamps_ns"] != window.sensor_timestamps_ns.tolist()
            ):
                msg = f"Output index conflicts with the current split contract: {destination}"
                raise ValueError(msg)  # noqa: TRY301 -- incompatible output is this clip's failure
            try:
                validate_output_index(existing, uri=uri, storage_profile=storage_profile)
            except (FileNotFoundError, ValueError, OSError):
                pass  # Replay incomplete payload publication using the same planned window.
            else:
                return {
                    "clip_id": clip_id,
                    "clip_location": destination,
                    "clip_bytes": row["media_size_bytes"],
                    **index_descriptor(existing, uri=uri),
                }, None
        with tempfile.TemporaryDirectory() as staging:
            local = Path(staging) / f"{clip_id}.mp4"
            written = encode_frames_to_path(
                _decode_window(window),
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
            descriptor = publish_output_index(
                local,
                destination=destination,
                uri=output_index_uri(destination),
                session_id=session_id,
                camera_key=sensor_key,
                alignment_timestamps_ns=timeline_ns,
                source_timestamps_ns=window.sensor_timestamps_ns,
                split_contract_digest=make_split_contract_digest(config),
                storage_profile=storage_profile,
            )
    except _StreamEndedError:
        # A decoder failure ends the shared pass. The encoder removes its partial file.
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
        "clip_bytes": descriptor["media_size_bytes"],
        **descriptor,
    }
    return clip, None


def _decode_window(window: StreamingCameraData) -> Iterator[npt.NDArray[np.uint8]]:
    """Yield pixels while distinguishing decode failures from encoder failures."""
    try:
        for _align_ns, _source_ns, frame in window.frames():
            yield frame
    except Exception as error:
        raise _StreamEndedError(str(error)) from error


def _stream_clips(  # noqa: PLR0913 -- one argument per part of a session's identity
    windows: Iterator[StreamingCameraData],
    spans: tuple[Span, ...],
    clip_ids: list[str],
    session_id: str,
    sensor_key: str,
    sensor_type: str,
    config: ResolvedMultimodalSplitConfig,
    storage_profile: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Encode one sampling window per clip, retaining outcomes if decoding ends early."""
    clips: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    planned = list(zip(spans, clip_ids, strict=True))
    for index, (span, clip_id) in enumerate(planned):
        try:
            window = next(windows)
        except Exception as error:  # noqa: BLE001 -- a missing or invalid window ends the shared pass
            failures.extend(_unproducible(planned[index:], session_id, sensor_key, error))
            break
        try:
            clip, walk_failure = _process_one_clip(
                window, span, clip_id, session_id, sensor_key, sensor_type, config, storage_profile
            )
        except _StreamEndedError as error:
            failures.extend(_unproducible(planned[index:], session_id, sensor_key, error))
            break
        _collect(clip, walk_failure, clips, failures)
    return clips, failures


def _unproducible(
    remaining: list[tuple[Span, str]],
    session_id: str,
    sensor_key: str,
    error: BaseException,
) -> list[dict[str, Any]]:
    """Record every planned span the camera can no longer decode.

    One per span rather than one for the camera, because the clips already
    written are good and a reader needs to know exactly which moments are
    missing -- the same reason an encode failure names its own span.
    """
    detail = str(error) or "its media holds fewer frames than its sensor timestamps describe"
    reason = _StreamEndedError(f"camera decoding could not complete this span: {detail}")
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
    """Frame iteration failed and the shared camera pass cannot continue."""


def _spec_for_spans(spans: tuple[Span, ...], config: ResolvedMultimodalSplitConfig) -> SamplingSpec:
    """Build one grid whose windows are the session's clip spans, in order.

    The streaming sensor serves a grid rather than inventing a timeline, so the
    spans a session was planned into become its windows. One window per clip
    means one payload per clip; gaps between planned clips have empty windows.
    """
    timestamps = np.concatenate([sampling_spec(span, config).grid.timestamps_ns for span in spans]).astype(np.int64)
    duration_ns = spans[0].duration_ns
    return SamplingSpec(
        grid=SamplingGrid(
            spans[0].start_ns,
            spans[-1].end_ns,
            timestamps,
            stride_ns=duration_ns,
            duration_ns=duration_ns,
        )
    )


def _open_camera_sensor(  # noqa: PLR0913 -- source, planned clock/timeline, and indexed storage location
    source: io.BufferedIOBase,
    sensor_uri: str,
    timeline_ns: npt.NDArray[np.int64],
    mp4_index_uri: str,
    storage_profile: str,
    alignment_field: str,
) -> StreamingCameraSensor:
    """Build a streaming sensor from the configured manifest record."""
    lookup = Mp4IndexLookup.open_required(mp4_index_uri, storage_profile=storage_profile)
    indexed = lookup.require_indexed_video_for(sensor_uri, storage_profile=storage_profile)
    sensor = StreamingCameraSensor(
        source,
        metadata=indexed.metadata,
        frame_index=indexed.frame_index,
        alignment_field=VideoFrameAlignmentField(alignment_field),
        moov_bytes=indexed.moov_bytes,
    )
    if not np.array_equal(sensor.timestamps_ns, timeline_ns):
        msg = "indexed camera timestamps changed since planning"
        raise ValueError(msg)
    return sensor


def _open_camera_windows(  # noqa: PLR0913 -- source, planned spans/timeline, and storage configuration
    source: io.BufferedIOBase,
    sensor_uri: str,
    spans: tuple[Span, ...],
    timeline_ns: npt.NDArray[np.int64],
    config: ResolvedMultimodalSplitConfig,
    storage_profile: str,
    *,
    camera_alignment_field: str | None,
) -> Iterator[StreamingCameraData]:
    """Serve planned spans using indexed timestamps or the legacy external timeline."""
    policy = NearestTimestampPolicy(max_delta_ns=config.video.max_delta_ns)
    spec = _spec_for_spans(spans, config)
    if config.input.mp4_index_uri is None:
        payloads = stream_camera_windows(iter_video_frames(source), timeline_ns, spec, policy=policy)
    else:
        if camera_alignment_field is None:
            msg = "indexed camera plan has no selected camera_alignment_field"
            raise ValueError(msg)
        sensor = _open_camera_sensor(
            source, sensor_uri, timeline_ns, config.input.mp4_index_uri, storage_profile, camera_alignment_field
        )
        payloads = sensor.sample(spec, policy=policy)
    return (window for window in payloads if window.align_timestamps_ns.size)
