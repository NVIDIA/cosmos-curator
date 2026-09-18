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
"""MCAP camera sensor for Foxglove CompressedVideo topics."""

import base64
import binascii
import json
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, cast

import av
import numpy as np
import numpy.typing as npt
from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
from mcap.records import Channel, Schema
from mcap.records import Message as McapMessage

from cosmos_curator.core.sensors.data.camera_data import CameraData
from cosmos_curator.core.sensors.data.video import VideoMetadata
from cosmos_curator.core.sensors.sampling.grid import SamplingWindow
from cosmos_curator.core.sensors.sampling.policy import NearestTimestampPolicy, require_nearest_timestamp_policy
from cosmos_curator.core.sensors.sampling.sampler import sample_window_indices
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors.group import STREAM_TIMESTAMPS_CAMERA_ONLY_MSG
from cosmos_curator.core.sensors.types.types import DataSource
from cosmos_curator.core.sensors.utils.mcap import McapTopicAccessor

_RGB_CHANNELS = 3
_COMPRESSED_VIDEO_SCHEMA_NAME = "foxglove.CompressedVideo"
_JSON_ENCODING = "json"
_PROTOBUF_ENCODING = "protobuf"
_SUPPORTED_FORMATS = {"h264": "h264", "h265": "hevc"}
_H264_IDR_NAL_TYPE = 5
_H265_IRAP_NAL_TYPE_MIN = 16
_H265_IRAP_NAL_TYPE_MAX = 21
_MAX_MCAP_TIME_NS = np.iinfo(np.int64).max


@dataclass
class _DecodeStreamState:
    """Mutable decoder state carried across adjacent sampling windows."""

    expected_format: str | None = None
    decoder: "_ForwardCompressedVideoDecoder | None" = None
    previous_frame_time: int | None = None
    decoded_log_times_ns: list[int] = field(default_factory=list)
    pending: tuple[Schema | None, Channel, McapMessage] | None = None


@dataclass
class _WindowValidationState:
    """Mutable state for incremental sampling-window validation."""

    previous_start: int | None = None
    previous_end: int | None = None


def _timestamp_to_ns(timestamp: object) -> int | None:
    """Convert an optional Foxglove/protobuf timestamp object to nanoseconds."""
    if timestamp is None:
        return None

    if isinstance(timestamp, dict):
        seconds = timestamp.get("sec", timestamp.get("seconds"))
        nanos = timestamp.get("nsec", timestamp.get("nanos"))
    else:
        seconds = getattr(timestamp, "seconds", None)
        nanos = getattr(timestamp, "nanos", None)

    if seconds is None and nanos is None:
        return None
    try:
        seconds_int = int(seconds or 0)
        nanos_int = int(nanos or 0)
    except (TypeError, ValueError) as e:
        msg = f"invalid Foxglove CompressedVideo timestamp: {timestamp!r}"
        raise ValueError(msg) from e
    if seconds_int < 0 or nanos_int < 0:
        msg = f"Foxglove CompressedVideo timestamp must be non-negative: {timestamp!r}"
        raise ValueError(msg)
    return seconds_int * 1_000_000_000 + nanos_int


def _decode_json_payload(data: bytes, *, topic: str) -> tuple[int | None, str, bytes]:
    """Parse one JSON-encoded Foxglove CompressedVideo payload."""
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        msg = f"failed to parse Foxglove CompressedVideo JSON message on topic {topic!r}"
        raise ValueError(msg) from e
    if not isinstance(payload, dict):
        msg = f"Foxglove CompressedVideo JSON message on topic {topic!r} must be an object"
        raise ValueError(msg)  # noqa: TRY004

    raw_format = payload.get("format")
    if not isinstance(raw_format, str):
        msg = f"Foxglove CompressedVideo JSON message on topic {topic!r} must contain string field 'format'"
        raise ValueError(msg)  # noqa: TRY004
    raw_data = payload.get("data")
    if not isinstance(raw_data, str):
        msg = f"Foxglove CompressedVideo JSON message on topic {topic!r} must contain base64 string field 'data'"
        raise ValueError(msg)  # noqa: TRY004
    try:
        compressed = base64.b64decode(raw_data, validate=True)
    except binascii.Error as e:
        msg = f"failed to decode Foxglove CompressedVideo data on topic {topic!r} as base64"
        raise ValueError(msg) from e
    return _timestamp_to_ns(payload.get("timestamp")), raw_format, compressed


def _decode_protobuf_payload(data: bytes, *, topic: str) -> tuple[int | None, str, bytes]:
    """Parse one protobuf-encoded Foxglove CompressedVideo payload."""
    message = CompressedVideo()
    try:
        message.ParseFromString(data)
    except Exception as e:
        msg = f"failed to parse Foxglove CompressedVideo protobuf message on topic {topic!r}"
        raise ValueError(msg) from e
    timestamp = message.timestamp if message.HasField("timestamp") else None
    return _timestamp_to_ns(timestamp), message.format, bytes(message.data)


def _normalize_format(format_name: str, *, topic: str) -> str:
    """Return a supported lowercase compressed video format."""
    normalized = format_name.strip().lower()
    if not normalized:
        msg = f"Foxglove CompressedVideo message on topic {topic!r} has an empty format"
        raise ValueError(msg)
    if normalized not in _SUPPORTED_FORMATS:
        msg = (
            f"unsupported Foxglove CompressedVideo format {format_name!r} on topic {topic!r}; "
            "supported formats are h264 and h265"
        )
        raise ValueError(msg)
    return normalized


def _validate_message_schema(schema: Schema | None, channel: Channel, *, topic: str) -> None:
    """Validate the MCAP channel/schema surface for CompressedVideo messages."""
    if channel.topic != topic:
        msg = f"expected MCAP topic {topic!r}, got {channel.topic!r}"
        raise ValueError(msg)
    if channel.message_encoding not in {_JSON_ENCODING, _PROTOBUF_ENCODING}:
        msg = (
            f"expected Foxglove CompressedVideo channel with message_encoding='json' or 'protobuf', "
            f"got {channel.message_encoding!r}"
        )
        raise ValueError(msg)
    if schema is not None and schema.name != _COMPRESSED_VIDEO_SCHEMA_NAME:
        msg = f"expected MCAP schema {_COMPRESSED_VIDEO_SCHEMA_NAME!r}, got {schema.name!r}"
        raise ValueError(msg)


def _compressed_video_payload(
    schema: Schema | None, channel: Channel, message: McapMessage, *, topic: str
) -> tuple[int, int | None, str, bytes]:
    """Normalize one MCAP message into timestamp, format, and codec bytes."""
    _validate_message_schema(schema, channel, topic=topic)
    if channel.message_encoding == _JSON_ENCODING:
        foxglove_timestamp_ns, raw_format, data = _decode_json_payload(message.data, topic=topic)
    else:
        foxglove_timestamp_ns, raw_format, data = _decode_protobuf_payload(message.data, topic=topic)
    if not data:
        msg = f"Foxglove CompressedVideo message on topic {topic!r} has empty data"
        raise ValueError(msg)
    return int(message.log_time), foxglove_timestamp_ns, _normalize_format(raw_format, topic=topic), data


def _annex_b_nal_units(data: bytes) -> Iterator[memoryview]:
    """Yield Annex B NAL unit payloads without their start codes."""
    start_codes: list[tuple[int, int]] = []
    i = 0
    while i < len(data) - 3:
        if data[i : i + 3] == b"\x00\x00\x01":
            start_codes.append((i, 3))
            i += 3
        elif i < len(data) - 4 and data[i : i + 4] == b"\x00\x00\x00\x01":
            start_codes.append((i, 4))
            i += 4
        else:
            i += 1

    view = memoryview(data)
    for index, (start, code_len) in enumerate(start_codes):
        payload_start = start + code_len
        payload_end = start_codes[index + 1][0] if index + 1 < len(start_codes) else len(data)
        if payload_start < payload_end:
            yield view[payload_start:payload_end]


def _contains_keyframe(data: bytes, format_name: str) -> bool:
    """Return whether an Annex B H.264/H.265 access unit can start decode."""
    for nal in _annex_b_nal_units(data):
        first_byte = int(nal[0])
        if format_name == "h264" and first_byte & 0x1F == _H264_IDR_NAL_TYPE:
            return True
        if format_name == "h265" and _H265_IRAP_NAL_TYPE_MIN <= ((first_byte >> 1) & 0x3F) <= _H265_IRAP_NAL_TYPE_MAX:
            return True
    return False


class _ForwardCompressedVideoDecoder:
    """Stateful packet-by-packet decoder for one CompressedVideo topic."""

    def __init__(self, format_name: str) -> None:
        self._format = format_name
        self._codec = _SUPPORTED_FORMATS[format_name]
        self._context: Any = av.CodecContext.create(self._codec, "r")
        self._started = False

    def decode(self, format_name: str, log_time_ns: int, data: bytes) -> npt.NDArray[np.uint8]:
        """Decode one access-unit packet and require one display frame."""
        if format_name != self._format:
            msg = f"mixed Foxglove CompressedVideo formats on topic: {self._format!r} then {format_name!r}"
            raise ValueError(msg)
        if not self._started:
            if not _contains_keyframe(data, format_name):
                msg = f"first {format_name} CompressedVideo message must contain a keyframe"
                raise ValueError(msg)
            self._started = True

        frames = self._context.decode(av.Packet(data))
        if len(frames) != 1:
            msg = (
                f"expected one decoded display frame per CompressedVideo message, "
                f"got {len(frames)} for log_time={log_time_ns}"
            )
            raise ValueError(msg)
        frame = frames[0]
        if bool(getattr(self._context, "has_b_frames", False)):
            msg = "CompressedVideo streams with B-frames or reordered decode output are not supported"
            raise ValueError(msg)
        return cast("npt.NDArray[np.uint8]", frame.to_ndarray(format="rgb24"))

    def finish(self) -> None:
        """Flush the decoder and reject delayed display frames."""
        delayed = self._context.decode(None)
        if delayed:
            msg = "CompressedVideo decode produced delayed frames; B-frame/reordered streams are not supported"
            raise ValueError(msg)

    def video_metadata(
        self,
        frame: npt.NDArray[np.uint8],
        *,
        timeline_ns: npt.NDArray[np.int64],
    ) -> VideoMetadata:
        """Build VideoMetadata from PyAV decode state and MCAP log_time cadence."""
        avg_frame_rate = Fraction(0, 1)
        if len(timeline_ns) > 1:
            duration_ns = int(timeline_ns[-1]) - int(timeline_ns[0])
            if duration_ns > 0:
                avg_frame_rate = Fraction((len(timeline_ns) - 1) * 1_000_000_000, duration_ns)
        profile = getattr(self._context, "profile", "") or ""
        pix_fmt = getattr(self._context, "pix_fmt", "") or ""
        bit_rate = getattr(self._context, "bit_rate", 0) or 0
        return VideoMetadata(
            codec_name=self._codec,
            has_bframes=bool(getattr(self._context, "has_b_frames", False)),
            codec_profile=str(profile),
            container_format="mcap",
            height=int(frame.shape[0]),
            width=int(frame.shape[1]),
            avg_frame_rate=avg_frame_rate,
            pix_fmt=str(pix_fmt),
            bit_rate_bps=int(bit_rate),
        )


def _validate_next_window(window: SamplingWindow, state: _WindowValidationState) -> None:
    """Reject overlapping or non-monotonic sampling windows incrementally."""
    if state.previous_start is not None and window.start_ns < state.previous_start:
        msg = "SamplingSpec.grid windows must be monotonically increasing"
        raise ValueError(msg)
    if state.previous_end is not None and window.start_ns < state.previous_end:
        msg = "McapCameraSensor requires non-overlapping sampling windows"
        raise ValueError(msg)
    state.previous_start = int(window.start_ns)
    state.previous_end = int(window.exclusive_end_ns)


class McapCameraSensor:
    """MCAP camera sensor for Foxglove CompressedVideo topics.

    The sensor reads the configured camera topic in MCAP ``log_time`` order and
    feeds compressed H.264/H.265 access units to a single forward PyAV decoder.
    It intentionally uses recorder ``message.log_time`` for sampling, alignment,
    and ``timestamps_ns``; the emitted ``CameraData.sensor_timestamps_ns`` and
    ``pts_stream`` currently carry that same recorder time. This sensor does not
    yet expose acquisition-time provenance. The Foxglove
    ``CompressedVideo.timestamp`` payload field may be present, but it is not
    used as the sampling timeline.
    """

    def __init__(self, source: DataSource, topic: str = "/camera/rgb") -> None:
        """Initialize the MCAP camera sensor."""
        self._topic = topic
        self._mcap = McapTopicAccessor(source, topic)
        self._video_metadata: VideoMetadata | None = None
        self._empty_camera_data: CameraData | None = None

    @property
    def video_metadata(self) -> VideoMetadata:
        """Return metadata derived from the first decoded compressed frame."""
        if self._video_metadata is None:
            self._video_metadata = self._load_video_metadata()
        return self._video_metadata

    def _load_video_metadata(self) -> VideoMetadata:
        """Decode the first display frame and derive minimal video metadata in one MCAP reader pass."""
        first_packet: tuple[int, int | None, str, bytes] | None = None
        timeline_ns: list[int] = []
        with self._mcap.open_reader() as reader:
            for schema, channel, message in reader.iter_messages(topics=self._topic, log_time_order=True):
                packet = _compressed_video_payload(schema, channel, message, topic=self._topic)
                timeline_ns.append(packet[0])
                if first_packet is None:
                    first_packet = packet

        if first_packet is None:
            msg = f"no MCAP messages on topic {self._topic!r}"
            raise ValueError(msg)

        _log_time_ns, _foxglove_timestamp_ns, format_name, data = first_packet
        decoder = _ForwardCompressedVideoDecoder(format_name)
        frame = decoder.decode(format_name, _log_time_ns, data)
        return decoder.video_metadata(frame, timeline_ns=np.array(timeline_ns, dtype=np.int64))

    @property
    def start_ns(self) -> int:
        """Earliest frame time on this topic, in nanoseconds."""
        return self._mcap.start_ns

    @property
    def end_ns(self) -> int:
        """Latest frame time on this topic, in nanoseconds."""
        return self._mcap.end_ns

    @property
    def max_gap_ns(self) -> int:
        """Return maximum expected gap duration in nanoseconds."""
        return self._mcap.max_gap_ns

    @property
    def timestamps_ns(self) -> npt.NDArray[np.int64]:
        """Canonical sampling timestamps in nanoseconds (raw MCAP ``message.log_time``)."""
        return self._mcap.timestamps_ns

    def stream_timestamps(self, batch_size: int = 0) -> Iterator[npt.NDArray[np.int64]]:
        """Not implemented: the timestamp stream is camera-only for now.

        See :meth:`CameraSensor.stream_timestamps`.
        """
        del batch_size
        raise NotImplementedError(STREAM_TIMESTAMPS_CAMERA_ONLY_MSG)

    def supports_sampling_policy(self, policy: object) -> bool:
        """Return whether this sensor can sample with *policy*."""
        return isinstance(policy, NearestTimestampPolicy)

    def _get_empty_camera_data(self, metadata: VideoMetadata | None = None) -> CameraData:
        """Return a cached empty batch preserving the expected frame shape."""
        if self._empty_camera_data is None:
            metadata = self.video_metadata if metadata is None else metadata
            empty_ts = np.empty(0, dtype=np.int64)
            empty_frames = np.empty((0, metadata.height, metadata.width, _RGB_CHANNELS), dtype=np.uint8)
            self._empty_camera_data = CameraData(
                align_timestamps_ns=empty_ts,
                sensor_timestamps_ns=empty_ts,
                pts_stream=empty_ts,
                frames=empty_frames,
                metadata=metadata,
            )
        return self._empty_camera_data

    def _sample_window(
        self,
        window: SamplingWindow,
        decoded_log_times_ns: list[int],
        decoded_frames: list[npt.NDArray[np.uint8]],
        *,
        policy: NearestTimestampPolicy,
        metadata: VideoMetadata,
    ) -> CameraData:
        """Build a ``CameraData`` batch for one decoded window."""
        if len(window) == 0 or not decoded_frames:
            return self._get_empty_camera_data(metadata)

        log_times_ns = np.array(decoded_log_times_ns, dtype=np.int64)
        indices, _counts = sample_window_indices(log_times_ns, window, policy=policy, dedup=False)
        sampled_sensor_timestamps_ns = log_times_ns[indices]
        sampled_sensor_timestamps_ns.flags.writeable = False
        frames = np.stack([decoded_frames[int(index)] for index in indices]).astype(np.uint8, copy=False)
        return CameraData(
            align_timestamps_ns=window.timestamps_ns,
            sensor_timestamps_ns=sampled_sensor_timestamps_ns,
            pts_stream=sampled_sensor_timestamps_ns,
            frames=frames,
            metadata=metadata,
        )

    def _decode_compressed_message(
        self,
        schema: Schema | None,
        channel: Channel,
        message: McapMessage,
        state: _DecodeStreamState,
    ) -> tuple[int, npt.NDArray[np.uint8]]:
        """Decode one CompressedVideo MCAP message and update stream state."""
        log_time_ns, _foxglove_timestamp_ns, format_name, data = _compressed_video_payload(
            schema, channel, message, topic=self._topic
        )
        if state.expected_format is None:
            state.expected_format = format_name
            state.decoder = _ForwardCompressedVideoDecoder(format_name)
        elif format_name != state.expected_format:
            msg = (
                f"mixed Foxglove CompressedVideo formats on topic {self._topic!r}: "
                f"{state.expected_format!r} then {format_name!r}"
            )
            raise ValueError(msg)
        if state.previous_frame_time is not None and log_time_ns <= state.previous_frame_time:
            msg = "MCAP CompressedVideo log_time values must be strictly increasing"
            raise ValueError(msg)
        state.previous_frame_time = log_time_ns

        if state.decoder is None:
            msg = "CompressedVideo decoder was not initialized"
            raise RuntimeError(msg)
        frame = state.decoder.decode(format_name, log_time_ns, data)
        state.decoded_log_times_ns.append(log_time_ns)
        self._video_metadata = state.decoder.video_metadata(
            frame, timeline_ns=np.array(state.decoded_log_times_ns, dtype=np.int64)
        )
        self._empty_camera_data = None
        return log_time_ns, frame

    def _decode_until_window_end(
        self,
        messages: Iterator[tuple[Schema | None, Channel, McapMessage]],
        window: SamplingWindow,
        state: _DecodeStreamState,
    ) -> tuple[list[int], list[npt.NDArray[np.uint8]]]:
        """Decode packets up to one window end and return frames inside it."""
        decoded_log_times_ns: list[int] = []
        decoded_frames: list[npt.NDArray[np.uint8]] = []
        while True:
            if state.pending is None:
                try:
                    state.pending = next(messages)
                except StopIteration:
                    break

            schema, channel, message = state.pending
            if int(message.log_time) >= window.exclusive_end_ns:
                break
            state.pending = None
            log_time_ns, frame = self._decode_compressed_message(schema, channel, message, state)
            if window.start_ns <= log_time_ns < window.exclusive_end_ns:
                decoded_log_times_ns.append(log_time_ns)
                decoded_frames.append(frame)
        return decoded_log_times_ns, decoded_frames

    def _ensure_metadata_available(
        self,
        messages: Iterator[tuple[Schema | None, Channel, McapMessage]],
        state: _DecodeStreamState,
    ) -> None:
        """Decode one pending or upcoming message so empty batches can carry metadata."""
        if self._video_metadata is not None:
            return
        if state.pending is None:
            try:
                state.pending = next(messages)
            except StopIteration as e:
                msg = f"no MCAP messages on topic {self._topic!r}"
                raise ValueError(msg) from e
        schema, channel, message = state.pending
        state.pending = None
        self._decode_compressed_message(schema, channel, message, state)

    def _sample_ready_windows(
        self,
        pending_windows: list[tuple[SamplingWindow, list[int], list[npt.NDArray[np.uint8]]]],
        *,
        policy: NearestTimestampPolicy,
    ) -> Generator[CameraData]:
        """Yield decoded windows once metadata is available."""
        if self._video_metadata is None:
            return
        metadata = self._video_metadata
        while pending_windows:
            window, log_times_ns, frames = pending_windows.pop(0)
            yield self._sample_window(window, log_times_ns, frames, policy=policy, metadata=metadata)

    def sample(self, spec: SamplingSpec, *, policy: NearestTimestampPolicy) -> Generator[CameraData]:
        """Sample camera frames according to a forward-only ``SamplingSpec``."""
        policy = require_nearest_timestamp_policy(policy, sensor_name=type(self).__name__)
        windows = iter(spec.grid)
        try:
            first_window = next(windows)
        except StopIteration:
            return

        window_state = _WindowValidationState()
        _validate_next_window(first_window, window_state)
        initial_windows = [first_window]
        try:
            second_window = next(windows)
        except StopIteration:
            pass
        else:
            _validate_next_window(second_window, window_state)
            initial_windows.append(second_window)

        decode_state = _DecodeStreamState()
        pending_windows: list[tuple[SamplingWindow, list[int], list[npt.NDArray[np.uint8]]]] = []
        with self._mcap.open_reader() as reader:
            messages = self._mcap.iter_messages(
                reader,
                0,
                _MAX_MCAP_TIME_NS,
                log_time_order=True,
            )
            try:
                for window in initial_windows:
                    log_times_ns, frames = self._decode_until_window_end(messages, window, decode_state)
                    pending_windows.append((window, log_times_ns, frames))
                    yield from self._sample_ready_windows(pending_windows, policy=policy)
                for window in windows:
                    _validate_next_window(window, window_state)
                    log_times_ns, frames = self._decode_until_window_end(messages, window, decode_state)
                    pending_windows.append((window, log_times_ns, frames))
                    yield from self._sample_ready_windows(pending_windows, policy=policy)
                if pending_windows:
                    self._ensure_metadata_available(messages, decode_state)
                    yield from self._sample_ready_windows(pending_windows, policy=policy)
            finally:
                if decode_state.decoder is not None:
                    decode_state.decoder.finish()
