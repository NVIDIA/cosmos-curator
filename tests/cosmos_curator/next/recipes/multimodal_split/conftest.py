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

"""Fixtures for ``multimodal-split`` processing tests."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, timestamp_pb2
from mcap.writer import Writer as McapWriter

from cosmos_curator.core.sensors.utils.timestamps import write_timestamps_ns
from cosmos_curator.next.recipes.multimodal_split.config import ResolvedMultimodalSplitConfig
from cosmos_curator.next.recipes.multimodal_split.plan import span_planner
from cosmos_curator.next.recipes.multimodal_split.split import split_session

_NS_PER_SECOND = 1_000_000_000
_WIDTH = 32
_HEIGHT = 16

# A recorder's clock when it started this session, taken from real footage: an
# uptime counter, nowhere near zero and nowhere near a wall-clock epoch. Fixtures
# default to it because a session that starts at zero cannot tell a sensor
# timeline apart from a presentation one, and that is the whole distinction the
# recipe now turns on.
DEVICE_ORIGIN_NS = 3_517_693_838_000


@pytest.fixture
def make_session() -> Callable[..., Path]:
    """Return a builder for a session directory holding one camera video.

    Frame ``i`` is a flat image of brightness ``i * step``, which makes the clip
    monotonically brighten and gives a decoded frame an approximate index. It is
    only approximate: at the default step of 2, lossy encoding collides adjacent
    frames, so decoded means are ascending but not distinct. A test that needs to
    name *which* source frame was selected should read
    ``selected_source_timestamps_ns``, which is exact, or raise the step first.
    """

    def build(
        root: Path,
        session_id: str,
        *,
        frames: int = 90,
        fps: int = 30,
        uri_pattern: str = "camera/front.mp4",
        step: int = 2,
        origin_ns: int = DEVICE_ORIGIN_NS,
        write_timestamps: bool = True,
    ) -> Path:
        path = root / session_id / uri_pattern
        path.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("libopenh264", rate=fps)
            stream.width, stream.height, stream.pix_fmt = _WIDTH, _HEIGHT, "yuv420p"
            for index in range(frames):
                image = np.full((_HEIGHT, _WIDTH, 3), (index * step) % 256, dtype=np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)

        # The sensor-timestamp artifact the recorder's converter writes beside
        # every camera. Both paths take their bounds from this instead of from the
        # container, so a session without one is not processable. It carries the
        # recorder's clock, which is what makes a camera clip joinable with the
        # other sensors; the container's own timeline starts at zero.
        # round(k * period), not k * round(period): a period that is not a whole
        # number of nanoseconds drifts otherwise, and the drift costs a clip.
        #
        # Skippable so a test can stand up a session whose camera has no capture
        # sidecar at all, which is the case ``allow_pts_fallback`` exists for.
        if write_timestamps:
            period_ns = _NS_PER_SECOND / fps
            timestamps_ns = np.array([origin_ns + round(index * period_ns) for index in range(frames)], dtype=np.int64)
            with (path.parent / f"{path.name}.timestamps.npy").open("wb") as timeline:
                write_timestamps_ns(timestamps_ns, timeline)
        return root / session_id

    return build


EGO_TOPIC = "/egotrajectory"
IMU_TOPIC = "/imu"


def _write_copied_sensor_mcap(
    session: Path,
    *,
    schema_name: str,
    first_ns: int,
    count: int,
    rate_hz: int,
    uri_pattern: str,
    topic: str,
    gap: tuple[int, int] | None,
    chunked: bool,
    extra_topic: str | None,
) -> Path:
    """Write one COPIED_SENSOR_TYPES sensor's MCAP -- the shared mechanics behind make_ego_mcap/make_imu_mcap.

    Written with the upstream MCAP writer rather than the recipe's own, because
    this stands in for what a recorder produced: the recipe reads these files and
    does not write them, and the options that matter here -- chunking, a second
    topic -- are properties of the recorder, not of the cut.

    The payloads are opaque bytes rather than encoded messages. Nothing in this
    recipe decodes them: planning reads the topic's first and last ``log_time``
    from the index, and the cut copies records through unexamined. A fixture that
    encoded real payloads would assert the recipe understands a schema it is
    deliberately built not to.

    ``gap`` is a dropout, as a half-open window of indices. A recorder that
    loses the sensor for a while still writes an MCAP whose bounds span the
    whole session, which is why planning cannot see the hole.
    """
    period_ns = _NS_PER_SECOND // rate_hz
    skipped = range(*gap) if gap is not None else range(0)
    path = session / uri_pattern
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        writer = McapWriter(handle, chunk_size=10_000, use_chunking=chunked)
        writer.start(library="test recorder")
        schema_id = writer.register_schema(name=schema_name, encoding="test-schema", data=b"schema-bytes")
        channels = {
            name: writer.register_channel(schema_id=schema_id, topic=name, message_encoding="test-bytes")
            for name in ([topic] if extra_topic is None else [topic, extra_topic])
        }
        for index in range(count):
            if index in skipped:
                continue
            log_time = first_ns + index * period_ns
            for name, channel_id in channels.items():
                writer.add_message(
                    channel_id=channel_id,
                    log_time=log_time,
                    data=f"{name}-{index}".encode(),
                    publish_time=log_time,
                    sequence=index,
                )
        writer.finish()
    return path


@pytest.fixture
def make_ego_mcap() -> Callable[..., Path]:
    """Return a builder for a session's egotrajectory MCAP -- see _write_copied_sensor_mcap for the mechanics."""

    def build(
        session: Path,
        *,
        first_ns: int = DEVICE_ORIGIN_NS,
        # One past the camera's last frame, which is how a recorder behaves: the
        # trajectory is still logging when the camera stops. At exactly 300 the
        # trajectory would end 10ms early and cost the session its third clip,
        # which is the narrowing rule working, not a fixture quirk -- the test
        # that wants it asks for it.
        count: int = 301,
        rate_hz: int = 100,
        uri_pattern: str = "ego/egotrajectory.mcap",
        topic: str = EGO_TOPIC,
        gap: tuple[int, int] | None = None,
        chunked: bool = True,
        extra_topic: str | None = None,
    ) -> Path:
        return _write_copied_sensor_mcap(
            session,
            schema_name="test.Pose",
            first_ns=first_ns,
            count=count,
            rate_hz=rate_hz,
            uri_pattern=uri_pattern,
            topic=topic,
            gap=gap,
            chunked=chunked,
            extra_topic=extra_topic,
        )

    return build


@pytest.fixture
def make_imu_mcap() -> Callable[..., Path]:
    """Return a builder for a session's IMU MCAP -- see _write_copied_sensor_mcap for the mechanics.

    Same shape as ``make_ego_mcap`` and for the same reason: IMU is a copied
    sensor too (see config.py's ``COPIED_SENSOR_TYPES`` docstring -- nearest
    neighbor/interpolate would each discard nearly every sample of a ~100-200 Hz
    inertial stream, and what a consumer actually wants is a pose delta
    integrated over the interval, not a point sample of one; that integration is
    deferred, not done here).
    """

    def build(
        session: Path,
        *,
        first_ns: int = DEVICE_ORIGIN_NS,
        # 376 at 125 Hz places the last message at exactly 3.000s, the same
        # camera-session coverage make_ego_mcap's 301-at-100-Hz default gives.
        count: int = 376,
        rate_hz: int = 125,
        uri_pattern: str = "imu/imu.mcap",
        topic: str = IMU_TOPIC,
        gap: tuple[int, int] | None = None,
        chunked: bool = True,
        extra_topic: str | None = None,
    ) -> Path:
        return _write_copied_sensor_mcap(
            session,
            schema_name="test.ImuSample",
            first_ns=first_ns,
            count=count,
            rate_hz=rate_hz,
            uri_pattern=uri_pattern,
            topic=topic,
            gap=gap,
            chunked=chunked,
            extra_topic=extra_topic,
        )

    return build


GPS_TOPIC = "/gps"

#: The Foxglove message the fleet's recorders write GPS with, reduced to the
#: fields the checked-in mapping reads plus the metadata map the real writer uses.
#:
#: Built as a descriptor rather than compiled from .proto text: a fixture that
#: shells out to protoc depends on a binary being present and executable in the
#: environment, which it is not here, and a unit test should not need a compiler.
_FIELD = descriptor_pb2.FieldDescriptorProto


def _location_fix_descriptor_set(*, timestamp_as_int: bool = False, singular_metadata: bool = False) -> bytes:
    """Build the LocationFix descriptor set the fixture MCAPs carry."""
    file_proto = descriptor_pb2.FileDescriptorProto(
        name="location_fix.proto",
        package="foxglove",
        syntax="proto3",
        # The real message stamps with the well-known type, and the checked-in
        # mapping reads it as one. A lookalike of the same shape is rejected.
        dependency=["google/protobuf/timestamp.proto"],
    )

    pair_proto = file_proto.message_type.add(name="KeyValuePair")
    pair_proto.field.add(name="key", number=1, type=_FIELD.TYPE_STRING, label=_FIELD.LABEL_OPTIONAL)
    pair_proto.field.add(name="value", number=2, type=_FIELD.TYPE_STRING, label=_FIELD.LABEL_OPTIONAL)

    fix_proto = file_proto.message_type.add(name="LocationFix")
    if timestamp_as_int:
        fix_proto.field.add(name="timestamp", number=1, type=_FIELD.TYPE_INT64, label=_FIELD.LABEL_OPTIONAL)
    else:
        fix_proto.field.add(
            name="timestamp",
            number=1,
            type=_FIELD.TYPE_MESSAGE,
            label=_FIELD.LABEL_OPTIONAL,
            type_name=".google.protobuf.Timestamp",
        )
    fix_proto.field.add(name="frame_id", number=2, type=_FIELD.TYPE_STRING, label=_FIELD.LABEL_OPTIONAL)
    for number, name in ((3, "latitude"), (4, "longitude"), (5, "altitude")):
        fix_proto.field.add(name=name, number=number, type=_FIELD.TYPE_DOUBLE, label=_FIELD.LABEL_OPTIONAL)
    fix_proto.field.add(
        name="metadata",
        number=10,
        type=_FIELD.TYPE_MESSAGE,
        label=_FIELD.LABEL_OPTIONAL if singular_metadata else _FIELD.LABEL_REPEATED,
        type_name=".foxglove.KeyValuePair",
    )

    timestamp_proto = descriptor_pb2.FileDescriptorProto()
    timestamp_proto.ParseFromString(timestamp_pb2.DESCRIPTOR.serialized_pb)
    return descriptor_pb2.FileDescriptorSet(file=[timestamp_proto, file_proto]).SerializeToString()


@pytest.fixture
def make_gps_mcap() -> Callable[..., Path]:
    """Return a builder for a session's GPS MCAP in the recorded Foxglove schema.

    Real positions rather than constants: alignment's whole job is to place a
    value between or onto fixes, so a fixture whose fixes are identical could not
    tell nearest selection from interpolation.
    """

    def build(
        session: Path,
        *,
        first_ns: int = DEVICE_ORIGIN_NS,
        count: int = 21,
        rate_hz: float = 1.0,
        uri_pattern: str = "gps/gps.mcap",
        topic: str = GPS_TOPIC,
        latitude: float = 37.4,
        longitude: float = -122.25,
        altitude: float = 30.0,
        # Roughly 11 m of latitude per fix, a plausible 11 m/s at 1 Hz. Every
        # component moves, and by a different amount, so a test can tell a
        # longitude written into the altitude field from one written correctly.
        step_deg: float = 0.0001,
        longitude_step_deg: float = 0.00025,
        altitude_step_m: float = 0.5,
        # Indices to leave out, so a test can drive the bound that refuses a row
        # whose supporting fixes are too far apart.
        dropped: frozenset[int] = frozenset(),
        topic_override: str | None = None,
        # Spell a field correctly but give it the wrong shape, so a test can
        # reach the case a name-only check would wave through.
        timestamp_as_int: bool = False,
        singular_metadata: bool = False,
        # Offset the payload timestamp from log_time, which is what a converter
        # that transforms one and not the other produces.
        payload_skew_ns: int = 0,
    ) -> Path:
        descriptor = _location_fix_descriptor_set(
            timestamp_as_int=timestamp_as_int, singular_metadata=singular_metadata
        )
        pool = descriptor_pool.DescriptorPool()
        for file_descriptor in descriptor_pb2.FileDescriptorSet.FromString(descriptor).file:
            pool.Add(file_descriptor)
        message_cls = message_factory.GetMessageClass(pool.FindMessageTypeByName("foxglove.LocationFix"))

        period_ns = int(_NS_PER_SECOND / rate_hz)
        path = session / uri_pattern
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            writer = McapWriter(handle, chunk_size=10_000)
            writer.start(library="test recorder")
            schema_id = writer.register_schema(name="foxglove.LocationFix", encoding="protobuf", data=descriptor)
            channel_id = writer.register_channel(
                schema_id=schema_id, topic=topic_override or topic, message_encoding="protobuf"
            )
            for index in range(count):
                if index in dropped:
                    continue
                log_time = first_ns + index * period_ns
                message = message_cls()
                payload_ns = log_time + payload_skew_ns
                if timestamp_as_int:
                    message.timestamp = payload_ns
                else:
                    message.timestamp.seconds = payload_ns // _NS_PER_SECOND
                    message.timestamp.nanos = payload_ns % _NS_PER_SECOND
                message.latitude = latitude + index * step_deg
                message.longitude = longitude + index * longitude_step_deg
                message.altitude = altitude + index * altitude_step_m
                writer.add_message(
                    channel_id=channel_id,
                    log_time=log_time,
                    data=message.SerializeToString(),
                    publish_time=log_time,
                    sequence=index,
                )
            writer.finish()
        return path

    return build


def gps_source_fixes(
    *,
    first_ns: int = DEVICE_ORIGIN_NS,
    count: int = 21,
    rate_hz: float = 1.0,
    latitude: float = 37.4,
    longitude: float = -122.25,
    altitude: float = 30.0,
    step_deg: float = 0.0001,
    longitude_step_deg: float = 0.00025,
    altitude_step_m: float = 0.5,
) -> dict[int, tuple[float, float, float]]:
    """Return what ``make_gps_mcap`` wrote, keyed by fix time.

    A test comparing a clip's rows against the source needs the source, and
    reading it back out of the fixture MCAP would let one bug hide another.
    """
    period_ns = int(_NS_PER_SECOND / rate_hz)
    return {
        first_ns + index * period_ns: (
            latitude + index * step_deg,
            longitude + index * longitude_step_deg,
            altitude + index * altitude_step_m,
        )
        for index in range(count)
    }


def split_all(
    item: dict[str, Any],
    *,
    config: ResolvedMultimodalSplitConfig,
    storage_profile: str = "default",
) -> list[dict[str, Any]]:
    """Plan one session and split every camera it planned, as the pipeline does.

    Splitting takes a planned camera now, not a session, so a test that wants a
    session's clips runs both stages -- which is also what keeps them honest
    about the row the planner actually hands over.
    """
    outcomes = [
        outcome
        for plan in span_planner(item, config=config, storage_profile=storage_profile)
        if plan["result_kind"] == "camera_plan"
        for outcome in split_session(plan, config=config, storage_profile=storage_profile)
    ]
    return [
        {
            **outcome,
            "clip_id": outcome["split_clip_id"],
            "sensor_key": outcome["sensor_key"],
            "clip_location": outcome["media_location"],
            "clip_bytes": outcome["media_size_bytes"],
        }
        for outcome in outcomes
        if outcome["status"] == "success"
    ]


@pytest.fixture
def make_config() -> Callable[..., ResolvedMultimodalSplitConfig]:
    """Return a builder for a resolved config pointed at a temporary tree."""

    def build(
        input_root: Path,
        media_root: Path,
        *,
        duration_s: float = 1.0,
        output_fps: int = 30,
        max_delta_s: float = 0.05,
        uri_pattern: str = "camera/front.mp4",
        camera_key: str = "front",
        cameras: dict[str, str] | None = None,
        optional_cameras: frozenset[str] = frozenset(),
        pts_fallback_cameras: frozenset[str] = frozenset(),
        mcap_sensors: dict[str, str] | None = None,
        optional_sensors: frozenset[str] = frozenset(),
        topic: str | None = None,
        imu_sensors: dict[str, str] | None = None,
        imu_topic: str | None = None,
        gps_sensors: dict[str, str] | None = None,
        # No default, because the config has none: a test that forgets to say
        # how its GPS is sampled should fail the way a config would.
        sampling: str | None = None,
        max_gap_ms: float | None = None,
        max_delta_ms: float | None = None,
        gps_topic: str | None = None,
        video_bitrate: str = "400K",
        gop_size: int = 5,
        progress: bool = False,
        storage_profile: str = "default",
        memory_gb: float | None = None,
        planner_num_cpus: float = 0.125,
        read_buffer_bytes: int | None = None,
    ) -> ResolvedMultimodalSplitConfig:
        camera_map = cameras or {camera_key: uri_pattern}
        return ResolvedMultimodalSplitConfig.model_validate(
            {
                "schema_version": 1,
                "kind": "multimodal-split",
                "input": {"input_path_prefix": str(input_root)},
                "sensors": {
                    **{
                        key: {
                            "sensor_type": "camera",
                            "uri_pattern": pattern,
                            "required": key not in optional_cameras,
                            "allow_pts_fallback": key in pts_fallback_cameras,
                        }
                        for key, pattern in camera_map.items()
                    },
                    **{
                        key: {
                            "sensor_type": "gps",
                            "uri_pattern": pattern,
                            "required": key not in optional_sensors,
                            **({} if sampling is None else {"sampling": sampling}),
                            **({} if max_gap_ms is None else {"max_gap_ms": max_gap_ms}),
                            **({} if max_delta_ms is None else {"max_delta_ms": max_delta_ms}),
                            **({} if gps_topic is None else {"topic": gps_topic}),
                        }
                        for key, pattern in (gps_sensors or {}).items()
                    },
                    **{
                        key: {
                            "sensor_type": "egotrajectory",
                            "uri_pattern": pattern,
                            "required": key not in optional_sensors,
                            "topic": topic,
                        }
                        for key, pattern in (mcap_sensors or {}).items()
                    },
                    **{
                        key: {
                            "sensor_type": "imu",
                            "uri_pattern": pattern,
                            "required": key not in optional_sensors,
                            "topic": imu_topic,
                        }
                        for key, pattern in (imu_sensors or {}).items()
                    },
                },
                "output": {"path": str(media_root)},
                "clip": {"duration_s": duration_s, "sampling_fps": output_fps},
                "video": {
                    "max_delta_s": max_delta_s,
                    "encoder": {"video_bitrate": video_bitrate, "gop_size": gop_size},
                },
                "execution": {
                    "progress": progress,
                    "storage_profile": storage_profile,
                    "split_session": {"memory_gb": memory_gb},
                    "span_planner": {"num_cpus": planner_num_cpus},
                    "read_buffer_bytes": read_buffer_bytes,
                },
            }
        )

    return build
