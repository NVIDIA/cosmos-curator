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
"""MCAP protobuf ego-trajectory sensor."""

from collections.abc import Generator, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from google.protobuf.message import Message
from mcap.reader import McapReader
from mcap.records import Channel, Schema

from cosmos_curator.core.sensors.alignment.ego_trajectory import align_ego_trajectory_nearest
from cosmos_curator.core.sensors.data.egotrajectory_data import EgoTrajectory
from cosmos_curator.core.sensors.data.extrinsics import SensorExtrinsics
from cosmos_curator.core.sensors.sampling.policy import NearestTimestampPolicy, require_nearest_timestamp_policy
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors.group import STREAM_TIMESTAMPS_CAMERA_ONLY_MSG
from cosmos_curator.core.sensors.types.types import DataSource
from cosmos_curator.core.sensors.utils.mcap import (
    EmptyMcapTopicError,
    McapProtobufMessageResolver,
    McapTopicAccessor,
    parse_protobuf_message,
)
from cosmos_curator.core.sensors.utils.protobuf_mapper import ProtobufRowMapper

DEFAULT_TOPIC = "/egotrajectory"
DEFAULT_FRAME = "world_enu"
_POSITION_COMPONENTS = 3
_QUATERNION_COMPONENTS = 4
REQUIRED_EGO_TRAJECTORY_MAPPING_FIELDS = frozenset(
    {"sensor_timestamp_ns", "position_m", "orientation_xyzw", "pose_valid"}
)

PositionVector = tuple[float, float, float]
QuaternionXyzw = tuple[float, float, float, float]


@dataclass(frozen=True)
class DecodedEgoTrajectorySample:
    """Decoded ego-trajectory protobuf fields normalized for ``EgoTrajectory``."""

    sensor_timestamp_ns: int
    align_timestamp_ns: int
    position_m: PositionVector
    orientation_xyzw: QuaternionXyzw
    pose_valid: bool = True
    host_timestamp_ns: int | None = None
    sequence_counter: int | None = None


def _ego_trajectory_from_samples(
    samples: list[DecodedEgoTrajectorySample],
    *,
    frame: str,
    host: bool,
    sequence: bool,
) -> EgoTrajectory:
    """Build an ``EgoTrajectory`` batch from decoded samples."""
    align = np.array([sample.align_timestamp_ns for sample in samples], dtype=np.int64)
    seq_values = [sample.sequence_counter for sample in samples]
    if sequence and any(v is None or int(v) < 0 or int(v) > int(np.iinfo(np.uint64).max) for v in seq_values):
        msg = "sequence_counter values must be present and fit uint64"
        raise ValueError(msg)
    host_values = [sample.host_timestamp_ns for sample in samples]
    if host and any(v is None for v in host_values):
        msg = "host_timestamp_ns must be present for every ego-trajectory sample"
        raise ValueError(msg)
    return EgoTrajectory(
        align_timestamps_ns=align,
        sensor_timestamps_ns=np.array([sample.sensor_timestamp_ns for sample in samples], dtype=np.int64),
        positions_m=np.asarray([sample.position_m for sample in samples], dtype=np.float64),
        orientations_xyzw=np.asarray([sample.orientation_xyzw for sample in samples], dtype=np.float64),
        pose_valid=np.array([sample.pose_valid for sample in samples], dtype=np.bool_),
        host_timestamps_ns=np.array(host_values, dtype=np.int64) if host else None,
        sequence_counter=np.array(seq_values, dtype=np.uint64) if sequence else None,
        frame=frame,
    )


def _empty_ego_trajectory(
    *,
    frame: str,
    host: bool,
    sequence: bool,
) -> EgoTrajectory:
    """Build an empty ego-trajectory sample batch."""
    empty_ts = np.empty(0, dtype=np.int64)
    return EgoTrajectory(
        align_timestamps_ns=empty_ts,
        sensor_timestamps_ns=empty_ts,
        positions_m=np.empty((0, _POSITION_COMPONENTS), dtype=np.float64),
        orientations_xyzw=np.empty((0, _QUATERNION_COMPONENTS), dtype=np.float64),
        pose_valid=np.empty(0, dtype=np.bool_),
        host_timestamps_ns=np.empty(0, dtype=np.int64) if host else None,
        sequence_counter=np.empty(0, dtype=np.uint64) if sequence else None,
        frame=frame,
    )


class EgoTrajectorySensor:
    """MCAP-backed ego-trajectory sensor.

    Requires ``NearestTimestampPolicy`` and decodes all mapped protobuf poses
    in the configured clip once per :meth:`sample` invocation. It selects one
    pose per requested timestamp from the clip's complete candidate set.
    ``schema_name`` selects the message declared by the MCAP's embedded
    descriptor, and ``protobuf_mapping`` maps that message into
    ``EgoTrajectory`` fields. Optional output arrays are present only when
    their corresponding values are mapped: ``host_timestamp_ns`` or
    ``sequence_counter``. Empty and populated output windows preserve the same
    optional-array presence.

    Poses are sensor-origin ``T_worldENU_from_sensorBody`` transforms.
    Configured rig extrinsics are retained for downstream pipeline composition;
    selected output preserves the wire pose. ``frame`` is library metadata
    naming the clip-local ENU world (default ``world_enu``).

    Every populated batch requires strictly increasing alignment timestamps;
    duplicate or decreasing values raise ``ValueError``. Mapping omission leaves
    source ``align_timestamps_ns`` equal to ``sensor_timestamps_ns``.
    """

    def __init__(  # noqa: PLR0913
        self,
        source: DataSource,
        topic: str = DEFAULT_TOPIC,
        *,
        schema_name: str,
        protobuf_mapping: str | Path | Mapping[str, Any],
        frame: str = DEFAULT_FRAME,
        extrinsics: SensorExtrinsics | None = None,
    ) -> None:
        """Initialize the MCAP ego-trajectory sensor."""
        if not frame:
            msg = "frame must be a non-empty string"
            raise ValueError(msg)
        self._topic = topic
        self._frame = frame
        self._extrinsics = extrinsics
        self._mcap = McapTopicAccessor(source, topic)
        self._protobuf_resolver = McapProtobufMessageResolver(schema_name, schema_label="ego-trajectory protobuf")
        self._protobuf_row_mapper = ProtobufRowMapper(protobuf_mapping, target_cls=DecodedEgoTrajectorySample)
        mapped = self._protobuf_row_mapper.mapped_target_names
        missing = sorted(REQUIRED_EGO_TRAJECTORY_MAPPING_FIELDS - mapped)
        if missing:
            msg = "ego-trajectory protobuf mapping must map required field(s): " + ", ".join(map(repr, missing))
            raise ValueError(msg)
        self._include_host = "host_timestamp_ns" in mapped
        self._include_sequence = "sequence_counter" in mapped
        self._empty_ego_trajectory: EgoTrajectory | None = None

    @property
    def start_ns(self) -> int:
        """Earliest ego-trajectory message time on this topic, in nanoseconds."""
        return self._mcap.start_ns

    @property
    def end_ns(self) -> int:
        """Latest ego-trajectory message time on this topic, in nanoseconds."""
        return self._mcap.end_ns

    @property
    def max_gap_ns(self) -> int:
        """Return maximum expected gap duration in nanoseconds."""
        return self._mcap.max_gap_ns

    @property
    def timestamps_ns(self) -> npt.NDArray[np.int64]:
        """Message times in nanoseconds from raw MCAP ``log_time`` values."""
        return self._mcap.timestamps_ns

    def stream_timestamps(self, batch_size: int = 0) -> Iterator[npt.NDArray[np.int64]]:
        """Not implemented: the timestamp stream is camera-only for now.

        See :meth:`CameraSensor.stream_timestamps`.
        """
        del batch_size
        raise NotImplementedError(STREAM_TIMESTAMPS_CAMERA_ONLY_MSG)

    def _resolve_message_class(self, reader: McapReader) -> type[Message] | None:
        """Resolve and cache the message class from the MCAP summary when available."""
        return self._protobuf_resolver.resolve_from_summary(reader, self._topic)

    def _message_class_for_message(self, schema: Schema | None, channel: Channel) -> type[Message]:
        """Resolve the dynamic message class and validate each message channel."""
        return self._protobuf_resolver.message_class_for_message(schema, channel, topic=self._topic)

    def _get_empty_ego_trajectory(self) -> EgoTrajectory:
        """Return a cached empty ego-trajectory sample batch."""
        if self._empty_ego_trajectory is None:
            self._empty_ego_trajectory = _empty_ego_trajectory(
                frame=self._frame,
                host=self._include_host,
                sequence=self._include_sequence,
            )
        return self._empty_ego_trajectory

    def _read_messages(
        self,
        reader: McapReader,
        start_ns: int,
        exclusive_end_ns: int,
    ) -> EgoTrajectory:
        """Read mapped ego-trajectory samples in one half-open MCAP log-time range."""
        samples: list[DecodedEgoTrajectorySample] = []
        for schema, channel, message in self._mcap.iter_messages(
            reader,
            start_ns,
            exclusive_end_ns,
            log_time_order=True,
        ):
            resolved_cls = self._message_class_for_message(schema, channel)
            message_obj = parse_protobuf_message(
                resolved_cls,
                message.data,
                topic=self._topic,
                sensor_label="ego-trajectory",
            )
            sample = self._protobuf_row_mapper(message_obj, mcap_logtime_ns=int(message.log_time))
            samples.append(sample)

        if not samples:
            return self._get_empty_ego_trajectory()
        return _ego_trajectory_from_samples(
            samples,
            frame=self._frame,
            host=self._include_host,
            sequence=self._include_sequence,
        )

    def read_all(self) -> EgoTrajectory:
        """Decode every message on the configured ego-trajectory topic into one batch."""
        with self._mcap.open_reader() as reader:
            self._resolve_message_class(reader)
            try:
                return self._read_messages(reader, self.start_ns, self.end_ns + 1)
            except EmptyMcapTopicError:
                return self._get_empty_ego_trajectory()

    def supports_sampling_policy(self, policy: object) -> bool:
        """Return whether this sensor can sample with *policy*."""
        return isinstance(policy, NearestTimestampPolicy)

    def sample(
        self,
        spec: SamplingSpec,
        *,
        policy: NearestTimestampPolicy,
    ) -> Generator[EgoTrajectory]:
        """Yield one nearest source pose for every requested window timestamp.

        Each ``sample()`` invocation decodes the configured clip source once
        before producing its first window. Every source pose in that clip is
        then eligible for every requested timestamp; windows batch output rows
        only. A clip the splitter wrote holds its half-open interior plus the
        pose immediately before it, which is eligible for nearest selection too,
        so a selected source timestamp can lie before the clip's span. No pose
        after the span is carried, so a request near the end of a clip selects
        the nearest pose that clip holds even where the session had a nearer one
        just beyond it. Nearest selection and tolerance checking use
        mapper-produced ``align_timestamps_ns``; depending on the mapping,
        those timestamps may themselves originate from MCAP ``log_time``.
        """
        policy = require_nearest_timestamp_policy(policy, sensor_name=type(self).__name__)
        source = self.read_all()
        for window in spec.grid:
            yield align_ego_trajectory_nearest(source, window.timestamps_ns, policy=policy)
