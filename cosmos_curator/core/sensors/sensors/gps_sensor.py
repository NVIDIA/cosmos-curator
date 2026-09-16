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
"""MCAP protobuf GPS sensor."""

from collections.abc import Callable, Collection, Generator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, SupportsInt

import numpy as np
import numpy.typing as npt
from google.protobuf.message import Message
from mcap.reader import McapReader
from mcap.records import Channel, Schema

from cosmos_curator.core.sensors.alignment.gps import align_gps
from cosmos_curator.core.sensors.data.extrinsics import SensorExtrinsics
from cosmos_curator.core.sensors.data.gps_data import GpsData
from cosmos_curator.core.sensors.sampling.policy import (
    GpsInterpolationPolicy,
    GpsSamplingPolicy,
    NearestTimestampPolicy,
    require_gps_sampling_policy,
)
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.types.types import DataSource
from cosmos_curator.core.sensors.utils.mcap import (
    EmptyMcapTopicError,
    McapProtobufMessageResolver,
    McapTopicAccessor,
    parse_protobuf_message,
)
from cosmos_curator.core.sensors.utils.protobuf_mapper import ProtobufRowMapper

DEFAULT_TOPIC = "/gps"
_VECTOR_COLUMNS = 3
REQUIRED_GPS_MAPPING_FIELDS = frozenset(
    {
        "sensor_timestamp_ns",
        "latitude_deg",
        "longitude_deg",
        "altitude_m",
    }
)
_OPTIONAL_SCALAR_FIELD_PAIRS = (
    ("satellites_used", "satellites_used_valid"),
    ("horizontal_accuracy_m", "horizontal_accuracy_m_valid"),
    ("vertical_accuracy_m", "vertical_accuracy_m_valid"),
    ("hdop", "hdop_valid"),
    ("vdop", "vdop_valid"),
    ("pdop", "pdop_valid"),
)


@dataclass(frozen=True)
class DecodedGpsSample:
    """Decoded GPS protobuf fields normalized for ``GpsData`` construction."""

    sensor_timestamp_ns: int
    align_timestamp_ns: int
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    position_valid: tuple[bool, bool, bool] = (True, True, True)
    host_timestamp_ns: int | None = None
    hdop: float = 0.0
    vdop: float = 0.0
    pdop: float = 0.0
    horizontal_accuracy_m: float = 0.0
    vertical_accuracy_m: float = 0.0
    satellites_used: int = 0
    hdop_valid: bool = False
    vdop_valid: bool = False
    pdop_valid: bool = False
    horizontal_accuracy_m_valid: bool = False
    vertical_accuracy_m_valid: bool = False
    satellites_used_valid: bool = False


def _coerce_uint32(value: SupportsInt) -> int:
    if isinstance(value, (bool, np.bool_)):
        msg = "uint32 GPS scalar value must not be bool"
        raise TypeError(msg)
    result = int(value)
    uint32_max = int(np.iinfo(np.uint32).max)
    if result < 0 or result > uint32_max:
        msg = f"value out of uint32 range: {result}"
        raise ValueError(msg)
    return result


def _optional_scalar_arrays(
    samples: list[DecodedGpsSample],
    field_names: tuple[str, str],
    *,
    coerce: Callable[..., Any],
    dtype: npt.DTypeLike,
    is_mapped: bool,
) -> tuple[npt.NDArray[Any] | None, npt.NDArray[np.bool_] | None]:
    """Return mapped optional values and their validity mask.

    Unmapped values are absent. Mapped values retain an array even when every
    row is invalid, preserving raw values so consumers can rely on a stable
    batch schema and use the validity mask to identify usable measurements.
    """
    value_name, validity_name = field_names
    validity = np.array([bool(getattr(sample, validity_name)) for sample in samples], dtype=np.bool_)
    if not is_mapped:
        return None, None
    values = [coerce(getattr(sample, value_name)) for sample in samples]
    return np.array(values, dtype=dtype), validity


def _optional_int64_array(
    samples: list[DecodedGpsSample],
    value_name: str,
) -> npt.NDArray[np.int64] | None:
    """Return an optional int64 array only when every row carries the field."""
    raw_values = [getattr(sample, value_name) for sample in samples]
    if all(value is None for value in raw_values):
        return None
    if any(value is None for value in raw_values):
        msg = f"{value_name} must be present for every GPS sample or omitted for every GPS sample"
        raise ValueError(msg)
    return np.array([int(value) for value in raw_values], dtype=np.int64)


def _gps_data_from_samples(
    samples: list[DecodedGpsSample],
    mapped_optional_scalar_names: Collection[str],
) -> GpsData:
    """Build a ``GpsData`` batch from decoded GPS samples."""
    align_timestamps_ns = np.array([sample.align_timestamp_ns for sample in samples], dtype=np.int64)
    nonincreasing_timestamps = align_timestamps_ns[1:][align_timestamps_ns[1:] <= align_timestamps_ns[:-1]]
    if len(nonincreasing_timestamps) > 0:
        msg = (
            f"GPS samples contain non-increasing align_timestamp_ns {int(nonincreasing_timestamps[0])}; "
            "raw GpsData requires strictly increasing align_timestamps_ns"
        )
        raise ValueError(msg)

    sensor_timestamps_ns = np.array([sample.sensor_timestamp_ns for sample in samples], dtype=np.int64)
    hdop, hdop_valid = _optional_scalar_arrays(
        samples,
        ("hdop", "hdop_valid"),
        coerce=float,
        dtype=np.float64,
        is_mapped="hdop" in mapped_optional_scalar_names,
    )
    vdop, vdop_valid = _optional_scalar_arrays(
        samples,
        ("vdop", "vdop_valid"),
        coerce=float,
        dtype=np.float64,
        is_mapped="vdop" in mapped_optional_scalar_names,
    )
    pdop, pdop_valid = _optional_scalar_arrays(
        samples,
        ("pdop", "pdop_valid"),
        coerce=float,
        dtype=np.float64,
        is_mapped="pdop" in mapped_optional_scalar_names,
    )
    horizontal_accuracy_m, horizontal_accuracy_m_valid = _optional_scalar_arrays(
        samples,
        ("horizontal_accuracy_m", "horizontal_accuracy_m_valid"),
        coerce=float,
        dtype=np.float64,
        is_mapped="horizontal_accuracy_m" in mapped_optional_scalar_names,
    )
    vertical_accuracy_m, vertical_accuracy_m_valid = _optional_scalar_arrays(
        samples,
        ("vertical_accuracy_m", "vertical_accuracy_m_valid"),
        coerce=float,
        dtype=np.float64,
        is_mapped="vertical_accuracy_m" in mapped_optional_scalar_names,
    )
    satellites_used, satellites_used_valid = _optional_scalar_arrays(
        samples,
        ("satellites_used", "satellites_used_valid"),
        coerce=_coerce_uint32,
        dtype=np.uint32,
        is_mapped="satellites_used" in mapped_optional_scalar_names,
    )

    return GpsData(
        align_timestamps_ns=align_timestamps_ns,
        sensor_timestamps_ns=sensor_timestamps_ns,
        latitude_deg=np.array([sample.latitude_deg for sample in samples], dtype=np.float64),
        longitude_deg=np.array([sample.longitude_deg for sample in samples], dtype=np.float64),
        altitude_m=np.array([sample.altitude_m for sample in samples], dtype=np.float64),
        position_valid=np.array([sample.position_valid for sample in samples], dtype=np.bool_),
        satellites_used=satellites_used,
        satellites_used_valid=satellites_used_valid,
        horizontal_accuracy_m=horizontal_accuracy_m,
        horizontal_accuracy_m_valid=horizontal_accuracy_m_valid,
        vertical_accuracy_m=vertical_accuracy_m,
        vertical_accuracy_m_valid=vertical_accuracy_m_valid,
        hdop=hdop,
        hdop_valid=hdop_valid,
        vdop=vdop,
        vdop_valid=vdop_valid,
        pdop=pdop,
        pdop_valid=pdop_valid,
        host_timestamps_ns=_optional_int64_array(samples, "host_timestamp_ns"),
    )


def _empty_gps_data(
    *,
    include_host_timestamps: bool,
    mapped_optional_scalar_names: Collection[str],
) -> GpsData:
    """Build an empty raw GPS sample batch."""
    empty_ts = np.empty(0, dtype=np.int64)
    host_timestamps_ns = np.empty(0, dtype=np.int64) if include_host_timestamps else None
    empty_float = np.empty(0, dtype=np.float64)
    empty_bool = np.empty(0, dtype=np.bool_)
    empty_uint32 = np.empty(0, dtype=np.uint32)
    return GpsData(
        align_timestamps_ns=empty_ts,
        sensor_timestamps_ns=empty_ts,
        latitude_deg=np.empty(0, dtype=np.float64),
        longitude_deg=np.empty(0, dtype=np.float64),
        altitude_m=np.empty(0, dtype=np.float64),
        position_valid=np.empty((0, _VECTOR_COLUMNS), dtype=np.bool_),
        satellites_used=empty_uint32 if "satellites_used" in mapped_optional_scalar_names else None,
        satellites_used_valid=empty_bool if "satellites_used" in mapped_optional_scalar_names else None,
        horizontal_accuracy_m=empty_float if "horizontal_accuracy_m" in mapped_optional_scalar_names else None,
        horizontal_accuracy_m_valid=empty_bool if "horizontal_accuracy_m" in mapped_optional_scalar_names else None,
        vertical_accuracy_m=empty_float if "vertical_accuracy_m" in mapped_optional_scalar_names else None,
        vertical_accuracy_m_valid=empty_bool if "vertical_accuracy_m" in mapped_optional_scalar_names else None,
        hdop=empty_float if "hdop" in mapped_optional_scalar_names else None,
        hdop_valid=empty_bool if "hdop" in mapped_optional_scalar_names else None,
        vdop=empty_float if "vdop" in mapped_optional_scalar_names else None,
        vdop_valid=empty_bool if "vdop" in mapped_optional_scalar_names else None,
        pdop=empty_float if "pdop" in mapped_optional_scalar_names else None,
        pdop_valid=empty_bool if "pdop" in mapped_optional_scalar_names else None,
        host_timestamps_ns=host_timestamps_ns,
    )


class GpsSensor:
    """MCAP-backed GPS sensor with explicit nearest or interpolation alignment.

    Reads mapped protobuf messages from an MCAP topic. ``schema_name`` selects
    the message declared by the MCAP's embedded descriptor, and
    ``protobuf_mapping`` maps that message into ``GpsData`` fields. ``sample``
    aligns the complete configured source with an explicit GPS policy; it does
    not perform sensor fusion.
    Mapping an optional scalar value without its matching validity field defaults
    that field's validity to true. Mapped optional scalar arrays remain present
    across every sampling window and preserve raw values; use their validity
    masks to identify usable values.
    """

    def __init__(
        self,
        source: DataSource,
        topic: str = DEFAULT_TOPIC,
        *,
        schema_name: str,
        protobuf_mapping: str | Path | Mapping[str, Any],
        extrinsics: SensorExtrinsics | None = None,
    ) -> None:
        """Initialize the MCAP GPS sensor."""
        self._topic = topic
        self._extrinsics = extrinsics
        self._mcap = McapTopicAccessor(source, topic)
        self._protobuf_resolver = McapProtobufMessageResolver(schema_name, schema_label="GPS protobuf")
        self._protobuf_row_mapper = ProtobufRowMapper(protobuf_mapping, target_cls=DecodedGpsSample)
        missing = sorted(REQUIRED_GPS_MAPPING_FIELDS - self._protobuf_row_mapper.mapped_target_names)
        if missing:
            msg = "GPS protobuf mapping must map required field(s): " + ", ".join(repr(name) for name in missing)
            raise ValueError(msg)
        mapped = self._protobuf_row_mapper.mapped_target_names
        self._include_empty_host_timestamps = "host_timestamp_ns" in mapped
        self._mapped_optional_scalar_names = frozenset(
            value_name for value_name, _validity_name in _OPTIONAL_SCALAR_FIELD_PAIRS if value_name in mapped
        )
        self._empty_gps_data: GpsData | None = None

    @property
    def start_ns(self) -> int:
        """Earliest GPS message time on this topic, in nanoseconds."""
        return self._mcap.start_ns

    @property
    def end_ns(self) -> int:
        """Latest GPS message time on this topic, in nanoseconds."""
        return self._mcap.end_ns

    @property
    def max_gap_ns(self) -> int:
        """Return maximum expected gap duration in nanoseconds."""
        return self._mcap.max_gap_ns

    @property
    def timestamps_ns(self) -> npt.NDArray[np.int64]:
        """Message times in nanoseconds from raw MCAP ``log_time`` values."""
        return self._mcap.timestamps_ns

    def _resolve_message_class(self, reader: McapReader) -> type[Message] | None:
        """Resolve and cache the GPS message class from the MCAP summary when available."""
        return self._protobuf_resolver.resolve_from_summary(reader, self._topic)

    def supports_sampling_policy(self, policy: object) -> bool:
        """Return whether this sensor can sample with *policy*."""
        return isinstance(policy, (NearestTimestampPolicy, GpsInterpolationPolicy))

    def _message_class_for_message(self, schema: Schema | None, channel: Channel) -> type[Message]:
        """Resolve the dynamic message class and validate each message channel."""
        return self._protobuf_resolver.message_class_for_message(schema, channel, topic=self._topic)

    def _get_empty_gps_data(self) -> GpsData:
        """Return a cached empty raw GPS sample batch."""
        if self._empty_gps_data is None:
            self._empty_gps_data = _empty_gps_data(
                include_host_timestamps=self._include_empty_host_timestamps,
                mapped_optional_scalar_names=self._mapped_optional_scalar_names,
            )
        return self._empty_gps_data

    def _read_messages(
        self,
        reader: McapReader,
        start_ns: int,
        exclusive_end_ns: int,
    ) -> GpsData:
        """Decode all GPS samples whose MCAP log times fall in a half-open range."""
        samples: list[DecodedGpsSample] = []
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
                sensor_label="GPS",
            )
            samples.append(self._protobuf_row_mapper(message_obj, mcap_logtime_ns=int(message.log_time)))

        if not samples:
            return self._get_empty_gps_data()
        return _gps_data_from_samples(samples, self._mapped_optional_scalar_names)

    def read_all(self) -> GpsData:
        """Decode every message on the configured GPS topic into one source batch."""
        with self._mcap.open_reader() as reader:
            self._resolve_message_class(reader)
            try:
                return self._read_messages(reader, self.start_ns, self.end_ns + 1)
            except EmptyMcapTopicError:
                return self._get_empty_gps_data()

    def sample(
        self,
        spec: SamplingSpec,
        *,
        policy: GpsSamplingPolicy,
    ) -> Generator[GpsData]:
        """Yield GPS rows aligned to each window's requested timestamps.

        Each ``sample()`` invocation decodes the configured source once before
        producing its first window. Every source fix in that MCAP is then
        eligible for every requested timestamp; windows batch output rows only.
        The splitter aligns a whole session at once and writes the rows, so a
        GPS clip arrives already on the episode's grid rather than as a
        clip-local source to align later. Every fix is present when alignment
        runs, which is what gives a boundary request brackets on both sides.

        A message's MCAP ``log_time`` and the measurement timestamp in its
        payload are one instant recorded twice. ``log_time`` is the one worth
        working from: it sits in the index, so a source timeline costs two seeks,
        while the payload's is knowable only by decoding every message. A
        converter that lets the two diverge breaks that, and splitting -- which
        bounds a clip window from the index and never decodes to plan -- is where
        it shows up.
        """
        policy = require_gps_sampling_policy(policy, sensor_name=type(self).__name__)
        source = self.read_all()
        for window in spec.grid:
            yield align_gps(source, window.timestamps_ns, policy=policy)
