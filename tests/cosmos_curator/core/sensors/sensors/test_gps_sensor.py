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
"""Tests for MCAP protobuf ``GpsSensor``."""

from pathlib import Path

import attrs
import numpy as np
import pytest
import yaml
from foxglove_schemas_protobuf.LocationFix_pb2 import LocationFix
from google.protobuf import descriptor_pb2
from google.protobuf.descriptor import FileDescriptor
from google.protobuf.message import Message

from cosmos_curator.core.sensors.data.gps_data import GpsData
from cosmos_curator.core.sensors.exceptions import AlignmentError, AlignmentFailureReason
from cosmos_curator.core.sensors.sampling.policy import GpsInterpolationPolicy, NearestTimestampPolicy, NoSamplingPolicy
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors.gps_sensor import DEFAULT_TOPIC, GpsSensor
from cosmos_curator.core.sensors.sensors.group import SensorGroup
from tests.cosmos_curator.core.sensors.test_utils import (
    McapSample,
    make_sampling_grid,
    one_window_spec,
    protobuf_descriptor_set_from_proto,
    protobuf_message_class,
    write_protobuf_mcap,
)

_FIELD_DESCRIPTOR_PROTO = descriptor_pb2.FieldDescriptorProto
_REPO_ROOT = Path(__file__).parents[5]
_REFERENCE_GPS_SCHEMA_NAME = "cosmos_curator.sensors.gps.v1.GpsSample"
_REFERENCE_GPS_MAPPING_PATH = (
    _REPO_ROOT / "cosmos_curator" / "core" / "sensors" / "examples" / "gps_protobuf_mapping.yaml"
)
_REFERENCE_GPS_PROTO_PATH = _REPO_ROOT / "cosmos_curator" / "core" / "sensors" / "schemas" / "gps.proto"
_FOXGLOVE_LOCATION_FIX_SCHEMA_NAME = "foxglove.LocationFix"
_FOXGLOVE_LOCATION_FIX_MAPPING_PATH = (
    _REPO_ROOT / "cosmos_curator" / "core" / "sensors" / "examples" / "foxglove_location_fix_protobuf_mapping.yaml"
)

_CUSTOM_TOPIC = "/vendor/gps"
_CUSTOM_GPS_SCHEMA_NAME = "vendor.gps.Envelope"
_CUSTOM_GPS_PROTO_FIELDS = (
    ("sensor_time_us", 1, _FIELD_DESCRIPTOR_PROTO.TYPE_UINT64),
    ("lat", 2, _FIELD_DESCRIPTOR_PROTO.TYPE_DOUBLE),
    ("lon", 3, _FIELD_DESCRIPTOR_PROTO.TYPE_DOUBLE),
    ("alt", 4, _FIELD_DESCRIPTOR_PROTO.TYPE_DOUBLE),
    ("hdop", 5, _FIELD_DESCRIPTOR_PROTO.TYPE_DOUBLE),
    ("align_time_us", 6, _FIELD_DESCRIPTOR_PROTO.TYPE_UINT64),
    ("host_time_us", 7, _FIELD_DESCRIPTOR_PROTO.TYPE_UINT64),
)


def _reference_gps_descriptor_set() -> descriptor_pb2.FileDescriptorSet:
    """Build the descriptor set for the checked-in reference GPS schema."""
    return protobuf_descriptor_set_from_proto(_REFERENCE_GPS_PROTO_PATH)


def _foxglove_location_fix_descriptor_set() -> descriptor_pb2.FileDescriptorSet:
    """Build LocationFix's embedded protobuf descriptor dependency closure."""
    descriptor_set = descriptor_pb2.FileDescriptorSet()
    added_file_names: set[str] = set()

    def add_file_and_dependencies(file_descriptor: FileDescriptor) -> None:
        if file_descriptor.name in added_file_names:
            return
        for dependency in file_descriptor.dependencies:
            add_file_and_dependencies(dependency)
        file_descriptor.CopyToProto(descriptor_set.file.add())
        added_file_names.add(file_descriptor.name)

    add_file_and_dependencies(LocationFix.DESCRIPTOR.file)
    return descriptor_set


def _reference_gps_payload(sensor_timestamp_ns: int, **overrides: object) -> bytes:
    """Serialize one reference GPS protobuf sample."""
    message_cls = protobuf_message_class(_reference_gps_descriptor_set(), _REFERENCE_GPS_SCHEMA_NAME)
    sample = message_cls()
    values: dict[str, object] = {
        "sensor_timestamp_ns": sensor_timestamp_ns,
        "host_timestamp_ns": sensor_timestamp_ns + 50,
        "latitude_deg": 47.1,
        "longitude_deg": 8.2,
        "altitude_m": 500.0,
        "course_deg": 90.0,
        "speed_m_s": 5.0,
        "climb_m_s": 0.0,
        "hdop": 0.8,
        "vdop": 1.2,
        "pdop": 1.6,
        "horizontal_accuracy_m": 0.0,
        "vertical_accuracy_m": 0.0,
        "satellites_used": 12,
        "latitude_valid": True,
        "longitude_valid": True,
        "altitude_valid": True,
        "course_valid": True,
        "speed_valid": True,
        "climb_valid": False,
        "hdop_valid": True,
        "vdop_valid": True,
        "pdop_valid": True,
        "horizontal_accuracy_m_valid": False,
        "vertical_accuracy_m_valid": False,
        "satellites_used_valid": True,
    }
    values.update(overrides)
    for name, value in values.items():
        setattr(sample, name, value)
    return sample.SerializeToString()


def _write_reference_gps_mcap(
    path: Path,
    samples: list[McapSample],
    *,
    chunk_size: int = 1_048_576,
) -> None:
    """Write an MCAP using the checked-in reference GPS schema contract."""
    write_protobuf_mcap(
        path,
        samples,
        topic=DEFAULT_TOPIC,
        schema_name=_REFERENCE_GPS_SCHEMA_NAME,
        schema_data=_reference_gps_descriptor_set().SerializeToString(),
        library="cosmos_curator reference gps sensor test",
        chunk_size=chunk_size,
    )


def _reference_gps_sensor(path: Path) -> GpsSensor:
    """Create a GPS sensor using the checked-in reference mapping."""
    return GpsSensor(
        path,
        topic=DEFAULT_TOPIC,
        schema_name=_REFERENCE_GPS_SCHEMA_NAME,
        protobuf_mapping=_REFERENCE_GPS_MAPPING_PATH,
    )


def _assert_equal_gps_data(actual: GpsData, expected: GpsData) -> None:
    """Require every GPS output field to be equal, including optional provenance."""
    for field in attrs.fields(GpsData):
        actual_value = getattr(actual, field.name)
        expected_value = getattr(expected, field.name)
        if actual_value is None or expected_value is None:
            assert actual_value is expected_value
        else:
            np.testing.assert_array_equal(actual_value, expected_value)


def _custom_gps_file_descriptor_set() -> descriptor_pb2.FileDescriptorSet:
    """Build a small customer-style nested protobuf descriptor set."""
    file_descriptor_set = descriptor_pb2.FileDescriptorSet()
    file_descriptor = file_descriptor_set.file.add()
    file_descriptor.name = "vendor/gps/envelope.proto"
    file_descriptor.package = "vendor.gps"
    file_descriptor.syntax = "proto3"

    fix_descriptor = file_descriptor.message_type.add()
    fix_descriptor.name = "Fix"
    for field_name, field_number, field_type in _CUSTOM_GPS_PROTO_FIELDS:
        field = fix_descriptor.field.add()
        field.name = field_name
        field.number = field_number
        field.label = _FIELD_DESCRIPTOR_PROTO.LABEL_OPTIONAL
        field.type = field_type

    envelope_descriptor = file_descriptor.message_type.add()
    envelope_descriptor.name = "Envelope"
    field = envelope_descriptor.field.add()
    field.name = "fix"
    field.number = 1
    field.label = _FIELD_DESCRIPTOR_PROTO.LABEL_OPTIONAL
    field.type = _FIELD_DESCRIPTOR_PROTO.TYPE_MESSAGE
    field.type_name = ".vendor.gps.Fix"
    return file_descriptor_set


def _custom_gps_message_class() -> type[Message]:
    """Create the dynamic customer-style message class used by synthetic fixtures."""
    return protobuf_message_class(_custom_gps_file_descriptor_set(), _CUSTOM_GPS_SCHEMA_NAME)


def _custom_gps_payload(
    *,
    sensor_time_us: int,
    lat: float = 47.1,
    lon: float = 8.2,
    alt: float = 500.0,
    hdop: float = 0.8,
    align_time_us: int | None = None,
    host_time_us: int | None = None,
) -> bytes:
    """Serialize one synthetic customer-style GPS protobuf sample."""
    message_cls = _custom_gps_message_class()
    sample = message_cls()
    sample.fix.sensor_time_us = sensor_time_us
    sample.fix.lat = lat
    sample.fix.lon = lon
    sample.fix.alt = alt
    sample.fix.hdop = hdop
    if align_time_us is not None:
        sample.fix.align_time_us = align_time_us
    if host_time_us is not None:
        sample.fix.host_time_us = host_time_us
    return sample.SerializeToString()


def _custom_gps_mapping(**extra_fields: object) -> dict[str, object]:
    return {
        "root": "fix",
        "fields": {
            "sensor_timestamp_ns": {"from": "sensor_time_us", "type": "timestamp", "unit": "us"},
            "latitude_deg": {"from": "lat", "type": "float"},
            "longitude_deg": {"from": "lon", "type": "float"},
            "altitude_m": {"from": "alt", "type": "float"},
            **extra_fields,
        },
    }


def _write_custom_gps_mcap(
    path: Path,
    samples: list[McapSample],
    *,
    topic: str = _CUSTOM_TOPIC,
    schema_name: str = _CUSTOM_GPS_SCHEMA_NAME,
    schema_encoding: str = "protobuf",
    message_encoding: str = "protobuf",
    schema_data: bytes | None = None,
) -> None:
    """Write a synthetic customer GPS protobuf MCAP."""
    write_protobuf_mcap(
        path,
        samples,
        topic=topic,
        schema_name=schema_name,
        schema_data=(_custom_gps_file_descriptor_set().SerializeToString() if schema_data is None else schema_data),
        schema_encoding=schema_encoding,
        message_encoding=message_encoding,
        library="cosmos_curator gps sensor test",
    )


def test_gps_sensor_reads_reference_schema_with_checked_in_mapping(tmp_path: Path) -> None:
    """The shipped reference schema and YAML should produce complete GPS data."""
    path = tmp_path / "reference_gps.mcap"
    _write_reference_gps_mcap(
        path,
        [
            McapSample(log_time_ns=100, data=_reference_gps_payload(100)),
            McapSample(
                log_time_ns=200,
                data=_reference_gps_payload(
                    200,
                    latitude_deg=47.2,
                    longitude_deg=8.3,
                    altitude_m=501.0,
                    satellites_used=14,
                ),
            ),
        ],
    )

    batch = _reference_gps_sensor(path).read_all()

    np.testing.assert_array_equal(batch.align_timestamps_ns, np.array([100, 200], dtype=np.int64))
    np.testing.assert_array_equal(batch.sensor_timestamps_ns, np.array([100, 200], dtype=np.int64))
    np.testing.assert_array_equal(batch.host_timestamps_ns, np.array([150, 250], dtype=np.int64))
    np.testing.assert_allclose(batch.latitude_deg, np.array([47.1, 47.2]))
    np.testing.assert_allclose(batch.longitude_deg, np.array([8.2, 8.3]))
    np.testing.assert_allclose(batch.altitude_m, np.array([500.0, 501.0]))
    np.testing.assert_array_equal(batch.position_valid, np.ones((2, 3), dtype=np.bool_))
    np.testing.assert_array_equal(batch.satellites_used, np.array([12, 14], dtype=np.uint32))


def test_gps_sensor_reads_foxglove_location_fix_with_yaml_mapping(tmp_path: Path) -> None:
    """A descriptor-embedded valid-only LocationFix should map without parser changes."""
    path = tmp_path / "foxglove_location_fix.mcap"
    first = LocationFix(latitude=37.402255555555556, longitude=-122.25870916666666, altitude=79.14180564880371)
    second = LocationFix(latitude=37.4023, longitude=-122.2588, altitude=79.2)
    first.timestamp.seconds = 1_696_434_500
    first.timestamp.nanos = 123
    second.timestamp.seconds = 1_696_434_501
    second.timestamp.nanos = 456
    _write_custom_gps_mcap(
        path,
        [
            McapSample(log_time_ns=1_696_434_428_000_000_000, data=first.SerializeToString()),
            McapSample(log_time_ns=1_696_434_429_000_000_000, data=second.SerializeToString()),
        ],
        topic=DEFAULT_TOPIC,
        schema_name=_FOXGLOVE_LOCATION_FIX_SCHEMA_NAME,
        schema_data=_foxglove_location_fix_descriptor_set().SerializeToString(),
    )

    sensor = GpsSensor(
        path,
        topic=DEFAULT_TOPIC,
        schema_name=_FOXGLOVE_LOCATION_FIX_SCHEMA_NAME,
        protobuf_mapping=_FOXGLOVE_LOCATION_FIX_MAPPING_PATH,
    )
    batch = sensor.read_all()

    np.testing.assert_array_equal(
        batch.sensor_timestamps_ns,
        np.array([1_696_434_500_000_000_123, 1_696_434_501_000_000_456], dtype=np.int64),
    )
    np.testing.assert_array_equal(batch.align_timestamps_ns, batch.sensor_timestamps_ns)
    np.testing.assert_allclose(batch.latitude_deg, np.array([37.402255555555556, 37.4023]))
    np.testing.assert_allclose(batch.longitude_deg, np.array([-122.25870916666666, -122.2588]))
    np.testing.assert_allclose(batch.altitude_m, np.array([79.14180564880371, 79.2]))
    np.testing.assert_array_equal(batch.position_valid, np.ones((2, 3), dtype=np.bool_))
    assert all(
        value is None
        for value in (
            batch.position_covariance_enu_m2,
            batch.velocity_enu_m_s,
            batch.velocity_valid,
            batch.fix_type,
            batch.satellites_used,
            batch.satellites_used_valid,
            batch.horizontal_accuracy_m,
            batch.horizontal_accuracy_m_valid,
            batch.vertical_accuracy_m,
            batch.vertical_accuracy_m_valid,
            batch.hdop,
            batch.hdop_valid,
            batch.vdop,
            batch.vdop_valid,
            batch.pdop,
            batch.pdop_valid,
            batch.host_timestamps_ns,
            batch.utc_timestamps_ns,
            batch.sequence_counter,
        )
    )


def test_gps_sensor_rejects_no_sampling_policy(tmp_path: Path) -> None:
    """GPS sampling requires an explicit alignment policy."""
    path = tmp_path / "reference_gps.mcap"
    _write_reference_gps_mcap(path, [McapSample(log_time_ns=100, data=_reference_gps_payload(100))])
    sensor = _reference_gps_sensor(path)

    assert sensor.supports_sampling_policy(NearestTimestampPolicy())
    assert sensor.supports_sampling_policy(GpsInterpolationPolicy())
    assert not sensor.supports_sampling_policy(NoSamplingPolicy())
    with pytest.raises(TypeError, match="GpsSensor requires NearestTimestampPolicy or GpsInterpolationPolicy"):
        next(sensor.sample(one_window_spec(100, 200), policy=NoSamplingPolicy()))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="GpsSensor requires NearestTimestampPolicy or GpsInterpolationPolicy"):
        next(sensor.sample(one_window_spec(100, 200), policy=None))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("policy", "expected_reason"),
    [
        (NearestTimestampPolicy(), AlignmentFailureReason.GPS_NO_ELIGIBLE_FIX),
        (GpsInterpolationPolicy(), AlignmentFailureReason.GPS_MISSING_BRACKET),
    ],
)
def test_gps_sensor_reports_an_empty_topic_as_a_structured_alignment_failure(
    tmp_path: Path,
    policy: NearestTimestampPolicy | GpsInterpolationPolicy,
    expected_reason: AlignmentFailureReason,
) -> None:
    """An empty configured GPS topic is bad alignment data under either GPS policy."""
    path = tmp_path / "empty_gps.mcap"
    _write_reference_gps_mcap(path, [])
    sensor = _reference_gps_sensor(path)

    with pytest.raises(AlignmentError) as error_info:
        next(SensorGroup({"gps": sensor}).sample(one_window_spec(100, 200), policies={"gps": policy}))

    error = error_info.value
    assert error.reason is expected_reason
    assert error.sensor_id == "gps"
    np.testing.assert_array_equal(error.align_timestamps_ns, [100])
    np.testing.assert_array_equal(error.known_source_timestamps_ns, [])


def test_gps_sensor_read_all_preserves_empty_mapped_batch_contract(tmp_path: Path) -> None:
    """An empty configured topic decodes to a fully shaped mapped GPS batch."""
    path = tmp_path / "empty_gps_read_all.mcap"
    _write_reference_gps_mcap(path, [])

    batch = _reference_gps_sensor(path).read_all()

    for timestamp_name in ("align_timestamps_ns", "sensor_timestamps_ns", "host_timestamps_ns"):
        timestamps = getattr(batch, timestamp_name)
        assert timestamps is not None
        assert timestamps.shape == (0,)
        assert timestamps.dtype == np.int64
    for coordinate_name in (
        "latitude_deg",
        "longitude_deg",
        "altitude_m",
        "horizontal_accuracy_m",
        "vertical_accuracy_m",
        "hdop",
        "vdop",
        "pdop",
    ):
        coordinate = getattr(batch, coordinate_name)
        assert coordinate is not None
        assert coordinate.shape == (0,)
        assert coordinate.dtype == np.float64
    assert batch.position_valid.shape == (0, 3)
    assert batch.position_valid.dtype == np.bool_
    assert batch.satellites_used is not None
    assert batch.satellites_used.shape == (0,)
    assert batch.satellites_used.dtype == np.uint32
    for validity_name in (
        "satellites_used_valid",
        "horizontal_accuracy_m_valid",
        "vertical_accuracy_m_valid",
        "hdop_valid",
        "vdop_valid",
        "pdop_valid",
    ):
        validity = getattr(batch, validity_name)
        assert validity is not None
        assert validity.shape == (0,)
        assert validity.dtype == np.bool_


def test_gps_sensor_alignment_is_invariant_to_mcap_chunk_boundaries(tmp_path: Path) -> None:
    """Equivalent MCAP chunk layouts decode to identical nearest and interpolated GPS rows."""
    samples = [
        McapSample(log_time_ns=100, data=_reference_gps_payload(100)),
        McapSample(log_time_ns=200, data=_reference_gps_payload(200)),
        McapSample(log_time_ns=400, data=_reference_gps_payload(400)),
    ]
    tiny_chunk_path = tmp_path / "tiny_chunks.mcap"
    large_chunk_path = tmp_path / "large_chunk.mcap"
    _write_reference_gps_mcap(tiny_chunk_path, samples, chunk_size=1)
    _write_reference_gps_mcap(large_chunk_path, samples, chunk_size=1_048_576)
    tiny_chunk_sensor = _reference_gps_sensor(tiny_chunk_path)
    large_chunk_sensor = _reference_gps_sensor(large_chunk_path)
    spec = one_window_spec(150, 250)

    for policy in (NearestTimestampPolicy(), GpsInterpolationPolicy()):
        tiny_chunk_output = next(tiny_chunk_sensor.sample(spec, policy=policy))
        large_chunk_output = next(large_chunk_sensor.sample(spec, policy=policy))

        _assert_equal_gps_data(tiny_chunk_output, large_chunk_output)


def test_gps_sensor_nearest_and_interpolation_use_clip_halo_fixes(tmp_path: Path) -> None:
    """Halo records copied into a clip stay eligible when the sensor aligns its own MCAP.

    The planned clip span is [100, 200). 90 and 200 are the halo fixes the split
    copies in and 140 is the only interior fix, so the first and last requests
    resolve using a halo under either policy.
    """
    path = tmp_path / "clip_gps.mcap"
    _write_reference_gps_mcap(
        path,
        [
            McapSample(log_time_ns=90, data=_reference_gps_payload(90)),
            McapSample(log_time_ns=140, data=_reference_gps_payload(140)),
            McapSample(log_time_ns=200, data=_reference_gps_payload(200)),
        ],
    )
    spec = SamplingSpec(
        grid=make_sampling_grid(
            timestamps_ns=np.array([100, 125, 150, 175, 200], dtype=np.int64),
            stride_ns=100,
            duration_ns=100,
        )
    )
    sensor = _reference_gps_sensor(path)

    nearest_batches = list(sensor.sample(spec, policy=NearestTimestampPolicy()))
    interpolated_batches = list(sensor.sample(spec, policy=GpsInterpolationPolicy()))

    assert len(nearest_batches) == 1
    np.testing.assert_array_equal(nearest_batches[0].align_timestamps_ns, [100, 125, 150, 175])
    # 100 and 175 are closer to the out-of-span halo fixes than to the interior one.
    np.testing.assert_array_equal(nearest_batches[0].sensor_timestamps_ns, [90, 140, 140, 200])

    assert len(interpolated_batches) == 1
    np.testing.assert_array_equal(
        interpolated_batches[0].source_bracket_timestamps_ns,
        [[90, 140], [90, 140], [140, 200], [140, 200]],
    )


def test_gps_reference_mapping_preserves_fully_invalid_optional_arrays(tmp_path: Path) -> None:
    """Mapped optional values should retain raw values and all-false validity arrays."""
    path = tmp_path / "invalid_optional_gps.mcap"
    _write_reference_gps_mcap(
        path,
        [
            McapSample(
                log_time_ns=100,
                data=_reference_gps_payload(
                    100,
                    latitude_valid=False,
                    longitude_valid=False,
                    altitude_valid=False,
                    hdop_valid=False,
                    vdop_valid=True,
                ),
            )
        ],
    )

    batch = _reference_gps_sensor(path).read_all()

    np.testing.assert_array_equal(batch.position_valid, np.array([[False, False, False]], dtype=np.bool_))
    np.testing.assert_allclose(batch.hdop, np.array([0.8]))
    np.testing.assert_array_equal(batch.hdop_valid, np.array([False], dtype=np.bool_))
    np.testing.assert_allclose(batch.vdop, np.array([1.2]))
    np.testing.assert_array_equal(batch.vdop_valid, np.array([True], dtype=np.bool_))


def test_gps_reference_mapping_preserves_partial_optional_validity(tmp_path: Path) -> None:
    """Partially valid optional values should preserve raw values with validity masks."""
    path = tmp_path / "partial_optional_gps.mcap"
    _write_reference_gps_mcap(
        path,
        [
            McapSample(log_time_ns=100, data=_reference_gps_payload(100)),
            McapSample(
                log_time_ns=200,
                data=_reference_gps_payload(
                    200,
                    hdop=99.0,
                    hdop_valid=False,
                    satellites_used=99,
                    satellites_used_valid=False,
                ),
            ),
        ],
    )

    batch = _reference_gps_sensor(path).read_all()

    np.testing.assert_allclose(batch.hdop, np.array([0.8, 99.0]))
    np.testing.assert_array_equal(batch.hdop_valid, np.array([True, False], dtype=np.bool_))
    np.testing.assert_array_equal(batch.satellites_used, np.array([12, 99], dtype=np.uint32))
    np.testing.assert_array_equal(batch.satellites_used_valid, np.array([True, False], dtype=np.bool_))


def test_gps_reference_mapping_preserves_invalid_raw_measurements(tmp_path: Path) -> None:
    """Invalid GPS measurements should retain raw values and their false masks."""
    path = tmp_path / "invalid_raw_gps.mcap"
    _write_reference_gps_mcap(
        path,
        [
            McapSample(
                log_time_ns=100,
                data=_reference_gps_payload(
                    100,
                    latitude_deg=np.nan,
                    longitude_deg=181.0,
                    altitude_m=np.inf,
                    latitude_valid=False,
                    longitude_valid=False,
                    altitude_valid=False,
                    hdop=np.nan,
                    hdop_valid=False,
                    horizontal_accuracy_m=-1.0,
                    horizontal_accuracy_m_valid=False,
                    satellites_used=99,
                    satellites_used_valid=False,
                ),
            )
        ],
    )

    batch = _reference_gps_sensor(path).read_all()

    assert np.isnan(batch.latitude_deg[0])
    np.testing.assert_allclose(batch.longitude_deg, np.array([181.0]))
    assert np.isinf(batch.altitude_m[0])
    np.testing.assert_array_equal(batch.position_valid, np.array([[False, False, False]], dtype=np.bool_))
    assert batch.hdop is not None
    assert np.isnan(batch.hdop[0])
    np.testing.assert_array_equal(batch.hdop_valid, np.array([False], dtype=np.bool_))
    np.testing.assert_allclose(batch.horizontal_accuracy_m, np.array([-1.0]))
    np.testing.assert_array_equal(batch.horizontal_accuracy_m_valid, np.array([False], dtype=np.bool_))
    np.testing.assert_array_equal(batch.satellites_used, np.array([99], dtype=np.uint32))
    np.testing.assert_array_equal(batch.satellites_used_valid, np.array([False], dtype=np.bool_))


def test_gps_sensor_reads_custom_schema_with_external_yaml_mapping(tmp_path: Path) -> None:
    """Custom protobuf GPS schemas should map through user YAML without a target field."""
    path = tmp_path / "custom_gps.mcap"
    mapping_path = tmp_path / "custom_gps_mapping.yaml"
    mapping_path.write_text(
        yaml.safe_dump(_custom_gps_mapping(hdop={"from": "hdop", "type": "float"})),
        encoding="utf-8",
    )
    _write_custom_gps_mcap(
        path,
        [
            McapSample(log_time_ns=10_000_000, data=_custom_gps_payload(sensor_time_us=100, hdop=0.8)),
            McapSample(log_time_ns=20_000_000, data=_custom_gps_payload(sensor_time_us=200, hdop=0.9)),
        ],
    )

    sensor = GpsSensor(
        path,
        topic=_CUSTOM_TOPIC,
        schema_name=_CUSTOM_GPS_SCHEMA_NAME,
        protobuf_mapping=mapping_path,
    )
    batch = sensor.read_all()

    np.testing.assert_array_equal(batch.align_timestamps_ns, np.array([100_000, 200_000], dtype=np.int64))
    np.testing.assert_array_equal(batch.sensor_timestamps_ns, np.array([100_000, 200_000], dtype=np.int64))
    np.testing.assert_allclose(batch.latitude_deg, np.array([47.1, 47.1], dtype=np.float64))
    np.testing.assert_allclose(batch.longitude_deg, np.array([8.2, 8.2], dtype=np.float64))
    np.testing.assert_allclose(batch.altitude_m, np.array([500.0, 500.0], dtype=np.float64))
    np.testing.assert_array_equal(batch.position_valid, np.array([[True, True, True], [True, True, True]]))
    np.testing.assert_allclose(batch.hdop, np.array([0.8, 0.9], dtype=np.float64))
    np.testing.assert_array_equal(batch.hdop_valid, np.array([True, True], dtype=np.bool_))
    assert batch.host_timestamps_ns is None
    assert batch.satellites_used is None


def test_gps_sensor_custom_mapping_reports_malformed_payload(tmp_path: Path) -> None:
    """Custom GPS payload parsing should retain sensor-specific topic context."""
    path = tmp_path / "malformed_custom_gps.mcap"
    _write_custom_gps_mcap(path, [McapSample(log_time_ns=100, data=b"\xff")])
    sensor = GpsSensor(
        path,
        topic=_CUSTOM_TOPIC,
        schema_name=_CUSTOM_GPS_SCHEMA_NAME,
        protobuf_mapping=_custom_gps_mapping(),
    )

    with pytest.raises(ValueError, match=r"failed to parse GPS protobuf message on topic .*gps"):
        sensor.read_all()


def test_gps_sensor_custom_mapping_can_use_mcap_logtime(tmp_path: Path) -> None:
    """Custom YAML may map sensor timestamps from MCAP log time."""
    path = tmp_path / "custom_gps_mcap_logtime.mcap"
    _write_custom_gps_mcap(
        path,
        [
            McapSample(log_time_ns=10_000_000, data=_custom_gps_payload(sensor_time_us=100)),
            McapSample(log_time_ns=20_000_000, data=_custom_gps_payload(sensor_time_us=100)),
        ],
    )

    sensor = GpsSensor(
        path,
        topic=_CUSTOM_TOPIC,
        schema_name=_CUSTOM_GPS_SCHEMA_NAME,
        protobuf_mapping=_custom_gps_mapping(
            sensor_timestamp_ns={"from": "$mcap.logtime", "type": "timestamp", "unit": "ns"}
        ),
    )
    batch = sensor.read_all()

    np.testing.assert_array_equal(batch.sensor_timestamps_ns, np.array([10_000_000, 20_000_000], dtype=np.int64))
    np.testing.assert_array_equal(batch.align_timestamps_ns, np.array([10_000_000, 20_000_000], dtype=np.int64))


def test_gps_sensor_custom_mapping_requires_position_fields(tmp_path: Path) -> None:
    """Custom GPS mappings must explicitly provide the core position fields."""
    mapping = {
        "root": "fix",
        "fields": {
            "sensor_timestamp_ns": {"from": "sensor_time_us", "type": "timestamp", "unit": "us"},
            "latitude_deg": {"from": "lat", "type": "float"},
            "altitude_m": {"from": "alt", "type": "float"},
        },
    }

    with pytest.raises(ValueError, match="longitude_deg"):
        GpsSensor(
            tmp_path / "custom_gps_missing_position.mcap",
            topic=_CUSTOM_TOPIC,
            schema_name=_CUSTOM_GPS_SCHEMA_NAME,
            protobuf_mapping=mapping,
        )


@pytest.mark.parametrize("host_mode", ["omitted", "source"])
def test_gps_sensor_custom_mapping_host_presence_matches_empty_windows(
    tmp_path: Path,
    host_mode: str,
) -> None:
    """Optional host timestamp presence should remain consistent across windows."""
    path = tmp_path / f"custom_gps_host_{host_mode}.mcap"
    host_mapping: dict[str, object] = {}
    if host_mode == "source":
        host_mapping["host_timestamp_ns"] = {"from": "host_time_us", "type": "timestamp", "unit": "us"}
    _write_custom_gps_mcap(
        path,
        [
            McapSample(
                log_time_ns=10_000_000,
                data=_custom_gps_payload(sensor_time_us=100, host_time_us=1_000 if host_mode == "source" else None),
            )
        ],
    )
    sensor = GpsSensor(
        path,
        topic=_CUSTOM_TOPIC,
        schema_name=_CUSTOM_GPS_SCHEMA_NAME,
        protobuf_mapping=_custom_gps_mapping(**host_mapping),
    )

    batch = sensor.read_all()

    if host_mode == "source":
        np.testing.assert_array_equal(batch.host_timestamps_ns, np.array([1_000_000]))
    else:
        assert batch.host_timestamps_ns is None


def test_gps_sensor_uses_explicit_mapped_align_timestamp(tmp_path: Path) -> None:
    """Custom YAML may map a protobuf align timestamp separately from sensor time."""
    path = tmp_path / "custom_gps_align.mcap"
    mapping = _custom_gps_mapping(align_timestamp_ns={"from": "align_time_us", "type": "timestamp", "unit": "us"})
    _write_custom_gps_mcap(
        path,
        [
            McapSample(
                log_time_ns=10_000_000,
                data=_custom_gps_payload(sensor_time_us=100, align_time_us=1_000),
            ),
            McapSample(
                log_time_ns=20_000_000,
                data=_custom_gps_payload(sensor_time_us=200, align_time_us=2_000),
            ),
        ],
    )

    sensor = GpsSensor(
        path,
        topic=_CUSTOM_TOPIC,
        schema_name=_CUSTOM_GPS_SCHEMA_NAME,
        protobuf_mapping=mapping,
    )
    batch = sensor.read_all()

    np.testing.assert_array_equal(batch.align_timestamps_ns, np.array([1_000_000, 2_000_000], dtype=np.int64))
    np.testing.assert_array_equal(batch.sensor_timestamps_ns, np.array([100_000, 200_000], dtype=np.int64))


def test_gps_sensor_rejects_duplicate_mapped_align_timestamps(tmp_path: Path) -> None:
    """Mapped GPS rows require strictly increasing alignment timestamps."""
    path = tmp_path / "duplicate_timestamps.mcap"
    _write_custom_gps_mcap(
        path,
        [
            McapSample(log_time_ns=10_000_000, data=_custom_gps_payload(sensor_time_us=100)),
            McapSample(log_time_ns=20_000_000, data=_custom_gps_payload(sensor_time_us=100)),
        ],
    )
    sensor = GpsSensor(
        path,
        topic=_CUSTOM_TOPIC,
        schema_name=_CUSTOM_GPS_SCHEMA_NAME,
        protobuf_mapping=_custom_gps_mapping(),
    )

    with pytest.raises(ValueError, match="strictly increasing align_timestamps_ns"):
        sensor.read_all()


def test_gps_sensor_exposes_mcap_topic_timeline_with_mapping(tmp_path: Path) -> None:
    """Timeline properties should remain independent of protobuf mapping."""
    path = tmp_path / "timeline.mcap"
    _write_custom_gps_mcap(
        path,
        [
            McapSample(log_time_ns=100, data=_custom_gps_payload(sensor_time_us=1)),
            McapSample(log_time_ns=200, data=_custom_gps_payload(sensor_time_us=2)),
            McapSample(log_time_ns=300, data=_custom_gps_payload(sensor_time_us=3)),
        ],
    )
    sensor = GpsSensor(
        path,
        topic=_CUSTOM_TOPIC,
        schema_name=_CUSTOM_GPS_SCHEMA_NAME,
        protobuf_mapping=_custom_gps_mapping(),
    )

    assert sensor.start_ns == 100
    assert sensor.end_ns == 300
    np.testing.assert_array_equal(sensor.timestamps_ns, np.array([100, 200, 300], dtype=np.int64))
    assert sensor.timestamps_ns.flags.writeable is False
    assert sensor.max_gap_ns == 0


def test_gps_sensor_rejects_empty_topic_timeline_with_mapping(tmp_path: Path) -> None:
    """Timeline properties should fail clearly when the configured topic has no messages."""
    path = tmp_path / "empty_topic.mcap"
    _write_custom_gps_mcap(path, [])
    sensor = GpsSensor(
        path,
        topic=_CUSTOM_TOPIC,
        schema_name=_CUSTOM_GPS_SCHEMA_NAME,
        protobuf_mapping=_custom_gps_mapping(),
    )

    with pytest.raises(ValueError, match=r"no MCAP messages on topic .*/vendor/gps"):
        _ = sensor.timestamps_ns
    with pytest.raises(ValueError, match=r"no MCAP messages on topic .*/vendor/gps"):
        _ = sensor.start_ns


def test_gps_sensor_aligns_each_window_against_the_complete_source(tmp_path: Path) -> None:
    """GPS windows batch requested rows but never limit source-fix eligibility."""
    path = tmp_path / "clip_gps.mcap"
    _write_custom_gps_mcap(
        path,
        [
            McapSample(log_time_ns=10, data=_custom_gps_payload(sensor_time_us=100)),
            McapSample(log_time_ns=20, data=_custom_gps_payload(sensor_time_us=300)),
        ],
    )
    sensor = GpsSensor(
        path,
        topic=_CUSTOM_TOPIC,
        schema_name=_CUSTOM_GPS_SCHEMA_NAME,
        protobuf_mapping=_custom_gps_mapping(),
    )

    spec = one_window_spec(200_000, 250_000)
    frames = list(SensorGroup({"gps": sensor}).sample(spec, policies={"gps": NearestTimestampPolicy()}))
    nearest = frames[0].sensor_data["gps"]
    interpolated = next(sensor.sample(one_window_spec(200_000, 250_000), policy=GpsInterpolationPolicy()))

    np.testing.assert_array_equal(nearest.align_timestamps_ns, [200_000])
    np.testing.assert_array_equal(nearest.sensor_timestamps_ns, [100_000])
    np.testing.assert_array_equal(interpolated.sensor_timestamps_ns, [200_000])
    np.testing.assert_array_equal(interpolated.source_bracket_timestamps_ns, [[100_000, 300_000]])


def test_gps_sensor_rejects_missing_mapped_topic(tmp_path: Path) -> None:
    """The configured mapped topic must exist in the MCAP."""
    path = tmp_path / "missing_topic.mcap"
    _write_custom_gps_mcap(
        path,
        [McapSample(log_time_ns=100, data=_custom_gps_payload(sensor_time_us=1))],
        topic="/other",
    )
    sensor = GpsSensor(
        path,
        topic=_CUSTOM_TOPIC,
        schema_name=_CUSTOM_GPS_SCHEMA_NAME,
        protobuf_mapping=_custom_gps_mapping(),
    )

    with pytest.raises(ValueError, match="no MCAP channel found for topic '/vendor/gps'"):
        sensor.read_all()


def test_gps_sensor_rejects_wrong_mapped_schema_name(tmp_path: Path) -> None:
    """The MCAP schema must match the mapped protobuf schema name."""
    path = tmp_path / "wrong_schema.mcap"
    _write_custom_gps_mcap(
        path,
        [McapSample(log_time_ns=100, data=_custom_gps_payload(sensor_time_us=1))],
        schema_name="vendor.gps.Other",
    )
    sensor = GpsSensor(
        path,
        topic=_CUSTOM_TOPIC,
        schema_name=_CUSTOM_GPS_SCHEMA_NAME,
        protobuf_mapping=_custom_gps_mapping(),
    )

    with pytest.raises(ValueError, match="expected MCAP schema"):
        sensor.read_all()


def test_gps_sensor_rejects_non_protobuf_mapped_channel(tmp_path: Path) -> None:
    """Mapped GPS channels must still use protobuf encoding."""
    path = tmp_path / "wrong_encoding.mcap"
    _write_custom_gps_mcap(
        path,
        [McapSample(log_time_ns=100, data=b"{}")],
        schema_encoding="jsonschema",
        message_encoding="json",
        schema_data=b"{}",
    )
    sensor = GpsSensor(
        path,
        topic=_CUSTOM_TOPIC,
        schema_name=_CUSTOM_GPS_SCHEMA_NAME,
        protobuf_mapping=_custom_gps_mapping(),
    )

    with pytest.raises(ValueError, match="expected protobuf channel"):
        sensor.read_all()
