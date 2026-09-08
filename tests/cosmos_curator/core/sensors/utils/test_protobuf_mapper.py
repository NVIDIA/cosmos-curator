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
"""Tests for generic protobuf-to-row-object mapper utilities."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import attrs
import pytest
import yaml
from foxglove_schemas_protobuf.PoseInFrame_pb2 import PoseInFrame
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.message import Message

from cosmos_curator.core.sensors.utils.protobuf_mapper import ProtobufRowMapper

_FIELD = descriptor_pb2.FieldDescriptorProto
_SCHEMA_NAME = "mapper_test.MapperMessage"
_ENVELOPE_SCHEMA_NAME = "mapper_test.Envelope"


@attrs.define(frozen=True)
class _GpsLikeRow:
    sensor_timestamp_ns: int
    align_timestamp_ns: int
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    hdop: float
    hdop_valid: bool
    position_valid: tuple[bool, bool, bool] = (True, True, True)


@dataclass(frozen=True)
class _OptionalValidityRow:
    sensor_timestamp_ns: int
    align_timestamp_ns: int
    hdop: float = 0.0
    hdop_valid: bool = False


@dataclass(frozen=True)
class _OptionalTemperatureRow:
    sensor_timestamp_ns: int
    align_timestamp_ns: int
    temperature_c: float | None = None
    temperature_valid: bool | None = None


@dataclass(frozen=True)
class _AmbiguousUnitValidityRow:
    value_m: float
    value_deg: float
    value_valid: bool


@dataclass(frozen=True)
class _MixedRow:
    count: int
    flag: bool
    temperature: float


@dataclass(frozen=True)
class _StringAnnotatedRow:
    sensor_timestamp_ns: "int"
    align_timestamp_ns: "int"
    latitude_deg: "float"
    hdop: "float"
    hdop_valid: "bool"


@dataclass(frozen=True)
class _TimestampRow:
    sensor_timestamp_ns: int


@dataclass(frozen=True)
class _DefaultRow:
    count: int


@dataclass(frozen=True)
class _NullableDefaultRow:
    count: int
    note: str | None


@dataclass(frozen=True)
class _NestedRootRow:
    sensor_timestamp_ns: int
    align_timestamp_ns: int
    latitude_deg: float


@dataclass(frozen=True)
class _FixedVectorRow:
    values: tuple[float, float, float]
    values_valid: tuple[bool, bool, bool]


@dataclass(frozen=True)
class _VariableTupleRow:
    values: tuple[float, ...]


@dataclass(frozen=True)
class _VariableListRow:
    values: list[float]
    values_valid: list[bool]


@dataclass(frozen=True)
class _ValidityVectorRow:
    values: tuple[bool, bool, bool]


@dataclass(frozen=True)
class _IntegerVectorRow:
    values: tuple[int, int]


@dataclass(frozen=True)
class _IncompatibleValidityShapeRow:
    values: list[float]
    values_valid: bool


@attrs.define(frozen=True)
class _NonInitFieldRow:
    count: int = attrs.field(init=False, default=0)


def _message_class() -> type[Message]:
    file_descriptor = descriptor_pb2.FileDescriptorProto()
    file_descriptor.name = "mapper_test.proto"
    file_descriptor.package = "mapper_test"
    file_descriptor.syntax = "proto3"

    quality_enum = file_descriptor.enum_type.add()
    quality_enum.name = "Quality"
    for name, number in (("QUALITY_UNKNOWN", 0), ("QUALITY_INITIALIZING", 1), ("QUALITY_OK", 3)):
        value = quality_enum.value.add()
        value.name = name
        value.number = number

    message_descriptor = file_descriptor.message_type.add()
    message_descriptor.name = "MapperMessage"
    for name, number, field_type in (
        ("sensor_timestamp_us", 1, _FIELD.TYPE_UINT64),
        ("epoch_ns", 2, _FIELD.TYPE_UINT64),
        ("align_timestamp_us", 14, _FIELD.TYPE_UINT64),
        ("latitude_deg", 3, _FIELD.TYPE_DOUBLE),
        ("longitude_deg", 4, _FIELD.TYPE_DOUBLE),
        ("altitude_m", 5, _FIELD.TYPE_DOUBLE),
        ("latitude_valid", 6, _FIELD.TYPE_BOOL),
        ("longitude_valid", 7, _FIELD.TYPE_BOOL),
        ("altitude_valid", 8, _FIELD.TYPE_BOOL),
        ("count", 9, _FIELD.TYPE_UINT32),
        ("flag", 10, _FIELD.TYPE_BOOL),
        ("temperature", 11, _FIELD.TYPE_DOUBLE),
        ("hdop", 12, _FIELD.TYPE_DOUBLE),
        ("hdop_valid", 13, _FIELD.TYPE_BOOL),
    ):
        field = message_descriptor.field.add()
        field.name = name
        field.number = number
        field.label = _FIELD.LABEL_OPTIONAL
        field.type = field_type

    field = message_descriptor.field.add()
    field.name = "latitude_samples"
    field.number = 15
    field.label = _FIELD.LABEL_REPEATED
    field.type = _FIELD.TYPE_DOUBLE

    field = message_descriptor.field.add()
    field.name = "quality_samples"
    field.number = 16
    field.label = _FIELD.LABEL_REPEATED
    field.type = _FIELD.TYPE_ENUM
    field.type_name = ".mapper_test.Quality"

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_descriptor)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName(_SCHEMA_NAME))


def _envelope_message_class() -> type[Message]:
    file_descriptor = descriptor_pb2.FileDescriptorProto()
    file_descriptor.name = "mapper_test_envelope.proto"
    file_descriptor.package = "mapper_test"
    file_descriptor.syntax = "proto3"

    fix_descriptor = file_descriptor.message_type.add()
    fix_descriptor.name = "Fix"
    for name, number, field_type in (
        ("sensor_timestamp_us", 1, _FIELD.TYPE_UINT64),
        ("latitude_deg", 2, _FIELD.TYPE_DOUBLE),
    ):
        field = fix_descriptor.field.add()
        field.name = name
        field.number = number
        field.label = _FIELD.LABEL_OPTIONAL
        field.type = field_type

    field = fix_descriptor.field.add()
    field.name = "samples"
    field.number = 3
    field.label = _FIELD.LABEL_REPEATED
    field.type = _FIELD.TYPE_DOUBLE

    envelope_descriptor = file_descriptor.message_type.add()
    envelope_descriptor.name = "Envelope"
    field = envelope_descriptor.field.add()
    field.name = "fix"
    field.number = 1
    field.label = _FIELD.LABEL_OPTIONAL
    field.type = _FIELD.TYPE_MESSAGE
    field.type_name = ".mapper_test.Fix"

    field = envelope_descriptor.field.add()
    field.name = "fixes"
    field.number = 2
    field.label = _FIELD.LABEL_REPEATED
    field.type = _FIELD.TYPE_MESSAGE
    field.type_name = ".mapper_test.Fix"

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_descriptor)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName(_ENVELOPE_SCHEMA_NAME))


def _message(**fields: object) -> Message:
    message = _message_class()()
    for name, value in fields.items():
        field = message.DESCRIPTOR.fields_by_name[name]
        if field.is_repeated:
            getattr(message, name).extend(value)
        else:
            setattr(message, name, value)
    return message


def _envelope_message(**fix_fields: object) -> Message:
    message = _envelope_message_class()()
    for name, value in fix_fields.items():
        field = message.fix.DESCRIPTOR.fields_by_name[name]
        if field.is_repeated:
            getattr(message.fix, name).extend(value)
        else:
            setattr(message.fix, name, value)
    return message


def _gps_like_spec() -> dict[str, object]:
    return {
        "fields": {
            "sensor_timestamp_ns": {"from": "sensor_timestamp_us", "type": "timestamp", "unit": "us"},
            "latitude_deg": {"from": "latitude_deg", "type": "float"},
            "longitude_deg": {"from": "longitude_deg", "type": "float"},
            "altitude_m": {"from": "altitude_m", "type": "float"},
            "hdop": {"from": "hdop", "type": "float"},
        },
    }


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ("fields: [unterminated", "failed to parse protobuf mapper YAML spec"),
        (
            {
                "fields": {
                    "missing_attr": {"from": "count", "type": "int"},
                },
            },
            "unknown target attribute 'missing_attr'",
        ),
        (
            {
                "fields": {
                    "latitude_deg": {"from": "latitude_deg", "type": "float"},
                },
            },
            "sensor_timestamp_ns",
        ),
        (
            {
                "fields": {
                    "sensor_timestamp_ns": {"from": "sensor_timestamp_us", "type": "timestamp", "unit": "us"},
                    "latitude_deg": {"from": "latitude_deg", "type": "decimal"},
                },
            },
            r"fields\.latitude_deg\.type",
        ),
        (
            {
                "fields": {
                    "sensor_timestamp_ns": {"from": "sensor_timestamp_us", "type": "timestamp", "unit": "minutes"},
                },
            },
            r"fields\.sensor_timestamp_ns\.unit",
        ),
        (
            {
                "fields": {
                    "sensor_timestamp_ns": {"from": "sensor_timestamp_us", "type": "timestamp", "unit": "us"},
                    "hdop_valid": {"from": "hdop_valid", "type": "bool"},
                },
            },
            r"fields\.hdop_valid maps validity",
        ),
    ],
)
def test_protobuf_row_mapper_validates_schema_independent_yaml_at_construction(
    spec: str | dict[str, object],
    match: str,
) -> None:
    """Reusable row mapper construction should catch YAML and target-shape errors eagerly."""
    with pytest.raises(ValueError, match=match):
        ProtobufRowMapper(spec, target_cls=_GpsLikeRow)


def test_protobuf_row_mapper_accepts_inline_yaml_string() -> None:
    """String specs are parsed as inline YAML."""
    yaml_text = "fields: {sensor_timestamp_ns: {default: 7, type: timestamp, unit: ns}}"
    mapper = ProtobufRowMapper(yaml_text, target_cls=_TimestampRow)

    assert mapper(_message()) == _TimestampRow(sensor_timestamp_ns=7)


def test_protobuf_row_mapper_does_not_open_string_path(tmp_path: Path) -> None:
    """File-backed specs require Path so string interpretation is deterministic."""
    spec_path = tmp_path / "gps_row.yaml"
    spec_path.write_text("fields: {sensor_timestamp_ns: {default: 5, type: timestamp, unit: ns}}", encoding="utf-8")

    with pytest.raises(ValueError, match="protobuf mapper YAML spec must be a mapping"):
        ProtobufRowMapper(str(spec_path), target_cls=_TimestampRow)


@pytest.mark.parametrize("spec_source", ["mapping", "yaml", "path"])
def test_protobuf_row_mapper_maps_gps_like_row_defaults(
    spec_source: str,
    tmp_path: Path,
) -> None:
    """YAML mapper should map scalar fields and apply row timestamp/validity defaults."""
    spec: Any = _gps_like_spec()
    if spec_source == "yaml":
        spec = yaml.safe_dump(spec)
    elif spec_source == "path":
        spec_path = tmp_path / "gps_row.yaml"
        spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
        spec = spec_path

    mapper = ProtobufRowMapper(spec, target_cls=_GpsLikeRow, message_cls=_message_class())

    row = mapper(
        _message(
            sensor_timestamp_us=123,
            latitude_deg=47.1,
            longitude_deg=8.2,
            altitude_m=500.5,
            hdop=0.8,
        )
    )

    assert row.sensor_timestamp_ns == 123_000
    assert row.align_timestamp_ns == 123_000
    assert row.latitude_deg == 47.1
    assert row.longitude_deg == 8.2
    assert row.altitude_m == 500.5
    assert row.position_valid == (True, True, True)
    assert row.hdop == 0.8
    assert row.hdop_valid is True


def test_protobuf_row_mapper_resolves_string_annotations_for_defaults() -> None:
    """Forward/string annotations should still drive float and validity defaults."""
    mapper = ProtobufRowMapper(
        {
            "fields": {
                "sensor_timestamp_ns": {"from": "sensor_timestamp_us", "type": "timestamp", "unit": "us"},
                "hdop": {"from": "hdop", "type": "float"},
            },
        },
        target_cls=_StringAnnotatedRow,
        message_cls=_message_class(),
    )

    row = mapper(_message(sensor_timestamp_us=123, hdop=0.8))

    assert row.sensor_timestamp_ns == 123_000
    assert row.align_timestamp_ns == 123_000
    assert math.isnan(row.latitude_deg)
    assert row.hdop == 0.8
    assert row.hdop_valid is True


def test_protobuf_row_mapper_rejects_explicit_null_defaults() -> None:
    """Optional targets are omitted from YAML instead of mapped to ``null``."""
    with pytest.raises(ValueError, match=r"fields\.note\.default must not be null"):
        ProtobufRowMapper(
            {
                "fields": {
                    "count": {"from": "count", "type": "int"},
                    "note": {"default": None},
                },
            },
            target_cls=_NullableDefaultRow,
        )


def test_protobuf_row_mapper_uses_explicit_align_timestamp_mapping() -> None:
    """Mapped align timestamps should override the default copy from sensor timestamp."""
    spec = _gps_like_spec()
    _cast_mapping(spec["fields"])["align_timestamp_ns"] = {
        "from": "align_timestamp_us",
        "type": "timestamp",
        "unit": "us",
    }
    mapper = ProtobufRowMapper(spec, target_cls=_GpsLikeRow, message_cls=_message_class())

    row = mapper(
        _message(
            sensor_timestamp_us=123,
            align_timestamp_us=456,
            latitude_deg=47.1,
            longitude_deg=8.2,
            altitude_m=500.5,
            hdop=0.8,
        )
    )

    assert row.sensor_timestamp_ns == 123_000
    assert row.align_timestamp_ns == 456_000


def test_protobuf_row_mapper_applies_group_conversion_per_element() -> None:
    """A grouped mapping should convert each source value and return a tuple."""
    mapper = ProtobufRowMapper(
        {
            "fields": {
                "values": {
                    "group": ["latitude_valid", "longitude_valid", "altitude_valid"],
                    "type": "bool",
                },
            },
        },
        target_cls=_ValidityVectorRow,
        message_cls=_message_class(),
    )

    row = mapper(_message(latitude_valid=True, longitude_valid=False, altitude_valid=True))

    assert row.values == (True, False, True)


def test_protobuf_row_mapper_maps_validity_code_with_valid_when_equals() -> None:
    """Validity/status codes should be mappable to bool without broad truthiness."""
    mapper = ProtobufRowMapper(
        {
            "fields": {
                "count": {"default": 1, "type": "int"},
                "flag": {"from": "count", "type": "bool", "valid_when": {"equals": 5}},
                "temperature": {"default": 0.0, "type": "float"},
            },
        },
        target_cls=_MixedRow,
        message_cls=_message_class(),
    )

    assert mapper(_message(count=5)).flag is True
    assert mapper(_message(count=6)).flag is False


def test_protobuf_row_mapper_rejects_incompatible_valid_when_equals_type() -> None:
    """Validity predicates must use a comparand compatible with the protobuf source."""
    with pytest.raises(ValueError, match=r"flag\.valid_when\.equals.*temperature"):
        ProtobufRowMapper(
            {
                "fields": {
                    "count": {"default": 1, "type": "int"},
                    "flag": {"from": "temperature", "type": "bool", "valid_when": {"equals": "valid"}},
                    "temperature": {"default": 0.0, "type": "float"},
                },
            },
            target_cls=_MixedRow,
            message_cls=_message_class(),
        )


def test_protobuf_row_mapper_applies_valid_when_equals_per_group_element() -> None:
    """Grouped validity/status codes should be compared one element at a time."""
    mapper = ProtobufRowMapper(
        {
            "fields": {
                "values": {
                    "group": ["count", "sensor_timestamp_us", "epoch_ns"],
                    "type": "bool",
                    "valid_when": {"equals": 5},
                },
            },
        },
        target_cls=_ValidityVectorRow,
        message_cls=_message_class(),
    )

    row = mapper(_message(count=5, sensor_timestamp_us=6, epoch_ns=5))

    assert row.values == (True, False, True)


def test_protobuf_row_mapper_maps_from_root_path() -> None:
    """A root path should let YAML map fields from a nested protobuf message."""
    mapper = ProtobufRowMapper(
        {
            "root": "fix",
            "fields": {
                "sensor_timestamp_ns": {"from": "sensor_timestamp_us", "type": "timestamp", "unit": "us"},
                "latitude_deg": {"from": "latitude_deg", "type": "float"},
            },
        },
        target_cls=_NestedRootRow,
        message_cls=_envelope_message_class(),
    )

    row = mapper(_envelope_message(sensor_timestamp_us=123, latitude_deg=47.1))

    assert row == _NestedRootRow(sensor_timestamp_ns=123_000, align_timestamp_ns=123_000, latitude_deg=47.1)


def test_protobuf_row_mapper_rejects_repeated_root_path() -> None:
    """Root paths cannot traverse repeated protobuf message fields."""
    with pytest.raises(ValueError, match=r"root path .*fixes.*unsupported repeated"):
        ProtobufRowMapper(
            {
                "root": "fixes",
                "fields": {
                    "sensor_timestamp_ns": {"from": "sensor_timestamp_us", "type": "timestamp", "unit": "us"},
                    "latitude_deg": {"from": "latitude_deg", "type": "float"},
                },
            },
            target_cls=_NestedRootRow,
            message_cls=_envelope_message_class(),
        )


def test_protobuf_row_mapper_converts_scalar_types() -> None:
    """Scalar int/bool/float conversions should populate dataclass targets."""
    mapper = ProtobufRowMapper(
        {
            "fields": {
                "count": {"from": "count", "type": "int"},
                "flag": {"from": "flag", "type": "bool"},
                "temperature": {"from": "temperature", "type": "float"},
            },
        },
        target_cls=_MixedRow,
        message_cls=_message_class(),
    )

    row = mapper(_message(count=7, flag=True, temperature=22.5))

    assert row == _MixedRow(count=7, flag=True, temperature=22.5)


def test_protobuf_row_mapper_allows_only_non_lossy_numeric_conversions() -> None:
    """Numeric conversions should reject truncation and precision loss."""
    mapper = ProtobufRowMapper(
        {
            "fields": {
                "count": {"from": "temperature", "type": "int"},
                "flag": {"from": "flag", "type": "bool"},
                "temperature": {"from": "count", "type": "float"},
            },
        },
        target_cls=_MixedRow,
        message_cls=_message_class(),
    )

    assert mapper(_message(count=1, flag=True, temperature=7.0)) == _MixedRow(count=7, flag=True, temperature=1.0)
    with pytest.raises(ValueError, match=r"count.*without losing information"):
        mapper(_message(count=1, flag=False, temperature=7.5))


def test_protobuf_row_mapper_requires_valid_when_for_numeric_bool_source() -> None:
    """Numeric and enum validity fields should use an explicit predicate."""
    with pytest.raises(ValueError, match=r"flag.*use valid_when"):
        ProtobufRowMapper(
            {
                "fields": {
                    "count": {"from": "count", "type": "int"},
                    "flag": {"from": "count", "type": "bool"},
                    "temperature": {"from": "temperature", "type": "float"},
                },
            },
            target_cls=_MixedRow,
            message_cls=_message_class(),
        )


def test_protobuf_row_mapper_rejects_bool_to_float_conversion() -> None:
    """Bool values should not be accepted as floats just because bool is int-like in Python."""
    mapper = ProtobufRowMapper(
        {
            "fields": {
                "count": {"default": 1, "type": "int"},
                "flag": {"default": True, "type": "bool"},
                "temperature": {"from": "flag", "type": "float"},
            },
        },
        target_cls=_MixedRow,
        message_cls=_message_class(),
    )

    with pytest.raises(ValueError, match=r"temperature.*float"):
        mapper(_message(flag=True))


def test_protobuf_row_mapper_rejects_fractional_nanosecond_timestamp_conversion() -> None:
    """Timestamp conversion should not silently truncate fractional nanoseconds."""
    mapper = ProtobufRowMapper(
        {
            "fields": {
                "sensor_timestamp_ns": {"from": "temperature", "type": "timestamp", "unit": "ns"},
            },
        },
        target_cls=_TimestampRow,
        message_cls=_message_class(),
    )

    with pytest.raises(ValueError, match="without losing information"):
        mapper(_message(temperature=1.5))


def test_protobuf_row_mapper_rejects_imprecise_epoch_seconds_timestamp_conversion() -> None:
    """Float epoch seconds must resolve exactly to an integer nanosecond count."""
    mapper = ProtobufRowMapper(
        {
            "fields": {
                "sensor_timestamp_ns": {"from": "temperature", "type": "timestamp", "unit": "s"},
            },
        },
        target_cls=_TimestampRow,
        message_cls=_message_class(),
    )

    with pytest.raises(ValueError, match="without losing information"):
        mapper(_message(temperature=1_753_920_000.0000002))


def test_protobuf_row_mapper_preserves_integer_timestamp_precision() -> None:
    """Integer timestamp conversion should not pass epoch-scale ns values through float."""
    mapper = ProtobufRowMapper(
        {
            "fields": {
                "sensor_timestamp_ns": {"from": "epoch_ns", "type": "timestamp", "unit": "ns"},
            },
        },
        target_cls=_TimestampRow,
        message_cls=_message_class(),
    )
    epoch_ns = 1_700_000_000_000_000_001

    row = mapper(_message(epoch_ns=epoch_ns))

    assert row.sensor_timestamp_ns == epoch_ns


@pytest.mark.parametrize(
    ("seconds", "nanos", "expected_ns"),
    [
        (1, 23, 1_000_000_023),
        (-1, 999_999_999, -1),
    ],
)
def test_protobuf_row_mapper_converts_protobuf_timestamp_to_nanoseconds(
    seconds: int,
    nanos: int,
    expected_ns: int,
) -> None:
    """A standard Timestamp message should preserve second/nanosecond precision."""
    mapper = ProtobufRowMapper(
        {"fields": {"sensor_timestamp_ns": {"from": "timestamp", "type": "protobuf_timestamp"}}},
        target_cls=_TimestampRow,
        message_cls=PoseInFrame,
    )
    message = PoseInFrame()
    message.timestamp.seconds = seconds
    message.timestamp.nanos = nanos

    assert mapper(message) == _TimestampRow(sensor_timestamp_ns=expected_ns)


@pytest.mark.parametrize(
    ("seconds", "nanos", "error"),
    [
        (1, 1_000_000_000, r"nanos must be in \[0, 1000000000\)"),
        (9_223_372_037, 0, "does not fit int64 nanoseconds"),
        (-9_223_372_037, 0, "does not fit int64 nanoseconds"),
    ],
)
def test_protobuf_row_mapper_rejects_invalid_protobuf_timestamp_values(
    seconds: int,
    nanos: int,
    error: str,
) -> None:
    """Malformed or unrepresentable Timestamp values should fail clearly."""
    mapper = ProtobufRowMapper(
        {"fields": {"sensor_timestamp_ns": {"from": "timestamp", "type": "protobuf_timestamp"}}},
        target_cls=_TimestampRow,
        message_cls=PoseInFrame,
    )
    message = PoseInFrame()
    message.timestamp.seconds = seconds
    message.timestamp.nanos = nanos

    with pytest.raises(ValueError, match=error):
        mapper(message)


def test_protobuf_row_mapper_rejects_non_timestamp_protobuf_timestamp_source() -> None:
    """The protobuf timestamp conversion accepts only the standard message type."""
    with pytest.raises(ValueError, match=r"is not google\.protobuf\.Timestamp"):
        ProtobufRowMapper(
            {
                "fields": {
                    "sensor_timestamp_ns": {
                        "from": "sensor_timestamp_us",
                        "type": "protobuf_timestamp",
                    }
                }
            },
            target_cls=_TimestampRow,
            message_cls=_message_class(),
        )


def test_protobuf_row_mapper_rejects_mcap_metadata_protobuf_timestamp_source() -> None:
    """Protobuf timestamp conversion must not be configured with MCAP log time."""
    with pytest.raises(ValueError, match="requires one protobuf message source, not MCAP metadata"):
        ProtobufRowMapper(
            {
                "fields": {
                    "sensor_timestamp_ns": {
                        "from": "$mcap.logtime",
                        "type": "protobuf_timestamp",
                    }
                }
            },
            target_cls=_TimestampRow,
        )


@pytest.mark.parametrize(
    ("target_cls", "fields", "expected"),
    [
        (_DefaultRow, {"count": {"default": 42, "type": "int"}}, _DefaultRow(count=42)),
        (
            _TimestampRow,
            {"sensor_timestamp_ns": {"default": 42, "type": "timestamp", "unit": "ms"}},
            _TimestampRow(sensor_timestamp_ns=42_000_000),
        ),
    ],
)
def test_protobuf_row_mapper_accepts_explicit_defaults(
    target_cls: type[object],
    fields: dict[str, object],
    expected: object,
) -> None:
    """Explicit YAML defaults populate ordinary and sensor timestamp fields."""
    mapper = ProtobufRowMapper({"fields": fields}, target_cls=target_cls)

    assert mapper(_message()) == expected


def test_protobuf_row_mapper_requires_align_timestamp_when_sensor_timestamp_uses_default() -> None:
    """GPS-like rows need an explicit align timestamp when sensor timestamp has no source field."""
    spec = _gps_like_spec()
    _cast_mapping(spec["fields"])["sensor_timestamp_ns"] = {"default": 0, "type": "timestamp", "unit": "ns"}

    with pytest.raises(ValueError, match=r"align_timestamp_ns.*static default"):
        ProtobufRowMapper(spec, target_cls=_GpsLikeRow, message_cls=_message_class())


def test_protobuf_row_mapper_accepts_explicit_align_timestamp_default() -> None:
    """Explicit align timestamp defaults should satisfy targets that need align time."""
    spec = _gps_like_spec()
    fields = _cast_mapping(spec["fields"])
    fields["sensor_timestamp_ns"] = {"default": 0, "type": "timestamp", "unit": "ns"}
    fields["align_timestamp_ns"] = {"default": 123, "type": "timestamp", "unit": "ns"}
    mapper = ProtobufRowMapper(spec, target_cls=_GpsLikeRow, message_cls=_message_class())

    row = mapper(_message(latitude_deg=47.1, longitude_deg=8.2, altitude_m=500.5, hdop=0.8))

    assert row.sensor_timestamp_ns == 0
    assert row.align_timestamp_ns == 123


def test_protobuf_row_mapper_keeps_default_when_value_and_validity_are_omitted() -> None:
    """An omitted value/validity pair should use the target row defaults."""
    mapper = ProtobufRowMapper(
        {
            "fields": {
                "sensor_timestamp_ns": {"from": "sensor_timestamp_us", "type": "timestamp", "unit": "us"},
            },
        },
        target_cls=_OptionalValidityRow,
        message_cls=_message_class(),
    )

    row = mapper(_message(sensor_timestamp_us=123))

    assert row.hdop == 0.0
    assert row.hdop_valid is False


def test_protobuf_row_mapper_defaults_optional_validity_only_when_value_is_mapped() -> None:
    """Optional values and validity are absent together unless the value is mapped."""
    base_fields = {
        "sensor_timestamp_ns": {"from": "sensor_timestamp_us", "type": "timestamp", "unit": "us"},
    }
    omitted_mapper = ProtobufRowMapper(
        {"fields": base_fields},
        target_cls=_OptionalTemperatureRow,
        message_cls=_message_class(),
    )
    mapped_mapper = ProtobufRowMapper(
        {"fields": {**base_fields, "temperature_c": {"from": "temperature", "type": "float"}}},
        target_cls=_OptionalTemperatureRow,
        message_cls=_message_class(),
    )

    omitted = omitted_mapper(_message(sensor_timestamp_us=123))
    mapped = mapped_mapper(_message(sensor_timestamp_us=123, temperature=19.5))

    assert (omitted.temperature_c, omitted.temperature_valid) == (None, None)
    assert (mapped.temperature_c, mapped.temperature_valid) == (19.5, True)


def test_protobuf_row_mapper_rejects_ambiguous_unit_suffixed_validity_pair() -> None:
    """A validity target must not select between multiple unit-suffixed values."""
    with pytest.raises(ValueError, match=r"value_valid.*value_deg, value_m"):
        ProtobufRowMapper(
            {"fields": {"value_valid": {"from": "flag", "type": "bool"}}},
            target_cls=_AmbiguousUnitValidityRow,
        )


def test_protobuf_row_mapper_rejects_unknown_protobuf_field_with_descriptor() -> None:
    """Source field names must exist in the protobuf descriptor when one is supplied."""
    spec = _gps_like_spec()
    _cast_mapping(spec["fields"])["latitude_deg"] = {"from": "missing_proto_field", "type": "float"}

    with pytest.raises(ValueError, match="missing_proto_field"):
        ProtobufRowMapper(spec, target_cls=_GpsLikeRow, message_cls=_message_class())


def test_protobuf_row_mapper_identifies_non_init_target_attribute() -> None:
    """Mapped target attributes must be accepted by the target constructor."""
    with pytest.raises(ValueError, match=r"fields\.count.*init=False.*cannot be populated"):
        ProtobufRowMapper(
            {"fields": {"count": {"from": "count", "type": "int"}}},
            target_cls=_NonInitFieldRow,
            message_cls=_message_class(),
        )


@pytest.mark.parametrize(
    ("target_cls", "mapping"),
    [
        (_DefaultRow, {"count": {"from": "count", "type": "float"}}),
        (_ValidityVectorRow, {"values": {"from": "quality_samples", "type": "int"}}),
    ],
)
def test_protobuf_row_mapper_rejects_mapping_type_mismatched_with_target(
    target_cls: type[object],
    mapping: dict[str, object],
) -> None:
    """Declared conversion types must match scalar or collection element annotations."""
    with pytest.raises(ValueError, match=r"uses type .* but target(?: elements)? require"):
        ProtobufRowMapper(
            {"fields": mapping},
            target_cls=target_cls,
            message_cls=_message_class(),
        )


def test_protobuf_row_mapper_maps_repeated_scalar_to_fixed_tuple() -> None:
    """Terminal repeated scalar fields should populate fixed-length tuples."""
    mapper = ProtobufRowMapper(
        {"fields": {"values": {"from": "latitude_samples", "type": "float"}}},
        target_cls=_FixedVectorRow,
        message_cls=_message_class(),
    )

    row = mapper(_message(latitude_samples=[1.0, 2.0, 3.0]))

    assert row.values == (1.0, 2.0, 3.0)
    assert row.values_valid == (True, True, True)


@pytest.mark.parametrize(
    ("target_cls", "expected", "expected_validity"),
    [
        (_VariableTupleRow, (1.0, 2.0, 3.0, 4.0), None),
        (_VariableListRow, [1.0, 2.0, 3.0, 4.0], [True, True, True, True]),
    ],
)
def test_protobuf_row_mapper_maps_repeated_scalar_to_variable_collection(
    target_cls: type[object],
    expected: tuple[float, ...] | list[float],
    expected_validity: list[bool] | None,
) -> None:
    """Repeated fields should use the tuple or list container declared by the target."""
    mapper = ProtobufRowMapper(
        {"fields": {"values": {"from": "latitude_samples", "type": "float"}}},
        target_cls=target_cls,
        message_cls=_message_class(),
    )

    row = mapper(_message(latitude_samples=[1.0, 2.0, 3.0, 4.0]))

    assert row.values == expected
    assert getattr(row, "values_valid", None) == expected_validity


def test_protobuf_row_mapper_rejects_incompatible_value_and_validity_shapes() -> None:
    """Automatic validity defaults require matching scalar or collection target shapes."""
    with pytest.raises(ValueError, match="must both be scalar or both be collections"):
        ProtobufRowMapper(
            {"fields": {"values": {"from": "latitude_samples", "type": "float"}}},
            target_cls=_IncompatibleValidityShapeRow,
            message_cls=_message_class(),
        )


def test_protobuf_row_mapper_applies_valid_when_to_repeated_enum() -> None:
    """Validity predicates should be evaluated separately for every repeated enum value."""
    mapper = ProtobufRowMapper(
        {
            "fields": {
                "values": {
                    "from": "quality_samples",
                    "type": "bool",
                    "valid_when": {"equals": 3},
                },
            },
        },
        target_cls=_ValidityVectorRow,
        message_cls=_message_class(),
    )

    row = mapper(_message(quality_samples=[3, 1, 3]))

    assert row.values == (True, False, True)


def test_protobuf_row_mapper_maps_nested_terminal_repeated_field() -> None:
    """A nested path may end at a repeated scalar field."""
    mapper = ProtobufRowMapper(
        {"fields": {"values": {"from": "fix.samples", "type": "float"}}},
        target_cls=_FixedVectorRow,
        message_cls=_envelope_message_class(),
    )

    row = mapper(_envelope_message(samples=[1.0, 2.0, 3.0]))

    assert row.values == (1.0, 2.0, 3.0)


def test_protobuf_row_mapper_rejects_wrong_repeated_field_length() -> None:
    """Fixed tuple targets should reject incorrectly sized repeated values."""
    mapper = ProtobufRowMapper(
        {"fields": {"values": {"from": "latitude_samples", "type": "float"}}},
        target_cls=_FixedVectorRow,
        message_cls=_message_class(),
    )

    with pytest.raises(ValueError, match=r"fields\.values expected 3 values"):
        mapper(_message(latitude_samples=[]))


def test_protobuf_row_mapper_rejects_repeated_source_for_scalar_target_eagerly() -> None:
    """Descriptor validation should reject repeated sources for scalar target attributes."""
    with pytest.raises(ValueError, match=r"fields\.count.*tuple or list target"):
        ProtobufRowMapper(
            {"fields": {"count": {"from": "latitude_samples", "type": "int"}}},
            target_cls=_DefaultRow,
            message_cls=_message_class(),
        )


def test_protobuf_row_mapper_identifies_repeated_element_conversion_error() -> None:
    """Repeated conversion errors should identify the source path and failing element index."""
    mapper = ProtobufRowMapper(
        {"fields": {"values": {"from": "latitude_samples", "type": "int"}}},
        target_cls=_IntegerVectorRow,
        message_cls=_message_class(),
    )

    with pytest.raises(ValueError, match=r"fields\.values.*latitude_samples\[1\].*without losing information"):
        mapper(_message(latitude_samples=[1.0, 1.5]))


def test_protobuf_row_mapper_materializes_group_for_list_target() -> None:
    """Grouped scalar fields should use the collection container declared by the target."""
    mapper = ProtobufRowMapper(
        {
            "fields": {
                "values": {
                    "group": ["latitude_deg", "longitude_deg"],
                    "type": "float",
                },
            },
        },
        target_cls=_VariableListRow,
        message_cls=_message_class(),
    )

    row = mapper(_message(latitude_deg=1.0, longitude_deg=2.0))

    assert row.values == [1.0, 2.0]


@pytest.mark.parametrize("source_path", ["fixes.latitude_deg", "fixes"])
def test_protobuf_row_mapper_rejects_repeated_message_paths_at_decode(source_path: str) -> None:
    """Deferred validation rejects repeated message sources and traversal through them."""
    mapper = ProtobufRowMapper(
        {"fields": {"values": {"from": source_path, "type": "float"}}},
        target_cls=_VariableTupleRow,
    )

    with pytest.raises(ValueError, match=r"fields\.values.*fixes.*unsupported repeated"):
        mapper(_envelope_message_class()())


def test_protobuf_row_mapper_rejects_repeated_field_inside_group_eagerly() -> None:
    """Groups remain collections of separately named scalar source fields."""
    with pytest.raises(ValueError, match=r"fields\.values.*latitude_samples.*unsupported repeated"):
        ProtobufRowMapper(
            {"fields": {"values": {"group": ["latitude_samples"], "type": "float"}}},
            target_cls=_VariableTupleRow,
            message_cls=_message_class(),
        )


def test_protobuf_row_mapper_maps_mcap_logtime_metadata() -> None:
    """MCAP-aware row mapper decode should allow YAML to read message log time."""
    message = _message()
    row_mapper = ProtobufRowMapper(
        {
            "fields": {
                "sensor_timestamp_ns": {"from": "$mcap.logtime", "type": "timestamp", "unit": "ns"},
            },
        },
        target_cls=_TimestampRow,
    )

    row = row_mapper(message, mcap_logtime_ns=987654321)

    assert row.sensor_timestamp_ns == 987654321


def test_protobuf_row_mapper_rejects_mcap_logtime_without_metadata() -> None:
    """Plain protobuf mappers should fail clearly when MCAP metadata is required."""
    mapper = ProtobufRowMapper(
        {
            "fields": {
                "sensor_timestamp_ns": {"from": "$mcap.logtime", "type": "timestamp", "unit": "ns"},
            },
        },
        target_cls=_TimestampRow,
        message_cls=_message_class(),
    )

    with pytest.raises(ValueError, match="requires MCAP metadata"):
        mapper(_message())


def test_protobuf_row_mapper_rejects_unsupported_mcap_metadata_source() -> None:
    """Only the MCAP metadata fields explicitly supported by YAML should be accepted."""
    with pytest.raises(ValueError, match="unsupported MCAP metadata source"):
        ProtobufRowMapper(
            {
                "fields": {
                    "sensor_timestamp_ns": {"from": "$mcap.publish_time", "type": "timestamp", "unit": "ns"},
                },
            },
            target_cls=_TimestampRow,
        )


def test_protobuf_row_mapper_rejects_unknown_protobuf_field_without_descriptor() -> None:
    """Source field errors still identify the YAML target field without an upfront descriptor."""
    spec = _gps_like_spec()
    _cast_mapping(spec["fields"])["latitude_deg"] = {"from": "missing_proto_field", "type": "float"}
    mapper = ProtobufRowMapper(spec, target_cls=_GpsLikeRow)

    with pytest.raises(ValueError, match=r"fields.latitude_deg.*missing_proto_field"):
        mapper(_message(sensor_timestamp_us=1, longitude_deg=8.2, altitude_m=500.5, hdop=0.8))


@pytest.mark.parametrize(
    ("bad_mapping", "match"),
    [
        ({"from": "count", "group": ["count"], "type": "int"}, r"exactly one"),
        ({"from": "count"}, r"fields.count.type is required"),
        ({"from": "count", "type": "int", "valid_when": {"equals": 5}}, r"valid_when"),
        ({"from": "count", "type": "bool", "valid_when": {"nonzero": True}}, r"valid_when"),
        ({"default": True, "type": "bool", "valid_when": {"equals": 5}}, r"valid_when"),
    ],
)
def test_protobuf_row_mapper_rejects_invalid_mapping_shapes(bad_mapping: dict[str, object], match: str) -> None:
    """Invalid mapping declarations fail clearly."""
    spec = {
        "fields": {
            "count": bad_mapping,
        },
    }

    with pytest.raises(ValueError, match=match):
        ProtobufRowMapper(spec, target_cls=_DefaultRow, message_cls=_message_class())


def test_protobuf_row_mapper_maps_decoded_message() -> None:
    """Reusable row mappers map decoded protobuf messages into target rows."""
    payload = _message(sensor_timestamp_us=123, latitude_deg=47.1, longitude_deg=8.2, altitude_m=500.5, hdop=0.8)
    row_mapper = ProtobufRowMapper(_gps_like_spec(), target_cls=_GpsLikeRow)

    row = row_mapper(payload)

    assert row.sensor_timestamp_ns == 123_000
    assert row.align_timestamp_ns == 123_000
    assert row.latitude_deg == 47.1
    assert row.longitude_deg == 8.2
    assert row.altitude_m == 500.5


def _cast_mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value
