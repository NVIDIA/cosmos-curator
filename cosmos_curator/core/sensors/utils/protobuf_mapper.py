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
"""YAML-built protobuf-to-row-object mappers."""

import dataclasses
import math
import types
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, Union, cast, get_args, get_origin, get_type_hints

import attrs
import yaml
from google.protobuf.descriptor import Descriptor, FieldDescriptor
from google.protobuf.message import Message

_OUTPUT_TYPES: Mapping[str, type[object]] = {
    "float": float,
    "int": int,
    "bool": bool,
    "timestamp": int,
    "protobuf_timestamp": int,
}
_SUPPORTED_TYPES = frozenset(_OUTPUT_TYPES)
_TIMESTAMP_UNITS_TO_NS = {"ns": 1, "us": 1_000, "ms": 1_000_000, "s": 1_000_000_000}
_MCAP_LOGTIME_SOURCE = "$mcap.logtime"
_PROTOBUF_TIMESTAMP_FULL_NAME = "google.protobuf.Timestamp"
_SENSOR_TIMESTAMP_FIELDS = ("sensor_timestamp_ns", "sensor_timestamps_ns")
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
PROTOBUF_NUMERIC_TYPES = frozenset(
    {
        FieldDescriptor.TYPE_DOUBLE,
        FieldDescriptor.TYPE_FLOAT,
        FieldDescriptor.TYPE_INT64,
        FieldDescriptor.TYPE_UINT64,
        FieldDescriptor.TYPE_INT32,
        FieldDescriptor.TYPE_FIXED64,
        FieldDescriptor.TYPE_FIXED32,
        FieldDescriptor.TYPE_UINT32,
        FieldDescriptor.TYPE_SFIXED32,
        FieldDescriptor.TYPE_SFIXED64,
        FieldDescriptor.TYPE_SINT32,
        FieldDescriptor.TYPE_SINT64,
        FieldDescriptor.TYPE_ENUM,
    }
)
_ALIGN_TIMESTAMP_FIELD_PAIRS = (
    ("align_timestamp_ns", "sensor_timestamp_ns"),
    ("align_timestamps_ns", "sensor_timestamps_ns"),
)
MISSING = object()


@dataclasses.dataclass(frozen=True)
class FieldMapping:
    """Normalized form of one YAML field mapping."""

    target_name: str
    sources: tuple[tuple[str, tuple[str, ...] | None], ...]
    is_group: bool
    default: object
    value_type: str | None
    unit: str | None
    valid_when_equals: object


def validate_valid_when_source(mapping: FieldMapping, source_name: str, source_type: int) -> None:
    """Validate a ``valid_when`` comparand against its protobuf source type."""
    if mapping.valid_when_equals is MISSING:
        return

    comparand = mapping.valid_when_equals
    if source_type == FieldDescriptor.TYPE_BOOL and isinstance(comparand, bool):
        return
    if (
        source_type in PROTOBUF_NUMERIC_TYPES
        and isinstance(comparand, (int, float))
        and not isinstance(comparand, bool)
    ):
        return

    if source_type == FieldDescriptor.TYPE_BOOL:
        expected = "a bool 'equals' value"
    elif source_type in PROTOBUF_NUMERIC_TYPES:
        expected = "a non-bool numeric or enum 'equals' value"
    else:
        expected = "a numeric, enum, or bool protobuf source"
    msg = (
        f"fields.{mapping.target_name}.valid_when.equals {comparand!r} is incompatible with protobuf source "
        f"{source_name!r}; expected {expected}"
    )
    raise ValueError(msg)


class ProtobufRowMapper[Target]:
    """Map decoded protobuf messages into row objects using compiled YAML."""

    def __init__(
        self,
        yaml_spec: str | Path | Mapping[str, Any],
        *,
        target_cls: type[Target],
        message_cls: type[Message] | None = None,
    ) -> None:
        """Compile a mapping and validate a known protobuf descriptor."""
        spec = load_spec(yaml_spec)
        if "target" in spec:
            msg = "protobuf mapper YAML must not define 'target'; the sensor supplies the target class"
            raise ValueError(msg)
        self._target_cls = validate_target_class(target_cls, "target_cls")
        target_defaults: set[str]
        non_init_target_names: set[str]
        self._target_annotations, target_defaults, non_init_target_names = target_metadata(self._target_cls)
        root_path = parse_optional_str(spec.get("root"), "root") or ""
        self._root_segments = path_segments(root_path, "root") if root_path else ()
        self._field_mappings = _parse_field_mappings(spec, self._target_annotations, non_init_target_names)
        (
            self._align_timestamp_defaults,
            float_defaults,
            scalar_validity_defaults,
            self._collection_validity_default_pairs,
        ) = _prepare_target_defaults(self._target_annotations, target_defaults, self._field_mappings)
        self._static_defaults = {
            **dict.fromkeys(float_defaults, math.nan),
            **scalar_validity_defaults,
        }
        self._validated_message_classes: set[type[Message]] = set()
        if message_cls is not None:
            self._validate_message_cls(message_cls)
            self._validated_message_classes.add(message_cls)

    @property
    def mapped_target_names(self) -> frozenset[str]:
        """Target attributes that the YAML mapping emits."""
        return frozenset(self._field_mappings)

    def __call__(self, message: Message, *, mcap_logtime_ns: int | None = None) -> Target:
        """Map one decoded protobuf message into the configured target row."""
        message_cls = type(message)
        if message_cls not in self._validated_message_classes:
            self._validate_message_cls(message_cls)
            self._validated_message_classes.add(message_cls)

        root: object = message
        if self._root_segments:
            root, _is_repeated = _resolve_path(message, self._root_segments)
            if root is MISSING:
                msg = f"root path {'.'.join(self._root_segments)!r} was not found in protobuf message"
                raise ValueError(msg)
        values = {
            mapping.target_name: _resolve_mapping_value(
                root,
                mapping,
                self._target_annotations[mapping.target_name],
                mcap_logtime_ns=mcap_logtime_ns,
            )
            for mapping in self._field_mappings.values()
        }
        for align_name, sensor_name in self._align_timestamp_defaults:
            values[align_name] = values[sensor_name]
        values.update(self._static_defaults)
        for validity_name, value_name in self._collection_validity_default_pairs:
            values[validity_name] = _validity_default(
                self._target_annotations[validity_name],
                values[value_name],
                validity_name,
            )
        try:
            return cast("Target", self._target_cls(**values))
        except TypeError as e:
            msg = f"failed to construct target {self._target_cls.__module__}.{self._target_cls.__qualname__}: {e}"
            raise ValueError(msg) from e

    def _validate_message_cls(self, message_cls: type[Message]) -> None:
        root_descriptor = message_cls.DESCRIPTOR
        if self._root_segments:
            root_path = ".".join(self._root_segments)
            root_field = resolve_descriptor_path(
                root_descriptor,
                self._root_segments,
                "root",
                allow_terminal_repeated=False,
            )
            if root_field.message_type is None:
                msg = f"root path {root_path!r} does not resolve to a protobuf message"
                raise ValueError(msg)
            root_descriptor = root_field.message_type
        for mapping in self._field_mappings.values():
            for source_name, source_path in mapping.sources:
                if source_path is None:
                    continue
                source_field = resolve_descriptor_path(
                    root_descriptor,
                    source_path,
                    f"fields.{mapping.target_name}",
                    allow_terminal_repeated=not mapping.is_group,
                )
                validate_valid_when_source(mapping, source_name, source_field.type)
                if mapping.value_type == "protobuf_timestamp" and (
                    source_field.message_type is None
                    or source_field.message_type.full_name != _PROTOBUF_TIMESTAMP_FULL_NAME
                ):
                    msg = (
                        f"fields.{mapping.target_name} uses type 'protobuf_timestamp', but source "
                        f"{source_name!r} is not {_PROTOBUF_TIMESTAMP_FULL_NAME}"
                    )
                    raise ValueError(msg)
                if (
                    mapping.value_type == "bool"
                    and mapping.valid_when_equals is MISSING
                    and source_field.type != FieldDescriptor.TYPE_BOOL
                ):
                    msg = (
                        f"fields.{mapping.target_name} uses direct bool conversion for non-bool protobuf field "
                        f"{source_name!r}; use valid_when for numeric or enum validity fields"
                    )
                    raise ValueError(msg)
                if source_field.is_repeated:
                    require_collection_target(self._target_annotations[mapping.target_name], mapping.target_name)


def load_spec(yaml_spec: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    """Load and validate a protobuf mapping specification."""
    if isinstance(yaml_spec, Mapping):
        return yaml_spec
    try:
        if isinstance(yaml_spec, Path):
            loaded = yaml.safe_load(yaml_spec.read_text(encoding="utf-8"))
        else:
            loaded = yaml.safe_load(yaml_spec)
    except yaml.YAMLError as e:
        msg = "failed to parse protobuf mapper YAML spec"
        raise ValueError(msg) from e
    return require_mapping(loaded, "protobuf mapper YAML spec")


def validate_target_class(target_cls: type[Any], name: str) -> type[Any]:
    """Require a dataclass or attrs mapping target class."""
    if not dataclasses.is_dataclass(target_cls) and not attrs.has(target_cls):
        msg = f"{name} must be a dataclass or attrs class"
        raise ValueError(msg)
    return target_cls


def target_metadata(target_cls: type[Any]) -> tuple[dict[str, object], set[str], set[str]]:
    """Return target annotations, defaulted fields, and non-init fields."""
    try:
        type_hints = get_type_hints(target_cls, include_extras=True)
    except (NameError, TypeError, AttributeError) as e:
        msg = f"failed to resolve type annotations for target {target_cls.__module__}.{target_cls.__qualname__}"
        raise ValueError(msg) from e

    if attrs.has(target_cls):
        fields = attrs.fields(target_cls)
        return (
            {field.name: type_hints.get(field.name, field.type) for field in fields if field.init},
            {field.name for field in fields if field.init and field.default is not attrs.NOTHING},
            {field.name for field in fields if not field.init},
        )

    fields = dataclasses.fields(target_cls)
    return (
        {field.name: type_hints.get(field.name, field.type) for field in fields if field.init},
        {
            field.name
            for field in fields
            if field.init
            and (field.default is not dataclasses.MISSING or field.default_factory is not dataclasses.MISSING)
        },
        {field.name for field in fields if not field.init},
    )


def resolve_descriptor_path(
    descriptor: Descriptor,
    path_segments: tuple[str, ...],
    context: str,
    *,
    allow_terminal_repeated: bool,
) -> FieldDescriptor:
    """Resolve and validate a field path in a protobuf descriptor."""
    current = descriptor
    segments = path_segments
    path = ".".join(segments)
    for segment in segments[:-1]:
        field = current.fields_by_name.get(segment)
        if field is None:
            msg = f"{context} references unknown protobuf field {segment!r} in path {path!r}"
            raise ValueError(msg)
        if field.is_repeated:
            msg = f"{context} path {path!r} references unsupported repeated protobuf field {segment!r}"
            raise ValueError(msg)
        if field.message_type is None:
            msg = f"{context} path {path!r} traverses non-message protobuf field {segment!r}"
            raise ValueError(msg)
        current = field.message_type

    terminal = segments[-1]
    field = current.fields_by_name.get(terminal)
    if field is None:
        msg = f"{context} references unknown protobuf field {terminal!r} in path {path!r}"
        raise ValueError(msg)
    if field.is_repeated and not allow_terminal_repeated:
        msg = f"{context} path {path!r} references unsupported repeated protobuf field {terminal!r}"
        raise ValueError(msg)
    if field.is_repeated and field.message_type is not None:
        msg = f"{context} path {path!r} references unsupported repeated protobuf message field {terminal!r}"
        raise ValueError(msg)
    return field


def _parse_field_mappings(
    spec: Mapping[str, Any],
    target_annotations: Mapping[str, object],
    non_init_target_names: set[str],
) -> dict[str, FieldMapping]:
    fields = require_mapping(spec.get("fields"), "fields")
    target_names = set(target_annotations)
    mappings: dict[str, FieldMapping] = {}
    for target_name, raw_mapping in fields.items():
        if not isinstance(target_name, str):
            msg = "fields keys must be target attribute names"
            raise ValueError(msg)  # noqa: TRY004
        if target_name in non_init_target_names:
            msg = (
                f"fields.{target_name} references target attribute {target_name!r}, but that attribute has init=False "
                "and cannot be populated by the mapper"
            )
            raise ValueError(msg)
        if target_name not in target_names:
            msg = f"unknown target attribute {target_name!r}; expected one of: {', '.join(sorted(target_names))}"
            raise ValueError(msg)
        field_mapping = parse_one_mapping(target_name, raw_mapping)
        validate_mapping_target_type(target_annotations[target_name], field_mapping)
        mappings[target_name] = field_mapping
    return mappings


def parse_one_mapping(target_name: str, raw_mapping: object) -> FieldMapping:
    """Parse and normalize one target field mapping."""
    raw_spec = require_mapping(raw_mapping, f"fields.{target_name}")
    source_name = parse_optional_str(raw_spec.get("from"), f"fields.{target_name}.from")
    group = _parse_optional_group(raw_spec.get("group"), f"fields.{target_name}.group")
    has_default = "default" in raw_spec
    active = sum([source_name is not None, group is not None, has_default])
    if active != 1:
        msg = f"fields.{target_name} must specify exactly one of 'from', 'group', or 'default'"
        raise ValueError(msg)
    if has_default and raw_spec["default"] is None:
        msg = f"fields.{target_name}.default must not be null; omit the field to leave an optional target absent"
        raise ValueError(msg)

    value_type = parse_optional_str(raw_spec.get("type"), f"fields.{target_name}.type")
    if value_type is not None and value_type not in _SUPPORTED_TYPES:
        msg = f"fields.{target_name}.type must be one of {sorted(_SUPPORTED_TYPES)}, got {value_type!r}"
        raise ValueError(msg)
    if (source_name is not None or group is not None) and value_type is None:
        msg = f"fields.{target_name}.type is required when using 'from' or 'group'"
        raise ValueError(msg)

    unit = parse_optional_str(raw_spec.get("unit"), f"fields.{target_name}.unit")
    if value_type == "timestamp":
        if unit not in _TIMESTAMP_UNITS_TO_NS:
            msg = f"fields.{target_name}.unit must be one of {sorted(_TIMESTAMP_UNITS_TO_NS)}, got {unit!r}"
            raise ValueError(msg)
    elif unit is not None:
        msg = f"fields.{target_name}.unit is only valid for timestamp mappings"
        raise ValueError(msg)
    if value_type == "protobuf_timestamp" and (group is not None or has_default or source_name == _MCAP_LOGTIME_SOURCE):
        msg = f"fields.{target_name} type 'protobuf_timestamp' requires one protobuf message source, not MCAP metadata"
        raise ValueError(msg)
    valid_when_equals = _parse_valid_when_equals(
        raw_spec.get("valid_when"),
        target_name,
        has_default=has_default,
        value_type=value_type,
    )

    context = f"fields.{target_name}"
    source_names = (source_name,) if source_name is not None else group or ()
    sources = tuple(
        (source, None if _validate_mcap_source(source, context) else path_segments(source, context))
        for source in source_names
    )
    field_mapping = FieldMapping(
        target_name=target_name,
        sources=sources,
        is_group=group is not None,
        default=raw_spec.get("default", MISSING),
        value_type=value_type,
        unit=unit,
        valid_when_equals=valid_when_equals,
    )
    if field_mapping.default is not MISSING:
        field_mapping = dataclasses.replace(
            field_mapping,
            default=_convert_value(field_mapping.default, field_mapping),
        )
    return field_mapping


def _validate_mcap_source(source_name: str, context: str) -> bool:
    if not (source_name == "$mcap" or source_name.startswith("$mcap.")):
        return False
    if source_name != _MCAP_LOGTIME_SOURCE:
        msg = (
            f"{context} references unsupported MCAP metadata source {source_name!r}; "
            f"supported source is {_MCAP_LOGTIME_SOURCE!r}"
        )
        raise ValueError(msg)
    return True


def validate_mapping_target_type(annotation: object, mapping: FieldMapping) -> None:
    """Validate that a mapping can populate its target annotation."""
    collection_target = _collection_target(annotation)
    if mapping.is_group:
        _collection_kind, expected_length = require_collection_target(annotation, mapping.target_name)
        if expected_length is not None and len(mapping.sources) != expected_length:
            msg = f"fields.{mapping.target_name} expected {expected_length} grouped values, got {len(mapping.sources)}"
            raise ValueError(msg)

    output_type = (
        bool
        if mapping.valid_when_equals is not MISSING
        else _OUTPUT_TYPES.get(mapping.value_type or "", type(mapping.default))
    )
    if output_type is types.NoneType:
        if types.NoneType not in get_args(annotation):
            msg = f"fields.{mapping.target_name} defaults to None, but the target annotation does not allow None"
            raise ValueError(msg)
        return
    if mapping.default is not MISSING and mapping.value_type is None and collection_target is not None:
        return

    resolved_annotation = _strip_none(annotation)
    expected_types = get_args(resolved_annotation) if collection_target is not None else (resolved_annotation,)
    if expected_types and expected_types[-1] is Ellipsis:
        expected_types = (expected_types[0],)
    for expected_type in expected_types:
        resolved_expected_type = _strip_none(expected_type)
        if resolved_expected_type in (Any, object) or resolved_expected_type is output_type:
            continue
        mapping_type = mapping.value_type or type(mapping.default).__name__
        target_label = "target elements" if collection_target is not None else "target"
        msg = (
            f"fields.{mapping.target_name} uses type {mapping_type!r}, but {target_label} require "
            f"{getattr(resolved_expected_type, '__name__', repr(resolved_expected_type))}"
        )
        raise ValueError(msg)


def _resolve_mapping_value(
    root: object,
    mapping: FieldMapping,
    target_annotation: object,
    *,
    mcap_logtime_ns: int | None,
) -> object:
    if mapping.default is not MISSING:
        return mapping.default
    if not mapping.is_group:
        source = mapping.sources[0]
        value, is_repeated = _source_field_value(
            root,
            source,
            mapping.target_name,
            mcap_logtime_ns=mcap_logtime_ns,
        )
        if is_repeated:
            return _convert_collection(
                value,
                mapping,
                target_annotation,
                source_names=(source[0],),
            )
        return _convert_value(value, mapping)

    grouped_values = tuple(
        _source_field_value(
            root,
            source,
            mapping.target_name,
            mcap_logtime_ns=mcap_logtime_ns,
        )[0]
        for source in mapping.sources
    )
    return _convert_collection(
        grouped_values,
        mapping,
        target_annotation,
        source_names=tuple(source_name for source_name, _source_path in mapping.sources),
    )


def _source_field_value(
    root: object,
    source: tuple[str, tuple[str, ...] | None],
    target_name: str,
    *,
    mcap_logtime_ns: int | None,
) -> tuple[object, bool]:
    source_name, source_path = source
    if source_path is None:
        if mcap_logtime_ns is None:
            msg = f"fields.{target_name} path {source_name!r} requires MCAP metadata"
            raise ValueError(msg)
        return mcap_logtime_ns, False

    value, is_repeated = _resolve_path(root, source_path)
    if value is MISSING:
        msg = f"fields.{target_name} path {source_name!r} was not found in protobuf message"
        raise ValueError(msg)
    return value, is_repeated


def _resolve_path(root: object, path_segments: tuple[str, ...]) -> tuple[object, bool]:
    """Read a descriptor-validated path while preserving message presence."""
    current = cast("Message", root)
    for index, segment in enumerate(path_segments):
        field = current.DESCRIPTOR.fields_by_name[segment]
        if field.is_repeated:
            return getattr(current, segment), True
        if field.message_type is not None and not current.HasField(segment):
            return MISSING, False
        value = getattr(current, segment)
        if index == len(path_segments) - 1:
            return value, False
        current = cast("Message", value)
    return current, False


def path_segments(path: str, context: str) -> tuple[str, ...]:
    """Split and validate a dot-separated protobuf field path."""
    if not isinstance(path, str) or not path:
        msg = f"{context} path must be a non-empty string"
        raise ValueError(msg)
    segments = tuple(path.split("."))
    if any(not segment for segment in segments):
        msg = f"{context} path {path!r} cannot contain empty segments"
        raise ValueError(msg)
    return segments


def _convert_collection(
    value: object,
    mapping: FieldMapping,
    target_annotation: object,
    *,
    source_names: tuple[str, ...],
) -> object:
    values = tuple(cast("Any", value))
    repeated_source = not mapping.is_group
    converted: list[object] = []
    for index, item in enumerate(values):
        source_name = source_names[0] if repeated_source else source_names[index]
        source_context = f"{source_name}[{index}]" if repeated_source else source_name
        try:
            converted.append(_convert_value(item, mapping))
        except ValueError as e:
            msg = f"fields.{mapping.target_name} source path {source_context!r} failed conversion: {e}"
            raise ValueError(msg) from e
    return _materialize_collection(converted, target_annotation, mapping.target_name)


def _materialize_collection(values: Sequence[object], annotation: object, target_name: str) -> object:
    collection_kind, expected_length = require_collection_target(annotation, target_name)
    if expected_length is not None and len(values) != expected_length:
        msg = f"fields.{target_name} expected {expected_length} values, got {len(values)}"
        raise ValueError(msg)
    if collection_kind == "tuple":
        return tuple(values)
    return list(values)


def _convert_value(value: object, mapping: FieldMapping) -> object:
    if mapping.value_type is None:
        converted = value
    elif mapping.valid_when_equals is not MISSING:
        converted = value == mapping.valid_when_equals
    elif mapping.value_type == "float":
        converted = _convert_float(value, mapping.target_name)
    elif mapping.value_type == "int":
        converted = _convert_int(value, mapping.target_name)
    elif mapping.value_type == "bool":
        if isinstance(value, bool):
            converted = value
        else:
            msg = f"fields.{mapping.target_name} requires a bool value; use valid_when for numeric validity fields"
            raise ValueError(msg)
    elif mapping.value_type == "protobuf_timestamp":
        converted = _protobuf_timestamp_to_ns(value, mapping.target_name)
    else:
        unit = cast("str", mapping.unit)
        converted = _timestamp_to_ns(value, unit, mapping.target_name)
    return converted


def _convert_float(value: object, target_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"fields.{target_name} cannot convert value {value!r} to float"
        raise ValueError(msg)  # noqa: TRY004
    if isinstance(value, float):
        return value
    try:
        converted = float(value)
    except OverflowError as e:
        msg = f"fields.{target_name} cannot convert value {value!r} to float"
        raise ValueError(msg) from e
    if int(converted) != value:
        msg = f"fields.{target_name} cannot convert value {value!r} to float without losing information"
        raise ValueError(msg)
    return converted


def _convert_int(value: object, target_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"fields.{target_name} cannot convert value {value!r} to int"
        raise ValueError(msg)  # noqa: TRY004
    if isinstance(value, int):
        return value
    if not math.isfinite(value) or not value.is_integer():
        msg = f"fields.{target_name} cannot convert value {value!r} to int without losing information"
        raise ValueError(msg)
    return int(value)


def _timestamp_to_ns(value: object, unit: str, target_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"fields.{target_name} timestamp value must be numeric"
        raise ValueError(msg)  # noqa: TRY004
    scale = _TIMESTAMP_UNITS_TO_NS[unit]
    if isinstance(value, int):
        return value * scale
    if not math.isfinite(value):
        msg = (
            f"fields.{target_name} timestamp value {value!r} with unit {unit!r} cannot be converted to integer "
            "nanoseconds without losing information"
        )
        raise ValueError(msg)
    numerator, denominator = value.as_integer_ratio()
    scaled_numerator = numerator * scale
    result, remainder = divmod(scaled_numerator, denominator)
    if remainder:
        msg = (
            f"fields.{target_name} timestamp value {value!r} with unit {unit!r} cannot be converted to integer "
            "nanoseconds without losing information"
        )
        raise ValueError(msg)
    return result


def _protobuf_timestamp_to_ns(value: object, target_name: str) -> int:
    """Convert one ``google.protobuf.Timestamp`` message to integer nanoseconds."""
    if not isinstance(value, Message) or value.DESCRIPTOR.full_name != _PROTOBUF_TIMESTAMP_FULL_NAME:
        msg = f"fields.{target_name} requires a {_PROTOBUF_TIMESTAMP_FULL_NAME} value"
        raise ValueError(msg)
    seconds = cast("int", value.seconds)
    nanos = cast("int", value.nanos)
    if not 0 <= nanos < _TIMESTAMP_UNITS_TO_NS["s"]:
        msg = f"fields.{target_name} protobuf timestamp nanos must be in [0, 1000000000), got {nanos}"
        raise ValueError(msg)
    timestamp_ns = seconds * _TIMESTAMP_UNITS_TO_NS["s"] + nanos
    if not _INT64_MIN <= timestamp_ns <= _INT64_MAX:
        msg = f"fields.{target_name} protobuf timestamp does not fit int64 nanoseconds"
        raise ValueError(msg)
    return timestamp_ns


def _prepare_target_defaults(
    target_annotations: Mapping[str, object],
    target_defaults: set[str],
    field_mappings: Mapping[str, FieldMapping],
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...], Mapping[str, object], tuple[tuple[str, str], ...]]:
    """Validate target conventions and precompute values omitted from YAML."""
    target_names = set(target_annotations)
    for timestamp_name in _SENSOR_TIMESTAMP_FIELDS:
        if timestamp_name in target_names and timestamp_name not in field_mappings:
            msg = f"protobuf mapper requires mapping or default for target timestamp field {timestamp_name!r}"
            raise ValueError(msg)

    for target_name in field_mappings:
        if not target_name.endswith("_valid"):
            continue
        paired_name = paired_target_name(target_name, target_names)
        if paired_name in target_names and paired_name not in field_mappings:
            msg = f"fields.{target_name} maps validity for {paired_name!r}, but fields.{paired_name} is not mapped"
            raise ValueError(msg)

    align_defaults: list[tuple[str, str]] = []
    for align_name, sensor_name in _ALIGN_TIMESTAMP_FIELD_PAIRS:
        if align_name not in target_names or align_name in field_mappings or align_name in target_defaults:
            continue
        sensor_mapping = field_mappings.get(sensor_name)
        if sensor_mapping is not None and sensor_mapping.sources and not sensor_mapping.is_group:
            align_defaults.append((align_name, sensor_name))
            continue
        msg = (
            f"protobuf mapper requires mapping or live sensor timestamp source for target timestamp field "
            f"{align_name!r}; map {align_name!r} explicitly when {sensor_name!r} uses a static default"
        )
        raise ValueError(msg)

    float_defaults = tuple(
        name
        for name, annotation in target_annotations.items()
        if name not in field_mappings and name not in target_defaults and _strip_none(annotation) is float
    )
    scalar_validity_defaults, collection_validity_pairs = _prepare_validity_defaults(
        target_annotations,
        field_mappings,
    )
    return tuple(align_defaults), float_defaults, scalar_validity_defaults, collection_validity_pairs


def _prepare_validity_defaults(
    target_annotations: Mapping[str, object],
    field_mappings: Mapping[str, FieldMapping],
) -> tuple[Mapping[str, object], tuple[tuple[str, str], ...]]:
    """Precompute omitted validity values and validate paired collection shapes."""
    scalar_defaults: dict[str, object] = {}
    collection_pairs: list[tuple[str, str]] = []
    for name, annotation in target_annotations.items():
        if name in field_mappings or not name.endswith("_valid") or not _is_bool_compatible_annotation(annotation):
            continue
        paired_name = paired_target_name(name, target_annotations)
        if paired_name not in target_annotations or paired_name not in field_mappings:
            continue
        validity_collection = _collection_target(annotation)
        value_collection = _collection_target(target_annotations[paired_name])
        if (validity_collection is None) != (value_collection is None):
            msg = f"target fields {paired_name!r} and {name!r} must both be scalar or both be collections"
            raise ValueError(msg)
        if validity_collection is None:
            scalar_defaults[name] = True
            continue
        if value_collection is not None:
            _validity_kind, validity_length = validity_collection
            _value_kind, value_length = value_collection
            if validity_length is not None and value_length is not None and validity_length != value_length:
                msg = (
                    f"target fields {paired_name!r} and {name!r} declare incompatible lengths "
                    f"{value_length} and {validity_length}"
                )
                raise ValueError(msg)
        collection_pairs.append((name, paired_name))
    return scalar_defaults, tuple(collection_pairs)


def paired_target_name(validity_name: str, target_names: Collection[str]) -> str:
    """Return a validity field's paired value target, including its closest unit-suffixed name."""
    base_name = validity_name.removesuffix("_valid")
    if base_name in target_names:
        return base_name
    unit_suffixed_names = [
        name for name in target_names if name.startswith(f"{base_name}_") and not name.endswith("_valid")
    ]
    if not unit_suffixed_names:
        return base_name

    suffix_lengths = {name: len(name.removeprefix(f"{base_name}_").split("_")) for name in unit_suffixed_names}
    closest_names = [name for name in unit_suffixed_names if suffix_lengths[name] == min(suffix_lengths.values())]
    if len(closest_names) > 1:
        msg = (
            f"validity target {validity_name!r} has ambiguous unit-suffixed value targets: "
            f"{', '.join(sorted(closest_names))}"
        )
        raise ValueError(msg)
    return closest_names[0]


def _validity_default(annotation: object, paired_value: object, target_name: str) -> object:
    if _strip_none(annotation) is bool:
        return True
    if isinstance(paired_value, str) or not isinstance(paired_value, Sequence):
        msg = f"fields.{target_name} requires a collection value to build its validity default"
        raise ValueError(msg)  # noqa: TRY004
    return _materialize_collection([True] * len(paired_value), annotation, target_name)


def require_collection_target(annotation: object, target_name: str) -> tuple[str, int | None]:
    """Return collection metadata or reject a scalar target annotation."""
    collection_target = _collection_target(annotation)
    if collection_target is None:
        msg = f"fields.{target_name} requires a tuple or list target annotation for collection values"
        raise ValueError(msg)
    return collection_target


def _collection_target(annotation: object) -> tuple[str, int | None] | None:
    annotation = _strip_none(annotation)
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is tuple:
        if args and args[-1] is Ellipsis:
            return "tuple", None
        return "tuple", len(args)
    if origin is list and len(args) == 1:
        return "list", None
    return None


def _is_bool_compatible_annotation(annotation: object) -> bool:
    annotation = _strip_none(annotation)
    if annotation is bool:
        return True
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is tuple:
        return bool(args) and (args == (bool, Ellipsis) or all(arg is bool for arg in args))
    if origin is list:
        return args == (bool,)
    return False


def _strip_none(annotation: object) -> object:
    origin = get_origin(annotation)
    if origin not in (types.UnionType, Union):
        return annotation
    args = tuple(arg for arg in get_args(annotation) if arg is not types.NoneType)
    if len(args) == 1:
        return args[0]
    return annotation


def require_mapping(value: object, name: str) -> Mapping[Any, Any]:
    """Require and return a mapping value."""
    if not isinstance(value, Mapping):
        msg = f"{name} must be a mapping"
        raise ValueError(msg)  # noqa: TRY004
    return value


def parse_optional_str(value: object, name: str) -> str | None:
    """Return an optional string or reject another value type."""
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"{name} must be a string"
        raise ValueError(msg)  # noqa: TRY004
    return value


def _parse_optional_group(value: object, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, Sequence):
        msg = f"{name} must be a sequence of protobuf field names"
        raise ValueError(msg)  # noqa: TRY004
    group = tuple(value)
    if not group or not all(isinstance(item, str) for item in group):
        msg = f"{name} must be a non-empty sequence of protobuf field names"
        raise ValueError(msg)
    for item in group:
        path_segments(item, name)
    return group


def _parse_valid_when_equals(
    value: object,
    target_name: str,
    *,
    has_default: bool,
    value_type: str | None,
) -> object:
    if value is None:
        return MISSING
    if value_type != "bool":
        msg = f"fields.{target_name}.valid_when is only valid for bool mappings"
        raise ValueError(msg)
    if has_default:
        msg = f"fields.{target_name}.valid_when cannot be used with default mappings"
        raise ValueError(msg)
    valid_when = require_mapping(value, f"fields.{target_name}.valid_when")
    if len(valid_when) != 1 or "equals" not in valid_when:
        msg = f"fields.{target_name}.valid_when must specify exactly one supported predicate: 'equals'"
        raise ValueError(msg)
    equals = valid_when["equals"]
    if equals is None:
        msg = f"fields.{target_name}.valid_when.equals must not be null"
        raise ValueError(msg)
    return equals
