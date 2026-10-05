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
"""Test MCAP utilities for the sensor library."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
from google.protobuf import descriptor_pb2
from google.protobuf.message import Message
from mcap.reader import make_reader as mcap_make_reader
from mcap.records import Channel, Schema
from mcap.summary import Summary
from mcap.writer import CompressionType, IndexType, Writer

from cosmos_curator.core.sensors.utils import mcap as sensor_mcap
from cosmos_curator.core.sensors.utils.mcap import (
    McapProtobufMessageResolver,
    McapTopicAccessor,
    channel_for_topic,
    channel_metadata_for_topic,
    get_metadata_record,
    iter_messages_log_time_ns,
    load_start_end_ns,
    load_timeline,
    require_channel_message_encoding,
    schema_for_channel,
)

_RGB_SCHEMA = {
    "type": "object",
    "title": "cosmos_curator.sensors.rgb8_frame",
    "description": "test",
}
_FIELD_DESCRIPTOR_PROTO = descriptor_pb2.FieldDescriptorProto
_PROTOBUF_SCHEMA_NAME = "cosmos_curator.test.v1.TestMessage"
_PROTOBUF_TOPIC = "/test/protobuf"


def _test_file_descriptor_set(message_name: str = "TestMessage") -> descriptor_pb2.FileDescriptorSet:
    """Build a minimal protobuf descriptor set for resolver tests."""
    file_descriptor_set = descriptor_pb2.FileDescriptorSet()
    file_descriptor = file_descriptor_set.file.add()
    file_descriptor.name = "cosmos_curator/core/sensors/schemas/test.proto"
    file_descriptor.package = "cosmos_curator.test.v1"
    file_descriptor.syntax = "proto3"

    message_descriptor = file_descriptor.message_type.add()
    message_descriptor.name = message_name
    field = message_descriptor.field.add()
    field.name = "sequence"
    field.number = 1
    field.label = _FIELD_DESCRIPTOR_PROTO.LABEL_OPTIONAL
    field.type = _FIELD_DESCRIPTOR_PROTO.TYPE_UINT64
    return file_descriptor_set


def _protobuf_schema(
    *,
    schema_id: int = 1,
    name: str = _PROTOBUF_SCHEMA_NAME,
    encoding: str = "protobuf",
    data: bytes | None = None,
) -> Schema:
    """Build an MCAP protobuf schema for resolver tests."""
    schema_data = data if data is not None else _test_file_descriptor_set().SerializeToString()
    return Schema(id=schema_id, data=schema_data, encoding=encoding, name=name)


def _protobuf_channel(
    *,
    channel_id: int = 1,
    schema_id: int = 1,
    message_encoding: str = "protobuf",
    topic: str = _PROTOBUF_TOPIC,
) -> Channel:
    """Build an MCAP channel for resolver tests."""
    return Channel(
        id=channel_id,
        topic=topic,
        message_encoding=message_encoding,
        metadata={},
        schema_id=schema_id,
    )


def _payload(width: int, height: int, fill: int) -> bytes:
    return bytes([fill]) * (width * height * 3)


def _write_rgb8_mcap(
    path: Path,
    topic: str,
    times_ns: list[int],
    width: int = 2,
    height: int = 2,
    *,
    indexed: bool = True,
) -> None:
    with path.open("wb") as out:
        writer = Writer(
            out,
            compression=CompressionType.ZSTD,
            index_types=IndexType.ALL if indexed else IndexType.NONE,
            repeat_channels=indexed,
            repeat_schemas=indexed,
            use_statistics=indexed,
            use_summary_offsets=indexed,
        )
        writer.start(library="cosmos_curator test")
        schema_id = writer.register_schema(
            name=_RGB_SCHEMA["title"],
            encoding="jsonschema",
            data=json.dumps(_RGB_SCHEMA).encode("utf-8"),
        )
        channel_id = writer.register_channel(
            schema_id=schema_id,
            topic=topic,
            message_encoding="rgb8",
            metadata={"width": str(width), "height": str(height)},
        )
        for i, time_ns in enumerate(times_ns):
            writer.add_message(
                channel_id=channel_id,
                log_time=time_ns,
                data=_payload(width, height, i + 1),
                publish_time=time_ns,
                sequence=i + 1,
            )
        writer.finish()  # type: ignore[no-untyped-call]


def test_load_start_end_ns_and_timeline(tmp_path: Path) -> None:
    """Generic MCAP topic timeline helpers should return ordered bounds and a read-only array."""
    path = tmp_path / "timeline.mcap"
    times_ns = [10, 20, 30]
    _write_rgb8_mcap(path, "/camera/rgb", times_ns)

    with path.open("rb") as stream:
        reader = mcap_make_reader(stream)  # type: ignore[no-untyped-call]
        start_ns, end_ns = load_start_end_ns(reader, "/camera/rgb")
        timeline = load_timeline(reader, "/camera/rgb")

    assert start_ns == 10
    assert end_ns == 30
    np.testing.assert_array_equal(timeline, np.array(times_ns, dtype=np.int64))
    assert not timeline.flags.writeable


def test_mcap_topic_accessor_loads_unindexed_bounds_before_timeline(tmp_path: Path) -> None:
    """Unindexed bounds should scan all records without depending on access order."""
    path = tmp_path / "unindexed_timeline.mcap"
    _write_rgb8_mcap(path, "/camera/rgb", [30, 10, 20], indexed=False)

    with path.open("rb") as stream:
        reader = mcap_make_reader(stream)  # type: ignore[no-untyped-call]
        assert reader.get_summary() is None

    accessor = McapTopicAccessor(path, "/camera/rgb")
    assert accessor.start_ns == 10
    assert accessor.end_ns == 30
    np.testing.assert_array_equal(accessor.timestamps_ns, np.array([10, 20, 30], dtype=np.int64))
    assert accessor.start_ns == 10
    assert accessor.end_ns == 30

    single_path = tmp_path / "unindexed_single.mcap"
    _write_rgb8_mcap(single_path, "/camera/rgb", [10], indexed=False)
    single_accessor = McapTopicAccessor(single_path, "/camera/rgb")
    assert single_accessor.start_ns == 10
    assert single_accessor.end_ns == 10


def test_load_timeline_raises_on_missing_topic(tmp_path: Path) -> None:
    """Generic MCAP topic timeline helpers should reject topics with no messages."""
    path = tmp_path / "missing_topic.mcap"
    _write_rgb8_mcap(path, "/camera/rgb", [10])

    with path.open("rb") as stream:
        reader = mcap_make_reader(stream)  # type: ignore[no-untyped-call]
        with pytest.raises(ValueError, match="no MCAP messages on topic"):
            load_start_end_ns(reader, "/camera/depth")
        with pytest.raises(ValueError, match="no MCAP messages on topic"):
            load_timeline(reader, "/camera/depth")


def test_channel_for_topic_returns_unique_match() -> None:
    """channel_for_topic should return the unique channel for a topic."""
    cam = SimpleNamespace(topic="/camera/rgb")
    depth = SimpleNamespace(topic="/camera/depth")
    summary = SimpleNamespace(channels={1: cam, 2: depth})

    assert channel_for_topic(summary, "/camera/rgb") is cam


def test_channel_for_topic_returns_none_when_topic_is_absent() -> None:
    """channel_for_topic should return None when the topic is missing."""
    summary = SimpleNamespace(channels={1: SimpleNamespace(topic="/camera/rgb")})

    assert channel_for_topic(summary, "/camera/depth") is None


def test_channel_for_topic_raises_on_duplicate_topic() -> None:
    """channel_for_topic should reject ambiguous same-topic channels."""
    summary = SimpleNamespace(
        channels={
            1: SimpleNamespace(topic="/camera/rgb"),
            2: SimpleNamespace(topic="/camera/rgb"),
        }
    )

    with pytest.raises(ValueError, match=r"expected exactly one MCAP channel for topic '/camera/rgb', found 2"):
        channel_for_topic(summary, "/camera/rgb")


def test_channel_metadata_for_topic_rejects_missing_summary() -> None:
    """channel_metadata_for_topic should reject a reader with no MCAP summary."""
    reader = SimpleNamespace(get_summary=lambda: None)

    with pytest.raises(ValueError, match="without an MCAP summary"):
        channel_metadata_for_topic(reader, "/camera/rgb")


def test_channel_metadata_for_topic_rejects_missing_channel() -> None:
    """channel_metadata_for_topic should reject a topic with no matching channel."""
    summary = SimpleNamespace(channels={1: SimpleNamespace(topic="/camera/depth")})
    reader = SimpleNamespace(get_summary=lambda: summary)

    with pytest.raises(ValueError, match="no MCAP channel found for topic '/camera/rgb'"):
        channel_metadata_for_topic(reader, "/camera/rgb")


def test_get_metadata_record_returns_unique_match() -> None:
    """get_metadata_record should return the single matching metadata payload."""
    reader = SimpleNamespace(
        iter_metadata=lambda: iter(
            [
                SimpleNamespace(name="a", metadata={"x": "1"}),
                SimpleNamespace(name="b", metadata={"y": "2"}),
            ]
        )
    )

    assert get_metadata_record(reader, "b") == {"y": "2"}


def test_get_metadata_record_raises_on_missing_or_duplicate_match() -> None:
    """get_metadata_record should reject missing or duplicate records."""
    missing_reader = SimpleNamespace(iter_metadata=lambda: iter([]))
    duplicate_reader = SimpleNamespace(
        iter_metadata=lambda: iter(
            [
                SimpleNamespace(name="dup", metadata={"x": "1"}),
                SimpleNamespace(name="dup", metadata={"x": "2"}),
            ]
        )
    )

    with pytest.raises(ValueError, match=r"required MCAP metadata record 'missing' not found"):
        get_metadata_record(missing_reader, "missing")

    with pytest.raises(ValueError, match=r"expected exactly one MCAP metadata record 'dup', found 2"):
        get_metadata_record(duplicate_reader, "dup")


def test_load_start_end_ns_raises_if_latest_lookup_fails_after_earliest() -> None:
    """load_start_end_ns should surface the rare broken-latest-message path clearly."""

    class _FakeReader:
        def get_summary(self) -> Summary:
            return cast("Summary", SimpleNamespace(chunk_indexes=[object()]))

        def iter_messages(
            self,
            *,
            topics: str,
            log_time_order: bool,
            reverse: bool = False,
        ) -> Iterator[tuple[Any, Any, Any]]:
            del topics, log_time_order
            if reverse:
                return iter(())
            return iter([(None, None, SimpleNamespace(log_time=10))])

    with pytest.raises(
        ValueError,
        match=r"failed to read latest MCAP message on topic '/camera/rgb' after reading earliest message",
    ):
        load_start_end_ns(_FakeReader(), "/camera/rgb")


def test_iter_messages_log_time_ns_forwards_to_reader_iter_messages() -> None:
    """iter_messages_log_time_ns should forward the expected bounds to reader.iter_messages."""
    calls: list[tuple[str, int, int, bool]] = []
    expected = [(None, None, SimpleNamespace(log_time=123))]

    class _FakeReader:
        def iter_messages(
            self,
            *,
            topics: str,
            start_time: int,
            end_time: int,
            log_time_order: bool,
        ) -> Iterator[tuple[Any, Any, Any]]:
            calls.append((topics, start_time, end_time, log_time_order))
            return iter(expected)

    got = list(iter_messages_log_time_ns(_FakeReader(), "/camera/rgb", 100, 200, log_time_order=False))

    assert got == expected
    assert calls == [("/camera/rgb", 100, 200, False)]


def test_iter_messages_log_time_ns_excludes_end_ns_exclusive_with_real_mcap(tmp_path: Path) -> None:
    """iter_messages_log_time_ns should exclude messages whose log_time equals end_ns_exclusive."""
    path = tmp_path / "exclusive_end.mcap"
    _write_rgb8_mcap(path, "/camera/rgb", [100, 200])

    with path.open("rb") as stream:
        reader = mcap_make_reader(stream)  # type: ignore[no-untyped-call]
        got = [
            int(message.log_time)
            for _schema, _channel, message in iter_messages_log_time_ns(
                reader,
                "/camera/rgb",
                100,
                200,
                log_time_order=True,
            )
        ]

    assert got == [100]


def test_mcap_topic_accessor_start_end_can_load_without_full_timeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """McapTopicAccessor should load bounds without materializing the full timeline."""
    timeline_calls = 0
    start_end_calls = 0

    @contextmanager
    def fake_open_data_source(source: object, mode: str = "rb") -> Iterator[object]:
        del source, mode
        yield object()

    def fake_make_reader(stream: object) -> object:
        del stream
        return object()

    def fake_load_timeline(reader: object, topic: str) -> npt.NDArray[np.int64]:
        nonlocal timeline_calls
        del reader, topic
        timeline_calls += 1
        return np.array([100, 200, 300], dtype=np.int64)

    def fake_load_start_end_ns(reader: object, topic: str) -> tuple[int, int]:
        nonlocal start_end_calls
        del reader, topic
        start_end_calls += 1
        return 100, 300

    monkeypatch.setattr("cosmos_curator.core.sensors.utils.mcap.open_data_source", fake_open_data_source)
    monkeypatch.setattr("cosmos_curator.core.sensors.utils.mcap.mcap_make_reader", fake_make_reader)
    monkeypatch.setattr("cosmos_curator.core.sensors.utils.mcap.load_timeline", fake_load_timeline)
    monkeypatch.setattr("cosmos_curator.core.sensors.utils.mcap.load_start_end_ns", fake_load_start_end_ns)

    accessor = McapTopicAccessor(b"not-used", "/camera/rgb")

    assert accessor.start_ns == 100
    assert accessor.end_ns == 300
    assert start_end_calls == 1
    assert timeline_calls == 0


def test_mcap_topic_accessor_timestamps_cache_populates_start_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """McapTopicAccessor should cache a read-only timeline and reuse it for bounds."""
    timeline_calls = 0
    start_end_calls = 0
    timeline = np.array([100, 200, 300], dtype=np.int64)
    timeline.flags.writeable = False

    @contextmanager
    def fake_open_data_source(source: object, mode: str = "rb") -> Iterator[object]:
        del source, mode
        yield object()

    def fake_make_reader(stream: object) -> object:
        del stream
        return object()

    def fake_load_timeline(reader: object, topic: str) -> npt.NDArray[np.int64]:
        nonlocal timeline_calls
        del reader, topic
        timeline_calls += 1
        return timeline

    def fake_load_start_end_ns(reader: object, topic: str) -> tuple[int, int]:
        nonlocal start_end_calls
        del reader, topic
        start_end_calls += 1
        return 0, 0

    monkeypatch.setattr("cosmos_curator.core.sensors.utils.mcap.open_data_source", fake_open_data_source)
    monkeypatch.setattr("cosmos_curator.core.sensors.utils.mcap.mcap_make_reader", fake_make_reader)
    monkeypatch.setattr("cosmos_curator.core.sensors.utils.mcap.load_timeline", fake_load_timeline)
    monkeypatch.setattr("cosmos_curator.core.sensors.utils.mcap.load_start_end_ns", fake_load_start_end_ns)

    accessor = McapTopicAccessor(b"not-used", "/camera/rgb")

    np.testing.assert_array_equal(accessor.timestamps_ns, timeline)
    np.testing.assert_array_equal(accessor.timestamps_ns, timeline)
    assert not accessor.timestamps_ns.flags.writeable
    assert accessor.start_ns == 100
    assert accessor.end_ns == 300
    assert accessor.max_gap_ns == 0
    assert timeline_calls == 1
    assert start_end_calls == 0


def test_mcap_topic_accessor_raises_on_missing_topic(tmp_path: Path) -> None:
    """McapTopicAccessor should preserve the existing empty-topic errors."""
    path = tmp_path / "missing_accessor_topic.mcap"
    _write_rgb8_mcap(path, "/camera/rgb", [10])
    accessor = McapTopicAccessor(path, "/camera/depth")

    with pytest.raises(ValueError, match="no MCAP messages on topic"):
        _ = accessor.start_ns
    with pytest.raises(ValueError, match="no MCAP messages on topic"):
        _ = accessor.timestamps_ns


def test_mcap_topic_accessor_iter_messages_excludes_end_ns_exclusive_with_real_mcap(tmp_path: Path) -> None:
    """McapTopicAccessor message iteration should preserve half-open windows."""
    path = tmp_path / "accessor_exclusive_end.mcap"
    _write_rgb8_mcap(path, "/camera/rgb", [100, 200])
    accessor = McapTopicAccessor(path, "/camera/rgb")

    with accessor.open_reader() as reader:
        got = [
            int(message.log_time)
            for _schema, _channel, message in accessor.iter_messages(
                reader,
                100,
                200,
                log_time_order=True,
            )
        ]

    assert got == [100]


def test_require_channel_message_encoding_accepts_expected_encoding() -> None:
    """require_channel_message_encoding should accept matching channel encodings."""
    require_channel_message_encoding(_protobuf_channel(), "protobuf")


def test_require_channel_message_encoding_rejects_unexpected_encoding() -> None:
    """require_channel_message_encoding should report mismatched channel encodings clearly."""
    with pytest.raises(ValueError, match="expected protobuf channel"):
        require_channel_message_encoding(_protobuf_channel(message_encoding="json"), "protobuf")


def test_schema_for_channel_returns_schema() -> None:
    """schema_for_channel should return the channel's referenced schema."""
    schema = _protobuf_schema()
    summary = SimpleNamespace(schemas={1: schema})

    assert schema_for_channel(summary, _protobuf_channel(), _PROTOBUF_TOPIC) is schema


def test_schema_for_channel_rejects_missing_schema() -> None:
    """schema_for_channel should reject missing schema references."""
    summary = SimpleNamespace(schemas={})

    with pytest.raises(ValueError, match=r"references missing schema_id 1"):
        schema_for_channel(summary, _protobuf_channel(), _PROTOBUF_TOPIC)


def test_mcap_protobuf_message_resolver_returns_message_class() -> None:
    """McapProtobufMessageResolver should build a dynamic protobuf class."""
    resolver = McapProtobufMessageResolver(_PROTOBUF_SCHEMA_NAME, schema_label="test protobuf")

    message_cls = resolver.message_class_for_message(_protobuf_schema(), _protobuf_channel(), topic=_PROTOBUF_TOPIC)

    assert issubclass(message_cls, Message)


def test_mcap_protobuf_message_resolver_resolves_from_summary() -> None:
    """McapProtobufMessageResolver should resolve schema metadata from an MCAP summary."""
    schema = _protobuf_schema()
    channel = _protobuf_channel()
    summary = SimpleNamespace(channels={1: channel}, schemas={1: schema})
    reader = SimpleNamespace(get_summary=lambda: summary)
    resolver = McapProtobufMessageResolver(_PROTOBUF_SCHEMA_NAME, schema_label="test protobuf")

    message_cls = resolver.resolve_from_summary(reader, _PROTOBUF_TOPIC)

    assert message_cls is not None
    assert issubclass(message_cls, Message)


def test_mcap_protobuf_message_resolver_rejects_bad_channel_encoding() -> None:
    """McapProtobufMessageResolver should require protobuf message encoding."""
    resolver = McapProtobufMessageResolver(_PROTOBUF_SCHEMA_NAME, schema_label="test protobuf")

    with pytest.raises(ValueError, match="expected protobuf channel"):
        resolver.message_class_for_message(
            _protobuf_schema(),
            _protobuf_channel(message_encoding="json"),
            topic=_PROTOBUF_TOPIC,
        )


def test_mcap_protobuf_message_resolver_rejects_missing_summary_schema() -> None:
    """McapProtobufMessageResolver should reject summary channels with missing schema ids."""
    channel = _protobuf_channel()
    summary = SimpleNamespace(channels={1: channel}, schemas={})
    reader = SimpleNamespace(get_summary=lambda: summary)
    resolver = McapProtobufMessageResolver(_PROTOBUF_SCHEMA_NAME, schema_label="test protobuf")

    with pytest.raises(ValueError, match=r"references missing schema_id 1"):
        resolver.resolve_from_summary(reader, _PROTOBUF_TOPIC)


def test_mcap_protobuf_message_resolver_rejects_invalid_descriptor() -> None:
    """McapProtobufMessageResolver should reject unparsable descriptor sets."""
    resolver = McapProtobufMessageResolver(_PROTOBUF_SCHEMA_NAME, schema_label="test protobuf")

    with pytest.raises(ValueError, match="failed to parse test protobuf schema descriptor set"):
        resolver.message_class_for_message(
            _protobuf_schema(data=b"\xff"),
            _protobuf_channel(),
            topic=_PROTOBUF_TOPIC,
        )


def test_mcap_protobuf_message_resolver_rejects_missing_message_descriptor() -> None:
    """McapProtobufMessageResolver should reject descriptors without the expected message."""
    resolver = McapProtobufMessageResolver(_PROTOBUF_SCHEMA_NAME, schema_label="test protobuf")
    descriptor_set = _test_file_descriptor_set(message_name="OtherMessage")

    with pytest.raises(ValueError, match="test protobuf schema descriptor set does not define"):
        resolver.message_class_for_message(
            _protobuf_schema(data=descriptor_set.SerializeToString()),
            _protobuf_channel(),
            topic=_PROTOBUF_TOPIC,
        )


def test_mcap_protobuf_message_resolver_allows_multiple_topics_with_different_schema_ids() -> None:
    """One resolver should be reusable across topics without schema-id cross-talk."""
    resolver = McapProtobufMessageResolver(_PROTOBUF_SCHEMA_NAME, schema_label="test protobuf")

    first_cls = resolver.message_class_for_message(
        _protobuf_schema(schema_id=1),
        _protobuf_channel(schema_id=1, topic="/test/first"),
        topic="/test/first",
    )
    second_cls = resolver.message_class_for_message(
        _protobuf_schema(schema_id=2),
        _protobuf_channel(channel_id=2, schema_id=2, topic="/test/second"),
        topic="/test/second",
    )

    assert issubclass(first_cls, Message)
    assert issubclass(second_cls, Message)


def test_mcap_protobuf_message_resolver_rejects_schema_id_changes() -> None:
    """McapProtobufMessageResolver should validate cached schema ids for each message."""
    resolver = McapProtobufMessageResolver(_PROTOBUF_SCHEMA_NAME, schema_label="test protobuf")

    assert issubclass(
        resolver.message_class_for_message(
            _protobuf_schema(schema_id=1),
            _protobuf_channel(schema_id=1),
            topic=_PROTOBUF_TOPIC,
        ),
        Message,
    )
    with pytest.raises(ValueError, match="changed schema_id from 1 to 2"):
        resolver.message_class_for_message(
            _protobuf_schema(schema_id=2),
            _protobuf_channel(channel_id=2, schema_id=2),
            topic=_PROTOBUF_TOPIC,
        )


def _other_sample_schema() -> sensor_mcap.ProtobufMessageSchema:
    """Return a fake non-camera/non-IMU schema for generic protobuf helper tests."""
    return sensor_mcap.ProtobufMessageSchema(
        full_name="example.sensor.v1.OtherSample",
        proto_file_name="example/sensor/v1/other.proto",
        package="example.sensor.v1",
        message_name="OtherSample",
        fields=(
            sensor_mcap.protobuf_field("timestamp_ns", 1, _FIELD_DESCRIPTOR_PROTO.TYPE_UINT64),
            sensor_mcap.protobuf_field("value", 2, _FIELD_DESCRIPTOR_PROTO.TYPE_DOUBLE),
            sensor_mcap.protobuf_field("valid", 3, _FIELD_DESCRIPTOR_PROTO.TYPE_BOOL),
            sensor_mcap.protobuf_field(
                "tags",
                4,
                _FIELD_DESCRIPTOR_PROTO.TYPE_STRING,
                label=_FIELD_DESCRIPTOR_PROTO.LABEL_REPEATED,
            ),
        ),
    )


def test_protobuf_schema_helpers_build_dynamic_message_class() -> None:
    """Generic schema helpers should build reusable descriptor sets and message classes."""
    schema = _other_sample_schema()

    descriptor_set = sensor_mcap.protobuf_file_descriptor_set(schema)
    message_cls = sensor_mcap.protobuf_message_class(schema)
    message = message_cls()
    message.timestamp_ns = 10
    message.value = 1.5
    message.valid = True

    assert descriptor_set.file[0].name == schema.proto_file_name
    assert descriptor_set.file[0].package == schema.package
    tags_field = descriptor_set.file[0].message_type[0].field[3]
    assert tags_field.name == "tags"
    assert tags_field.label == _FIELD_DESCRIPTOR_PROTO.LABEL_REPEATED
    assert message.SerializeToString()


def test_iter_decoded_protobuf_messages_decodes_fake_sensor_schema(tmp_path: Path) -> None:
    """The core protobuf iterator should work for a future sensor schema."""
    schema = _other_sample_schema()
    message_cls = sensor_mcap.protobuf_message_class(schema)
    first = message_cls()
    first.timestamp_ns = 10
    first.value = 1.5
    first.valid = True
    second = message_cls()
    second.timestamp_ns = 20
    second.value = 2.5
    second.valid = False
    path = tmp_path / "other.mcap"

    with path.open("wb") as stream:
        writer = Writer(stream, compression=CompressionType.ZSTD)
        writer.start(library="cosmos_curator test")
        schema_id = writer.register_schema(
            name=schema.full_name,
            encoding=sensor_mcap.PROTOBUF_ENCODING,
            data=sensor_mcap.protobuf_file_descriptor_set(schema).SerializeToString(),
        )
        channel_id = writer.register_channel(
            schema_id=schema_id,
            topic="/other",
            message_encoding=sensor_mcap.PROTOBUF_ENCODING,
        )
        writer.add_message(
            channel_id=channel_id,
            log_time=10,
            publish_time=11,
            sequence=1,
            data=first.SerializeToString(),
        )
        writer.add_message(
            channel_id=channel_id,
            log_time=20,
            publish_time=21,
            sequence=2,
            data=second.SerializeToString(),
        )
        writer.finish()  # type: ignore[no-untyped-call]

    with path.open("rb") as stream:
        reader = mcap_make_reader(stream)  # type: ignore[no-untyped-call]
        decoded = list(
            sensor_mcap.iter_decoded_protobuf_messages(
                reader,
                "/other",
                expected_schema_name=schema.full_name,
                schema_label="test protobuf",
                log_time_order=False,
            )
        )

    assert [message.log_time_ns for message in decoded] == [10, 20]
    assert [message.publish_time_ns for message in decoded] == [11, 21]
    assert [message.sequence for message in decoded] == [1, 2]
    assert [message.message.value for message in decoded] == [1.5, 2.5]
    assert [message.message.valid for message in decoded] == [True, False]


def test_iter_decoded_protobuf_messages_validates_each_indexed_message_schema() -> None:
    """Indexed summaries must not bypass per-message schema ID validation."""
    summary_schema = _protobuf_schema(schema_id=1)
    summary_channel = _protobuf_channel(schema_id=1)
    conflicting_schema = _protobuf_schema(schema_id=2)
    conflicting_channel = _protobuf_channel(channel_id=2, schema_id=2)
    reader = SimpleNamespace(
        get_summary=lambda: SimpleNamespace(channels={1: summary_channel}, schemas={1: summary_schema}),
        iter_messages=lambda **_kwargs: iter(
            [
                (
                    conflicting_schema,
                    conflicting_channel,
                    SimpleNamespace(data=b"", log_time=1, publish_time=1, sequence=1),
                )
            ]
        ),
    )

    with pytest.raises(ValueError, match="changed schema_id from 1 to 2"):
        list(
            sensor_mcap.iter_decoded_protobuf_messages(
                reader,
                _PROTOBUF_TOPIC,
                expected_schema_name=_PROTOBUF_SCHEMA_NAME,
                schema_label="test protobuf",
            )
        )


def test_iter_decoded_protobuf_messages_rejects_wrong_schema_name(tmp_path: Path) -> None:
    """Schema validation should happen in the generic decoded-message iterator."""
    schema = _other_sample_schema()
    message_cls = sensor_mcap.protobuf_message_class(schema)
    message = message_cls()
    message.timestamp_ns = 10
    path = tmp_path / "wrong_schema.mcap"

    with path.open("wb") as stream:
        writer = Writer(stream, compression=CompressionType.ZSTD)
        writer.start(library="cosmos_curator test")
        schema_id = writer.register_schema(
            name=schema.full_name,
            encoding=sensor_mcap.PROTOBUF_ENCODING,
            data=sensor_mcap.protobuf_file_descriptor_set(schema).SerializeToString(),
        )
        channel_id = writer.register_channel(
            schema_id=schema_id,
            topic="/other",
            message_encoding=sensor_mcap.PROTOBUF_ENCODING,
        )
        writer.add_message(channel_id=channel_id, log_time=10, publish_time=10, data=message.SerializeToString())
        writer.finish()  # type: ignore[no-untyped-call]

    with path.open("rb") as stream:
        reader = mcap_make_reader(stream)  # type: ignore[no-untyped-call]
        with pytest.raises(ValueError, match="expected MCAP schema"):
            list(
                sensor_mcap.iter_decoded_protobuf_messages(
                    reader,
                    "/other",
                    expected_schema_name="example.sensor.v1.DifferentSample",
                    schema_label="test protobuf",
                    log_time_order=False,
                )
            )


def test_write_mcap_messages_preserves_schema_channel_and_message_metadata(tmp_path: Path) -> None:
    """The shared writer should not assume protobuf payloads or sensor-specific topics."""
    output_path = tmp_path / "generic.mcap"
    schema = sensor_mcap.McapSchemaSpec(
        name="example.raw.Sample",
        encoding="example-schema",
        data=b"schema-bytes",
        message_encoding="example-bytes",
    )
    records = [
        sensor_mcap.McapMessageRecord(data=b"first", log_time_ns=20, publish_time_ns=10, sequence=1),
        sensor_mcap.McapMessageRecord(data=b"second", log_time_ns=40, publish_time_ns=30, sequence=2),
    ]

    stats = sensor_mcap.write_mcap_messages(
        output_path,
        stream=sensor_mcap.McapStreamSpec(topic="/example", schema=schema, library="test writer"),
        records=records,
    )

    assert stats.message_count == 2
    assert stats.start_log_time_ns == 20
    assert stats.end_log_time_ns == 40
    assert stats.start_publish_time_ns == 10
    assert stats.end_publish_time_ns == 30

    with output_path.open("rb") as stream:
        reader = mcap_make_reader(stream)  # type: ignore[no-untyped-call]
        decoded = list(reader.iter_messages(log_time_order=False))

    assert len(decoded) == 2
    assert decoded[0][0].name == schema.name
    assert decoded[0][0].encoding == schema.encoding
    assert decoded[0][0].data == schema.data
    assert decoded[0][1].topic == "/example"
    assert decoded[0][1].message_encoding == schema.message_encoding
    assert [message.data for _schema, _channel, message in decoded] == [b"first", b"second"]
    assert [message.sequence for _schema, _channel, message in decoded] == [1, 2]


def test_write_mcap_messages_writes_optional_metadata_records(tmp_path: Path) -> None:
    """Optional stream metadata should land as named MCAP metadata records."""
    output_path = tmp_path / "with_metadata.mcap"
    schema = sensor_mcap.McapSchemaSpec(
        name="example.generic.Sample",
        encoding="example-schema",
        data=b"schema-bytes",
        message_encoding="example-bytes",
    )

    sensor_mcap.write_mcap_messages(
        output_path,
        stream=sensor_mcap.McapStreamSpec(
            topic="/example",
            schema=schema,
            library="test writer",
            metadata={"example.meta.v1": {"version": "1.0.0", "kind": "test"}},
        ),
        records=[sensor_mcap.McapMessageRecord(data=b"payload", log_time_ns=1, publish_time_ns=1, sequence=1)],
    )

    with output_path.open("rb") as stream:
        reader = mcap_make_reader(stream)  # type: ignore[no-untyped-call]
        assert sensor_mcap.get_metadata_record(reader, "example.meta.v1") == {
            "version": "1.0.0",
            "kind": "test",
        }


def test_write_mcap_messages_rejects_existing_output_without_overwrite(tmp_path: Path) -> None:
    """Existing output files should be protected unless overwrite is requested."""
    output_path = tmp_path / "exists.mcap"
    output_path.write_bytes(b"already here")

    with pytest.raises(FileExistsError, match="Output path already exists"):
        sensor_mcap.write_mcap_messages(
            output_path,
            stream=sensor_mcap.McapStreamSpec(
                topic="/example",
                schema=sensor_mcap.McapSchemaSpec("example.Schema", "schema", b"{}", "bytes"),
                library="test writer",
            ),
            records=[],
        )

    assert output_path.read_bytes() == b"already here"


def test_write_mcap_messages_removes_temporary_file_on_record_failure(tmp_path: Path) -> None:
    """A failing record stream should not leave partial output or temp files behind."""
    output_path = tmp_path / "broken.mcap"

    def records() -> Iterator[sensor_mcap.McapMessageRecord]:
        yield sensor_mcap.McapMessageRecord(data=b"first", log_time_ns=1, publish_time_ns=1, sequence=1)
        msg = "synthetic record failure"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="synthetic record failure"):
        sensor_mcap.write_mcap_messages(
            output_path,
            stream=sensor_mcap.McapStreamSpec(
                topic="/example",
                schema=sensor_mcap.McapSchemaSpec("example.Schema", "schema", b"{}", "bytes"),
                library="test writer",
            ),
            records=records(),
        )

    assert not output_path.exists()
    assert not list(tmp_path.glob(".broken.mcap.*.tmp"))


def test_mcap_stream_spec_rejects_missing_schema_and_message_encoding() -> None:
    """Construction requires at least one of schema or message_encoding."""
    with pytest.raises(ValueError, match="requires either schema or message_encoding"):
        sensor_mcap.McapStreamSpec(topic="/x", library="test")


def test_mcap_stream_spec_rejects_both_schema_and_message_encoding() -> None:
    """Setting both schema and message_encoding is ambiguous and rejected."""
    schema = sensor_mcap.McapSchemaSpec("n", "protobuf", b"", "protobuf")
    with pytest.raises(ValueError, match="cannot set both schema and message_encoding"):
        sensor_mcap.McapStreamSpec(
            topic="/x",
            library="test",
            schema=schema,
            message_encoding="other",
        )


def test_write_schemaless_mcap_uses_schema_id_zero_and_channel_metadata(tmp_path: Path) -> None:
    """A schemaless stream should register no schema and expose the caller's channel metadata."""
    output_path = tmp_path / "schemaless.mcap"
    records = [
        sensor_mcap.McapMessageRecord(data=b"packet0", log_time_ns=10, publish_time_ns=10, sequence=1),
        sensor_mcap.McapMessageRecord(data=b"packet1", log_time_ns=20, publish_time_ns=20, sequence=2),
    ]

    stats = sensor_mcap.write_mcap_messages(
        output_path,
        stream=sensor_mcap.McapStreamSpec(
            topic="/lidar/top",
            library="test schemaless writer",
            message_encoding="vendor.sensor.udp",
            channel_metadata={
                "sensor_model": "GENERIC_SENSOR_V1",
                "return_mode": "last_and_strongest",
            },
        ),
        records=records,
    )

    assert stats.message_count == 2
    assert stats.start_log_time_ns == 10
    assert stats.end_log_time_ns == 20

    with output_path.open("rb") as stream:
        reader = mcap_make_reader(stream)  # type: ignore[no-untyped-call]
        summary = reader.get_summary()

    assert summary is not None
    # No schemas registered for the schemaless channel.
    assert not summary.schemas, f"expected no schemas, got {list(summary.schemas.values())}"

    channels = list(summary.channels.values())
    assert len(channels) == 1
    channel = channels[0]
    assert channel.topic == "/lidar/top"
    assert channel.schema_id == 0
    assert channel.message_encoding == "vendor.sensor.udp"
    assert dict(channel.metadata) == {
        "sensor_model": "GENERIC_SENSOR_V1",
        "return_mode": "last_and_strongest",
    }

    with output_path.open("rb") as stream:
        reader = mcap_make_reader(stream)  # type: ignore[no-untyped-call]
        payloads = [record.data for _schema, _channel, record in reader.iter_messages(log_time_order=False)]

    assert payloads == [b"packet0", b"packet1"]


def test_write_schemaless_mcap_defaults_channel_metadata_to_empty(tmp_path: Path) -> None:
    """A schemaless stream without channel_metadata should still produce a valid empty-metadata channel."""
    output_path = tmp_path / "schemaless_no_meta.mcap"
    records = [sensor_mcap.McapMessageRecord(data=b"payload", log_time_ns=1, publish_time_ns=1, sequence=1)]

    sensor_mcap.write_mcap_messages(
        output_path,
        stream=sensor_mcap.McapStreamSpec(
            topic="/schemaless",
            library="test schemaless writer",
            message_encoding="application/octet-stream",
        ),
        records=records,
    )

    with output_path.open("rb") as stream:
        reader = mcap_make_reader(stream)  # type: ignore[no-untyped-call]
        summary = reader.get_summary()

    assert summary is not None
    channels = list(summary.channels.values())
    assert len(channels) == 1
    assert channels[0].schema_id == 0
    assert dict(channels[0].metadata) == {}
