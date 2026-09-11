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
"""MCAP utilities for the sensor library."""

import uuid
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.message import DecodeError
from google.protobuf.message import Message as ProtobufMessage
from mcap.reader import McapReader
from mcap.reader import make_reader as mcap_make_reader
from mcap.records import Channel, Schema
from mcap.records import Message as McapMessage
from mcap.summary import Summary
from mcap.writer import CompressionType, Writer

from cosmos_curator.core.sensors.types.types import DataSource
from cosmos_curator.core.sensors.utils.io import open_data_source

VIDEO_METADATA_RECORD_NAME = "cosmos_curator.video_metadata.v1"
PROTOBUF_ENCODING = "protobuf"
NS_PER_USEC = 1_000


class EmptyMcapTopicError(ValueError):
    """Raised when an otherwise readable MCAP contains no messages for a topic."""


def parse_protobuf_message(
    message_cls: type[ProtobufMessage],
    data: bytes,
    *,
    topic: str,
    sensor_label: str,
) -> ProtobufMessage:
    """Parse one protobuf payload and add sensor/topic context to decode errors."""
    message = message_cls()
    try:
        message.ParseFromString(data)
    except DecodeError as e:
        msg = f"failed to parse {sensor_label} protobuf message on topic {topic!r}"
        raise ValueError(msg) from e
    return message


def channel_for_topic(summary: Summary, topic: str) -> Channel | None:
    """Return exactly one channel in *summary* whose topic equals *topic*.

    Returns ``None`` when the topic is absent.

    Raises:
        ValueError: If multiple channels share the same topic.

    """
    matches = [ch for ch in summary.channels.values() if ch.topic == topic]
    if not matches:
        return None
    if len(matches) != 1:
        msg = f"expected exactly one MCAP channel for topic {topic!r}, found {len(matches)}"
        raise ValueError(msg)
    return matches[0]


def get_metadata_record(reader: McapReader, name: str) -> dict[str, str]:
    """Return exactly one metadata record by *name*.

    Raises:
        ValueError: if the named metadata record is missing or duplicated.

    """
    matches = [record.metadata for record in reader.iter_metadata() if record.name == name]
    if not matches:
        msg = f"required MCAP metadata record {name!r} not found"
        raise ValueError(msg)
    if len(matches) != 1:
        msg = f"expected exactly one MCAP metadata record {name!r}, found {len(matches)}"
        raise ValueError(msg)
    return matches[0]


def require_channel_message_encoding(channel: Channel, expected_encoding: str) -> None:
    """Raise if *channel* does not use the expected message encoding."""
    if channel.message_encoding != expected_encoding:
        msg = f"expected {expected_encoding} channel, got message_encoding={channel.message_encoding!r}"
        raise ValueError(msg)


def schema_for_channel(summary: Summary, channel: Channel, topic: str) -> Schema:
    """Return the schema referenced by *channel* in *summary*.

    Raises:
        ValueError: if the channel references a missing schema id.

    """
    schema = summary.schemas.get(channel.schema_id)
    if schema is None:
        msg = f"MCAP channel for topic {topic!r} references missing schema_id {channel.schema_id}"
        raise ValueError(msg)
    return schema


def load_start_end_ns(reader: McapReader, topic: str) -> tuple[int, int]:
    """Load the first and last message log times for one topic.

    Indexed seekable readers use direct forward and reverse lookups. Readers
    without chunk indexes are scanned once because the MCAP non-seeking
    fallback does not honor reverse iteration.
    """
    summary = reader.get_summary()
    if summary is not None and summary.chunk_indexes:
        earliest = reader.iter_messages(topics=topic, log_time_order=True, reverse=False)
        try:
            _schema, _channel, first_msg = next(earliest)
        except StopIteration as e:
            msg = f"no MCAP messages on topic {topic!r}"
            raise EmptyMcapTopicError(msg) from e

        latest = reader.iter_messages(topics=topic, log_time_order=True, reverse=True)
        try:
            _schema, _channel, last_msg = next(latest)
        except StopIteration as e:
            msg = f"failed to read latest MCAP message on topic {topic!r} after reading earliest message"
            raise ValueError(msg) from e
        return int(first_msg.log_time), int(last_msg.log_time)

    start_ns: int | None = None
    end_ns: int | None = None
    for _schema, _channel, message in reader.iter_messages(topics=topic, log_time_order=False):
        log_time_ns = int(message.log_time)
        start_ns = log_time_ns if start_ns is None else min(start_ns, log_time_ns)
        end_ns = log_time_ns if end_ns is None else max(end_ns, log_time_ns)
    if start_ns is None or end_ns is None:
        msg = f"no MCAP messages on topic {topic!r}"
        raise EmptyMcapTopicError(msg)
    return start_ns, end_ns


def load_timeline(reader: McapReader, topic: str) -> npt.NDArray[np.int64]:
    """Load the full ordered ``log_time`` timeline for *topic*.

    Args:
        reader: MCAP reader positioned on the source file/stream.
        topic: Topic name to query.

    Returns:
        Read-only ``int64`` array of message ``log_time`` values in ascending
        order.

    Raises:
        ValueError: If the topic has no messages.

    """
    times = [
        int(message.log_time)
        for _schema, _channel, message in reader.iter_messages(
            topics=topic,
            log_time_order=True,
        )
    ]
    if not times:
        msg = f"no MCAP messages on topic {topic!r}"
        raise EmptyMcapTopicError(msg)
    arr = np.array(times, dtype=np.int64)
    arr.flags.writeable = False
    return arr


class McapTopicAccessor:
    """Reusable source/topic mechanics for MCAP-backed sensors."""

    def __init__(self, source: DataSource, topic: str) -> None:
        """Initialize an accessor for one MCAP source and topic."""
        self.source = source
        self.topic = topic
        self._message_log_times_ns_cache: npt.NDArray[np.int64] | None = None
        self._start_ns: int | None = None
        self._end_ns: int | None = None

    @contextmanager
    def open_reader(self) -> Iterator[McapReader]:
        """Open the source and yield an MCAP reader."""
        with open_data_source(self.source, mode="rb") as stream:
            yield mcap_make_reader(stream)  # type: ignore[no-untyped-call]

    @property
    def start_ns(self) -> int:
        """Earliest message time on this topic, in nanoseconds."""
        self._ensure_start_end_ns_cached()
        if self._start_ns is None:
            msg = "start_ns was not loaded"
            raise ValueError(msg)
        return self._start_ns

    @property
    def end_ns(self) -> int:
        """Latest message time on this topic, in nanoseconds."""
        self._ensure_start_end_ns_cached()
        if self._end_ns is None:
            msg = "end_ns was not loaded"
            raise ValueError(msg)
        return self._end_ns

    @property
    def max_gap_ns(self) -> int:
        """Return maximum expected gap duration in nanoseconds."""
        return 0

    @property
    def timestamps_ns(self) -> npt.NDArray[np.int64]:
        """Message times in nanoseconds from raw MCAP ``log_time`` values."""
        return self._ensure_timeline_cached()

    def _ensure_start_end_ns_cached(self) -> None:
        """Cache only the first and last message timestamps for the topic."""
        if self._start_ns is not None and self._end_ns is not None:
            return

        full = self._message_log_times_ns_cache
        if full is not None:
            self._cache_start_end_from_timeline(full)
            return

        with self.open_reader() as reader:
            self._start_ns, self._end_ns = load_start_end_ns(reader, self.topic)

    def _cache_start_end_from_timeline(self, timeline_ns: npt.NDArray[np.int64]) -> None:
        """Cache start and end timestamps from a non-empty topic timeline."""
        if len(timeline_ns) == 0:
            msg = f"no MCAP messages on topic {self.topic!r}"
            raise EmptyMcapTopicError(msg)
        self._start_ns = int(timeline_ns[0])
        self._end_ns = int(timeline_ns[-1])

    def _ensure_timeline_cached(self) -> npt.NDArray[np.int64]:
        """Load and cache the full ordered timeline for the topic."""
        if self._message_log_times_ns_cache is not None:
            return self._message_log_times_ns_cache

        with self.open_reader() as reader:
            arr = load_timeline(reader, self.topic)
        self._cache_start_end_from_timeline(arr)
        self._message_log_times_ns_cache = arr
        return arr

    def iter_messages(
        self,
        reader: McapReader,
        start_ns: int,
        end_ns_exclusive: int,
        *,
        log_time_order: bool = True,
    ) -> Iterator[tuple[Schema | None, Channel, McapMessage]]:
        """Yield topic messages whose ``log_time`` falls in the half-open interval."""
        yield from iter_messages_log_time_ns(
            reader,
            self.topic,
            start_ns,
            end_ns_exclusive,
            log_time_order=log_time_order,
        )


def iter_messages_log_time_ns(
    reader: McapReader,
    topic: str,
    start_ns: int,
    end_ns_exclusive: int,
    *,
    log_time_order: bool = True,
) -> Iterator[tuple[Schema | None, Channel, McapMessage]]:
    """Yield messages on *topic* with ``start_ns <= log_time < end_ns_exclusive``.

    For seekable streams with MCAP summary chunk indexes, the underlying
    ``mcap`` ``SeekingReader`` first selects chunk records whose time span
    overlaps the requested interval, then filters each ``Message`` by
    ``log_time``. This is not a full linear read of the file.

    For non-seekable streams, or MCAP files without chunk indexes, behavior
    falls back to the library's non-indexed path (possibly reading or
    buffering the whole stream).

    Args:
        reader: MCAP reader (from :func:`make_reader` or ``mcap.reader.make_reader``).
        topic: Topic name (single channel).
        start_ns: Inclusive lower bound on ``Message.log_time`` (nanoseconds).
        end_ns_exclusive: Exclusive upper bound on ``Message.log_time`` (nanoseconds).
        log_time_order: If True, yield in ascending ``log_time`` order.

    Yields:
        ``(schema, channel, message)`` tuples matching ``iter_messages``.

    """
    yield from reader.iter_messages(
        topics=topic,
        start_time=start_ns,
        end_time=end_ns_exclusive,
        log_time_order=log_time_order,
    )


@dataclass(frozen=True)
class ProtobufFieldSpec:
    """One field in a dynamically generated protobuf message."""

    name: str
    number: int
    field_type: int


@dataclass(frozen=True)
class ProtobufMessageSchema:
    """Metadata needed to build and validate a dynamic protobuf message."""

    full_name: str
    proto_file_name: str
    package: str
    message_name: str
    fields: Sequence[ProtobufFieldSpec]


@dataclass(frozen=True)
class DecodedProtobufMessage:
    """One decoded protobuf MCAP message plus MCAP timing metadata."""

    message: ProtobufMessage
    log_time_ns: int
    publish_time_ns: int
    sequence: int


@dataclass(frozen=True)
class McapSchemaSpec:
    """Schema and channel encoding metadata for an MCAP stream."""

    name: str
    encoding: str
    data: bytes
    message_encoding: str


@dataclass(frozen=True)
class McapStreamSpec:
    """One MCAP output stream with a single topic and schema."""

    topic: str
    schema: McapSchemaSpec
    library: str
    overwrite: bool = False


@dataclass(frozen=True)
class McapMessageRecord:
    """One serialized MCAP message ready to write."""

    data: bytes
    log_time_ns: int
    publish_time_ns: int
    sequence: int


@dataclass(frozen=True)
class McapWriteStats:
    """Summary of messages written to one MCAP file."""

    message_count: int
    start_log_time_ns: int | None
    end_log_time_ns: int | None
    start_publish_time_ns: int | None
    end_publish_time_ns: int | None


def protobuf_field(name: str, number: int, field_type: int) -> ProtobufFieldSpec:
    """Create one dynamic protobuf field spec."""
    return ProtobufFieldSpec(name=name, number=number, field_type=field_type)


def protobuf_file_descriptor_proto(schema: ProtobufMessageSchema) -> descriptor_pb2.FileDescriptorProto:
    """Build the protobuf file descriptor for a dynamic sensor schema."""
    file_descriptor = descriptor_pb2.FileDescriptorProto()
    file_descriptor.name = schema.proto_file_name
    file_descriptor.package = schema.package
    file_descriptor.syntax = "proto3"

    message_descriptor = file_descriptor.message_type.add()
    message_descriptor.name = schema.message_name

    for field_spec in schema.fields:
        field = message_descriptor.field.add()
        field.name = field_spec.name
        field.number = field_spec.number
        field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        field.type = field_spec.field_type

    return file_descriptor


def protobuf_file_descriptor_set(schema: ProtobufMessageSchema) -> descriptor_pb2.FileDescriptorSet:
    """Return the descriptor set embedded in a protobuf MCAP schema record."""
    descriptor_set = descriptor_pb2.FileDescriptorSet()
    descriptor_set.file.append(protobuf_file_descriptor_proto(schema))
    return descriptor_set


def protobuf_message_class(schema: ProtobufMessageSchema) -> type[ProtobufMessage]:
    """Create a dynamic protobuf class for serializing one sensor schema."""
    pool = descriptor_pool.DescriptorPool()
    pool.Add(protobuf_file_descriptor_proto(schema))
    message_descriptor = pool.FindMessageTypeByName(schema.full_name)
    return message_factory.GetMessageClass(message_descriptor)


def protobuf_mcap_schema_spec(schema: ProtobufMessageSchema) -> McapSchemaSpec:
    """Build writer schema metadata for a protobuf message schema."""
    return McapSchemaSpec(
        name=schema.full_name,
        encoding=PROTOBUF_ENCODING,
        data=protobuf_file_descriptor_set(schema).SerializeToString(),
        message_encoding=PROTOBUF_ENCODING,
    )


def timestamp_usec_to_ns(timestamp_usec: int, name: str) -> int:
    """Convert a non-negative microsecond timestamp to nanoseconds."""
    timestamp_ns = timestamp_usec * NS_PER_USEC
    if timestamp_ns < 0:
        msg = f"{name} must be non-negative, got {timestamp_usec}"
        raise ValueError(msg)
    return timestamp_ns


def format_optional_timestamp_ns(value: int | None) -> str:
    """Format an optional nanosecond timestamp for CLI output."""
    return str(value) if value is not None else "N/A"


def write_mcap_messages(
    dest: str | Path,
    *,
    stream: McapStreamSpec,
    records: Iterable[McapMessageRecord],
) -> McapWriteStats:
    """Write serialized records to an MCAP file using one schema and one channel."""
    dest_path = Path(dest)
    if dest_path.exists() and not stream.overwrite:
        msg = f"Output path already exists: {dest_path}"
        raise FileExistsError(msg)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.parent / f".{dest_path.name}.{uuid.uuid4().hex}.tmp"
    writer: Writer | None = None
    stats = _MutableWriteStats()

    try:
        with temp_path.open("xb") as out_file:
            writer = Writer(out_file, compression=CompressionType.ZSTD)
            writer.start(library=stream.library)
            schema_id = writer.register_schema(
                name=stream.schema.name,
                encoding=stream.schema.encoding,
                data=stream.schema.data,
            )
            channel_id = writer.register_channel(
                schema_id=schema_id,
                topic=stream.topic,
                message_encoding=stream.schema.message_encoding,
            )

            for record in records:
                stats.add(record)
                writer.add_message(
                    channel_id=channel_id,
                    log_time=record.log_time_ns,
                    data=record.data,
                    publish_time=record.publish_time_ns,
                    sequence=record.sequence,
                )

            try:
                writer.finish()  # type: ignore[no-untyped-call]
            finally:
                writer = None

        temp_path.replace(dest_path)
    except Exception:
        if writer is not None:
            with suppress(Exception):
                writer.finish()  # type: ignore[no-untyped-call]
        temp_path.unlink(missing_ok=True)
        raise

    return stats.freeze()


def iter_decoded_protobuf_messages(
    reader: McapReader,
    topic: str,
    *,
    expected_schema_name: str,
    schema_label: str = "MCAP protobuf",
    log_time_order: bool = True,
) -> Iterator[DecodedProtobufMessage]:
    """Yield decoded protobuf messages for one MCAP topic."""
    resolver = McapProtobufMessageResolver(expected_schema_name, schema_label=schema_label)
    resolver.resolve_from_summary(reader, topic)
    yielded = False

    for schema, channel, message in reader.iter_messages(topics=topic, log_time_order=log_time_order):
        message_cls = resolver.message_class_for_message(schema, channel, topic=topic)

        sample = message_cls()
        try:
            sample.ParseFromString(message.data)
        except DecodeError as e:
            msg = f"failed to parse {schema_label} message on topic {topic!r}"
            raise ValueError(msg) from e

        yielded = True
        yield DecodedProtobufMessage(
            message=sample,
            log_time_ns=int(message.log_time),
            publish_time_ns=int(message.publish_time),
            sequence=int(message.sequence),
        )

    if not yielded:
        msg = f"no MCAP messages on topic {topic!r}"
        raise ValueError(msg)


@dataclass
class _MutableWriteStats:
    """Mutable accumulator used while streaming records to the writer."""

    message_count: int = 0
    start_log_time_ns: int | None = None
    end_log_time_ns: int | None = None
    start_publish_time_ns: int | None = None
    end_publish_time_ns: int | None = None

    def add(self, record: McapMessageRecord) -> None:
        """Track one written record."""
        self.message_count += 1
        self.start_log_time_ns = _min_optional(self.start_log_time_ns, record.log_time_ns)
        self.end_log_time_ns = _max_optional(self.end_log_time_ns, record.log_time_ns)
        self.start_publish_time_ns = _min_optional(self.start_publish_time_ns, record.publish_time_ns)
        self.end_publish_time_ns = _max_optional(self.end_publish_time_ns, record.publish_time_ns)

    def freeze(self) -> McapWriteStats:
        """Return an immutable snapshot."""
        return McapWriteStats(
            message_count=self.message_count,
            start_log_time_ns=self.start_log_time_ns,
            end_log_time_ns=self.end_log_time_ns,
            start_publish_time_ns=self.start_publish_time_ns,
            end_publish_time_ns=self.end_publish_time_ns,
        )


def _min_optional(current: int | None, value: int) -> int:
    """Return the minimum while allowing an empty accumulator."""
    return value if current is None else min(current, value)


def _max_optional(current: int | None, value: int) -> int:
    """Return the maximum while allowing an empty accumulator."""
    return value if current is None else max(current, value)


class McapProtobufMessageResolver:
    """Resolve and cache dynamic protobuf message classes from MCAP schemas."""

    def __init__(self, expected_schema_name: str, *, schema_label: str = "MCAP protobuf") -> None:
        """Initialize a resolver for one expected protobuf schema name."""
        self.expected_schema_name = expected_schema_name
        self.schema_label = schema_label
        self._message_cls_by_topic: dict[str, type[ProtobufMessage]] = {}
        self._schema_id_by_topic: dict[str, int] = {}

    def resolve_from_summary(self, reader: McapReader, topic: str) -> type[ProtobufMessage] | None:
        """Resolve the message class from the MCAP summary when one is available."""
        if topic in self._message_cls_by_topic:
            return self._message_cls_by_topic[topic]

        summary = reader.get_summary()
        if summary is None:
            return None

        channel = channel_for_topic(summary, topic)
        if channel is None:
            msg = f"no MCAP channel found for topic {topic!r}"
            raise ValueError(msg)
        require_channel_message_encoding(channel, PROTOBUF_ENCODING)

        schema = schema_for_channel(summary, channel, topic)
        return self._message_class_for_topic(schema, int(channel.schema_id), topic)

    def message_class_for_message(
        self,
        schema: Schema | None,
        channel: Channel,
        *,
        topic: str,
    ) -> type[ProtobufMessage]:
        """Resolve the dynamic message class and validate each message channel."""
        require_channel_message_encoding(channel, PROTOBUF_ENCODING)
        return self._message_class_for_topic(schema, int(channel.schema_id), topic)

    def _message_class_for_topic(
        self,
        schema: Schema | None,
        schema_id: int,
        topic: str,
    ) -> type[ProtobufMessage]:
        """Resolve one topic's message class while allowing resolver reuse across topics."""
        cached_schema_id = self._schema_id_by_topic.get(topic)
        if cached_schema_id is not None and schema_id != cached_schema_id:
            msg = f"MCAP topic {topic!r} changed schema_id from {cached_schema_id} to {schema_id}"
            raise ValueError(msg)
        if topic in self._message_cls_by_topic:
            return self._message_cls_by_topic[topic]

        message_cls = self._message_class_from_schema(schema)
        self._message_cls_by_topic[topic] = message_cls
        self._schema_id_by_topic[topic] = schema_id
        return message_cls

    def _message_class_from_schema(self, schema: Schema | None) -> type[ProtobufMessage]:
        """Build a dynamic protobuf message class from an MCAP schema record."""
        if schema is None:
            article = "an" if self.schema_label[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
            msg = f"MCAP message is missing {article} {self.schema_label} schema"
            raise ValueError(msg)
        if schema.name != self.expected_schema_name:
            msg = f"expected MCAP schema {self.expected_schema_name!r}, got {schema.name!r}"
            raise ValueError(msg)
        if schema.encoding != PROTOBUF_ENCODING:
            msg = f"expected protobuf schema encoding, got {schema.encoding!r}"
            raise ValueError(msg)

        file_descriptor_set = descriptor_pb2.FileDescriptorSet()
        try:
            file_descriptor_set.ParseFromString(schema.data)
        except DecodeError as e:
            msg = f"failed to parse {self.schema_label} schema descriptor set"
            raise ValueError(msg) from e

        pool = descriptor_pool.DescriptorPool()
        try:
            for file_descriptor in file_descriptor_set.file:
                pool.Add(file_descriptor)
            message_descriptor = pool.FindMessageTypeByName(self.expected_schema_name)
        except (KeyError, TypeError, ValueError) as e:
            msg = f"{self.schema_label} schema descriptor set does not define {self.expected_schema_name!r}"
            raise ValueError(msg) from e
        return message_factory.GetMessageClass(message_descriptor)
