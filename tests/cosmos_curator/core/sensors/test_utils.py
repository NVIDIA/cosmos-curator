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
"""Shared helpers for sensor tests."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.message import Message
from mcap.writer import CompressionType, Writer

from cosmos_curator.core.sensors.sampling.grid import SamplingGrid
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec

# Time origins for sampling tests. These pin origin invariance: at wall-clock epoch magnitudes
# adjacent float64 values are ~238 ns apart, so any code that routes an absolute nanosecond
# timestamp through a float64 loses detail here that survives near zero.
ZERO_ORIGIN_NS = 0
EPOCH_ROUND_NS = 1_700_000_000_000_000_000
EPOCH_ODD_NS = 1_700_000_000_123_456_789

_PROTO_SCALAR_TYPES = {
    "bool": descriptor_pb2.FieldDescriptorProto.TYPE_BOOL,
    "double": descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE,
    "fixed32": descriptor_pb2.FieldDescriptorProto.TYPE_FIXED32,
    "fixed64": descriptor_pb2.FieldDescriptorProto.TYPE_FIXED64,
    "float": descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT,
    "int32": descriptor_pb2.FieldDescriptorProto.TYPE_INT32,
    "int64": descriptor_pb2.FieldDescriptorProto.TYPE_INT64,
    "sfixed32": descriptor_pb2.FieldDescriptorProto.TYPE_SFIXED32,
    "sfixed64": descriptor_pb2.FieldDescriptorProto.TYPE_SFIXED64,
    "sint32": descriptor_pb2.FieldDescriptorProto.TYPE_SINT32,
    "sint64": descriptor_pb2.FieldDescriptorProto.TYPE_SINT64,
    "string": descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
    "uint32": descriptor_pb2.FieldDescriptorProto.TYPE_UINT32,
    "uint64": descriptor_pb2.FieldDescriptorProto.TYPE_UINT64,
}


@dataclass(frozen=True)
class McapSample:
    """Serialized protobuf payload and its MCAP log time."""

    log_time_ns: int
    data: bytes


def protobuf_descriptor_set_from_proto(proto_path: Path) -> descriptor_pb2.FileDescriptorSet:
    """Build a descriptor set from one flat proto3 message in a checked-in schema."""
    proto_text = proto_path.read_text(encoding="utf-8")
    package_match = re.search(r"^package\s+([A-Za-z_][\w.]*)\s*;$", proto_text, flags=re.MULTILINE)
    message_match = re.search(
        r"^message\s+([A-Za-z_]\w*)\s*\{\s*(.*?)^\}",
        proto_text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if 'syntax = "proto3";' not in proto_text or package_match is None or message_match is None:
        msg = f"{proto_path} must contain one flat proto3 package and message"
        raise ValueError(msg)

    descriptor_set = descriptor_pb2.FileDescriptorSet()
    file_descriptor = descriptor_set.file.add()
    file_descriptor.name = proto_path.name
    file_descriptor.package = package_match.group(1)
    file_descriptor.syntax = "proto3"
    message_descriptor = file_descriptor.message_type.add()
    message_descriptor.name = message_match.group(1)

    field_pattern = re.compile(
        r"^\s*(?:(repeated)\s+)?(\w+)\s+([A-Za-z_]\w*)\s*=\s*(\d+)\s*;$",
        flags=re.MULTILINE,
    )
    for repeated, field_type, field_name, field_number in field_pattern.findall(message_match.group(2)):
        try:
            descriptor_type = _PROTO_SCALAR_TYPES[field_type]
        except KeyError as e:
            msg = f"{proto_path} uses unsupported test fixture field type {field_type!r}"
            raise ValueError(msg) from e
        field = message_descriptor.field.add()
        field.name = field_name
        field.number = int(field_number)
        field.label = (
            descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
            if repeated
            else descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        )
        field.type = descriptor_type

    return descriptor_set


def protobuf_message_class(
    descriptor_set: descriptor_pb2.FileDescriptorSet,
    message_name: str,
) -> type[Message]:
    """Build a dynamic protobuf class from a synthetic descriptor set."""
    pool = descriptor_pool.DescriptorPool()
    for file_descriptor in descriptor_set.file:
        pool.Add(file_descriptor)
    return cast("type[Message]", message_factory.GetMessageClass(pool.FindMessageTypeByName(message_name)))


def write_protobuf_mcap(
    path: Path,
    samples: list[McapSample],
    *,
    topic: str,
    schema_name: str,
    schema_data: bytes,
    schema_encoding: str = "protobuf",
    message_encoding: str = "protobuf",
    library: str = "cosmos_curator sensor test",
    chunk_size: int = 1_048_576,
) -> None:
    """Write serialized protobuf samples to one synthetic MCAP channel."""
    with path.open("wb") as out_file:
        writer = Writer(out_file, compression=CompressionType.ZSTD, chunk_size=chunk_size)
        writer.start(library=library)
        schema_id = writer.register_schema(name=schema_name, encoding=schema_encoding, data=schema_data)
        channel_id = writer.register_channel(
            schema_id=schema_id,
            topic=topic,
            message_encoding=message_encoding,
        )
        for sequence, sample in enumerate(samples):
            writer.add_message(
                channel_id=channel_id,
                log_time=sample.log_time_ns,
                data=sample.data,
                publish_time=sample.log_time_ns + 50,
                sequence=sequence,
            )
        writer.finish()  # type: ignore[no-untyped-call]


def one_window_spec(start_ns: int, exclusive_end_ns: int) -> SamplingSpec:
    """Build a one-window sampling spec with explicit half-open bounds."""
    grid = make_sampling_grid(
        timestamps_ns=np.array([start_ns, exclusive_end_ns], dtype=np.int64),
        stride_ns=exclusive_end_ns - start_ns,
        duration_ns=exclusive_end_ns - start_ns,
    )
    return SamplingSpec(grid=grid)


def make_sampling_grid(
    timestamps_ns: npt.NDArray[np.int64],
    stride_ns: int,
    duration_ns: int,
) -> SamplingGrid:
    """Build a SamplingGrid from a boundary-inclusive timestamp series.

    Previously, SamplingGrid took the same args as this function, but
    the signature was expanded.

    This helper uses the old semantics and translates to the expanded
    signature.
    """
    if len(timestamps_ns) == 0:
        start_ns = 0
        exclusive_end_ns = 0
    elif len(timestamps_ns) == 1:
        start_ns = int(timestamps_ns[0])
        exclusive_end_ns = int(timestamps_ns[0]) + duration_ns
    else:
        start_ns = int(timestamps_ns[0])
        exclusive_end_ns = int(timestamps_ns[-1])

    return SamplingGrid(
        start_ns=start_ns,
        exclusive_end_ns=exclusive_end_ns,
        timestamps_ns=timestamps_ns[:-1],
        stride_ns=stride_ns,
        duration_ns=duration_ns,
    )
