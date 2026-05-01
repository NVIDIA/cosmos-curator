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

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
from mcap.reader import McapReader
from mcap.records import Channel, Message, Schema
from mcap.summary import Summary

VIDEO_METADATA_RECORD_NAME = "cosmos_curator.video_metadata.v1"


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


def load_start_end_ns(reader: McapReader, topic: str) -> tuple[int, int]:
    """Load the first and last message ``log_time`` values for *topic*.

    Args:
        reader: MCAP reader positioned on the source file/stream.
        topic: Topic name to query.

    Returns:
        Tuple of ``(start_ns, end_ns)`` in nanoseconds.

    Raises:
        ValueError: If the topic has no messages.

    """
    earliest = reader.iter_messages(topics=topic, log_time_order=True, reverse=False)
    try:
        _schema, _channel, first_msg = next(earliest)
    except StopIteration as e:
        msg = f"no MCAP messages on topic {topic!r}"
        raise ValueError(msg) from e

    latest = reader.iter_messages(topics=topic, log_time_order=True, reverse=True)
    try:
        _schema, _channel, last_msg = next(latest)
    except StopIteration as e:
        msg = f"failed to read latest MCAP message on topic {topic!r} after reading earliest message"
        raise ValueError(msg) from e
    return int(first_msg.log_time), int(last_msg.log_time)


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
        raise ValueError(msg)
    arr = np.array(times, dtype=np.int64)
    arr.flags.writeable = False
    return arr


def iter_messages_log_time_ns(
    reader: McapReader,
    topic: str,
    start_ns: int,
    end_ns_exclusive: int,
    *,
    log_time_order: bool = True,
) -> Iterator[tuple[Schema | None, Channel, Message]]:
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
