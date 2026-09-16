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

"""Cutting one session-long MCAP topic into a clip's worth at a time.

Records are copied, never decoded: the source schema is written through as the
bytes it arrived as, and each message keeps its own payload. So a topic is
sliceable without anything here knowing what its messages mean -- the same code
serves a Curator schema and a Foxglove one, and a schema nobody has written a
mapping for yet still cuts correctly.

That is also what makes a clip's MCAP readable by whatever reads the session's.
A consumer opens the clip with the code it already has, because the schema
travelled with the messages.
"""

import io
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

from mcap.reader import make_reader
from mcap.records import Channel, Message, Schema
from mcap.summary import Summary

from cosmos_curator.core.sensors.utils.mcap import (
    McapMessageRecord,
    McapSchemaSpec,
    McapStreamSpec,
    channel_for_topic,
    iter_messages_log_time_ns,
    load_start_end_ns,
    schema_for_channel,
    write_mcap_messages,
)
from cosmos_curator.next.media.spans import Span

McapMessage = tuple[Schema | None, Channel, Message]


def _log_time(message: McapMessage) -> int:
    """Return the envelope time the cut is made on, which is never the payload's."""
    return int(message[2].log_time)


@dataclass(frozen=True)
class SpanCut:
    """One span's messages, selected but not yet written.

    Selection and writing are separate so the caller can name the destination
    from facts the cutter does not have -- a clip id -- without the cutter
    taking a second pass to find the messages again.
    """

    span: Span
    leading: McapMessage | None
    body: list[McapMessage]


_LIBRARY = "cosmos-curator multimodal-split"

# Past any recorder's clock. The reader takes a half-open interval and there is
# no sentinel for "to the end", so the whole topic is asked for as one.
_NO_END = 2**63 - 1


def resolve_topic(summary: Summary | None, configured: str | None) -> str:
    """Return the topic to cut, defaulting to the only one the file carries.

    A recorder writes one sensor per MCAP, so naming the topic is usually
    restating what the file already says. It stays configurable for a producer
    that packs several onto one file, and is required there: picking one of
    several would silently drop the rest.

    Raises:
        ValueError: If the file has no channels, or has several and none was named.

    """
    if summary is None:
        msg = "MCAP has no summary, so its topics cannot be listed without reading it whole"
        raise ValueError(msg)
    if configured is not None:
        return configured
    topics = sorted({channel.topic for channel in summary.channels.values()})
    if not topics:
        msg = "MCAP carries no channels, so there is no topic to cut"
        raise ValueError(msg)
    if len(topics) > 1:
        msg = f"MCAP carries {len(topics)} topics, so one must be named in config: {', '.join(topics)}"
        raise ValueError(msg)
    return topics[0]


def read_bounds(stream: io.BufferedIOBase, configured_topic: str | None) -> tuple[int, int]:
    """Return the first and last message times on the topic, in nanoseconds.

    Two indexed seeks on a chunked file, so this stays flat as a session grows:
    a three-hour recording answers for about the same bytes as a twenty-second
    one. Planning uses it to decide coverage without reading a topic it is not
    going to cut.
    """
    reader = make_reader(cast("BinaryIO", stream))
    topic = resolve_topic(reader.get_summary(), configured_topic)
    return load_start_end_ns(reader, topic)


class McapTopicWriter:
    """Holds one buffered MCAP open, and writes files carrying its topic's schema.

    Buffered rather than streamed. Windowed reads against a remote object re-fetch
    a chunk for every span that touches it, and these files are small enough to
    hold whole where a camera is not, so the object is fetched once and every span
    is served from memory.

    Shared by the two ways a sensor becomes a clip -- cutting records out of the
    source, and writing rows derived from it -- because both produce a file whose
    schema and topic are the source's, and only what goes in the messages differs.
    """

    def __init__(self, mcap_bytes: bytes, configured_topic: str | None) -> None:
        """Open one whole MCAP, resolving the topic its output will carry."""
        self._reader = make_reader(io.BytesIO(mcap_bytes))
        summary = self._reader.get_summary()
        if summary is None:
            msg = "MCAP has no summary, so its channels cannot be read without reading it whole"
            raise ValueError(msg)
        self.topic = resolve_topic(summary, configured_topic)
        channel = channel_for_topic(summary, self.topic)
        if channel is None:
            msg = f"MCAP carries no channel for topic {self.topic!r}"
            raise ValueError(msg)
        schema = schema_for_channel(summary, channel, self.topic)
        self.schema_name = schema.name
        self._stream = McapStreamSpec(
            topic=self.topic,
            schema=McapSchemaSpec(
                name=schema.name,
                encoding=schema.encoding,
                data=schema.data,
                message_encoding=channel.message_encoding,
            ),
            library=_LIBRARY,
        )


class McapSpanWriter(McapTopicWriter):
    """Cuts one buffered MCAP topic into a file per span."""

    def __init__(self, mcap_bytes: bytes, configured_topic: str | None) -> None:
        """Open one whole MCAP for cutting, and note whether it can be seeked."""
        super().__init__(mcap_bytes, configured_topic)
        # Whether a reverse seek is available at all. Without chunk indexes the
        # mcap reader falls back to a non-seeking scan that drops ``reverse``
        # silently and yields ascending, so a reverse lookup would return the
        # earliest message rather than the latest. ``load_start_end_ns`` branches
        # on the same fact for the same reason.
        summary = self._reader.get_summary()
        self._seekable = bool(summary is not None and summary.chunk_indexes)

    def iter_spans(self, spans: Sequence[Span]) -> Iterator[SpanCut]:
        """Select every span's messages in one forward pass over the topic.

        One pass, not one indexed lookup per span, for the reason the camera
        splitter walks a video once: a chunk holds far more messages than a clip,
        so seeking per clip decompresses the same chunk once for every clip that
        lands in it. Without chunk indexes it is worse, because each lookup
        rescans from the start of the file, making the cost quadratic in the
        number of clips.

        Spans must be ascending and non-overlapping, which is what the planner
        tiles. The cursor only moves forward, so a span behind it would silently
        come back empty.

        A failure cutting one span is the caller's to handle and does not stop
        the pass: the caller decides whether a clip that could not be written
        ends the sensor.
        """
        previous: McapMessage | None = None
        messages = iter_messages_log_time_ns(self._reader, self.topic, 0, _NO_END)
        pending = next(messages, None)
        last_end_ns: int | None = None
        for span in spans:
            if last_end_ns is not None and span.start_ns < last_end_ns:
                msg = f"spans must ascend and not overlap, but {span.start_ns} follows {last_end_ns}"
                raise ValueError(msg)
            last_end_ns = span.end_ns

            # Everything before the span is behind the cursor; the last of them is
            # the bracket. Keeping only the last is what bounds this to a clip's
            # worth of messages rather than the session's.
            while pending is not None and _log_time(pending) < span.start_ns:
                previous = pending
                pending = next(messages, None)
            body: list[McapMessage] = []
            while pending is not None and _log_time(pending) < span.end_ns:
                body.append(pending)
                pending = next(messages, None)

            yield SpanCut(span=span, leading=previous, body=body)
            if body:
                previous = body[-1]

    def write_cut(self, cut: SpanCut, dest: Path) -> tuple[int, int]:
        """Write one cut span to *dest*; return how many messages, and how many are its own.

        Half-open on the bounds the cameras were cut on, plus the one message
        immediately before ``start`` where a recording has one.

        That bracket is what makes a clip answerable at its own first instant. A
        consumer asking for a value at exactly ``start`` needs a sample at or
        before it, and a strict half-open cut leaves the first sample strictly
        after.

        One message, not a duration, because one is what bracketing an instant
        takes. The cost is that adjacent clips share a message, so a consumer
        concatenating clips has to drop it -- which is why the count of the
        clip's own messages is returned beside the total.
        """
        selected = ([cut.leading] if cut.leading is not None else []) + cut.body
        stats = write_mcap_messages(
            dest,
            stream=self._stream,
            records=(
                McapMessageRecord(
                    data=message.data,
                    log_time_ns=message.log_time,
                    publish_time_ns=message.publish_time,
                    sequence=message.sequence,
                )
                for _schema, _channel, message in selected
            ),
        )
        return stats.message_count, len(cut.body)
