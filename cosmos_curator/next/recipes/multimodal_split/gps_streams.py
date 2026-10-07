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

"""Writing a clip's GPS rows in the schema its session was recorded in.

A GPS clip is not a cut of the source the way a trajectory clip is. The source
arrives at 1 Hz against a 30 Hz grid, so the grid is denser than the fixes and
every row is placed rather than copied -- selected from the nearest fix, or
derived between the two that bracket it.

The rows go back in the source's own schema, encoded against the descriptor the
source MCAP carries, so a clip opens with whatever already reads the session and
nothing here needs a compiled protobuf. That is a schema the writer is handed,
not a schema it understands: the position fields it fills are named here, so a
producer whose message spells them differently needs this module changed. What it
buys is that the clip and the session are the same message type, and a field this
module does not fill is simply absent from a clip's rows rather than wrong.
"""

from pathlib import Path

import numpy as np
from google.protobuf.descriptor import Descriptor, FieldDescriptor
from google.protobuf.message import Message

from cosmos_curator.core.sensors.data.gps_data import GpsData
from cosmos_curator.core.sensors.utils.mcap import (
    McapMessageRecord,
    McapProtobufMessageResolver,
    write_mcap_messages,
)
from cosmos_curator.next.recipes.multimodal_split.mcap_streams import McapTopicWriter

#: Provenance keys written into the message's own metadata map.
#:
#: The source already uses that map, so this rides in the schema rather than
#: beside it. Which fixes produced a row is not recoverable from the row itself
#: once it sits on the grid, and it is what tells a reader whether a position is
#: a measurement or a derivation, and how far it reached for one.
SAMPLING_KEY = "cosmos_curator.sampling"
SOURCE_TIMESTAMP_KEY = "cosmos_curator.source_timestamp_ns"
SOURCE_BRACKET_KEY = "cosmos_curator.source_bracket_timestamps_ns"

#: What ``_encode`` needs of the message it fills, so a source that cannot carry
#: a row is refused when the writer opens rather than once per clip from deep
#: inside the cut.
#:
#: Shape, not just names. A descriptor can spell every field correctly and still
#: be unwritable -- a ``timestamp`` that is an integer rather than a message, or
#: a singular ``metadata`` -- and that fails identically per clip, which is the
#: outcome this check exists to convert into one named failure.
_REQUIRED_FIELDS: tuple[tuple[str, int, int, tuple[str, ...]], ...] = (
    ("timestamp", FieldDescriptor.TYPE_MESSAGE, FieldDescriptor.LABEL_OPTIONAL, ("seconds", "nanos")),
    ("latitude", FieldDescriptor.TYPE_DOUBLE, FieldDescriptor.LABEL_OPTIONAL, ()),
    ("longitude", FieldDescriptor.TYPE_DOUBLE, FieldDescriptor.LABEL_OPTIONAL, ()),
    ("altitude", FieldDescriptor.TYPE_DOUBLE, FieldDescriptor.LABEL_OPTIONAL, ()),
    ("metadata", FieldDescriptor.TYPE_MESSAGE, FieldDescriptor.LABEL_REPEATED, ("key", "value")),
)


def _describe_mismatch(descriptor: Descriptor, required: tuple[str, int, int, tuple[str, ...]]) -> str | None:
    """Return why *descriptor* cannot carry this field, or None when it can."""
    name, field_type, label, children = required
    field = descriptor.fields_by_name.get(name)
    if field is None:
        return f"carries no {name} field"
    if field.type != field_type:
        return f"has {name} as type {field.type}, which cannot hold what a row needs"
    if field.is_repeated:
        actual_label = FieldDescriptor.LABEL_REPEATED
    elif field.is_required:
        actual_label = FieldDescriptor.LABEL_REQUIRED
    else:
        actual_label = FieldDescriptor.LABEL_OPTIONAL
    if actual_label != label:
        plural = "repeated" if label == FieldDescriptor.LABEL_REPEATED else "singular"
        return f"has {name} where a {plural} one is needed"
    missing = [child for child in children if child not in field.message_type.fields_by_name]
    if missing:
        return f"has {name} without its {', '.join(missing)}"
    return None


class GpsClipWriter(McapTopicWriter):
    """Writes aligned GPS rows using the schema and topic of the source MCAP."""

    def __init__(
        self,
        mcap_bytes: bytes,
        configured_topic: str | None,
        required_fields: tuple[tuple[str, int, int, tuple[str, ...]], ...] = _REQUIRED_FIELDS,
    ) -> None:
        """Resolve the source's topic, schema and message class once per session."""
        super().__init__(mcap_bytes, configured_topic)
        resolver = McapProtobufMessageResolver(self.schema_name, schema_label="GPS protobuf")
        message_cls = resolver.resolve_from_summary(self._reader, self.topic)
        if message_cls is None:
            msg = f"MCAP has no summary, so the message type for topic {self.topic!r} cannot be resolved"
            raise ValueError(msg)
        mismatches = [
            reason
            for required in required_fields
            if (reason := _describe_mismatch(message_cls.DESCRIPTOR, required)) is not None
        ]
        if mismatches:
            msg = f"GPS rows cannot be written as {self.schema_name!r}, which {'; and '.join(mismatches)}"
            raise ValueError(msg)
        self._message_cls = message_cls

    def verify_log_time_matches_payload(self) -> None:
        """Refuse a source whose envelope and payload disagree about when a fix happened.

        A message's MCAP ``log_time`` and the timestamp inside its payload are the
        same instant recorded twice: when the receiver observed the fix. Not when
        the recorder received it -- that is ``publish_time``, which trails by a
        sensor-specific transport latency and is deliberately a different number.

        Planning bounds a clip window from ``log_time`` and alignment runs on the
        payload's, so a converter that lets them diverge makes the window and the
        samples describe different clocks. Every clip then fails with something
        that reads like missing data rather than a broken converter.

        Checked once when the sensor opens rather than per clip, and against every
        message rather than the first: a per-sample mapping error shows up part way
        through a recording, not at its head.

        Raises:
            ValueError: If any message's payload timestamp differs from its
                ``log_time``, naming the first one and the size of the gap.

        """
        for _schema, _channel, message in self._reader.iter_messages(topics=self.topic, log_time_order=True):
            fix = self._message_cls.FromString(message.data)
            payload_ns = fix.timestamp.seconds * 1_000_000_000 + fix.timestamp.nanos
            if payload_ns != message.log_time:
                msg = (
                    f"{self.topic!r} carries a payload timestamp of {payload_ns} against a log_time of "
                    f"{message.log_time}, a gap of {payload_ns - message.log_time} ns. They are one instant "
                    f"recorded twice, so the converter that wrote this is at fault."
                )
                raise ValueError(msg)

    def write_rows(self, aligned: GpsData, sampling: str, dest: Path) -> int:
        """Write one message per grid row; return how many.

        ``log_time`` is the grid timestamp, which is what the row is *for* and
        what makes row *i* here the same instant as frame *i* of the video. The
        payload's own timestamp carries it too, matching the source, where the
        two agree.
        """
        records = [
            McapMessageRecord(
                data=self._encode(aligned, index, sampling).SerializeToString(),
                log_time_ns=int(aligned.align_timestamps_ns[index]),
                publish_time_ns=int(aligned.align_timestamps_ns[index]),
                sequence=index,
            )
            for index in range(len(aligned.align_timestamps_ns))
        ]
        stats = write_mcap_messages(dest, stream=self._stream, records=records)
        return stats.message_count

    def _encode(self, aligned: GpsData, index: int, sampling: str) -> Message:
        """Build one row's message, recording where its position came from."""
        message = self._message_cls()
        align_ns = int(aligned.align_timestamps_ns[index])
        message.timestamp.seconds = align_ns // 1_000_000_000
        message.timestamp.nanos = align_ns % 1_000_000_000
        message.latitude = float(aligned.latitude_deg[index])
        message.longitude = float(aligned.longitude_deg[index])
        message.altitude = float(aligned.altitude_m[index])

        provenance = {SAMPLING_KEY: sampling}
        brackets = aligned.source_bracket_timestamps_ns
        if brackets is not None:
            # Interpolated: the two fixes the position was derived between, which
            # is the only record that it was derived at all.
            provenance[SOURCE_BRACKET_KEY] = ",".join(str(int(value)) for value in np.asarray(brackets[index]))
        else:
            # Selected: the fix that actually served this row. A reader wanting
            # staleness subtracts it from the row's own timestamp.
            provenance[SOURCE_TIMESTAMP_KEY] = str(int(aligned.sensor_timestamps_ns[index]))
        for key, value in provenance.items():
            entry = message.metadata.add()
            entry.key = key
            entry.value = value
        return message
