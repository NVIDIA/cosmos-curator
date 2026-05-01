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
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from mcap.reader import make_reader as mcap_make_reader
from mcap.writer import CompressionType, Writer

from cosmos_curator.core.sensors.utils.mcap import (
    channel_for_topic,
    get_metadata_record,
    iter_messages_log_time_ns,
    load_start_end_ns,
    load_timeline,
)

_RGB_SCHEMA = {
    "type": "object",
    "title": "cosmos_curator.sensors.rgb8_frame",
    "description": "test",
}


def _payload(width: int, height: int, fill: int) -> bytes:
    return bytes([fill]) * (width * height * 3)


def _write_rgb8_mcap(path: Path, topic: str, times_ns: list[int], width: int = 2, height: int = 2) -> None:
    with path.open("wb") as out:
        writer = Writer(out, compression=CompressionType.ZSTD)
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
