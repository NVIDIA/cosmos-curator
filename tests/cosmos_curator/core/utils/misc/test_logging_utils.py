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

"""Tests for make_tagged_logger in logging_utils."""

from typing import Any

import pytest
from loguru import logger

from cosmos_curator.core.utils.misc.logging_utils import make_tagged_logger


@pytest.fixture(autouse=True)
def _isolate_loguru_sinks() -> None:
    """Remove all sinks before each test and restore after."""
    logger.remove()
    yield
    logger.remove()


def _capture_messages(records: list[dict[str, Any]]) -> int:
    """Add a raw-record sink to the global logger and return its sink id."""
    return logger.add(lambda msg: records.append(msg.record), format="{message}", level="DEBUG")


# -- Normal tag prefixing --


def test_normal_tag_prefixes_message() -> None:
    """Tagged logger prepends the tag to every emitted message."""
    records: list[dict[str, Any]] = []
    _capture_messages(records)

    log = make_tagged_logger("[vLLM]")
    log.info("engine ready")

    assert len(records) == 1
    assert records[0]["message"] == "[vLLM] engine ready"


# -- Empty and whitespace-only tags --


def test_empty_tag_leaves_message_unchanged() -> None:
    """Empty string tag produces no prefix and no leading space."""
    records: list[dict[str, Any]] = []
    _capture_messages(records)

    log = make_tagged_logger("")
    log.info("hello")

    assert len(records) == 1
    assert records[0]["message"] == "hello"


def test_whitespace_only_tag_leaves_message_unchanged() -> None:
    """Whitespace-only tag is treated as empty -- no prefix added."""
    records: list[dict[str, Any]] = []
    _capture_messages(records)

    log = make_tagged_logger("   ")
    log.info("hello")

    assert len(records) == 1
    assert records[0]["message"] == "hello"


# -- Special characters --


def test_special_character_tag() -> None:
    """Tags containing brackets, colons, and symbols are preserved."""
    records: list[dict[str, Any]] = []
    _capture_messages(records)

    log = make_tagged_logger("[ray:worker-0/gpu:3]")
    log.info("started")

    assert records[0]["message"] == "[ray:worker-0/gpu:3] started"


# -- Multi-line messages --


def test_multiline_message_prefixed_once() -> None:
    """Tag is prepended once; multi-line body is preserved as-is."""
    records: list[dict[str, Any]] = []
    _capture_messages(records)

    log = make_tagged_logger("[tag]")
    log.info("line1\nline2\nline3")

    assert records[0]["message"] == "[tag] line1\nline2\nline3"


# -- Parameterized / formatted messages --


def test_parameterized_message() -> None:
    """Loguru positional placeholders work with tagged logger."""
    records: list[dict[str, Any]] = []
    _capture_messages(records)

    log = make_tagged_logger("[db]")
    log.info("rows={}, table={}", 42, "users")

    assert records[0]["message"] == "[db] rows=42, table=users"


def test_fstring_formatted_message() -> None:
    """Pre-formatted f-string messages are tagged correctly."""
    records: list[dict[str, Any]] = []
    _capture_messages(records)

    log = make_tagged_logger("[net]")
    status = 200
    log.info(f"status={status}")  # noqa: G004 -- intentionally testing f-string passthrough

    assert records[0]["message"] == "[net] status=200"


# -- Tag is stripped (leading/trailing whitespace normalized) --


def test_tag_whitespace_is_stripped() -> None:
    """Leading and trailing whitespace in tag is normalized away."""
    records: list[dict[str, Any]] = []
    _capture_messages(records)

    log = make_tagged_logger("  [padded]  ")
    log.info("ok")

    assert records[0]["message"] == "[padded] ok"


# -- Shared sinks with global logger --


def test_tagged_logger_uses_global_sink() -> None:
    """Tagged logger routes messages to sinks registered on the global logger."""
    records: list[dict[str, Any]] = []
    _capture_messages(records)

    log = make_tagged_logger("[shared]")
    log.info("visible")

    assert len(records) == 1
    assert "[shared] visible" in records[0]["message"]


def test_tagged_logger_receives_sink_added_after_creation() -> None:
    """Sinks added to global logger after patch creation still receive messages."""
    log = make_tagged_logger("[late]")

    records: list[dict[str, Any]] = []
    _capture_messages(records)

    log.info("arrived")

    assert len(records) == 1
    assert records[0]["message"] == "[late] arrived"


# -- Shared levels with global logger --


def test_tagged_logger_respects_global_minimum_level() -> None:
    """Tagged logger obeys the minimum level set on the global sink."""
    records: list[dict[str, Any]] = []
    logger.add(lambda msg: records.append(msg.record), format="{message}", level="WARNING")

    log = make_tagged_logger("[lvl]")
    log.debug("should be filtered")
    log.info("should be filtered")
    log.warning("should pass")

    assert len(records) == 1
    assert records[0]["message"] == "[lvl] should pass"


# -- Shared filters with global logger --


def test_tagged_logger_respects_global_filter() -> None:
    """Tagged logger obeys filters attached to the global sink."""
    records: list[dict[str, Any]] = []
    logger.add(
        lambda msg: records.append(msg.record),
        format="{message}",
        level="DEBUG",
        filter=lambda record: "keep" in record["message"],
    )

    log = make_tagged_logger("[flt]")
    log.info("drop this")
    log.info("keep this")

    assert len(records) == 1
    assert records[0]["message"] == "[flt] keep this"
