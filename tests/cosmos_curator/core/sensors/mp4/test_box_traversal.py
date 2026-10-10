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

"""Tests for scoped UUID dispatch and box bounds."""

import struct

import pytest

from cosmos_curator.core.sensors.mp4.box_primitives import MalformedBoxHeaderError
from cosmos_curator.core.sensors.mp4.box_traversal import (
    BoxLocation,
    BoxScope,
    BoxTraversal,
    UuidCallbackError,
    UuidHandler,
)
from tests.cosmos_curator.core.sensors.mp4._box_fixtures import pack_box

_IDENTIFIER = b"1234567890abcdef"


def test_nested_dispatch_preserves_context_and_runs_once() -> None:
    """Repeated scope reads preserve order and context without repeating callbacks."""
    events: list[tuple[bytes, BoxLocation]] = []
    buffer = pack_box("moof", pack_box("uuid", _IDENTIFIER + b"first") + pack_box("uuid", _IDENTIFIER + b"second"))
    traversal = BoxTraversal(
        buffer,
        uuid_handlers=[
            UuidHandler(_IDENTIFIER, frozenset({("moof",)}), lambda payload, box: events.append((payload, box)))
        ],
    )
    root = BoxScope(0, len(buffer), (), track_id=7, fragment_index=3)
    moof = traversal.scan_children(root)[0]
    scope = moof.child_scope()
    children = traversal.scan_children(scope)
    assert traversal.scan_children(scope) == children
    assert [payload for payload, _box in events] == [b"first", b"second"]
    assert [box.offset for _payload, box in events] == [8, 37]
    assert all(box.parent.path == ("moof",) for _payload, box in events)
    assert all(box.parent.track_id == 7 and box.parent.fragment_index == 3 for _payload, box in events)


def test_uuid_matching_requires_both_identifier_and_parent_path() -> None:
    """Dispatch requires a matching identifier and supported parent path."""
    events: list[bytes] = []
    buffer = pack_box("uuid", b"other-identifier" + b"ignored") + pack_box("uuid", _IDENTIFIER + b"matched")
    handler = UuidHandler(_IDENTIFIER, frozenset({("moof",)}), lambda payload, _box: events.append(payload))
    traversal = BoxTraversal(buffer, uuid_handlers=[handler])
    assert len(traversal.scan_children(BoxScope(0, len(buffer), ("stbl",)))) == 2
    assert events == []
    traversal.scan_children(BoxScope(0, len(buffer), ("moof",)))
    assert events == [b"matched"]


def test_leaf_payload_is_not_traversed() -> None:
    """Scanning siblings does not descend into a leaf payload."""
    events: list[bytes] = []
    buffer = pack_box("mdat", pack_box("uuid", _IDENTIFIER + b"not a child box"))
    traversal = BoxTraversal(
        buffer,
        uuid_handlers=[UuidHandler(_IDENTIFIER, frozenset({("mdat",)}), lambda payload, _box: events.append(payload))],
    )
    assert len(traversal.scan_children(BoxScope(0, len(buffer), ()))) == 1
    assert events == []


@pytest.mark.parametrize("extended", [False, True])
def test_uuid_payload_excludes_identifier_and_box_header(*, extended: bool) -> None:
    """Handlers receive only the bytes following the UUID identifier."""
    events: list[bytes] = []
    content = _IDENTIFIER + b"payload"
    buffer = struct.pack(">I4sQ", 1, b"uuid", 16 + len(content)) + content if extended else pack_box("uuid", content)
    traversal = BoxTraversal(
        buffer, uuid_handlers=[UuidHandler(_IDENTIFIER, frozenset({()}), lambda payload, _box: events.append(payload))]
    )
    traversal.scan_children(BoxScope(0, len(buffer), ()))
    assert events == [b"payload"]


@pytest.mark.parametrize(
    "buffer",
    [pack_box("uuid", b"short"), struct.pack(">I4s", 1, b"uuid"), struct.pack(">I4s", 100, b"free")],
)
def test_malformed_boxes_raise_before_dispatch(buffer: bytes) -> None:
    """Malformed boxes fail with the sensor parsing exception."""
    with pytest.raises(MalformedBoxHeaderError):
        BoxTraversal(buffer).scan_children(BoxScope(0, len(buffer), ()))


def test_size_zero_and_trailing_padding() -> None:
    """Size-zero boxes and short trailing padding follow existing walking conventions."""
    buffer = pack_box("free", b"") + b"xx"
    assert len(BoxTraversal(buffer).scan_children(BoxScope(0, len(buffer), ()))) == 1
    buffer = struct.pack(">I4s", 0, b"uuid") + _IDENTIFIER
    assert BoxTraversal(buffer).scan_children(BoxScope(0, len(buffer), ()))[0].end == len(buffer)


def test_registration_validation_and_independent_traversals() -> None:
    """Registrations reject ambiguity and do not share traversal state."""
    events: list[bytes] = []
    handler = UuidHandler(_IDENTIFIER, frozenset({()}), lambda payload, _box: events.append(payload))
    with pytest.raises(ValueError, match="16 bytes"):
        UuidHandler(b"short", frozenset({()}), handler.callback)
    with pytest.raises(ValueError, match="Duplicate"):
        BoxTraversal(b"", uuid_handlers=[handler, handler])
    buffer = pack_box("uuid", _IDENTIFIER + b"payload")
    for _attempt in range(2):
        BoxTraversal(buffer, uuid_handlers=[handler]).scan_children(BoxScope(0, len(buffer), ()))
    assert events == [b"payload", b"payload"]


@pytest.mark.parametrize("error_type", [IndexError, struct.error, RuntimeError, ValueError])
def test_callback_errors_preserve_the_original_exception(error_type: type[Exception]) -> None:
    """Handler failures carry their original exception through traversal."""
    error = error_type("extension bug")

    def fail(_payload: bytes, _box: BoxLocation) -> None:
        raise error

    buffer = pack_box("uuid", _IDENTIFIER)
    with pytest.raises(UuidCallbackError, match="extension bug") as caught:
        BoxTraversal(buffer, uuid_handlers=[UuidHandler(_IDENTIFIER, frozenset({()}), fail)]).scan_children(
            BoxScope(0, len(buffer), ())
        )
    assert caught.value.error is error
    assert caught.value.__cause__ is error
