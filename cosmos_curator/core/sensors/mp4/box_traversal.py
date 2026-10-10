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

"""Scoped box traversal with invocation-local UUID handlers."""

import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from cosmos_curator.core.sensors.mp4.box_primitives import (
    EXTENDED_HEADER_BYTES,
    EXTENDED_SIZE_MARKER,
    STANDARD_HEADER_BYTES,
    BoxNotFoundError,
    MalformedBoxHeaderError,
    read_box_header,
)

_UUID_BYTES = 16


class UuidCallbackError(Exception):
    """A handler failure, separate from standard MP4 parsing errors."""

    def __init__(self, error: Exception) -> None:
        """Retain the original exception for caller-specific failure handling."""
        self.error = error
        super().__init__(str(error))


@dataclass(frozen=True)
class BoxScope:
    """A child-box range with its parent path and optional track/fragment identity."""

    start: int
    end: int
    path: tuple[str, ...]
    track_id: int | None = None
    fragment_index: int | None = None


@dataclass(frozen=True)
class BoxLocation:
    """One box's buffer-relative bounds and enclosing scope."""

    box_type: str
    offset: int
    content_start: int
    end: int
    parent: BoxScope

    def child_scope(self, *, track_id: int | None = None) -> BoxScope:
        """Describe this container's children, optionally identifying its track."""
        return BoxScope(
            self.content_start,
            self.end,
            (*self.parent.path, self.box_type),
            self.parent.track_id if track_id is None else track_id,
            self.parent.fragment_index,
        )


@dataclass(frozen=True)
class UuidHandler:
    """Decode one UUID identifier at explicitly supported parent paths."""

    identifier: bytes
    parent_paths: frozenset[tuple[str, ...]]
    callback: Callable[[bytes, BoxLocation], None]

    def __post_init__(self) -> None:
        """Reject identifiers that cannot match a UUID box."""
        if len(self.identifier) != _UUID_BYTES:
            msg = "UUID handler identifiers must contain 16 bytes"
            raise ValueError(msg)


class BoxTraversal:
    """Index requested scopes and dispatch their UUID boxes once per traversal.

    Callers descend only into known container layouts. Reusing a scope returns
    its recorded children without invoking handlers again. Create a new traversal
    for each buffer and parsing attempt.
    """

    def __init__(self, buffer: bytes, *, uuid_handlers: Sequence[UuidHandler] = ()) -> None:
        """Bind handlers to one buffer without shared registration state."""
        self._buffer = buffer
        self._handlers: dict[tuple[bytes, tuple[str, ...]], UuidHandler] = {}
        for handler in uuid_handlers:
            for path in handler.parent_paths:
                key = (handler.identifier, path)
                if key in self._handlers:
                    msg = "Duplicate UUID handler for the same identifier and parent path"
                    raise ValueError(msg)
                self._handlers[key] = handler
        self._scopes: dict[BoxScope, tuple[BoxLocation, ...]] = {}
        self._dispatched: set[int] = set()

    def scan_children(self, scope: BoxScope) -> tuple[BoxLocation, ...]:
        """Return ordered child locations, invoking matching handlers during the scan."""
        if scope in self._scopes:
            return self._scopes[scope]
        if not 0 <= scope.start <= scope.end <= len(self._buffer):
            msg = "Box scope extends outside its buffer"
            raise MalformedBoxHeaderError(msg)
        children: list[BoxLocation] = []
        offset = scope.start
        while scope.end - offset >= STANDARD_HEADER_BYTES:
            size32 = struct.unpack_from(">I", self._buffer, offset)[0]
            if size32 == EXTENDED_SIZE_MARKER and scope.end - offset < EXTENDED_HEADER_BYTES:
                msg = "Truncated extended box header"
                raise MalformedBoxHeaderError(msg)
            box_type, size, header_len = read_box_header(self._buffer, offset, scope.end)
            end = offset + size
            if end > scope.end:
                msg = "Box extends past its parent scope"
                raise MalformedBoxHeaderError(msg)
            location = BoxLocation(box_type, offset, offset + header_len, end, scope)
            children.append(location)
            if box_type == "uuid":
                self._dispatch(location)
            offset = end
        result = tuple(children)
        self._scopes[scope] = result
        return result

    def find(self, scope: BoxScope, path: tuple[str, ...]) -> BoxLocation:
        """Locate a descendant using recorded children at each path segment."""
        for index, name in enumerate(path):
            location = next((box for box in self.scan_children(scope) if box.box_type == name), None)
            if location is None:
                msg = f"{'/'.join(path[: index + 1])} not found"
                raise BoxNotFoundError(msg)
            if index == len(path) - 1:
                return location
            scope = location.child_scope()
        msg = "Box path must not be empty"
        raise ValueError(msg)

    def find_bounds(self, scope: BoxScope, path: tuple[str, ...]) -> tuple[int, int]:
        """Return a descendant's content bounds for existing field decoders."""
        box = self.find(scope, path)
        return box.content_start, box.end

    def extension_boxes(self, scope: BoxScope) -> tuple[tuple[bytes, bytes], ...]:
        """Preserve UUID identifiers and payloads alongside handler dispatch."""
        return tuple(
            (
                self._buffer[box.content_start : box.content_start + _UUID_BYTES],
                self._buffer[box.content_start + _UUID_BYTES : box.end],
            )
            for box in self.scan_children(scope)
            if box.box_type == "uuid"
        )

    def find_bytes(self, scope: BoxScope, path: tuple[str, ...]) -> bytes:
        """Return a located box including its header."""
        box = self.find(scope, path)
        return self._buffer[box.offset : box.end]

    def _dispatch(self, location: BoxLocation) -> None:
        if location.end - location.content_start < _UUID_BYTES:
            msg = "UUID box is missing its 16-byte identifier"
            raise MalformedBoxHeaderError(msg)
        identifier_end = location.content_start + _UUID_BYTES
        identifier = self._buffer[location.content_start : identifier_end]
        handler = self._handlers.get((identifier, location.parent.path))
        if handler is None or location.offset in self._dispatched:
            return
        try:
            handler.callback(self._buffer[identifier_end : location.end], location)
        except Exception as exc:
            raise UuidCallbackError(exc) from exc
        self._dispatched.add(location.offset)
