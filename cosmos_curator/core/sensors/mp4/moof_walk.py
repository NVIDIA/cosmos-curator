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

"""Walk a fragmented MP4 once, yielding every ``moof`` box verbatim."""

import struct
from collections.abc import Iterator
from typing import IO

from cosmos_curator.core.sensors.mp4.box_primitives import (
    EXTENDED_HEADER_BYTES,
    EXTENDED_SIZE_MARKER,
    STANDARD_HEADER_BYTES,
    read_box_header,
)
from cosmos_curator.core.sensors.mp4.errors import UnindexableVideoError


class NoFragmentsFoundError(UnindexableVideoError):
    """A file has no ``moof`` boxes at all -- not fragmented, or not fragmented the way expected."""


def _consume_skip(fetched: bytes, skip_remaining: int) -> tuple[bytes, int]:
    """Drop up to *skip_remaining* bytes off the front of *fetched*.

    Returns ``(remaining fetched, remaining skip_remaining)``.
    """
    if not skip_remaining:
        return fetched, 0
    skip_now = min(skip_remaining, len(fetched))
    return fetched[skip_now:], skip_remaining - skip_now


def walk_moofs(file_size: int, handle: IO[bytes], *, chunk_bytes: int) -> Iterator[tuple[int, bytes]]:
    """Yield ``(moof_start, moof_bytes)`` for every top-level ``moof`` box, in file order.

    ``moof_bytes`` preserves content unread. Reads from *handle* without opening or closing it --
    that's the caller's responsibility.

    *file_size* resolves a size-0 ("extends to end of file") box's true extent -- it is not
    used for loop termination, which is detected from the stream's own EOF.
    """
    leftover = b""
    leftover_start = 0
    skip_remaining = 0
    while True:
        fetched = handle.read(chunk_bytes)
        if not fetched:
            return
        fetched_len = len(fetched)
        unskipped, skip_remaining = _consume_skip(fetched, skip_remaining)
        leftover_start += fetched_len - len(unskipped)  # bytes of this chunk the ongoing skip consumed
        if not unskipped:
            continue  # the whole chunk was skipped -- nothing new to parse yet
        buf = leftover + unskipped
        buf_base = leftover_start
        # buf-relative file end, passed to read_box_header instead of len(buf) -- so a size-0
        # box resolves to the real file end, not wherever this chunk happens to stop.
        file_end_in_buf = file_size - buf_base
        cursor = 0
        while cursor + STANDARD_HEADER_BYTES <= len(buf):
            size32 = struct.unpack_from(">I", buf, cursor)[0]
            header_bytes = EXTENDED_HEADER_BYTES if size32 == EXTENDED_SIZE_MARKER else STANDARD_HEADER_BYTES
            if cursor + header_bytes > len(buf):
                break  # header itself not fully buffered yet -- carry into the next chunk
            box_type, box_size, _header_len = read_box_header(buf, cursor, file_end_in_buf)
            if box_size <= 0:
                break  # box not fully present yet -- carry it into the next chunk
            if cursor + box_size > len(buf):
                if box_type == "moof":
                    break  # carry the whole partial moof into leftover -- its bytes are the point
                # Its content is never read -- skip the rest of it instead of buffering it, so
                # memory doesn't scale with this one box's own size (a GOP-sized mdat, say).
                skip_remaining = box_size - (len(buf) - cursor)
                cursor = len(buf)
                break
            if box_type == "moof":
                yield buf_base + cursor, buf[cursor : cursor + box_size]
            cursor += box_size
        leftover = buf[cursor:]
        leftover_start = buf_base + cursor
