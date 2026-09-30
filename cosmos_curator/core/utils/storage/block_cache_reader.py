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
"""Serve random-access reads of a cloud object from a small cache of aligned blocks.

``smart_open``'s S3 reader turns every ``seek`` that leaves its read-ahead buffer into a
fresh ``GetObject``, and libav seeks far more than it reads: indexing a fragmented MP4
probes the file size several times per fragment, each probe a ``seek(-1, SEEK_END)``
and a seek back because PyAV answers ``AVSEEK_SIZE`` with -1.

:class:`BlockCacheReader` makes ``seek`` / ``tell`` pure bookkeeping and fetches data
in fixed-size aligned blocks held in a small LRU. On a miss the wrapped stream keeps
streaming from its open response when the block starts where it already is, or a
short gap ahead of it, because reading through a gap costs only its bytes while
repositioning costs a round trip. Wider forward gaps and backward jumps seek, so a
sparse walk over large boxes -- a regular MP4's ``mdat``, a coarsely fragmented MP4's
fragments -- transfers only the blocks it touches.

Measurements and the choice of defaults are in
``docs/curator/design/sensor-library-cloud-storage.md``.
"""

import io
import sys
from collections import OrderedDict
from collections.abc import Buffer, Iterator
from typing import BinaryIO

DEFAULT_BLOCK_SIZE = 64 * 1024
DEFAULT_MAX_BLOCKS = 16
DEFAULT_SKIP_THRESHOLD = 256 * 1024


def _limit(size: int | None) -> int:
    return sys.maxsize if size is None or size < 0 else size


class BlockCacheReader(io.BufferedIOBase):
    """Read-only, seekable view of ``wrapped`` that serves reads from cached aligned blocks.

    ``wrapped`` is borrowed: closing this reader drops the cache but leaves ``wrapped``
    open for whoever owns it. ``wrapped.read(n)`` must return fewer than ``n`` bytes only
    at end of stream (the :class:`io.BufferedIOBase` contract), which is how the object
    size is usually learned for free. Otherwise the size is read once, on the first
    ``seek(..., SEEK_END)`` or read that sought past the end, from
    ``wrapped.seek(0, SEEK_END)`` -- request-free on ``smart_open``'s S3 and Azure
    readers, whose open already fetched it. Not thread-safe.
    """

    def __init__(
        self,
        wrapped: BinaryIO,
        *,
        block_size: int = DEFAULT_BLOCK_SIZE,
        max_blocks: int = DEFAULT_MAX_BLOCKS,
        skip_threshold: int = DEFAULT_SKIP_THRESHOLD,
    ) -> None:
        """Wrap ``wrapped`` without touching it beyond reading its current position.

        Args:
            wrapped: Readable, seekable binary stream, positioned where this reader starts.
            block_size: Bytes per cached block; also the largest single ``wrapped.read``.
            max_blocks: Blocks kept before the least recently used one is evicted.
            skip_threshold: Widest forward gap between the wrapped stream's position
                and a missed block that is read and dropped rather than sought over.
                ``0`` seeks over every gap.

        Raises:
            ValueError: If ``block_size`` or ``max_blocks`` is not positive, or
                ``skip_threshold`` is negative.

        """
        if block_size <= 0 or max_blocks <= 0:
            msg = f"block_size and max_blocks must be positive, got {block_size} and {max_blocks}"
            raise ValueError(msg)
        if skip_threshold < 0:
            msg = f"skip_threshold must be non-negative, got {skip_threshold}"
            raise ValueError(msg)
        super().__init__()
        self._wrapped = wrapped
        self._block_size = block_size
        self._max_blocks = max_blocks
        self._skip_threshold = skip_threshold
        self._blocks: OrderedDict[int, bytes] = OrderedDict()
        self._wrapped_pos = self._pos = wrapped.tell()
        self._size: int | None = None
        # PyAV reads ``name`` for format probing and error messages.
        if (name := getattr(wrapped, "name", None)) is not None:
            self.name = name

    def readable(self) -> bool:
        """Return True."""
        return True

    def seekable(self) -> bool:
        """Return True."""
        return True

    def tell(self) -> int:
        """Return the current position without touching the wrapped stream."""
        self._check_open()
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Move the current position; the wrapped stream is not repositioned until a read misses.

        Raises:
            ValueError: If ``whence`` is invalid or the target position is negative.

        """
        self._check_open()
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._pos + offset
        elif whence == io.SEEK_END:
            position = self._object_size() + offset
        else:
            msg = f"invalid whence ({whence}, should be {io.SEEK_SET}, {io.SEEK_CUR} or {io.SEEK_END})"
            raise ValueError(msg)
        if position < 0:
            msg = f"negative seek position {position}"
            raise ValueError(msg)
        self._pos = position
        return position

    def read(self, size: int | None = -1) -> bytes:
        """Read up to ``size`` bytes, or to end of stream if ``size`` is negative or None."""
        self._check_open()
        return b"".join(self._chunks(_limit(size)))

    def read1(self, size: int | None = -1) -> bytes:
        """Read up to ``size`` bytes from at most one block."""
        self._check_open()
        return bytes(self._next_chunk(_limit(size))) if size != 0 else b""

    def readinto(self, buffer: Buffer) -> int:
        """Read into ``buffer`` and return the number of bytes written."""
        self._check_open()
        target = memoryview(buffer).cast("B")
        written = 0
        for chunk in self._chunks(len(target)):
            target[written : written + len(chunk)] = chunk
            written += len(chunk)
        return written

    def close(self) -> None:
        """Drop the cache and mark this reader closed; the wrapped stream stays open."""
        self._blocks.clear()
        super().close()

    def _check_open(self) -> None:
        if self.closed:
            msg = "I/O operation on closed file."
            raise ValueError(msg)

    def _object_size(self) -> int:
        if self._size is None:
            self._wrapped_pos = -1
            self._size = self._wrapped_pos = self._wrapped.seek(0, io.SEEK_END)
        return self._size

    def _chunks(self, limit: int) -> Iterator[memoryview]:
        while limit and (chunk := self._next_chunk(limit)):
            limit -= len(chunk)
            yield chunk

    def _next_chunk(self, limit: int) -> memoryview:
        index, offset = divmod(self._pos, self._block_size)
        chunk = memoryview(self._block(index))[offset : offset + limit]
        self._pos += len(chunk)
        return chunk

    def _block(self, index: int) -> bytes:
        block = self._blocks.get(index)
        if block is not None:
            self._blocks.move_to_end(index)
            return block
        start = index * self._block_size
        if self._size is not None and start >= self._size:
            return b""
        # Unknown until the fetch returns, so a retry after a failed seek or read re-seeks.
        wrapped_pos, self._wrapped_pos = self._wrapped_pos, -1
        sought = wrapped_pos < 0 or not 0 <= start - wrapped_pos <= self._skip_threshold
        if sought:
            wrapped_pos = self._wrapped.seek(start)
        while wrapped_pos < start:
            skipped = len(self._wrapped.read(min(start - wrapped_pos, self._block_size)))
            if not skipped:
                self._size = self._wrapped_pos = wrapped_pos
                return b""
            wrapped_pos += skipped
        block = self._wrapped.read(self._block_size)
        self._wrapped_pos = wrapped_pos + len(block)
        if not block:
            # A stream that doesn't clamp seeks past its end (``BytesIO``) lands on ``start``
            # wherever the object ends, so only a position reached by reading is its size.
            self._size = self._object_size() if sought else wrapped_pos
            return block
        if len(block) < self._block_size:
            self._size = self._wrapped_pos
        self._blocks[index] = block
        if len(self._blocks) > self._max_blocks:
            self._blocks.popitem(last=False)
        return block
