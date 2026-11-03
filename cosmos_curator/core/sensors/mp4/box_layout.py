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

"""Lay out one video's extracted ``moov``/``moof`` boxes as a file libav can open without its media."""

import bisect
import io
import struct
from collections.abc import Sequence
from typing import Any

from cosmos_curator.core.sensors.mp4.errors import UnindexableVideoError

_BOX_HEADER_BYTES = 8
_ZEROS = bytes(1 << 20)


class BoxLayoutUnsupportedError(UnindexableVideoError):
    """The extracted boxes cannot be laid out as a file libav walks the way it walks the source."""


def _filler_box_header(box_type: bytes, size: int) -> bytes:
    """Return a header for a *size*-byte box whose content libav skips unread."""
    if size < _BOX_HEADER_BYTES:
        msg = f"a {size}-byte gap between extracted boxes cannot hold a box header"
        raise BoxLayoutUnsupportedError(msg)
    if size < 1 << 32:
        return struct.pack(">I4s", size, box_type)
    return struct.pack(">I4sQ", 1, box_type, size)


class _SparseFile(io.RawIOBase):
    """A read-only file of *size* bytes holding *segments* at their offsets and zeros elsewhere."""

    def __init__(self, size: int, segments: list[tuple[int, bytes]]) -> None:
        super().__init__()
        self._size = size
        self._segments = segments
        self._starts = [start for start, _ in segments]
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._position, io.SEEK_END: self._size}[whence]
        self._position = max(base + offset, 0)
        return self._position

    def readinto(self, buffer: Any) -> int:  # noqa: ANN401 -- io.RawIOBase's own writable-buffer protocol
        view = memoryview(buffer).cast("B")
        start = self._position
        end = min(start + len(view), self._size)
        if end <= start:
            return 0
        filled = start
        index = max(bisect.bisect_right(self._starts, start) - 1, 0)
        while filled < end:
            segment_start, data = self._segments[index] if index < len(self._segments) else (end, b"")
            segment_end = segment_start + len(data)
            if segment_end <= filled:
                index += 1
                continue
            zeros_end = min(max(segment_start, filled), end)
            while filled < zeros_end:
                step = min(zeros_end - filled, len(_ZEROS))
                view[filled - start : filled - start + step] = _ZEROS[:step]
                filled += step
            copy_end = min(segment_end, end)
            if filled < copy_end:
                view[filled - start : copy_end - start] = data[filled - segment_start : copy_end - segment_start]
                filled = copy_end
            index += 1
        self._position = end
        return end - start


def boxes_only_stream(
    file_size: int,
    moov_bytes: bytes,
    moof_rows: Sequence[tuple[int, bytes]] = (),
    *,
    ftyp_bytes: bytes | None = None,
) -> io.BufferedReader:
    """Lay out one video's extracted boxes as a seekable stream the size of its source.

    libav takes every sample's decode time, composition offset, byte range, and duration from
    the boxes, and reads a sample's media only to return it as a packet, so this stream demuxes
    to the source's packet index without reading the source.

    ``ftyp``, when given, goes at offset 0 and ``moov`` right after it; otherwise ``moov`` goes at
    offset 0. Extraction keeps no record of where they sat, and no sample table refers to them.
    Each ``moof`` goes at its source offset, which its ``trun`` data offsets may be relative to.
    A ``free`` box spans the gap after ``moov`` and an ``mdat`` box the gap after each ``moof``,
    so libav steps from box to box as it would through the source.

    Args:
        file_size: the source file's size, so sample byte ranges stay within the same bounds.
        moov_bytes: exactly one ``moov`` box, header included.
        moof_rows: every ``(moof_start, moof_bytes)`` pair, in file order; empty if nonfragmented.
        ftyp_bytes: exactly one ``ftyp`` box, header included, or ``None`` to lay out none.

    Raises:
        BoxLayoutUnsupportedError: ``ftyp`` and ``moov`` do not fit before the first ``moof``, the
            ``moof`` boxes overlap or run past the end of the file, or a gap is too small for a box header.

    """
    segments = [(0, moov_bytes)] if ftyp_bytes is None else [(0, ftyp_bytes), (len(ftyp_bytes), moov_bytes)]
    leading_end = segments[-1][0] + len(moov_bytes)
    boundaries = [start for start, _ in moof_rows] + [file_size]
    if boundaries[0] < leading_end:
        leading = "moov" if ftyp_bytes is None else "ftyp and moov"
        msg = f"{leading} ({leading_end} bytes) cannot fit before the first moof at offset {boundaries[0]}"
        raise BoxLayoutUnsupportedError(msg)
    if gap := boundaries[0] - leading_end:
        segments.append((leading_end, _filler_box_header(b"free", gap)))
    for (start, moof_bytes), next_start in zip(moof_rows, boundaries[1:], strict=True):
        gap = next_start - start - len(moof_bytes)
        if gap < 0:
            msg = f"moof at offset {start} ({len(moof_bytes)} bytes) overlaps the next box at offset {next_start}"
            raise BoxLayoutUnsupportedError(msg)
        segments.append((start, moof_bytes))
        if gap:
            segments.append((start + len(moof_bytes), _filler_box_header(b"mdat", gap)))
    return io.BufferedReader(_SparseFile(file_size, segments))
