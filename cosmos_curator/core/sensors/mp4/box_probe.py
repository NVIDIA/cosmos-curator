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

"""Index a video's extracted ``moov``/``moof`` boxes with PyAV.

Sample byte offsets remain relative to the source file. Media bytes are replaced
with zeros; sync flags come from libav's header index.
"""

import bisect
import io
import struct
from collections.abc import Sequence
from typing import Any

import attrs
import av
import numpy as np
import numpy.typing as npt
from av.container import InputContainer

from cosmos_curator.core.sensors.data.mp4 import Mp4Metadata
from cosmos_curator.core.sensors.data.video import VideoProbe
from cosmos_curator.core.sensors.mp4.box_primitives import (
    BoxNotFoundError,
    MalformedBoxHeaderError,
    find_all_boxes,
    find_box,
    find_box_bytes,
    read_box_header,
)
from cosmos_curator.core.sensors.mp4.errors import UnindexableVideoError
from cosmos_curator.core.sensors.mp4.moov_track import EmptySampleTableError, MediaHeader, trak_track_id
from cosmos_curator.core.sensors.types.types import VideoIndexCreationMethod
from cosmos_curator.core.sensors.utils.video import (
    HeaderIndexUnavailableError,
    IncompletePacketError,
    probe_open_video_stream,
)

_BOX_HEADER_BYTES = 8
_ZEROS = bytes(1 << 20)
#: libav reads ``mdhd``'s unsigned timescale as a signed int and replaces one it cannot hold, 0 or
#: 2**31 and above, with 1, leaving the ticks counted in the original.
_LIBAV_TIMESCALE_LIMIT = 1 << 31


class BoxLayoutUnsupportedError(UnindexableVideoError):
    """The extracted boxes cannot be laid out as a file libav walks the way it walks the source."""


class InvalidFrameByteRangeError(UnindexableVideoError):
    """An indexed frame's byte range lies outside the source media."""


class BoxProbeReadError(UnindexableVideoError):
    """libav cannot open or demux the extracted boxes."""


class BoxProbeMismatchError(UnindexableVideoError):
    """libav's header index and its demux of the same boxes describe different samples."""


class UnsupportedTimescaleError(UnindexableVideoError):
    """The ``mdhd`` timescale is one libav replaces with 1 while keeping the ticks counted in it."""


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
    file_size: int, moov_bytes: bytes, moof_rows: Sequence[tuple[int, bytes]] = ()
) -> io.BufferedReader:
    """Lay out one video's extracted boxes as a seekable stream the size of its source.

    libav takes every sample's decode time, composition offset, byte range, and duration from
    the boxes, and reads a sample's media only to return it as a packet, so this stream demuxes
    to the source's packet index without reading the source.

    ``moov`` goes at offset 0: extraction keeps no record of where it sat, and no sample table
    refers to it. Each ``moof`` goes at its source offset, which its ``trun`` data offsets
    may be relative to. A ``free`` box spans the gap after ``moov`` and an ``mdat`` box the gap
    after each ``moof``, so libav steps from box to box as it would through the source.

    Args:
        file_size: the source file's size, so sample byte ranges stay within the same bounds.
        moov_bytes: exactly one ``moov`` box, header included.
        moof_rows: every ``(moof_start, moof_bytes)`` pair, in file order; empty if nonfragmented.

    Raises:
        BoxLayoutUnsupportedError: ``moov`` does not fit before the first ``moof``, the ``moof``
            boxes overlap or run past the end of the file, or a gap is too small for a box header.

    """
    segments = [(0, moov_bytes)]
    boundaries = [start for start, _ in moof_rows] + [file_size]
    if boundaries[0] < len(moov_bytes):
        msg = f"moov ({len(moov_bytes)} bytes) does not fit before the first moof at offset {boundaries[0]}"
        raise BoxLayoutUnsupportedError(msg)
    if gap := boundaries[0] - len(moov_bytes):
        segments.append((len(moov_bytes), _filler_box_header(b"free", gap)))
    for (start, moof_bytes), next_start in zip(moof_rows, boundaries[1:], strict=True):
        gap = next_start - start - len(moof_bytes)
        if gap < 0:
            msg = f"moof at offset {start} ({len(moof_bytes)} bytes) overlaps the next box at offset {next_start}"
            raise BoxLayoutUnsupportedError(msg)
        segments.append((start, moof_bytes))
        if gap:
            segments.append((start + len(moof_bytes), _filler_box_header(b"mdat", gap)))
    return io.BufferedReader(_SparseFile(file_size, segments))


def check_frame_byte_ranges(offset: npt.NDArray[np.int64], size: npt.NDArray[np.int64], file_size: int) -> None:
    """Raise unless every frame's byte range ``[offset, offset + size)`` lies within the source.

    Raises:
        InvalidFrameByteRangeError: naming the first frame whose range does not.

    """
    # Subtract from the file size to avoid overflowing int64 when a range extends past EOF.
    remaining = file_size - offset.clip(min=0)
    invalid = (offset < 0) | (size < 0) | (size > remaining)
    if invalid.any():
        index = int(invalid.argmax())
        start, length = int(offset[index]), int(size[index])
        msg = f"Frame {index} has byte range [{start}, {start + length}) outside media size {file_size}"
        raise InvalidFrameByteRangeError(msg)


@attrs.define(frozen=True)
class _TrakBoxes:
    timescale: int
    elst_bytes: bytes | None


def _trak_boxes(moov_bytes: bytes, track_id: int) -> _TrakBoxes:
    """Return the ``mdhd`` timescale and ``edts/elst`` box of the ``trak`` whose ``tkhd`` has *track_id*.

    Raises:
        BoxProbeMismatchError: no ``trak``, or more than one, has *track_id*.
        BoxNotFoundError: that ``trak`` has no ``mdia/mdhd``.
        MalformedBoxHeaderError: *moov_bytes* is truncated or corrupt.

    """
    try:
        _box_type, box_size, header_len = read_box_header(moov_bytes, 0, len(moov_bytes))
        matching = [
            (trak_start, trak_end)
            for trak_start, trak_end in find_all_boxes(moov_bytes, "trak", header_len, box_size)
            if trak_track_id(moov_bytes, trak_start, trak_end) == track_id
        ]
        if len(matching) != 1:
            which = "no trak has" if not matching else f"{len(matching)} traks share"
            msg = f"libav lists a video stream for track {track_id}, but {which} that track ID"
            raise BoxProbeMismatchError(msg)
        [(trak_start, trak_end)] = matching
        mdhd_start, mdhd_end = find_box(moov_bytes, ("mdia", "mdhd"), trak_start, trak_end)
        try:
            edts_start, edts_end = find_box(moov_bytes, ("edts",), trak_start, trak_end)
        except BoxNotFoundError:
            elst_bytes = None
        else:
            elst_bytes = find_box_bytes(moov_bytes, "elst", edts_start, edts_end)
        return _TrakBoxes(MediaHeader.parse(moov_bytes, mdhd_start, mdhd_end).timescale, elst_bytes)
    except (struct.error, IndexError) as exc:
        msg = f"moov buffer ({len(moov_bytes)} byte(s)) ran out mid-parse -- truncated or corrupt: {exc}"
        raise MalformedBoxHeaderError(msg) from exc


@attrs.define(frozen=True)
class BoxIndex:
    """One video's samples and stream metadata, as PyAV reads them from its extracted boxes."""

    probe: VideoProbe
    metadata: Mp4Metadata


def _index_best_video_stream(container: InputContainer, file_size: int, moov_bytes: bytes) -> BoxIndex:
    """Index and describe the video stream libav ranks best in *container*."""
    stream = container.streams.best("video")
    if stream is None:
        msg = "libav lists no video stream in the extracted moov"
        raise BoxNotFoundError(msg)
    position = [video.index for video in container.streams.video].index(stream.index)
    trak = _trak_boxes(moov_bytes, stream.id)
    if not 0 < trak.timescale < _LIBAV_TIMESCALE_LIMIT:
        msg = f"mdhd timescale {trak.timescale} is one libav replaces with 1, keeping the ticks counted in it"
        raise UnsupportedTimescaleError(msg)
    try:
        header = probe_open_video_stream(container, position, VideoIndexCreationMethod.FROM_HEADER)
    except HeaderIndexUnavailableError as exc:
        # libav builds no header index for an empty track, so only a demux tells emptiness apart.
        if not len(probe_open_video_stream(container, position, VideoIndexCreationMethod.FULL_DEMUX)):
            msg = "libav reads no video samples from the extracted boxes"
            raise EmptySampleTableError(msg) from exc
        msg = "libav built no header index from the extracted boxes, so it reports no sync flags"
        raise BoxProbeMismatchError(msg) from exc
    check_frame_byte_ranges(header.offset, header.size, file_size)
    demux = probe_open_video_stream(container, position, VideoIndexCreationMethod.FULL_DEMUX)
    for name in ("dts", "offset", "size"):
        if not np.array_equal(getattr(header, name), getattr(demux, name)):
            msg = f"libav's header index and demux of the extracted boxes disagree on {name}"
            raise BoxProbeMismatchError(msg)
    probe = attrs.evolve(demux, is_keyframe=header.is_keyframe)
    metadata = Mp4Metadata.from_video_probe(
        probe, encoder_tag=stream.metadata.get("encoder"), elst_bytes=trak.elst_bytes
    )
    return BoxIndex(probe=probe, metadata=metadata)


def index_from_boxes(file_size: int, moov_bytes: bytes, moof_rows: Sequence[tuple[int, bytes]] = ()) -> BoxIndex:
    """Index and describe one video's best video stream with PyAV from its extracted boxes.

    Every sample field comes from a full demux of :func:`boxes_only_stream`, apart from
    ``is_keyframe``: libav also derives a demuxed packet's key flag from its payload, so
    zero-filled media demuxes with no keyframes. Sync flags come from libav's header index of
    the same boxes, built from ``stss`` or ``trun`` sample flags. Byte ranges are checked
    against the header index before the demux, which would drop or truncate out-of-file samples.
    The header index and the demux share one open of the boxes, since libav parses every
    ``moof`` while opening them.

    The metadata is
    :meth:`~cosmos_curator.core.sensors.data.mp4.Mp4Metadata.from_video_probe`'s, apart from the
    stream's ``edts/elst`` box, copied verbatim from its ``trak``: libav is opened without reading
    the edit list, so that decode times are the stored ones.

    Raises:
        BoxLayoutUnsupportedError: see :func:`boxes_only_stream`.
        BoxNotFoundError: libav lists no video stream, or its ``trak`` has no ``mdhd``.
        UnsupportedTimescaleError: the stream's ``mdhd`` timescale is 0 or 2**31 and above.
        EmptySampleTableError: libav reads no samples from the boxes.
        InvalidFrameByteRangeError: a sample's byte range lies outside the source.
        BoxProbeMismatchError: no ``trak`` has the stream's track ID, or libav's header index is
            unavailable or disagrees with the demux on any sample's decode time or byte range.
        BoxProbeReadError: libav cannot open or demux the boxes, or a demuxed packet lacks dts,
            pts, or byte position.
        MalformedBoxHeaderError: *moov_bytes* is truncated or corrupt.

    """
    boxes = boxes_only_stream(file_size, moov_bytes, moof_rows)
    try:
        with av.open(boxes, mode="r", options={"ignore_editlist": "1"}) as container:
            return _index_best_video_stream(container, file_size, moov_bytes)
    except (av.error.FFmpegError, IncompletePacketError) as exc:
        msg = f"libav cannot index the extracted boxes: {exc}"
        raise BoxProbeReadError(msg) from exc
