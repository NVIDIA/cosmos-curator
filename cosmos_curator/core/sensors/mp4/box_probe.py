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

import struct
from collections.abc import Sequence

import attrs
import av
import numpy as np
import numpy.typing as npt
from av.container import InputContainer

from cosmos_curator.core.sensors.data.mp4 import Mp4Metadata
from cosmos_curator.core.sensors.data.video import VideoProbe
from cosmos_curator.core.sensors.mp4.box_layout import boxes_only_stream
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
    mp4_track_id,
    probe_open_video_stream,
)

#: libav reads ``mdhd``'s unsigned timescale as a signed int and replaces one it cannot hold, 0 or
#: 2**31 and above, with 1, leaving the ticks counted in the original.
_LIBAV_TIMESCALE_LIMIT = 1 << 31
#: ``stsd``'s version, flags, and entry count (8), then its first VisualSampleEntry's box header (8),
#: reserved and data reference index (8), and pre_defined/reserved (16), before width and height.
_STSD_FRAME_SIZE_OFFSET = 40


class InvalidFrameByteRangeError(UnindexableVideoError):
    """An indexed frame's byte range lies outside the source media."""


class BoxProbeReadError(UnindexableVideoError):
    """libav cannot open or demux the extracted boxes."""


class BoxProbeMismatchError(UnindexableVideoError):
    """libav's header index and its demux of the same boxes describe different samples."""


class UnsupportedTimescaleError(UnindexableVideoError):
    """The ``mdhd`` timescale is one libav replaces with 1 while keeping the ticks counted in it."""


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
    frame_size: tuple[int, int] | None


def _sample_entry_frame_size(moov_bytes: bytes, trak_start: int, trak_end: int) -> tuple[int, int] | None:
    """Return the width and height of a ``trak``'s first ``stsd`` sample entry, or None if unreadable."""
    try:
        stsd_start, stsd_end = find_box(moov_bytes, ("mdia", "minf", "stbl", "stsd"), trak_start, trak_end)
    except (BoxNotFoundError, MalformedBoxHeaderError, struct.error):
        return None
    if stsd_end - stsd_start < _STSD_FRAME_SIZE_OFFSET + 4:
        return None
    width, height = struct.unpack_from(">HH", moov_bytes, stsd_start + _STSD_FRAME_SIZE_OFFSET)
    return width, height


def _trak_boxes(moov_bytes: bytes, track_id: int) -> _TrakBoxes:
    """Return the ``mdhd`` timescale and ``edts/elst`` box of the ``trak`` whose ``tkhd`` has *track_id*.

    Also returns that ``trak``'s :func:`_sample_entry_frame_size`.

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
        return _TrakBoxes(
            MediaHeader.parse(moov_bytes, mdhd_start, mdhd_end).timescale,
            elst_bytes,
            _sample_entry_frame_size(moov_bytes, trak_start, trak_end),
        )
    except (struct.error, IndexError) as exc:
        msg = f"moov buffer ({len(moov_bytes)} byte(s)) ran out mid-parse -- truncated or corrupt: {exc}"
        raise MalformedBoxHeaderError(msg) from exc


@attrs.define(frozen=True)
class BoxIndex:
    """One video's samples and stream metadata, as PyAV reads them from its extracted boxes."""

    probe: VideoProbe
    metadata: Mp4Metadata


def _index_video_stream(
    container: InputContainer, file_size: int, moov_bytes: bytes, ftyp_bytes: bytes | None, stream_idx: int | None
) -> BoxIndex:
    """Index the requested video stream, or libav's preferred stream when no index is given."""
    if stream_idx is not None and not 0 <= stream_idx < len(container.streams.video):
        msg = f"video stream index {stream_idx} is out of range"
        raise BoxNotFoundError(msg)
    stream = container.streams.best("video") if stream_idx is None else container.streams.video[stream_idx]
    if stream is None:
        msg = "libav lists no video stream in the extracted moov"
        raise BoxNotFoundError(msg)
    position = [video.index for video in container.streams.video].index(stream.index)
    track_id = mp4_track_id(stream)
    trak = _trak_boxes(moov_bytes, track_id)
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
    width, height = demux.width, demux.height
    if not (width and height) and trak.frame_size is not None:
        width, height = trak.frame_size
    probe = attrs.evolve(demux, is_keyframe=header.is_keyframe, width=width, height=height)
    metadata = Mp4Metadata.from_video_probe(
        probe,
        track_id=track_id,
        encoder_tag=stream.metadata.get("encoder"),
        elst_bytes=trak.elst_bytes,
        ftyp_bytes=ftyp_bytes,
    )
    return BoxIndex(probe=probe, metadata=metadata)


def index_from_boxes(
    file_size: int,
    moov_bytes: bytes,
    moof_rows: Sequence[tuple[int, bytes]] = (),
    *,
    ftyp_bytes: bytes | None = None,
    stream_idx: int | None = None,
) -> BoxIndex:
    """Index and describe one video stream with PyAV from its extracted boxes.

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
    the edit list, so that decode times are the stored ones. When libav reports a width or height
    of 0, both come from the first sample entry of that ``trak``'s ``stsd``, if it has one.
    ``ftyp_bytes`` is laid out ahead of ``moov`` and recorded in the metadata verbatim.

    ``stream_idx`` selects a stream by video-stream position. When omitted, libav's
    preferred video stream is selected.

    Raises:
        BoxLayoutUnsupportedError: see :func:`boxes_only_stream`.
        BoxNotFoundError: the video stream index is out of range, libav lists no video stream,
            or its ``trak`` has no ``mdhd``.
        UnsupportedTimescaleError: the stream's ``mdhd`` timescale is 0 or 2**31 and above.
        EmptySampleTableError: libav reads no samples from the boxes.
        InvalidFrameByteRangeError: a sample's byte range lies outside the source.
        BoxProbeMismatchError: no ``trak`` has the stream's track ID, or libav's header index is
            unavailable or disagrees with the demux on any sample's decode time or byte range.
        BoxProbeReadError: libav cannot open or demux the boxes, or a demuxed packet lacks dts,
            pts, or byte position.
        MalformedBoxHeaderError: *moov_bytes* is truncated or corrupt.

    """
    boxes = boxes_only_stream(file_size, moov_bytes, moof_rows, ftyp_bytes=ftyp_bytes)
    try:
        with av.open(boxes, mode="r", options={"ignore_editlist": "1"}) as container:
            return _index_video_stream(container, file_size, moov_bytes, ftyp_bytes, stream_idx)
    except (av.error.FFmpegError, IncompletePacketError) as exc:
        msg = f"libav cannot index the extracted boxes: {exc}"
        raise BoxProbeReadError(msg) from exc
