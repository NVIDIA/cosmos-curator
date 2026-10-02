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

"""Generic ISO-BMFF box walking -- header parsing, child walking, path descent, and locating a video track."""

import struct

from cosmos_curator.core.sensors.mp4.errors import UnindexableVideoError

#: size (4) + type (4) -- the ordinary case. See EXTENDED_SIZE_MARKER for the 64-bit form.
STANDARD_HEADER_BYTES = 8
#: size (4) + type (4) + largesize (8) -- the 64-bit form, used when EXTENDED_SIZE_MARKER is read.
EXTENDED_HEADER_BYTES = 16
EXTENDED_SIZE_MARKER = 1
TO_END_MARKER = 0

#: Every ``stsd`` video sample entry fourcc this package recognizes, mapped to the codec name a
#: reader should report for it.
VIDEO_FOURCC_CODEC_NAMES = {
    "avc1": "h264",
    "avc3": "h264",
    "hev1": "hevc",
    "hvc1": "hevc",
}
_VIDEO_SAMPLE_ENTRY_TYPES = frozenset(VIDEO_FOURCC_CODEC_NAMES)
_UUID_EXTENSION_TYPE_BYTES = 16


class BoxNotFoundError(UnindexableVideoError):
    """A required box does not exist within the searched range."""


class AmbiguousVideoTrackError(UnindexableVideoError):
    """More than one ``trak`` under ``moov`` has a video-codec ``stsd`` entry.

    Raised instead of silently indexing the first match; a file that genuinely carries more
    than one video track needs a caller-specific policy this generic primitive can't supply.
    """


class MalformedBoxHeaderError(UnindexableVideoError):
    """A box declares a total size smaller than its own header.

    Raised instead of truncating the walk -- undetected, this would silently misalign every box
    read after the malformed one and publish a wrong-but-plausible :class:`~.moov_track.VideoTrack`
    with no signal anything was cut short.
    """


def read_box_header(buf: bytes, offset: int, end: int) -> tuple[str, int, int]:
    """Return ``(box_type, box_size, header_len)`` for the box starting at *offset*.

    ``box_size`` is the box's total size, header included. Callers must have at least
    ``EXTENDED_HEADER_BYTES`` buffered whenever the box might use the extended-size form --
    underbuffering raises an uncaught ``struct.error``, not :class:`MalformedBoxHeaderError`.

    Raises:
        MalformedBoxHeaderError: see that class's own docstring.

    """
    size32 = struct.unpack_from(">I", buf, offset)[0]
    box_type = buf[offset + 4 : offset + 8].decode("ascii", errors="replace")
    if size32 == EXTENDED_SIZE_MARKER:
        size64 = struct.unpack_from(">Q", buf, offset + 8)[0]
        if size64 < EXTENDED_HEADER_BYTES:
            msg = f"box {box_type!r} at offset {offset}: extended size {size64} is smaller than its own header"
            raise MalformedBoxHeaderError(msg)
        return box_type, size64, EXTENDED_HEADER_BYTES
    if size32 == TO_END_MARKER:
        return box_type, end - offset, STANDARD_HEADER_BYTES
    if size32 < STANDARD_HEADER_BYTES:
        msg = f"box {box_type!r} at offset {offset}: size {size32} is smaller than its own header"
        raise MalformedBoxHeaderError(msg)
    return box_type, size32, STANDARD_HEADER_BYTES


def iter_child_boxes(buf: bytes, start: int, end: int) -> list[tuple[str, int, int]]:
    """Return ``[(box_type, content_start, content_end), ...]`` for one level of children.

    Raises:
        MalformedBoxHeaderError: a child's declared size extends past *end* -- downstream
            parsers trust these bounds, so an unbounded child could read into whatever
            follows the parent.

    """
    children: list[tuple[str, int, int]] = []
    offset = start
    while offset < end:
        if end - offset < STANDARD_HEADER_BYTES:
            break  # trailing padding, not a malformed box
        box_type, box_size, header_len = read_box_header(buf, offset, end)
        if box_size <= 0:
            break
        content_end = offset + box_size
        if content_end > end:
            msg = f"box {box_type!r} at offset {offset}: declared size {box_size} extends past parent end {end}"
            raise MalformedBoxHeaderError(msg)
        children.append((box_type, offset + header_len, content_end))
        offset += box_size
    return children


def find_box(buf: bytes, path: tuple[str, ...], start: int, end: int) -> tuple[int, int]:
    """Descend *path* (e.g. ``("trak", "mdia", "mdhd")``) from one level of children.

    Raises:
        BoxNotFoundError: any segment of *path* is missing.

    """
    content_start, content_end = start, end
    for i, name in enumerate(path):
        children = iter_child_boxes(buf, content_start, content_end)
        match = next((c for c in children if c[0] == name), None)
        if match is None:
            msg = f"{'/'.join(path[: i + 1])} not found"
            raise BoxNotFoundError(msg)
        _, content_start, content_end = match
    return content_start, content_end


def find_all_boxes(buf: bytes, box_type: str, start: int, end: int) -> list[tuple[int, int]]:
    """Return every top-level child under [*start*, *end*) matching *box_type*."""
    return [(cs, ce) for t, cs, ce in iter_child_boxes(buf, start, end) if t == box_type]


def find_box_bytes(buf: bytes, box_type: str, start: int, end: int) -> bytes | None:
    """Return one child box's own bytes (header included) within [*start*, *end*), or None if absent."""
    offset = start
    while offset < end:
        found_type, box_size, _header_len = read_box_header(buf, offset, end)
        if box_size <= 0:
            return None
        if found_type == box_type:
            return bytes(buf[offset : offset + box_size])
        offset += box_size
    return None


def uuid_extension_boxes(buf: bytes, start: int, end: int) -> tuple[tuple[bytes, bytes], ...]:
    """Return every ``uuid`` box's own ``(extension_type, payload)`` pair within [*start*, *end*), verbatim."""
    return tuple(
        (bytes(buf[cs : cs + _UUID_EXTENSION_TYPE_BYTES]), bytes(buf[cs + _UUID_EXTENSION_TYPE_BYTES : ce]))
        for cs, ce in find_all_boxes(buf, "uuid", start, end)
    )


def _is_video_sample_entry(buf: bytes, stsd_start: int, stsd_end: int) -> str | None:
    """Return the sample entry's fourcc if ``stsd``'s first entry is a video codec, else None."""
    # FullBox header (4B) + entry_count (4B); the first SampleEntry starts right after.
    if stsd_end - stsd_start < STANDARD_HEADER_BYTES:
        return None
    entry_start = stsd_start + STANDARD_HEADER_BYTES
    if entry_start + 8 > stsd_end:
        return None
    fourcc = buf[entry_start + 4 : entry_start + 8].decode("ascii", errors="replace")
    return fourcc if fourcc in _VIDEO_SAMPLE_ENTRY_TYPES else None


def find_video_trak(buf: bytes, moov_start: int, moov_end: int) -> tuple[int, int, int, int, int, int, str]:
    """Return ``(trak_start, trak_end, stbl_start, stbl_end, stsd_start, stsd_end, sample_entry_fourcc)``.

    Raises:
        BoxNotFoundError: no ``trak`` in ``moov`` has a video sample entry.
        AmbiguousVideoTrackError: more than one ``trak`` in ``moov`` has a
            video sample entry.

    """
    matches: list[tuple[int, int, int, int, int, int, str]] = []
    for trak_start, trak_end in find_all_boxes(buf, "trak", moov_start, moov_end):
        try:
            stbl_start, stbl_end = find_box(buf, ("mdia", "minf", "stbl"), trak_start, trak_end)
            stsd_start, stsd_end = find_box(buf, ("stsd",), stbl_start, stbl_end)
        except BoxNotFoundError:
            continue
        fourcc = _is_video_sample_entry(buf, stsd_start, stsd_end)
        if fourcc is not None:
            matches.append((trak_start, trak_end, stbl_start, stbl_end, stsd_start, stsd_end, fourcc))
    if not matches:
        msg = "no trak with a video sample entry found under moov"
        raise BoxNotFoundError(msg)
    if len(matches) > 1:
        msg = f"{len(matches)} traks with a video sample entry found under moov, expected exactly one"
        raise AmbiguousVideoTrackError(msg)
    return matches[0]
