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

"""The non-fragmented combiner: a faithful, struct-of-arrays decode of one ``moov``'s pooled ``stbl``.

Box layout this module reads (everything below lives inside one ``moov``, under ``stbl``)::

    moov
    └── trak
        └── mdia
            └── minf
                └── stbl         "sample table" -- everything about the samples themselves
                    ├── stts      decode time-to-sample (run-length deltas) -> dts_ticks
                    ├── ctts      composition time-to-sample (optional) -> CttsBox
                    ├── stss      sync sample table (optional) -> is_keyframe
                    ├── stsz      sample sizes -> size
                    ├── stsc      sample-to-chunk grouping   -+
                    └── stco/co64 chunk byte offsets          +-> offset
                                    (32-bit vs 64-bit variant) -+
"""

import struct
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.mp4.box_primitives import (
    BoxNotFoundError,
    MalformedBoxHeaderError,
    read_box_header,
)
from cosmos_curator.core.sensors.mp4.box_traversal import BoxScope, BoxTraversal, UuidHandler
from cosmos_curator.core.sensors.mp4.errors import UnindexableVideoError
from cosmos_curator.core.sensors.mp4.moov_track import (
    CttsBox,
    MediaHeader,
    MovieHeader,
    StsdBox,
    TrackHeader,
    VideoTrack,
    video_track_scopes,
)


class MalformedCompositionOffsetError(UnindexableVideoError):
    """``ctts``'s total sample count doesn't match ``stts``'s.

    ISO/IEC 14496-12 8.6.1.3.3 defines ``CT(n) = DT(n) + CTTS(n)`` for
    sample ``n`` -- every sample in the track, the same domain ``stts``
    itself defines. A ``ctts`` whose run-length table covers a different
    number of samples than ``stts`` has no well-defined offset for at least
    one sample the formula requires one for, so the file is non-conformant.
    """


class MalformedSampleTableError(UnindexableVideoError):
    """``stsz``'s sample count or ``stsc``/``stco``'s expansion doesn't match ``stts``'s.

    ``stts`` defines the sample domain every other per-sample table
    (``stsz``, and ``stsc``/``stco`` after expansion) must cover exactly --
    a ``VideoTrack`` with ``dts_ticks``, ``size``, and ``offset`` of
    disagreeing lengths has no well-defined byte range for at least one
    sample, so the file is non-conformant.
    """


def parse_stts(buf: bytes, start: int, end: int) -> list[tuple[int, int]]:  # noqa: ARG001 -- end kept for symmetry with sibling parsers
    """Parse ``stts`` into ``[(sample_count, sample_delta), ...]`` run-length pairs."""
    entry_count = struct.unpack_from(">I", buf, start + 4)[0]
    pairs = struct.unpack_from(f">{2 * entry_count}I", buf, start + 8)
    return list(zip(pairs[0::2], pairs[1::2], strict=True))


def parse_ctts(buf: bytes, start: int, end: int) -> list[tuple[int, int]]:  # noqa: ARG001 -- end kept for symmetry with sibling parsers
    """Parse ``ctts`` into ``[(sample_count, sample_offset), ...]`` run-length pairs.

    ``sample_offset`` is signed from version 1 on; version 0 is unsigned but
    never negative in practice, so reading both as signed ``i`` is exact either
    way -- an unsigned version-0 offset large enough to look negative as signed
    would already be a multi-gigasample file no MP4 track reaches.
    """
    entry_count = struct.unpack_from(">I", buf, start + 4)[0]
    pairs = struct.unpack_from(f">{2 * entry_count}i", buf, start + 8)
    return list(zip(pairs[0::2], pairs[1::2], strict=True))


def parse_stss(buf: bytes, start: int, end: int) -> set[int]:  # noqa: ARG001 -- end kept for symmetry with sibling parsers
    """Parse ``stss`` into the 1-indexed sample numbers that are keyframes."""
    entry_count = struct.unpack_from(">I", buf, start + 4)[0]
    return set(struct.unpack_from(f">{entry_count}I", buf, start + 8))


def parse_stsz(buf: bytes, start: int, end: int) -> list[int]:  # noqa: ARG001 -- end kept for symmetry with sibling parsers
    """Parse ``stsz`` into per-sample byte sizes."""
    sample_size, sample_count = struct.unpack_from(">II", buf, start + 4)
    if sample_size != 0:
        return [sample_size] * sample_count
    return list(struct.unpack_from(f">{sample_count}I", buf, start + 12))


def parse_stco(buf: bytes, start: int, end: int) -> list[int]:  # noqa: ARG001 -- end kept for symmetry with sibling parsers
    """Parse ``stco`` (32-bit chunk offsets)."""
    entry_count = struct.unpack_from(">I", buf, start + 4)[0]
    return list(struct.unpack_from(f">{entry_count}I", buf, start + 8))


def parse_co64(buf: bytes, start: int, end: int) -> list[int]:  # noqa: ARG001 -- end kept for symmetry with sibling parsers
    """Parse ``co64`` (64-bit chunk offsets, used once a file exceeds 4 GB)."""
    entry_count = struct.unpack_from(">I", buf, start + 4)[0]
    return list(struct.unpack_from(f">{entry_count}Q", buf, start + 8))


def parse_stsc(buf: bytes, start: int, end: int) -> list[tuple[int, int, int]]:  # noqa: ARG001
    """Parse ``stsc`` into ``[(first_chunk, samples_per_chunk, sample_description_index), ...]``."""
    entry_count = struct.unpack_from(">I", buf, start + 4)[0]
    triples = struct.unpack_from(f">{3 * entry_count}I", buf, start + 8)
    return list(zip(triples[0::3], triples[1::3], triples[2::3], strict=True))


def _expand_dts_ticks(stts: list[tuple[int, int]]) -> npt.NDArray[np.int64]:
    """Expand ``stts``'s run-length ``(sample_count, sample_delta)`` pairs into per-sample cumulative ticks.

    Vectorized per run (one ``np.arange`` per ``stts`` entry, not one Python
    step per sample) -- a run-length table typically has few entries even
    when the samples they cover number in the tens of thousands.
    """
    runs: list[npt.NDArray[np.int64]] = []
    tick = 0
    for sample_count, sample_delta in stts:
        runs.append(tick + np.arange(sample_count, dtype=np.int64) * sample_delta)
        tick += sample_count * sample_delta
    if not runs:
        return np.array([], dtype=np.int64)
    return np.concatenate(runs)


def _chunk_offsets(buf: bytes, traversal: BoxTraversal, scope: BoxScope) -> list[int]:
    try:
        start, end = traversal.find_bounds(scope, ("stco",))
        return parse_stco(buf, start, end)
    except BoxNotFoundError:
        start, end = traversal.find_bounds(scope, ("co64",))
        return parse_co64(buf, start, end)


def _expand_sample_offsets(
    stsc: list[tuple[int, int, int]], chunk_offsets: list[int], sample_sizes: list[int]
) -> npt.NDArray[np.int64]:
    """Walk ``stsc``'s run-length chunk groups -- samples within a chunk are packed contiguously, no padding."""
    offsets: list[int] = []
    sample_idx = 0
    num_chunks = len(chunk_offsets)
    for group_idx, (first_chunk, samples_per_chunk, _sample_desc_index) in enumerate(stsc):
        next_first_chunk = stsc[group_idx + 1][0] if group_idx + 1 < len(stsc) else num_chunks + 1
        for chunk_number in range(first_chunk, next_first_chunk):
            running_offset = chunk_offsets[chunk_number - 1]  # stco/co64 is 1-indexed, chunk_offsets isn't
            for _ in range(samples_per_chunk):
                offsets.append(running_offset)
                running_offset += sample_sizes[sample_idx]
                sample_idx += 1
    return np.array(offsets, dtype=np.int64)


def parse_video_track(buf: bytes, *, uuid_handlers: Sequence[UuidHandler] = ()) -> VideoTrack:
    """Parse one non-fragmented file's ``moov`` box (header included) into a faithful :class:`VideoTrack`.

    Args:
        buf: exactly one ``moov`` box, header included.
        uuid_handlers: invocation-local handlers for UUID boxes in visited scopes.

    Raises:
        BoxNotFoundError: *buf* is not a ``moov`` box, or carries no video
            ``trak``.
        MalformedBoxHeaderError: *buf* is truncated or corrupt partway through parsing.
        UuidCallbackError: a registered UUID handler failed; ``error`` retains its exception.

    """
    try:
        return _parse_video_track(buf, uuid_handlers=uuid_handlers)
    except (struct.error, IndexError) as exc:
        msg = f"moov buffer ({len(buf)} byte(s)) ran out mid-parse -- truncated or corrupt: {exc}"
        raise MalformedBoxHeaderError(msg) from exc


def _check_sample_count(actual: int, frame_count: int, source: str) -> None:
    """Raise :class:`MalformedSampleTableError` if *actual* (from *source*) disagrees with ``stts``'s *frame_count*."""
    if actual != frame_count:
        msg = f"stts defines {frame_count} sample(s) but {source} has {actual} -- non-conformant file"
        raise MalformedSampleTableError(msg)


def _parse_video_track(buf: bytes, *, uuid_handlers: Sequence[UuidHandler]) -> VideoTrack:
    box_type, box_size, header_len = read_box_header(buf, 0, len(buf))
    if box_type != "moov":
        msg = f"expected a moov box, found {box_type!r}"
        raise BoxNotFoundError(msg)
    moov_start, moov_end = header_len, box_size

    traversal = BoxTraversal(buf, uuid_handlers=uuid_handlers)
    moov_scope = BoxScope(moov_start, moov_end, ("moov",))
    trak_scope, stbl_scope, stsd_scope, fourcc = video_track_scopes(buf, moov_scope)
    # Visit the selected track's intermediate containers as well as stbl,
    # so UUID handlers registered for mdia/minf are dispatched with track context.
    traversal.find(trak_scope, ("mdia", "minf", "stbl"))
    traversal.scan_children(stbl_scope)

    mvhd_start, mvhd_end = traversal.find_bounds(moov_scope, ("mvhd",))
    movie_header = MovieHeader.parse(buf, mvhd_start, mvhd_end)
    moov_extension_boxes = traversal.extension_boxes(moov_scope)

    tkhd_start, tkhd_end = traversal.find_bounds(trak_scope, ("tkhd",))
    track_header = TrackHeader.parse(buf, tkhd_start, tkhd_end)
    try:
        elst_bytes = traversal.find_bytes(trak_scope, ("edts", "elst"))
    except BoxNotFoundError:
        elst_bytes = None

    mdhd_start, mdhd_end = traversal.find_bounds(trak_scope, ("mdia", "mdhd"))
    media_header = MediaHeader.parse(buf, mdhd_start, mdhd_end)

    sample_description = StsdBox.parse(buf, traversal, stsd_scope, fourcc)

    stts_start, stts_end = traversal.find_bounds(stbl_scope, ("stts",))
    dts_ticks = _expand_dts_ticks(parse_stts(buf, stts_start, stts_end))
    frame_count = len(dts_ticks)

    try:
        stss_start, stss_end = traversal.find_bounds(stbl_scope, ("stss",))
        keyframe_numbers = parse_stss(buf, stss_start, stss_end)
        is_keyframe = np.array([(i + 1) in keyframe_numbers for i in range(frame_count)], dtype=np.bool_)
    except BoxNotFoundError:
        is_keyframe = None  # faithful: stss absent, not defaulted to "every sample is a keyframe"

    ctts_boxes = [(box.content_start, box.end) for box in traversal.scan_children(stbl_scope) if box.box_type == "ctts"]
    if ctts_boxes:
        ctts_start, ctts_end = ctts_boxes[0]
        ctts_pairs = parse_ctts(buf, ctts_start, ctts_end)
        ctts_sample_count = sum(count for count, _offset in ctts_pairs)
        if ctts_sample_count != frame_count:
            msg = f"ctts covers {ctts_sample_count} sample(s) but stts defines {frame_count} -- non-conformant file"
            raise MalformedCompositionOffsetError(msg)
        composition_offsets = CttsBox(
            sample_counts=np.array([count for count, _offset in ctts_pairs], dtype=np.int64),
            sample_offsets=np.array([offset for _count, offset in ctts_pairs], dtype=np.int64),
        )
    else:
        composition_offsets = None

    stsz_start, stsz_end = traversal.find_bounds(stbl_scope, ("stsz",))
    sample_sizes = parse_stsz(buf, stsz_start, stsz_end)
    _check_sample_count(len(sample_sizes), frame_count, "stsz")

    stsc_start, stsc_end = traversal.find_bounds(stbl_scope, ("stsc",))
    stsc = parse_stsc(buf, stsc_start, stsc_end)
    chunk_offsets = _chunk_offsets(buf, traversal, stbl_scope)
    offsets = _expand_sample_offsets(stsc, chunk_offsets, sample_sizes)
    _check_sample_count(len(offsets), frame_count, "stsc/stco expansion")

    return VideoTrack(
        timescale=media_header.timescale,
        duration=media_header.duration,
        sample_description=sample_description,
        movie_header=movie_header,
        track_header=track_header,
        moov_extension_boxes=moov_extension_boxes,
        elst_bytes=elst_bytes,
        dts_ticks=dts_ticks,
        offset=offsets,
        size=np.array(sample_sizes, dtype=np.int64),
        is_keyframe=is_keyframe,
        composition_offsets=composition_offsets,
    )
