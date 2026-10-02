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

"""Decode a fragmented MP4's ``moov`` and every ``moof``/``traf``/``trun`` into :class:`VideoTrack`."""

import struct

import numpy as np

from cosmos_curator.core.sensors.mp4.box_primitives import (
    BoxNotFoundError,
    MalformedBoxHeaderError,
    find_box,
    find_box_bytes,
    find_video_trak,
    read_box_header,
    uuid_extension_boxes,
)
from cosmos_curator.core.sensors.mp4.errors import UnindexableVideoError
from cosmos_curator.core.sensors.mp4.moov_track import (
    MediaHeader,
    MovieHeader,
    StsdBox,
    TrackHeader,
    VideoTrack,
)

# tfhd flags (ISO/IEC 14496-12 §8.8.7)
_TFHD_BASE_DATA_OFFSET_PRESENT = 0x000001
_TFHD_SAMPLE_DESCRIPTION_INDEX_PRESENT = 0x000002
_TFHD_DEFAULT_SAMPLE_DURATION_PRESENT = 0x000008
_TFHD_DEFAULT_SAMPLE_SIZE_PRESENT = 0x000010
_TFHD_DEFAULT_SAMPLE_FLAGS_PRESENT = 0x000020
_TFHD_DEFAULT_BASE_IS_MOOF = 0x020000

# trun flags (ISO/IEC 14496-12 §8.8.8)
_TRUN_DATA_OFFSET_PRESENT = 0x000001
_TRUN_FIRST_SAMPLE_FLAGS_PRESENT = 0x000004
#: Read but unused -- dts_ticks comes from each fragment's own tfdt, not accumulated trun
#: durations. Still named and checked, to advance past this field correctly for whatever
#: follows it when the flag is set.
_TRUN_SAMPLE_DURATION_PRESENT = 0x000100
_TRUN_SAMPLE_SIZE_PRESENT = 0x000200
_TRUN_SAMPLE_FLAGS_PRESENT = 0x000400
_TRUN_SAMPLE_COMPOSITION_TIME_OFFSETS_PRESENT = 0x000800

#: Within a sample_flags word, whether this sample is *not* a sync sample (i.e. not a keyframe).
_SAMPLE_IS_NON_SYNC_SAMPLE_BIT = 0x00010000


class FragmentLayoutUnsupportedError(UnindexableVideoError):
    """A fragment's layout doesn't match what every real production file examined so far looks like."""


class CompositionOffsetUnsupportedError(UnindexableVideoError):
    """A sample carries a nonzero composition time offset -- never observed in real production data."""


def _parse_tfhd(buf: bytes, start: int) -> dict[str, int | bool | None]:
    """Parse ``tfhd``'s content into the fields a fragment's sample may need."""
    flags = int.from_bytes(buf[start + 1 : start + 4], "big")
    pos = start + 4  # skip version(1) + flags(3)
    pos += 4  # track_ID, unused here -- a camera recording has exactly one track
    if flags & _TFHD_BASE_DATA_OFFSET_PRESENT:
        pos += 8
    sample_description_index = None
    if flags & _TFHD_SAMPLE_DESCRIPTION_INDEX_PRESENT:
        sample_description_index = struct.unpack_from(">I", buf, pos)[0]
        pos += 4
    default_sample_duration = None
    if flags & _TFHD_DEFAULT_SAMPLE_DURATION_PRESENT:
        default_sample_duration = struct.unpack_from(">I", buf, pos)[0]
        pos += 4
    default_sample_size = None
    if flags & _TFHD_DEFAULT_SAMPLE_SIZE_PRESENT:
        default_sample_size = struct.unpack_from(">I", buf, pos)[0]
        pos += 4
    default_sample_flags = None
    if flags & _TFHD_DEFAULT_SAMPLE_FLAGS_PRESENT:
        default_sample_flags = struct.unpack_from(">I", buf, pos)[0]
        pos += 4
    return {
        "default_base_is_moof": bool(flags & _TFHD_DEFAULT_BASE_IS_MOOF),
        "sample_description_index": sample_description_index,
        "default_sample_duration": default_sample_duration,
        "default_sample_size": default_sample_size,
        "default_sample_flags": default_sample_flags,
    }


def _parse_tfdt(buf: bytes, start: int) -> int:
    """Parse ``tfdt``'s ``baseMediaDecodeTime`` (version-0 32-bit or version-1 64-bit)."""
    version = buf[start]
    if version == 1:
        return int(struct.unpack_from(">Q", buf, start + 4)[0])
    return int(struct.unpack_from(">I", buf, start + 4)[0])


def _parse_trun(buf: bytes, start: int) -> dict[str, int | None]:
    """Parse a single-sample ``trun`` -- any other sample count is refused rather than guessed at.

    Raises:
        FragmentLayoutUnsupportedError: ``sample_count`` is not exactly 1.

    """
    flags = int.from_bytes(buf[start + 1 : start + 4], "big")
    sample_count = struct.unpack_from(">I", buf, start + 4)[0]
    if sample_count != 1:
        msg = f"trun at offset {start} carries {sample_count} samples, expected exactly 1"
        raise FragmentLayoutUnsupportedError(msg)

    pos = start + 8
    data_offset = None
    if flags & _TRUN_DATA_OFFSET_PRESENT:
        data_offset = struct.unpack_from(">i", buf, pos)[0]
        pos += 4
    first_sample_flags = None
    if flags & _TRUN_FIRST_SAMPLE_FLAGS_PRESENT:
        first_sample_flags = struct.unpack_from(">I", buf, pos)[0]
        pos += 4
    if flags & _TRUN_SAMPLE_DURATION_PRESENT:
        pos += 4
    sample_size = None
    if flags & _TRUN_SAMPLE_SIZE_PRESENT:
        sample_size = struct.unpack_from(">I", buf, pos)[0]
        pos += 4
    sample_flags = None
    if flags & _TRUN_SAMPLE_FLAGS_PRESENT:
        sample_flags = struct.unpack_from(">I", buf, pos)[0]
        pos += 4
    sample_composition_offset = None
    if flags & _TRUN_SAMPLE_COMPOSITION_TIME_OFFSETS_PRESENT:
        sample_composition_offset = struct.unpack_from(">i", buf, pos)[0]
        pos += 4
    return {
        "data_offset": data_offset,
        "sample_size": sample_size,
        "sample_flags": first_sample_flags if first_sample_flags is not None else sample_flags,
        "sample_composition_offset": sample_composition_offset,
    }


def _parse_trex_defaults(moov_buf: bytes, moov_content_start: int, moov_content_end: int) -> dict[str, int | None]:
    """Parse ``mvex``/``trex``'s fallback values, or all-``None`` if the track has no ``trex``."""
    try:
        trex_start, _trex_end = find_box(moov_buf, ("mvex", "trex"), moov_content_start, moov_content_end)
    except BoxNotFoundError:
        return {"default_sample_size": None, "default_sample_flags": None}
    _track_id, _description_index, _duration, size, flags = struct.unpack_from(">5I", moov_buf, trex_start + 4)
    return {"default_sample_size": size, "default_sample_flags": flags}


def _parse_fragment(
    moof_start: int, moof_bytes: bytes, trex_defaults: dict[str, int | None]
) -> tuple[int, int, bool, int]:
    """Parse one ``moof``'s single sample into ``(offset, size, is_keyframe, decode_time)``.

    Args:
        moof_start: the ``moof`` box's absolute offset in the source file.
        moof_bytes: the entire ``moof`` box, header included.
        trex_defaults: this video's ``mvex``/``trex`` fallback, parsed once from ``moov``.

    Raises:
        FragmentLayoutUnsupportedError: the fragment's ``tfhd``/``trun`` don't match the one
            shape verified against production files; refused rather than guessed at.
        CompositionOffsetUnsupportedError: the sample carries a nonzero composition time offset.

    """
    _box_type, box_size, header_len = read_box_header(moof_bytes, 0, len(moof_bytes))
    moof_content_start, moof_content_end = header_len, box_size
    traf_start, traf_end = find_box(moof_bytes, ("traf",), moof_content_start, moof_content_end)
    tfhd_start, _tfhd_end = find_box(moof_bytes, ("tfhd",), traf_start, traf_end)
    tfhd = _parse_tfhd(moof_bytes, tfhd_start)
    tfdt_start, _tfdt_end = find_box(moof_bytes, ("tfdt",), traf_start, traf_end)
    decode_time = _parse_tfdt(moof_bytes, tfdt_start)
    trun_start, _trun_end = find_box(moof_bytes, ("trun",), traf_start, traf_end)
    trun = _parse_trun(moof_bytes, trun_start)

    if trun["sample_composition_offset"]:
        msg = f"moof at offset {moof_start}: trun carries a nonzero composition time offset"
        raise CompositionOffsetUnsupportedError(msg)
    if not tfhd["default_base_is_moof"]:
        msg = f"moof at offset {moof_start}: tfhd lacks default-base-is-moof -- unverified layout"
        raise FragmentLayoutUnsupportedError(msg)
    if trun["data_offset"] is None:
        msg = f"moof at offset {moof_start}: trun lacks data-offset-present -- unverified layout"
        raise FragmentLayoutUnsupportedError(msg)

    sample_size = trun["sample_size"] if trun["sample_size"] is not None else tfhd["default_sample_size"]
    if sample_size is None:
        sample_size = trex_defaults["default_sample_size"]
    sample_flags = trun["sample_flags"] if trun["sample_flags"] is not None else tfhd["default_sample_flags"]
    if sample_flags is None:
        sample_flags = trex_defaults["default_sample_flags"]
    if sample_size is None or sample_flags is None:
        msg = f"moof at offset {moof_start}: sample size/flags unresolved by trun, tfhd, or trex"
        raise FragmentLayoutUnsupportedError(msg)

    is_keyframe = not bool(sample_flags & _SAMPLE_IS_NON_SYNC_SAMPLE_BIT)
    absolute_offset = moof_start + trun["data_offset"]
    return absolute_offset, sample_size, is_keyframe, decode_time


def parse_fragment_table(moov_bytes: bytes, moof_rows: list[tuple[int, bytes]]) -> VideoTrack:
    """Parse a fragmented video's ``moov`` plus every published ``moof`` row into a faithful :class:`VideoTrack`.

    Args:
        moov_bytes: exactly one ``moov`` box, header included.
        moof_rows: every ``(moof_start, moof_bytes)`` pair for this video, in file order.

    Raises:
        BoxNotFoundError: *moov_bytes* is not a ``moov`` box, or carries no video ``trak``.
        FragmentLayoutUnsupportedError: a fragment's layout doesn't match what this module has
            verified against production files.
        CompositionOffsetUnsupportedError: a sample carries a nonzero composition offset.
        MalformedBoxHeaderError: *moov_bytes* or a *moof_rows* entry is truncated or corrupt
            partway through parsing.

    """
    try:
        return _parse_fragment_table(moov_bytes, moof_rows)
    except (struct.error, IndexError) as exc:
        msg = f"moov/moof buffer ran out mid-parse -- truncated or corrupt: {exc}"
        raise MalformedBoxHeaderError(msg) from exc


def _parse_fragment_table(moov_bytes: bytes, moof_rows: list[tuple[int, bytes]]) -> VideoTrack:
    box_type, box_size, header_len = read_box_header(moov_bytes, 0, len(moov_bytes))
    if box_type != "moov":
        msg = f"expected a moov box, found {box_type!r}"
        raise BoxNotFoundError(msg)
    moov_start, moov_end = header_len, box_size

    trak_start, trak_end, _stbl_start, _stbl_end, stsd_start, stsd_end, fourcc = find_video_trak(
        moov_bytes, moov_start, moov_end
    )

    mvhd_start, mvhd_end = find_box(moov_bytes, ("mvhd",), moov_start, moov_end)
    movie_header = MovieHeader.parse(moov_bytes, mvhd_start, mvhd_end)
    moov_extension_boxes = uuid_extension_boxes(moov_bytes, moov_start, moov_end)

    tkhd_start, tkhd_end = find_box(moov_bytes, ("tkhd",), trak_start, trak_end)
    track_header = TrackHeader.parse(moov_bytes, tkhd_start, tkhd_end)
    try:
        edts_start, edts_end = find_box(moov_bytes, ("edts",), trak_start, trak_end)
        elst_bytes = find_box_bytes(moov_bytes, "elst", edts_start, edts_end)
    except BoxNotFoundError:
        elst_bytes = None

    mdhd_start, mdhd_end = find_box(moov_bytes, ("mdia", "mdhd"), trak_start, trak_end)
    media_header = MediaHeader.parse(moov_bytes, mdhd_start, mdhd_end)

    sample_description = StsdBox.parse(moov_bytes, stsd_start, stsd_end, fourcc)

    trex_defaults = _parse_trex_defaults(moov_bytes, moov_start, moov_end)

    offsets: list[int] = []
    sizes: list[int] = []
    is_keyframe: list[bool] = []
    decode_times: list[int] = []
    for moof_start, moof_bytes in moof_rows:
        offset, size, keyframe, decode_time = _parse_fragment(moof_start, moof_bytes, trex_defaults)
        offsets.append(offset)
        sizes.append(size)
        is_keyframe.append(keyframe)
        decode_times.append(decode_time)

    return VideoTrack(
        timescale=media_header.timescale,
        duration=media_header.duration,
        sample_description=sample_description,
        movie_header=movie_header,
        track_header=track_header,
        moov_extension_boxes=moov_extension_boxes,
        elst_bytes=elst_bytes,
        dts_ticks=np.array(decode_times, dtype=np.int64),
        offset=np.array(offsets, dtype=np.int64),
        size=np.array(sizes, dtype=np.int64),
        is_keyframe=np.array(is_keyframe, dtype=np.bool_),
        composition_offsets=None,
    )
