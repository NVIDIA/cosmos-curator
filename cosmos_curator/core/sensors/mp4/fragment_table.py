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
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from cosmos_curator.core.sensors.mp4.box_primitives import (
    BoxNotFoundError,
    MalformedBoxHeaderError,
    find_all_boxes,
    find_box,
    read_box_header,
)
from cosmos_curator.core.sensors.mp4.box_traversal import BoxScope, BoxTraversal, UuidHandler
from cosmos_curator.core.sensors.mp4.errors import UnindexableVideoError
from cosmos_curator.core.sensors.mp4.moov_track import (
    MediaHeader,
    MovieHeader,
    StsdBox,
    TrackHeader,
    VideoTrack,
    video_track_scopes,
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
_TRUN_SAMPLE_DURATION_PRESENT = 0x000100
_TRUN_SAMPLE_SIZE_PRESENT = 0x000200
_TRUN_SAMPLE_FLAGS_PRESENT = 0x000400
_TRUN_SAMPLE_COMPOSITION_TIME_OFFSETS_PRESENT = 0x000800

#: Within a sample_flags word, whether this sample is *not* a sync sample (i.e. not a keyframe).
_SAMPLE_IS_NON_SYNC_SAMPLE_BIT = 0x00010000

#: Fewer samples than this leaves no decode-time interval to derive a duration span from.
_MIN_SAMPLES_FOR_A_DURATION_SPAN = 2


class FragmentLayoutUnsupportedError(UnindexableVideoError):
    """A fragment uses a layout outside the supported single-track, single-run profiles."""


class CompositionOffsetUnsupportedError(UnindexableVideoError):
    """A sample carries a nonzero composition time offset -- never observed in real production data."""


def _parse_tfhd(buf: bytes, start: int, end: int, *, expected_track_id: int | None) -> dict[str, int | bool | None]:
    """Parse ``tfhd``'s content into the fields a fragment's sample may need."""
    buf = buf[start:end]
    flags = int.from_bytes(buf[1:4], "big")
    track_id = struct.unpack_from(">I", buf, 4)[0]
    if track_id != expected_track_id:
        msg = f"tfhd at offset {start}: track ID {track_id} does not match moov track ID {expected_track_id}"
        raise FragmentLayoutUnsupportedError(msg)
    pos = 8  # FullBox header and track_ID; only single-track files are supported.
    base_data_offset = None
    if flags & _TFHD_BASE_DATA_OFFSET_PRESENT:
        base_data_offset = struct.unpack_from(">Q", buf, pos)[0]
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
        "base_data_offset": base_data_offset,
        "default_base_is_moof": bool(flags & _TFHD_DEFAULT_BASE_IS_MOOF),
        "sample_description_index": sample_description_index,
        "default_sample_duration": default_sample_duration,
        "default_sample_size": default_sample_size,
        "default_sample_flags": default_sample_flags,
    }


def _parse_tfdt(buf: bytes, start: int) -> int:
    """Parse ``tfdt``'s ``baseMediaDecodeTime`` (version-0 32-bit or version-1 64-bit)."""
    version = buf[start]
    if version not in (0, 1):
        msg = f"tfdt at offset {start}: unsupported version {version}"
        raise FragmentLayoutUnsupportedError(msg)
    if version == 1:
        return int(struct.unpack_from(">Q", buf, start + 4)[0])
    return int(struct.unpack_from(">I", buf, start + 4)[0])


@dataclass(frozen=True)
class _TrackRun:
    sample_count: int
    data_offset: int | None
    first_sample_flags: int | None
    durations: list[int] | None
    sizes: list[int] | None
    flags: list[int] | None
    composition_offsets: list[int] | None


def _parse_trun(buf: bytes, start: int, end: int) -> _TrackRun:
    """Read a run's sample columns within its declared box bounds."""
    buf = buf[start:end]
    version = buf[0]
    if version not in (0, 1):
        msg = f"trun at offset {start}: unsupported version {version}"
        raise FragmentLayoutUnsupportedError(msg)
    flags = int.from_bytes(buf[1:4], "big")
    sample_count = struct.unpack_from(">I", buf, 4)[0]
    if sample_count == 0:
        msg = f"trun at offset {start}: empty runs are unsupported"
        raise FragmentLayoutUnsupportedError(msg)
    if flags & _TRUN_FIRST_SAMPLE_FLAGS_PRESENT and flags & _TRUN_SAMPLE_FLAGS_PRESENT:
        msg = f"trun at offset {start}: first-sample flags and per-sample flags are mutually exclusive"
        raise FragmentLayoutUnsupportedError(msg)

    pos = 8
    data_offset = None
    if flags & _TRUN_DATA_OFFSET_PRESENT:
        data_offset = struct.unpack_from(">i", buf, pos)[0]
        pos += 4
    first_sample_flags = None
    if flags & _TRUN_FIRST_SAMPLE_FLAGS_PRESENT:
        first_sample_flags = struct.unpack_from(">I", buf, pos)[0]
        pos += 4

    columns = _parse_sample_columns(buf, pos, sample_count, flags, version)
    return _TrackRun(sample_count, data_offset, first_sample_flags, *columns)


def _parse_sample_columns(
    buf: bytes, pos: int, count: int, flags: int, version: int
) -> tuple[list[int] | None, list[int] | None, list[int] | None, list[int] | None]:
    fields = (
        _TRUN_SAMPLE_DURATION_PRESENT,
        _TRUN_SAMPLE_SIZE_PRESENT,
        _TRUN_SAMPLE_FLAGS_PRESENT,
        _TRUN_SAMPLE_COMPOSITION_TIME_OFFSETS_PRESENT,
    )
    columns: list[list[int] | None] = [[] if flags & field else None for field in fields]
    entry_size = 4 * sum(column is not None for column in columns)
    if pos + count * entry_size > len(buf):
        msg = "trun contains truncated sample entries"
        raise MalformedBoxHeaderError(msg)
    if entry_size:
        for _ in range(count):
            for field, column in zip(fields, columns, strict=True):
                if column is not None:
                    fmt = ">i" if field == _TRUN_SAMPLE_COMPOSITION_TIME_OFFSETS_PRESENT and version == 1 else ">I"
                    column.append(struct.unpack_from(fmt, buf, pos)[0])
                    pos += 4
    return columns[0], columns[1], columns[2], columns[3]


def _parse_trex_defaults(moov_buf: bytes, moov_content_start: int, moov_content_end: int) -> dict[str, int | None]:
    """Parse ``mvex``/``trex``'s fallback values, or all-``None`` if the track has no ``trex``."""
    try:
        trex_start, _trex_end = find_box(moov_buf, ("mvex", "trex"), moov_content_start, moov_content_end)
    except BoxNotFoundError:
        return {"default_sample_duration": None, "default_sample_size": None, "default_sample_flags": None}
    _track_id, _description_index, duration, size, flags = struct.unpack_from(">5I", moov_buf, trex_start + 4)
    return {"default_sample_duration": duration or None, "default_sample_size": size, "default_sample_flags": flags}


def _find_single_box(buf: bytes, name: str, start: int, end: int) -> tuple[int, int]:
    boxes = find_all_boxes(buf, name, start, end)
    if not boxes:
        msg = f"{name} not found"
        raise BoxNotFoundError(msg)
    if len(boxes) != 1:
        msg = f"expected exactly one {name}, found {len(boxes)}"
        raise FragmentLayoutUnsupportedError(msg)
    return boxes[0]


def _sample_value(
    column: list[int] | None, index: int, tfhd_default: int | None, trex_default: int | None
) -> int | None:
    if column is not None:
        return column[index]
    return tfhd_default if tfhd_default is not None else trex_default


def _parse_fragment(
    moof_row: tuple[int, bytes],
    trex_defaults: dict[str, int | None],
    *,
    uuid_handlers: Sequence[UuidHandler],
    fragment_index: int,
    expected_track_id: int | None,
) -> Iterator[tuple[int, int, bool, int, int | None]]:
    """Yield each sample's offset, size, keyframe flag, DTS, and duration in file order."""
    moof_start, moof_bytes = moof_row
    _box_type, box_size, header_len = read_box_header(moof_bytes, 0, len(moof_bytes))
    _find_single_box(moof_bytes, "traf", header_len, box_size)
    traversal = BoxTraversal(moof_bytes, uuid_handlers=uuid_handlers)
    moof_scope = BoxScope(header_len, box_size, ("moof",), fragment_index=fragment_index)
    traf = traversal.find(moof_scope, ("traf",))
    tfhd_start, tfhd_end = _find_single_box(moof_bytes, "tfhd", traf.content_start, traf.end)
    tfhd = _parse_tfhd(moof_bytes, tfhd_start, tfhd_end, expected_track_id=expected_track_id)
    track_id = struct.unpack_from(">I", moof_bytes[tfhd_start:tfhd_end], 4)[0]
    traf_scope = traf.child_scope(track_id=track_id)
    traversal.scan_children(traf_scope)
    tfdt_start, tfdt_end = _find_single_box(moof_bytes, "tfdt", traf.content_start, traf.end)
    decode_time = _parse_tfdt(moof_bytes[tfdt_start:tfdt_end], 0)
    trun_start, trun_end = _find_single_box(moof_bytes, "trun", traf.content_start, traf.end)
    trun = _parse_trun(moof_bytes, trun_start, trun_end)

    if trun.composition_offsets is not None and any(trun.composition_offsets):
        msg = f"moof at offset {moof_start}: trun carries a nonzero composition time offset"
        raise CompositionOffsetUnsupportedError(msg)
    base_offset = tfhd["base_data_offset"]
    if base_offset is None:
        if not tfhd["default_base_is_moof"]:
            msg = f"moof at offset {moof_start}: tfhd lacks base-data-offset and default-base-is-moof"
            raise FragmentLayoutUnsupportedError(msg)
        base_offset = moof_start
    if trun.data_offset is None:
        msg = f"moof at offset {moof_start}: trun lacks data-offset-present -- unverified layout"
        raise FragmentLayoutUnsupportedError(msg)
    absolute_offset = base_offset + trun.data_offset

    for index in range(trun.sample_count):
        size = _sample_value(trun.sizes, index, tfhd["default_sample_size"], trex_defaults["default_sample_size"])
        flags = _sample_value(trun.flags, index, tfhd["default_sample_flags"], trex_defaults["default_sample_flags"])
        if index == 0 and trun.first_sample_flags is not None:
            flags = trun.first_sample_flags
        if size is None or flags is None:
            msg = f"moof at offset {moof_start}: sample size/flags unresolved by trun, tfhd, or trex"
            raise FragmentLayoutUnsupportedError(msg)
        duration = _sample_value(
            trun.durations, index, tfhd["default_sample_duration"], trex_defaults["default_sample_duration"]
        )
        if duration is None and index + 1 < trun.sample_count:
            msg = f"moof at offset {moof_start}: sample duration unresolved by trun, tfhd, or trex"
            raise FragmentLayoutUnsupportedError(msg)
        yield absolute_offset, size, not bool(flags & _SAMPLE_IS_NON_SYNC_SAMPLE_BIT), decode_time, duration
        absolute_offset += size
        if duration is not None:
            decode_time += duration


def parse_fragment_table(
    moov_bytes: bytes, moof_rows: list[tuple[int, bytes]], *, uuid_handlers: Sequence[UuidHandler] = ()
) -> VideoTrack:
    """Parse a fragmented video's ``moov`` plus every published ``moof`` row into a faithful :class:`VideoTrack`.

    Args:
        moov_bytes: exactly one ``moov`` box, header included.
        moof_rows: every ``(moof_start, moof_bytes)`` pair for this video, in file order.
        uuid_handlers: invocation-local handlers for UUID boxes in visited scopes.

    Raises:
        BoxNotFoundError: *moov_bytes* is not a ``moov`` box, or carries no video ``trak``.
        FragmentLayoutUnsupportedError: a fragment's layout doesn't match what this module has
            verified against production files.
        CompositionOffsetUnsupportedError: a sample carries a nonzero composition offset.
        MalformedBoxHeaderError: *moov_bytes* or a *moof_rows* entry is truncated or corrupt
            partway through parsing.
        UuidCallbackError: a registered UUID handler failed; ``error`` retains its exception.

    """
    try:
        return _parse_fragment_table(moov_bytes, moof_rows, uuid_handlers=uuid_handlers)
    except (struct.error, IndexError) as exc:
        msg = f"moov/moof buffer ran out mid-parse -- truncated or corrupt: {exc}"
        raise MalformedBoxHeaderError(msg) from exc


def _duration_from_decode_times(decode_times: list[int], sample_durations: list[int | None]) -> int:
    """Return the decode span plus the last sample's duration, in track timescale ticks.

    Fragmented files may have a stale ``mdhd.duration``. Use the last sample's explicit
    duration when available; otherwise estimate it from the mean decode interval.
    With fewer than two samples, return the available sample duration or zero.
    """
    if len(decode_times) < _MIN_SAMPLES_FOR_A_DURATION_SPAN:
        return (sample_durations[-1] if sample_durations else None) or 0
    span = decode_times[-1] - decode_times[0]
    last_interval = sample_durations[-1] if sample_durations[-1] is not None else span // (len(decode_times) - 1)
    return span + last_interval


def _reject_initial_samples(buf: bytes, stbl_scope: BoxScope) -> None:
    """Refuse initial sample tables that the fragment-only index would silently omit."""
    for name in ("stts", "stsz", "stz2", "stsc", "stco", "co64"):
        for start, end in find_all_boxes(buf, name, stbl_scope.start, stbl_scope.end):
            count_offset = 8 if name in ("stsz", "stz2") else 4
            count = struct.unpack_from(">I", buf[start:end], count_offset)[0]
            if count:
                msg = f"moov contains a nonempty {name} table; initial samples alongside fragments are unsupported"
                raise FragmentLayoutUnsupportedError(msg)


def _parse_fragment_table(
    moov_bytes: bytes, moof_rows: list[tuple[int, bytes]], *, uuid_handlers: Sequence[UuidHandler]
) -> VideoTrack:
    box_type, box_size, header_len = read_box_header(moov_bytes, 0, len(moov_bytes))
    if box_type != "moov":
        msg = f"expected a moov box, found {box_type!r}"
        raise BoxNotFoundError(msg)
    moov_start, moov_end = header_len, box_size
    _find_single_box(moov_bytes, "trak", moov_start, moov_end)

    traversal = BoxTraversal(moov_bytes, uuid_handlers=uuid_handlers)
    moov_scope = BoxScope(moov_start, moov_end, ("moov",))
    trak_scope, stbl_scope, stsd_scope, fourcc = video_track_scopes(moov_bytes, moov_scope)
    # Visit the selected track's intermediate containers as well as stbl,
    # so UUID handlers registered for mdia/minf are dispatched with track context.
    traversal.find(trak_scope, ("mdia", "minf", "stbl"))
    traversal.scan_children(stbl_scope)
    _reject_initial_samples(moov_bytes, stbl_scope)

    mvhd_start, mvhd_end = traversal.find_bounds(moov_scope, ("mvhd",))
    movie_header = MovieHeader.parse(moov_bytes, mvhd_start, mvhd_end)
    moov_extension_boxes = traversal.extension_boxes(moov_scope)

    tkhd_start, tkhd_end = traversal.find_bounds(trak_scope, ("tkhd",))
    track_header = TrackHeader.parse(moov_bytes, tkhd_start, tkhd_end)
    try:
        elst_bytes = traversal.find_bytes(trak_scope, ("edts", "elst"))
    except BoxNotFoundError:
        elst_bytes = None

    mdhd_start, mdhd_end = traversal.find_bounds(trak_scope, ("mdia", "mdhd"))
    media_header = MediaHeader.parse(moov_bytes, mdhd_start, mdhd_end)

    sample_description = StsdBox.parse(moov_bytes, traversal, stsd_scope, fourcc)

    trex_defaults = _parse_trex_defaults(moov_bytes, moov_start, moov_end)

    offsets: list[int] = []
    sizes: list[int] = []
    is_keyframe: list[bool] = []
    decode_times: list[int] = []
    sample_durations: list[int | None] = []
    for fragment_index, moof_row in enumerate(moof_rows):
        for offset, size, keyframe, decode_time, sample_duration in _parse_fragment(
            moof_row,
            trex_defaults,
            uuid_handlers=uuid_handlers,
            fragment_index=fragment_index,
            expected_track_id=trak_scope.track_id,
        ):
            offsets.append(offset)
            sizes.append(size)
            is_keyframe.append(keyframe)
            decode_times.append(decode_time)
            sample_durations.append(sample_duration)

    return VideoTrack(
        timescale=media_header.timescale,
        duration=_duration_from_decode_times(decode_times, sample_durations),
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
