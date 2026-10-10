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

"""MP4 frame indexes and metadata built from parsed tracks or published Arrow records."""

import struct
from fractions import Fraction
from typing import Any, Self

import attrs
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc

from cosmos_curator.core.sensors.mp4.box_primitives import VIDEO_FOURCC_CODEC_NAMES
from cosmos_curator.core.sensors.mp4.errors import UnindexableVideoError
from cosmos_curator.core.sensors.mp4.moov_track import VideoTrack
from cosmos_curator.core.sensors.utils.validation import AttrsAttribute, optional_int64_array

_NS_PER_SECOND = 1_000_000_000

type TimestampArray = npt.NDArray[np.int64] | np.ma.MaskedArray[Any, np.dtype[np.int64]]


def video_track_sample_arrays(
    video_track: VideoTrack,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64] | None, npt.NDArray[np.bool_]]:
    """Return ``(dts_ticks, composition_offset_ticks, is_keyframe)`` in decode order.

    Expand ``ctts`` runs into per-sample offsets, or return ``None`` when absent.
    Without ``stss``, every sample is a keyframe. No edit-list shift or time-base
    conversion is applied.
    """
    frame_count = len(video_track.dts_ticks)
    composition_offsets = video_track.composition_offsets
    composition_offset_ticks = (
        None
        if composition_offsets is None
        else np.repeat(composition_offsets.sample_offsets, composition_offsets.sample_counts).astype(np.int64)
    )
    is_keyframe = (
        video_track.is_keyframe if video_track.is_keyframe is not None else np.ones(frame_count, dtype=np.bool_)
    )
    return video_track.dts_ticks, composition_offset_ticks, is_keyframe


#: elst's own media_rate_integer/media_rate_fraction pair for normal (1x) playback --
#: see UnsupportedEditListError's own docstring for why only this rate is supported.
_ELST_NORMAL_RATE = (1, 0)


class UnsupportedEditListError(UnindexableVideoError):
    """An ``edts/elst`` box this reader doesn't know how to apply to ``pts_ns``.

    Only the common shape is applied: a single entry, playing at normal (1x) rate, with a
    non-negative ``media_time`` (the lead-in the encoder buffered before presentation
    starts, trimmed by shifting every ``pts_ns`` back by that amount). Multiple entries, an
    empty edit (``media_time == -1``, meaning "insert a gap here"), a non-1x rate, or bytes
    too short to even hold the shape being checked all describe an edit list this reader
    refuses to guess at rather than silently produce a wrong shift.
    """


class InvalidTimeBaseError(UnindexableVideoError):
    """A ``time_base_den`` of 0 -- a degenerate ``mdhd`` with ``timescale == 0`` -- can't convert ticks to ns."""


def _edit_list_media_time_ticks(elst_bytes: bytes | None) -> int:
    """Return the single supported ``elst`` entry's own ``media_time``, in the track's own timescale.

    ``media_time`` is in the *media* (``mdhd``) timescale per ISO/IEC 14496-12 8.6.6 --
    the same tick domain ``dts_ticks`` is already in, so the caller can convert it to
    nanoseconds with the same ``time_base_num``/``time_base_den`` it already has, no
    separate movie-timescale lookup needed. Returns 0 when *elst_bytes* is ``None``
    (no edit list -- nothing to shift).

    Raises:
        UnsupportedEditListError: see that class's own docstring.

    """
    if elst_bytes is None:
        return 0
    try:
        version = elst_bytes[8]
        entry_count = struct.unpack_from(">I", elst_bytes, 12)[0]
        if entry_count != 1:
            msg = f"elst has {entry_count} entries -- only a single-entry edit list is supported"
            raise UnsupportedEditListError(msg)
        if version == 1:
            _segment_duration, media_time, rate_integer, rate_fraction = struct.unpack_from(">Qqhh", elst_bytes, 16)
        else:
            _segment_duration, media_time, rate_integer, rate_fraction = struct.unpack_from(">Iihh", elst_bytes, 16)
    except (struct.error, IndexError) as exc:
        msg = f"elst ({len(elst_bytes)} byte(s)) is truncated or corrupt -- cannot read its own entries"
        raise UnsupportedEditListError(msg) from exc
    if media_time < 0:
        msg = "elst's single entry is an empty edit (media_time == -1) -- not supported"
        raise UnsupportedEditListError(msg)
    if (rate_integer, rate_fraction) != _ELST_NORMAL_RATE:
        msg = f"elst's single entry plays at rate {rate_integer}.{rate_fraction} -- only normal (1x) rate is supported"
        raise UnsupportedEditListError(msg)
    return int(media_time)


#: Bounds `avg_frame_rate`'s denominator (after `Fraction.limit_denominator`) so its
#: numerator fits `VIDEO_METADATA_SCHEMA`'s int32 field. Motivated by a real video
#: (frame_count=31525, timescale=1000000) that overflowed int32 without this bound.
_AVG_FRAME_RATE_MAX_DENOMINATOR = 100_000


def _ticks_to_ns(ticks: npt.NDArray[np.int64], time_base_num: int, time_base_den: int) -> npt.NDArray[np.int64]:
    """Convert a tick array to nanoseconds: ``ticks * time_base_num * 1e9 // time_base_den``.

    Integer arithmetic throughout, never a per-sample Python loop. The
    intermediate product goes through Python's arbitrary-precision ints
    (``object``-dtype array), not straight ``int64``: at a ~1,000,000
    timescale, ``ticks * time_base_num * 1e9`` alone overflows ``int64`` past
    ~2.5 hours of media, even though the true nanosecond result comfortably
    fits. The final cast back to ``int64`` is checked rather than trusted,
    so a genuinely unrepresentable result raises instead of wrapping
    silently.

    This is the one real place this conversion happens: the result is
    computed at read time, never persisted.

    Raises:
        InvalidTimeBaseError: *time_base_den* is 0.

    """
    if time_base_den == 0:
        msg = f"time base denominator is 0 (time_base={time_base_num}/{time_base_den}) -- cannot convert ticks"
        raise InvalidTimeBaseError(msg)
    wide = ticks.astype(object) * int(time_base_num) * _NS_PER_SECOND // int(time_base_den)
    result = wide.astype(np.int64)
    if not np.array_equal(result.astype(object), wide):
        msg = f"tick-to-nanosecond conversion overflowed int64 (time_base={time_base_num}/{time_base_den})"
        raise OverflowError(msg)
    return result


def _stream_rate_stats(
    frame_count: int, sample_sizes: npt.NDArray[np.int64], duration_ticks: int, timescale: int
) -> tuple[Fraction, int]:
    """Return ``(avg_frame_rate, bit_rate_bps)`` -- both zero for a zero-duration track.

    The ratio is built from integers first, never through a float duration
    intermediate: a real video, during this recipe's own real-S3 validation,
    produced a numerator ~14x too large for ``VIDEO_METADATA_SCHEMA``'s int32
    field when the duration was floated first.
    """
    if not duration_ticks or not timescale:
        return Fraction(0), 0
    avg_frame_rate = Fraction(frame_count * timescale, duration_ticks).limit_denominator(
        _AVG_FRAME_RATE_MAX_DENOMINATOR
    )
    duration_seconds = duration_ticks / timescale
    bit_rate_bps = int(int(sample_sizes.sum()) * 8 / duration_seconds)
    return avg_frame_rate, bit_rate_bps


@attrs.define(frozen=True)
class Mp4FrameIndex:
    """Struct-of-arrays frame index in decode order.

    Timestamps, presentation order, and keyframe indices are computed at construction.
    The track's time base converts ticks to nanoseconds; a supported ``edts/elst`` edit
    shifts ``pts_ns`` but leaves ``dts_ns`` unchanged.

    Optional upstream-provided EOF/SOF/MOE arrays use int64 nanoseconds and share
    the frame index's decode order. Container time-base conversion, edit-list shifts,
    and presentation sorting never transform these values.

    ``eof_ns``, ``sof_ns``, and ``moe_ns`` are independently optional; recipes decide
    which timestamps are required. ``sof_ns`` records readout start; ``moe_ns`` records exposure midpoint
    and may precede SOF. SOF/MOE are independently absent when the recorder did not
    populate them. Partially missing SOF/MOE use masked int64 arrays, with ``True``
    marking unavailable frames; entirely missing columns are ``None``. Fully populated
    columns remain ordinary NumPy arrays. Exposure midpoint is not derived from SOF/EOF.

    ``pts_sort_order[rank]`` maps a presentation-order rank to a decode-order sample
    index, without reordering the frame arrays.
    """

    dts_ticks: npt.NDArray[np.int64]
    composition_offset_ticks: npt.NDArray[np.int64] | None  # None iff every row's own value is null
    dts_ns: npt.NDArray[np.int64]
    composition_offset_ns: npt.NDArray[np.int64] | None  # None iff every row's own value is null
    pts_ns: npt.NDArray[np.int64]  # dts_ns + (composition_offset_ns or 0) - edit_list_shift_ns, decode order
    pts_sort_order: npt.NDArray[np.int64]  # argsort(pts_ns) -- a decode-order-index permutation
    keyframe_indices: npt.NDArray[np.int64]  # flatnonzero(is_keyframe), already sorted
    is_keyframe: npt.NDArray[np.bool_]
    offset: npt.NDArray[np.int64]
    size: npt.NDArray[np.int64]
    eof_ns: npt.NDArray[np.int64] | None = attrs.field(default=None, validator=optional_int64_array)
    sof_ns: TimestampArray | None = attrs.field(default=None, validator=optional_int64_array)
    moe_ns: TimestampArray | None = attrs.field(default=None, validator=optional_int64_array)

    @eof_ns.validator
    @sof_ns.validator
    @moe_ns.validator
    def _validate_camera_timing(self, attribute: AttrsAttribute, value: TimestampArray | None) -> None:
        """Require supplied arrays to match the frame count; EOF cannot contain masked entries."""
        if value is None:
            return
        if len(value) != len(self):
            msg = f"{attribute.name} length must equal the frame count ({len(self)}), got {len(value)}"
            raise ValueError(msg)
        if attribute.name == "eof_ns" and isinstance(value, np.ma.MaskedArray) and np.any(value.mask):
            msg = "eof_ns must be populated for every frame"
            raise ValueError(msg)

    def __len__(self) -> int:
        """Return the sample count."""
        return len(self.dts_ticks)

    @classmethod
    def _assemble(  # noqa: PLR0913 -- one argument per real field the two constructors don't already share
        cls,
        *,
        dts_ticks: npt.NDArray[np.int64],
        composition_offset_ticks: npt.NDArray[np.int64] | None,
        is_keyframe: npt.NDArray[np.bool_],
        offset: npt.NDArray[np.int64],
        size: npt.NDArray[np.int64],
        time_base_num: int,
        time_base_den: int,
        elst_bytes: bytes | None,
        eof_ns: npt.NDArray[np.int64] | None = None,
        sof_ns: TimestampArray | None = None,
        moe_ns: TimestampArray | None = None,
    ) -> Self:
        """Shared construction path for both real constructors below -- one place ticks become ns/pts/order.

        Raises:
            UnsupportedEditListError: *elst_bytes* is present but not the one shape this
                reader knows how to apply -- see that class's own docstring.
            InvalidTimeBaseError: *time_base_den* is 0.

        """
        dts_ns = _ticks_to_ns(dts_ticks, time_base_num, time_base_den)
        composition_offset_ns = (
            None
            if composition_offset_ticks is None
            else _ticks_to_ns(composition_offset_ticks, time_base_num, time_base_den)
        )
        pts_ns = dts_ns if composition_offset_ns is None else dts_ns + composition_offset_ns
        edit_list_media_time_ticks = _edit_list_media_time_ticks(elst_bytes)
        if edit_list_media_time_ticks:
            edit_list_shift_ns = (
                int(edit_list_media_time_ticks) * int(time_base_num) * _NS_PER_SECOND // int(time_base_den)
            )
            pts_ns = pts_ns - edit_list_shift_ns
        return cls(
            dts_ticks=dts_ticks,
            composition_offset_ticks=composition_offset_ticks,
            dts_ns=dts_ns,
            composition_offset_ns=composition_offset_ns,
            pts_ns=pts_ns,
            pts_sort_order=np.argsort(pts_ns, kind="stable"),
            keyframe_indices=np.flatnonzero(is_keyframe),
            is_keyframe=is_keyframe,
            offset=offset,
            size=size,
            eof_ns=eof_ns,
            sof_ns=sof_ns,
            moe_ns=moe_ns,
        )

    @classmethod
    def from_frame_index_table(
        cls, table: pa.Table, *, time_base_num: int, time_base_den: int, elst_bytes: bytes | None = None
    ) -> Self:
        """Build from a ``FRAME_INDEX_SCHEMA`` table and the video's accompanying metadata.

        Pass ``time_base_num``, ``time_base_den``, and ``elst_bytes`` from ``Mp4Metadata``
        to convert ticks and apply the presentation-time shift.

        Raises:
            ValueError: Composition offsets or EOF contain a mix of null and non-null values;
                those columns must be all-null or fully populated. Partially missing SOF/MOE
                are represented as masked int64 arrays without rounding through float64.
            UnsupportedEditListError: The edit list has an unsupported shape.
            InvalidTimeBaseError: ``time_base_den`` is zero.

        """
        dts_ticks = table.column("dts_ticks").to_numpy()
        composition_offset_column = table.column("composition_offset_ticks")
        if composition_offset_column.null_count == len(table):
            composition_offset_ticks: npt.NDArray[np.int64] | None = None
        elif composition_offset_column.null_count != 0:
            msg = (
                f"composition_offset_ticks is null on {composition_offset_column.null_count} of {len(table)} "
                "row(s) -- expected all-null or fully populated, never a mix"
            )
            raise ValueError(msg)
        else:
            composition_offset_ticks = composition_offset_column.to_numpy(zero_copy_only=False)
        # Fill missing values before NumPy conversion to retain exact int64 values.
        # The independent mask, not the fill value, represents unavailable timing.
        timing: dict[str, TimestampArray | None] = {}
        for name in ("eof_ns", "sof_ns", "moe_ns"):
            column = table.column(name)
            if column.null_count == len(table):
                timing[name] = None
            elif column.null_count:
                if name == "eof_ns":
                    msg = "eof_ns must be populated for every frame or entirely absent"
                    raise ValueError(msg)
                # The pinned NumPy stubs lack a typed masked-array constructor.
                timing[name] = np.ma.MaskedArray(  # type: ignore[no-untyped-call]
                    pc.fill_null(column, 0).to_numpy(), mask=column.is_null().to_numpy(), copy=False
                )
            else:
                timing[name] = column.to_numpy()
        return cls._assemble(
            dts_ticks=dts_ticks,
            composition_offset_ticks=composition_offset_ticks,
            is_keyframe=table.column("is_keyframe").to_numpy(),
            offset=table.column("offset").to_numpy(),
            size=table.column("size").to_numpy(),
            time_base_num=time_base_num,
            time_base_den=time_base_den,
            elst_bytes=elst_bytes,
            eof_ns=timing["eof_ns"],
            sof_ns=timing["sof_ns"],
            moe_ns=timing["moe_ns"],
        )

    @classmethod
    def from_video_track(cls, video_track: VideoTrack) -> Self:
        """Build directly from a real, already-parsed ``VideoTrack`` -- no sidecar round-trip.

        Raises:
            UnsupportedEditListError: *video_track*'s own ``elst_bytes`` is present but not
                the one shape this reader knows how to apply.
            InvalidTimeBaseError: *video_track*'s own ``timescale`` is 0.

        """
        dts_ticks, composition_offset_ticks, is_keyframe = video_track_sample_arrays(video_track)
        return cls._assemble(
            dts_ticks=dts_ticks,
            composition_offset_ticks=composition_offset_ticks,
            is_keyframe=is_keyframe,
            offset=video_track.offset,
            size=video_track.size,
            time_base_num=1,
            time_base_den=video_track.timescale,
            elst_bytes=video_track.elst_bytes,
        )


@attrs.define(frozen=True)
class Mp4Metadata:
    """Typed mirror of one ``VIDEO_METADATA_SCHEMA`` row -- purely what the box states about the stream."""

    codec_name: str
    extradata: bytes
    time_base_num: int
    time_base_den: int
    frame_count: int
    keyframe_count: int
    width: int
    height: int
    avc_profile_idc: int | None
    avc_constraint_flags: int | None
    avc_level_idc: int | None
    nal_length_size: int | None
    chroma_format_idc: int | None
    bit_depth_luma_minus8: int | None
    bit_depth_chroma_minus8: int | None
    buffer_size_db: int | None
    max_bitrate_bps: int | None
    avg_bitrate_bps: int | None
    movie_creation_time: int
    movie_modification_time: int
    track_creation_time: int
    track_modification_time: int
    # VIDEO_METADATA_SCHEMA's own list<int32> comes back from a Lance/pyarrow row as a plain
    # list -- converted to a tuple here so a frozen Mp4Metadata is actually immutable.
    track_matrix: tuple[int, ...] = attrs.field(converter=tuple)
    track_display_width: int
    track_display_height: int
    elst_bytes: bytes | None
    avg_frame_rate_num: int
    avg_frame_rate_den: int
    bit_rate_bps: int
    duration_ns: int
    encoder_tag: str | None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Self:
        """Build from one real ``VIDEO_METADATA_SCHEMA`` row (e.g. ``read_video_record``'s metadata row)."""
        return cls(**{field.name: row[field.name] for field in attrs.fields(cls)})

    @classmethod
    def from_video_track(cls, video_track: VideoTrack) -> Self:
        """Build directly from a real, already-parsed ``VideoTrack`` -- no sidecar round-trip.

        ``encoder_tag`` stays ``None`` -- nothing in a ``VideoTrack`` states it.
        """
        sd = video_track.sample_description
        avg_frame_rate, bit_rate_bps = _stream_rate_stats(
            len(video_track.dts_ticks), video_track.size, video_track.duration, video_track.timescale
        )
        # Treat a zero timescale as unknown duration, matching the rate statistics.
        duration_ns = (
            0
            if video_track.timescale == 0
            else int(_ticks_to_ns(np.array([video_track.duration], dtype=np.int64), 1, video_track.timescale)[0])
        )
        return cls(
            codec_name=VIDEO_FOURCC_CODEC_NAMES[sd.fourcc],
            extradata=sd.extradata,
            time_base_num=1,
            time_base_den=video_track.timescale,
            frame_count=len(video_track.dts_ticks),
            keyframe_count=(
                int(video_track.is_keyframe.sum())
                if video_track.is_keyframe is not None
                else len(video_track.dts_ticks)
            ),
            width=sd.width,
            height=sd.height,
            avc_profile_idc=sd.avc_profile_idc,
            avc_constraint_flags=sd.avc_constraint_flags,
            avc_level_idc=sd.avc_level_idc,
            nal_length_size=sd.nal_length_size,
            chroma_format_idc=sd.chroma_format_idc,
            bit_depth_luma_minus8=sd.bit_depth_luma_minus8,
            bit_depth_chroma_minus8=sd.bit_depth_chroma_minus8,
            buffer_size_db=sd.buffer_size_db,
            max_bitrate_bps=sd.max_bitrate_bps,
            avg_bitrate_bps=sd.avg_bitrate_bps,
            movie_creation_time=video_track.movie_header.creation_time,
            movie_modification_time=video_track.movie_header.modification_time,
            track_creation_time=video_track.track_header.creation_time,
            track_modification_time=video_track.track_header.modification_time,
            # mypy's attrs plugin mis-infers converter=tuple's own parameter type; already tuple[int, ...] at runtime.
            track_matrix=video_track.track_header.matrix,  # type: ignore[arg-type]
            track_display_width=video_track.track_header.width,
            track_display_height=video_track.track_header.height,
            elst_bytes=video_track.elst_bytes,
            avg_frame_rate_num=avg_frame_rate.numerator,
            avg_frame_rate_den=avg_frame_rate.denominator,
            bit_rate_bps=bit_rate_bps,
            duration_ns=duration_ns,
            encoder_tag=None,
        )
