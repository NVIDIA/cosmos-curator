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

"""MP4-specific decode-relevant typed views: ``Mp4FrameIndex``/``Mp4Metadata``.

``Mp4FrameIndex``/``Mp4Metadata`` are genuinely MP4-only (built from
ISO-BMFF box parsing), unlike this package's own ``data.video.VideoIndex``/
``VideoMetadata``, which are acquisition-path agnostic. The ``Mp4`` prefix
states that plainly; sitting next to ``data/video.py`` without it would
read as a close variant of those two, which would be backwards.

``Mp4FrameIndex``/``Mp4Metadata`` can be built two ways: from an
already-published sidecar table (``from_frame_index_table``/``from_row``),
or directly from a real ``cosmos_curator.core.sensors.mp4.moov_track.VideoTrack``
(``from_video_track``). ``processing.py`` calls ``Mp4Metadata.from_video_track``
the same way, but builds frame-index rows from :func:`video_track_sample_arrays`
directly instead of ``Mp4FrameIndex.from_video_track``.
"""

import struct
from fractions import Fraction
from typing import Any, Self

import attrs
import numpy as np
import numpy.typing as npt
import pyarrow as pa

from cosmos_curator.core.sensors.mp4.box_primitives import VIDEO_FOURCC_CODEC_NAMES
from cosmos_curator.core.sensors.mp4.errors import UnindexableVideoError
from cosmos_curator.core.sensors.mp4.moov_track import VideoTrack

_NS_PER_SECOND = 1_000_000_000


def video_track_sample_arrays(
    video_track: VideoTrack,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64] | None, npt.NDArray[np.bool_]]:
    """Return ``(dts_ticks, composition_offset_ticks, is_keyframe)`` -- elst/ns-independent, decode order.

    Shared by :meth:`Mp4FrameIndex.from_video_track` and ``processing.py``'s own frame-row
    construction, so the ``ctts``-expansion/``is_keyframe``-default logic has one copy, not two.

    ``composition_offset_ticks`` is expanded from ``ctts``'s run-length pairs via ``np.repeat``
    when present, ``None`` otherwise. ``is_keyframe`` defaults to "every sample is a keyframe"
    (the spec-sanctioned inference) when ``stbl`` carries no ``stss`` at all.
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
    """Struct-of-arrays copy of one video's ``FRAME_INDEX_SCHEMA`` rows, decode order (by ``dts_ticks``).

    Read once, sliced/searched many times -- a consumer wanting a bounded
    sample range or a keyframe lookup works against an already-loaded
    ``numpy`` array, not a re-read of the sidecar.

    ``dts_ns``/``composition_offset_ns``/``pts_ns``/``pts_sort_order``/``keyframe_indices`` are
    all computed once here, at construction, from ``dts_ticks``/``composition_offset_ticks`` plus
    the stream's own ``time_base``. ``pts_ns`` also applies the track's own ``edts/elst`` edit
    list, when the one supported shape is present -- ``dts_ns`` does not, staying the
    container's own undecoded decode timeline.

    Arrays stay in decode order throughout, never reordered into
    presentation order the way the sensor library's own ``VideoIndex``
    reorders its arrays -- decoding is fundamentally a decode-order
    operation (every consumer of ``offset``/``size``/``is_keyframe`` needs
    decode order, never presentation order), and reordering everything
    would mean carrying a second, un-reordered view alongside the
    reordered one. ``pts_sort_order`` is the *only* thing a
    presentation-order search needs: ``pts_ns[pts_sort_order]`` is a
    presentation-order view with no extra copy, and ``pts_sort_order[rank]``
    *is* the decode-order sample index -- no reverse mapping to derive
    separately.
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
        )

    @classmethod
    def from_frame_index_table(
        cls, table: pa.Table, *, time_base_num: int, time_base_den: int, elst_bytes: bytes | None = None
    ) -> Self:
        """Build from a real ``FRAME_INDEX_SCHEMA`` table (e.g. ``read_frame_index``'s own return).

        *time_base_num*/*time_base_den*/*elst_bytes* come from the same video's paired
        ``Mp4Metadata`` (``VIDEO_METADATA_SCHEMA.time_base_num``/``time_base_den``/
        ``elst_bytes``) -- converting ticks to nanoseconds, and applying an edit list to
        ``pts_ns``, both need data that lives in that other sidecar, not this table.

        Raises:
            ValueError: ``composition_offset_ticks`` is null on some rows but not others.
                Expected all-null (no ``ctts``) or fully populated (``ctts`` present) for
                one video -- a partial mix would otherwise reach
                :func:`pyarrow.Array.to_numpy`, which silently upcasts a partially-null
                int64 column to ``float64`` with ``NaN``.
            UnsupportedEditListError: *elst_bytes* is present but not the one shape this
                reader knows how to apply.
            InvalidTimeBaseError: *time_base_den* is 0.

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
        return cls._assemble(
            dts_ticks=dts_ticks,
            composition_offset_ticks=composition_offset_ticks,
            is_keyframe=table.column("is_keyframe").to_numpy(),
            offset=table.column("offset").to_numpy(),
            size=table.column("size").to_numpy(),
            time_base_num=time_base_num,
            time_base_den=time_base_den,
            elst_bytes=elst_bytes,
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
    encoder_tag: str | None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Self:
        """Build from one real ``VIDEO_METADATA_SCHEMA`` row (e.g. read back from ``read_video_metadata``)."""
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
            encoder_tag=None,
        )
