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

"""Parses ``moov``'s ``mvhd``/``tkhd``/``mdhd``/``stsd`` and defines :class:`VideoTrack`.

Box layout this module reads (everything below can sit inside either a non-fragmented or a
fragmented file's ``moov``)::

    moov
    ├── mvhd                 movie header: creation/modification time
    └── trak                 (the video track -- find_video_trak() picks this one by
        │                      fourcc if a file has multiple tracks, e.g. video + audio)
        ├── tkhd              track header: creation/modification time, transform, display size
        ├── edts
        │   └── elst          edit list (optional), verbatim, undecoded
        └── mdia
            ├── mdhd          media header: timescale, duration
            └── minf
                └── stbl
                    └── stsd  sample description: codec fourcc, width/height,
                              nested avcC/hvcC -> avc_profile_idc/avc_level_idc, extradata

``stbl``'s own per-sample content (``stts``/``ctts``/``stss``/``stsz``/``stsc``/``stco``/``co64``)
is non-fragmented-only and lives in ``moov.py``. Its fragmented equivalent
(``moof``/``traf``/``tfhd``/``tfdt``/``trun``) lives in ``fragment_table.py``.
"""

import struct
from typing import TYPE_CHECKING, Any, Self

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.mp4.box_primitives import BoxNotFoundError, find_box, uuid_extension_boxes
from cosmos_curator.core.sensors.mp4.errors import UnindexableVideoError
from cosmos_curator.core.sensors.utils.helpers import as_optional_readonly_view, as_readonly_view
from cosmos_curator.core.sensors.utils.validation import int64_array, optional_bool_array

if TYPE_CHECKING:
    AttrsAttribute = attrs.Attribute[Any]
else:
    AttrsAttribute = attrs.Attribute

#: SampleEntry base (8B+6B+2B) + VisualSampleEntry's own fixed fields (70B) = 86B;
#: nested config boxes (``avcC`` etc.) start right after.
_VISUAL_SAMPLE_ENTRY_FIXED_BYTES = 86
_WIDTH_OFFSET_IN_ENTRY = 32
_AVCC_PROFILE_BYTE = 1
_AVCC_CONSTRAINT_BYTE = 2
_AVCC_LEVEL_BYTE = 3
_AVCC_LENGTH_SIZE_BYTE = 4
_AVCC_NUM_SPS_BYTE = 5
_AVCC_SPS_ARRAY_START = 6
_AVCC_HIGH_PROFILE_MIN = 100
_AVCC_EXTENSION_FIELD_BYTES = 4
_TKHD_FIXED_TAIL_BYTES = 16  # reserved(8)+layer(2)+alternate_group(2)+volume(2)+reserved(2)
_TKHD_MATRIX_BYTES = 36  # 9 x int32(4)


class AmbiguousSampleDescriptionError(UnindexableVideoError):
    """``stsd.entry_count > 1`` -- a genuinely different track mid-stream codec switch."""


def _walk_length_prefixed_entries(data: bytes, position: int, count: int) -> int | None:
    """Advance *position* past *count* 2-byte-length-prefixed entries, or None if *data* runs out.

    Shared by both of ``avcC``'s variable-length arrays (SPS, then PPS) --
    their content is never decoded, only walked to find where they end.
    """
    for _ in range(count):
        if position + 2 > len(data):
            return None
        entry_length = struct.unpack_from(">H", data, position)[0]
        position += 2 + entry_length
        if position > len(data):
            return None
    return position


def _avcc_extension_offset(extradata: bytes) -> int | None:
    """Return the byte offset in *extradata* immediately after ``avcC``'s SPS/PPS arrays, or None if unreadable."""
    if len(extradata) <= _AVCC_NUM_SPS_BYTE:
        return None
    num_sps = extradata[_AVCC_NUM_SPS_BYTE] & 0x1F
    position = _walk_length_prefixed_entries(extradata, _AVCC_SPS_ARRAY_START, num_sps)
    if position is None or position >= len(extradata):
        return None
    num_pps = extradata[position]
    return _walk_length_prefixed_entries(extradata, position + 1, num_pps)


@attrs.define(frozen=True)
class StsdBox:
    """``stsd`` -- "sample description box": what decodes the track, and at what size.

    ISO/IEC 14496-12's own sample description box. It can in principle hold more than one
    entry; this type only represents the first. Its content is a fixed-layout
    ``VisualSampleEntry`` (fourcc, width, height) wrapping one nested, codec-specific decoder
    config box: ``avcC`` (``AVCDecoderConfigurationRecord``) for an ``avc1``/``avc3`` entry,
    or ``hvcC`` for ``hev1``/``hvc1``.
    """

    fourcc: str
    width: int
    height: int
    extradata: bytes
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
    stsd_extension_boxes: tuple[tuple[bytes, bytes], ...]

    @classmethod
    def parse(cls, buf: bytes, stsd_start: int, stsd_end: int, fourcc: str) -> Self:  # noqa: ARG003 -- stsd_end kept for symmetry with sibling parsers
        """Parse ``stsd``'s one sample-description entry.

        Called by both combiners -- a fragmented track's ``moov`` still carries this box,
        with an empty ``stbl`` around it.

        Fields:
            fourcc: the sample entry's own box type, verbatim (``"avc1"``, ``"hev1"``, ...).
            avc_profile_idc, avc_constraint_flags, avc_level_idc: the undecoded first three
                bytes of ``avcC``'s ``AVCDecoderConfigurationRecord`` (after its version byte),
                ``None`` when unreadable. ``hvcC``'s differently-laid-out equivalent fields
                are never parsed.
            nal_length_size, chroma_format_idc, bit_depth_luma_minus8, bit_depth_chroma_minus8:
                likewise undecoded ``avcC`` fields -- the last three present only for High
                profile and above (per spec), ``None`` otherwise.
            buffer_size_db, max_bitrate_bps, avg_bitrate_bps: ``btrt``'s own three undecoded
                fields when a ``btrt`` sibling of ``avcC`` exists, else ``None``.
            stsd_extension_boxes: every ``uuid`` box sibling of ``avcC``/``btrt`` within this
                sample entry, verbatim, as ``(extension_type, payload)`` pairs -- an empty
                tuple, not ``None``, when none are present.

        Raises:
            AmbiguousSampleDescriptionError: ``stsd.entry_count > 1`` -- checked
                before reading anything past ``entry_count`` itself.

        """
        entry_count = struct.unpack_from(">I", buf, stsd_start + 4)[0]
        if entry_count > 1:
            msg = f"stsd has {entry_count} sample-description entries, expected exactly one"
            raise AmbiguousSampleDescriptionError(msg)

        entry_start = stsd_start + 8  # FullBox header (4B) + entry_count (4B)
        entry_size = struct.unpack_from(">I", buf, entry_start)[0]
        entry_end = entry_start + entry_size

        width, height = struct.unpack_from(">HH", buf, entry_start + _WIDTH_OFFSET_IN_ENTRY)
        config_start = entry_start + _VISUAL_SAMPLE_ENTRY_FIXED_BYTES

        try:
            btrt_start, _btrt_end = find_box(buf, ("btrt",), config_start, entry_end)
            buffer_size_db, max_bitrate_bps, avg_bitrate_bps = struct.unpack_from(">III", buf, btrt_start)
        except BoxNotFoundError:
            buffer_size_db = max_bitrate_bps = avg_bitrate_bps = None
        stsd_extension_boxes = uuid_extension_boxes(buf, config_start, entry_end)

        box_name = "avcC" if fourcc in ("avc1", "avc3") else "hvcC"
        try:
            config_content_start, config_content_end = find_box(buf, (box_name,), config_start, entry_end)
        except BoxNotFoundError:
            return cls(
                fourcc,
                width,
                height,
                b"",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                buffer_size_db,
                max_bitrate_bps,
                avg_bitrate_bps,
                stsd_extension_boxes,
            )

        extradata = bytes(buf[config_content_start:config_content_end])
        if fourcc not in ("avc1", "avc3") or len(extradata) <= _AVCC_LEVEL_BYTE:
            return cls(
                fourcc,
                width,
                height,
                extradata,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                buffer_size_db,
                max_bitrate_bps,
                avg_bitrate_bps,
                stsd_extension_boxes,
            )

        avc_profile_idc = extradata[_AVCC_PROFILE_BYTE]
        nal_length_size = (
            (extradata[_AVCC_LENGTH_SIZE_BYTE] & 0x03) + 1 if len(extradata) > _AVCC_LENGTH_SIZE_BYTE else None
        )
        chroma_format_idc = bit_depth_luma_minus8 = bit_depth_chroma_minus8 = None
        extension_offset = _avcc_extension_offset(extradata)
        if (
            extension_offset is not None
            and avc_profile_idc >= _AVCC_HIGH_PROFILE_MIN
            and len(extradata) - extension_offset >= _AVCC_EXTENSION_FIELD_BYTES
        ):
            chroma_format_idc = extradata[extension_offset] & 0x03
            bit_depth_luma_minus8 = extradata[extension_offset + 1] & 0x07
            bit_depth_chroma_minus8 = extradata[extension_offset + 2] & 0x07

        return cls(
            fourcc,
            width,
            height,
            extradata,
            avc_profile_idc,
            extradata[_AVCC_CONSTRAINT_BYTE],
            extradata[_AVCC_LEVEL_BYTE],
            nal_length_size,
            chroma_format_idc,
            bit_depth_luma_minus8,
            bit_depth_chroma_minus8,
            buffer_size_db,
            max_bitrate_bps,
            avg_bitrate_bps,
            stsd_extension_boxes,
        )


@attrs.define(hash=False, frozen=True)
class CttsBox:
    """``ctts`` -- "composition time to sample box": the gap between decode order and presentation order.

    ISO/IEC 14496-12's own composition-offset box, present only when a track's presentation
    order diverges from its decode order -- the case that arises with B-frames. Like ``stts``,
    it's a run-length-encoded table: ``sample_counts[i]`` consecutive samples each carry the
    composition offset ``sample_offsets[i]``, added to that sample's decode timestamp to get
    its presentation timestamp.

    Kept exactly as the box encodes it -- never expanded to one value per sample, and never
    added to :attr:`VideoTrack.dts_ticks` to compute a presentation timestamp here. A
    zero-offset ``ctts`` and a genuinely reordering one are represented identically by this
    type.
    """

    __hash__ = None  # type: ignore[assignment]

    sample_counts: npt.NDArray[np.int64] = attrs.field(converter=as_readonly_view, validator=int64_array)
    sample_offsets: npt.NDArray[np.int64] = attrs.field(converter=as_readonly_view, validator=int64_array)

    def __eq__(self, other: object) -> bool:
        """Compare array fields with ``np.array_equal`` -- plain ``==`` on an array isn't a bool."""
        if not isinstance(other, CttsBox):
            return False
        return bool(np.array_equal(self.sample_counts, other.sample_counts)) and bool(
            np.array_equal(self.sample_offsets, other.sample_offsets)
        )


@attrs.define(frozen=True)
class MovieHeader:
    """``mvhd``: file-level creation/modification time, undecoded."""

    creation_time: int  # seconds since 1904-01-01, version-0 (32-bit) or version-1 (64-bit)
    modification_time: int

    @classmethod
    def parse(cls, buf: bytes, start: int, end: int) -> Self:  # noqa: ARG003 -- end kept for symmetry with sibling parsers
        """Parse a ``mvhd`` box's content (version-0 or version-1 layout) -- creation/modification time only."""
        version = buf[start]
        if version == 1:
            creation_time, modification_time = struct.unpack_from(">QQ", buf, start + 4)
        else:
            creation_time, modification_time = struct.unpack_from(">II", buf, start + 4)
        return cls(creation_time=creation_time, modification_time=modification_time)


@attrs.define(frozen=True)
class TrackHeader:
    """``tkhd``: this specific track's own creation/modification time, transform, and display size."""

    creation_time: int
    modification_time: int
    matrix: tuple[int, ...]
    width: int
    height: int

    @classmethod
    def parse(cls, buf: bytes, start: int, end: int) -> Self:  # noqa: ARG003 -- end kept for symmetry with sibling parsers
        """Parse a ``tkhd`` box's content (version-0 or version-1 layout).

        Fields:
            matrix: ``tkhd``'s own 9 values, undecoded 16.16/2.30 fixed-point, in the box's
                own layout.
            width, height: undecoded 16.16 fixed-point display size -- can differ from
                ``stsd``'s own coded width/height (e.g. anamorphic or explicitly scaled
                content).
        """
        version = buf[start]
        if version == 1:
            creation_time, modification_time = struct.unpack_from(">QQ", buf, start + 4)
            after_duration = start + 4 + 8 + 8 + 4 + 4 + 8  # +track_ID(4)+reserved(4)+duration(8)
        else:
            creation_time, modification_time = struct.unpack_from(">II", buf, start + 4)
            after_duration = start + 4 + 4 + 4 + 4 + 4 + 4  # +track_ID(4)+reserved(4)+duration(4)
        matrix_start = after_duration + _TKHD_FIXED_TAIL_BYTES
        matrix = struct.unpack_from(">9i", buf, matrix_start)
        width, height = struct.unpack_from(">II", buf, matrix_start + _TKHD_MATRIX_BYTES)
        return cls(
            creation_time=creation_time, modification_time=modification_time, matrix=matrix, width=width, height=height
        )


@attrs.define(frozen=True)
class MediaHeader:
    """``mdhd``: the clock a track's sample deltas are expressed in."""

    timescale: int
    duration: int

    @classmethod
    def parse(cls, buf: bytes, start: int, end: int) -> Self:  # noqa: ARG003 -- end kept for symmetry with sibling parsers
        """Parse a ``mdhd`` box's content (version-0 or version-1 layout)."""
        version = buf[start]
        if version == 1:
            timescale = struct.unpack_from(">I", buf, start + 4 + 8 + 8)[0]
            duration = struct.unpack_from(">Q", buf, start + 4 + 8 + 8 + 4)[0]
        else:
            timescale = struct.unpack_from(">I", buf, start + 4 + 4 + 4)[0]
            duration = struct.unpack_from(">I", buf, start + 4 + 4 + 4 + 4)[0]
        return cls(timescale=timescale, duration=duration)


class EmptySampleTableError(UnindexableVideoError):
    """A parse produced zero samples -- never a valid index, regardless of which combiner built it.

    A genuinely fragmented track's own ``stbl`` is real but empty by design (samples live in
    ``moof``/``traf``/``trun`` instead), so an empty result can't be told apart from a broken or
    truncated file using ``moov`` content alone -- confirmed against a real corpus: a
    no-fragments-at-all video parsed "successfully" via the non-fragmented path with zero
    samples. Refusing both combiners' empty results uniformly here is correct: neither is a
    valid zero-frame index worth publishing as a success.
    """


def _nonempty_sample_table(
    instance: object,  # noqa: ARG001
    attribute: AttrsAttribute,
    value: npt.NDArray[np.int64],
) -> None:
    """Attrs validator refusing a zero-length sample table -- see :class:`EmptySampleTableError`."""
    if len(value) == 0:
        msg = f"{attribute.name} is empty -- parsed to zero samples"
        raise EmptySampleTableError(msg)


@attrs.define(hash=False, frozen=True)
class VideoTrack:
    """A faithful, struct-of-arrays decode of one track: its ``mdhd``/``stsd`` context, plus its sample data.

    ``is_keyframe`` is ``None`` when the source carries no sync-sample info; ``composition_offsets``
    is ``None`` the same way, when ``ctts`` is absent. Both are real, common, valid states for a
    track to be in; this type reports them as absent rather than resolving them into a value the
    box never stated.

    ``moov_extension_boxes`` is every
    ``uuid`` box that is a direct child of ``moov`` itself (not nested inside
    any ``trak``, so this never double-counts ``sample_description``'s own
    ``stsd_extension_boxes``, which sit many levels deeper in the tree),
    verbatim, as ``(extension_type, payload)`` pairs. ``elst_bytes`` is the
    track's own ``edts/elst`` box, verbatim, header included, when present,
    else ``None`` -- not decoded here: a track with an edit list has a
    presentation timeline that isn't simply ``mdhd``'s duration/timescale.

    Built by two alternative combiners depending on container shape -- ``moov.py``'s own
    non-fragmented combiner (real ``stbl``) or ``fragment_table.py``'s own fragmented combiner
    (``moof``/``traf`` instead) -- so nothing about this type's own fields says which path
    built it.
    """

    __hash__ = None  # type: ignore[assignment]

    timescale: int
    duration: int
    sample_description: StsdBox
    movie_header: MovieHeader
    track_header: TrackHeader
    moov_extension_boxes: tuple[tuple[bytes, bytes], ...]
    elst_bytes: bytes | None
    dts_ticks: npt.NDArray[np.int64] = attrs.field(
        converter=as_readonly_view, validator=attrs.validators.and_(int64_array, _nonempty_sample_table)
    )
    offset: npt.NDArray[np.int64] = attrs.field(converter=as_readonly_view, validator=int64_array)
    size: npt.NDArray[np.int64] = attrs.field(converter=as_readonly_view, validator=int64_array)
    is_keyframe: npt.NDArray[np.bool_] | None = attrs.field(
        converter=as_optional_readonly_view, validator=optional_bool_array
    )
    composition_offsets: CttsBox | None

    def __eq__(self, other: object) -> bool:
        """Compare array fields with ``np.array_equal`` -- plain ``==`` on an array isn't a bool."""
        if not isinstance(other, VideoTrack):
            return False
        self_keyframe, other_keyframe = self.is_keyframe, other.is_keyframe
        if (self_keyframe is None) != (other_keyframe is None):
            return False
        is_keyframe_equal = self_keyframe is None or (
            other_keyframe is not None and bool(np.array_equal(self_keyframe, other_keyframe))
        )
        return (
            self.timescale == other.timescale
            and self.duration == other.duration
            and self.sample_description == other.sample_description
            and self.movie_header == other.movie_header
            and self.track_header == other.track_header
            and self.moov_extension_boxes == other.moov_extension_boxes
            and self.elst_bytes == other.elst_bytes
            and bool(np.array_equal(self.dts_ticks, other.dts_ticks))
            and bool(np.array_equal(self.offset, other.offset))
            and bool(np.array_equal(self.size, other.size))
            and is_keyframe_equal
            and self.composition_offsets == other.composition_offsets
        )
