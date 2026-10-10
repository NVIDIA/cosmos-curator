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
"""Tests for the faithful, struct-of-arrays ``moov`` decode in ``moov.py``.

Focus is on what makes this reader faithful rather than assembled: it never
resolves an absent ``stss``/``ctts`` into a default, and it never maps a
fourcc or profile byte to a friendly name.
"""

import struct

import pytest

from cosmos_curator.core.sensors.mp4 import moov as st
from cosmos_curator.core.sensors.mp4.box_primitives import BoxNotFoundError, MalformedBoxHeaderError, find_box
from cosmos_curator.core.sensors.mp4.box_traversal import (
    BoxLocation,
    BoxScope,
    BoxTraversal,
    UuidCallbackError,
    UuidHandler,
)
from cosmos_curator.core.sensors.mp4.moov_track import AmbiguousSampleDescriptionError
from tests.cosmos_curator.core.sensors.mp4 import _box_fixtures as bf
from tests.cosmos_curator.core.sensors.mp4._box_fixtures import pack_box as _box


def test_parse_video_track_end_to_end() -> None:
    """A fully synthetic moov decodes into the expected SoA shape."""
    moov = bf.synthetic_moov(frame_count=4, keyframe_index=0, ctts_offset=None)

    table = st.parse_video_track(moov)

    assert table.timescale == 12288
    assert table.sample_description.fourcc == "avc1"
    assert table.sample_description.width == 640
    assert table.sample_description.height == 480
    assert table.sample_description.avc_profile_idc == 0x42
    assert table.sample_description.avc_constraint_flags == 0xC0
    assert table.sample_description.avc_level_idc == 0x1E
    assert table.dts_ticks.tolist() == [0, 512, 1024, 1536]
    assert table.size.tolist() == [1000, 1001, 1002, 1003]
    assert table.offset.tolist() == [5000, 6000, 7001, 8003]
    assert table.is_keyframe is not None
    assert table.is_keyframe.tolist() == [True, False, False, False]
    assert table.composition_offsets is None


def test_parse_video_track_leaves_is_keyframe_none_when_stss_is_absent() -> None:
    """No stss at all means this reader reports absence, not a defaulted all-True array."""
    moov = bf.synthetic_moov(frame_count=3, keyframe_index=None, ctts_offset=None)

    table = st.parse_video_track(moov)

    assert table.is_keyframe is None


def test_parse_video_track_carries_ctts_as_undecoded_run_length_pairs() -> None:
    """A nonzero ctts offset is carried through, not rejected and not folded into dts_ticks."""
    moov = bf.synthetic_moov(frame_count=2, keyframe_index=0, ctts_offset=2048)

    table = st.parse_video_track(moov)

    assert table.composition_offsets is not None
    assert table.composition_offsets.sample_counts.tolist() == [2]
    assert table.composition_offsets.sample_offsets.tolist() == [2048]
    # dts_ticks is untouched by the ctts offset -- combining them is a layer-2 decision.
    assert table.dts_ticks.tolist() == [0, 512]


def test_parse_video_track_carries_zero_offset_ctts_the_same_way_as_nonzero() -> None:
    """A present-but-zero ctts is not special-cased away -- it's still carried through as-is."""
    moov = bf.synthetic_moov(frame_count=2, keyframe_index=0, ctts_offset=0)

    table = st.parse_video_track(moov)

    assert table.composition_offsets is not None
    assert table.composition_offsets.sample_offsets.tolist() == [0]


def test_parse_video_track_rejects_ctts_whose_sample_count_disagrees_with_stts() -> None:
    """ISO/IEC 14496-12 8.6.1.3.3: CT(n) = DT(n) + CTTS(n) for the same samples stts defines.

    A ctts covering a different number of samples than stts is non-conformant
    -- refused outright, not silently truncated or left to crash a caller
    indexing past the shorter array.
    """
    moov = bf.synthetic_moov(frame_count=4, keyframe_index=0, ctts_offset=10, ctts_sample_count=3)

    with pytest.raises(st.MalformedCompositionOffsetError):
        st.parse_video_track(moov)


def test_parse_video_track_rejects_stsz_sample_count_disagreeing_with_stts() -> None:
    """Stsz declaring fewer samples than stts defines is non-conformant, not silently truncated.

    Nothing else cross-checks stts/stsz/stsc agreement, so a mismatch here would otherwise
    reach VideoTrack with size shorter than dts_ticks and fail confusingly several layers up.
    """
    moov = bf.synthetic_moov(frame_count=4, keyframe_index=0, ctts_offset=None, stsz_sample_count=3)

    with pytest.raises(st.MalformedSampleTableError):
        st.parse_video_track(moov)


def test_parse_video_track_rejects_stsc_expansion_disagreeing_with_stts() -> None:
    """stsc/stco expanding to fewer offsets than stts defines is non-conformant, not silently truncated.

    Mirrors the stsz check: a single chunk declaring fewer samples-per-chunk than the
    track's total sample count expands to too few byte offsets for the frame count.
    """
    moov = bf.synthetic_moov(frame_count=4, keyframe_index=0, ctts_offset=None, stsc_samples_per_chunk=2)

    with pytest.raises(st.MalformedSampleTableError):
        st.parse_video_track(moov)


def test_parse_video_track_reports_undecoded_fourcc_not_a_friendly_codec_name() -> None:
    """The sample entry's own box type is kept verbatim -- no avc1/avc3 -> "h264" mapping."""
    moov = bf.synthetic_moov(frame_count=1, keyframe_index=0, ctts_offset=None)

    table = st.parse_video_track(moov)

    assert table.sample_description.fourcc == "avc1"


def test_parse_video_track_leaves_avc_fields_none_for_an_entry_without_a_config_box() -> None:
    """An hev1 entry with no hvcC at all yields width/height but no profile/level -- no guess is made."""
    moov = bf.synthetic_moov(frame_count=1, keyframe_index=0, ctts_offset=None, fourcc="hevc")

    table = st.parse_video_track(moov)

    assert table.sample_description.fourcc == "hev1"
    assert table.sample_description.width == 640
    assert table.sample_description.extradata == b""
    assert table.sample_description.avc_profile_idc is None
    assert table.sample_description.avc_level_idc is None
    assert table.sample_description.nal_length_size is None
    assert table.sample_description.chroma_format_idc is None
    assert table.sample_description.buffer_size_db is None
    assert table.sample_description.stsd_extension_boxes == ()


def test_parse_video_track_rejects_a_non_moov_box() -> None:
    """A buffer that isn't a moov box at all fails loudly, not with a confusing downstream error."""
    not_moov = _box("free", b"payload")

    with pytest.raises(BoxNotFoundError):
        st.parse_video_track(not_moov)


def test_parse_video_track_raises_malformed_box_header_error_on_truncated_content() -> None:
    """A moov cut short partway through parsing (not just at the header) is refused, not crashed on."""
    moov = bf.synthetic_moov(frame_count=4, keyframe_index=0, ctts_offset=None)
    truncated = moov[: len(moov) // 2]

    with pytest.raises(MalformedBoxHeaderError, match="ran out mid-parse"):
        st.parse_video_track(truncated)


def test_parse_video_track_arrays_are_read_only() -> None:
    """Every array field comes back as a read-only view, per the sensor library's own convention."""
    moov = bf.synthetic_moov(frame_count=2, keyframe_index=0, ctts_offset=2048)

    table = st.parse_video_track(moov)

    assert table.dts_ticks.flags.writeable is False
    assert table.offset.flags.writeable is False
    assert table.size.flags.writeable is False
    assert table.is_keyframe is not None
    assert table.is_keyframe.flags.writeable is False
    assert table.composition_offsets is not None
    assert table.composition_offsets.sample_counts.flags.writeable is False
    assert table.composition_offsets.sample_offsets.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        table.dts_ticks[0] = 999


def test_parse_video_track_multi_chunk_offsets() -> None:
    """Offset assembly across multiple stsc groups and chunks still works under the SoA reader."""
    width, height = 640, 480
    extradata = bytes([1, 0x42, 0xC0, 0x1E])
    stsd = bf.avc1_stsd(width, height, extradata)
    stts = bf.pack_full_box("stts", 0, struct.pack(">I", 1) + struct.pack(">II", 4, 512))
    sizes = [10, 20, 30, 40]
    stsz = bf.pack_full_box("stsz", 0, struct.pack(">II", 0, 4) + struct.pack(">IIII", *sizes))
    # Two stsc groups: chunk 1 has 2 samples, chunks 2+ have 1 sample each.
    stsc = bf.pack_full_box("stsc", 0, struct.pack(">I", 2) + struct.pack(">IIIIII", 1, 2, 1, 2, 1, 1))
    stco = bf.pack_full_box("stco", 0, struct.pack(">I", 3) + struct.pack(">III", 1000, 2000, 3000))
    stbl = _box("stbl", stsd + stts + stsz + stsc + stco)
    minf = _box("minf", stbl)
    mdhd = bf.pack_full_box("mdhd", 0, struct.pack(">III", 0, 0, 12288) + struct.pack(">I", 4 * 512))
    mdia = _box("mdia", mdhd + minf)
    trak = _box("trak", bf.tkhd_box() + mdia)
    moov = _box("moov", bf.mvhd_box() + trak)

    table = st.parse_video_track(moov)

    assert table.offset.tolist() == [1000, 1010, 2000, 3000]


def test_parse_video_track_decodes_avcc_extension_fields_for_high_profile() -> None:
    """A real, hand-decoded High-profile avcC's extension fields decode exactly as verified by hand."""
    moov = bf.synthetic_moov(frame_count=1, keyframe_index=0, ctts_offset=None, extradata=bf.avcc_extradata(100))

    sd = st.parse_video_track(moov).sample_description

    assert sd.avc_profile_idc == 100
    assert sd.nal_length_size == 4
    assert sd.chroma_format_idc == 1
    assert sd.bit_depth_luma_minus8 == 0
    assert sd.bit_depth_chroma_minus8 == 0


def test_parse_video_track_extension_fields_none_for_high_profile_with_no_extension_bytes() -> None:
    """High profile alone isn't enough -- the extension bytes still have to actually be present."""
    moov = bf.synthetic_moov(
        frame_count=1, keyframe_index=0, ctts_offset=None, extradata=bf.avcc_extradata(100, include_extension=False)
    )

    sd = st.parse_video_track(moov).sample_description

    assert sd.avc_profile_idc == 100
    assert sd.chroma_format_idc is None
    assert sd.bit_depth_luma_minus8 is None
    assert sd.bit_depth_chroma_minus8 is None


def test_parse_video_track_extension_fields_none_below_high_profile_even_with_trailing_bytes() -> None:
    """Trailing bytes after PPS are not misread as the extension when profile_idc < 100."""
    moov = bf.synthetic_moov(frame_count=1, keyframe_index=0, ctts_offset=None, extradata=bf.avcc_extradata(66))

    sd = st.parse_video_track(moov).sample_description

    assert sd.avc_profile_idc == 66
    assert sd.chroma_format_idc is None
    assert sd.bit_depth_luma_minus8 is None
    assert sd.bit_depth_chroma_minus8 is None


def test_parse_video_track_btrt_present() -> None:
    """A btrt sibling of avcC decodes into the three undecoded bitrate fields."""
    btrt = _box("btrt", struct.pack(">III", 422464, 79977840, 481704))
    moov = bf.synthetic_moov(frame_count=1, keyframe_index=0, ctts_offset=None, stsd_extra_children=btrt)

    sd = st.parse_video_track(moov).sample_description

    assert sd.buffer_size_db == 422464
    assert sd.max_bitrate_bps == 79977840
    assert sd.avg_bitrate_bps == 481704


def test_parse_video_track_btrt_absent() -> None:
    """No btrt sibling means None, not a raised error."""
    moov = bf.synthetic_moov(frame_count=1, keyframe_index=0, ctts_offset=None)

    sd = st.parse_video_track(moov).sample_description

    assert sd.buffer_size_db is None
    assert sd.max_bitrate_bps is None
    assert sd.avg_bitrate_bps is None


def test_parse_video_track_stsd_extension_boxes_present() -> None:
    """A uuid sibling of avcC/btrt is captured verbatim as an (extension_type, payload) pair."""
    uuid = bf.uuid_box(b"usrdef_gopsize\x00\x00", struct.pack(">I", 5))
    moov = bf.synthetic_moov(frame_count=1, keyframe_index=0, ctts_offset=None, stsd_extra_children=uuid)

    sd = st.parse_video_track(moov).sample_description

    assert sd.stsd_extension_boxes == ((b"usrdef_gopsize\x00\x00", struct.pack(">I", 5)),)


def test_parse_video_track_stsd_extension_boxes_empty_when_absent() -> None:
    """No uuid sibling means an empty tuple, not None."""
    moov = bf.synthetic_moov(frame_count=1, keyframe_index=0, ctts_offset=None)

    sd = st.parse_video_track(moov).sample_description

    assert sd.stsd_extension_boxes == ()


def test_parse_video_track_raises_ambiguous_sample_description_for_multiple_entries() -> None:
    """stsd.entry_count > 1 refuses outright rather than parsing only the first entry silently."""
    moov = bf.synthetic_moov(frame_count=1, keyframe_index=0, ctts_offset=None, stsd_entry_count=2)

    with pytest.raises(AmbiguousSampleDescriptionError):
        st.parse_video_track(moov)


def test_parse_video_track_movie_and_track_header_version_0() -> None:
    """A version-0 mvhd/tkhd decodes creation/modification time and the track's own display size."""
    moov = bf.synthetic_moov(frame_count=1, keyframe_index=0, ctts_offset=None, mvhd_version=0, tkhd_version=0)

    table = st.parse_video_track(moov)

    assert table.movie_header.creation_time == 0
    assert table.movie_header.modification_time == 0
    assert table.track_header.matrix == bf.TKHD_MATRIX_IDENTITY
    assert table.track_header.width == 640 << 16
    assert table.track_header.height == 480 << 16


def test_parse_video_track_movie_and_track_header_version_1() -> None:
    """A version-1 (64-bit time) mvhd/tkhd decodes through the same fields."""
    moov = bf.synthetic_moov(frame_count=1, keyframe_index=0, ctts_offset=None, mvhd_version=1, tkhd_version=1)

    table = st.parse_video_track(moov)

    assert table.movie_header.creation_time == 0
    assert table.movie_header.modification_time == 0
    assert table.track_header.width == 640 << 16
    assert table.track_header.height == 480 << 16


def test_parse_video_track_moov_extension_boxes_captures_direct_children_only() -> None:
    """Two moov-level uuid boxes are both captured; a stsd-level one is not double-counted into this list."""
    uuid_a = bf.uuid_box(b"a" * 16, b"1111")
    uuid_b = bf.uuid_box(b"b" * 16, b"2222")
    stsd_uuid = bf.uuid_box(b"c" * 16, b"3333")
    moov = bf.synthetic_moov(
        frame_count=1,
        keyframe_index=0,
        ctts_offset=None,
        stsd_extra_children=stsd_uuid,
        moov_extra_children=uuid_a + uuid_b,
    )

    table = st.parse_video_track(moov)

    assert set(table.moov_extension_boxes) == {(b"a" * 16, b"1111"), (b"b" * 16, b"2222")}
    assert table.sample_description.stsd_extension_boxes == ((b"c" * 16, b"3333"),)


def test_parse_video_track_elst_bytes_present() -> None:
    """A track with an edts/elst carries its own bytes through verbatim, header included."""
    moov = bf.synthetic_moov(frame_count=1, keyframe_index=0, ctts_offset=None, include_elst=True)

    table = st.parse_video_track(moov)

    assert table.elst_bytes is not None
    assert table.elst_bytes[4:8] == b"elst"


def test_parse_video_track_elst_bytes_absent() -> None:
    """A track with no edts at all carries None, not an empty placeholder."""
    moov = bf.synthetic_moov(frame_count=1, keyframe_index=0, ctts_offset=None)

    table = st.parse_video_track(moov)

    assert table.elst_bytes is None


def test_chunk_offsets_falls_back_to_co64_when_stco_is_absent() -> None:
    """A stbl with co64 but no stco (the >4GB shape) still yields chunk offsets, via the 64-bit path."""
    entries = [5_000_000_000, 5_000_001_000]  # past stco's own 32-bit range
    co64 = bf.pack_full_box("co64", 0, struct.pack(">I", len(entries)) + struct.pack(">2Q", *entries))
    stbl = _box("stbl", co64)

    offsets = st._chunk_offsets(stbl, BoxTraversal(stbl), BoxScope(8, len(stbl), ("stbl",)))

    assert offsets == entries


def test_chunk_offsets_prefers_stco_when_both_are_present() -> None:
    """``stco`` is tried first -- a stbl carrying both (never expected in practice) uses the 32-bit one."""
    stco = bf.pack_full_box("stco", 0, struct.pack(">I", 1) + struct.pack(">I", 1234))
    co64 = bf.pack_full_box("co64", 0, struct.pack(">I", 1) + struct.pack(">Q", 5_000_000_000))
    stbl = _box("stbl", stco + co64)

    offsets = st._chunk_offsets(stbl, BoxTraversal(stbl), BoxScope(8, len(stbl), ("stbl",)))

    assert offsets == [1234]


@pytest.mark.parametrize("error_type", [IndexError, struct.error, RuntimeError, ValueError])
def test_uuid_callback_errors_are_not_malformed_mp4_data(error_type: type[Exception]) -> None:
    """Handler failures remain distinct from standard buffer-read failures."""
    identifier = b"1234567890abcdef"
    moov = bf.synthetic_moov(
        frame_count=1,
        keyframe_index=0,
        ctts_offset=None,
        stbl_extra_children=bf.uuid_box(identifier, b"payload"),
    )
    error = error_type("handler failed")

    def fail(_payload: bytes, _box: BoxLocation) -> None:
        raise error

    handler = UuidHandler(identifier, frozenset({("moov", "trak", "mdia", "minf", "stbl")}), fail)
    with pytest.raises(UuidCallbackError) as caught:
        st.parse_video_track(moov, uuid_handlers=[handler])
    assert caught.value.error is error
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("tkhd_version", [0, 1])
def test_uuid_handlers_follow_selected_video_scopes(tkhd_version: int) -> None:
    """Handlers receive video track context and leave preserved UUID bytes unchanged."""
    identifier = b"1234567890abcdef"
    events: list[tuple[bytes, BoxLocation]] = []
    moov = bf.synthetic_moov(
        frame_count=2,
        keyframe_index=0,
        ctts_offset=None,
        tkhd_version=tkhd_version,
        stbl_extra_children=bf.uuid_box(identifier, b"frames"),
        stsd_extra_children=bf.uuid_box(identifier, b"description"),
        moov_extra_children=bf.uuid_box(identifier, b"movie"),
    )
    paths = frozenset(
        {("moov",), ("moov", "trak", "mdia", "minf", "stbl"), ("moov", "trak", "mdia", "minf", "stbl", "stsd", "avc1")}
    )
    handler = UuidHandler(identifier, paths, lambda payload, box: events.append((payload, box)))
    track = st.parse_video_track(moov, uuid_handlers=[handler])
    assert len(events) == 3
    by_payload = dict(events)
    assert by_payload[b"movie"].parent.track_id is None
    assert by_payload[b"frames"].parent.track_id == 1
    assert by_payload[b"description"].parent.track_id == 1
    assert track.moov_extension_boxes == ((identifier, b"movie"),)
    assert track.sample_description.stsd_extension_boxes == ((identifier, b"description"),)
    assert track.size.tolist() == st.parse_video_track(moov).size.tolist()


def test_uuid_handlers_do_not_run_for_unselected_tracks() -> None:
    """An unselected track's UUID boxes cannot contribute to the video index."""
    identifier = b"1234567890abcdef"
    events: list[bytes] = []
    video = bf.synthetic_moov(
        frame_count=1, keyframe_index=0, ctts_offset=None, stbl_extra_children=bf.uuid_box(identifier, b"video")
    )
    other = video.replace(b"avc1", b"mp4a").replace(b"video", b"audio")
    start, end = find_box(other, ("trak",), 8, len(other))
    moov = _box("moov", _box("trak", other[start:end]) + video[8:])
    handler = UuidHandler(
        identifier, frozenset({("moov", "trak", "mdia", "minf", "stbl")}), lambda payload, _box: events.append(payload)
    )
    st.parse_video_track(moov, uuid_handlers=[handler])
    assert events == [b"video"]


@pytest.mark.parametrize("parent", ["mdia", "minf"])
def test_uuid_handlers_visit_intermediate_track_containers(parent: str) -> None:
    """Track-container UUIDs dispatch exactly once with the selected track identity."""
    identifier = b"1234567890abcdef"
    payload = b"track timing"
    extras = {f"{parent}_extra_children": bf.uuid_box(identifier, payload)}
    moov = bf.synthetic_moov(frame_count=1, keyframe_index=0, ctts_offset=None, **extras)
    events: list[tuple[bytes, BoxLocation]] = []
    parent_path = ("moov", "trak", "mdia") + (("minf",) if parent == "minf" else ())
    handler = UuidHandler(identifier, frozenset({parent_path}), lambda value, box: events.append((value, box)))

    track = st.parse_video_track(moov, uuid_handlers=[handler])

    assert track.size.tolist()
    assert len(events) == 1
    assert events[0][0] == payload
    assert events[0][1].parent.path == parent_path
    assert events[0][1].parent.track_id == 1
    assert events[0][1].parent.fragment_index is None
