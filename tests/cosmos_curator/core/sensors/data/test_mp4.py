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
"""Construct Mp4FrameIndex/Mp4Metadata from real FRAME_INDEX_SCHEMA/VIDEO_METADATA_SCHEMA rows, VideoTracks."""

import pathlib
from fractions import Fraction
from functools import partial
from typing import Any, cast

import attrs
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pytest

from cosmos_curator.core.sensors.data.mp4 import (
    InvalidTimeBaseError,
    Mp4FrameIndex,
    Mp4Metadata,
    UnsupportedEditListError,
    _stream_rate_stats,
    _ticks_to_ns,
)
from cosmos_curator.core.sensors.data.video import VideoProbe
from cosmos_curator.core.sensors.mp4.box_primitives import read_box_header
from cosmos_curator.core.sensors.mp4.errors import UnindexableVideoError
from cosmos_curator.core.sensors.mp4.fragment_table import parse_fragment_table
from cosmos_curator.core.sensors.mp4.moov import parse_video_track
from cosmos_curator.core.sensors.types.types import VideoIndexCreationMethod
from cosmos_curator.core.sensors.utils.video import make_index_and_metadata
from cosmos_curator.next.recipes.mp4_index.contracts import FRAME_INDEX_SCHEMA
from tests.cosmos_curator.core.sensors.mp4 import _box_fixtures as bf

#: Same real, checked-in B-frame clip core.sensors.video's own tests use
#: (tests/cosmos_curator/core/sensors/video/conftest.py) -- reused here, not
#: re-fetched, so pts_ns/pts_sort_order/keyframe_indices are validated against
#: real composition-offset reordering, not just synthetic single-value ctts.
_BFRAMES_MP4 = (
    pathlib.Path(__file__).resolve().parents[3] / "pipelines" / "video" / "data" / "test_clip_10s_bframes.mp4"
)

_METADATA_ROW = {
    "codec_name": "h264",
    "extradata": b"\x01\x64\x00\x1f",
    "time_base_num": 1,
    "time_base_den": 12288,
    "frame_count": 2,
    "keyframe_count": 1,
    "width": 640,
    "height": 480,
    "avc_profile_idc": 100,
    "avc_constraint_flags": 0,
    "avc_level_idc": 31,
    "nal_length_size": 4,
    "chroma_format_idc": 1,
    "bit_depth_luma_minus8": 0,
    "bit_depth_chroma_minus8": 0,
    "buffer_size_db": None,
    "max_bitrate_bps": None,
    "avg_bitrate_bps": None,
    "movie_creation_time": 0,
    "movie_modification_time": 0,
    "track_creation_time": 0,
    "track_modification_time": 0,
    "track_matrix": [0] * 9,
    "track_display_width": 0,
    "track_display_height": 0,
    "elst_bytes": None,
    "avg_frame_rate_num": 30,
    "avg_frame_rate_den": 1,
    "bit_rate_bps": 1000,
    "duration_ns": 1_000_000_000,
    "encoder_tag": None,
}

#: timescale=12288, matching _METADATA_ROW's own time_base_den -- ticks chosen so
#: ticks * 1_000_000_000 // 12288 lands on round nanosecond values, easy to hand-verify.
_TIME_BASE_NUM, _TIME_BASE_DEN = 1, 12288

_FRAME_ROWS: list[dict[str, Any]] = [
    {"dts_ticks": 12, "composition_offset_ticks": None, "is_keyframe": True, "offset": 0, "size": 100},
    {"dts_ticks": 422, "composition_offset_ticks": 6, "is_keyframe": False, "offset": 100, "size": 50},
    {"dts_ticks": 832, "composition_offset_ticks": None, "is_keyframe": False, "offset": 150, "size": 42},
]

#: A real per-video shape: composition_offset_ticks is either entirely null (no ctts) or entirely
#: populated (ctts present) for every sample -- process_one_video never mixes the two within one
#: video, unlike _FRAME_ROWS above (built for the generic Arrow round-trip test, not this).
_FRAME_ROWS_WITH_CTTS: list[dict[str, Any]] = [
    {"dts_ticks": 12, "composition_offset_ticks": 24, "is_keyframe": True, "offset": 0, "size": 100},
    {"dts_ticks": 422, "composition_offset_ticks": 0, "is_keyframe": False, "offset": 100, "size": 50},
    {"dts_ticks": 832, "composition_offset_ticks": -6, "is_keyframe": False, "offset": 150, "size": 42},
]


def _expected_ns(ticks: int) -> int:
    return (ticks * _TIME_BASE_NUM * 1_000_000_000) // _TIME_BASE_DEN


def test_mp4_metadata_from_row_reads_every_field_and_converts_track_matrix_to_a_tuple() -> None:
    """Every VIDEO_METADATA_SCHEMA field lands on the matching attribute; track_matrix becomes immutable."""
    metadata = Mp4Metadata.from_row(_METADATA_ROW)

    for field in attrs.fields(Mp4Metadata):
        actual = getattr(metadata, field.name)
        expected = _METADATA_ROW[field.name]
        if field.name == "track_matrix":
            assert actual == tuple(cast("list[int]", expected))
        else:
            assert actual == expected


def test_mp4_metadata_from_row_raises_on_a_row_missing_a_field() -> None:
    """A row out of sync with the schema (a stale producer, a bad join) fails loudly, not partially."""
    with pytest.raises(KeyError):
        Mp4Metadata.from_row({"video_id": "x", "codec_name": "h264"})


def test_mp4_frame_index_from_table_matches_the_source_rows() -> None:
    """Every FRAME_INDEX_SCHEMA field round-trips into Mp4FrameIndex's own array fields."""
    table = pa.Table.from_pylist(_FRAME_ROWS_WITH_CTTS, schema=FRAME_INDEX_SCHEMA)

    frame_index = Mp4FrameIndex.from_frame_index_table(
        table, time_base_num=_TIME_BASE_NUM, time_base_den=_TIME_BASE_DEN
    )

    assert len(frame_index) == len(_FRAME_ROWS_WITH_CTTS)
    assert frame_index.dts_ticks.tolist() == [row["dts_ticks"] for row in _FRAME_ROWS_WITH_CTTS]
    assert frame_index.dts_ns.tolist() == [_expected_ns(row["dts_ticks"]) for row in _FRAME_ROWS_WITH_CTTS]
    assert frame_index.is_keyframe.tolist() == [row["is_keyframe"] for row in _FRAME_ROWS_WITH_CTTS]
    assert frame_index.offset.tolist() == [row["offset"] for row in _FRAME_ROWS_WITH_CTTS]
    assert frame_index.size.tolist() == [row["size"] for row in _FRAME_ROWS_WITH_CTTS]
    assert frame_index.composition_offset_ticks is not None
    assert frame_index.composition_offset_ticks.tolist() == [
        row["composition_offset_ticks"] for row in _FRAME_ROWS_WITH_CTTS
    ]
    assert frame_index.composition_offset_ns is not None
    assert frame_index.composition_offset_ns.tolist() == [
        _expected_ns(row["composition_offset_ticks"]) for row in _FRAME_ROWS_WITH_CTTS
    ]


def test_mp4_frame_index_composition_offset_fields_are_none_when_every_row_is_null() -> None:
    """A track with no ctts box -- every composition_offset_ticks null -- collapses to None, not an all-null array."""
    rows = [{**row, "composition_offset_ticks": None} for row in _FRAME_ROWS]
    table = pa.Table.from_pylist(rows, schema=FRAME_INDEX_SCHEMA)

    frame_index = Mp4FrameIndex.from_frame_index_table(
        table, time_base_num=_TIME_BASE_NUM, time_base_den=_TIME_BASE_DEN
    )

    assert frame_index.composition_offset_ticks is None
    assert frame_index.composition_offset_ns is None


def test_mp4_frame_index_from_table_raises_on_partially_null_composition_offset_ticks() -> None:
    """A composition_offset_ticks column null on some rows but not others fails loudly, not as a NaN-cast crash.

    A partial mix would otherwise reach to_numpy(zero_copy_only=False), which silently
    upcasts a partially-null int64 column to float64 with NaN, violating this field's own
    NDArray[np.int64] contract several frames later with an opaque "cannot convert float
    NaN to integer" error.
    """
    table = pa.Table.from_pylist(_FRAME_ROWS, schema=FRAME_INDEX_SCHEMA)

    with pytest.raises(ValueError, match="composition_offset_ticks"):
        Mp4FrameIndex.from_frame_index_table(table, time_base_num=_TIME_BASE_NUM, time_base_den=_TIME_BASE_DEN)


def test_stream_rate_stats_is_zero_for_a_zero_duration_track() -> None:
    """A track with duration_ticks=0 (a real, malformed-but-parseable mdhd) returns zero, not a ZeroDivisionError."""
    avg_frame_rate, bit_rate_bps = _stream_rate_stats(
        frame_count=3, sample_sizes=np.array([100, 100, 100], dtype=np.int64), duration_ticks=0, timescale=12288
    )

    assert avg_frame_rate == Fraction(0)
    assert bit_rate_bps == 0


def test_stream_rate_stats_is_zero_for_a_zero_timescale_track() -> None:
    """A track with timescale=0 is likewise refused into a zero result, not a ZeroDivisionError."""
    avg_frame_rate, bit_rate_bps = _stream_rate_stats(
        frame_count=3, sample_sizes=np.array([100, 100, 100], dtype=np.int64), duration_ticks=512, timescale=0
    )

    assert avg_frame_rate == Fraction(0)
    assert bit_rate_bps == 0


def test_stream_rate_stats_computes_real_rate_and_bit_rate() -> None:
    """A real, non-degenerate track: avg_frame_rate and bit_rate_bps both come out non-zero and correct."""
    avg_frame_rate, bit_rate_bps = _stream_rate_stats(
        frame_count=30, sample_sizes=np.array([1000] * 30, dtype=np.int64), duration_ticks=12288, timescale=12288
    )

    assert avg_frame_rate == Fraction(30, 1)
    assert bit_rate_bps == 30 * 1000 * 8  # 30 frames * 1000 bytes * 8 bits, over 1 second


def test_ticks_to_ns_does_not_overflow_int64_past_a_few_hours_at_a_high_timescale() -> None:
    """A ~1,000,000 timescale (this module's own cited real-world value) wraps int64 past ~2.5 hours.

    ``ticks * time_base_num * 1e9`` alone exceeds int64's max before the division by
    ``time_base_den`` ever happens, even though the true result -- a handful of hours of
    nanoseconds -- comfortably fits. Silent wraparound would hand a caller a large negative
    timestamp instead of the true value or an explicit error.
    """
    timescale = 1_000_000
    ticks = np.array([9_223_372_037], dtype=np.int64)  # ~2 hours 33 minutes at this timescale

    result = _ticks_to_ns(ticks, 1, timescale)

    assert result.tolist() == [9_223_372_037_000]


def test_mp4_frame_index_from_video_track_matches_hand_computed_ticks() -> None:
    """A real, synthetic moov's own stts/stss/ctts/stsz/stco values land on Mp4FrameIndex exactly."""
    moov = bf.synthetic_moov(frame_count=3, keyframe_index=0, ctts_offset=100)
    video_track = parse_video_track(moov)

    frame_index = Mp4FrameIndex.from_video_track(video_track)

    assert len(frame_index) == 3
    assert frame_index.dts_ticks.tolist() == [0, 512, 1024]
    assert frame_index.composition_offset_ticks is not None
    assert frame_index.composition_offset_ticks.tolist() == [100, 100, 100]
    assert frame_index.is_keyframe.tolist() == [True, False, False]
    assert frame_index.offset.tolist() == list(video_track.offset)
    assert frame_index.size.tolist() == list(video_track.size)
    # timescale=12288 (synthetic_moov's own mdhd): ticks -> ns via the same formula the
    # constructor uses, hand-verified here against the real parsed VideoTrack.
    assert frame_index.dts_ns.tolist() == [(t * 1_000_000_000) // 12288 for t in [0, 512, 1024]]


def test_mp4_frame_index_from_video_track_defaults_is_keyframe_when_stss_absent() -> None:
    """No stss box -- every sample is a keyframe, the same spec-sanctioned inference processing.py already made."""
    moov = bf.synthetic_moov(frame_count=2, keyframe_index=None, ctts_offset=None)
    video_track = parse_video_track(moov)

    frame_index = Mp4FrameIndex.from_video_track(video_track)

    assert frame_index.is_keyframe.tolist() == [True, True]
    assert frame_index.composition_offset_ticks is None
    assert frame_index.composition_offset_ns is None


def test_mp4_metadata_from_video_track_matches_the_real_box_fields() -> None:
    """A real, synthetic moov's own stsd/mvhd/tkhd values land on Mp4Metadata exactly."""
    moov = bf.synthetic_moov(frame_count=3, keyframe_index=0, ctts_offset=None)
    video_track = parse_video_track(moov)

    metadata = Mp4Metadata.from_video_track(video_track)

    assert metadata.codec_name == "h264"
    assert metadata.time_base_num == 1
    assert metadata.time_base_den == 12288
    assert metadata.frame_count == 3
    assert metadata.keyframe_count == 1
    assert metadata.width == 640
    assert metadata.height == 480
    assert metadata.avc_profile_idc == 0x42
    assert metadata.duration_ns == 125_000_000  # 3 samples * 512 ticks each, at timescale=12288


def _video_probe(dts: list[int], duration: list[int] | None) -> VideoProbe:
    count = len(dts)
    return VideoProbe(
        index_method=VideoIndexCreationMethod.FULL_DEMUX,
        time_base=Fraction(1, 1000),
        dts=np.array(dts, dtype=np.int64),
        composition_offset=None,
        is_keyframe=np.array([index == 0 for index in range(count)]),
        offset=np.arange(count, dtype=np.int64) * 100,
        size=np.full(count, 100, dtype=np.int64),
        duration=None if duration is None else np.array(duration, dtype=np.int64),
        is_discard=np.zeros(count, dtype=np.bool_),
        codec_name="h264",
        extradata=b"\x01\x42",
        width=640,
        height=480,
        has_b_frames=False,
        avg_frame_rate=Fraction(25),
        container_format="mov,mp4,m4a,3gp,3g2,mj2",
    )


def test_mp4_metadata_from_video_probe_spans_decode_times_plus_the_last_packet_duration() -> None:
    """Duration, rates, and decoder setup come from libav; fields it does not report are None."""
    probe = _video_probe([0, 40, 80, 120], [40, 40, 40, 40])

    metadata = Mp4Metadata.from_video_probe(probe, encoder_tag="Lavc libx264", elst_bytes=b"elst")

    assert metadata.duration_ns == 160_000_000
    assert (metadata.avg_frame_rate_num, metadata.avg_frame_rate_den) == (25, 1)
    assert metadata.bit_rate_bps == 4 * 100 * 8 * 1000 // 160
    assert (metadata.time_base_num, metadata.time_base_den) == (1, 1000)
    assert (metadata.frame_count, metadata.keyframe_count) == (4, 1)
    assert (metadata.codec_name, metadata.extradata, metadata.width, metadata.height) == ("h264", b"\x01\x42", 640, 480)
    assert (metadata.encoder_tag, metadata.elst_bytes) == ("Lavc libx264", b"elst")
    assert metadata.avc_profile_idc is None
    assert metadata.track_matrix is None
    assert metadata.movie_creation_time is None


def test_mp4_metadata_from_video_probe_spans_only_decode_times_without_packet_durations() -> None:
    """With no durations reported, the last sample adds nothing to the span."""
    metadata = Mp4Metadata.from_video_probe(_video_probe([0, 40, 80], None), encoder_tag=None, elst_bytes=None)

    assert metadata.duration_ns == 80_000_000


def test_mp4_metadata_from_video_track_tolerates_a_zero_timescale_with_duration_ns_zero() -> None:
    """A malformed/zero mdhd timescale returns duration_ns=0 instead of raising."""
    moov = bf.synthetic_moov(frame_count=1, keyframe_index=0, ctts_offset=None, timescale=0)
    video_track = parse_video_track(moov)

    metadata = Mp4Metadata.from_video_track(video_track)

    assert metadata.duration_ns == 0
    assert metadata.avg_frame_rate_num == 0


def test_mp4_metadata_from_video_track_maps_non_avc_fourcc_to_hevc() -> None:
    """A hev1/hvc1 stsd entry maps to codec_name='hevc', not 'h264'."""
    moov = bf.synthetic_moov(frame_count=1, keyframe_index=0, ctts_offset=None, fourcc="hev1")
    video_track = parse_video_track(moov)

    metadata = Mp4Metadata.from_video_track(video_track)

    assert metadata.codec_name == "hevc"


def test_mp4_frame_index_from_video_track_works_for_a_fragment_sourced_table() -> None:
    """from_video_track needs no fragmented-specific constructor -- a fragment_table.py-built table just works.

    dts_ticks here are each fragment's own tfdt decode time, not accumulated stts deltas
    (fragment_table.py's own docstring), but from_video_track's own contract only cares that
    dts_ticks/offset/size/is_keyframe are real per-sample arrays in the stream's own timescale --
    it never assumes how they were produced.
    """
    moov = bf.synthetic_fragmented_moov(timescale=30, duration=3)
    moof_rows = [
        (1000, bf.pack_moof(bf.real_traf(decode_time=0, data_offset=32, sample_size=500, is_keyframe=True))),
        (1600, bf.pack_moof(bf.real_traf(decode_time=1, data_offset=32, sample_size=400, is_keyframe=False))),
    ]
    video_track = parse_fragment_table(moov, moof_rows)

    frame_index = Mp4FrameIndex.from_video_track(video_track)
    metadata = Mp4Metadata.from_video_track(video_track)

    assert frame_index.dts_ticks.tolist() == [0, 1]
    assert frame_index.offset.tolist() == [1032, 1632]
    assert frame_index.size.tolist() == [500, 400]
    assert frame_index.is_keyframe.tolist() == [True, False]
    assert frame_index.composition_offset_ticks is None
    assert metadata.codec_name == "h264"
    assert metadata.time_base_den == 30
    assert metadata.frame_count == 2
    assert metadata.keyframe_count == 1


def _extract_top_level_box(buf: bytes, box_type: str) -> bytes:
    """Find one top-level box, header included -- same approach core.sensors.video's own conftest.py uses."""
    offset = 0
    while offset < len(buf):
        found_type, box_size, _header_len = read_box_header(buf, offset, len(buf))
        if found_type == box_type:
            return buf[offset : offset + box_size]
        offset += box_size
    msg = f"no top-level {box_type!r} box in {len(buf)}-byte buffer"
    raise ValueError(msg)


def _real_bframes_frame_index() -> Mp4FrameIndex:
    mp4_bytes = _BFRAMES_MP4.read_bytes()
    moov_bytes = _extract_top_level_box(mp4_bytes, "moov")
    video_track = parse_video_track(moov_bytes)
    return Mp4FrameIndex.from_video_track(video_track)


_REAL_MP4_FIXTURES_DIR = pathlib.Path(__file__).resolve().parents[3] / "pipelines" / "video" / "data"
_REAL_MP4_FIXTURES = (
    _REAL_MP4_FIXTURES_DIR / "test_clip_10s.mp4",
    _BFRAMES_MP4,
    _REAL_MP4_FIXTURES_DIR / "test_video_30s.mp4",
)


@pytest.mark.parametrize("mp4_path", _REAL_MP4_FIXTURES, ids=lambda p: p.name)
def test_mp4_frame_index_matches_pyav_on_real_fixtures(mp4_path: pathlib.Path) -> None:
    """offset/size/is_keyframe/pts_ns match PyAV's own make_index_and_metadata, sample for sample, on real files.

    Every other parser test in this package uses hand-packed synthetic boxes, which checks this
    reader against our own reading of the spec, not against real files. This checks it against
    PyAV -- the VideoIndex CameraSensor/multimodal_split already use today -- matched by byte
    offset (a unique key into either index) since the two readers order their arrays differently
    (decode order here, presentation order there).
    """
    mp4_bytes = mp4_path.read_bytes()
    moov_bytes = _extract_top_level_box(mp4_bytes, "moov")
    video_track = parse_video_track(moov_bytes)
    frame_index = Mp4FrameIndex.from_video_track(video_track)

    pyav_index, _metadata = make_index_and_metadata(mp4_path, index_method=VideoIndexCreationMethod.FULL_DEMUX)

    def _by_offset(
        offset: np.ndarray, size: np.ndarray, is_keyframe: np.ndarray, pts_ns: np.ndarray
    ) -> dict[int, tuple[int, bool, int]]:
        return {
            int(o): (int(s), bool(k), int(p))
            for o, s, k, p in zip(offset.tolist(), size.tolist(), is_keyframe.tolist(), pts_ns.tolist(), strict=True)
        }

    ours = _by_offset(frame_index.offset, frame_index.size, frame_index.is_keyframe, frame_index.pts_ns)
    pyav = _by_offset(pyav_index.offset, pyav_index.size, pyav_index.is_keyframe, pyav_index.pts_ns)
    assert ours == pyav


def test_mp4_frame_index_keyframe_indices_matches_hand_computed_flatnonzero() -> None:
    """keyframe_indices is exactly flatnonzero(is_keyframe), already sorted -- checked against a real B-frame clip."""
    frame_index = _real_bframes_frame_index()

    assert frame_index.keyframe_indices.tolist() == np.flatnonzero(frame_index.is_keyframe).tolist()
    assert len(frame_index.keyframe_indices) > 0
    assert np.array_equal(frame_index.keyframe_indices, np.sort(frame_index.keyframe_indices))


def test_mp4_frame_index_pts_ns_matches_hand_computed_dts_plus_composition_offset_minus_edit_list_shift() -> None:
    """pts_ns = dts_ns + (composition_offset_ns or 0) - edit_list_shift_ns, computed once, real B-frame data.

    This clip's own elst has a single normal-rate entry with media_time=1024 ticks at a
    12288 timescale -- 83_333_333 ns, hand-verified directly against its own bytes.
    """
    frame_index = _real_bframes_frame_index()

    assert frame_index.composition_offset_ns is not None  # a real B-frame clip has a nonzero ctts
    edit_list_shift_ns = 1024 * 1_000_000_000 // 12288
    expected_pts_ns = frame_index.dts_ns + frame_index.composition_offset_ns - edit_list_shift_ns
    assert np.array_equal(frame_index.pts_ns, expected_pts_ns)
    assert frame_index.pts_ns[0] == 0  # the clip's first frame is exactly the edit list's own trim point


def test_mp4_frame_index_pts_sort_order_is_a_valid_permutation_that_sorts_pts_ns() -> None:
    """pts_sort_order is a real permutation of decode-order indices, and pts_ns[pts_sort_order] is sorted."""
    frame_index = _real_bframes_frame_index()

    assert sorted(frame_index.pts_sort_order.tolist()) == list(range(len(frame_index)))
    sorted_pts = frame_index.pts_ns[frame_index.pts_sort_order]
    assert np.array_equal(sorted_pts, np.sort(frame_index.pts_ns))


def test_mp4_frame_index_pts_ns_equals_dts_ns_when_no_composition_offset() -> None:
    """No ctts box -- pts_ns is exactly dts_ns, and pts_sort_order matches dts_ns's own sort order."""
    moov = bf.synthetic_moov(frame_count=4, keyframe_index=0, ctts_offset=None)
    video_track = parse_video_track(moov)

    frame_index = Mp4FrameIndex.from_video_track(video_track)

    assert frame_index.composition_offset_ns is None
    assert np.array_equal(frame_index.pts_ns, frame_index.dts_ns)


def test_mp4_frame_index_pts_ns_shifts_by_a_single_normal_rate_edit_lists_own_media_time() -> None:
    """A single-entry, normal-rate elst shifts pts_ns back by media_time, converted to ns -- dts_ns untouched."""
    elst_content = bf.edit_list_box(entries=[(4 * 512, 256, 1, 0)])  # media_time=256 ticks at a 12288 timescale
    moov = bf.synthetic_moov(
        frame_count=4, keyframe_index=0, ctts_offset=None, include_elst=True, elst_content=elst_content
    )
    video_track = parse_video_track(moov)

    frame_index = Mp4FrameIndex.from_video_track(video_track)

    expected_shift_ns = 256 * 1_000_000_000 // 12288
    assert np.array_equal(frame_index.pts_ns, frame_index.dts_ns - expected_shift_ns)
    assert np.array_equal(frame_index.dts_ns, _ticks_to_ns(video_track.dts_ticks, 1, 12288))


def test_mp4_frame_index_pts_ns_unaffected_by_a_zero_media_time_edit_list() -> None:
    """An elst present but with media_time=0 (the fixture default) is a no-op shift, not skipped detection."""
    moov = bf.synthetic_moov(frame_count=4, keyframe_index=0, ctts_offset=None, include_elst=True)
    video_track = parse_video_track(moov)

    frame_index = Mp4FrameIndex.from_video_track(video_track)

    assert np.array_equal(frame_index.pts_ns, frame_index.dts_ns)


def test_mp4_frame_index_raises_on_a_multi_entry_edit_list() -> None:
    """More than one elst entry is refused, not silently reduced to the first."""
    elst_content = bf.edit_list_box(entries=[(1000, 0, 1, 0), (1000, 512, 1, 0)])
    moov = bf.synthetic_moov(
        frame_count=4, keyframe_index=0, ctts_offset=None, include_elst=True, elst_content=elst_content
    )
    video_track = parse_video_track(moov)

    with pytest.raises(UnsupportedEditListError, match="entries"):
        Mp4FrameIndex.from_video_track(video_track)


def test_mp4_frame_index_raises_on_an_empty_edit_list_entry() -> None:
    """media_time=-1 (an empty edit -- "insert a gap here") is refused, not treated as a zero shift."""
    elst_content = bf.edit_list_box(entries=[(1000, -1, 1, 0)])
    moov = bf.synthetic_moov(
        frame_count=4, keyframe_index=0, ctts_offset=None, include_elst=True, elst_content=elst_content
    )
    video_track = parse_video_track(moov)

    with pytest.raises(UnsupportedEditListError, match="empty edit"):
        Mp4FrameIndex.from_video_track(video_track)


def test_mp4_frame_index_raises_on_a_non_normal_rate_edit_list() -> None:
    """A media_rate other than 1x (a real, spec-legal slow-motion/fast-forward edit) is refused."""
    elst_content = bf.edit_list_box(entries=[(1000, 0, 2, 0)])
    moov = bf.synthetic_moov(
        frame_count=4, keyframe_index=0, ctts_offset=None, include_elst=True, elst_content=elst_content
    )
    video_track = parse_video_track(moov)

    with pytest.raises(UnsupportedEditListError, match="rate"):
        Mp4FrameIndex.from_video_track(video_track)


def test_mp4_frame_index_raises_on_a_version_1_edit_list_with_an_unsupported_shape() -> None:
    """A version-1 (64-bit) elst decodes its own wider fields -- and still refuses an empty edit."""
    elst_content = bf.edit_list_box(entries=[(1000, -1, 1, 0)], version=1)
    moov = bf.synthetic_moov(
        frame_count=4, keyframe_index=0, ctts_offset=None, include_elst=True, elst_content=elst_content
    )
    video_track = parse_video_track(moov)

    with pytest.raises(UnsupportedEditListError, match="empty edit"):
        Mp4FrameIndex.from_video_track(video_track)


def test_unsupported_edit_list_error_is_an_unindexable_video_error() -> None:
    """processing.py's per-video catch (UnindexableVideoError) must reach this -- not just ValueError."""
    assert issubclass(UnsupportedEditListError, UnindexableVideoError)


def test_mp4_frame_index_raises_on_a_truncated_edit_list() -> None:
    """A real elst box's bytes cut short mid-entry fails loudly as UnsupportedEditListError, not a struct.error."""
    moov = bf.synthetic_moov(frame_count=4, keyframe_index=0, ctts_offset=None)
    video_track = parse_video_track(moov)
    truncated_elst_bytes = bf.elst_box()[:14]  # cuts off before entry_count's own 4 bytes can be read in full
    video_track = attrs.evolve(video_track, elst_bytes=truncated_elst_bytes)

    with pytest.raises(UnsupportedEditListError, match="truncated"):
        Mp4FrameIndex.from_video_track(video_track)


def test_ticks_to_ns_raises_on_zero_time_base_denominator() -> None:
    """A zero time_base_den (a real, malformed-but-parseable mdhd with timescale=0) fails loudly, not a crash."""
    with pytest.raises(InvalidTimeBaseError, match="0"):
        _ticks_to_ns(np.array([1, 2, 3], dtype=np.int64), 1, 0)


def test_mp4_frame_index_raises_on_a_zero_timescale_video_track() -> None:
    """A VideoTrack with timescale=0 fails loudly as InvalidTimeBaseError, not a ZeroDivisionError."""
    moov = bf.synthetic_moov(frame_count=4, keyframe_index=0, ctts_offset=None)
    video_track = parse_video_track(moov)
    video_track = attrs.evolve(video_track, timescale=0)

    with pytest.raises(InvalidTimeBaseError):
        Mp4FrameIndex.from_video_track(video_track)


def _frame_table_with_timing(timing: pa.Table) -> pa.Table:
    rows = [
        {"dts_ticks": i * 10, "composition_offset_ticks": None, "is_keyframe": i == 0, "offset": i * 100, "size": 100}
        for i in range(len(timing))
    ]
    table = pa.Table.from_pylist(rows, schema=FRAME_INDEX_SCHEMA).drop(["eof_ns", "sof_ns", "moe_ns"])
    for field in timing.schema:
        if field.name in ("eof_ns", "sof_ns", "moe_ns"):
            table = table.append_column(field, timing.column(field.name))
    return table


def _index_with_timing(timing: pa.Table) -> Mp4FrameIndex:
    return Mp4FrameIndex.from_frame_index_table(
        _frame_table_with_timing(timing), time_base_num=1, time_base_den=1_000_000
    )


#: A timing record with EOF, SOF, and MOE populated.
_ROWS_WITH_SOF_MOE: list[dict[str, Any]] = [
    {"eof_ns": 1_000_000, "sof_ns": 900_000, "moe_ns": 950_000},
    {"eof_ns": 2_000_000, "sof_ns": 1_900_000, "moe_ns": 1_950_000},
    {"eof_ns": 3_000_000, "sof_ns": 2_900_000, "moe_ns": 2_950_000},
]

#: A timing record carrying EOF only; optional SOF/MOE are absent.
_ROWS_EOF_ONLY: list[dict[str, Any]] = [
    {"eof_ns": 1_000_000, "sof_ns": None, "moe_ns": None},
    {"eof_ns": 2_000_000, "sof_ns": None, "moe_ns": None},
]


def test_mp4_frame_index_timing_reads_every_field_when_sof_moe_present() -> None:
    """A real v2+ box shape: eof_ns/sof_ns/moe_ns all populated and round-trip exactly."""
    table = pa.Table.from_pylist(_ROWS_WITH_SOF_MOE, schema=FRAME_INDEX_SCHEMA)

    camera_timing = _index_with_timing(table)

    assert len(camera_timing) == 3
    assert camera_timing.eof_ns is not None
    assert camera_timing.eof_ns.tolist() == [row["eof_ns"] for row in _ROWS_WITH_SOF_MOE]
    assert camera_timing.sof_ns is not None
    assert camera_timing.sof_ns.tolist() == [row["sof_ns"] for row in _ROWS_WITH_SOF_MOE]
    assert camera_timing.moe_ns is not None
    assert camera_timing.moe_ns.tolist() == [row["moe_ns"] for row in _ROWS_WITH_SOF_MOE]


def test_mp4_frame_index_timing_collapses_all_null_sof_moe_to_none() -> None:
    """A real v0/v1 box shape: sof_ns/moe_ns entirely null collapse to None, not all-null arrays."""
    table = pa.Table.from_pylist(_ROWS_EOF_ONLY, schema=FRAME_INDEX_SCHEMA)

    camera_timing = _index_with_timing(table)

    assert len(camera_timing) == 2
    assert camera_timing.eof_ns is not None
    assert camera_timing.eof_ns.tolist() == [row["eof_ns"] for row in _ROWS_EOF_ONLY]
    assert camera_timing.sof_ns is None
    assert camera_timing.moe_ns is None


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("eof_present", [False, True])
def test_mp4_frame_index_preserves_partial_timing_masks_and_int64_precision(
    *, chunked: bool, eof_present: bool
) -> None:
    """Different SOF/MOE gaps preserve decode order and integers above float64 precision."""
    large = (1 << 62) + 1
    table = pa.Table.from_pylist(
        [
            {"eof_ns": large + 100 if eof_present else None, "sof_ns": large, "moe_ns": None},
            {"eof_ns": large + 200 if eof_present else None, "sof_ns": None, "moe_ns": large + 2},
            {"eof_ns": large + 300 if eof_present else None, "sof_ns": large + 1, "moe_ns": large + 3},
        ],
        schema=FRAME_INDEX_SCHEMA,
    )
    if chunked:
        table = pa.concat_tables([table.slice(0, 1), table.slice(1)])

    index = _index_with_timing(table)

    if eof_present:
        assert index.eof_ns is not None
        assert index.eof_ns.tolist() == [large + 100, large + 200, large + 300]
    else:
        assert index.eof_ns is None
    for name, values, mask in (
        ("sof_ns", [large, None, large + 1], [False, True, False]),
        ("moe_ns", [None, large + 2, large + 3], [True, False, False]),
    ):
        column = getattr(index, name)
        assert isinstance(column, np.ma.MaskedArray)
        assert column.dtype == np.int64
        assert column.tolist() == values
        assert np.ma.getmaskarray(column).tolist() == mask
        assert column.compressed().tolist() == [value for value in values if value is not None]


@pytest.mark.parametrize("construction", ["direct", "evolve"])
@pytest.mark.parametrize("field", ["sof_ns", "moe_ns"])
@pytest.mark.parametrize("eof_present", [False, True])
def test_optional_timing_accepts_masked_int64_arrays(construction: str, field: str, *, eof_present: bool) -> None:
    """Direct construction and attrs.evolve preserve the same validity and integer contract."""
    index = _index_with_timing(pa.Table.from_pylist(_ROWS_EOF_ONLY, schema=FRAME_INDEX_SCHEMA))
    values = np.ma.array(np.array([(1 << 62) + 1, 0], dtype=np.int64), mask=[False, True])
    construct = (
        partial(Mp4FrameIndex, **attrs.asdict(index, recurse=False))
        if construction == "direct"
        else partial(attrs.evolve, index)
    )
    updated = construct(**{field: values, "eof_ns": index.eof_ns if eof_present else None})
    assert (updated.eof_ns is not None) == eof_present
    assert getattr(updated, field) is values
    assert getattr(updated, field).tolist() == [(1 << 62) + 1, None]


@pytest.mark.parametrize("construction", ["direct", "evolve"])
def test_eof_rejects_masked_missing_frames(construction: str) -> None:
    """When supplied, EOF remains a complete int64 array on every construction path."""
    index = _index_with_timing(pa.Table.from_pylist(_ROWS_EOF_ONLY, schema=FRAME_INDEX_SCHEMA))
    construct = (
        partial(Mp4FrameIndex, **attrs.asdict(index, recurse=False))
        if construction == "direct"
        else partial(attrs.evolve, index)
    )
    with pytest.raises(ValueError, match="eof_ns"):
        construct(eof_ns=np.ma.array(np.array([10, 20], dtype=np.int64), mask=[False, True]))


@pytest.mark.parametrize("fields", [("sof_ns",), ("moe_ns",), ("sof_ns", "moe_ns")])
def test_optional_timing_reads_without_eof(fields: tuple[str, ...]) -> None:
    """SOF-only, MOE-only, and combined upstream timing are usable independently of EOF."""
    large = (1 << 62) + 1
    rows = [
        {"eof_ns": None, "sof_ns": None, "moe_ns": None, **dict.fromkeys(fields, value)} for value in (large, large + 2)
    ]
    index = _index_with_timing(pa.Table.from_pylist(rows, schema=FRAME_INDEX_SCHEMA))
    assert index.eof_ns is None
    for field in ("sof_ns", "moe_ns"):
        values = getattr(index, field)
        if field in fields:
            assert values.dtype == np.int64
            assert values.tolist() == [large, large + 2]
        else:
            assert values is None


def test_float_timing_is_rejected() -> None:
    """Readers must not accept timing data that lost integer precision."""
    table = pa.table({"eof_ns": [1.0], "sof_ns": [None], "moe_ns": [None]})
    with pytest.raises(ValueError, match="int64"):
        _index_with_timing(table)


def test_mp4_frame_index_preserves_recorder_clock_with_presentation_reordering_and_edit_shift() -> None:
    """Container sorting and time shifts must preserve exact recorder nanoseconds in decode order."""
    eof = [9_007_199_254_740_993, 9_007_199_254_741_003, 9_007_199_254_741_013]
    timing = pa.Table.from_pylist(
        [{"eof_ns": value, "sof_ns": None, "moe_ns": value - 5} for value in eof],
        schema=FRAME_INDEX_SCHEMA,
    )
    frames = _frame_table_with_timing(timing)
    frames = frames.set_column(
        frames.schema.get_field_index("composition_offset_ticks"),
        FRAME_INDEX_SCHEMA.field("composition_offset_ticks"),
        pa.array([30, 0, -5], type=pa.int64()),
    )
    index = Mp4FrameIndex.from_frame_index_table(
        frames,
        time_base_num=1,
        time_base_den=1_000_000,
        elst_bytes=bf.edit_list_box(entries=[(1000, 5, 1, 0)]),
    )
    assert index.dts_ns.tolist() == [0, 10_000, 20_000]
    assert index.pts_ns.tolist() == [25_000, 5_000, 10_000]
    assert index.pts_sort_order.tolist() == [1, 2, 0]
    assert index.eof_ns is not None
    assert index.eof_ns.dtype == np.int64
    assert index.eof_ns.tolist() == eof
    assert index.sof_ns is None
    assert index.moe_ns is not None
    assert index.moe_ns.tolist() == [value - 5 for value in eof]


def test_ordinary_mp4_frame_index_has_no_recorder_timing() -> None:
    """Ordinary Arrow rows and parsed tracks do not synthesize recorder timestamps."""
    table = pa.Table.from_pylist(_FRAME_ROWS_WITH_CTTS, schema=FRAME_INDEX_SCHEMA)
    index = Mp4FrameIndex.from_frame_index_table(table, time_base_num=1, time_base_den=1_000_000)
    assert index.eof_ns is index.sof_ns is index.moe_ns is None
    track_index = Mp4FrameIndex.from_video_track(
        parse_video_track(bf.synthetic_moov(frame_count=2, keyframe_index=0, ctts_offset=None))
    )
    assert track_index.eof_ns is track_index.sof_ns is track_index.moe_ns is None


def test_partially_null_eof_is_rejected() -> None:
    """A supplied EOF column must be fully populated; it has no per-frame validity mask."""
    timing = pa.Table.from_pylist(
        [{"eof_ns": 1_000_000, "sof_ns": None, "moe_ns": None}, {"eof_ns": None, "sof_ns": None, "moe_ns": None}],
        schema=FRAME_INDEX_SCHEMA,
    )
    with pytest.raises(ValueError, match="eof_ns"):
        _index_with_timing(timing)


@pytest.mark.parametrize("construction", ["direct", "evolve"])
@pytest.mark.parametrize("field", ["eof_ns", "sof_ns", "moe_ns"])
@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.array([1.0, 2.0]), "dtype int64"),
        (np.array([[1, 2]], dtype=np.int64), "1-D"),
        (np.array([1], dtype=np.int64), "length must equal the frame count"),
    ],
)
def test_camera_timing_fields_validate_all_construction_paths(
    construction: str, field: str, value: npt.NDArray[np.int64] | npt.NDArray[np.float64], message: str
) -> None:
    """Model validation applies even when no Arrow reader is involved."""
    index = _index_with_timing(pa.Table.from_pylist(_ROWS_EOF_ONLY, schema=FRAME_INDEX_SCHEMA))
    changes = {field: value}
    construct = (
        partial(Mp4FrameIndex, **attrs.asdict(index, recurse=False))
        if construction == "direct"
        else partial(attrs.evolve, index)
    )
    with pytest.raises(ValueError, match=message):
        construct(**changes)


@pytest.mark.parametrize("construction", ["direct", "evolve"])
@pytest.mark.parametrize("fields", [("sof_ns",), ("moe_ns",), ("sof_ns", "moe_ns")])
def test_camera_timing_fields_are_independent_of_eof(construction: str, fields: tuple[str, ...]) -> None:
    """Direct construction and attrs.evolve accept optional timing while removing EOF."""
    index = _index_with_timing(pa.Table.from_pylist(_ROWS_EOF_ONLY, schema=FRAME_INDEX_SCHEMA))
    values = np.array([(1 << 62) + 1, (1 << 62) + 3], dtype=np.int64)
    changes = {"eof_ns": None, **dict.fromkeys(fields, values)}
    construct = (
        partial(Mp4FrameIndex, **attrs.asdict(index, recurse=False))
        if construction == "direct"
        else partial(attrs.evolve, index)
    )
    updated = construct(**changes)
    assert updated.eof_ns is None
    for field in ("sof_ns", "moe_ns"):
        assert getattr(updated, field) is (values if field in fields else None)


@pytest.mark.parametrize("field", ["sof_ns", "moe_ns"])
def test_camera_timing_field_validation_preserves_values_and_optional_columns(field: str) -> None:
    """Zero, negative, and nonmonotonic upstream timestamps remain untouched."""
    index = _index_with_timing(pa.Table.from_pylist(_ROWS_WITH_SOF_MOE, schema=FRAME_INDEX_SCHEMA))
    values = np.array([0, -1, -1], dtype=np.int64)
    updated = attrs.evolve(index, eof_ns=values, **{field: values})
    np.testing.assert_array_equal(updated.eof_ns, values)
    np.testing.assert_array_equal(getattr(updated, field), values)
    without_optional = attrs.evolve(updated, sof_ns=None, moe_ns=None)
    assert without_optional.sof_ns is without_optional.moe_ns is None
    ordinary = attrs.evolve(without_optional, eof_ns=None)
    assert ordinary.eof_ns is ordinary.sof_ns is ordinary.moe_ns is None
