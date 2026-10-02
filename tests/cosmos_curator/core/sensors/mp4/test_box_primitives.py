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
"""Tests for the generic ISO-BMFF box-walking primitives any reader in this package can share."""

import struct

import pytest

from cosmos_curator.core.sensors.mp4 import box_primitives as bp
from tests.cosmos_curator.core.sensors.mp4._box_fixtures import pack_box as _box


def test_read_box_header_standard_32_bit_size() -> None:
    """A standard 32-bit size box reads back type, size, and an 8-byte header."""
    buf = _box("free", b"1234")
    box_type, size, header_len = bp.read_box_header(buf, 0, len(buf))
    assert (box_type, size, header_len) == ("free", 12, 8)


def test_read_box_header_extended_64_bit_size() -> None:
    """A size-field-1 box uses the following 8 bytes as a 64-bit size."""
    content = b"x" * 10
    buf = struct.pack(">I", 1) + b"free" + struct.pack(">Q", 16 + len(content)) + content
    box_type, size, header_len = bp.read_box_header(buf, 0, len(buf))
    assert (box_type, size, header_len) == ("free", 16 + len(content), 16)


def test_read_box_header_size_zero_extends_to_buffer_end() -> None:
    """A size-field-0 box extends to the end of the buffer it was read from."""
    buf = struct.pack(">I", 0) + b"free" + b"trailing content"
    box_type, size, header_len = bp.read_box_header(buf, 0, len(buf))
    assert box_type == "free"
    assert size == len(buf)
    assert header_len == 8


def test_read_box_header_accepts_a_standard_size_exactly_equal_to_its_own_header() -> None:
    """An empty box (size == header_len, no content) is well-formed, not malformed."""
    buf = struct.pack(">I", 8) + b"free"
    box_type, size, header_len = bp.read_box_header(buf, 0, len(buf))
    assert (box_type, size, header_len) == ("free", 8, 8)


def test_read_box_header_raises_when_standard_size_is_smaller_than_its_own_header() -> None:
    """A standard-size box declaring a total size smaller than its own 8-byte header is malformed.

    Undetected, this produces a content range where start > end (offset + header_len > offset +
    box_size) and advances the walk by fewer bytes than a header occupies -- silent corruption,
    not a crash, in any caller that doesn't itself re-check the returned box_size.
    """
    buf = struct.pack(">I", 4) + b"free"
    with pytest.raises(bp.MalformedBoxHeaderError):
        bp.read_box_header(buf, 0, len(buf))


def test_read_box_header_accepts_an_extended_size_exactly_equal_to_its_own_header() -> None:
    """An empty extended-size box (size64 == 16, no content) is well-formed, not malformed."""
    buf = struct.pack(">I", 1) + b"free" + struct.pack(">Q", 16)
    box_type, size, header_len = bp.read_box_header(buf, 0, len(buf))
    assert (box_type, size, header_len) == ("free", 16, 16)


def test_read_box_header_raises_when_extended_size_is_smaller_than_its_own_header() -> None:
    """A size-field-1 box whose 64-bit size is smaller than its own 16-byte header is malformed."""
    buf = struct.pack(">I", 1) + b"free" + struct.pack(">Q", 10)
    with pytest.raises(bp.MalformedBoxHeaderError):
        bp.read_box_header(buf, 0, len(buf))


def test_iter_child_boxes_walks_siblings() -> None:
    """Every sibling box at one level is returned, in order."""
    buf = _box("aaaa", b"1") + _box("bbbb", b"22") + _box("cccc", b"333")
    children = bp.iter_child_boxes(buf, 0, len(buf))
    assert [c[0] for c in children] == ["aaaa", "bbbb", "cccc"]


def test_iter_child_boxes_raises_when_a_declared_size_overruns_the_parent_end() -> None:
    """A child declaring a size past its own parent's end is refused, not handed back as-is.

    Every content parser downstream trusts these bounds -- an unbounded child would let a
    ``stbl`` whose declared size overruns its ``minf`` parent read sample tables out of
    whatever bytes follow, producing a plausible-looking but wrong VideoTrack.
    """
    buf = struct.pack(">I", 100) + b"abcd" + b"\x00" * 200
    with pytest.raises(bp.MalformedBoxHeaderError):
        bp.iter_child_boxes(buf, 0, 16)


def test_iter_child_boxes_stops_before_a_header_shorter_than_standard_header_bytes() -> None:
    """A trailing run too short to hold a full header is skipped, not read into past *end*."""
    buf = _box("free", b"\x00" * 8) + b"zz"
    children = bp.iter_child_boxes(buf, 0, len(buf))
    assert [c[0] for c in children] == ["free"]


def test_find_box_descends_a_path() -> None:
    """A multi-segment path descends through nested boxes."""
    inner = _box("mdhd", b"target")
    mdia = _box("mdia", inner)
    trak = _box("trak", mdia)
    start, end = bp.find_box(trak, ("mdia", "mdhd"), 8, len(trak))
    assert trak[start:end] == b"target"


def test_find_box_raises_on_missing_segment() -> None:
    """A missing segment anywhere in the path raises, not just the last one."""
    trak = _box("trak", _box("mdia", b""))
    with pytest.raises(bp.BoxNotFoundError):
        bp.find_box(trak, ("mdia", "stbl"), 8, len(trak))


def test_find_all_boxes_returns_every_top_level_match() -> None:
    """Two same-typed siblings at one level are both returned, not just the first."""
    buf = _box("trak", b"1") + _box("free", b"") + _box("trak", b"22")
    matches = bp.find_all_boxes(buf, "trak", 0, len(buf))
    assert [buf[start:end] for start, end in matches] == [b"1", b"22"]


def test_find_video_trak_skips_a_trak_with_no_video_sample_entry() -> None:
    """A non-video (e.g. audio) trak is skipped in favor of the one with a video sample entry."""
    audio_stsd = _box("stsd", struct.pack(">II", 0, 1) + _box("mp4a", b"\x00" * 20))
    audio_stbl = _box("stbl", audio_stsd)
    audio_trak = _box("trak", _box("mdia", _box("minf", audio_stbl)))

    video_entry = struct.pack(">6sH", b"\x00" * 6, 1) + b"\x00" * 78  # minimal fixed VisualSampleEntry fields
    video_stsd = _box("stsd", struct.pack(">II", 0, 1) + _box("avc1", video_entry))
    video_stbl = _box("stbl", video_stsd)
    video_trak = _box("trak", _box("mdia", _box("minf", video_stbl)))

    moov = _box("moov", audio_trak + video_trak)
    trak_start, trak_end, stbl_start, stbl_end, stsd_start, stsd_end, fourcc = bp.find_video_trak(moov, 8, len(moov))

    assert fourcc == "avc1"
    assert moov[trak_start:trak_end] == video_trak[8:]
    assert moov[stbl_start:stbl_end] == video_stbl[8:]
    assert moov[stsd_start:stsd_end] == video_stsd[8:]


def test_find_video_trak_raises_when_no_trak_has_a_video_sample_entry() -> None:
    """A moov with only non-video (or no) traks refuses rather than picking one arbitrarily."""
    audio_stsd = _box("stsd", struct.pack(">II", 0, 1) + _box("mp4a", b"\x00" * 20))
    audio_trak = _box("trak", _box("mdia", _box("minf", _box("stbl", audio_stsd))))
    moov = _box("moov", audio_trak)

    with pytest.raises(bp.BoxNotFoundError):
        bp.find_video_trak(moov, 8, len(moov))


def _video_trak(fourcc: str = "avc1") -> bytes:
    video_entry = struct.pack(">6sH", b"\x00" * 6, 1) + b"\x00" * 78  # minimal fixed VisualSampleEntry fields
    video_stsd = _box("stsd", struct.pack(">II", 0, 1) + _box(fourcc, video_entry))
    video_stbl = _box("stbl", video_stsd)
    return _box("trak", _box("mdia", _box("minf", video_stbl)))


def test_find_video_trak_raises_ambiguous_when_more_than_one_trak_has_a_video_sample_entry() -> None:
    """A moov with two video-shaped traks refuses rather than silently picking the first."""
    moov = _box("moov", _video_trak() + _video_trak())

    with pytest.raises(bp.AmbiguousVideoTrackError):
        bp.find_video_trak(moov, 8, len(moov))


def test_video_fourcc_codec_names_is_the_single_source_of_truth_for_accepted_fourccs() -> None:
    """The video-track acceptance set is derived from the codec-name table, not a second hand-kept list.

    Guards against the two drifting apart silently: a fourcc this package accepts as a video track
    must always have a codec name, and vice versa -- see VIDEO_FOURCC_CODEC_NAMES's own comment.
    """
    assert frozenset(bp.VIDEO_FOURCC_CODEC_NAMES) == bp._VIDEO_SAMPLE_ENTRY_TYPES


def test_video_fourcc_codec_names_raises_on_an_unrecognized_fourcc() -> None:
    """A fourcc outside the table fails loudly rather than being silently mislabeled."""
    with pytest.raises(KeyError):
        _ = bp.VIDEO_FOURCC_CODEC_NAMES["av01"]
