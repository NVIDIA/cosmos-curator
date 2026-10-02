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
"""Tests for the faithful, moof/traf/trun-sourced ``VideoTrack`` decode in ``fragment_table.py``.

Focus is on the two things genuinely new here: per-fragment sample data (offset/size/keyframe/
decode-time from tfhd/tfdt/trun) and the refusal policy for any shape not yet proven against a
real corpus file -- everything else (mvhd/tkhd/mdhd/stsd) is exercised by ``test_moov.py``
already, since this module reuses those parsers verbatim.
"""

import pytest

from cosmos_curator.core.sensors.mp4 import fragment_table as ft
from cosmos_curator.core.sensors.mp4.box_primitives import BoxNotFoundError, MalformedBoxHeaderError
from tests.cosmos_curator.core.sensors.mp4 import _box_fixtures as bf


def _moof(
    *,
    moof_start: int,
    decode_time: int,
    data_offset: int,
    sample_size: int,
    is_keyframe: bool = True,
    default_sample_duration: int | None = 3000,
    default_base_is_moof: bool = True,
    sample_count: int = 1,
    composition_offset: int | None = None,
) -> tuple[int, bytes]:
    traf = bf.real_traf(
        decode_time=decode_time,
        data_offset=data_offset,
        sample_size=sample_size,
        is_keyframe=is_keyframe,
        default_sample_duration=default_sample_duration,
        default_base_is_moof=default_base_is_moof,
        sample_count=sample_count,
        composition_offset=composition_offset,
    )
    return moof_start, bf.pack_moof(traf)


def test_parse_fragment_table_end_to_end() -> None:
    """A fully synthetic fragmented moov + moof rows decode into the expected SoA shape."""
    moov = bf.synthetic_fragmented_moov(timescale=30, duration=90)
    moof_rows = [
        _moof(moof_start=1000, decode_time=0, data_offset=32, sample_size=500, is_keyframe=True),
        _moof(moof_start=1600, decode_time=1, data_offset=32, sample_size=400, is_keyframe=False),
        _moof(moof_start=2100, decode_time=2, data_offset=32, sample_size=420, is_keyframe=False),
    ]

    table = ft.parse_fragment_table(moov, moof_rows)

    assert table.timescale == 30
    assert table.duration == 90
    assert table.sample_description.fourcc == "avc1"
    assert table.sample_description.width == 640
    assert table.sample_description.height == 480
    assert table.dts_ticks.tolist() == [0, 1, 2]
    assert table.size.tolist() == [500, 400, 420]
    assert table.offset.tolist() == [1032, 1632, 2132]  # moof_start + data_offset
    assert table.is_keyframe is not None
    assert table.is_keyframe.tolist() == [True, False, False]
    assert table.composition_offsets is None


def test_parse_fragment_table_raises_malformed_box_header_error_on_truncated_moov() -> None:
    """A moov cut short partway through parsing (not just at the header) is refused, not crashed on."""
    moov = bf.synthetic_fragmented_moov(timescale=30, duration=90)
    truncated = moov[: len(moov) // 2]

    with pytest.raises(MalformedBoxHeaderError, match="ran out mid-parse"):
        ft.parse_fragment_table(truncated, [])


def test_parse_fragment_table_falls_back_to_trex_when_trun_and_tfhd_omit_sample_size() -> None:
    """A trun/tfhd that both omit sample size falls back to mvex/trex's own default."""
    moov = bf.synthetic_fragmented_moov()
    # real_trun always sets sample-size-present; build a traf by hand without it to exercise
    # the trex fallback specifically.
    tfhd = bf.real_tfhd(default_sample_duration=None)
    tfdt = bf.real_tfdt(base_media_decode_time=5)
    trun_flags = bf._TRUN_DATA_OFFSET_PRESENT | bf._TRUN_SAMPLE_FLAGS_PRESENT
    trun_content = bytes([0]) + trun_flags.to_bytes(3, "big")
    trun_content += (1).to_bytes(4, "big") + (32).to_bytes(4, "big", signed=True) + (0).to_bytes(4, "big")
    trun = bf.pack_box("trun", trun_content)
    traf = bf.pack_box("traf", tfhd + tfdt + trun)
    moof_rows = [(2000, bf.pack_moof(traf))]

    table = ft.parse_fragment_table(moov, moof_rows)

    assert table.size.tolist() == [0]  # trex_box()'s own default_sample_size default


def test_parse_fragment_table_raises_when_sample_size_is_unresolved_by_trun_tfhd_and_trex() -> None:
    """No mvex/trex at all, and trun/tfhd both omit sample size: every fallback is exhausted -- refused."""
    moov = bf.synthetic_fragmented_moov(include_trex=False)
    tfhd = bf.real_tfhd(default_sample_duration=None)
    tfdt = bf.real_tfdt(base_media_decode_time=5)
    trun_flags = bf._TRUN_DATA_OFFSET_PRESENT | bf._TRUN_SAMPLE_FLAGS_PRESENT
    trun_content = bytes([0]) + trun_flags.to_bytes(3, "big")
    trun_content += (1).to_bytes(4, "big") + (32).to_bytes(4, "big", signed=True) + (0).to_bytes(4, "big")
    trun = bf.pack_box("trun", trun_content)
    traf = bf.pack_box("traf", tfhd + tfdt + trun)
    moof_rows = [(2000, bf.pack_moof(traf))]

    with pytest.raises(ft.FragmentLayoutUnsupportedError, match="unresolved by trun, tfhd, or trex"):
        ft.parse_fragment_table(moov, moof_rows)


def test_parse_fragment_table_raises_when_trun_carries_more_than_one_sample() -> None:
    """Refused: a multi-sample trun (see this module's own refusal policy)."""
    moov = bf.synthetic_fragmented_moov()
    moof_rows = [_moof(moof_start=1000, decode_time=0, data_offset=32, sample_size=500, sample_count=2)]

    with pytest.raises(ft.FragmentLayoutUnsupportedError, match="2 samples"):
        ft.parse_fragment_table(moov, moof_rows)


def test_parse_fragment_table_raises_when_tfhd_lacks_default_base_is_moof() -> None:
    """Refused: a tfhd without default-base-is-moof (see this module's own refusal policy)."""
    moov = bf.synthetic_fragmented_moov()
    moof_rows = [_moof(moof_start=1000, decode_time=0, data_offset=32, sample_size=500, default_base_is_moof=False)]

    with pytest.raises(ft.FragmentLayoutUnsupportedError, match="default-base-is-moof"):
        ft.parse_fragment_table(moov, moof_rows)


def test_parse_fragment_table_raises_on_nonzero_composition_offset() -> None:
    """Refused: a nonzero composition offset (see this module's own refusal policy)."""
    moov = bf.synthetic_fragmented_moov()
    moof_rows = [
        _moof(moof_start=1000, decode_time=0, data_offset=32, sample_size=500, composition_offset=100),
    ]

    with pytest.raises(ft.CompositionOffsetUnsupportedError, match="composition time offset"):
        ft.parse_fragment_table(moov, moof_rows)


def test_parse_fragment_table_accepts_a_zero_composition_offset() -> None:
    """A present-but-zero composition offset is not the same as a nonzero one -- not refused."""
    moov = bf.synthetic_fragmented_moov()
    moof_rows = [_moof(moof_start=1000, decode_time=0, data_offset=32, sample_size=500, composition_offset=0)]

    table = ft.parse_fragment_table(moov, moof_rows)

    assert table.size.tolist() == [500]


def test_parse_fragment_table_rejects_a_non_moov_box() -> None:
    """A buffer that isn't a moov box at all fails loudly, matching parse_video_track's own contract."""
    with pytest.raises(BoxNotFoundError, match="expected a moov box"):
        ft.parse_fragment_table(bf.pack_box("free", b""), [])


def test_parse_fragment_table_reuses_real_elst_when_present() -> None:
    """edts/elst on a fragmented track's trak is read the same faithful way as the non-fragmented path."""
    moov = bf.synthetic_fragmented_moov(include_elst=True)
    moof_rows = [_moof(moof_start=1000, decode_time=0, data_offset=32, sample_size=500)]

    table = ft.parse_fragment_table(moov, moof_rows)

    assert table.elst_bytes is not None


def test_parse_fragment_table_elst_is_none_when_absent() -> None:
    """No edts at all means this reader reports absence, not a defaulted value."""
    moov = bf.synthetic_fragmented_moov(include_elst=False)
    moof_rows = [_moof(moof_start=1000, decode_time=0, data_offset=32, sample_size=500)]

    table = ft.parse_fragment_table(moov, moof_rows)

    assert table.elst_bytes is None
