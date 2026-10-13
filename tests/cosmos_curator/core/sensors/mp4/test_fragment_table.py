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

import struct

import pytest

from cosmos_curator.core.sensors.data.mp4 import Mp4Metadata
from cosmos_curator.core.sensors.mp4 import fragment_table as ft
from cosmos_curator.core.sensors.mp4.box_primitives import BoxNotFoundError, MalformedBoxHeaderError
from cosmos_curator.core.sensors.mp4.box_traversal import BoxLocation, UuidCallbackError, UuidHandler
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
    trun_sample_duration: int | None = None,
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
        trun_sample_duration=trun_sample_duration,
    )
    return moof_start, bf.pack_moof(traf)


def test_parse_fragment_table_end_to_end() -> None:
    """A fully synthetic fragmented moov + moof rows decode into the expected SoA shape."""
    moov = bf.synthetic_fragmented_moov(timescale=30, duration=90)
    moof_rows = [
        _moof(
            moof_start=1000,
            decode_time=0,
            data_offset=32,
            sample_size=500,
            is_keyframe=True,
            default_sample_duration=None,
        ),
        _moof(
            moof_start=1600,
            decode_time=1,
            data_offset=32,
            sample_size=400,
            is_keyframe=False,
            default_sample_duration=None,
        ),
        _moof(
            moof_start=2100,
            decode_time=2,
            data_offset=32,
            sample_size=420,
            is_keyframe=False,
            default_sample_duration=None,
        ),
    ]

    table = ft.parse_fragment_table(moov, moof_rows)

    assert table.timescale == 30
    # Derived from the fragments' own tfdt span (0..2), not mdhd's stale 90 -- see
    # test_parse_fragment_table_duration_is_derived_from_fragment_decode_times.
    assert table.duration == 3
    assert table.sample_description.fourcc == "avc1"
    assert table.sample_description.width == 640
    assert table.sample_description.height == 480
    assert table.dts_ticks.tolist() == [0, 1, 2]
    assert table.size.tolist() == [500, 400, 420]
    assert table.offset.tolist() == [1032, 1632, 2132]  # moof_start + data_offset
    assert table.is_keyframe is not None
    assert table.is_keyframe.tolist() == [True, False, False]
    assert table.composition_offsets is None


def test_parse_fragment_table_duration_is_derived_from_fragment_decode_times() -> None:
    """Fragment decode times determine duration when ``mdhd.duration`` is zero or stale."""
    moov = bf.synthetic_fragmented_moov(timescale=30, duration=0)
    moof_rows = [
        _moof(
            moof_start=1000,
            decode_time=0,
            data_offset=32,
            sample_size=500,
            is_keyframe=True,
            default_sample_duration=None,
        ),
        _moof(
            moof_start=1600,
            decode_time=10,
            data_offset=32,
            sample_size=400,
            is_keyframe=False,
            default_sample_duration=None,
        ),
        _moof(
            moof_start=2100,
            decode_time=20,
            data_offset=32,
            sample_size=420,
            is_keyframe=False,
            default_sample_duration=None,
        ),
        _moof(
            moof_start=2600,
            decode_time=30,
            data_offset=32,
            sample_size=410,
            is_keyframe=False,
            default_sample_duration=None,
        ),
    ]

    table = ft.parse_fragment_table(moov, moof_rows)

    # span (30-0=30) + one average interval (30/3=10) -- the span between 4 samples covers only
    # 3 intervals; the 4th sample's own screen time extends one interval past the last decode time.
    # No fragment states its own default_sample_duration here, so there's no real value to prefer
    # over this estimate -- see test_parse_fragment_table_duration_prefers_the_last_fragments_own_sample_duration.
    assert table.duration == 40


def test_parse_fragment_table_duration_prefers_the_last_fragments_own_sample_duration() -> None:
    """The last sample's explicit duration takes precedence over the mean decode interval."""
    moov = bf.synthetic_fragmented_moov(timescale=30, duration=0)
    moof_rows = [
        _moof(
            moof_start=1000,
            decode_time=0,
            data_offset=32,
            sample_size=500,
            is_keyframe=True,
            default_sample_duration=None,
        ),
        _moof(
            moof_start=1600,
            decode_time=10,
            data_offset=32,
            sample_size=400,
            is_keyframe=False,
            default_sample_duration=None,
        ),
        _moof(
            moof_start=2600,
            decode_time=30,
            data_offset=32,
            sample_size=410,
            is_keyframe=False,
            default_sample_duration=12,
        ),
    ]

    table = ft.parse_fragment_table(moov, moof_rows)

    # span (30-0=30) + the last fragment's own stated duration (12), not the averaged interval (15).
    assert table.duration == 42


def test_parse_fragment_table_duration_uses_the_fragments_own_sample_duration_for_a_single_fragment() -> None:
    """A single sample uses its ``tfhd`` duration even without a decode interval."""
    moov = bf.synthetic_fragmented_moov(timescale=30, duration=12345)
    moof_rows = [
        _moof(
            moof_start=1000,
            decode_time=7,
            data_offset=32,
            sample_size=500,
            is_keyframe=True,
            default_sample_duration=3000,
        )
    ]

    table = ft.parse_fragment_table(moov, moof_rows)

    assert table.duration == 3000


def test_parse_fragment_table_duration_is_zero_for_a_single_fragment_with_no_known_sample_duration() -> None:
    """A single sample with no decode-time span and no tfhd default_sample_duration -- 0, not mdhd's stale value."""
    moov = bf.synthetic_fragmented_moov(timescale=30, duration=12345)
    moof_rows = [
        _moof(
            moof_start=1000,
            decode_time=7,
            data_offset=32,
            sample_size=500,
            is_keyframe=True,
            default_sample_duration=None,
        )
    ]

    table = ft.parse_fragment_table(moov, moof_rows)

    assert table.duration == 0


def test_parse_fragment_table_duration_prefers_truns_own_sample_duration_over_tfhds_default() -> None:
    """A ``trun`` sample duration overrides the ``tfhd`` default and works when that default is absent."""
    moov = bf.synthetic_fragmented_moov(timescale=30, duration=12345)
    moof_rows = [
        _moof(
            moof_start=1000,
            decode_time=7,
            data_offset=32,
            sample_size=500,
            is_keyframe=True,
            default_sample_duration=None,
            trun_sample_duration=3500,
        )
    ]

    table = ft.parse_fragment_table(moov, moof_rows)

    assert table.duration == 3500


def test_parse_fragment_table_duration_uses_trex_for_a_single_fragment() -> None:
    """A trex-only sample duration produces nonzero duration and frame rate."""
    moov = bf.synthetic_fragmented_moov(timescale=90_000, trex_sample_duration=3000)
    moof = _moof(moof_start=1000, decode_time=0, data_offset=32, sample_size=500, default_sample_duration=None)

    track = ft.parse_fragment_table(moov, [moof])
    metadata = Mp4Metadata.from_video_track(track)

    assert track.duration == 3000
    assert metadata.duration_ns == 33_333_333
    assert (metadata.avg_frame_rate_num, metadata.avg_frame_rate_den) == (30, 1)


def test_parse_fragment_table_duration_prefers_trex_over_the_mean_decode_interval() -> None:
    """Use the trex duration for the last sample rather than estimating its interval."""
    moov = bf.synthetic_fragmented_moov(timescale=90_000, trex_sample_duration=3000)
    moofs = [
        _moof(
            moof_start=1000 + index * 1000,
            decode_time=decode_time,
            data_offset=32,
            sample_size=500,
            default_sample_duration=None,
        )
        for index, decode_time in enumerate((0, 4000))
    ]

    assert ft.parse_fragment_table(moov, moofs).duration == 7000


@pytest.mark.parametrize(
    ("trun_duration", "tfhd_duration", "expected"),
    [(None, 2000, 2000), (1000, None, 1000), (1000, 2000, 1000), (0, 2000, 0), (None, 0, 0)],
)
def test_parse_fragment_table_duration_preserves_trun_and_tfhd_precedence(
    trun_duration: int | None, tfhd_duration: int | None, expected: int
) -> None:
    """Trun and tfhd durations override trex, including explicit zero values."""
    moov = bf.synthetic_fragmented_moov(timescale=90_000, trex_sample_duration=3000)
    moof = _moof(
        moof_start=1000,
        decode_time=0,
        data_offset=32,
        sample_size=500,
        default_sample_duration=tfhd_duration,
        trun_sample_duration=trun_duration,
    )

    assert ft.parse_fragment_table(moov, [moof]).duration == expected


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


def test_parse_fragment_table_raises_when_trun_sample_entries_are_truncated() -> None:
    """The run declares two samples but contains only one entry."""
    moov = bf.synthetic_fragmented_moov()
    moof_rows = [_moof(moof_start=1000, decode_time=0, data_offset=32, sample_size=500, sample_count=2)]

    with pytest.raises(MalformedBoxHeaderError, match="truncated sample entries"):
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


@pytest.mark.parametrize("error_type", [IndexError, struct.error, RuntimeError, ValueError])
@pytest.mark.parametrize("parent_path", [("moov",), ("moof",), ("moof", "traf")])
def test_uuid_callback_errors_are_not_malformed_mp4_data(
    error_type: type[Exception], parent_path: tuple[str, ...]
) -> None:
    """Failures in either input buffer remain distinct from malformed MP4 data."""
    identifier = b"1234567890abcdef"
    uuid = bf.uuid_box(identifier, b"payload")
    moov = bf.synthetic_fragmented_moov()
    traf = bf.real_traf(decode_time=0, data_offset=32, sample_size=500, is_keyframe=True)
    if parent_path == ("moov",):
        moov = bf.pack_box("moov", moov[8:] + uuid)
    elif parent_path == ("moof", "traf"):
        traf = bf.pack_box("traf", traf[8:] + uuid)
    moof = bf.pack_moof(traf, siblings=uuid if parent_path == ("moof",) else b"")
    error = error_type("handler failed")

    def fail(_payload: bytes, _box: BoxLocation) -> None:
        raise error

    handler = UuidHandler(identifier, frozenset({parent_path}), fail)
    with pytest.raises(UuidCallbackError) as caught:
        ft.parse_fragment_table(moov, [(1000, moof)], uuid_handlers=[handler])
    assert caught.value.error is error
    assert caught.value.__cause__ is error


def test_uuid_handlers_preserve_fragment_order_and_scope() -> None:
    """Moof UUIDs are fragment-scoped; traf UUIDs retain their declared track ID."""
    identifier = b"1234567890abcdef"
    events: list[tuple[bytes, BoxLocation]] = []
    rows: list[tuple[int, bytes]] = []
    for index in range(2):
        traf = bf.real_traf(decode_time=index * 3000, data_offset=32, sample_size=500, is_keyframe=True)
        traf = bf.pack_box("traf", traf[8:] + bf.uuid_box(identifier, b"track"))
        rows.append((1000 + index * 1000, bf.pack_moof(traf, siblings=bf.uuid_box(identifier, bytes([index])))))
    handler = UuidHandler(
        identifier, frozenset({("moof",), ("moof", "traf")}), lambda payload, box: events.append((payload, box))
    )
    track = ft.parse_fragment_table(bf.synthetic_fragmented_moov(), rows, uuid_handlers=[handler])
    assert [payload for payload, _box in events] == [b"\x00", b"track", b"\x01", b"track"]
    assert [box.parent.fragment_index for _payload, box in events] == [0, 0, 1, 1]
    assert [box.parent.track_id for _payload, box in events] == [None, 1, None, 1]
    assert track.dts_ticks.tolist() == [0, 3000]


@pytest.mark.parametrize("parent", ["mdia", "minf"])
def test_uuid_handlers_visit_intermediate_track_containers(parent: str) -> None:
    """Track-container UUIDs dispatch exactly once with the selected track identity."""
    identifier = b"1234567890abcdef"
    payload = b"track timing"
    extras = {f"{parent}_extra_children": bf.uuid_box(identifier, payload)}
    moov = bf.synthetic_fragmented_moov(**extras)
    events: list[tuple[bytes, BoxLocation]] = []
    parent_path = ("moov", "trak", "mdia") + (("minf",) if parent == "minf" else ())
    handler = UuidHandler(identifier, frozenset({parent_path}), lambda value, box: events.append((value, box)))

    track = ft.parse_fragment_table(
        moov, [_moof(moof_start=1000, decode_time=0, data_offset=32, sample_size=500)], uuid_handlers=[handler]
    )

    assert track.size.tolist()
    assert len(events) == 1
    assert events[0][0] == payload
    assert events[0][1].parent.path == parent_path
    assert events[0][1].parent.track_id == 1
    assert events[0][1].parent.fragment_index is None
