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

"""Frame-identity regression for the smart-cut head and the full re-encode fallback.

A cut that emits the right number of frames can still start on the wrong one, which is
exactly how the original defect passed validation. The re-encode therefore stops relying on
``-ss`` to choose frames: it seeks to a keyframe at or before the start purely to bound decode
work, selects the range by integer source PTS, and then verifies the frames it actually
decoded. These tests cover each of those three parts, plus the index that feeds them.

No ffmpeg: the seek arithmetic is modelled here, and the end-to-end check against real ffmpeg
on both MP4 and MPEG-TS lives in ``test_smart_cut_seek_integration.py``.
"""

import io
import json
import tempfile
from fractions import Fraction
from types import SimpleNamespace

import pytest

from cosmos_curator.next.media import smart_cut
from cosmos_curator.next.media.smart_cut import (
    _encode_cmd,
    _error_tail,
    _Index,
    _keyframe_seek_seconds,
    _probe_index,
    _run_encode,
    _selected_source_pts_match,
    _unambiguous_keyframes,
)

# (num, den) stream timebases, coarse through fine.
TIMEBASES = [
    pytest.param(1, 30, id="1/30"),
    pytest.param(1, 25, id="1/25"),
    pytest.param(1001, 30000, id="1001/30000-29.97fps"),
    pytest.param(1, 60, id="1/60"),
    pytest.param(1, 120, id="1/120"),
    pytest.param(1, 1000, id="1/1000"),
    pytest.param(1, 90000, id="1/90000"),
    pytest.param(1, 1000000, id="1/1000000"),
]


def _tick_ffmpeg_seeks_to(seek: str, num: int, den: int) -> Fraction:
    """Return the source tick ffmpeg resolves *seek* to.

    ``-ss`` is parsed into whole microseconds with finer digits discarded, then rescaled to
    source ticks with nearest rounding (half away from zero).
    """
    microseconds = int(Fraction(seek) * 1_000_000)
    ticks = Fraction(microseconds, 1_000_000) * den / num
    return Fraction(int(ticks + Fraction(1, 2)))


def _index(pts: list[int], keyframes: tuple[int, ...], num: int = 1, den: int = 30) -> _Index:
    return _Index(num, den, pts, stream_copy_safe=False, keyframes=keyframes, smart_cut_safe=True)


# --------------------------------------------------------------------------------------
# _keyframe_seek_seconds: the seek must never overshoot the keyframe it anchors on.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("num", "den"), TIMEBASES)
def test_keyframe_seek_never_lands_past_the_keyframe(num: int, den: int) -> None:
    """The anchor seek resolves at or before its keyframe at every timebase.

    Landing even one tick past the keyframe starts decoding without the reference the
    requested frames depend on, which is the MPEG-TS failure this anchoring exists to avoid.
    Landing earlier is harmless -- ``trim=start_pts`` still selects the frames.
    """
    over = [t for t in range(2001) if _tick_ffmpeg_seeks_to(_keyframe_seek_seconds(t, num, den), num, den) > t]
    assert over == [], f"timebase {num}/{den}: {len(over)} seeks overshot, first few {over[:5]}"


@pytest.mark.parametrize(("num", "den"), TIMEBASES)
def test_keyframe_seek_stays_within_one_tick(num: int, den: int) -> None:
    """The seek is not merely safe but tight, so it does not decode a whole extra GOP."""
    slack = [t - _tick_ffmpeg_seeks_to(_keyframe_seek_seconds(t, num, den), num, den) for t in range(2001)]
    assert max(slack) <= 1, f"timebase {num}/{den}: seek fell {max(slack)} ticks short of the keyframe"


def test_rounding_instead_of_flooring_overshoots_below_a_microsecond() -> None:
    """Record where flooring is actually load-bearing, so the choice is not cargo cult.

    At every ordinary timebase a rounded conversion is self-inverting and lands on the keyframe
    too, so these tests would stay green either way. Flooring earns its place only once a tick
    is finer than a microsecond, where rounding overshoots on roughly half of all positions.
    """
    num, den = 1, 10_000_000
    rounded = [
        t
        for t in range(2001)
        if _tick_ffmpeg_seeks_to(f"{round(t * num * 1_000_000 / den) / 1_000_000:.6f}", num, den) > t
    ]
    assert rounded, "rounding no longer overshoots; re-check whether flooring is still load-bearing"
    assert _tick_ffmpeg_seeks_to(_keyframe_seek_seconds(2001, num, den), num, den) <= 2001


def test_first_frame_seeks_to_zero() -> None:
    """A clip anchored on the first keyframe seeks to the start of the stream."""
    assert _keyframe_seek_seconds(0, 1, 30) == "0.000000"


# --------------------------------------------------------------------------------------
# Keyframe index: presentation-order mapping and ambiguity.
# --------------------------------------------------------------------------------------


def test_keyframe_at_or_before_picks_the_anchor() -> None:
    """The anchor is the last keyframe at or before the frame, or None below the first."""
    index = _index(list(range(12)), (0, 4, 8))
    assert [index.keyframe_at_or_before(f) for f in (0, 3, 4, 5, 8, 11)] == [0, 0, 4, 4, 8, 8]
    assert _index(list(range(12)), (4, 8)).keyframe_at_or_before(3) is None
    assert _index(list(range(12)), ()).keyframe_at_or_before(7) is None


def test_unambiguous_keyframes_drops_duplicated_timestamps() -> None:
    """A keyframe sharing a timestamp with a neighbour cannot be seeked to unambiguously."""
    pts = [0, 1, 1, 2, 3, 4, 5, 5, 6]
    # 1 and 2 share PTS 1; 6 and 7 share PTS 5. Only 0, 4 and 8 are addressable.
    assert _unambiguous_keyframes(pts, (0, 1, 2, 4, 6, 7, 8)) == (0, 4, 8)
    # Both ends are checked, so a duplicate at either boundary is still rejected.
    assert _unambiguous_keyframes([7, 7], (0, 1)) == ()
    assert _unambiguous_keyframes([0, 1, 2], (0, 1, 2)) == (0, 1, 2)


def _fake_ffprobe(packets: list[dict[str, object]]):  # noqa: ANN202
    def run(args: list[str]) -> SimpleNamespace:
        if "-show_packets" in args:
            return SimpleNamespace(returncode=0, stdout=json.dumps({"packets": packets}), stderr="")
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(
                {
                    "streams": [{"time_base": "1/30", "codec_name": "h264", "field_order": "progressive"}],
                    "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
                }
            ),
        )

    return run


def test_keyframes_are_mapped_to_presentation_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """With B-frame reordering, keyframe file positions are not frame indices.

    Packets arrive 0, 4, 1, 2, 3, so the keyframe at PTS 4 sits at file row 1 but is frame 4.
    Taking the row would anchor the decode on the wrong frame.
    """
    packets = [{"pts": p, "flags": "K_" if p in (0, 4) else "__"} for p in (0, 4, 1, 2, 3)]
    monkeypatch.setattr(smart_cut, "_ffprobe", _fake_ffprobe(packets))
    index = _probe_index("src.mp4")
    assert index.keyframes == (0, 4)
    # Packets out of presentation order, so copying packets in file order is still unsafe.
    assert not index.smart_cut_safe


def test_duplicate_pts_keep_only_unambiguous_keyframes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Duplicate timestamps disable copying but must not discard the whole keyframe index.

    Previously any duplicate emptied ``keyframes``, which would leave the re-encode with no
    anchor and send it back to decoding from the start of the file.
    """
    packets = [{"pts": p, "flags": "K_" if p in (0, 1, 4) else "__"} for p in (0, 1, 1, 2, 3, 4)]
    monkeypatch.setattr(smart_cut, "_ffprobe", _fake_ffprobe(packets))
    index = _probe_index("src.mp4")
    assert index.keyframes == (0, 5)
    assert not index.smart_cut_safe
    assert not index.stream_copy_safe


# --------------------------------------------------------------------------------------
# _encode_cmd: the seek bounds decode work, the trim selects frames.
# --------------------------------------------------------------------------------------


def test_encode_cmd_anchors_on_the_keyframe_and_trims_by_source_pts() -> None:
    """The seek targets the anchor keyframe; the requested frame is selected by its own PTS."""
    index = _index(list(range(120)), (0, 50, 100))
    cmd = _encode_cmd("src.mp4", "out.mp4", index, 63, 7, "4M", 1)
    # Frame 63's own timestamp is 2.1s; the seek is the keyframe at frame 50 instead.
    assert cmd[cmd.index("-ss") + 1] == _keyframe_seek_seconds(50, 1, 30) == "1.666666"
    vf = cmd[cmd.index("-vf") + 1]
    assert "trim=start_pts=63" in vf
    assert "trim=end_frame=7" in vf
    # showinfo must log before setpts, which is what destroys the source PTS.
    assert vf.index("showinfo@source") < vf.index("setpts=PTS-STARTPTS")
    # The trim owns selection, so the seek is deliberately inexact and timestamps are preserved.
    assert "-copyts" in cmd
    assert "-noaccurate_seek" in cmd
    assert cmd[cmd.index("-frames:v") + 1] == "7"


def test_encode_cmd_decodes_from_the_start_without_a_usable_keyframe() -> None:
    """With no keyframe at or before the start, decode from the beginning rather than guess."""
    cmd = _encode_cmd("src.mp4", "out.mp4", _index(list(range(10)), ()), 3, 2, "4M", 1)
    assert "-ss" not in cmd
    assert "trim=start_pts=3" in cmd[cmd.index("-vf") + 1]


def test_encode_cmd_logs_at_info_so_source_pts_are_observable() -> None:
    """The verification depends on the log level, so pin it rather than leave it incidental."""
    cmd = _encode_cmd("src.mp4", "out.mp4", _index(list(range(10)), (0,)), 0, 2, "4M", 1)
    assert cmd[cmd.index("-v") + 1] == "info"


# --------------------------------------------------------------------------------------
# Verification: identity of the frames that were actually decoded.
# --------------------------------------------------------------------------------------


def _log(pts_values: list[int], *, prefix: bytes = b"") -> io.BytesIO:
    errf = io.BytesIO(prefix)
    errf.seek(0, 2)
    for n, pts in enumerate(pts_values):
        errf.write(f"[Parsed_showinfo@source_3 @ 0x1] n:{n:4d} pts:{pts:7d} pts_time:0.1\n".encode())
    return errf


def test_selected_source_pts_match_accepts_the_requested_range() -> None:
    """The frames the decoder actually emitted are the ones the plan asked for."""
    errf = _log([63, 64, 65])
    assert _selected_source_pts_match(errf, 0, [63, 64, 65])


def test_selected_source_pts_match_rejects_a_same_length_substitution() -> None:
    """The defect's signature: the right number of frames, taken from the wrong place.

    Every frame-count and duration check passes for this case, so it is the one the PTS
    comparison exists to catch.
    """
    assert not _selected_source_pts_match(_log([100, 101, 102]), 0, [63, 64, 65])


def test_selected_source_pts_match_rejects_a_short_decode() -> None:
    """A truncated decode is rejected too, not only a shifted one."""
    assert not _selected_source_pts_match(_log([63, 64]), 0, [63, 64, 65])


def test_selected_source_pts_match_ignores_earlier_runs() -> None:
    """A failed earlier attempt leaves its own trace in the shared log; only this run counts."""
    prefix_log = _log([1, 2, 3])
    log_start = prefix_log.tell()
    for n, pts in enumerate([63, 64, 65]):
        prefix_log.write(f"[Parsed_showinfo@source_3 @ 0x1] n:{n:4d} pts:{pts:7d}\n".encode())
    assert _selected_source_pts_match(prefix_log, log_start, [63, 64, 65])


def test_run_encode_rejects_a_wrong_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean ffmpeg exit is not enough: the wrong frames must still fail the cut."""
    index = _index(list(range(120)), (0, 50, 100))

    def fake_run_ff(cmd: list[str], errf) -> bool:  # noqa: ANN001, ARG001
        errf.write(b"".join(f"[Parsed_showinfo@source_3] n:{n} pts:{100 + n}\n".encode() for n in range(3)))
        return True

    monkeypatch.setattr(smart_cut, "_run_ff", fake_run_ff)
    with tempfile.TemporaryFile() as errf:
        assert not _run_encode("src.mp4", "out.mp4", index, 63, 3, "4M", 1, errf)
        errf.seek(0)
        assert b"Decoded source PTS do not match" in errf.read()


def test_run_encode_accepts_the_requested_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """The check passes through a correct encode rather than failing closed on every run."""
    index = _index(list(range(120)), (0, 50, 100))

    def fake_run_ff(cmd: list[str], errf) -> bool:  # noqa: ANN001, ARG001
        errf.write(b"".join(f"[Parsed_showinfo@source_3] n:{n} pts:{63 + n}\n".encode() for n in range(3)))
        return True

    monkeypatch.setattr(smart_cut, "_run_ff", fake_run_ff)
    with tempfile.TemporaryFile() as errf:
        assert _run_encode("src.mp4", "out.mp4", index, 63, 3, "4M", 1, errf)


def test_error_tail_keeps_the_ffmpeg_error_past_the_frame_trace() -> None:
    """``-v info`` emits a line per frame; unfiltered, the tail would be only frame dumps."""
    errf = _log(list(range(500)))
    errf.write(b"[libopenh264 @ 0x1] Encoding failed: no memory\n")
    tail = _error_tail(errf)
    assert "Encoding failed: no memory" in tail
    assert "showinfo@source" not in tail
