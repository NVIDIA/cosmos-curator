# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fast-path compatibility and decoded output validation regressions."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cosmos_curator.next.media import smart_cut


def _index(codec: str = "h264") -> smart_cut._Index:
    return smart_cut._Index(
        1,
        90000,
        [i * 1800 for i in range(100)],
        stream_copy_safe=False,
        smart_cut_safe=True,
        keyframes=(0, 50),
        codec_name=codec,
    )


def _stub_output(monkeypatch: pytest.MonkeyPatch, frames: int) -> list[tuple]:
    warnings = []
    monkeypatch.setattr(smart_cut, "_run_ff", lambda *_a, **_kw: True)
    monkeypatch.setattr(smart_cut, "_run_encode", lambda *_a, **_kw: True)
    monkeypatch.setattr(smart_cut, "_validate_output", lambda *_a: (frames, None))
    monkeypatch.setattr(smart_cut.logger, "warning", lambda *args: warnings.append(args))
    return warnings


@pytest.mark.parametrize("codec", ["hevc", "av1", ""])
def test_incompatible_hybrid_falls_back_before_splicing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    codec: str,
) -> None:
    """Non-H.264 packets must not be spliced behind an H.264 head."""
    warnings = _stub_output(monkeypatch, 10)
    splices = []
    monkeypatch.setattr(smart_cut, "_smartcut", lambda *a, **_kw: splices.append(a) or True)
    record = smart_cut._cut_one(
        "source.mp4",
        {"startFrame": 45, "endFrame": 54, "output": str(tmp_path / "cut.mp4")},
        _index(codec),
        "2M",
        1,
        allow_stream_copy=True,
        smart_cut=True,
    )
    assert not splices
    assert record["success"]
    assert record["mode"] == "encode"
    assert "codec" in record["fallback_reason"]
    assert not warnings


def test_codec_fallbacks_are_summarized_once_per_plan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Expected codec fallbacks retain diagnostics without a warning per clip."""
    warnings = _stub_output(monkeypatch, 10)
    messages = []
    monkeypatch.setattr(smart_cut.logger, "info", lambda *args: messages.append(args))
    monkeypatch.setattr(smart_cut, "_probe_index", lambda *_args, **_kwargs: _index("hevc"))
    cuts = [{"startFrame": 45, "endFrame": 54, "output": str(tmp_path / f"{i}.mp4")} for i in range(2)]
    results = smart_cut.cut_plan("source.mp4", cuts, smart_cut=True)
    assert all(record["success"] and "hevc" in record["fallback_reason"] for record in results)
    assert not warnings
    assert len(messages) == 1
    assert messages[0][1:] == (2, 2, 0, "hevc")


def test_compatible_h264_retains_hybrid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Do not disable compatible hybrid cuts as a workaround."""
    _stub_output(monkeypatch, 10)
    monkeypatch.setattr(smart_cut, "_smartcut", lambda *_a, **_kw: True)
    record = smart_cut._cut_one(
        "source.mp4",
        {"startFrame": 45, "endFrame": 54, "output": str(tmp_path / "cut.mp4")},
        _index(),
        "2M",
        1,
        allow_stream_copy=True,
        smart_cut=True,
    )
    assert record["success"]
    assert record["mode"] == "smartcut"


@pytest.mark.parametrize("all_intra", [False, True])
def test_hevc_pure_copy_does_not_require_h264_compatibility(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    all_intra: bool,
) -> None:
    """Pure copying does not mix encoders and remains valid for HEVC."""
    _stub_output(monkeypatch, 10)
    index = _index("hevc")
    index.stream_copy_safe = all_intra
    record = smart_cut._cut_one(
        "source.mp4",
        {"startFrame": 50, "endFrame": 59, "output": str(tmp_path / "cut.mp4")},
        index,
        "2M",
        1,
        allow_stream_copy=True,
        smart_cut=True,
    )
    assert record["success"]
    assert record["mode"] == ("copy" if all_intra else "smartcopy")


def _probe(monkeypatch: pytest.MonkeyPatch, pts: list[int], *, duration: str = "0.08", returncode: int = 0) -> None:
    payload = {
        "streams": [{"time_base": "1/50", "nb_frames": "4", "nb_read_frames": str(len(pts)), "duration": duration}],
        "frames": [{"best_effort_timestamp": p} for p in pts],
    }
    monkeypatch.setattr(
        smart_cut,
        "_ffprobe",
        lambda _args: SimpleNamespace(returncode=returncode, stdout=json.dumps(payload), stderr=""),
    )


def _validate() -> tuple[int, str | None]:
    return smart_cut._validate_output("cut.mp4", _index(), 2, 4)


def test_validation_accepts_decoded_count_and_source_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A correctly decoded and timed clip passes validation."""
    _probe(monkeypatch, [0, 1, 2, 3])
    assert _validate() == (4, None)


def test_validation_rejects_matching_header_but_short_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Container sample counts must not mask missing displayed frames."""
    _probe(monkeypatch, [0])
    count, error = _validate()
    assert count == 1
    assert error is not None


@pytest.mark.parametrize("pts", [[0, 7, 14, 21], [1, 2, 3, 4]])
def test_validation_rejects_wrong_timestamps(monkeypatch: pytest.MonkeyPatch, pts: list[int]) -> None:
    """A correct count cannot conceal time-base corruption or a shifted start."""
    _probe(monkeypatch, pts)
    count, error = _validate()
    assert count == 4
    assert error is not None


def test_validation_rejects_wrong_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Terminal sample duration is part of the requested interval."""
    _probe(monkeypatch, [0, 1, 2, 3], duration="8.015625")
    assert _validate()[1] is not None


def test_validation_rejects_failed_probe_with_partial_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Partial JSON from a failed decoder is not evidence of a valid clip."""
    _probe(monkeypatch, [0, 1, 2, 3], returncode=1)
    assert _validate()[1] is not None


def test_invalid_fast_path_retries_full_encode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An invalid fast path is replaced rather than published as success."""
    _stub_output(monkeypatch, 10)
    monkeypatch.setattr(smart_cut, "_smartcut", lambda *_a, **_kw: True)
    validations = iter([(1, "decoded 1/10 frames"), (10, None)])
    monkeypatch.setattr(smart_cut, "_validate_output", lambda *_a: next(validations))
    record = smart_cut._cut_one(
        "source.mp4",
        {"startFrame": 45, "endFrame": 54, "output": str(tmp_path / "cut.mp4")},
        _index(),
        "2M",
        1,
        allow_stream_copy=True,
        smart_cut=True,
    )
    assert record["success"]
    assert record["mode"] == "encode"
    assert "decoded" in record["fallback_reason"]


def test_invalid_full_encode_is_not_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A failed fallback stays failed; it cannot inherit fast-path success."""
    _stub_output(monkeypatch, 10)
    monkeypatch.setattr(smart_cut, "_validate_output", lambda *_a: (1, "decoded 1/10 frames"))
    record = smart_cut._cut_one(
        "source.mp4",
        {"startFrame": 45, "endFrame": 54, "output": str(tmp_path / "cut.mp4")},
        _index(),
        "2M",
        1,
        allow_stream_copy=False,
    )
    assert not record["success"]
    assert "decoded" in record["error"]


def test_full_encode_keeps_container_specific_options_out() -> None:
    """Only internal MP4 hybrid segments need the explicit track timescale."""
    command = smart_cut._encode_cmd("source.mp4", "output.mkv", _index(), 2, 4, "2M", 1)
    assert "-video_track_timescale" not in command


def test_validation_preserves_variable_frame_intervals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare source PTS rather than assuming a constant output frame rate."""
    index = smart_cut._Index(1, 90000, [0, 1800, 5400, 7200, 10800], stream_copy_safe=False)
    _probe(monkeypatch, [0, 2, 3], duration="0.10")
    assert smart_cut._validate_output("cut.mp4", index, 1, 3) == (3, None)


def test_validation_checks_available_terminal_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cuts ending at EOF use the source stream's end, not an invented CFR duration."""
    index = _index()
    index.end_pts = 180000
    _probe(monkeypatch, [0, 1], duration="0.04")
    assert smart_cut._validate_output("cut.mp4", index, 98, 2) == (2, None)
    _probe(monkeypatch, [0, 1], duration="0.08")
    assert smart_cut._validate_output("cut.mp4", index, 98, 2)[1] is not None


def test_vfr_sample_duration_is_not_the_next_pts_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gap after a selected sample is not part of that sample's duration."""
    index = smart_cut._Index(1, 1000, [0, 40, 120], stream_copy_safe=False, durations=[40, 40, 40])
    _probe(monkeypatch, [0, 2], duration="0.08")
    assert smart_cut._validate_output("cut.mp4", index, 0, 2) == (2, None)


def test_unknown_terminal_duration_does_not_fail_valid_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing source end metadata is not evidence that a correctly decoded clip is bad."""
    index = smart_cut._Index(1, 50, [0, 1], stream_copy_safe=False)
    _probe(monkeypatch, [0, 1], duration="0.04")
    assert smart_cut._validate_output("cut.mp4", index, 0, 2) == (2, None)


@pytest.mark.parametrize("pts", [[0], [0, 3]])
def test_unknown_terminal_duration_keeps_count_and_pts_checks(
    monkeypatch: pytest.MonkeyPatch,
    pts: list[int],
) -> None:
    """Skipping an unavailable end check must not skip the known count or timestamps."""
    index = smart_cut._Index(1, 50, [0, 1], stream_copy_safe=False)
    _probe(monkeypatch, pts, duration="0.04")
    assert smart_cut._validate_output("cut.mp4", index, 0, 2)[1] is not None


def test_unknown_terminal_duration_does_not_retry_copy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Retrying the encoder cannot supply source duration metadata that never existed."""
    index = smart_cut._Index(1, 50, [0, 1], stream_copy_safe=True)
    _probe(monkeypatch, [0, 1], duration="0.04")
    monkeypatch.setattr(smart_cut, "_run_ff", lambda *_a: True)
    monkeypatch.setattr(smart_cut, "_run_encode", lambda *_a, **_kw: pytest.fail("unnecessary encode retry"))
    record = smart_cut._cut_one(
        "source.mp4",
        {"startFrame": 0, "endFrame": 1, "output": str(tmp_path / "copy.mp4")},
        index,
        "2M",
        1,
        allow_stream_copy=True,
    )
    assert record["success"]
    assert record["mode"] == "copy"
    assert record["fallback_reason"] is None


@pytest.mark.parametrize(
    ("source_pts", "output_pts", "origin", "duration"),
    [([8, 32, 53, 79, 112], [0, 2, 3, 4], 0, "0.12"), ([12, 48, 73, 99, 128], [1, 2, 4, 5], 1, "0.10")],
)
def test_copy_accepts_independently_rounded_pts_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_pts: list[int],
    output_pts: list[int],
    origin: int,
    duration: str,
) -> None:
    """Rounding absolute endpoints to 20 ms can shift their difference by more than 10 ms."""
    index = smart_cut._Index(1, 1000, source_pts, stream_copy_safe=True)
    payload = {
        "streams": [{"time_base": "1/50", "start_pts": origin, "nb_read_frames": "4", "duration": duration}],
        "frames": [{"best_effort_timestamp": pts} for pts in output_pts],
    }
    monkeypatch.setattr(
        smart_cut,
        "_ffprobe",
        lambda _args: SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(payload)),
    )
    monkeypatch.setattr(smart_cut, "_run_ff", lambda *_a: True)
    monkeypatch.setattr(smart_cut, "_run_encode", lambda *_a, **_kw: pytest.fail("unnecessary encode retry"))
    record = smart_cut._cut_one(
        "source.mp4",
        {"startFrame": 0, "endFrame": 3, "output": str(tmp_path / "copy.mp4")},
        index,
        "2M",
        1,
        allow_stream_copy=True,
    )
    assert record["success"]
    assert record["mode"] == "copy"


def test_validation_rejects_more_than_endpoint_rounding(monkeypatch: pytest.MonkeyPatch) -> None:
    """One output tick of rounding slack must not hide a two-tick timing error."""
    _probe(monkeypatch, [0, 3, 4, 5])
    assert _validate()[1] is not None


def test_packet_duration_validates_eof_without_stream_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    """A terminal packet duration provides the endpoint even for Matroska inputs."""
    index = smart_cut._Index(1, 50, [0, 1], stream_copy_safe=False, durations=[1, 1])
    _probe(monkeypatch, [0, 1], duration="0.04")
    assert smart_cut._validate_output("cut.mp4", index, 0, 2) == (2, None)
    _probe(monkeypatch, [0, 1], duration="160.96")
    assert smart_cut._validate_output("cut.mp4", index, 0, 2)[1] is not None


def test_index_keeps_sample_durations_aligned_with_pts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain codec and presentation-ordered durations through packet reordering."""
    metadata = {
        "streams": [{"time_base": "1/1000", "codec_name": "hevc", "field_order": "progressive"}],
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
    }
    packets = {
        "packets": [
            {"pts": 0, "duration": 40, "flags": "K_"},
            {"pts": 120, "duration": 30, "flags": "__"},
            {"pts": 40, "duration": 20, "flags": "__"},
        ]
    }
    monkeypatch.setattr(
        smart_cut,
        "_ffprobe",
        lambda args: SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(packets if "-show_packets" in args else metadata),
        ),
    )
    index = smart_cut._probe_index("source.mp4")
    assert index.codec_name == "hevc"
    assert index.pts == [0, 40, 120]
    assert index.durations == [40, 20, 30]
    assert index.end_of_frame(2) == 150


@pytest.mark.parametrize("container", ["ts", "mkv"])
def test_validation_accounts_for_container_timing(monkeypatch: pytest.MonkeyPatch, container: str) -> None:
    """TS has a muxer offset; Matroska exposes duration at format rather than stream level."""
    offset = 1400 if container == "ts" else 0
    stream = {"time_base": "1/1000", "start_pts": offset, "nb_read_frames": "4"}
    if container == "ts":
        stream["duration"] = "0.08"
    payload = {
        "streams": [stream],
        "format": {"duration": "0.08"},
        "frames": [{"best_effort_timestamp": offset + p} for p in [0, 20, 40, 60]],
    }
    monkeypatch.setattr(
        smart_cut,
        "_ffprobe",
        lambda _args: SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr=""),
    )
    assert _validate() == (4, None)
