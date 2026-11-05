# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smart-cut failure reporting under the shared media failure contract.

A failure is a ``MediaError`` only on positive evidence about the source or cut;
a caller that publishes per-cut failures permanently must never see a full disk,
a timeout, or a killed ffmpeg as one.
"""

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from cosmos_curator.next.media import smart_cut
from cosmos_curator.next.media.failures import (
    CutError,
    FFmpegRunError,
    InvalidMediaError,
    MediaError,
    ProbeError,
    UnsupportedMediaError,
)
from cosmos_curator.next.media.smart_cut import _Index


def _completed(returncode: int, stderr: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["ffprobe"], returncode=returncode, stdout="", stderr=stderr)


@pytest.mark.parametrize(
    ("diagnosis", "expected"),
    [
        (InvalidMediaError("FFprobe reported invalid media data"), InvalidMediaError),
        # The transcode metadata contract says nothing about why the cut probe failed.
        (UnsupportedMediaError("Media metadata does not satisfy the v1 contract"), ProbeError),
        (ProbeError("FFprobe exited with status 1"), ProbeError),
        (None, ProbeError),
    ],
)
def test_a_failed_probe_is_terminal_only_when_ffprobe_identifies_the_media(
    monkeypatch: pytest.MonkeyPatch, diagnosis: Exception | None, expected: type[Exception]
) -> None:
    """The structured FFprobe contract decides, never stderr; a source that probes cleanly is inconclusive."""

    def diagnose(_source: str, **_kwargs: object) -> None:
        if diagnosis is not None:
            raise diagnosis

    monkeypatch.setattr(smart_cut, "probe_video_source", diagnose)
    stderr = "source.mp4: Invalid data found when processing input"
    error = smart_cut._probe_failure("source.mp4", _completed(1, stderr), "ffprobe failed for 'source.mp4'")
    assert type(error) is expected
    assert stderr in str(error)


def _probed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["ffprobe"], returncode=0, stdout=stdout, stderr="")


@pytest.mark.parametrize(
    ("decoded_frames", "expected"),
    [
        (0, ProbeError),  # A missing or failed decoder exits 0 with no frames: no evidence about the packets.
        (10, UnsupportedMediaError),  # Frames decoded, but fewer than the packets: the index is not a frame index.
    ],
)
def test_only_a_decoded_count_can_reject_the_packet_index(
    monkeypatch: pytest.MonkeyPatch, decoded_frames: int, expected: type[Exception]
) -> None:
    """The prefix check rejects an AV1 source only when the decoder produced frames that disagree."""
    stream = {"time_base": "1/30", "codec_name": "av1", "field_order": "progressive"}
    meta = json.dumps({"streams": [stream], "format": {"format_name": "mov,mp4"}})
    packets = '{"packets": [' + ",".join(f'{{"pts": {i}, "flags": "K__"}}' for i in range(20)) + "]}"
    responses = iter([_probed(meta), _probed(packets)])
    monkeypatch.setattr(smart_cut, "_ffprobe", lambda _args: next(responses))
    monkeypatch.setattr(smart_cut, "_prefix_packet_and_frame_counts", lambda _source, _seconds: (20, decoded_frames))
    with pytest.raises(expected):
        smart_cut._probe_index("source.mp4")


def test_a_persistent_probe_timeout_is_a_probe_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ray retries a ProbeError by name; a bare TimeoutExpired would match none of its patterns."""

    def timeout(*_args: object, **_kwargs: object) -> None:
        cmd = "ffprobe"
        raise subprocess.TimeoutExpired(cmd, 600)

    monkeypatch.setattr(smart_cut.subprocess, "run", timeout)
    monkeypatch.setattr(smart_cut.time, "sleep", lambda _s: None)
    with pytest.raises(ProbeError, match="timed out"):
        smart_cut._ffprobe(["source.mp4"])


def test_a_killed_ffmpeg_raises_instead_of_reading_as_a_failed_cut(tmp_path: Path) -> None:
    """A signal ends the run, so no fallback encode runs and the output is not judged."""
    with (tmp_path / "err").open("w+b") as errf, pytest.raises(FFmpegRunError, match="signal 9"):
        smart_cut._run_ff(["sh", "-c", "kill -KILL $$"], errf)


def _index() -> _Index:
    return _Index(1, 30, list(range(60)), stream_copy_safe=False, keyframes=(0, 30), smart_cut_safe=False)


def _cut(tmp_path: Path) -> dict[str, Any]:
    return {"startFrame": 10, "endFrame": 19, "output": str(tmp_path / "clip.mp4")}


def _cut_one(tmp_path: Path, cut: dict[str, Any] | None = None) -> dict[str, Any]:
    return smart_cut._cut_one(
        "source.mp4", cut or _cut(tmp_path), _index(), "2M", 1, allow_stream_copy=False, smart_cut=True
    )


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (subprocess.TimeoutExpired("ffmpeg", 600), FFmpegRunError),
        (FFmpegRunError("ffmpeg was killed by signal 9"), FFmpegRunError),
        (OSError(28, "ENOSPC"), FFmpegRunError),
        (ProbeError("ffprobe timed out after 600s"), ProbeError),
    ],
)
def test_environmental_cut_failures_are_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: BaseException, expected: type[Exception]
) -> None:
    """Timeouts, killed processes, OS errors, and failed probes leave the cut retryable."""

    def fail(*_args: object, **_kwargs: object) -> bool:
        raise failure

    monkeypatch.setattr(smart_cut, "_run_encode", fail)
    record = _cut_one(tmp_path)
    assert record["success"] is False
    assert type(record["failure"]) is expected
    assert record["error"] == str(record["failure"])


def test_a_failed_reencode_is_inconclusive_whatever_ffmpeg_says(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ffmpeg's free-text log never establishes a media problem, so a failed run is retried."""

    def failed_encode(*args: Any, **_kwargs: object) -> bool:  # noqa: ANN401
        args[7].write(b"Invalid NAL unit size")
        return False

    monkeypatch.setattr(smart_cut, "_run_encode", failed_encode)
    record = _cut_one(tmp_path)
    assert isinstance(record["failure"], FFmpegRunError)
    assert "Invalid NAL unit size" in record["error"]


def test_an_encode_that_logs_no_source_frames_is_inconclusive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no showinfo trace the selection was never observed, so it is not evidence of wrong frames."""
    monkeypatch.setattr(smart_cut, "_run_ff", lambda _cmd, _errf: True)
    with (tmp_path / "err").open("w+b") as errf, pytest.raises(FFmpegRunError, match="no decoded source frames"):
        smart_cut._run_encode("source.mp4", str(tmp_path / "clip.mp4"), _index(), 10, 10, "2M", 1, errf)


def test_geometry_selection_and_frame_count_failures_are_cut_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cut past the source, a wrong decoded selection, or an exact-count miss is a property of the source."""
    beyond = _cut_one(tmp_path, {"startFrame": 50, "endFrame": 70, "output": str(tmp_path / "late.mp4")})
    assert isinstance(beyond["failure"], CutError)

    def wrong_frames(*_args: object, **_kwargs: object) -> bool:
        msg = "decoded source PTS do not match the requested frame range"
        raise CutError(msg)

    monkeypatch.setattr(smart_cut, "_run_encode", wrong_frames)
    assert isinstance(_cut_one(tmp_path)["failure"], CutError)

    monkeypatch.setattr(smart_cut, "_run_encode", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(smart_cut, "_output_frame_count", lambda _path: 9)
    short = _cut_one(tmp_path)
    assert isinstance(short["failure"], CutError)
    assert short["error"] == "wrote 9/10 frames"

    monkeypatch.setattr(smart_cut, "_output_frame_count", lambda _path: -1)
    uncounted = _cut_one(tmp_path)["failure"]
    assert isinstance(uncounted, FFmpegRunError)
    assert not isinstance(uncounted, MediaError)


def test_a_wrong_smart_cut_head_falls_back_to_the_full_reencode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the final re-encode judges the cut, so a bad head selection is not reported on its own."""
    calls: list[int] = []

    def encode(*args: Any, **_kwargs: object) -> bool:  # noqa: ANN401
        calls.append(args[4])  # nframes
        if len(calls) == 1:
            msg = "decoded source PTS do not match the requested frame range"
            raise CutError(msg)
        return True

    index = _Index(1, 30, list(range(60)), stream_copy_safe=False, keyframes=(0, 15, 30), smart_cut_safe=True)
    monkeypatch.setattr(smart_cut, "_run_encode", encode)
    monkeypatch.setattr(smart_cut, "_output_frame_count", lambda _path: 10)
    record = smart_cut._cut_one(
        "source.mp4", _cut(tmp_path), index, "2M", 1, allow_stream_copy=True, smart_cut=True, tmp_dir=str(tmp_path)
    )
    assert (record["success"], record["mode"], record["failure"]) == (True, "encode", None)
    assert calls == [5, 10]


def test_on_cut_receives_each_record_as_its_cut_finishes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers can consume and remove each output while later cuts run."""
    monkeypatch.setattr(smart_cut, "_probe_index", lambda _source, verify_index="auto": _index())  # noqa: ARG005
    monkeypatch.setattr(
        smart_cut, "_cut_one", lambda _source, cut, *_args, **_kwargs: {"output": cut["output"], "success": True}
    )
    seen: list[str] = []
    cuts = [{"startFrame": i, "endFrame": i + 1, "output": str(tmp_path / f"{i}.mp4")} for i in (0, 10, 20)]

    records = smart_cut.cut_plan("source.mp4", cuts, workers=2, on_cut=lambda record: seen.append(record["output"]))

    assert sorted(seen) == sorted(cut["output"] for cut in cuts)
    assert [record["output"] for record in records] == [cut["output"] for cut in cuts]
