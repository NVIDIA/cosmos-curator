# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distinguish invalid local media from inconclusive FFprobe failures."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from cosmos_curator.next.media import ffmpeg

_LOCAL = Path("/staged/source.mp4")
_SIGNED = "https://private.invalid/source.mp4?signature=secret"
_INVALID_DATA = b'{"error": {"code": -1094995529, "string": "Invalid data found when processing input"}}'


@pytest.mark.parametrize(
    ("source", "returncode", "stdout", "expected"),
    [
        pytest.param(_LOCAL, 1, _INVALID_DATA, ffmpeg.InvalidMediaError, id="local-invalid-data"),
        pytest.param(_SIGNED, 1, _INVALID_DATA, ffmpeg.ProbeError, id="remote-invalid-data"),
        pytest.param(_LOCAL, -9, _INVALID_DATA, ffmpeg.ProbeError, id="killed-after-reporting-error"),
        pytest.param(_LOCAL, -9, b"", ffmpeg.ProbeError, id="killed-without-output"),
        pytest.param(_LOCAL, 1, b'{"error": {"code": -12}}', ffmpeg.ProbeError, id="out-of-memory"),
        pytest.param(_LOCAL, 1, b'{"error": {"code": -5}}', ffmpeg.ProbeError, id="io-error"),
        pytest.param(_LOCAL, 1, b'{"error": {"code": -2}}', ffmpeg.ProbeError, id="missing-local-file"),
        pytest.param(_LOCAL, 1, b"", ffmpeg.ProbeError, id="stderr-alone-is-inconclusive"),
        pytest.param(_LOCAL, 1, b'{"error":', ffmpeg.ProbeError, id="truncated-json"),
        pytest.param(_LOCAL, 1, b"\xff", ffmpeg.ProbeError, id="invalid-encoding"),
        pytest.param(_LOCAL, 1, b"[]", ffmpeg.ProbeError, id="invalid-payload"),
        pytest.param(_LOCAL, 1, b'{"error": null}', ffmpeg.ProbeError, id="missing-error-code"),
    ],
)
def test_probe_failure_requires_explicit_local_media_error(
    monkeypatch: pytest.MonkeyPatch,
    source: str | Path,
    returncode: int,
    stdout: bytes,
    expected: type[RuntimeError],
) -> None:
    """Neither stderr text, remote invalid data, nor killed probes prove permanent corruption."""

    def fail(command: list[str], **_: object) -> None:
        assert "-show_error" in command
        raise subprocess.CalledProcessError(
            returncode,
            command,
            output=stdout,
            stderr=f"{source}: Invalid data found when processing input".encode(),
        )

    monkeypatch.setattr(ffmpeg.subprocess, "run", fail)
    with pytest.raises(expected) as exc_info:
        ffmpeg.probe_video_source(source)
    assert isinstance(exc_info.value, ffmpeg.MediaError) == (expected is ffmpeg.InvalidMediaError)
    assert "secret" not in str(exc_info.value)
    assert str(source) not in str(exc_info.value)


def test_probe_timeout_is_not_a_terminal_media_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    """Timeout classification belongs to the wrapper, without caller inspection of causes."""

    def timeout(command: list[str], **_: object) -> None:
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(ffmpeg.subprocess, "run", timeout)
    with pytest.raises(ffmpeg.ProbeError, match="timed out after 1s") as exc_info:
        ffmpeg.probe_video_source(_SIGNED, timeout_s=1)
    assert not isinstance(exc_info.value, ffmpeg.MediaError)
    assert "secret" not in str(exc_info.value)


def test_missing_probe_binary_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """An absent executable is an environment failure, not an invalid video."""

    def missing(*_: object, **__: object) -> None:
        message = "ffprobe"
        raise FileNotFoundError(message)

    monkeypatch.setattr(ffmpeg.subprocess, "run", missing)
    with pytest.raises(FileNotFoundError, match="ffprobe"):
        ffmpeg.probe_video_source(_LOCAL)


@pytest.mark.parametrize("stdout", [b"not json", b"\xff", b"[]", b"{}", b'{"streams": null}'])
def test_malformed_probe_response_is_not_terminal(monkeypatch: pytest.MonkeyPatch, stdout: bytes) -> None:
    """Malformed responses cannot establish an unusable source even after a successful exit."""
    monkeypatch.setattr(ffmpeg.subprocess, "run", lambda *_, **__: SimpleNamespace(stdout=stdout))
    with pytest.raises(ffmpeg.ProbeError):
        ffmpeg.probe_video_source(_LOCAL)


def test_readable_media_without_video_is_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid probe response can establish a permanent media-contract mismatch."""
    monkeypatch.setattr(
        ffmpeg.subprocess,
        "run",
        lambda *_, **__: SimpleNamespace(stdout=b'{"streams": [{"codec_type": "audio"}]}'),
    )
    with pytest.raises(ffmpeg.UnsupportedMediaError, match="No video stream"):
        ffmpeg.probe_video_source(_LOCAL)


@pytest.mark.env("default")
def test_real_probe_classifies_corruption_and_missing_path_differently(tmp_path: Path) -> None:
    """Real FFprobe distinguishes two failures that both exit with status one."""
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"invalid media")
    with pytest.raises(ffmpeg.InvalidMediaError):
        ffmpeg.probe_video_source(corrupt)
    with pytest.raises(ffmpeg.ProbeError):
        ffmpeg.probe_video_source(tmp_path / "missing.mp4")
