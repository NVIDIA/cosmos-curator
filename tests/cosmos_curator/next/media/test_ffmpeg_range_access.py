# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that cloud transcoding uses FFmpeg's range-capable input path."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from cosmos_curator.next.media import ffmpeg
from cosmos_curator.next.media.spans import Span
from cosmos_curator.next.recipes.video_transcode.config import TranscodeConfig


def test_http_transcode_uses_input_seek_before_signed_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Input-side seeking lets FFmpeg issue HTTP ranges instead of staging a whole object."""
    seen: list[str] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        seen.extend(command)
        Path(command[-1]).write_bytes(b"mp4")
        return SimpleNamespace(stdout=b"", stderr=b"")

    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)
    destination = tmp_path / "clip.mp4"
    source = "https://s3.example/object?X-Amz-Signature=secret"

    ffmpeg.transcode_span_to_path(
        source,
        destination,
        Span(start_ns=10_000_000_000, end_ns=20_000_000_000),
        TranscodeConfig(),
    )

    assert seen.index("-ss") < seen.index("-i")
    assert seen[seen.index("-i") + 1] == source
    assert seen[seen.index("-seekable") + 1] == "1"
    assert destination.read_bytes() == b"mp4"


def test_batched_transcode_maps_each_seekable_input_to_its_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A batch repeats input-side seeking and maps each input to one MP4."""
    seen: list[str] = []
    destinations = (tmp_path / "first.mp4", tmp_path / "second.mp4")

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        seen.extend(command)
        for destination in destinations:
            destination.write_bytes(b"mp4")
        return SimpleNamespace(stdout=b"", stderr=b"")

    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)
    source = "https://s3.example/object?X-Amz-Signature=secret"

    ffmpeg.transcode_spans_to_paths(
        source,
        (
            (Span(start_ns=0, end_ns=10_000_000_000), destinations[0]),
            (Span(start_ns=10_000_000_000, end_ns=20_000_000_000), destinations[1]),
        ),
        TranscodeConfig(),
        encoder_threads=2,
    )

    input_positions = [index for index, value in enumerate(seen) if value == "-i"]
    seek_positions = [index for index, value in enumerate(seen) if value == "-ss"]
    assert len(input_positions) == len(seek_positions) == 2
    assert all(seek < input_ for seek, input_ in zip(seek_positions, input_positions, strict=True))
    assert [seen[index + 1] for index in input_positions] == [source, source]
    filter_threads_position = seen.index("-filter_threads")
    assert filter_threads_position < input_positions[0]
    assert seen[filter_threads_position + 1] == "1"
    assert [seen[index + 1] for index, value in enumerate(seen) if value == "-map"] == [
        "0:v:0",
        "0:a:0?",
        "1:v:0",
        "1:a:0?",
    ]
    assert seen.count("+faststart") == 2
    assert [seen[index + 1] for index, value in enumerate(seen) if value == "-threads"] == ["1", "1", "2", "2"]
    assert all(destination.read_bytes() == b"mp4" for destination in destinations)


def test_incomplete_batched_transcode_removes_partial_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A successful exit is still a failure unless every requested MP4 exists."""
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"

    def fake_run(_command: list[str], **_kwargs: object) -> SimpleNamespace:
        first.write_bytes(b"partial batch")
        return SimpleNamespace(stdout=b"", stderr=b"")

    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)

    with pytest.raises(ffmpeg.TranscodeError, match="1 of 2"):
        ffmpeg.transcode_spans_to_paths(
            tmp_path / "source",
            (
                (Span(start_ns=0, end_ns=1_000_000_000), first),
                (Span(start_ns=1_000_000_000, end_ns=2_000_000_000), second),
            ),
            TranscodeConfig(),
        )

    assert not first.exists()
    assert not second.exists()


def test_signed_url_is_redacted_from_ffmpeg_diagnostic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Durable source diagnostics never persist presigned credentials."""
    source = "https://s3.example/object?X-Amz-Signature=secret"

    def fail(command: list[str], **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, command, stderr=f"failed to read {source}".encode())

    monkeypatch.setattr(ffmpeg.subprocess, "run", fail)

    with pytest.raises(ffmpeg.TranscodeError) as exc_info:
        ffmpeg.transcode_span_to_path(
            source,
            tmp_path / "clip.mp4",
            Span(start_ns=0, end_ns=1_000_000_000),
            TranscodeConfig(),
        )

    assert "secret" not in str(exc_info.value)
    assert "<source>" in str(exc_info.value)
