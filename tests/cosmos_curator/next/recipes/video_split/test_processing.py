# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for streaming source transcoding and independent clip upload."""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

from cosmos_curator.core.utils.misc import retry_utils
from cosmos_curator.next.media.ffmpeg import (
    InvalidMediaError,
    ProbeError,
    TranscodeError,
    UnsupportedMediaError,
    VideoMetadata,
)
from cosmos_curator.next.media.spans import Span
from cosmos_curator.next.recipes.video_split import processing
from cosmos_curator.next.recipes.video_split.config import ResolvedVideoSplitConfig, resolve_config_data
from cosmos_curator.next.recipes.video_split.identities import make_source_id

_SOURCE_URI = "s3://example-bucket/raw/source.mp4"


class _NullLogger:
    def warning(self, _message: str, *_args: object) -> None:
        pass


@pytest.fixture(autouse=True)
def _worker_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep real FFmpeg, backoff waits, and noisy worker logs outside unit tests."""
    monkeypatch.setattr(processing, "assert_video_encoder_available", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(processing, "logger", _NullLogger())
    monkeypatch.setattr(retry_utils, "logger", _NullLogger())
    monkeypatch.setattr(retry_utils.time, "sleep", lambda _: None)


def _config(*, ffmpeg_batch_size: int = 16, probe_attempts: int = 1) -> ResolvedVideoSplitConfig:
    return resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-split",
            "input": {"uris": [_SOURCE_URI]},
            "split": {"duration_s": 10.0, "stride_s": 10.0, "min_duration_s": 2.0},
            "output": {"media_root": "s3://example-bucket/output"},
            "execution": {
                "storage_attempts": 1,
                "probe_attempts": probe_attempts,
                "transcode_attempts": 1,
                "ffmpeg_batch_size": ffmpeg_batch_size,
            },
        }
    )


def _metadata(duration_ns: int = 25_000_000_000) -> VideoMetadata:
    return VideoMetadata(
        duration_ns=duration_ns,
        width=1920,
        height=1080,
        frame_rate=30.0,
        frame_count=750,
        video_codec="h264",
    )


def _happy_media(
    monkeypatch: pytest.MonkeyPatch,
    *,
    duration_ns: int = 25_000_000_000,
    observed: dict[str, Any] | None = None,
) -> None:
    state = observed if observed is not None else {}

    def fake_download(source_uri: str, **_kwargs: object) -> bytes:
        state.setdefault("downloads", []).append(source_uri)
        return b"source bytes"

    def fake_probe(path: Path, **_kwargs: object) -> VideoMetadata:
        state.setdefault("probes", []).append(path.name)
        return _metadata(duration_ns) if path.name == "source.mp4" else _metadata(10_000_000_000)

    def fake_transcode(source: Path, destination: Path, span: Span, *_args: object, **_kwargs: object) -> None:
        assert source.read_bytes() == b"source bytes"
        state.setdefault("transcode_sources", []).append(source)
        state.setdefault("spans", []).append(span)
        state.setdefault("clip_paths", []).append(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"clip bytes")

    def fake_transcode_batch(
        source: Path,
        outputs: Sequence[tuple[Span, Path]],
        *_args: object,
        **_kwargs: object,
    ) -> None:
        state.setdefault("ffmpeg_batch_sizes", []).append(len(outputs))
        for span, destination in outputs:
            fake_transcode(source, destination, span)

    def fake_upload(payload: bytes, destination_uri: str, **_kwargs: object) -> None:
        state.setdefault("uploads", []).append((destination_uri, payload))

    monkeypatch.setattr(processing, "download_bytes", fake_download)
    monkeypatch.setattr(processing, "probe_video_path", fake_probe)
    monkeypatch.setattr(processing, "transcode_span_to_path", fake_transcode)
    monkeypatch.setattr(processing, "transcode_spans_to_paths", fake_transcode_batch)
    monkeypatch.setattr(processing, "upload_bytes", fake_upload)


def _transcode(config: ResolvedVideoSplitConfig) -> list[dict[str, Any]]:
    downloaded = processing.download_and_plan_source({"source_uri": _SOURCE_URI}, config=config)
    return list(processing.transcode_source(downloaded, config=config))


def test_source_is_downloaded_once_then_clips_upload_in_a_separate_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All transcodes share one local source while upload consumes streamed bytes later."""
    observed: dict[str, Any] = {}
    _happy_media(monkeypatch, observed=observed)
    config = _config()

    transcoded = _transcode(config)

    assert observed["downloads"] == [_SOURCE_URI]
    assert len(set(observed["transcode_sources"])) == 1
    assert observed["ffmpeg_batch_sizes"] == [3]
    assert [(span.start_ns, span.end_ns) for span in observed["spans"]] == [
        (0, 10_000_000_000),
        (10_000_000_000, 20_000_000_000),
        (20_000_000_000, 25_000_000_000),
    ]
    assert "uploads" not in observed
    assert all(row["record_type"] == "clip" and row["clip_bytes"] == b"clip bytes" for row in transcoded)
    assert all(not path.exists() for path in observed["clip_paths"])

    uploaded = [processing.upload_clip(row, config=config) for row in transcoded]

    assert len(observed["uploads"]) == 3
    assert all(payload == b"clip bytes" for _, payload in observed["uploads"])
    assert all(row["clip_bytes"] == b"" for row in uploaded)


def test_transcode_generator_yields_before_finishing_the_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """A long source cannot accumulate all clip payloads in one Python list."""
    observed: dict[str, Any] = {}
    _happy_media(monkeypatch, duration_ns=45_000_000_000, observed=observed)
    config = _config(ffmpeg_batch_size=2)
    downloaded = processing.download_and_plan_source({"source_uri": _SOURCE_URI}, config=config)

    records = processing.transcode_source(downloaded, config=config)
    assert "ffmpeg_batch_sizes" not in observed

    first = next(records)
    assert first["clip_bytes"] == b"clip bytes"
    assert observed["ffmpeg_batch_sizes"] == [2]

    next(records)
    assert observed["ffmpeg_batch_sizes"] == [2]

    next(records)
    assert observed["ffmpeg_batch_sizes"] == [2, 2]

    assert len(list(records)) == 2
    assert len(observed["spans"]) == 5


def test_valid_short_source_emits_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Success is represented only by clips, so a valid zero-clip source is silent."""
    _happy_media(monkeypatch, duration_ns=1_000_000_000)

    assert _transcode(_config()) == []


def test_known_partial_source_reuses_metadata_and_plans_only_missing_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical source metadata avoids a second probe while one download serves missing clips."""
    observed: dict[str, Any] = {}
    _happy_media(monkeypatch, observed=observed)
    row = {
        "source_uri": _SOURCE_URI,
        "source_id": make_source_id(_SOURCE_URI),
        "source_known": True,
        "source_size_bytes": 1024,
        "source_duration_ns": 25_000_000_000,
        "source_width": 1920,
        "source_height": 1080,
        "source_frame_rate": 30.0,
        "source_frame_count": 750,
        "source_video_codec": "h264",
        "missing_spans_json": json.dumps([[10_000_000_000, 20_000_000_000]]),
    }
    config = _config()

    downloaded = processing.download_and_plan_source(row, config=config)
    records = list(processing.transcode_source(downloaded, config=config))

    assert observed["downloads"] == [_SOURCE_URI]
    assert "source.mp4" not in observed["probes"]
    assert [(span.start_ns, span.end_ns) for span in observed["spans"]] == [(10_000_000_000, 20_000_000_000)]
    assert len(records) == 1


def _raise(error: Exception) -> Callable[..., object]:
    def _raiser(*_args: object, **_kwargs: object) -> object:
        raise error

    return _raiser


@pytest.mark.parametrize(
    ("failing", "error", "stage"),
    [
        ("download_bytes", OSError("connection reset"), "source-read"),
        ("probe_video_path", InvalidMediaError("unreadable header"), "source-probe"),
    ],
)
def test_expected_source_failures_flow_as_error_records(
    monkeypatch: pytest.MonkeyPatch,
    failing: str,
    error: Exception,
    stage: str,
) -> None:
    """A source that cannot be planned contributes one sparse error row."""
    _happy_media(monkeypatch)
    monkeypatch.setattr(processing, failing, _raise(error))

    records = _transcode(_config())

    assert len(records) == 1
    assert records[0]["record_type"] == "error"
    assert records[0]["clip_id"] == ""
    assert records[0]["error_stage"] == stage
    assert processing.upload_clip(records[0], config=_config()) == records[0]


@pytest.mark.parametrize("error", [KeyError("width"), ProbeError("inconclusive probe")])
def test_unexpected_source_error_fails_the_task(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    """Bugs and unresolved probes must not turn every source into terminal media errors."""
    _happy_media(monkeypatch)
    monkeypatch.setattr(processing, "probe_video_path", _raise(error))

    with pytest.raises(type(error)):
        processing.download_and_plan_source({"source_uri": _SOURCE_URI}, config=_config())


@pytest.mark.parametrize("ffmpeg_batch_size", [1, 16])
def test_clip_probe_retries_without_retranscoding(monkeypatch: pytest.MonkeyPatch, ffmpeg_batch_size: int) -> None:
    """A transient probe failure reuses the encoded clip and preserves its siblings."""
    observed: dict[str, Any] = {}
    _happy_media(monkeypatch, observed=observed)
    config = _config(ffmpeg_batch_size=ffmpeg_batch_size, probe_attempts=3)
    downloaded = processing.download_and_plan_source({"source_uri": _SOURCE_URI}, config=config)
    probes: list[Path] = []

    def flaky_probe(path: Path, **_kwargs: object) -> VideoMetadata:
        probes.append(path)
        assert path.read_bytes() == b"clip bytes"
        if len(probes) < 3:
            message = "inconclusive clip probe"
            raise ProbeError(message)
        return _metadata(10_000_000_000)

    monkeypatch.setattr(processing, "probe_video_path", flaky_probe)
    records = list(processing.transcode_source(downloaded, config=config))

    assert [row["record_type"] for row in records] == ["clip", "clip", "clip"]
    assert all(row["clip_bytes"] == b"clip bytes" for row in records)
    assert len(observed["spans"]) == 3
    assert probes == [observed["clip_paths"][0]] * 3 + observed["clip_paths"][1:]
    assert all(not path.exists() for path in observed["clip_paths"])


@pytest.mark.parametrize("ffmpeg_batch_size", [1, 16])
def test_exhausted_clip_probe_fails_the_task(monkeypatch: pytest.MonkeyPatch, ffmpeg_batch_size: int) -> None:
    """An inconclusive clip stays retryable after local attempts and releases temporary files."""
    observed: dict[str, Any] = {}
    _happy_media(monkeypatch, observed=observed)
    config = _config(ffmpeg_batch_size=ffmpeg_batch_size, probe_attempts=2)
    downloaded = processing.download_and_plan_source({"source_uri": _SOURCE_URI}, config=config)
    probes: list[Path] = []

    def failed_probe(path: Path, **_kwargs: object) -> VideoMetadata:
        probes.append(path)
        message = "inconclusive clip probe"
        raise ProbeError(message)

    monkeypatch.setattr(processing, "probe_video_path", failed_probe)
    with pytest.raises(ProbeError, match="inconclusive clip probe"):
        list(processing.transcode_source(downloaded, config=config))

    assert probes == [observed["clip_paths"][0]] * 2
    assert all(not path.exists() for path in observed["clip_paths"])


@pytest.mark.parametrize("ffmpeg_batch_size", [1, 16])
@pytest.mark.parametrize("error_type", [InvalidMediaError, UnsupportedMediaError])
def test_invalid_clip_probe_keeps_successful_siblings(
    monkeypatch: pytest.MonkeyPatch,
    ffmpeg_batch_size: int,
    error_type: type[InvalidMediaError | UnsupportedMediaError],
) -> None:
    """Confirmed bad media emits one clip error without spending the probe retry budget."""
    observed: dict[str, Any] = {}
    _happy_media(monkeypatch, observed=observed)
    config = _config(ffmpeg_batch_size=ffmpeg_batch_size, probe_attempts=3)
    downloaded = processing.download_and_plan_source({"source_uri": _SOURCE_URI}, config=config)
    failed_clip_id = downloaded["clip_work"][1]["clip_id"]
    probes: list[Path] = []

    def invalid_probe(path: Path, **_kwargs: object) -> VideoMetadata:
        probes.append(path)
        if path.stem == failed_clip_id:
            message = "unusable clip"
            raise error_type(message)
        return _metadata(10_000_000_000)

    monkeypatch.setattr(processing, "probe_video_path", invalid_probe)
    records = list(processing.transcode_source(downloaded, config=config))

    assert [row["record_type"] for row in records] == ["clip", "error", "clip"]
    assert records[1]["clip_id"] == failed_clip_id
    assert records[1]["error_stage"] == "clip-probe"
    assert records[1]["error_message"] == "unusable clip"
    assert records[1]["clip_bytes"] == b""
    assert len(observed["spans"]) == 3
    assert probes == observed["clip_paths"]
    assert all(not path.exists() for path in observed["clip_paths"])


def test_one_clip_failure_keeps_its_successful_siblings(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed multi-output command is retried clip-by-clip and emits one error."""
    observed: dict[str, Any] = {}
    _happy_media(monkeypatch, observed=observed)
    batch_calls: list[tuple[int, ...]] = []
    single_calls: list[int] = []

    def fail_batch(
        _source: Path,
        outputs: Sequence[tuple[Span, Path]],
        *_args: object,
        **_kwargs: object,
    ) -> None:
        batch_calls.append(tuple(span.start_ns for span, _ in outputs))
        message = "batched command failed"
        raise TranscodeError(message)

    def fail_middle_clip(
        source: Path,
        destination: Path,
        span: Span,
        *_args: object,
        **_kwargs: object,
    ) -> None:
        single_calls.append(span.start_ns)
        if span.start_ns == 10_000_000_000:
            message = "broken GOP"
            raise TranscodeError(message)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

    monkeypatch.setattr(processing, "transcode_spans_to_paths", fail_batch)
    monkeypatch.setattr(processing, "transcode_span_to_path", fail_middle_clip)

    records = _transcode(_config())

    assert [row["record_type"] for row in records] == ["clip", "error", "clip"]
    assert records[1]["error_stage"] == "transcode"
    assert records[1]["error_message"] == "broken GOP"
    assert batch_calls == [(0, 10_000_000_000, 20_000_000_000)]
    assert single_calls == [0, 10_000_000_000, 20_000_000_000]


def test_media_write_failure_fails_the_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken output destination is operational failure, not clip data."""
    _happy_media(monkeypatch)
    record = _transcode(_config())[0]
    error = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "write permission denied"}},
        "PutObject",
    )
    monkeypatch.setattr(processing, "upload_bytes", _raise(error))

    with pytest.raises(ClientError):
        processing.upload_clip(record, config=_config())


def test_unexpected_clip_error_fails_the_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """A programming error in transcoding is not converted into an error row."""
    _happy_media(monkeypatch)
    monkeypatch.setattr(processing, "transcode_spans_to_paths", _raise(ValueError("bad argument")))

    with pytest.raises(ValueError, match="bad argument"):
        _transcode(_config())
