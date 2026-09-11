# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remote probing, fallback, and source failure classification."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest
from botocore.exceptions import ClientError

from cosmos_curator.next.media import ffmpeg
from cosmos_curator.next.media.ffmpeg import InvalidMediaError, ProbeError, UnsupportedMediaError, VideoMetadata
from cosmos_curator.next.recipes.video_cut import processing
from cosmos_curator.next.recipes.video_cut.config import ResolvedVideoCutConfig, resolve_config_data
from cosmos_curator.next.recipes.video_cut.contracts import INPUT_SCHEMA, make_source_id

_URI = "s3://test-bucket/source.mp4"
_SIGNED = "https://private.invalid/source.mp4?signature=secret"
_METADATA = VideoMetadata(5_000_000_000, 64, 48, 30.0, None, "h264")


def _input() -> pa.Table:
    return pa.Table.from_pylist([{"source_id": make_source_id(_URI), "source_uri": _URI}], schema=INPUT_SCHEMA)


@pytest.fixture
def fake_storage(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record downloads while keeping source access independent of credentials."""
    downloads: list[Path] = []

    def download(_prefix: object, path: Path) -> None:
        downloads.append(path)
        path.write_bytes(b"media")

    client = SimpleNamespace(
        s3=SimpleNamespace(
            head_object=lambda **_: {"ContentLength": 5}, generate_presigned_url=lambda *_, **__: _SIGNED
        ),
        download_to_path=download,
    )
    monkeypatch.setattr(processing, "s3_client", lambda _: client)
    return downloads


def test_remote_metadata_avoids_download(
    config: ResolvedVideoCutConfig,
    fake_storage: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful remote probe produces deterministic spans without staging media."""
    seen = []

    def probe(source: str | Path, **_: object) -> VideoMetadata:
        seen.append(source)
        return _METADATA

    monkeypatch.setattr(processing, "probe_video_source", probe)
    row = processing.plan_sources(_input(), split=config.split, execution=config.execution).to_pylist()[0]
    assert seen == [_SIGNED]
    assert fake_storage == []
    assert [(s["start_ns"], s["end_ns"]) for s in row["spans"]] == [
        (0, 2_000_000_000),
        (2_000_000_000, 4_000_000_000),
        (4_000_000_000, 5_000_000_000),
    ]
    assert "secret" not in str(row)
    assert row["source_frame_count"] is None


def test_probe_fallback_downloads_once(
    config: ResolvedVideoCutConfig,
    fake_storage: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP probing failure falls back to a local probe; temporary bytes are removed."""

    def probe(source: str | Path, **_: object) -> VideoMetadata:
        if isinstance(source, str):
            message = "HTTP probe failed"
            raise ProbeError(message)
        assert source.read_bytes() == b"media"
        return _METADATA

    monkeypatch.setattr(processing, "probe_video_source", probe)
    row = processing.plan_sources(_input(), split=config.split, execution=config.execution).to_pylist()[0]
    assert row["status"] == "success"
    assert len(fake_storage) == 1
    assert not fake_storage[0].exists()


def test_explicit_download_skips_remote_metadata_requests(
    config: ResolvedVideoCutConfig,
    fake_storage: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download mode obtains size from the staged bytes without a preliminary HEAD."""
    config = resolve_config_data(config.model_dump(mode="json"), overrides=["execution.probe_mode=download"])
    client = processing.s3_client(config.execution.storage_profile)

    def unexpected_request(*_: object, **__: object) -> None:
        pytest.fail("Download mode must not request remote probe metadata")

    def probe(source: Path, **_: object) -> VideoMetadata:
        assert source.read_bytes() == b"media"
        return _METADATA

    monkeypatch.setattr(client.s3, "head_object", unexpected_request)
    monkeypatch.setattr(client.s3, "generate_presigned_url", unexpected_request)
    monkeypatch.setattr(processing, "probe_video_source", probe)
    row = processing.plan_sources(_input(), split=config.split, execution=config.execution).to_pylist()[0]
    assert row["status"] == "success"
    assert row["source_size_bytes"] == 5
    assert len(fake_storage) == 1
    assert not fake_storage[0].exists()


@pytest.mark.parametrize("error_type", [InvalidMediaError, UnsupportedMediaError])
def test_bad_media_is_terminal_but_empty_plans_are_successful(
    config: ResolvedVideoCutConfig,
    fake_storage: list[Path],
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[ffmpeg.MediaError],
) -> None:
    """Zero-span and permanent failure outcomes both retain their source identity."""

    def invalid(*_: object, **__: object) -> VideoMetadata:
        message = "invalid media"
        raise error_type(message)

    monkeypatch.setattr(processing, "probe_video_source", invalid)
    row = processing.plan_sources(_input(), split=config.split, execution=config.execution).to_pylist()[0]
    assert row["status"] == "error"
    assert row["spans"] == []
    assert row["error_stage"] == "source-probe"
    assert row["source_id"] == make_source_id(_URI)
    assert len(fake_storage) == 1
    empty = processing.source_result(_URI, VideoMetadata(100, 64, 48, 30.0, 1, "h264"), 5, config.split)
    assert empty["status"] == "success"
    assert empty["spans"] == []


@pytest.mark.parametrize(
    ("code", "terminal"),
    [("404", True), ("AccessDenied", False), ("SlowDown", False), ("InvalidObjectState", False)],
)
def test_storage_errors_are_not_all_bad_media(
    config: ResolvedVideoCutConfig,
    monkeypatch: pytest.MonkeyPatch,
    code: str,
    *,
    terminal: bool,
) -> None:
    """Only known object-level failures become durable terminal outcomes."""

    def fail(*_: object) -> object:
        raise ClientError({"Error": {"Code": code, "Message": "failed"}}, "HeadObject")

    monkeypatch.setattr(processing, "s3_client", fail)
    if terminal:
        assert (
            processing.plan_sources(_input(), split=config.split, execution=config.execution).to_pylist()[0][
                "error_stage"
            ]
            == "source-read"
        )
    else:
        with pytest.raises(ClientError):
            processing.plan_sources(_input(), split=config.split, execution=config.execution)


@pytest.mark.parametrize("probe_mode", ["auto", "download"])
@pytest.mark.parametrize(
    "failure",
    [
        subprocess.TimeoutExpired("ffprobe", 1),
        subprocess.CalledProcessError(-9, "ffprobe"),
        subprocess.CalledProcessError(1, "ffprobe", output=b'{"error": {"code": -12}}'),
        subprocess.CalledProcessError(1, "ffprobe", stderr=b"unclassified failure"),
    ],
    ids=["timeout", "sigkill", "out-of-memory", "unknown-exit"],
)
def test_inconclusive_local_probe_remains_retryable(
    config: ResolvedVideoCutConfig,
    fake_storage: list[Path],
    monkeypatch: pytest.MonkeyPatch,
    probe_mode: str,
    failure: subprocess.SubprocessError,
) -> None:
    """An overloaded probe must not permanently quarantine a valid source."""
    config = resolve_config_data(config.model_dump(mode="json"), overrides=[f"execution.probe_mode={probe_mode}"])

    def fail(*_: object, **__: object) -> None:
        raise failure

    monkeypatch.setattr(ffmpeg.subprocess, "run", fail)
    with pytest.raises(ProbeError):
        processing.plan_sources(_input(), split=config.split, execution=config.execution)
    assert len(fake_storage) == 1
    assert not fake_storage[0].exists()
