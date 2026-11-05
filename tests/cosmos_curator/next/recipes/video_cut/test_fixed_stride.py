# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-stride tasks: remote probing, in-place download fallback, and failure classes."""

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pyarrow as pa
import pytest
from botocore.exceptions import ClientError

from cosmos_curator.next.media import ffmpeg
from cosmos_curator.next.media.failures import InvalidMediaError, MediaError, ProbeError, UnsupportedMediaError
from cosmos_curator.next.media.ffmpeg import VideoMetadata
from cosmos_curator.next.recipes.video_cut import fixed_stride, pipeline
from cosmos_curator.next.recipes.video_cut.config import ResolvedVideoCutConfig, resolve_config_data
from cosmos_curator.next.recipes.video_cut.contracts import INPUT_SCHEMA, make_source_id
from cosmos_curator.next.utils import source_objects
from tests.cosmos_curator.next.recipes.video_cut.conftest import probed_source

_URI = "s3://test-bucket/source.mp4"
_SIGNED = "https://private.invalid/source.mp4?signature=secret"
_METADATA = VideoMetadata(5_000_000_000, 64, 48, 30.0, None, "h264")


def _input() -> pa.Table:
    return pa.Table.from_pylist([{"source_id": make_source_id(_URI), "source_uri": _URI}], schema=INPUT_SCHEMA)


@pytest.fixture
def downloads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record complete downloads to local files, keeping source access free of credentials."""
    downloaded: list[Path] = []

    def download(_prefix: object, path: Path) -> None:
        downloaded.append(path)
        path.write_bytes(b"media")

    client = SimpleNamespace(
        s3=SimpleNamespace(
            head_object=lambda **_: {"ContentLength": 5}, generate_presigned_url=lambda *_, **__: _SIGNED
        ),
        download_to_path=download,
    )
    monkeypatch.setattr(fixed_stride, "s3_client", lambda _: client)
    return downloaded


def _cut(config: ResolvedVideoCutConfig) -> dict[str, Any]:
    """Run one source through the fixed-stride task without Ray."""
    planned = fixed_stride.plan_sources(_input(), split=config.split, execution=config.execution)
    assert "video_bytes" not in planned.column_names
    return planned.to_pylist()[0]


def test_remote_metadata_avoids_download(
    config: ResolvedVideoCutConfig, downloads: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful remote probe plans the source without downloading it."""
    seen = []

    def probe(source: str | Path, **_: object) -> VideoMetadata:
        seen.append(source)
        return _METADATA

    monkeypatch.setattr(fixed_stride, "probe_video_source", probe)
    row = _cut(config)
    assert seen == [_SIGNED]
    assert downloads == []
    assert [(span["start_ns"], span["end_ns"]) for span in row["spans"]] == [
        (0, 2_000_000_000),
        (2_000_000_000, 4_000_000_000),
        (4_000_000_000, 5_000_000_000),
    ]
    assert (row["source_size_bytes"], row["source_frame_count"], row["error_stage"]) == (5, None, None)
    assert "secret" not in str(row)


@pytest.mark.parametrize("error_type", [ProbeError, InvalidMediaError, UnsupportedMediaError])
def test_remote_failure_falls_back_to_one_local_probe(
    config: ResolvedVideoCutConfig,
    downloads: list[Path],
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[RuntimeError],
) -> None:
    """An inconclusive remote probe downloads and probes locally within the same task."""

    def remote(*_: object, **__: object) -> VideoMetadata:
        message = "HTTP probe failed"
        raise error_type(message)

    def local(path: Path, **_: object) -> VideoMetadata:
        assert path.read_bytes() == b"media"
        return _METADATA

    monkeypatch.setattr(fixed_stride, "probe_video_source", remote)
    monkeypatch.setattr(fixed_stride, "probe_video_path", local)
    row = _cut(config)
    assert len(downloads) == 1
    assert not downloads[0].exists()
    assert (row["error_stage"], row["source_size_bytes"], len(row["spans"])) == (None, 5, 3)


@pytest.mark.parametrize("probe_cpus", [0.25, 1.0])
def test_reading_probing_and_planning_share_one_task(config: ResolvedVideoCutConfig, probe_cpus: float) -> None:
    """The task CPU reservation bounds probing, download fallback, and planning without an actor pool.

    Ray fuses the read into the probe task only while both reserve the same CPUs and set no other remote args.
    """
    config = resolve_config_data(
        config.model_dump(mode="json"),
        overrides=[f"execution.fixed_stride.probe_cpus={probe_cpus}"],
    )
    processing = Mock()
    pipeline.cut_sources(processing, config)
    processing.read.assert_called_once_with(num_cpus=probe_cpus)
    selection = processing.read.return_value
    selection.map.assert_not_called()
    selection.map_batches.assert_called_once_with(
        fixed_stride.plan_sources,
        batch_size=None,
        batch_format="pyarrow",
        fn_kwargs={"split": config.split, "execution": config.execution},
        num_cpus=probe_cpus,
    )
    processing.write.assert_called_once_with(selection.map_batches.return_value)


@pytest.mark.parametrize("error_type", [InvalidMediaError, UnsupportedMediaError])
def test_bad_media_is_terminal_but_empty_plans_are_successful(
    config: ResolvedVideoCutConfig,
    downloads: list[Path],
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[MediaError],
) -> None:
    """A local probe's media error finishes the source; a short source is a successful empty plan."""

    def invalid(*_: object, **__: object) -> VideoMetadata:
        message = "invalid media"
        raise error_type(message)

    monkeypatch.setattr(fixed_stride, "probe_video_source", invalid)
    monkeypatch.setattr(fixed_stride, "probe_video_path", invalid)
    row = _cut(config)
    assert (row["error_stage"], row["error_message"], row["spans"]) == ("source-probe", "invalid media", [])
    assert row["source_id"] == make_source_id(_URI)
    assert len(downloads) == 1
    assert not downloads[0].exists()
    empty = fixed_stride.planned_source(probed_source(_URI, 100), config.split)
    assert (empty["error_stage"], empty["spans"]) == (None, [])


@pytest.mark.parametrize("remote", [True, False], ids=["head", "download"])
@pytest.mark.parametrize(
    ("code", "bucket_exists", "terminal"),
    [
        ("404", True, True),
        ("404", False, False),
        ("NoSuchKey", True, True),
        ("InvalidObjectState", True, True),
        ("NoSuchBucket", False, False),
        ("AccessDenied", True, False),
        ("SlowDown", True, False),
    ],
    ids=["missing-key", "missing-bucket", "no-such-key", "archived", "no-such-bucket", "access-denied", "throttled"],
)
def test_only_unreadable_objects_are_terminal(
    config: ResolvedVideoCutConfig,
    monkeypatch: pytest.MonkeyPatch,
    code: str,
    *,
    bucket_exists: bool,
    terminal: bool,
    remote: bool,
) -> None:
    """A missing or archived object stays unreadable; environment failures propagate from either request."""

    def fail(*_: object, **__: object) -> object:
        raise ClientError({"Error": {"Code": code, "Message": "failed"}}, "GetObject")

    def head_bucket(**_: object) -> dict[str, object]:
        if not bucket_exists:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadBucket")
        return {}

    def inconclusive(*_: object, **__: object) -> VideoMetadata:
        message = "HTTP probe failed"
        raise ProbeError(message)

    client = SimpleNamespace(
        s3=SimpleNamespace(
            head_object=fail if remote else lambda **_: {"ContentLength": 5},
            head_bucket=head_bucket,
            generate_presigned_url=lambda *_, **__: _SIGNED,
        ),
        download_to_path=fail,
    )
    monkeypatch.setattr(fixed_stride, "s3_client", lambda _: client)
    monkeypatch.setattr(source_objects, "s3_client", lambda _: client)
    monkeypatch.setattr(fixed_stride, "probe_video_source", inconclusive)
    if terminal:
        row = _cut(config)
        assert (row["error_stage"], row["error_message"]) == ("source-read", f"S3 source unavailable: {code}")
    else:
        with pytest.raises(ClientError):
            _cut(config)


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
def test_inconclusive_probes_remain_retryable(
    config: ResolvedVideoCutConfig,
    downloads: list[Path],
    monkeypatch: pytest.MonkeyPatch,
    failure: subprocess.SubprocessError,
) -> None:
    """An overloaded probe must not permanently quarantine a valid source, remotely or locally."""

    def fail(*_: object, **__: object) -> None:
        raise failure

    monkeypatch.setattr(ffmpeg.subprocess, "run", fail)
    with pytest.raises(ProbeError):
        _cut(config)
    assert len(downloads) == 1
    assert not downloads[0].exists()
