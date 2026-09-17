# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download reuse, FFmpeg batching, and media durability before checkpointing."""

from pathlib import Path

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.media.ffmpeg import TranscodeError, VideoMetadata
from cosmos_curator.next.recipes.video_transcode import processing
from cosmos_curator.next.recipes.video_transcode.config import ResolvedVideoTranscodeConfig
from cosmos_curator.next.recipes.video_transcode.contracts import CLIP_FIELDS, PAYLOAD_SCHEMA, RESULT_SCHEMA
from cosmos_curator.next.recipes.video_transcode.pipeline import SpanDatasource


def test_download_once_and_stream_bounded_clips(
    config: ResolvedVideoTranscodeConfig,
    dataset: lance.LanceDataset,
    source_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A downloaded source crosses the stage boundary without relying on its original path."""
    tasks = SpanDatasource(dataset, config).get_read_tasks(2)
    spans = pa.concat_tables([table for task in tasks for table in task()])
    calls = []
    download = processing.download_bytes

    def download_once(source: str, **kwargs: str) -> bytes:
        calls.append("download")
        return download(source, **kwargs)

    def encode(_source: Path, outputs: tuple, _config: object) -> None:
        assert _source.read_bytes() == b"source-video"
        calls.append(f"encode-{len(outputs)}")
        for _, path in outputs:
            path.write_bytes(b"encoded-clip")

    monkeypatch.setattr(processing, "download_bytes", download_once)
    monkeypatch.setattr(processing, "assert_video_encoder_available", lambda _: None)
    monkeypatch.setattr(processing, "_encode", encode)
    monkeypatch.setattr(
        processing, "probe_video_path", lambda *_args, **_kwargs: VideoMetadata(1_000_000_000, 64, 64, 10.0, 10, "h264")
    )
    downloaded = processing.SourceDownloader(config=config)(processing.group_source_spans(spans).to_pylist()[0])
    assert calls == ["download"]
    source_path.unlink()
    iterator = processing.transcode_source(downloaded, config=config)
    first = next(iterator)
    assert calls == ["download", "encode-2"]
    assert "source_bytes" not in downloaded
    rows = [first, *iterator]
    assert calls == ["download", "encode-2", "encode-2"]
    assert len(rows) == 4
    terminal = processing.upload_clips(pa.Table.from_pylist(rows), config=config)
    assert terminal.schema.equals(RESULT_SCHEMA, check_metadata=True)
    for uri in terminal["clip_uri"].to_pylist():
        assert Path(uri).read_bytes() == b"encoded-clip"


@pytest.mark.parametrize("num_clips", [0, 1, 4])
def test_upload_batch_preserves_result_schema(
    config: ResolvedVideoTranscodeConfig, results: pa.Table, num_clips: int
) -> None:
    """Empty and inferred batches normalize field types, metadata, and span addresses."""
    expected = results.slice(0, num_clips)
    field = RESULT_SCHEMA.field("clip_frame_count")
    expected = expected.set_column(expected.schema.get_field_index(field.name), field, pa.nulls(num_clips, field.type))
    payloads = [bytes([index]) * 50 for index in range(num_clips)]
    batch = expected.append_column(PAYLOAD_SCHEMA.field("clip_bytes"), pa.array(payloads, type=pa.large_binary()))
    if num_clips:
        batch = pa.Table.from_pylist(batch.to_pylist())
    uploaded = processing.upload_clips(batch, config=config)
    assert uploaded.schema.equals(RESULT_SCHEMA, check_metadata=True)
    assert uploaded.equals(expected)
    for uri, payload in zip(uploaded["clip_uri"].to_pylist(), payloads, strict=True):
        assert Path(uri).read_bytes() == payload


def test_download_rejects_changed_source(
    config: ResolvedVideoTranscodeConfig, dataset: lance.LanceDataset, source_path: Path
) -> None:
    """A size mismatch fails in the download stage before scheduling any encoding."""
    spans = pa.concat_tables([table for task in SpanDatasource(dataset, config).get_read_tasks(2) for table in task()])
    source_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source size changed since video-cut"):
        processing.SourceDownloader(config=config)(processing.group_source_spans(spans).to_pylist()[0])


def test_download_failure_produces_no_source(
    config: ResolvedVideoTranscodeConfig, dataset: lance.LanceDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exhausted source read cannot reach transcoding or a durable checkpoint."""
    spans = pa.concat_tables([table for task in SpanDatasource(dataset, config).get_read_tasks(2) for table in task()])

    def fail(*_args: object, **_kwargs: object) -> bytes:
        message = "download unavailable"
        raise OSError(message)

    monkeypatch.setattr(processing, "download_bytes", fail)
    with pytest.raises(OSError, match="download unavailable"):
        processing.SourceDownloader(config=config)(processing.group_source_spans(spans).to_pylist()[0])


def test_upload_failure_produces_no_result(
    config: ResolvedVideoTranscodeConfig, results: pa.Table, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial batch upload cannot return metadata for a failed or unattempted clip."""
    batch = results.append_column(
        PAYLOAD_SCHEMA.field("clip_bytes"), pa.array([b"x" * 50] * results.num_rows, type=pa.large_binary())
    )
    upload = processing.upload_bytes
    uris = results["clip_uri"].to_pylist()

    def fail(payload: bytes, uri: str, *, storage_profile: str) -> None:
        if uri == uris[1]:
            message = "upload unavailable"
            raise OSError(message)
        upload(payload, uri, storage_profile=storage_profile)

    monkeypatch.setattr(processing, "upload_bytes", fail)
    with pytest.raises(OSError, match="upload unavailable"):
        processing.upload_clips(batch, config=config)
    assert Path(uris[0]).read_bytes() == b"x" * 50
    assert all(not Path(uri).exists() for uri in uris[1:])


def test_failed_ffmpeg_batch_falls_back_to_individual_spans(
    config: ResolvedVideoTranscodeConfig, dataset: lance.LanceDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multi-output FFmpeg failures are retried per span without losing identity."""
    spans = pa.concat_tables([table for task in SpanDatasource(dataset, config).get_read_tasks(2) for table in task()])
    sizes = []

    def encode(_source: Path, outputs: tuple, _config: object) -> None:
        sizes.append(len(outputs))
        if len(outputs) > 1:
            message = "batch failed"
            raise TranscodeError(message)
        outputs[0][1].write_bytes(b"clip")

    monkeypatch.setattr(processing, "assert_video_encoder_available", lambda _: None)
    monkeypatch.setattr(processing, "_encode", encode)
    monkeypatch.setattr(
        processing, "probe_video_path", lambda *_args, **_kwargs: VideoMetadata(1_000_000_000, 64, 64, 10.0, 10, "h264")
    )
    downloaded = processing.download_source(processing.group_source_spans(spans).to_pylist()[0], config=config)
    output = pa.Table.from_pylist(list(processing.transcode_source(downloaded, config=config)))
    assert sizes == [2, 1, 1, 2, 1, 1]
    assert set(output["span_id"].to_pylist()) == set(spans["span_id"].to_pylist())
    assert set(CLIP_FIELDS.names).issubset(output.column_names)
    assert output["clip_bytes"].to_pylist() == [b"clip"] * 4
