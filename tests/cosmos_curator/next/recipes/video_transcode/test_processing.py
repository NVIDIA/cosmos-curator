# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download reuse, FFmpeg batching, and media durability before checkpointing."""

from contextlib import closing
from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa
import pytest

from cosmos_curator.core.utils.misc import retry_utils
from cosmos_curator.next.enrichment import contracts
from cosmos_curator.next.enrichment.execution import selection_datasource
from cosmos_curator.next.media.ffmpeg import ProbeError, TranscodeError, VideoMetadata
from cosmos_curator.next.recipes.video_transcode import processing
from cosmos_curator.next.recipes.video_transcode.config import ResolvedVideoTranscodeConfig
from cosmos_curator.next.recipes.video_transcode.contracts import CLIP_FIELDS, READ_COLUMNS
from tests.cosmos_curator.next.enrichment.helpers import attempt_all

_CLIP_BYTES = pa.field("clip_bytes", pa.large_binary(), nullable=False)


def _read_spans(enrichment_input: contracts.EnrichmentInput) -> pa.Table:
    """Read every pending span with row addresses, as the checkpointed plan does, without Ray."""
    datasource = selection_datasource(attempt_all(enrichment_input), columns=READ_COLUMNS)
    return pa.concat_tables([table for task in datasource.get_read_tasks(2) for table in task()])


def test_download_once_and_stream_bounded_clips(
    config: ResolvedVideoTranscodeConfig,
    enrichment_input: contracts.EnrichmentInput,
    source_path: Path,
    staged_schema: pa.Schema,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A downloaded source crosses the stage boundary without relying on its original path."""
    spans = _read_spans(enrichment_input)
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
    terminal = processing.upload_clips(pa.Table.from_pylist(rows), config=config, staged_schema=staged_schema)
    assert terminal.schema.equals(staged_schema, check_metadata=True)
    for uri in terminal["clip_uri"].to_pylist():
        assert Path(uri).read_bytes() == b"encoded-clip"


@pytest.mark.parametrize("num_clips", [0, 1, 4])
def test_upload_batch_preserves_result_schema(
    config: ResolvedVideoTranscodeConfig, results: pa.Table, staged_schema: pa.Schema, num_clips: int
) -> None:
    """Empty and inferred batches normalize field types, metadata, and span addresses."""
    expected = results.slice(0, num_clips)
    field = staged_schema.field("clip_frame_count")
    expected = expected.set_column(expected.schema.get_field_index(field.name), field, pa.nulls(num_clips, field.type))
    payloads = [bytes([index]) * 50 for index in range(num_clips)]
    batch = expected.append_column(_CLIP_BYTES, pa.array(payloads, type=pa.large_binary()))
    if num_clips:
        batch = pa.Table.from_pylist(batch.to_pylist())
    uploaded = processing.upload_clips(batch, config=config, staged_schema=staged_schema)
    assert uploaded.schema.equals(staged_schema, check_metadata=True)
    assert uploaded.equals(expected)
    for uri, payload in zip(uploaded["clip_uri"].to_pylist(), payloads, strict=True):
        assert Path(uri).read_bytes() == payload


def test_download_rejects_changed_source(
    config: ResolvedVideoTranscodeConfig, enrichment_input: contracts.EnrichmentInput, source_path: Path
) -> None:
    """A size mismatch fails in the download stage before scheduling any encoding."""
    spans = _read_spans(enrichment_input)
    source_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source size changed since video-cut"):
        processing.SourceDownloader(config=config)(processing.group_source_spans(spans).to_pylist()[0])


def test_download_failure_produces_no_source(
    config: ResolvedVideoTranscodeConfig, enrichment_input: contracts.EnrichmentInput, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exhausted source read cannot reach transcoding or a durable checkpoint."""
    spans = _read_spans(enrichment_input)

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
    batch = results.append_column(_CLIP_BYTES, pa.array([b"x" * 50] * results.num_rows, type=pa.large_binary()))
    upload = processing.upload_bytes
    uris = results["clip_uri"].to_pylist()

    def fail(payload: bytes, uri: str, *, storage_profile: str) -> None:
        if uri == uris[1]:
            message = "upload unavailable"
            raise OSError(message)
        upload(payload, uri, storage_profile=storage_profile)

    monkeypatch.setattr(processing, "upload_bytes", fail)
    with pytest.raises(OSError, match="upload unavailable"):
        processing.upload_clips(batch, config=config, staged_schema=results.schema)
    assert Path(uris[0]).read_bytes() == b"x" * 50
    assert all(not Path(uri).exists() for uri in uris[1:])


def test_failed_ffmpeg_batch_falls_back_to_individual_spans(
    config: ResolvedVideoTranscodeConfig, enrichment_input: contracts.EnrichmentInput, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multi-output FFmpeg failures are retried per span without losing identity."""
    spans = _read_spans(enrichment_input)
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


@pytest.mark.parametrize("ffmpeg_batch_size", [1, 2])
@pytest.mark.parametrize("probe_recovers", [True, False])
def test_clip_probe_retries_reuse_encoded_media(
    config: ResolvedVideoTranscodeConfig,
    enrichment_input: contracts.EnrichmentInput,
    monkeypatch: pytest.MonkeyPatch,
    ffmpeg_batch_size: int,
    *,
    probe_recovers: bool,
) -> None:
    """Retry inspection without re-encoding; exhausted probes must not yield a clip."""
    config = config.model_copy(
        update={
            "execution": config.execution.model_copy(
                update={"ffmpeg_batch_size": ffmpeg_batch_size, "probe_attempts": 2}
            )
        }
    )
    spans = _read_spans(enrichment_input)
    probed_paths: list[Path] = []
    encoded_paths: list[Path] = []

    def encode(_source: Path, outputs: tuple, _config: object) -> None:
        for _, path in outputs:
            encoded_paths.append(path)
            path.write_bytes(b"clip")

    def probe(path: Path, **_kwargs: object) -> VideoMetadata:
        probed_paths.append(path)
        assert path.read_bytes() == b"clip"
        if len(probed_paths) == 1 or not probe_recovers:
            message = "temporary probe failure"
            raise ProbeError(message)
        return VideoMetadata(1_000_000_000, 64, 64, 10.0, None, "h264")

    monkeypatch.setattr(retry_utils.time, "sleep", lambda _: None)
    monkeypatch.setattr(retry_utils, "logger", Mock())
    monkeypatch.setattr(processing, "assert_video_encoder_available", lambda _: None)
    monkeypatch.setattr(processing, "_encode", encode)
    monkeypatch.setattr(processing, "probe_video_path", probe)
    source = processing.download_source(processing.group_source_spans(spans).to_pylist()[0], config=config)
    with closing(processing.transcode_source(source, config=config)) as clips:
        if probe_recovers:
            clip = next(clips)
            assert clip["clip_bytes"] == b"clip"
            assert clip["clip_frame_count"] is None
        else:
            with pytest.raises(ProbeError, match="temporary probe failure"):
                next(clips)
    assert len(encoded_paths) == ffmpeg_batch_size
    assert probed_paths == [encoded_paths[0], encoded_paths[0]]
    assert all(not path.exists() for path in encoded_paths)
