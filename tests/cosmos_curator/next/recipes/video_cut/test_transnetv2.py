# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native-cadence decoding, scene policy, and source outcomes."""

import sys
from functools import partial
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import av
import numpy as np
import pyarrow as pa
import pytest
import torch
from botocore.exceptions import ClientError

from cosmos_curator.core.utils import environment
from cosmos_curator.models import transnetv2 as model_module
from cosmos_curator.models.registry import resolve_registered_model
from cosmos_curator.next.media.ffmpeg import InvalidMediaError, UnsupportedMediaError, probe_video_path
from cosmos_curator.next.recipes.video_cut import transnetv2, transnetv2_nvdec
from cosmos_curator.next.recipes.video_cut.config import ExecutionConfig, TransNetV2Config
from cosmos_curator.next.recipes.video_cut.contracts import SOURCE_SCHEMA, make_source_id, make_span_id

_URI = "s3://test-bucket/source.mp4"


@pytest.mark.parametrize(
    ("transitions", "expected"),
    [
        ([False, False, False, False], [(0, 1_000_000_000), (1_000_000_000, 2_000_000_000)]),
        ([False, True, False, False], [(0, 200_000_000), (700_000_000, 1_700_000_000), (1_700_000_000, 2_000_000_000)]),
        ([True, True, False, True], [(700_000_000, 1_000_000_000)]),
        ([True, True, True, True], []),
    ],
)
def test_scene_boundaries_use_timestamps_and_include_the_last_frame(
    transitions: list[bool], expected: list[tuple[int, int]]
) -> None:
    """Transition runs are omitted, long scenes tile, and the last scene ends at source duration."""
    spans = transnetv2.scene_spans(
        np.asarray(transitions),
        np.asarray([0, 200_000_000, 700_000_000, 1_000_000_000]),
        duration_ns=2_000_000_000,
        split=TransNetV2Config(min_duration_s=0.1, max_duration_s=1.0),
    )
    assert [(span.start_ns, span.end_ns) for span in spans] == expected


def test_short_scenes_and_tails_are_dropped() -> None:
    """The minimum applies to both detected scenes and the remainder of a long scene."""
    spans = transnetv2.scene_spans(
        np.asarray([False, True, False]),
        np.asarray([0, 100_000_000, 200_000_000]),
        duration_ns=1_300_000_000,
        split=TransNetV2Config(min_duration_s=0.2, max_duration_s=1.0),
    )
    assert [(span.start_ns, span.end_ns) for span in spans] == [(200_000_000, 1_200_000_000)]


def test_decode_retains_native_pts_and_resizes(video_path: Path) -> None:
    """Decoding does not replace variable frame timing with average-FPS estimates."""
    frames, timestamps = transnetv2.decode_frames(video_path, cpus=1)
    assert frames.shape == (4, 27, 48, 3)
    assert frames.dtype == np.uint8
    np.testing.assert_array_equal(timestamps, [0, 40_000_000, 120_000_000, 240_000_000])


def test_decode_transport_preserves_frames_and_drops_encoded_bytes(video_path: Path) -> None:
    """Mixed success/error batches carry only small RGB frames and exact PTS across Arrow."""
    execution = ExecutionConfig()
    source = {"source_id": make_source_id(_URI), "source_uri": _URI, "video_bytes": video_path.read_bytes()}
    decoded = transnetv2.decode_sources(
        pa.Table.from_pylist(
            [source | {"download_error": ""}, source | {"video_bytes": b"", "download_error": "missing source"}]
        ),
        execution=execution,
    )
    assert "video_bytes" not in decoded.column_names
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, decoded.schema) as writer:
        writer.write_table(decoded)
    rows = pa.ipc.open_stream(sink.getvalue()).read_all().to_pylist()
    frames, timestamps = transnetv2.decode_frames(video_path, cpus=execution.transnetv2.cpus)
    np.testing.assert_array_equal(np.frombuffer(rows[0]["frame_bytes"], dtype=np.uint8).reshape(frames.shape), frames)
    np.testing.assert_array_equal(rows[0]["timestamps_ns"], timestamps)
    assert rows[0]["source_size_bytes"] == video_path.stat().st_size
    assert rows[0]["metadata"]["duration_ns"] == probe_video_path(video_path).duration_ns
    assert rows[1]["error_stage"] == "source-read"
    assert rows[1]["error_message"] == "missing source"
    assert rows[1]["frame_bytes"] == b""
    assert rows[1]["timestamps_ns"] == []


@pytest.mark.parametrize(
    ("operation", "stage"), [("probe_video_path", "source-probe"), ("decode_frames", "source-decode")]
)
@pytest.mark.parametrize("failure", [InvalidMediaError, RuntimeError])
def test_decode_failure_classification(
    video_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str, stage: str, failure: type[RuntimeError]
) -> None:
    """Moving CPU work upstream preserves terminal media errors and retryable operational failures."""
    monkeypatch.setattr(transnetv2, operation, Mock(side_effect=failure("injected failure")))
    source = {
        "source_id": make_source_id(_URI),
        "source_uri": _URI,
        "video_bytes": video_path.read_bytes(),
        "download_error": "",
    }
    batch = pa.Table.from_pylist([source])
    if failure is RuntimeError:
        with pytest.raises(RuntimeError, match="injected failure"):
            transnetv2.decode_sources(batch, execution=ExecutionConfig())
    else:
        row = transnetv2.decode_sources(batch, execution=ExecutionConfig()).to_pylist()[0]
        assert row["source_id"] == source["source_id"]
        assert row["error_stage"] == stage
        assert row["error_message"] == "injected failure"
        assert row["metadata"] is None
        assert row["frame_bytes"] == b""


def test_weights_must_be_staged_without_checksum_verification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preflight requires a file and leaves its contents to staging and model loading."""
    monkeypatch.setattr(environment, "CONTAINER_PATHS_MODEL_WEIGHT_CACHE_DIR", tmp_path)
    path = tmp_path / resolve_registered_model("transnetv2").model_id / "transnetv2-pytorch-weights.pth"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"staged checkpoint")
    assert transnetv2.validate_weights() == path
    path.unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint file is not staged"):
        transnetv2.validate_weights()
    path.mkdir()
    with pytest.raises(FileNotFoundError, match="checkpoint file is not staged"):
        transnetv2.validate_weights()


@pytest.mark.parametrize("code", ["404", "NoSuchKey", "AccessDenied", "SlowDown", "InvalidObjectState"])
def test_download_failure_classification(code: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing objects are terminal; shared or retryable storage failures propagate."""
    error = ClientError({"Error": {"Code": code}}, "GetObject")
    monkeypatch.setattr(transnetv2, "download_bytes", Mock(side_effect=error))
    downloader = transnetv2.SourceDownloader(storage_profile="default")
    source = {"source_id": make_source_id(_URI), "source_uri": _URI}
    if code in {"404", "NoSuchKey"}:
        result = downloader(source)
        assert result["download_error"] == f"S3 source unavailable: {code}"
        assert result["video_bytes"] == b""
    else:
        with pytest.raises(ClientError):
            downloader(source)


@pytest.mark.parametrize("nvdec", [False, True])
def test_actor_reuses_model_and_emits_common_source_results(
    video_path: Path, monkeypatch: pytest.MonkeyPatch, *, nvdec: bool
) -> None:
    """Actual CPU decode and windowing feed the common publication schema with stable span IDs."""
    loader = Mock(return_value=lambda inputs: torch.zeros((*inputs.shape[:2], 1)))
    monkeypatch.setattr(model_module, "load_transnetv2_model", loader)
    monkeypatch.setattr(transnetv2, "validate_weights", Mock(return_value=Path("/config/models/test.pth")))
    monkeypatch.setattr(transnetv2, "predict_transitions", partial(transnetv2.predict_transitions, device="cpu"))
    planner = transnetv2.TransNetV2Planner(
        split=TransNetV2Config(min_duration_s=0.01, decoder="nvdec" if nvdec else "pyav"), execution=ExecutionConfig()
    )
    source = {
        "source_id": make_source_id(_URI),
        "source_uri": _URI,
        "video_bytes": video_path.read_bytes(),
        "download_error": "",
    }
    decode_gpu = Mock(side_effect=partial(transnetv2.decode_frames, cpus=1))
    monkeypatch.setattr(transnetv2_nvdec, "decode_frames_nvdec", decode_gpu)
    decoded = pa.Table.from_pylist([source])
    if not nvdec:
        decoded = transnetv2.decode_sources(decoded, execution=ExecutionConfig())
    monkeypatch.setattr(transnetv2, "decode_frames", Mock(side_effect=AssertionError("GPU actors must not decode")))
    for _ in range(2):
        result = planner(decoded)
        assert result.schema == SOURCE_SCHEMA
        row = result.to_pylist()[0]
        assert row["status"] == "success"
        duration_ns = probe_video_path(video_path).duration_ns
        assert row["spans"] == [
            {"span_id": make_span_id(source["source_id"], 0, duration_ns), "start_ns": 0, "end_ns": duration_ns}
        ]
        assert row["source_size_bytes"] == len(source["video_bytes"])
    loader.assert_called_once()
    assert decode_gpu.call_count == (2 if nvdec else 0)
    missing = pa.Table.from_pylist([source | {"download_error": "S3 source unavailable: 404"}])
    if not nvdec:
        missing = transnetv2.decode_sources(missing, execution=ExecutionConfig())
    failed = planner(missing).to_pylist()[0]
    assert failed["status"] == "error"
    assert failed["error_stage"] == "source-read"
    assert failed["spans"] == []


def test_nvdec_invalid_pyav_input_becomes_source_error(video_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid data found after probing becomes a terminal decode error before GPU initialization."""
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", ModuleType("PyNvVideoCodec"))
    monkeypatch.setattr(
        transnetv2_nvdec.av, "open", Mock(side_effect=av.error.InvalidDataError(0, "invalid video data"))
    )
    source = {
        "source_id": make_source_id(_URI),
        "source_uri": _URI,
        "video_bytes": video_path.read_bytes(),
        "download_error": "",
    }
    row = transnetv2._decode_source(source, ExecutionConfig(), nvdec=True)
    assert row["error_stage"] == "source-decode"
    assert row["error_message"] == "Source contains invalid video data"
    assert row["frame_bytes"] == b""
    assert row["timestamps_ns"] == []


def test_nvdec_stream_mismatch_becomes_terminal_source_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A different selected stream is reported without GPU initialization or inference."""
    path = tmp_path / "multistream.mp4"
    with av.open(str(path), "w") as container:
        streams = [container.add_stream("mpeg4", rate=25) for _ in range(2)]
        for stream in streams:
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for index, stream in enumerate(streams):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), index * 255, dtype=np.uint8), format="rgb24")
            container.mux(stream.encode(frame))
            container.mux(stream.encode())
    with av.open(str(path)) as container:
        assert len(container.streams.video) == 2
        selected_stream = container.streams.video[1].index

    demuxer = Mock()
    demuxer.GetVideoStreamId.return_value = selected_stream
    nvc = ModuleType("PyNvVideoCodec")
    monkeypatch.setattr(nvc, "CreateDemuxer", Mock(return_value=demuxer), raising=False)
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", nvc)
    cuda = Mock(side_effect=AssertionError("Stream mismatch must be rejected before CUDA initialization"))
    monkeypatch.setattr(torch.cuda, "current_stream", cuda)
    with pytest.raises(UnsupportedMediaError, match="different video stream"):
        transnetv2_nvdec.decode_frames_nvdec(path)

    model = Mock(side_effect=AssertionError("Unsupported sources must not reach inference"))
    monkeypatch.setattr(model_module, "load_transnetv2_model", Mock(return_value=model))
    monkeypatch.setattr(transnetv2, "validate_weights", Mock(return_value=Path("/config/models/test.pth")))
    planner = transnetv2.TransNetV2Planner(split=TransNetV2Config(decoder="nvdec"), execution=ExecutionConfig())
    source = {
        "source_id": make_source_id(_URI),
        "source_uri": _URI,
        "video_bytes": path.read_bytes(),
        "download_error": "",
    }
    result = planner(pa.Table.from_pylist([source]))
    assert result.schema == SOURCE_SCHEMA
    row = result.to_pylist()[0]
    assert row["source_id"] == source["source_id"]
    assert row["status"] == "error"
    assert row["error_stage"] == "source-decode"
    assert row["error_message"] == "NVDEC selected a different video stream; use split.decoder=pyav for this source"
    assert row["spans"] == []
    cuda.assert_not_called()
    model.assert_not_called()


@pytest.mark.parametrize("failure", [InvalidMediaError, RuntimeError])
def test_nvdec_errors_preserve_retry_policy(
    video_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type[RuntimeError]
) -> None:
    """Recognized media errors are terminal; NVDEC/CUDA infrastructure errors propagate."""
    monkeypatch.setattr(transnetv2_nvdec, "decode_frames_nvdec", Mock(side_effect=failure("NVDEC failed")))
    source = {
        "source_id": make_source_id(_URI),
        "source_uri": _URI,
        "video_bytes": video_path.read_bytes(),
        "download_error": "",
    }
    if failure is RuntimeError:
        with pytest.raises(RuntimeError, match="NVDEC failed"):
            transnetv2._decode_source(source, ExecutionConfig(), nvdec=True)
    else:
        row = transnetv2._decode_source(source, ExecutionConfig(), nvdec=True)
        assert (row["error_stage"], row["error_message"]) == ("source-decode", "NVDEC failed")


def test_actor_keeps_model_failures_retryable(video_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An inference failure must never become a durably skipped bad-media source."""
    monkeypatch.setattr(model_module, "load_transnetv2_model", Mock())
    monkeypatch.setattr(transnetv2, "validate_weights", Mock(return_value=Path("/config/models/test.pth")))
    monkeypatch.setattr(transnetv2, "predict_transitions", Mock(side_effect=RuntimeError("inference failed")))
    planner = transnetv2.TransNetV2Planner(split=TransNetV2Config(decoder="pyav"), execution=ExecutionConfig())
    source = {
        "source_id": make_source_id(_URI),
        "source_uri": _URI,
        "video_bytes": video_path.read_bytes(),
        "download_error": "",
    }
    decoded = transnetv2.decode_sources(pa.Table.from_pylist([source]), execution=ExecutionConfig())
    with pytest.raises(RuntimeError, match="inference failed"):
        planner(decoded)
    monkeypatch.setattr(transnetv2, "decode_frames", Mock(side_effect=InvalidMediaError("invalid media")))
    failed_decode = transnetv2.decode_sources(pa.Table.from_pylist([source]), execution=ExecutionConfig())
    row = planner(failed_decode).to_pylist()[0]
    assert row["status"] == "error"
    assert row["error_stage"] == "source-decode"
