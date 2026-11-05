# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TransNetV2 stages: download and GPU admission, native-cadence decoding, scene policy, and source outcomes."""

import sys
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import av
import numpy as np
import pyarrow as pa
import pytest
import ray
import torch
from botocore.exceptions import ClientError
from ray.data import ActorPoolStrategy
from ray.data._internal.execution.execution_callback import ExecutionCallback
from ray.data.context import DataContext

from cosmos_curator.core.utils import environment
from cosmos_curator.models import transnetv2 as model_module
from cosmos_curator.models.registry import resolve_registered_model
from cosmos_curator.next.append.attempts import AppendAttempt, reconcile_and_select_attempt
from cosmos_curator.next.append.execution import checkpointed_processing
from cosmos_curator.next.core.ray_autoscaling import ReleasePendingDownloadActors
from cosmos_curator.next.media.failures import InvalidMediaError, UnsupportedMediaError
from cosmos_curator.next.media.ffmpeg import probe_video_path
from cosmos_curator.next.recipes.video_cut import pipeline, transnetv2, transnetv2_nvdec
from cosmos_curator.next.recipes.video_cut.bindings import resolve_bindings
from cosmos_curator.next.recipes.video_cut.config import (
    ExecutionConfig,
    ResolvedVideoCutConfig,
    TransNetV2Config,
    resolve_config_data,
)
from cosmos_curator.next.recipes.video_cut.contracts import SOURCE_SCHEMA, make_source_id, make_span_id
from cosmos_curator.next.recipes.video_cut.discovery import discover_items
from cosmos_curator.next.utils import source_objects
from tests.cosmos_curator.next.recipes.video_cut.conftest import SOURCE_URIS

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
            [
                source | {"error_stage": None, "error_message": None},
                source | {"video_bytes": b"", "error_stage": "source-read", "error_message": "missing source"},
            ]
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
    assert rows[0]["source_duration_ns"] == probe_video_path(video_path).duration_ns
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
        "error_stage": None,
        "error_message": None,
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
        assert row["source_duration_ns"] is None
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


@pytest.mark.parametrize(
    ("code", "terminal"),
    [
        ("404", True),
        ("NoSuchKey", True),
        ("InvalidObjectState", True),
        ("NoSuchBucket", False),
        ("AccessDenied", False),
        ("SlowDown", False),
    ],
)
def test_download_failure_classification(code: str, monkeypatch: pytest.MonkeyPatch, *, terminal: bool) -> None:
    """Missing or archived objects are terminal, as in fixed-stride; environment failures propagate."""
    error = ClientError({"Error": {"Code": code}}, "GetObject")
    monkeypatch.setattr(transnetv2, "download_bytes", Mock(side_effect=error))
    monkeypatch.setattr(
        source_objects, "s3_client", lambda _: SimpleNamespace(s3=SimpleNamespace(head_bucket=lambda **_: {}))
    )
    downloader = transnetv2.SourceDownloader(storage_profile="default")
    source = {"source_id": make_source_id(_URI), "source_uri": _URI}
    if terminal:
        result = downloader(source)
        assert result["error_message"] == f"S3 source unavailable: {code}"
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
        split=TransNetV2Config(min_duration_s=0.01, decoder="nvdec" if nvdec else "pyav"),
        execution=ExecutionConfig(),
    )
    source = {
        "source_id": make_source_id(_URI),
        "source_uri": _URI,
        "video_bytes": video_path.read_bytes(),
        "error_stage": None,
        "error_message": None,
    }
    decode_gpu = Mock(side_effect=partial(transnetv2.decode_frames, cpus=1))
    monkeypatch.setattr(transnetv2_nvdec, "decode_frames_nvdec", decode_gpu)
    decoded = pa.Table.from_pylist([source])
    if not nvdec:
        decoded = transnetv2.decode_sources(decoded, execution=ExecutionConfig())
    monkeypatch.setattr(transnetv2, "decode_frames", Mock(side_effect=AssertionError("GPU actors must not decode")))
    for _ in range(2):
        result = planner(decoded)
        assert result.schema.equals(SOURCE_SCHEMA, check_metadata=True)
        row = result.to_pylist()[0]
        assert row["error_stage"] is None
        duration_ns = probe_video_path(video_path).duration_ns
        assert row["spans"] == [
            {"span_id": make_span_id(source["source_id"], 0, duration_ns), "start_ns": 0, "end_ns": duration_ns}
        ]
        assert row["source_size_bytes"] == len(source["video_bytes"])
    loader.assert_called_once()
    assert decode_gpu.call_count == (2 if nvdec else 0)
    missing = pa.Table.from_pylist(
        [source | {"error_stage": "source-read", "error_message": "S3 source unavailable: 404"}]
    )
    if not nvdec:
        missing = transnetv2.decode_sources(missing, execution=ExecutionConfig())
    failed = planner(missing).to_pylist()[0]
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
        "error_stage": None,
        "error_message": None,
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
        "error_stage": None,
        "error_message": None,
    }
    result = planner(pa.Table.from_pylist([source]))
    assert result.schema.equals(SOURCE_SCHEMA, check_metadata=True)
    row = result.to_pylist()[0]
    assert row["source_id"] == source["source_id"]
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
        "error_stage": None,
        "error_message": None,
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
        "error_stage": None,
        "error_message": None,
    }
    decoded = transnetv2.decode_sources(pa.Table.from_pylist([source]), execution=ExecutionConfig())
    with pytest.raises(RuntimeError, match="inference failed"):
        planner(decoded)
    monkeypatch.setattr(transnetv2, "decode_frames", Mock(side_effect=InvalidMediaError("invalid media")))
    failed_decode = transnetv2.decode_sources(pa.Table.from_pylist([source]), execution=ExecutionConfig())
    row = planner(failed_decode).to_pylist()[0]
    assert row["error_stage"] == "source-decode"


def _cut_sources(config: ResolvedVideoCutConfig, *, fail: bool) -> AppendAttempt:
    """Run the recipe's plan in a checkpoint scope over a new selection of every source."""
    assert config.input.uris is not None
    attempt = reconcile_and_select_attempt(resolve_bindings(config), lambda: discover_items(config))
    with ExitStack() as stack:
        if fail:
            stack.enter_context(pytest.raises(RuntimeError, match="injected write failure"))
        with checkpointed_processing(attempt) as scope:
            pipeline.cut_sources(scope, config)
    return attempt


@pytest.mark.parametrize("fail_write", [False, True])
def test_transnetv2_actor_resources_and_checkpoint_scope(
    config: ResolvedVideoCutConfig, monkeypatch: pytest.MonkeyPatch, *, fail_write: bool
) -> None:
    """CPU decode and GPU inference scale separately, without freezing startup resource capacity."""
    raw = config.model_dump(mode="json")
    raw["split"] = {"method": "transnetv2"}
    config = resolve_config_data(raw)
    context = DataContext.get_current()
    monkeypatch.setattr(context, "custom_execution_callback_classes", [ExecutionCallback])
    monkeypatch.setattr(context.autoscaling_config, "actor_pool_util_upscaling_threshold", 1.75)
    plan = Mock()
    for method in ("map", "map_batches", "repartition"):
        getattr(plan, method).return_value = plan

    def read(*_args: object, **kwargs: object) -> Mock:
        assert context.checkpoint_config.id_column == "source_id"
        assert kwargs == {"override_num_blocks": len(SOURCE_URIS), "num_cpus": None}
        plan.context = context.copy()
        return plan

    monkeypatch.setattr(ray.data, "read_datasource", read)
    monkeypatch.setattr(transnetv2, "validate_weights", Mock())
    snapshot = Mock(side_effect=AssertionError("Actor pools must not freeze startup resource capacity"))
    monkeypatch.setattr(ray, "cluster_resources", snapshot)
    if fail_write:
        plan.write_datasink.side_effect = RuntimeError("injected write failure")
    attempt = _cut_sources(config, fail=fail_write)
    assert attempt.processing_complete() is not fail_write
    download_args, download = plan.map.call_args
    assert download_args == (transnetv2.SourceDownloader,)
    assert download["resources"] == {"curator_io": 1.0}
    assert download["num_cpus"] == 0.25
    # Decoding, inference, then the shared validation before checkpointed staging.
    assert plan.map_batches.call_count == 3
    decode_args, decode = plan.map_batches.call_args_list[0]
    assert decode_args == (transnetv2.decode_sources,)
    assert (decode["num_cpus"], decode["num_gpus"], decode["memory"]) == (3, 0, 2 * 1024**3)
    assert decode["batch_size"] == 1
    assert decode["fn_kwargs"] == {"execution": config.execution}
    assert "resources" not in decode
    assert "compute" not in decode  # Stateless CPU tasks release their reservations after each task.
    inference_args, inference = plan.map_batches.call_args_list[1]
    assert inference_args == (transnetv2.TransNetV2Planner,)
    assert (inference["num_cpus"], inference["num_gpus"]) == (1, 0.25)
    assert inference["batch_size"] == 1
    assert inference["memory"] == 2 * 1024**3
    assert inference["runtime_env"]["env_vars"]["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "0"
    assert "resources" not in inference
    for stage in (download, inference):
        assert isinstance(stage["compute"], ActorPoolStrategy)
        assert stage["compute"].max_size == float("inf")
        assert stage["compute"].max_tasks_in_flight_per_actor == 1
    assert plan.context.autoscaling_config.actor_pool_util_upscaling_threshold == 1.0
    assert plan.context.custom_execution_callback_classes == [ExecutionCallback, ReleasePendingDownloadActors]
    assert context.custom_execution_callback_classes == [ExecutionCallback]
    assert context.autoscaling_config.actor_pool_util_upscaling_threshold == 1.75
    assert context.checkpoint_config is None
    snapshot.assert_not_called()


def test_nvdec_shares_one_gpu_actor_with_inference(
    config: ResolvedVideoCutConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GPU decode avoids a CPU decode stage and reserves memory for the combined work."""
    raw = config.model_dump(mode="json")
    raw["split"] = {"method": "transnetv2", "decoder": "nvdec"}
    config = resolve_config_data(
        raw,
        overrides=["execution.transnetv2.memory_gib=3", "execution.transnetv2.inference_memory_gib=2"],
    )
    dataset = Mock()
    dataset.context = DataContext.get_current().copy()
    dataset.map.return_value = dataset
    monkeypatch.setattr(transnetv2, "validate_weights", Mock())
    transnetv2.plan(dataset, config)
    dataset.map_batches.assert_called_once()
    args, settings = dataset.map_batches.call_args
    assert args == (transnetv2.TransNetV2Planner,)
    assert settings["fn_constructor_kwargs"]["split"].decoder == "nvdec"
    assert (settings["num_cpus"], settings["num_gpus"], settings["memory"]) == (1, 0.25, 5 * 1024**3)
    assert settings["compute"].max_size == float("inf")
