# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for separate download, preparation, and batched GPU actor semantics."""

from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pytest
import ray
from ray.data import ActorPoolStrategy

from cosmos_curator.core.utils import environment
from cosmos_curator.next.recipes.multimodal_embedding import inference
from cosmos_curator.next.recipes.multimodal_embedding.config import resolve_config_data
from cosmos_curator.next.recipes.multimodal_embedding.contracts import (
    embedding_contract_digest,
    resolve_model,
    result_field_names,
)
from cosmos_curator.next.recipes.multimodal_embedding.episode_input import EPISODE_MAPPED_INPUT_SCHEMA
from cosmos_curator.next.recipes.multimodal_embedding.video_media import decode_selected_frames, inspect_video
from cosmos_curator.next.recipes.multimodal_split.cpu_video_encode import encode_frames_to_path
from cosmos_curator.next.utils.media_reader import MediaObjectReader, MediaSystemicError, MediaTransportError


def _video(path: Path, *, frames: int = 30, pixel_value: int | None = None) -> None:
    encode_frames_to_path(
        (
            np.full((16, 32, 3), index % 256 if pixel_value is None else pixel_value, dtype=np.uint8)
            for index in range(frames)
        ),
        path,
        fps=30,
        encoder="libopenh264",
        bitrate="400K",
        gop_size=30,
        faststart=False,
    )


def _source(path: Path, *, episode_id: str, duration_ns: int = 1_000_000_000) -> dict[str, Any]:
    payload = path.read_bytes() if path.exists() else b""
    return {
        "fragment_id": 0,
        "row_offset": 0,
        "episode_id": episode_id,
        "materialization_digest": "materialization",
        "start_ns": 0,
        "exclusive_end_ns": duration_ns,
        "camera_key": "front_wide",
        "media_location": str(path),
        "media_size_bytes": len(payload) if payload else 1,
        "media_sha256": sha256(payload).hexdigest() if payload else "a" * 64,
    }


def _download_actor() -> inference.VideoDownloadActor:
    return inference.VideoDownloadActor(
        camera_key="front_wide",
        max_episode_duration_s=60.0,
        storage_profile="default",
        media_attempts=1,
    )


def _media_actor(*, model_key: str = "cosmos_embed1_336p", decode_cpus: int = 1) -> inference.VideoPreparationActor:
    return inference.VideoPreparationActor(
        model=resolve_model(model_key), camera_key="front_wide", decode_cpus=decode_cpus
    )


def _prepare(source: pa.Table) -> pa.Table:
    return _media_actor()(_download_actor()(source))


def _gpu_actor(*, model_key: str = "cosmos_embed1_336p") -> inference.VideoEmbeddingBatchActor:
    model = resolve_model(model_key)
    return inference.VideoEmbeddingBatchActor(
        model=model,
        camera_key="front_wide",
        result_key="default",
        digest=embedding_contract_digest(model, camera_key="front_wide", result_key="default"),
        max_episode_duration_s=60.0,
    )


def _fake_preprocessor(frame_count: int) -> object:
    class FakePreprocessor:
        def preprocess(self, frames: tuple[np.ndarray, ...]) -> np.ndarray:
            assert len(frames) == frame_count
            return np.full((frame_count, 3, 8, 8), frames[0].mean(), dtype=np.float32)

    return FakePreprocessor()


class _CPUPreparationActor(inference.VideoPreparationActor):
    """Use production media handling with a lightweight CPU frame processor."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._preprocessor = _fake_preprocessor(self._model.video_frame_count)  # type: ignore[assignment]


class _CPUEmbeddingActor(inference.VideoEmbeddingBatchActor):
    """Use production batch handling with a deterministic CPU encoder."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        dimension = self._model.vector_dimension

        class Encoder:
            def encode_batch(self, videos: np.ndarray) -> np.ndarray:
                vectors = np.zeros((len(videos), dimension), dtype=np.float32)
                for index, video in enumerate(videos):
                    vectors[index, int(float(video.flat[0]) >= 64)] = 1.0
                return vectors

        self._encoder = Encoder()


@pytest.mark.parametrize("model_key", ["cosmos_embed1_224p", "cosmos_embed1_336p", "cosmos_embed1_448p"])
@pytest.mark.parametrize("decode_cpus", [1, 2])
def test_stage_boundary_preserves_selected_frames_and_tensor(
    model_key: str, decode_cpus: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Transporting the verified MP4 preserves the existing sampling and preprocessing input."""
    video = tmp_path / "video.mp4"
    _video(video)
    model = resolve_model(model_key)
    source = pa.Table.from_pylist([_source(video, episode_id="a")], schema=EPISODE_MAPPED_INPUT_SCHEMA)

    class FrameProcessor:
        def preprocess(self, frames: tuple[np.ndarray, ...]) -> np.ndarray:
            return np.stack(frames).transpose(0, 3, 1, 2).astype(np.float32) / 255

    processor = FrameProcessor()

    def load(_model: object, *, cpus: int) -> object:
        assert cpus == decode_cpus
        return processor

    monkeypatch.setattr(inference, "load_video_preprocessor", load)
    downloaded = _download_actor()(source)
    assert downloaded["media_bytes"][0].as_py() == video.read_bytes()
    prepared = _media_actor(model_key=model_key, decode_cpus=decode_cpus)(downloaded).to_pylist()[0]
    expected = processor.preprocess(
        decode_selected_frames(
            video.read_bytes(), inspect_video(video.read_bytes(), frame_count=model.video_frame_count)
        ).frames
    )
    actual = np.frombuffer(prepared["prepared_video"], dtype="<f4").reshape(prepared["prepared_shape"])
    np.testing.assert_array_equal(actual, expected)
    assert "media_bytes" not in prepared


def test_unsupported_duration_skips_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unsupported episode does not touch storage or load a processor."""

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Unsupported media must not be downloaded or preprocessed")

    monkeypatch.setattr(MediaObjectReader, "read", unexpected)
    monkeypatch.setattr(inference, "load_video_preprocessor", unexpected)
    source = _source(tmp_path / "absent.mp4", episode_id="long", duration_ns=120_000_000_000)
    prepared = _prepare(pa.Table.from_pylist([source], schema=EPISODE_MAPPED_INPUT_SCHEMA)).to_pylist()[0]
    assert prepared["preparation_status"] == "unsupported_duration"
    assert prepared["prepared_video"] is None


def test_oversized_media_is_refused_before_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared reader enforces the episode's declared size cap before transferring bytes."""
    video = tmp_path / "video.mp4"
    _video(video)
    source = _source(video, episode_id="oversized")
    source["media_size_bytes"] = 1

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Oversized media must be refused from metadata without downloading")

    monkeypatch.setattr(MediaObjectReader, "_download", unexpected)
    row = _download_actor()(pa.Table.from_pylist([source], schema=EPISODE_MAPPED_INPUT_SCHEMA)).to_pylist()[0]
    assert row["download_error_code"] == "media_object_error"
    assert row["media_bytes"] is None


@pytest.mark.parametrize("mismatch", ["size", "sha256"])
def test_download_checks_exact_media_identity(mismatch: str, tmp_path: Path) -> None:
    """Successful capped reads still need to match the episode's exact size and hash."""
    video = tmp_path / "video.mp4"
    _video(video)
    source = _source(video, episode_id="mismatch")
    if mismatch == "size":
        source["media_size_bytes"] += 1
    else:
        source["media_sha256"] = "0" * 64
    row = _download_actor()(pa.Table.from_pylist([source], schema=EPISODE_MAPPED_INPUT_SCHEMA)).to_pylist()[0]
    assert row["download_error_code"] == f"{mismatch}_mismatch"
    assert row["media_bytes"] is None


def test_media_stage_carries_success_error_and_unsupported_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exactly one row survives per episode; only eligible present media is processed."""
    video = tmp_path / "video.mp4"
    _video(video)
    missing = tmp_path / "missing.mp4"
    loads: list[int] = []

    def load(_model: object, *, cpus: int) -> object:
        assert cpus == 1
        loads.append(1)
        return _fake_preprocessor(8)

    monkeypatch.setattr(inference, "load_video_preprocessor", load)
    rows = [_source(video, episode_id="success"), _source(missing, episode_id="missing")]
    rows.append(_source(video, episode_id="unsupported", duration_ns=120_000_000_000))
    prepared = _prepare(pa.Table.from_pylist(rows, schema=EPISODE_MAPPED_INPUT_SCHEMA)).to_pylist()
    assert [row["episode_id"] for row in prepared] == ["success", "missing", "unsupported"]
    assert [row["preparation_status"] for row in prepared] == ["ready", "error", "unsupported_duration"]
    assert prepared[0]["prepared_shape"] == [8, 3, 8, 8]
    assert len(prepared[0]["prepared_video"]) == 8 * 3 * 8 * 8 * 4
    assert prepared[0]["media_duration_ns"] == 1_000_000_000
    assert prepared[1]["preparation_error_code"] == "media_object_error"
    assert prepared[1]["prepared_video"] is None
    assert prepared[2]["prepared_video"] is None
    assert loads == [1]


def test_gpu_stage_batches_ready_rows_and_passes_other_rows_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The GPU model sees one batch containing only eligible prepared videos."""
    video = tmp_path / "video.mp4"
    _video(video)
    monkeypatch.setattr(inference, "load_video_preprocessor", lambda _model, **_kwargs: _fake_preprocessor(8))
    sources = [
        _source(video, episode_id="a"),
        _source(tmp_path / "missing.mp4", episode_id="missing"),
        _source(video, episode_id="b"),
        _source(video, episode_id="long", duration_ns=120_000_000_000),
    ]
    prepared = _prepare(pa.Table.from_pylist(sources, schema=EPISODE_MAPPED_INPUT_SCHEMA))
    calls: list[tuple[int, ...]] = []

    class FakeEncoder:
        def encode_batch(self, videos: np.ndarray) -> np.ndarray:
            calls.append(videos.shape)
            return np.eye(768, dtype=np.float32)[: len(videos)]

    monkeypatch.setattr(inference, "load_video_encoder", lambda _model: FakeEncoder())
    result = _gpu_actor()(prepared).to_pylist()
    model = resolve_model("cosmos_embed1_336p")
    vector_name, metadata_name = result_field_names("default", model_key=model.key)
    assert calls == [(2, 8, 3, 8, 8)]
    assert [row["episode_id"] for row in result] == ["a", "missing", "b", "long"]
    assert [row[metadata_name]["status"] for row in result] == ["success", "error", "success", "unsupported_duration"]
    np.testing.assert_array_equal(result[0][vector_name], np.eye(768, dtype=np.float32)[0])
    np.testing.assert_array_equal(result[2][vector_name], np.eye(768, dtype=np.float32)[1])
    assert result[1][vector_name] is None
    assert result[3][vector_name] is None


def test_one_invalid_projection_is_an_item_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad projection does not turn another row's valid vector into an error."""
    video = tmp_path / "video.mp4"
    _video(video)
    monkeypatch.setattr(inference, "load_video_preprocessor", lambda _model, **_kwargs: _fake_preprocessor(8))
    sources = [_source(video, episode_id="good"), _source(video, episode_id="bad")]
    prepared = _prepare(pa.Table.from_pylist(sources, schema=EPISODE_MAPPED_INPUT_SCHEMA))

    class FakeEncoder:
        def encode_batch(self, videos: np.ndarray) -> np.ndarray:
            assert len(videos) == 2
            result = np.full((2, 768), 1 / np.sqrt(768), dtype=np.float32)
            result[1] = np.nan
            return result

    monkeypatch.setattr(inference, "load_video_encoder", lambda _model: FakeEncoder())
    rows = _gpu_actor()(prepared).to_pylist()
    metadata_name = result_field_names("default", model_key="cosmos_embed1_336p")[1]
    assert [row[metadata_name]["status"] for row in rows] == ["success", "error"]
    assert rows[1][metadata_name]["error_code"] == "invalid_vector"


def test_whole_batch_gpu_failure_aborts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA or OOM failures remain retryable attempt failures."""
    video = tmp_path / "video.mp4"
    _video(video)
    monkeypatch.setattr(inference, "load_video_preprocessor", lambda _model, **_kwargs: _fake_preprocessor(8))
    source = pa.Table.from_pylist([_source(video, episode_id="a")], schema=EPISODE_MAPPED_INPUT_SCHEMA)
    prepared = _prepare(source)

    class FailingEncoder:
        def encode_batch(self, _videos: np.ndarray) -> np.ndarray:
            msg = "CUDA out of memory"
            raise RuntimeError(msg)

    monkeypatch.setattr(inference, "load_video_encoder", lambda _model: FailingEncoder())
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        _gpu_actor()(prepared)


@pytest.mark.parametrize("error_type", [MediaTransportError, MediaSystemicError])
def test_shared_media_outage_aborts_before_gpu(
    error_type: type[Exception], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Storage outages are never recorded as final episode errors."""
    video = tmp_path / "video.mp4"
    _video(video)

    def unavailable(*_args: object, **_kwargs: object) -> None:
        msg = "storage unavailable"
        raise error_type(msg)

    monkeypatch.setattr(MediaObjectReader, "read", unavailable)
    with pytest.raises(error_type, match="storage unavailable"):
        _download_actor()(pa.Table.from_pylist([_source(video, episode_id="a")], schema=EPISODE_MAPPED_INPUT_SCHEMA))


@pytest.mark.parametrize(
    ("inference_concurrency", "num_input_rows", "expected_ceiling"),
    [("auto", 64, 64), ("auto", 1, 1), (8, 64, 8), (128, 64, 64)],
)
@pytest.mark.parametrize(("media_concurrency", "decode_concurrency"), [("auto", "auto"), (2, 3)])
def test_ray_mapping_separates_io_and_gpu_resources(
    tmp_path: Path,
    inference_concurrency: int | Literal["auto"],
    num_input_rows: int,
    expected_ceiling: int,
    media_concurrency: int | Literal["auto"],
    decode_concurrency: int | Literal["auto"],
) -> None:
    """IO reservations stay separate from a GPU pool bounded by pending work."""
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "multimodal-embedding",
            "input": {"episodes_lance_uri": str(tmp_path / "episodes.lance")},
            "video": {"camera_key": "front_wide"},
            "execution": {
                "media_concurrency": media_concurrency,
                "decode_concurrency": decode_concurrency,
                "inference_concurrency": inference_concurrency,
                "decode_cpus": 2,
                "decode_memory_gib": 3,
            },
        }
    )
    calls: list[tuple[object, dict[str, Any]]] = []

    class FakeDataset:
        def __init__(self) -> None:
            self.context = ray.data.DataContext()
            self.context.autoscaling_config.actor_pool_max_upscaling_delta = None

        def map_batches(self, actor_type: object, **kwargs: object) -> "FakeDataset":
            calls.append((actor_type, kwargs))
            return self

    source: Any = FakeDataset()
    assert inference.map_video_embeddings(source, config, num_input_rows=num_input_rows) is source
    assert source.context.autoscaling_config.actor_pool_util_upscaling_threshold == 1.0
    assert source.context.autoscaling_config.actor_pool_max_upscaling_delta == 1
    assert [actor for actor, _kwargs in calls] == [
        inference.VideoDownloadActor,
        inference.VideoPreparationActor,
        inference.VideoEmbeddingBatchActor,
    ]
    media = calls[0][1]
    cpu = calls[1][1]
    gpu = calls[2][1]
    assert media["resources"] == {"curator_io": 1.0}
    assert media["num_cpus"] == 0.25
    assert "num_gpus" not in media
    assert media["batch_size"] == 1
    assert media["max_restarts"] == cpu["max_restarts"] == gpu["max_restarts"] == 5
    assert media["max_task_retries"] == cpu["max_task_retries"] == gpu["max_task_retries"] == 5
    assert media["compute"].max_size == (float("inf") if media_concurrency == "auto" else media_concurrency)
    assert "runtime_env" not in media
    assert media["compute"].max_tasks_in_flight_per_actor == 1
    assert media["max_concurrency"] == 1
    assert "resources" not in cpu
    assert "num_gpus" not in cpu
    assert cpu["num_cpus"] == 2
    assert cpu["fn_constructor_kwargs"]["decode_cpus"] == 2
    assert cpu["memory"] == 3 * 1024**3
    assert cpu["batch_size"] == 1
    assert cpu["compute"].max_size == (float("inf") if decode_concurrency == "auto" else decode_concurrency)
    assert cpu["compute"].max_tasks_in_flight_per_actor == 1
    assert "legacy-transformers" in str(cpu["runtime_env"])
    assert "resources" not in gpu
    assert gpu["num_gpus"] == 1.0
    assert gpu["batch_size"] == 4
    assert gpu["compute"].max_size == expected_ceiling
    assert gpu["compute"].max_tasks_in_flight_per_actor == 1
    assert gpu["max_concurrency"] == 1


@pytest.mark.integration
def test_ray_actor_boundary_preserves_unsupported_rows(curator_io_cluster: None, tmp_path: Path) -> None:  # noqa: ARG001
    """Ray Data passes an over-limit row through three actors without media or a model."""
    model = resolve_model("cosmos_embed1_336p")
    source = _source(tmp_path / "absent.mp4", episode_id="long", duration_ns=120_000_000_000)
    rows = ray.data.from_arrow(pa.Table.from_pylist([source], schema=EPISODE_MAPPED_INPUT_SCHEMA))
    downloaded = rows.map_batches(
        inference.VideoDownloadActor,
        fn_constructor_kwargs={
            "camera_key": "front_wide",
            "max_episode_duration_s": 60.0,
            "storage_profile": "default",
            "media_attempts": 1,
        },
        batch_format="pyarrow",
        batch_size=1,
        compute=ActorPoolStrategy(min_size=1, max_tasks_in_flight_per_actor=1),
        num_cpus=0.25,
        max_concurrency=1,
        resources={environment.CURATOR_IO_RESOURCE_NAME: 1.0},
    )
    prepared = downloaded.map_batches(
        _CPUPreparationActor,
        fn_constructor_kwargs={"model": model, "camera_key": "front_wide"},
        batch_format="pyarrow",
        batch_size=1,
        compute=ActorPoolStrategy(min_size=1, max_size=1, max_tasks_in_flight_per_actor=1),
        num_cpus=1.0,
        memory=128 * 1024**2,
        max_concurrency=1,
    )
    embedded = prepared.map_batches(
        inference.VideoEmbeddingBatchActor,
        fn_constructor_kwargs={
            "model": model,
            "camera_key": "front_wide",
            "result_key": "default",
            "digest": embedding_contract_digest(model, camera_key="front_wide", result_key="default"),
            "max_episode_duration_s": 60.0,
        },
        batch_format="pyarrow",
        batch_size=1,
        compute=ActorPoolStrategy(min_size=1, max_size=1),
        num_cpus=1.0,
    )
    result = embedded.take_all()
    vector_name, metadata_name = result_field_names("default", model_key=model.key)
    assert len(result) == 1
    assert result[0]["episode_id"] == "long"
    assert result[0][vector_name] is None
    assert result[0][metadata_name]["status"] == "unsupported_duration"


@pytest.mark.integration
def test_ray_production_plan_preserves_episode_vectors(
    curator_io_cluster: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute the production three-stage plan on CPU, with distinct vectors and item outcomes."""
    del curator_io_cluster
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    _video(first, pixel_value=24)
    _video(second, pixel_value=160)
    sources = [
        _source(first, episode_id="a"),
        _source(tmp_path / "missing.mp4", episode_id="missing"),
        _source(second, episode_id="b"),
        _source(tmp_path / "unsupported.mp4", episode_id="long", duration_ns=120_000_000_000),
    ]
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "multimodal-embedding",
            "input": {"episodes_lance_uri": str(tmp_path / "episodes.lance")},
            "video": {"camera_key": "front_wide"},
            "execution": {
                "media_concurrency": 1,
                "decode_concurrency": 1,
                "inference_concurrency": 1,
                "decode_memory_gib": 0.125,
                "media_attempts": 1,
            },
        }
    )
    # Only the test removes GPU placement and isolated runtimes; production
    # composition, batching, IO admission, actor retries and row handling run unchanged.
    config = config.model_copy(update={"execution": config.execution.model_copy(update={"inference_gpus": 0.0})})
    monkeypatch.setattr(inference, "VideoPreparationActor", _CPUPreparationActor)
    monkeypatch.setattr(inference, "VideoEmbeddingBatchActor", _CPUEmbeddingActor)
    monkeypatch.setattr(inference, "PixiRuntimeEnv", lambda _environment: {})
    monkeypatch.setattr(inference, "ray_data_gpu_runtime_env", lambda _environment: {})
    rows = ray.data.from_arrow(pa.Table.from_pylist(sources, schema=EPISODE_MAPPED_INPUT_SCHEMA))
    result = inference.map_video_embeddings(rows, config, num_input_rows=len(sources)).take_all()
    vector_name, metadata_name = result_field_names("default", model_key=config.model.variant)
    by_id = {row["episode_id"]: row for row in result}
    assert len(result) == len(by_id) == len(sources)
    expected = np.eye(768, dtype=np.float32)
    np.testing.assert_array_equal(by_id["a"][vector_name], expected[0])
    np.testing.assert_array_equal(by_id["b"][vector_name], expected[1])
    assert by_id["a"][metadata_name]["status"] == by_id["b"][metadata_name]["status"] == "success"
    assert by_id["missing"][vector_name] is None
    assert by_id["missing"][metadata_name]["error_code"] == "media_object_error"
    assert by_id["long"][vector_name] is None
    assert by_id["long"][metadata_name]["status"] == "unsupported_duration"
