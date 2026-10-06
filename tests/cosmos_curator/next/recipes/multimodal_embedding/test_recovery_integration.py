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

"""Local Ray staging and Lance publication with a deterministic CPU encoder."""

import json
from collections.abc import Iterator
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import lance
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import ray

from cosmos_curator.core.sensors.sampling.grid import make_ts_grid
from cosmos_curator.core.sensors.types.types import TimestampSource
from cosmos_curator.core.utils import environment
from cosmos_curator.next.recipes.multimodal_embedding import inference, pipeline
from cosmos_curator.next.recipes.multimodal_embedding import publication as embedding_publication
from cosmos_curator.next.recipes.multimodal_embedding.config import (
    ResolvedMultimodalEmbeddingConfig,
    resolve_config_data,
)
from cosmos_curator.next.recipes.multimodal_embedding.contracts import (
    CameraEmbeddingPlan,
    EmbeddingRunPlan,
    EmbeddingTerminalResult,
    VideoEmbeddingModelSpec,
    build_terminal_result_row,
    embedding_contract_digest,
    resolve_model,
    result_field_names,
    result_fields,
    staged_result_schema,
)
from cosmos_curator.next.recipes.multimodal_embedding.episode_input import (
    CAMERA_WORK_ID,
    EPISODE_SOURCE_COLUMNS,
    camera_work_id,
    project_camera_work,
)
from cosmos_curator.next.recipes.multimodal_split.records import EPISODE_SCHEMA, EpisodeFacts, build_episode_record
from cosmos_curator.next.recipes.video_caption import publication
from cosmos_curator.next.recipes.video_caption.bindings import CaptionRunBindings
from cosmos_curator.next.recipes.video_caption.lance_state import CaptionAttempt, capture_attempt, ensure_caption_fields
from cosmos_curator.next.recipes.video_caption.workspace import (
    CaptionWorkspace,
    ensure_workspace,
    phase_a_completion_covers,
    record_phase_a_completion,
    resolve_workspace,
)
from cosmos_curator.next.utils.identity import canonical_digest
from tests.cosmos_curator.next.recipes.multimodal_embedding.test_inference import (
    _CPUEmbeddingActor,
    _CPUPreparationActor,
    _video,
)


def _test_run_plan(
    config: ResolvedMultimodalEmbeddingConfig,
    model: VideoEmbeddingModelSpec,
    keys: tuple[str, ...],
    pending: dict[int, tuple[str, ...]],
) -> EmbeddingRunPlan:
    return EmbeddingRunPlan(
        config.input.episodes_lance_uri,
        config.output.result_key,
        model,
        tuple(
            CameraEmbeddingPlan.resolve(model, camera_key=key, result_key=config.result_key_for_camera(key))
            for key in keys
        ),
        pending,
    )


def _test_camera_bindings(
    config: ResolvedMultimodalEmbeddingConfig, model: VideoEmbeddingModelSpec, *, camera_key: str
) -> CaptionRunBindings:
    camera = CameraEmbeddingPlan.resolve(
        model, camera_key=camera_key, result_key=config.result_key_for_camera(camera_key)
    )
    return embedding_publication.camera_bindings(camera, model, uri=config.input.episodes_lance_uri)


def _episode(index: int) -> dict[str, object]:
    start_ns = 0
    end_ns = 1_000_000_000
    _, _, timestamps = make_ts_grid(start_ns, exclusive_end_ns=end_ns, sample_rate_hz=30)
    facts = EpisodeFacts(
        episode_id=f"episode-{index}",
        split_clip_id=f"split-{index}",
        source_id=f"source-{index}",
        source_session_id=f"session-{index}",
        source_session_location=f"/input/session-{index}",
        start_ns=start_ns,
        exclusive_end_ns=end_ns,
        sampling_fps=30,
        row_count=len(timestamps),
        timestamp_source=TimestampSource.SENSOR,
        split_contract_digest="split-contract",
    )
    cameras = [
        {
            "camera_key": "front_wide",
            "source_media_location": f"/source/{index}.mp4",
            "media_location": f"/output/{index}.mp4",
            "media_size_bytes": 10,
            "media_sha256": str(index + 1) * 64,
            "timeline_location": f"/output/{index}.timestamps.npy",
            "timeline_sha256": "a" * 64,
            "sensor_timestamps_location": f"/output/{index}.sensor.npy",
            "sensor_timestamps_sha256": "b" * 64,
        }
    ]
    return build_episode_record(facts, cameras, [])


def _multi_episode(index: int, video: Path) -> dict[str, object]:
    row = _episode(index)
    payload = video.read_bytes()
    cameras = row["camera_media"]
    front = {
        **cameras[0],
        "media_location": str(video),
        "media_size_bytes": len(payload),
        "media_sha256": sha256(payload).hexdigest(),
    }
    row["camera_media"] = [front, {**front, "camera_key": "rear"}]
    return row


def _multi_cpu_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute production Ray topology with CPU stand-ins only for model/processor assets."""
    monkeypatch.setattr(pipeline, "validate_cosmos_model_directory", lambda model: model.runtime_model_dir)
    monkeypatch.setattr(inference, "VideoPreparationActor", _CPUPreparationActor)
    monkeypatch.setattr(inference, "VideoEmbeddingBatchActor", _CPUEmbeddingActor)
    monkeypatch.setattr(inference, "PixiRuntimeEnv", lambda _name: {})
    monkeypatch.setattr(inference, "ray_data_gpu_runtime_env", lambda _name: {})
    original = ray.data.Dataset.map_batches

    def cpu_batches(self: ray.data.Dataset, fn: object, **kwargs: object) -> ray.data.Dataset:
        if fn in {_CPUEmbeddingActor, _UnexpectedEmbeddingActor}:
            kwargs["num_gpus"] = 0
        if fn in {_CPUPreparationActor, _CPUEmbeddingActor, _UnexpectedEmbeddingActor}:
            kwargs["memory"] = 128 * 1024**2
        return original(self, fn, **kwargs)

    monkeypatch.setattr(ray.data.Dataset, "map_batches", cpu_batches)


def _multi_config(uri: str, *, selection: object = "all") -> ResolvedMultimodalEmbeddingConfig:
    return resolve_config_data(
        {
            "schema_version": 1,
            "kind": "multimodal-embedding",
            "input": {"episodes_lance_uri": uri},
            "video": {"camera_keys": selection},
            "execution": {
                "media_concurrency": 1,
                "decode_concurrency": 1,
                "inference_concurrency": 1,
                "inference_batch_size": 2,
                "lance_read_batch_size": 1,
                "parquet_rows_per_file": 1,
            },
        }
    )


@pytest.mark.integration
def test_multi_camera_mixed_results_keep_camera_and_episode_bindings(
    tmp_path: Path, curator_io_cluster: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Different vectors and a missing rear view survive wide publication and receipt hashing."""
    del curator_io_cluster
    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    _video(first, pixel_value=24)
    _video(second, pixel_value=160)
    uri = str(tmp_path / "episodes.lance")
    for index, video in enumerate((first, first, second)):
        episode = _multi_episode(index, video)
        if index == 1:
            episode["camera_media"][1]["media_location"] = str(tmp_path / "missing.mp4")
        lance.write_dataset(
            pa.Table.from_pylist([episode], schema=EPISODE_SCHEMA),
            uri,
            mode="create" if index == 0 else "append",
            data_storage_version="2.2",
        )
    _multi_cpu_runtime(monkeypatch)
    summary = pipeline.run_config(_multi_config(uri))
    assert summary["published_fragments"] == 3
    assert summary["cameras"]["front_wide"]["terminal_counts"]["success"] == 3
    assert summary["cameras"]["rear"]["terminal_counts"] == {
        "success": 2,
        "error": 1,
        "unsupported_duration": 0,
    }
    rows = {row["episode_id"]: row for row in lance.dataset(uri).to_table().to_pylist()}
    expected = np.eye(768, dtype=np.float32)
    for key, camera in summary["cameras"].items():
        for index, vector_index in ((0, 0), (1, 0), (2, 1)):
            row = rows[f"episode-{index}"]
            vector, metadata = row[camera["embedding_field"]], row[camera["metadata_field"]]
            assert metadata["episode_id"] == row["episode_id"]
            assert metadata["camera_key"] == key
            if key == "rear" and index == 1:
                assert vector is None
                assert metadata["error_code"] == "media_object_error"
            else:
                np.testing.assert_array_equal(vector, expected[vector_index])
    version = lance.dataset(uri).version
    rerun = pipeline.run_config(_multi_config(uri))
    assert rerun["published_fragments"] == 0
    assert lance.dataset(uri).version == version


@pytest.mark.parametrize(
    "video",
    [None, {}, {"camera_keys": "all"}, {"camera_keys": ["front_wide"]}],
)
def test_empty_table_is_noop_for_every_camera_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, video: dict[str, object] | None
) -> None:
    """No selector starts Ray or model work on an empty canonical table."""
    uri = str(tmp_path / "episodes.lance")
    dataset = lance.write_dataset(pa.Table.from_pylist([], schema=EPISODE_SCHEMA), uri, data_storage_version="2.2")
    original_version = int(dataset.version)
    config_payload = {"schema_version": 1, "kind": "multimodal-embedding", "input": {"episodes_lance_uri": uri}}
    if video is not None:
        config_payload["video"] = video
    config = resolve_config_data(config_payload)

    def unexpected_work(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Empty input must not start Ray, staging, or model work")

    for name in (
        "ensure_ray_initialized",
        "validate_cosmos_model_directory",
        "ensure_workspace",
        "run_inference_phase",
    ):
        monkeypatch.setattr(pipeline, name, unexpected_work)
    summary = pipeline.run_config(config)
    assert summary["selected_fragments"] == summary["pending_fragments"] == summary["published_fragments"] == 0
    assert summary["cleanup_complete"] is True
    current = lance.dataset(uri)
    assert current.count_rows() == 0
    if config.video.camera_keys == "all":
        assert summary["camera_keys"] == []
        assert summary["cameras"] == {}
        assert summary["episodes_lance_version"] == original_version == int(current.version)
        assert current.schema.equals(EPISODE_SCHEMA, check_metadata=True)


@pytest.mark.integration
def test_explicit_untagged_namespace_uses_pair_pipeline_and_reruns_without_work(
    tmp_path: Path, curator_io_cluster: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The naming opt-out uses the current camera-pair contract and has no scalar execution path."""
    del curator_io_cluster
    video = tmp_path / "camera.mp4"
    _video(video)
    uri = str(tmp_path / "episodes.lance")
    lance.write_dataset(
        pa.Table.from_pylist([_multi_episode(0, video)], schema=EPISODE_SCHEMA), uri, data_storage_version="2.2"
    )
    _multi_cpu_runtime(monkeypatch)
    payload = _multi_config(uri, selection=["rear"]).model_dump(mode="json")
    payload["output"]["tag_camera_key"] = False
    config = resolve_config_data(payload)
    summary = pipeline.run_config(config)
    camera = summary["cameras"]["rear"]
    assert camera["effective_result_key"] == "default"
    assert camera["tag_camera_key"] is False
    assert camera["terminal_counts"]["success"] == 1
    metadata = lance.dataset(uri).to_table(columns=[camera["metadata_field"]]).to_pylist()[0]
    assert metadata[camera["metadata_field"]]["camera_key"] == "rear"
    version = lance.dataset(uri).version

    def unexpected_inference(*_args: object, **_kwargs: object) -> None:
        pytest.fail("A completed camera-pair run must not repeat inference")

    monkeypatch.setattr(pipeline, "run_inference_phase", unexpected_inference)
    assert pipeline.run_config(config)["published_fragments"] == 0
    assert lance.dataset(uri).version == version


@pytest.mark.integration
def test_camera_lists_publish_once_and_reuse_completed_cameras(  # noqa: PLR0915 -- one-camera, append-camera and no-op sequence
    tmp_path: Path, curator_io_cluster: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two real camera downloads share the pipeline; adding a camera retains published front vectors."""
    del curator_io_cluster
    video = tmp_path / "camera.mp4"
    _video(video)
    uri = str(tmp_path / "episodes.lance")
    for index in range(2):
        lance.write_dataset(
            pa.Table.from_pylist([_multi_episode(index, video)], schema=EPISODE_SCHEMA),
            uri,
            mode="create" if index == 0 else "append",
            data_storage_version="2.2",
        )
    # Old untagged fields are unrelated data, never a compatibility shortcut.
    model = resolve_model("cosmos_embed1_336p")
    old_fields = result_fields(
        "default",
        model_key=model.key,
        digest=embedding_contract_digest(model, camera_key="front_wide", result_key="default"),
    )
    lance.dataset(uri).add_columns(old_fields)
    old_before = lance.dataset(uri).to_table(columns=old_fields.names)
    _multi_cpu_runtime(monkeypatch)
    multi = _multi_config(uri)
    config_payload = multi.model_dump(mode="json")
    config_payload["video"] = {"camera_keys": ["front_wide"]}
    front = pipeline.run_config(resolve_config_data(config_payload))["cameras"]["front_wide"]
    assert front["base_result_key"] == "default"
    assert front["effective_result_key"] == "default_front_wide"
    assert front["embedding_field"] not in old_fields.names
    assert lance.dataset(uri).to_table(columns=old_fields.names).equals(old_before)
    previous = lance.dataset(uri).to_table(columns=[front["embedding_field"], front["metadata_field"]])
    workloads = []
    original_inference = pipeline.run_inference_phase

    def observe(*args: object, **kwargs: object) -> None:
        workloads.append(kwargs["pending_cameras"])
        original_inference(*args, **kwargs)

    monkeypatch.setattr(pipeline, "run_inference_phase", observe)
    preflight = Mock(wraps=pipeline.validate_episode_cameras)
    monkeypatch.setattr(pipeline, "validate_episode_cameras", preflight)
    capture = Mock(wraps=pipeline.capture_attempt)
    monkeypatch.setattr(pipeline, "capture_attempt", capture)
    summary = pipeline.run_config(multi)
    assert preflight.call_count == 2
    assert capture.call_count == 2, "Classify once per camera; reuse those results for the aggregate attempt"
    assert summary["pending_fragments"] == 2
    assert summary["skipped_complete_fragments"] == 0
    assert all(keys == ("rear",) for keys in workloads[0].values())
    assert summary["cameras"]["front_wide"]["embedding_field"] == front["embedding_field"]
    current = lance.dataset(uri)
    assert current.to_table(columns=previous.column_names).equals(previous)
    transaction = current.read_transaction(int(current.version))
    assert isinstance(transaction.operation, lance.LanceOperation.Update)
    assert len(transaction.operation.updated_fragments) == 2
    assert summary["published_fragments"] == 2
    for item in summary["cameras"].values():
        assert item["terminal_counts"] == {"success": 2, "error": 0, "unsupported_duration": 0}
    rear = summary["cameras"]["rear"]
    assert rear["base_result_key"] == "default"
    assert rear["effective_result_key"] == "default_rear"
    assert pipeline.run_config(resolve_config_data(config_payload))["published_fragments"] == 0
    version = int(current.version)
    reordered = _multi_config(uri, selection=["rear", "front_wide"])
    preflight.reset_mock()
    capture.reset_mock()
    rerun = pipeline.run_config(reordered)
    assert preflight.call_count == 2
    assert capture.call_count == 2
    assert rerun["pending_fragments"] == 0
    assert rerun["skipped_complete_fragments"] == 2
    assert rerun["published_fragments"] == 0
    assert int(lance.dataset(uri).version) == version
    assert len(workloads) == 1


@pytest.mark.integration
@pytest.mark.parametrize("failure", ["before_commit", "lost_response"])
def test_multi_camera_atomic_recovery_never_exposes_partial_cameras(
    tmp_path: Path,
    curator_io_cluster: None,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    """All camera columns are published together; retries reuse pair checkpoints."""
    del curator_io_cluster
    video = tmp_path / "camera.mp4"
    _video(video)
    uri = str(tmp_path / "episodes.lance")
    lance.write_dataset(
        pa.Table.from_pylist([_multi_episode(0, video)], schema=EPISODE_SCHEMA), uri, data_storage_version="2.2"
    )
    _multi_cpu_runtime(monkeypatch)
    config = _multi_config(uri)
    original_commit = embedding_publication._commit_atomic_update

    def failing_commit(*args: object, **kwargs: object) -> None:
        if failure == "lost_response":
            original_commit(*args, **kwargs)
        msg = "simulated multi-camera commit response failure"
        raise OSError(msg)

    monkeypatch.setattr(embedding_publication, "_commit_atomic_update", failing_commit)
    if failure == "before_commit":
        with pytest.raises(publication.PublicationError, match="Could not reconcile atomic"):
            pipeline.run_config(config)
        capfd.readouterr()
        table = lance.dataset(uri).to_table()
        vector_fields = [
            name
            for name in table.column_names
            if name.startswith("embedding_video_") and not name.endswith("_metadata")
        ]
        assert len(vector_fields) == 2
        assert all(table[name].null_count == table.num_rows for name in vector_fields)
    else:
        assert pipeline.run_config(config)["already_committed_fragments"] == 1
    monkeypatch.setattr(embedding_publication, "_commit_atomic_update", original_commit)
    monkeypatch.setattr(
        pipeline,
        "run_inference_phase",
        lambda *_args, **_kwargs: pytest.fail("must reuse completed camera inference"),
    )
    summary = pipeline.run_config(config)
    assert all(value["terminal_counts"]["success"] == 1 for value in summary["cameras"].values())


@pytest.mark.integration
@pytest.mark.parametrize(
    ("selection", "winner", "race_at"),
    [
        (["front_wide", "rear"], ["front_wide"], "preparation"),
        (["front_wide", "rear"], ["front_wide"], "commit"),
        (["front_wide"], ["rear"], "preparation"),
        (["front_wide"], ["rear"], "commit"),
        (["front_wide", "rear"], ["front_wide", "rear"], "commit"),
    ],
)
def test_camera_publication_allows_disjoint_writers_and_rejects_overlap(  # noqa: C901, PLR0915 -- explicit competing-writer scenarios
    tmp_path: Path,
    curator_io_cluster: None,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    selection: list[str],
    winner: list[str],
    race_at: str,
) -> None:
    """Disjoint writers coexist; overlapping fields fail without exposing remaining results."""
    del curator_io_cluster
    video = tmp_path / "camera.mp4"
    _video(video)
    uri = str(tmp_path / "episodes.lance")
    for index in range(2):
        lance.write_dataset(
            pa.Table.from_pylist([_multi_episode(index, video)], schema=EPISODE_SCHEMA),
            uri,
            mode="create" if index == 0 else "append",
            data_storage_version="2.2",
        )
    _multi_cpu_runtime(monkeypatch)
    publish = pipeline.publish_atomic_staged_results
    commit = embedding_publication._commit_atomic_update
    fired = False
    winner_columns: dict[str, list[object]] = {}

    def publish_winner() -> None:
        nonlocal fired
        fired = True
        winner_config = _multi_config(uri, selection=winner)
        winner_config = winner_config.model_copy(
            update={
                "output": winner_config.output.model_copy(update={"staging_root_uri": str(tmp_path / "winner-staging")})
            }
        )
        summary = pipeline.run_config(winner_config)
        # Distinct vectors prove that publication never replaces the other writer
        # with this run's axis-zero staged projections.
        for camera in summary["cameras"].values():
            field = camera["embedding_field"]
            for fragment_id in range(2):
                vector = [0.0] * 768
                vector[2] = 1.0
                _commit_episode_columns(uri, fragment_id, [{"episode_id": f"episode-{fragment_id}", field: vector}])
            for name in (field, camera["metadata_field"]):
                winner_columns[name] = lance.dataset(uri).to_table(columns=[name]).column(0).to_pylist()

    def competing_publish(*args: object, **kwargs: object) -> object:
        if not fired and race_at == "preparation":
            publish_winner()
        return publish(*args, **kwargs)

    def competing_commit(*args: object, **kwargs: object) -> None:
        if not fired and race_at == "commit":
            publish_winner()
            # Do not rely on Lance's automatic conflict handling: force the
            # loser's stale transaction back through our reconciliation path.
            msg = "camera writer won the commit race"
            raise RuntimeError(msg)
        commit(*args, **kwargs)

    monkeypatch.setattr(pipeline, "publish_atomic_staged_results", competing_publish)
    monkeypatch.setattr(embedding_publication, "_commit_atomic_update", competing_commit)
    if set(selection) & set(winner):
        with pytest.raises(publication.PublicationError, match="one active writer per result-field set"):
            pipeline.run_config(_multi_config(uri, selection=selection))
        assert fired
        latest = lance.dataset(uri)
        for field, values in winner_columns.items():
            assert latest.to_table(columns=[field]).column(0).to_pylist() == values
        for camera_key in set(selection) - set(winner):
            field = result_field_names(f"default_{camera_key}", model_key="cosmos_embed1_336p")[0]
            assert latest.to_table(columns=[field]).column(0).null_count == 2
        capfd.readouterr()
        return
    summary = pipeline.run_config(_multi_config(uri, selection=selection))
    assert fired
    latest = lance.dataset(uri)
    for field, values in winner_columns.items():
        assert latest.to_table(columns=[field]).column(0).to_pylist() == values
    for camera in summary["cameras"].values():
        assert camera["terminal_counts"]["success"] == 2
    assert summary["published_fragments"] == 2
    transaction = latest.read_transaction(latest.version)
    modified = set(transaction.operation.fields_modified)
    winning_ids = publication.leaf_field_ids(latest, tuple(winner_columns))
    assert not modified & winning_ids
    monkeypatch.setattr(
        pipeline,
        "run_inference_phase",
        lambda *_args, **_kwargs: pytest.fail("published camera results must remain reusable"),
    )
    version = latest.version
    assert pipeline.run_config(_multi_config(uri, selection=selection))["published_fragments"] == 0
    assert lance.dataset(uri).version == version
    capfd.readouterr()


@pytest.mark.integration
def test_concurrent_camera_publication_rejects_changed_completed_camera(
    tmp_path: Path,
    curator_io_cluster: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed camera is immutable even while another camera is being embedded."""
    del curator_io_cluster
    video = tmp_path / "camera.mp4"
    _video(video)
    uri = str(tmp_path / "episodes.lance")
    lance.write_dataset(
        pa.Table.from_pylist([_multi_episode(0, video)], schema=EPISODE_SCHEMA),
        uri,
        data_storage_version="2.2",
    )
    _multi_cpu_runtime(monkeypatch)
    front = pipeline.run_config(_multi_config(uri, selection=["front_wide"]))
    field = front["cameras"]["front_wide"]["embedding_field"]
    publish = pipeline.publish_atomic_staged_results

    def mutate(*args: object, **kwargs: object) -> object:
        vector = [0.0] * 768
        vector[2] = 1.0
        _commit_episode_columns(uri, 0, [{"episode_id": "episode-0", field: vector}])
        return publish(*args, **kwargs)

    monkeypatch.setattr(pipeline, "publish_atomic_staged_results", mutate)
    with pytest.raises(publication.PublicationError, match="Selected embedding fields changed"):
        pipeline.run_config(_multi_config(uri))
    rear, _ = result_field_names("default_rear", model_key="cosmos_embed1_336p")
    assert lance.dataset(uri).to_table(columns=[rear]).column(0).null_count == 1


@pytest.mark.integration
@pytest.mark.parametrize("completion_marker", [True, False])
def test_multi_camera_recovery_after_another_selection_publishes_a_camera(
    tmp_path: Path,
    curator_io_cluster: None,
    monkeypatch: pytest.MonkeyPatch,
    *,
    completion_marker: bool,
) -> None:
    """Old pair staging remains reusable when a subset run publishes one of its cameras."""
    del curator_io_cluster
    video = tmp_path / "camera.mp4"
    _video(video)
    uri = str(tmp_path / "episodes.lance")
    lance.write_dataset(
        pa.Table.from_pylist([_multi_episode(0, video)], schema=EPISODE_SCHEMA), uri, data_storage_version="2.2"
    )
    _multi_cpu_runtime(monkeypatch)
    config = _multi_config(uri, selection=["front_wide", "rear"])
    # Leave a CPU for the source read alongside Ray's checkpoint-filter actor on the four-CPU fixture.
    config = config.model_copy(update={"execution": config.execution.model_copy(update={"inference_cpus": 0.5})})
    original_publish = pipeline.publish_atomic_staged_results

    def interrupted_publish(*_args: object, **_kwargs: object) -> None:
        msg = "interrupted before camera publication"
        raise InterruptedError(msg)

    monkeypatch.setattr(pipeline, "publish_atomic_staged_results", interrupted_publish)
    with pytest.raises(InterruptedError, match="before camera publication"):
        pipeline.run_config(config)
    markers = list(Path(config.output.staging_root_uri).rglob("phase-a-complete.json"))
    assert len(markers) == 1
    if not completion_marker:
        markers[0].unlink()
    monkeypatch.setattr(pipeline, "publish_atomic_staged_results", original_publish)
    front = pipeline.run_config(_multi_config(uri, selection=["front_wide"]))
    front_names = [front["cameras"]["front_wide"][name] for name in ("embedding_field", "metadata_field")]
    published_front = lance.dataset(uri).to_table(columns=front_names)
    before_resume = int(lance.dataset(uri).version)
    if completion_marker:
        monkeypatch.setattr(
            pipeline,
            "run_inference_phase",
            lambda *_args, **_kwargs: pytest.fail("must reuse completed camera staging"),
        )
    else:
        # The native row checkpoint must suppress all expensive work even without the completion marker.
        monkeypatch.setattr(
            inference,
            "VideoEmbeddingBatchActor",
            _UnexpectedEmbeddingActor,
        )
    summary = pipeline.run_config(config)
    current = lance.dataset(uri)
    assert int(current.version) == before_resume + 1
    assert current.to_table(columns=front_names).equals(published_front)
    assert summary["published_fragments"] == 1
    assert all(item["terminal_counts"]["success"] == 1 for item in summary["cameras"].values())
    assert not markers[0].parent.exists()


class _UnexpectedEmbeddingActor:
    """Fail if recovery sends any already checkpointed camera back through inference."""

    def __init__(self, **_kwargs: object) -> None:
        pass

    def __call__(self, _batch: pa.Table) -> pa.Table:
        msg = "checkpointed cameras must not repeat inference"
        raise AssertionError(msg)


def _tracked_camera_batch(
    batch: pa.Table, *, camera_results: dict[str, tuple[str, str]], calls_dir: str, fail_id: str | None
) -> pa.Table:
    model = resolve_model("cosmos_embed1_336p")
    rows = []
    for source in batch.to_pylist():
        identity = source[CAMERA_WORK_ID]
        Path(calls_dir, f"{identity}-{uuid4().hex}").touch()
        if identity == fail_id:
            msg = "simulated camera interruption"
            raise RuntimeError(msg)
        result_key, digest = camera_results[source["camera_key"]]
        vector_name, metadata_name = result_field_names(result_key, model_key=model.key)
        result = build_terminal_result_row(
            source,
            model=model,
            result_key=result_key,
            digest=digest,
            result=EmbeddingTerminalResult(
                "success", [1.0] + [0.0] * (model.vector_dimension - 1), 60.0, None, None, 1_000_000_000
            ),
        )
        rows.append({**source, "embedding": result[vector_name], "embedding_metadata": result[metadata_name]})
    return pa.Table.from_pylist(rows, schema=inference.camera_staged_schema(model))


@pytest.mark.parametrize(
    "corruption", ["missing_pair", "unexpected_camera", "hash_mismatch", "unpublished_camera", "stale_hash_mismatch"]
)
def test_multi_camera_assembly_rejects_incomplete_or_wrong_camera_bindings(tmp_path: Path, corruption: str) -> None:
    """A wide episode update cannot hide a missing camera or wrong media provenance."""
    video = tmp_path / "camera.mp4"
    _video(video)
    uri = str(tmp_path / "episodes.lance")
    dataset = lance.write_dataset(
        pa.Table.from_pylist([_multi_episode(0, video)], schema=EPISODE_SCHEMA), uri, data_storage_version="2.2"
    )
    config = _multi_config(uri)
    model = resolve_model(config.model.variant)
    result_keys = {key: config.result_key_for_camera(key) for key in ("front_wide", "rear")}
    cameras = {key: _test_camera_bindings(config, model, camera_key=key) for key in result_keys}
    for binding in cameras.values():
        dataset = ensure_caption_fields(uri, storage_options=None, bindings=binding, attempts=1)
    fragment = dataset.get_fragments()[0]
    source = fragment.scanner(columns=list(EPISODE_SOURCE_COLUMNS), with_row_address=True).to_table()
    work = project_camera_work(source, camera_keys=("front_wide", "rear"))
    calls = tmp_path / "calls"
    calls.mkdir()
    staged = _tracked_camera_batch(
        work,
        camera_results={key: (result_keys[key], binding.result.digest) for key, binding in cameras.items()},
        calls_dir=str(calls),
        fail_id=None,
    )
    rows = staged.to_pylist()
    pending = tuple(result_keys)
    if corruption == "missing_pair":
        rows.pop()
    elif corruption == "unexpected_camera":
        rows[-1]["camera_key"] = "another_camera"
    elif corruption == "hash_mismatch":
        rows[-1]["media_sha256"] = "bad_hash"
    else:
        pending = ("rear",)
        if corruption == "stale_hash_mismatch":
            rows[0]["media_sha256"] = "bad_hash"
    with pytest.raises((TypeError, ValueError), match=r"cover every|unexpected|media binding|requires metadata"):
        embedding_publication._prepare_camera_rows(
            pa.Table.from_pylist(rows, schema=staged.schema),
            fragment,
            cameras=tuple(
                CameraEmbeddingPlan.resolve(model, camera_key=key, result_key=value)
                for key, value in result_keys.items()
            ),
            model=model,
            pending_cameras={int(fragment.fragment_id): pending},
        )
    table = dataset.to_table()
    assert all(table[binding.result.caption_field_name].null_count == table.num_rows for binding in cameras.values())


@pytest.mark.integration
def test_multi_camera_mid_staging_restart_preserves_pair_checkpoints(
    tmp_path: Path, curator_io_cluster: None, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """A checkpointed camera is skipped without suppressing another camera from the same episode."""
    del curator_io_cluster
    video = tmp_path / "camera.mp4"
    _video(video)
    uri = str(tmp_path / "episodes.lance")
    for index in range(4):
        lance.write_dataset(
            pa.Table.from_pylist([_multi_episode(index, video)], schema=EPISODE_SCHEMA),
            uri,
            mode="create" if index == 0 else "append",
            data_storage_version="2.2",
        )
    config = _multi_config(uri)
    calls = tmp_path / "calls"
    calls.mkdir()
    fail_id = camera_work_id("episode-3", "rear")

    def fake_map(
        source: ray.data.Dataset, _config: object, *, camera_results: dict[str, tuple[str, str]], **_kwargs: object
    ) -> ray.data.Dataset:
        return source.map_batches(
            _tracked_camera_batch,
            batch_format="pyarrow",
            batch_size=1,
            fn_kwargs={"camera_results": camera_results, "calls_dir": str(calls), "fail_id": fail_id},
        )

    monkeypatch.setattr(pipeline, "validate_cosmos_model_directory", lambda model: model.runtime_model_dir)
    monkeypatch.setattr(inference, "map_video_embeddings", fake_map)
    with pytest.raises(Exception, match="simulated camera interruption"):
        pipeline.run_config(config)
    capfd.readouterr()
    model = resolve_model(config.model.variant)
    result_keys = {key: config.result_key_for_camera(key) for key in ("front_wide", "rear")}
    bindings = embedding_publication.publication_bindings(_test_run_plan(config, model, tuple(result_keys), {}))
    workspace = resolve_workspace(
        config.output.staging_root_uri,
        bindings.result.caption_field_name,
        bindings.result.digest,
        storage_profile="default",
        policy_namespace=canonical_digest(config.selection.model_dump(mode="json")),
    )
    checkpointed = {
        identity
        for path in Path(workspace.checkpoints_uri).glob("*.parquet")
        for identity in pq.read_table(path, columns=[CAMERA_WORK_ID])[CAMERA_WORK_ID].to_pylist()
    }
    assert checkpointed, "at least one camera must have reached its durable row checkpoint"
    counts = {identity: len(list(calls.glob(f"{identity}-*"))) for identity in checkpointed}
    fail_id = None
    summary = pipeline.run_config(config)
    assert all(item["terminal_counts"]["success"] == 4 for item in summary["cameras"].values())
    assert all(len(list(calls.glob(f"{identity}-*"))) == count for identity, count in counts.items())


def _fake_batch(
    batch: pa.Table,
    *,
    model_key: str,
    camera_key: str,
    result_key: str,
    max_input_rows: int | None = None,
    error_id: str | None = None,
) -> pa.Table:
    if max_input_rows is not None:
        assert batch.num_rows <= max_input_rows, "Large source fragment was not split before inference"
    model = resolve_model(model_key)
    digest = embedding_contract_digest(model, camera_key=camera_key, result_key=result_key)
    rows = []
    for source in batch.to_pylist():
        vector = np.zeros(model.vector_dimension, dtype=np.float32)
        vector[int(source["episode_id"].removeprefix("episode-")) % model.vector_dimension] = 1.0
        is_error = source["episode_id"] == error_id
        result = EmbeddingTerminalResult(
            status="error" if is_error else "success",
            vector=None if is_error else vector,
            max_episode_duration_s=60.0,
            error_code="media_object_error" if is_error else None,
            error_message="missing media" if is_error else None,
            media_duration_ns=1_000_000_000,
        )
        rows.append(
            {
                **source,
                **build_terminal_result_row(source, model=model, result_key=result_key, digest=digest, result=result),
            }
        )
    if CAMERA_WORK_ID in batch.column_names:
        vector_name, metadata_name = result_field_names(result_key, model_key=model_key)
        rows = [
            {
                **{name: row[name] for name in inference.CAMERA_MAPPED_INPUT_SCHEMA.names},
                "embedding": row[vector_name],
                "embedding_metadata": row[metadata_name],
            }
            for row in rows
        ]
        return pa.Table.from_pylist(rows, schema=inference.camera_staged_schema(model))
    return pa.Table.from_pylist(rows, schema=staged_result_schema(result_key, model_key=model_key, digest=digest))


def _fake_error_batch(batch: pa.Table, *, model_key: str, camera_key: str, result_key: str) -> pa.Table:
    model = resolve_model(model_key)
    digest = embedding_contract_digest(model, camera_key=camera_key, result_key=result_key)
    rows = []
    for source in batch.to_pylist():
        result = EmbeddingTerminalResult(
            status="error",
            vector=None,
            max_episode_duration_s=60.0,
            error_code="media_object_error",
            error_message="media absent during replay",
        )
        rows.append(
            {
                **source,
                **build_terminal_result_row(source, model=model, result_key=result_key, digest=digest, result=result),
            }
        )
    if CAMERA_WORK_ID in batch.column_names:
        vector_name, metadata_name = result_field_names(result_key, model_key=model_key)
        rows = [
            {
                **{name: row[name] for name in inference.CAMERA_MAPPED_INPUT_SCHEMA.names},
                "embedding": row[vector_name],
                "embedding_metadata": row[metadata_name],
            }
            for row in rows
        ]
        return pa.Table.from_pylist(rows, schema=inference.camera_staged_schema(model))
    return pa.Table.from_pylist(rows, schema=staged_result_schema(result_key, model_key=model_key, digest=digest))


def test_embedding_field_registration_and_pending_classification(tmp_path: Path) -> None:
    """The real episode schema can acquire a fixed-size vector field before Ray starts."""
    uri = str(tmp_path / "episodes.lance")
    lance.write_dataset(pa.Table.from_pylist([_episode(0)], schema=EPISODE_SCHEMA), uri, data_storage_version="2.2")
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "multimodal-embedding",
            "input": {"episodes_lance_uri": uri},
            "video": {"camera_keys": ["front_wide"]},
        }
    )
    bindings = _test_camera_bindings(config, resolve_model(config.model.variant), camera_key="front_wide")
    ensure_caption_fields(uri, storage_options=None, bindings=bindings, attempts=1)
    registered = lance.dataset(uri)
    assert capture_attempt(registered, bindings=bindings).pending_fragment_ids
    vector_field, metadata_field = result_field_names("default_front_wide", model_key=config.model.variant)
    assert registered.to_table(columns=[vector_field, metadata_field]).to_pylist() == [
        {vector_field: None, metadata_field: None}
    ]


def test_recipe_rejects_existing_incompatible_result_fields_before_inference(tmp_path: Path) -> None:
    """The runner refuses a reused result key with different field metadata."""
    uri = str(tmp_path / "episodes.lance")
    lance.write_dataset(pa.Table.from_pylist([_episode(0)], schema=EPISODE_SCHEMA), uri, data_storage_version="2.2")
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "multimodal-embedding",
            "input": {"episodes_lance_uri": uri},
            "video": {"camera_keys": ["front_wide"]},
        }
    )
    model = resolve_model(config.model.variant)
    digest = embedding_contract_digest(
        model, camera_key="front_wide", result_key=config.result_key_for_camera("front_wide")
    )
    fields = result_fields(config.result_key_for_camera("front_wide"), model_key=model.key, digest=digest)
    incompatible = pa.schema(
        [
            field.with_metadata({**(field.metadata or {}), b"cosmos_curator.contract_digest": b"other-digest"})
            for field in fields
        ]
    )
    lance.dataset(uri).add_columns(incompatible)
    before = lance.dataset(uri)
    with pytest.raises(ValueError, match="incompatible"):
        pipeline.run_config(config)
    after = lance.dataset(uri)
    assert after.version == before.version
    assert after.to_table(columns=list(incompatible.names)).to_pylist() == [dict.fromkeys(incompatible.names)]


@pytest.mark.parametrize("camera_mode", ["list", "all"])
@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("invalid", ["duplicate_id", "record_version", "media_version"])
def test_append_between_preflight_and_capture_is_validated(
    invalid: str, camera_mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, complete: bool
) -> None:
    """The selected snapshot includes appends, and IDs are checked against completed rows too."""
    uri = str(tmp_path / "episodes.lance")
    lance.write_dataset(pa.Table.from_pylist([_episode(0)], schema=EPISODE_SCHEMA), uri, data_storage_version="2.2")
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "multimodal-embedding",
            "input": {"episodes_lance_uri": uri},
            "video": {"camera_keys": ["front_wide"]},
        }
    )
    config = _multi_config(uri, selection=["front_wide"] if camera_mode == "list" else "all")
    bindings = _test_camera_bindings(config, resolve_model(config.model.variant), camera_key="front_wide")
    if complete:
        registered = ensure_caption_fields(uri, storage_options=None, bindings=bindings, attempts=1)
        source = registered.scanner(columns=list(bindings.input.source_columns), with_row_address=True).to_table()
        staged = _fake_batch(
            bindings.input.project_batch(source),
            model_key=config.model.variant,
            camera_key="front_wide",
            result_key=config.result_key_for_camera("front_wide"),
        )
        _commit_episode_columns(uri, 0, staged.select(["episode_id", *bindings.result.fields.names]).to_pylist())
    original_ensure = pipeline.ensure_caption_fields

    def append_after_registration(*args: object, **kwargs: object) -> lance.LanceDataset:
        registered = original_ensure(*args, **kwargs)  # type: ignore[arg-type]
        appended = _episode(0 if invalid == "duplicate_id" else 1)
        if invalid == "record_version":
            appended["record_schema_version"] = -1
        elif invalid == "media_version":
            appended["media_contract_version"] = -1
        lance.write_dataset(pa.Table.from_pylist([appended], schema=EPISODE_SCHEMA), uri, mode="append")
        return registered

    capture = Mock(wraps=pipeline.capture_attempt)
    ray_init = Mock()
    run_inference = Mock()
    monkeypatch.setattr(pipeline, "ensure_caption_fields", append_after_registration)
    monkeypatch.setattr(pipeline, "capture_attempt", capture)
    monkeypatch.setattr(pipeline, "ensure_ray_initialized", ray_init)
    monkeypatch.setattr(pipeline, "run_inference_phase", run_inference)
    with pytest.raises(ValueError, match="not globally unique" if invalid == "duplicate_id" else "contract versions"):
        pipeline.run_config(config)
    assert lance.dataset(uri).count_rows() == 2
    capture.assert_not_called()
    ray_init.assert_not_called()
    run_inference.assert_not_called()


def test_all_camera_append_revalidates_camera_union_before_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new camera in a registration-racing append cannot silently escape an all-camera selection."""
    uri = str(tmp_path / "episodes.lance")
    lance.write_dataset(pa.Table.from_pylist([_episode(0)], schema=EPISODE_SCHEMA), uri, data_storage_version="2.2")
    original_ensure = pipeline.ensure_caption_fields

    def append_new_camera(*args: object, **kwargs: object) -> lance.LanceDataset:
        registered = original_ensure(*args, **kwargs)  # type: ignore[arg-type]
        appended = _episode(1)
        front = appended["camera_media"][0]
        appended["camera_media"] = [front, {**front, "camera_key": "rear"}]
        lance.write_dataset(pa.Table.from_pylist([appended], schema=EPISODE_SCHEMA), uri, mode="append")
        return registered

    capture = Mock(wraps=pipeline.capture_attempt)
    ray_init = Mock()
    monkeypatch.setattr(pipeline, "ensure_caption_fields", append_new_camera)
    monkeypatch.setattr(pipeline, "capture_attempt", capture)
    monkeypatch.setattr(pipeline, "ensure_ray_initialized", ray_init)
    with pytest.raises(ValueError, match="0 camera descriptors for 'rear'"):
        pipeline.run_config(_multi_config(uri))
    capture.assert_not_called()
    ray_init.assert_not_called()


@pytest.mark.parametrize("asset_state", ["missing", "config_only", "weights_only", "bad_processor", "complete"])
def test_cosmos_dependency_preflight_before_ray(
    asset_state: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Weights and both matching configurations must be staged before Ray initialization."""
    monkeypatch.setattr(environment, "CONTAINER_PATHS_MODEL_WEIGHT_CACHE_DIR", tmp_path / "models")
    uri = str(tmp_path / "episodes.lance")
    lance.write_dataset(pa.Table.from_pylist([_episode(0)], schema=EPISODE_SCHEMA), uri, data_storage_version="2.2")
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "multimodal-embedding",
            "input": {"episodes_lance_uri": uri},
            "video": {"camera_keys": ["front_wide"]},
        }
    )
    model = resolve_model(config.model.variant)
    model_dir = model.runtime_model_dir
    if asset_state != "missing":
        model_dir.mkdir(parents=True)
        if asset_state != "weights_only":
            (model_dir / "config.json").write_text(
                json.dumps({"num_video_frames": 8, "resolution": 336, "embed_dim": 768})
            )
            (model_dir / "processor_config.json").write_text(
                json.dumps({"num_video_frames": 8, "resolution": 224 if asset_state == "bad_processor" else 336})
            )
        if asset_state != "config_only":
            (model_dir / "model.safetensors").touch()
    ray_init = Mock(side_effect=RuntimeError("Ray initialized after preflight"))
    inference_mock = Mock()
    monkeypatch.setattr(pipeline, "ensure_ray_initialized", ray_init)
    monkeypatch.setattr(pipeline, "run_inference_phase", inference_mock)
    if asset_state == "complete":
        with pytest.raises(RuntimeError, match="Ray initialized after preflight"):
            pipeline.run_config(config)
        ray_init.assert_called_once()
    else:
        with pytest.raises(ValueError if asset_state == "bad_processor" else FileNotFoundError):
            pipeline.run_config(config)
        ray_init.assert_not_called()
    inference_mock.assert_not_called()


def test_policy_workspace_changes_without_changing_vector_contract(tmp_path: Path) -> None:
    """Eligibility gets an isolated staging scope while the field contract remains stable."""
    base = {
        "schema_version": 1,
        "kind": "multimodal-embedding",
        "input": {"episodes_lance_uri": str(tmp_path / "episodes.lance")},
        "video": {"camera_keys": ["front_wide"]},
    }
    first = resolve_config_data(base)
    raised = resolve_config_data(base | {"selection": {"max_episode_duration_s": 120.0}})
    retry = resolve_config_data(base | {"selection": {"retry_errors": True}})
    model = resolve_model(first.model.variant)
    digest = embedding_contract_digest(model, camera_key="front_wide", result_key="default_front_wide")
    assert (
        _test_camera_bindings(first, model, camera_key="front_wide").result.digest
        == _test_camera_bindings(raised, model, camera_key="front_wide").result.digest
    )
    field, _ = result_field_names("default_front_wide", model_key=model.key)
    roots = [
        resolve_workspace(
            config.output.staging_root_uri,
            field,
            digest,
            storage_profile="default",
            policy_namespace=canonical_digest(config.selection.model_dump(mode="json")),
        ).root_uri
        for config in (first, raised, retry)
    ]
    assert roots[0] != roots[1]
    assert roots[0] != roots[2]
    assert roots[0].startswith(str(tmp_path / "staging"))
    old_workspace = resolve_workspace(
        first.output.staging_root_uri,
        field,
        digest,
        storage_profile="default",
        policy_namespace=canonical_digest(first.selection.model_dump(mode="json")),
    )
    new_workspace = resolve_workspace(
        raised.output.staging_root_uri,
        field,
        digest,
        storage_profile="default",
        policy_namespace=canonical_digest(raised.selection.model_dump(mode="json")),
    )
    attempt = CaptionAttempt(version=1, pending_fragment_ids=(0,), complete_fragment_ids=())
    ensure_workspace(old_workspace, storage_profile="default", manifest={"test": "old-policy"})
    record_phase_a_completion(old_workspace, attempt, digest)
    assert phase_a_completion_covers(old_workspace, attempt, digest)
    assert not phase_a_completion_covers(new_workspace, attempt, digest)


def test_retry_errors_rejected_until_completed_fragment_rewrite(tmp_path: Path) -> None:
    """The staged-only recipe cannot silently ignore its future rewrite switch."""
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "multimodal-embedding",
            "input": {"episodes_lance_uri": str(tmp_path / "episodes.lance")},
            "video": {"camera_keys": ["front_wide"]},
            "selection": {"retry_errors": True},
        }
    )
    with pytest.raises(ValueError, match=r"not supported yet.*new output\.result_key"):
        pipeline.run_config(config)


def test_mixed_vector_fingerprint_ignores_hidden_null_metadata_bytes(tmp_path: Path) -> None:
    """Lance dictionary strings behind nulls do not change logically identical vector results."""
    uri = str(tmp_path / "results.lance")
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "multimodal-embedding",
            "input": {"episodes_lance_uri": uri},
            "video": {"camera_keys": ["front_wide"]},
        }
    )
    bindings = _test_camera_bindings(config, resolve_model(config.model.variant), camera_key="front_wide")
    source = pa.Table.from_pylist([_episode(index) for index in range(3)], schema=EPISODE_SCHEMA).append_column(
        "_rowaddr", pa.array([0, 1, 2], type=pa.uint64())
    )
    staged = _fake_batch(
        bindings.input.project_batch(source),
        model_key=config.model.variant,
        camera_key="front_wide",
        result_key=config.result_key_for_camera("front_wide"),
        error_id="episode-1",
    )
    schema = pa.schema([bindings.input.mapped_schema.field("episode_id"), *bindings.result.fields])
    expected = staged.select(schema.names).cast(schema)
    observed = lance.write_dataset(expected, uri, data_storage_version="2.2").to_table()
    assert expected.equals(observed)
    fingerprint = publication._caption_table_fingerprint(expected, bindings)
    assert fingerprint == publication._caption_table_fingerprint(observed, bindings)
    changed = observed.to_pylist()
    changed[0][bindings.result.caption_field_name] = changed[2][bindings.result.caption_field_name]
    assert fingerprint != publication._caption_table_fingerprint(pa.Table.from_pylist(changed, schema=schema), bindings)


def test_large_completed_fragment_uses_batch_vector_validation(tmp_path: Path) -> None:
    """A no-op scan validates vector values by Arrow batch, never Python row lists."""
    uri = str(tmp_path / "episodes.lance")
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "multimodal-embedding",
            "input": {"episodes_lance_uri": uri},
            "video": {"camera_keys": ["front_wide"]},
        }
    )
    model = resolve_model(config.model.variant)
    digest = embedding_contract_digest(model, camera_key="front_wide", result_key="default_front_wide")
    fields = result_fields("default_front_wide", model_key=model.key, digest=digest)
    records = []
    for index in range(128):
        vector = np.zeros(model.vector_dimension, dtype=np.float32)
        vector[index] = 1.0
        episode = _episode(index)
        source = {
            "episode_id": episode["episode_id"],
            "materialization_digest": episode["materialization_digest"],
            "camera_key": "front_wide",
            "media_sha256": episode["camera_media"][0]["media_sha256"],  # type: ignore[index]
            "start_ns": episode["start_ns"],
            "exclusive_end_ns": episode["exclusive_end_ns"],
        }
        terminal = build_terminal_result_row(
            source,
            model=model,
            result_key="default_front_wide",
            digest=digest,
            result=EmbeddingTerminalResult(
                status="success",
                vector=vector,
                max_episode_duration_s=60.0,
                error_code=None,
                error_message=None,
                media_duration_ns=1_000_000_000,
            ),
        )
        records.append({**episode, **terminal})
    schema = pa.schema([*EPISODE_SCHEMA, *fields])
    dataset = lance.write_dataset(pa.Table.from_pylist(records, schema=schema), uri, data_storage_version="2.2")
    bindings = _test_camera_bindings(config, model, camera_key="front_wide")
    batch_calls = 0
    original_batch = bindings.result.validate_terminal_batch
    assert original_batch is not None

    def counted_batch(batch: pa.Table) -> None:
        nonlocal batch_calls
        batch_calls += 1
        original_batch(batch)

    def forbid_row(_value: object, _metadata: dict[str, object] | None) -> None:
        pytest.fail("complete scan materialized vectors as Python rows")

    bindings = replace(
        bindings,
        result=replace(bindings.result, validate_terminal_value=forbid_row, validate_terminal_batch=counted_batch),
    )
    attempt = capture_attempt(dataset, bindings=bindings)
    assert attempt.complete_fragment_ids
    assert not attempt.pending_fragment_ids
    assert batch_calls >= 1
    assert batch_calls < 128


def _tracked_batch(
    batch: pa.Table,
    *,
    model_key: str,
    camera_key: str,
    result_key: str,
    calls_dir: str,
    fail_id: str | None,
) -> pa.Table:
    for episode_id, work_id in zip(batch["episode_id"].to_pylist(), batch[CAMERA_WORK_ID].to_pylist(), strict=True):
        Path(calls_dir, f"{work_id}-{uuid4().hex}").touch()
        if episode_id == fail_id:
            msg = "simulated encoder interruption"
            raise RuntimeError(msg)
    return _fake_batch(batch, model_key=model_key, camera_key=camera_key, result_key=result_key)


@pytest.mark.integration
def test_published_vectors_stay_bound_to_episodes_across_an_error(
    tmp_path: Path, curator_io_cluster: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Staging and publication preserve distinct successful vectors around an item error."""
    del curator_io_cluster
    uri = str(tmp_path / "episodes.lance")
    lance.write_dataset(
        pa.Table.from_pylist([_episode(index) for index in range(3)], schema=EPISODE_SCHEMA),
        uri,
        data_storage_version="2.2",
    )
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "multimodal-embedding",
            "input": {"episodes_lance_uri": uri},
            "video": {"camera_keys": ["front_wide"]},
        }
    )
    monkeypatch.setattr(pipeline, "validate_cosmos_model_directory", lambda model: model.runtime_model_dir)

    def fake_map(
        source: ray.data.Dataset, _config: object, *, num_input_rows: int, **_kwargs: object
    ) -> ray.data.Dataset:
        assert num_input_rows == 3
        return source.map_batches(
            _fake_batch,
            batch_format="pyarrow",
            batch_size=3,
            fn_kwargs={
                "model_key": config.model.variant,
                "camera_key": "front_wide",
                "result_key": config.result_key_for_camera("front_wide"),
                "error_id": "episode-1",
            },
        )

    monkeypatch.setattr(inference, "map_video_embeddings", fake_map)
    summary = pipeline.run_config(config)
    vector_field, metadata_field = result_field_names("default_front_wide", model_key=config.model.variant)
    rows = lance.dataset(uri).to_table(columns=["episode_id", vector_field, metadata_field]).to_pylist()
    by_id = {row["episode_id"]: row for row in rows}
    assert len(rows) == len(by_id) == 3
    for index in (0, 2):
        expected = np.zeros(768, dtype=np.float32)
        expected[index] = 1.0
        np.testing.assert_array_equal(by_id[f"episode-{index}"][vector_field], expected)
        assert by_id[f"episode-{index}"][metadata_field]["status"] == "success"
    assert by_id["episode-1"][vector_field] is None
    assert by_id["episode-1"][metadata_field]["error_code"] == "media_object_error"
    assert summary["cameras"]["front_wide"]["terminal_counts"] == {"success": 2, "error": 1, "unsupported_duration": 0}


@pytest.mark.integration
def test_publish_resume_and_noop(  # noqa: PLR0915 -- one recovery scenario also checks appended pending work
    tmp_path: Path,
    curator_io_cluster: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupted attempt publishes staged vectors once, then skips a rerun."""
    del curator_io_cluster
    uri = str(tmp_path / "episodes.lance")
    lance.write_dataset(
        pa.Table.from_pylist([_episode(0), _episode(1)], schema=EPISODE_SCHEMA), uri, data_storage_version="2.2"
    )
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "multimodal-embedding",
            "input": {"episodes_lance_uri": uri},
            "video": {"camera_keys": ["front_wide"]},
            "execution": {"inference_batch_size": 2, "parquet_rows_per_file": 2},
        }
    )
    monkeypatch.setattr(pipeline, "validate_cosmos_model_directory", lambda model: model.runtime_model_dir)

    pending_row_counts: list[int] = []

    def fake_map(
        source: ray.data.Dataset, _config: object, *, num_input_rows: int, **_kwargs: object
    ) -> ray.data.Dataset:
        pending_row_counts.append(num_input_rows)
        return source.map_batches(
            _fake_batch,
            batch_format="pyarrow",
            fn_kwargs={
                "model_key": config.model.variant,
                "camera_key": "front_wide",
                "result_key": "default_front_wide",
            },
        )

    monkeypatch.setattr(inference, "map_video_embeddings", fake_map)
    original_publish = pipeline.publish_atomic_staged_results
    original_batch_validator = embedding_publication.validate_terminal_batch
    workspace: CaptionWorkspace | None = None

    def interrupt(_uri: str, _attempt: object, staged: CaptionWorkspace, _bindings: object, **_kwargs: object) -> None:
        nonlocal workspace
        workspace = staged
        msg = "interrupted after staging"
        raise InterruptedError(msg)

    monkeypatch.setattr(pipeline, "publish_atomic_staged_results", interrupt)
    with pytest.raises(InterruptedError, match="after staging"):
        pipeline.run_config(config)
    assert workspace is not None
    assert Path(workspace.phase_a_completion_uri).exists()

    monkeypatch.setattr(pipeline, "publish_atomic_staged_results", original_publish)
    monkeypatch.setattr(
        pipeline,
        "run_inference_phase",
        lambda *_args, **_kwargs: pytest.fail("completed staging must be reused"),
    )
    summary = pipeline.run_config(config)
    model = resolve_model(config.model.variant)
    vector_field, metadata_field = result_field_names("default_front_wide", model_key=model.key)
    rows = lance.dataset(uri).to_table(columns=["episode_id", vector_field, metadata_field]).to_pylist()
    assert summary["published_fragments"] == 1
    assert summary["cameras"]["front_wide"]["terminal_counts"] == {"success": 2, "error": 0, "unsupported_duration": 0}
    assert [row[metadata_field]["episode_id"] for row in rows] == ["episode-0", "episode-1"]
    for index, row in enumerate(rows):
        expected = np.zeros(model.vector_dimension, dtype=np.float32)
        expected[index] = 1.0
        np.testing.assert_array_equal(row[vector_field], expected)
    version = lance.dataset(uri).version
    batch_checks = 0

    def counted_batch_validator(*args: object, **kwargs: object) -> None:
        nonlocal batch_checks
        batch_checks += 1
        original_batch_validator(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(embedding_publication, "validate_terminal_batch", counted_batch_validator)
    monkeypatch.setattr(
        embedding_publication,
        "validate_terminal_row",
        lambda *_args, **_kwargs: pytest.fail("complete scans must use the Arrow batch validator"),
    )
    assert pipeline.run_config(config)["published_fragments"] == 0
    assert lance.dataset(uri).version == version
    assert batch_checks > 0

    monkeypatch.undo()
    monkeypatch.setattr(pipeline, "validate_cosmos_model_directory", lambda model: model.runtime_model_dir)
    monkeypatch.setattr(inference, "map_video_embeddings", fake_map)
    lance.write_dataset(pa.Table.from_pylist([_episode(2)], schema=EPISODE_SCHEMA), uri, mode="append")
    appended_summary = pipeline.run_config(config)
    assert appended_summary["published_fragments"] == 1
    assert lance.dataset(uri).count_rows() == 3
    assert pending_row_counts == [2, 1]


@pytest.mark.integration
def test_large_fragment_is_split_before_inference_without_changing_row_identity(
    tmp_path: Path, curator_io_cluster: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single Lance fragment becomes small work blocks and still publishes exact episode coverage."""
    del curator_io_cluster
    uri = str(tmp_path / "episodes.lance")
    dataset = lance.write_dataset(
        pa.Table.from_pylist([_episode(index) for index in range(7)], schema=EPISODE_SCHEMA),
        uri,
        data_storage_version="2.2",
    )
    assert len(dataset.get_fragments()) == 1
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "multimodal-embedding",
            "input": {"episodes_lance_uri": uri},
            "video": {"camera_keys": ["front_wide"]},
            "execution": {"lance_read_batch_size": 2},
        }
    )

    def fake_map(source: ray.data.Dataset, _config: object, **_kwargs: object) -> ray.data.Dataset:
        return source.map_batches(
            _fake_batch,
            batch_format="pyarrow",
            batch_size=None,
            fn_kwargs={
                "model_key": config.model.variant,
                "camera_key": "front_wide",
                "result_key": "default_front_wide",
                "max_input_rows": 2,
            },
        )

    monkeypatch.setattr(pipeline, "validate_cosmos_model_directory", lambda model: model.runtime_model_dir)
    monkeypatch.setattr(inference, "map_video_embeddings", fake_map)
    summary = pipeline.run_config(config)
    assert summary["cameras"]["front_wide"]["terminal_counts"] == {"success": 7, "error": 0, "unsupported_duration": 0}
    assert summary["published_fragments"] == 1
    assert lance.dataset(uri).to_table(columns=["episode_id"])["episode_id"].to_pylist() == [
        f"episode-{index}" for index in range(7)
    ]


@pytest.mark.integration
def test_mid_staging_restart_reuses_committed_row_checkpoints(
    tmp_path: Path,
    curator_io_cluster: None,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Rows in committed Parquet checkpoints are not encoded on a fresh attempt."""
    del curator_io_cluster
    uri = str(tmp_path / "episodes.lance")
    lance.write_dataset(pa.Table.from_pylist([_episode(0)], schema=EPISODE_SCHEMA), uri, data_storage_version="2.2")
    for index in range(1, 4):
        lance.write_dataset(pa.Table.from_pylist([_episode(index)], schema=EPISODE_SCHEMA), uri, mode="append")
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "multimodal-embedding",
            "input": {"episodes_lance_uri": uri},
            "video": {"camera_keys": ["front_wide"]},
            "execution": {
                "inference_batch_size": 1,
                "parquet_rows_per_file": 1,
                "lance_read_batch_size": 1,
            },
        }
    )
    model = resolve_model(config.model.variant)
    camera = _test_camera_bindings(config, model, camera_key="front_wide")
    combined = embedding_publication.publication_bindings(
        _test_run_plan(config, model, tuple({"front_wide": config.result_key_for_camera("front_wide")}), {})
    )
    digest = combined.result.digest
    vector_field = camera.result.caption_field_name
    workspace = resolve_workspace(
        config.output.staging_root_uri,
        vector_field,
        digest,
        storage_profile="default",
        policy_namespace=canonical_digest(config.selection.model_dump(mode="json")),
    )
    calls_dir = tmp_path / "calls"
    calls_dir.mkdir()
    fail_id: str | None = "episode-3"

    def fake_map(source: ray.data.Dataset, _config: object, **_kwargs: object) -> ray.data.Dataset:
        return source.map_batches(
            _tracked_batch,
            batch_format="pyarrow",
            batch_size=1,
            fn_kwargs={
                "model_key": model.key,
                "camera_key": "front_wide",
                "result_key": "default_front_wide",
                "calls_dir": str(calls_dir),
                "fail_id": fail_id,
            },
        )

    monkeypatch.setattr(pipeline, "validate_cosmos_model_directory", lambda model: model.runtime_model_dir)
    monkeypatch.setattr(inference, "map_video_embeddings", fake_map)
    with pytest.raises(Exception, match="simulated encoder interruption"):
        pipeline.run_config(config)
    capfd.readouterr()
    checkpoint_files = list(Path(workspace.checkpoints_uri).glob("*.parquet"))
    assert checkpoint_files, "at least one successful row must reach a durable checkpoint before failure"
    checkpointed_ids = {
        str(work_id)
        for path in checkpoint_files
        for work_id in pq.read_table(path, columns=[CAMERA_WORK_ID])[CAMERA_WORK_ID].to_pylist()
    }
    assert checkpointed_ids
    before = {episode_id: len(list(calls_dir.glob(f"{episode_id}-*"))) for episode_id in checkpointed_ids}
    fail_id = None
    summary = pipeline.run_config(config)
    assert summary["cameras"]["front_wide"]["terminal_counts"] == {"success": 4, "error": 0, "unsupported_duration": 0}
    assert all(len(list(calls_dir.glob(f"{episode_id}-*"))) == count for episode_id, count in before.items())


@pytest.mark.integration
def test_orphan_staged_result_is_reconciled_after_checkpoint_loss(
    tmp_path: Path,
    curator_io_cluster: None,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """A crash after a result write but before its row checkpoint can replay safely."""
    del curator_io_cluster
    uri = str(tmp_path / "episodes.lance")
    lance.write_dataset(pa.Table.from_pylist([_episode(0)], schema=EPISODE_SCHEMA), uri, data_storage_version="2.2")
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "multimodal-embedding",
            "input": {"episodes_lance_uri": uri},
            "video": {"camera_keys": ["front_wide"]},
            "execution": {"inference_batch_size": 1, "parquet_rows_per_file": 1},
        }
    )

    def fake_map(source: ray.data.Dataset, _config: object, **_kwargs: object) -> ray.data.Dataset:
        return source.map_batches(
            _fake_batch,
            batch_format="pyarrow",
            fn_kwargs={
                "model_key": config.model.variant,
                "camera_key": "front_wide",
                "result_key": "default_front_wide",
            },
        )

    monkeypatch.setattr(pipeline, "validate_cosmos_model_directory", lambda model: model.runtime_model_dir)
    monkeypatch.setattr(inference, "map_video_embeddings", fake_map)
    original_publish = pipeline.publish_atomic_staged_results
    workspace: CaptionWorkspace | None = None

    def interrupt(_uri: str, _attempt: object, staged: CaptionWorkspace, _bindings: object, **_kwargs: object) -> None:
        nonlocal workspace
        workspace = staged
        msg = "after result write"
        raise InterruptedError(msg)

    monkeypatch.setattr(pipeline, "publish_atomic_staged_results", interrupt)
    with pytest.raises(InterruptedError, match="after result write"):
        pipeline.run_config(config)
    capfd.readouterr()
    assert workspace is not None
    result_paths = list(Path(workspace.results_uri).glob("*.parquet"))
    assert len(result_paths) == 1
    checkpoint_paths = list(Path(workspace.checkpoints_uri).glob("*.parquet"))
    assert checkpoint_paths
    for path in checkpoint_paths:
        path.unlink()
    Path(workspace.phase_a_completion_uri).unlink()

    def replay_with_error(source: ray.data.Dataset, _config: object, **_kwargs: object) -> ray.data.Dataset:
        return source.map_batches(
            _fake_error_batch,
            batch_format="pyarrow",
            fn_kwargs={
                "model_key": config.model.variant,
                "camera_key": "front_wide",
                "result_key": "default_front_wide",
            },
        )

    monkeypatch.setattr(inference, "map_video_embeddings", replay_with_error)

    def publish_replayed(*args: object, **kwargs: object) -> object:
        assert workspace is not None
        assert len(list(Path(workspace.results_uri).glob("*.parquet"))) == 2
        return original_publish(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pipeline, "publish_atomic_staged_results", publish_replayed)
    summary = pipeline.run_config(config)
    assert summary["published_fragments"] == 1
    vector_field, metadata_field = result_field_names("default_front_wide", model_key=config.model.variant)
    row = lance.dataset(uri).to_table(columns=[vector_field, metadata_field]).to_pylist()[0]
    assert row[vector_field][0] == 1.0
    assert row[metadata_field]["status"] == "success"


@pytest.mark.integration
def test_append_with_current_full_schema_keeps_split_binding_stable(
    tmp_path: Path,
    curator_io_cluster: None,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Null result leaves in an appended data file do not change Split-owned identity."""
    del curator_io_cluster
    uri = str(tmp_path / "episodes.lance")
    lance.write_dataset(pa.Table.from_pylist([_episode(0)], schema=EPISODE_SCHEMA), uri, data_storage_version="2.2")
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "multimodal-embedding",
            "input": {"episodes_lance_uri": uri},
            "video": {"camera_keys": ["front_wide"]},
            "execution": {"inference_batch_size": 1, "parquet_rows_per_file": 1},
        }
    )

    def fake_map(source: ray.data.Dataset, _config: object, **_kwargs: object) -> ray.data.Dataset:
        return source.map_batches(
            _fake_batch,
            batch_format="pyarrow",
            fn_kwargs={
                "model_key": config.model.variant,
                "camera_key": "front_wide",
                "result_key": "default_front_wide",
            },
        )

    monkeypatch.setattr(pipeline, "validate_cosmos_model_directory", lambda model: model.runtime_model_dir)
    monkeypatch.setattr(inference, "map_video_embeddings", fake_map)
    assert pipeline.run_config(config)["published_fragments"] == 1
    capfd.readouterr()
    current_schema = lance.dataset(uri).schema
    lance.write_dataset(pa.Table.from_pylist([_episode(1)], schema=current_schema), uri, mode="append")
    assert pipeline.run_config(config)["published_fragments"] == 1
    assert lance.dataset(uri).count_rows() == 2


@pytest.mark.integration
def test_terminal_item_error_publishes_null_fixed_width_vector(
    tmp_path: Path,
    curator_io_cluster: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nullable staged list can publish as the fixed-width Lance result field."""
    del curator_io_cluster
    uri = str(tmp_path / "episodes.lance")
    lance.write_dataset(pa.Table.from_pylist([_episode(0)], schema=EPISODE_SCHEMA), uri, data_storage_version="2.2")
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "multimodal-embedding",
            "input": {"episodes_lance_uri": uri},
            "video": {"camera_keys": ["front_wide"]},
            "execution": {"inference_batch_size": 1, "parquet_rows_per_file": 1},
        }
    )

    def error_map(source: ray.data.Dataset, _config: object, **_kwargs: object) -> ray.data.Dataset:
        return source.map_batches(
            _fake_error_batch,
            batch_format="pyarrow",
            fn_kwargs={
                "model_key": config.model.variant,
                "camera_key": "front_wide",
                "result_key": "default_front_wide",
            },
        )

    monkeypatch.setattr(pipeline, "validate_cosmos_model_directory", lambda model: model.runtime_model_dir)
    monkeypatch.setattr(inference, "map_video_embeddings", error_map)
    summary = pipeline.run_config(config)
    assert summary["cameras"]["front_wide"]["terminal_counts"] == {"success": 0, "error": 1, "unsupported_duration": 0}
    assert summary["cameras"]["front_wide"]["error_codes"] == {"media_object_error": 1}
    vector_field, metadata_field = result_field_names("default_front_wide", model_key=config.model.variant)
    table = lance.dataset(uri).to_table(columns=[vector_field, metadata_field])
    assert pa.types.is_fixed_size_list(table.schema.field(vector_field).type)
    assert table.to_pylist()[0][vector_field] is None
    assert table.to_pylist()[0][metadata_field]["status"] == "error"


@pytest.mark.integration
@pytest.mark.parametrize("tag_camera_key", [True, False])
@pytest.mark.parametrize("failure", ["before_commit", "after_commit", "lost_response"])
def test_atomic_publication_recovery(
    failure: str,
    tmp_path: Path,
    curator_io_cluster: None,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    *,
    tag_camera_key: bool,
) -> None:
    """A finite selection is visible together; each interruption reuses durable embedding work."""
    del curator_io_cluster
    uri = str(tmp_path / "episodes.lance")
    lance.write_dataset(pa.Table.from_pylist([_episode(0)], schema=EPISODE_SCHEMA), uri, data_storage_version="2.2")
    lance.write_dataset(pa.Table.from_pylist([_episode(1)], schema=EPISODE_SCHEMA), uri, mode="append")
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "multimodal-embedding",
            "input": {"episodes_lance_uri": uri},
            "video": {"camera_keys": ["front_wide"]},
            "output": {"tag_camera_key": tag_camera_key},
            "execution": {"inference_batch_size": 1, "parquet_rows_per_file": 1},
        }
    )

    def fake_map(source: ray.data.Dataset, _config: object, **_kwargs: object) -> ray.data.Dataset:
        return source.map_batches(
            _fake_batch,
            batch_format="pyarrow",
            fn_kwargs={
                "model_key": config.model.variant,
                "camera_key": "front_wide",
                "result_key": config.result_key_for_camera("front_wide"),
            },
        )

    monkeypatch.setattr(pipeline, "validate_cosmos_model_directory", lambda model: model.runtime_model_dir)
    monkeypatch.setattr(inference, "map_video_embeddings", fake_map)
    original_commit = embedding_publication._commit_atomic_update
    attempts = 0
    before_version: int | None = None
    field, _ = result_field_names(config.result_key_for_camera("front_wide"), model_key=config.model.variant)

    def interrupt_commit(*args: object, **kwargs: object) -> None:
        nonlocal attempts, before_version
        attempts += 1
        before = lance.dataset(uri)
        before_version = int(before.version)
        assert before.to_table(columns=[field])[field].null_count == 2
        if failure == "before_commit":
            msg = "storage unavailable before commit"
            raise OSError(msg)
        original_commit(*args, **kwargs)  # type: ignore[arg-type]
        after = lance.dataset(uri)
        assert after.to_table(columns=[field])[field].null_count == 0
        assert int(after.version) == before_version + 1
        if failure == "after_commit":
            msg = "simulated driver exit after atomic commit"
            raise SystemExit(msg)
        msg = "lost commit response"
        raise OSError(msg)

    monkeypatch.setattr(embedding_publication, "_commit_atomic_update", interrupt_commit)
    if failure == "lost_response":
        summary = pipeline.run_config(config)
        assert summary["already_committed_fragments"] == 2
        assert summary["published_fragments"] == 0
    else:
        expected_error = publication.PublicationError if failure == "before_commit" else SystemExit
        with pytest.raises(expected_error):
            pipeline.run_config(config)
        assert lance.dataset(uri).to_table(columns=[field])[field].null_count == (
            2 if failure == "before_commit" else 0
        )
    capfd.readouterr()
    assert attempts == (config.execution.commit_attempts if failure == "before_commit" else 1)
    monkeypatch.setattr(embedding_publication, "_commit_atomic_update", original_commit)
    monkeypatch.setattr(
        pipeline, "run_inference_phase", lambda *_args, **_kwargs: pytest.fail("must reuse staged results")
    )
    summary = pipeline.run_config(config)
    assert summary["skipped_complete_fragments"] == (0 if failure == "before_commit" else 2)
    assert summary["published_fragments"] == (2 if failure == "before_commit" else 0)
    assert summary["cameras"]["front_wide"]["terminal_counts"] == {"success": 2, "error": 0, "unsupported_duration": 0}


@pytest.mark.integration
def test_concurrent_episode_binding_change_rejects_publication(
    tmp_path: Path,
    curator_io_cluster: None,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """A changed split-owned binding cannot receive a vector from older staging."""
    del curator_io_cluster
    uri = str(tmp_path / "episodes.lance")
    source = _episode(0)
    lance.write_dataset(pa.Table.from_pylist([source], schema=EPISODE_SCHEMA), uri, data_storage_version="2.2")
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "multimodal-embedding",
            "input": {"episodes_lance_uri": uri},
            "video": {"camera_keys": ["front_wide"]},
            "execution": {"inference_batch_size": 1, "parquet_rows_per_file": 1},
        }
    )

    def fake_map(rows: ray.data.Dataset, _config: object, **_kwargs: object) -> ray.data.Dataset:
        return rows.map_batches(
            _fake_batch,
            batch_format="pyarrow",
            fn_kwargs={
                "model_key": config.model.variant,
                "camera_key": "front_wide",
                "result_key": "default_front_wide",
            },
        )

    monkeypatch.setattr(pipeline, "validate_cosmos_model_directory", lambda model: model.runtime_model_dir)
    monkeypatch.setattr(inference, "map_video_embeddings", fake_map)
    original_publish = pipeline.publish_atomic_staged_results

    def mutate_before_publish(*args: object, **kwargs: object) -> object:
        dataset = lance.dataset(uri)
        fragment = dataset.get_fragment(0)
        assert fragment is not None
        schema = pa.schema([dataset.schema.field("episode_id"), dataset.schema.field("materialization_digest")])
        replacement = pa.Table.from_pylist(
            [{"episode_id": source["episode_id"], "materialization_digest": "concurrent-change"}], schema=schema
        )
        metadata, field_ids = fragment.update_columns(
            pa.RecordBatchReader.from_batches(schema, replacement.to_batches()),
            left_on="episode_id",
            right_on="episode_id",
        )
        transaction = lance.Transaction(
            read_version=dataset.version,
            operation=lance.LanceOperation.Update(updated_fragments=[metadata], fields_modified=field_ids),
        )
        lance.LanceDataset.commit(uri, transaction, max_retries=0)
        return original_publish(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pipeline, "publish_atomic_staged_results", mutate_before_publish)
    with pytest.raises(publication.PublicationError, match="changed a multimodal-split-owned physical binding"):
        pipeline.run_config(config)
    capfd.readouterr()
    field, _ = result_field_names("default_front_wide", model_key=config.model.variant)
    assert lance.dataset(uri).to_table(columns=[field]).to_pylist() == [{field: None}]


def _atomic_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    error_id: str | None = None,
) -> tuple[
    ResolvedMultimodalEmbeddingConfig,
    CaptionAttempt,
    CaptionWorkspace,
    CaptionRunBindings,
    tuple[publication.PreparedFragmentPublication, ...],
]:
    """Use real Lance/Parquet while replacing only Ray execution with in-process workers."""
    uri = str(tmp_path / "episodes.lance")
    for index in range(2):
        lance.write_dataset(
            pa.Table.from_pylist([_episode(index)], schema=EPISODE_SCHEMA),
            uri,
            mode="create" if index == 0 else "append",
            data_storage_version="2.2",
        )
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "multimodal-embedding",
            "input": {"episodes_lance_uri": uri},
            "video": {"camera_keys": ["front_wide"]},
        }
    )
    model = resolve_model(config.model.variant)
    camera = _test_camera_bindings(config, model, camera_key="front_wide")
    cameras = {"front_wide": camera}
    dataset = ensure_caption_fields(uri, storage_options=None, bindings=camera, attempts=1)
    pending = {int(fragment.fragment_id): ("front_wide",) for fragment in dataset.get_fragments()}
    bindings = embedding_publication.publication_bindings(
        _test_run_plan(config, model, tuple({"front_wide": config.result_key_for_camera("front_wide")}), pending)
    )
    attempt = capture_attempt(dataset, bindings=bindings)
    workspace = resolve_workspace(
        config.output.staging_root_uri,
        bindings.result.caption_field_name,
        bindings.result.digest,
        storage_profile="default",
    )
    Path(workspace.results_uri).mkdir(parents=True)
    prepared = []
    for fragment in dataset.get_fragments():
        source = fragment.scanner(columns=list(bindings.input.source_columns), with_row_address=True).to_table()
        staged = _fake_batch(
            project_camera_work(source, camera_keys=("front_wide",)),
            model_key=model.key,
            camera_key="front_wide",
            result_key=config.result_key_for_camera("front_wide"),
            error_id=error_id,
        )
        pq.write_table(staged, Path(workspace.results_uri) / f"{fragment.fragment_id}.parquet")
        payload = embedding_publication._prepare_camera_fragment(
            staged,
            uri=uri,
            attempt_version=attempt.version,
            current_version=attempt.version,
            storage_options=None,
            plan=_test_run_plan(config, model, tuple(cameras), pending),
        )
        prepared.append(
            embedding_publication._parse_camera_publication(payload["payload"][0].as_py(), int(fragment.fragment_id))
        )
    candidates = tuple(prepared)
    monkeypatch.setattr(embedding_publication, "_prepare_camera_publications", lambda *_args, **_kwargs: candidates)
    monkeypatch.setattr(embedding_publication, "_iter_camera_publications", iter)

    def prepare_one(fragment_id: int, **kwargs: object) -> publication.PreparedFragmentPublication:
        staged = pq.read_table(Path(workspace.results_uri) / f"{fragment_id}.parquet")
        payload = embedding_publication._prepare_camera_fragment(
            staged,
            uri=kwargs["uri"],
            attempt_version=kwargs["attempt_version"],
            current_version=kwargs["current_version"],
            storage_options=None,
            plan=kwargs["plan"],
        )
        return embedding_publication._parse_camera_publication(payload["payload"][0].as_py(), fragment_id)

    monkeypatch.setattr(embedding_publication, "_reprepare_camera_fragment", prepare_one)
    monkeypatch.setattr(ray, "init", lambda *_args, **_kwargs: pytest.fail("unit tests must not start Ray"))
    return config, attempt, workspace, bindings, candidates


def test_camera_reprepare_reports_missing_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An absent workspace produces the publication error without starting real Ray."""
    reprepare = embedding_publication._reprepare_camera_fragment
    config, attempt, workspace, _bindings, _ = _atomic_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(embedding_publication, "_reprepare_camera_fragment", reprepare)
    empty = replace(workspace, results_uri=str(tmp_path / "absent-results"))
    monkeypatch.setattr(ray, "remote", lambda **_kwargs: lambda fn: Mock(remote=fn))
    monkeypatch.setattr(ray, "get", lambda value: value)
    with pytest.raises(publication.PublicationError, match=r"No staged camera results.*fragment 0"):
        embedding_publication._reprepare_camera_fragment(
            0,
            uri=config.input.episodes_lance_uri,
            attempt_version=attempt.version,
            current_version=attempt.version,
            workspace=empty,
            plan=_test_run_plan(config, resolve_model(config.model.variant), ("front_wide",), {}),
            storage_options=None,
        )


def _atomic_publish(
    config: ResolvedMultimodalEmbeddingConfig,
    attempt: CaptionAttempt,
    workspace: CaptionWorkspace,
    bindings: CaptionRunBindings,
) -> publication.PublicationSummary:
    del bindings  # shared adapters are tested independently; publication receives the explicit plan
    return embedding_publication.publish_atomic_staged_results(
        config.input.episodes_lance_uri,
        attempt,
        workspace,
        _test_run_plan(
            config,
            resolve_model(config.model.variant),
            ("front_wide",),
            dict.fromkeys(attempt.pending_fragment_ids, ("front_wide",)),
        ),
        storage_options=None,
        commit_attempts=3,
    )


def _commit_episode_columns(uri: str, fragment_id: int, rows: list[dict[str, object]]) -> None:
    dataset = lance.dataset(uri)
    fragment = dataset.get_fragment(fragment_id)
    assert fragment is not None
    schema = pa.schema([dataset.schema.field(name) for name in rows[0]])
    table = pa.Table.from_pylist(rows, schema=schema)
    updated, modified = fragment.update_columns(table, left_on="episode_id", right_on="episode_id")
    lance.LanceDataset.commit(
        uri,
        lance.Transaction(
            read_version=dataset.version,
            operation=lance.LanceOperation.Update(updated_fragments=[updated], fields_modified=modified),
        ),
        max_retries=0,
    )


@pytest.mark.parametrize("error_id", [None, "episode-1"])
def test_atomic_selection_publishes_one_version_and_noop_stays_put(
    error_id: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two fragments produce one result transaction, with a scoped recovery receipt."""
    config, attempt, workspace, bindings, _ = _atomic_setup(tmp_path, monkeypatch, error_id=error_id)
    uri = config.input.episodes_lance_uri
    before = lance.dataset(uri).version
    summary = _atomic_publish(config, attempt, workspace, bindings)
    latest = lance.dataset(uri)
    assert latest.version == before + 1
    assert summary.published_fragments == 2
    assert latest.to_table(columns=[bindings.result.caption_field_name]).column(0).null_count == int(
        error_id is not None
    )
    transaction = latest.read_transaction(latest.version)
    assert len(transaction.transaction_properties[embedding_publication._ATOMIC_PUBLICATION_KEY]) == 64
    complete = capture_attempt(latest, bindings=bindings)
    assert _atomic_publish(config, complete, workspace, bindings).published_fragments == 0
    assert lance.dataset(uri).version == latest.version


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "fields"])
def test_atomic_invalid_descriptors_publish_nothing(
    invalid: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad candidate anywhere in the finite selection leaves every embedding pending."""
    config, attempt, workspace, bindings, prepared = _atomic_setup(tmp_path, monkeypatch)
    if invalid == "missing":
        prepared = prepared[:1]
    elif invalid == "duplicate":
        prepared = (prepared[0], prepared[0])
    else:
        descriptor = prepared[1].descriptor
        assert descriptor is not None
        prepared = (prepared[0], replace(prepared[1], descriptor=replace(descriptor, modified_field_ids=(99999,))))
    monkeypatch.setattr(embedding_publication, "_prepare_camera_publications", lambda *_args, **_kwargs: prepared)
    before = lance.dataset(config.input.episodes_lance_uri)
    with pytest.raises(publication.PublicationError):
        _atomic_publish(config, attempt, workspace, bindings)
    after = lance.dataset(config.input.episodes_lance_uri)
    assert after.version == before.version
    assert after.to_table(columns=[bindings.result.caption_field_name]).column(0).null_count == 2


def test_partial_preparation_failure_keeps_all_results_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Receiving the first descriptor cannot expose its results before the other worker finishes."""
    config, attempt, workspace, bindings, prepared = _atomic_setup(tmp_path, monkeypatch)

    def interrupt(_values: object) -> Iterator[publication.PreparedFragmentPublication]:
        yield prepared[0]
        msg = "preparation worker interrupted"
        raise OSError(msg)

    monkeypatch.setattr(embedding_publication, "_iter_camera_publications", interrupt)
    before = lance.dataset(config.input.episodes_lance_uri).version
    with pytest.raises(OSError, match="worker interrupted"):
        _atomic_publish(config, attempt, workspace, bindings)
    assert lance.dataset(config.input.episodes_lance_uri).version == before


def test_append_during_atomic_publication_remains_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A producer append is outside the pinned selection and receives no stale vector."""
    config, attempt, workspace, bindings, _ = _atomic_setup(tmp_path, monkeypatch)
    lance.write_dataset(
        pa.Table.from_pylist([_episode(2)], schema=EPISODE_SCHEMA), config.input.episodes_lance_uri, mode="append"
    )
    assert _atomic_publish(config, attempt, workspace, bindings).published_fragments == 2
    rows = (
        lance.dataset(config.input.episodes_lance_uri)
        .to_table(columns=[bindings.result.caption_field_name])
        .column(0)
        .to_pylist()
    )
    assert rows[0] is not None
    assert rows[1] is not None
    assert rows[2] is None


def test_atomic_disjoint_caption_commit_race_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """A same-fragment caption update wins its race, and embedding descriptors are rebuilt."""
    config, attempt, workspace, bindings, _ = _atomic_setup(tmp_path, monkeypatch)
    uri = config.input.episodes_lance_uri
    original = embedding_publication._commit_atomic_update
    commits = 0

    def race(*args: object, **kwargs: object) -> None:
        nonlocal commits
        commits += 1
        if commits == 1:
            lance.dataset(uri).add_columns(pa.schema([pa.field("caption_test", pa.string())]))
            _commit_episode_columns(uri, 0, [{"episode_id": "episode-0", "caption_test": "a car turns left"}])
        original(*args, **kwargs)

    monkeypatch.setattr(embedding_publication, "_commit_atomic_update", race)
    assert _atomic_publish(config, attempt, workspace, bindings).published_fragments == 2
    assert commits == 2
    latest = lance.dataset(uri)
    assert latest.to_table(columns=["caption_test"]).column(0).to_pylist() == ["a car turns left", None]
    assert latest.to_table(columns=[bindings.result.caption_field_name]).column(0).null_count == 0
    capfd.readouterr()


@pytest.mark.parametrize("written_fragments", [1, 2])
def test_competing_camera_writer_is_rejected(
    written_fragments: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without this run's receipt, even identical competing results are a conflict."""
    config, attempt, workspace, bindings, _ = _atomic_setup(tmp_path, monkeypatch)
    uri = config.input.episodes_lance_uri
    columns = ["episode_id", bindings.result.caption_field_name, bindings.result.metadata_field_name]
    for fragment_id in range(written_fragments):
        staged = pq.read_table(Path(workspace.results_uri) / f"{fragment_id}.parquet")
        _commit_episode_columns(uri, fragment_id, _wide_staged(staged, bindings).select(columns).to_pylist())
    before = lance.dataset(uri).version
    with pytest.raises(publication.PublicationError, match="one active writer per result-field set"):
        _atomic_publish(config, attempt, workspace, bindings)
    assert lance.dataset(uri).version == before
    assert (
        lance.dataset(uri).to_table(columns=[bindings.result.caption_field_name]).column(0).null_count
        == 2 - written_fragments
    )


def test_stale_descriptor_rejects_changed_staged_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unrelated column writes permit rebuilding, but cannot hide checkpoint value drift."""
    config, attempt, workspace, bindings, _ = _atomic_setup(tmp_path, monkeypatch)
    uri = config.input.episodes_lance_uri
    lance.dataset(uri).add_columns(pa.schema([pa.field("caption_test", pa.string())]))
    _commit_episode_columns(uri, 0, [{"episode_id": "episode-0", "caption_test": "a car turns left"}])
    path = Path(workspace.results_uri) / "0.parquet"
    staged = pq.read_table(path)
    rows = staged.to_pylist()
    rows[0]["embedding"] = np.eye(768, dtype=np.float32)[2].tolist()
    pq.write_table(pa.Table.from_pylist(rows, schema=staged.schema), path)
    before = lance.dataset(uri).version
    with pytest.raises(publication.PublicationError, match="differs from the authoritative staged selection"):
        _atomic_publish(config, attempt, workspace, bindings)
    latest = lance.dataset(uri)
    assert latest.version == before
    assert latest.to_table(columns=[bindings.result.caption_field_name]).column(0).null_count == 2


def test_atomic_receipt_requires_matching_canonical_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An old receipt cannot authorize success after a competing writer changes the vector."""
    config, attempt, workspace, bindings, _ = _atomic_setup(tmp_path, monkeypatch)
    uri = config.input.episodes_lance_uri
    _atomic_publish(config, attempt, workspace, bindings)
    field = bindings.result.caption_field_name
    vector = [0.0] * resolve_model(config.model.variant).vector_dimension
    vector[1] = 1.0
    _commit_episode_columns(uri, 0, [{"episode_id": "episode-0", field: vector}])
    before = lance.dataset(uri).version
    with pytest.raises(publication.PublicationError, match="differs from its authoritative staged result"):
        _atomic_publish(config, attempt, workspace, bindings)
    assert lance.dataset(uri).version == before


@pytest.mark.parametrize("camera", ["Front-Wide", "front-wide", "x" * 41])
def test_all_camera_tagging_rejects_unsafe_names_before_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, camera: str
) -> None:
    """Discovered camera names must satisfy the exact effective-key contract before any writes."""
    row = _episode(0)
    row["camera_media"][0]["camera_key"] = camera
    uri = str(tmp_path / "episodes.lance")
    dataset = lance.write_dataset(pa.Table.from_pylist([row], schema=EPISODE_SCHEMA), uri, data_storage_version="2.2")
    before = dataset.version
    monkeypatch.setattr(pipeline, "ensure_caption_fields", lambda *_a, **_kw: pytest.fail("must not register fields"))
    monkeypatch.setattr(pipeline, "ensure_ray_initialized", lambda *_a, **_kw: pytest.fail("must not start Ray"))
    with pytest.raises(ValueError, match="Invalid effective result key"):
        pipeline.run_config(_multi_config(uri))
    assert lance.dataset(uri).version == before


@pytest.mark.parametrize("visibility", ["stale", "transport"])
def test_lost_commit_response_waits_for_visible_verified_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, visibility: str
) -> None:
    """Brief visibility or read failures reconcile without another result commit."""
    config, attempt, workspace, bindings, _ = _atomic_setup(tmp_path, monkeypatch)
    uri = config.input.episodes_lance_uri
    open_dataset = lance.dataset
    stale_snapshot = open_dataset(uri)
    commit = embedding_publication._commit_atomic_update
    committed = False
    reads = 0
    commits = 0
    sleeps: list[float] = []

    def lose_response(*args: object, **kwargs: object) -> None:
        nonlocal committed, commits
        commits += 1
        commit(*args, **kwargs)  # type: ignore[arg-type]
        committed = True
        msg = "lost commit response"
        raise OSError(msg)

    def delayed_read(*args: object, **kwargs: object) -> lance.LanceDataset:
        nonlocal reads
        if committed and not kwargs.get("version"):
            reads += 1
            if reads <= 2:
                if visibility == "transport":
                    msg = "temporary read failure"
                    raise OSError(msg)
                return stale_snapshot
        return open_dataset(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(embedding_publication, "_commit_atomic_update", lose_response)
    monkeypatch.setattr(lance, "dataset", delayed_read)
    monkeypatch.setattr(embedding_publication.time, "sleep", sleeps.append)
    summary = _atomic_publish(config, attempt, workspace, bindings)
    assert commits == 1
    assert reads == 3
    assert sleeps == [0.25, 0.5]
    assert summary.already_committed_fragments == 2
    latest = open_dataset(uri)
    assert latest.version == stale_snapshot.version + 1
    assert latest.to_table(columns=[bindings.result.caption_field_name]).column(0).null_count == 0


def test_persistent_stale_reads_keep_staging_for_later_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exhausting visibility retries preserves canonical results and durable recovery files."""
    config, attempt, workspace, bindings, _ = _atomic_setup(tmp_path, monkeypatch)
    uri = config.input.episodes_lance_uri
    open_dataset = lance.dataset
    stale_snapshot = open_dataset(uri)
    commit = embedding_publication._commit_atomic_update
    committed = False

    def lose_response(*args: object, **kwargs: object) -> None:
        nonlocal committed
        commit(*args, **kwargs)  # type: ignore[arg-type]
        committed = True
        msg = "lost commit response"
        raise OSError(msg)

    def stale_read(*args: object, **kwargs: object) -> lance.LanceDataset:
        if committed and not kwargs.get("version"):
            return stale_snapshot
        return open_dataset(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(embedding_publication, "_commit_atomic_update", lose_response)
    monkeypatch.setattr(lance, "dataset", stale_read)
    monkeypatch.setattr(embedding_publication.time, "sleep", lambda _seconds: None)
    with pytest.raises(publication.PublicationError, match="Staged results are retained"):
        _atomic_publish(config, attempt, workspace, bindings)
    assert list(Path(workspace.results_uri).glob("*.parquet"))
    assert open_dataset(uri).version == stale_snapshot.version + 1
    monkeypatch.setattr(lance, "dataset", open_dataset)
    summary = _atomic_publish(config, attempt, workspace, bindings)
    assert summary.already_committed_fragments == 2
    assert open_dataset(uri).version == stale_snapshot.version + 1


def test_cleanup_failure_preserves_successful_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Workspace cleanup cannot turn verified canonical results into a failed run."""
    config, attempt, workspace, bindings, _ = _atomic_setup(tmp_path, monkeypatch)
    ensure_workspace(
        workspace,
        storage_profile="default",
        manifest={
            "workspace_schema_version": pipeline.WORKSPACE_SCHEMA_VERSION,
            "episodes_lance_uri": config.input.episodes_lance_uri,
            "camera_contracts": {
                "front_wide": pipeline.normalized_embedding_contract(
                    resolve_model(config.model.variant),
                    camera_key="front_wide",
                    result_key=config.result_key_for_camera("front_wide"),
                )
            },
            "selection": config.selection.model_dump(mode="json"),
            "checkpoint_adapter_version": pipeline.CHECKPOINT_ADAPTER_VERSION,
            "staging_vector_layout": "episode_camera_rows_v1",
        },
    )
    record_phase_a_completion(workspace, attempt, bindings.result.digest)
    monkeypatch.setattr(pipeline, "resolve_workspace", lambda *_args, **_kwargs: workspace)
    ray_init = Mock()
    monkeypatch.setattr(pipeline, "ensure_ray_initialized", ray_init)
    monkeypatch.setattr(pipeline, "run_inference_phase", lambda *_args, **_kwargs: pytest.fail("no-op must not infer"))

    def fail_cleanup(*_args: object, **_kwargs: object) -> None:
        msg = "simulated cleanup failure"
        raise OSError(msg)

    # Capture the expected warning without leaking it into passing CI logs.
    captured_logger = Mock()
    monkeypatch.setattr(pipeline, "logger", captured_logger)
    monkeypatch.setattr(pipeline, "cleanup_workspace", fail_cleanup)
    before = lance.dataset(config.input.episodes_lance_uri).version
    summary = pipeline.run_config(config)
    assert summary["cleanup_complete"] is False
    assert summary["published_fragments"] == 2
    assert summary["skipped_complete_fragments"] == 0
    assert summary["cameras"]["front_wide"]["terminal_counts"]["success"] == 2
    assert summary["episodes_lance_version"] == before + 1
    assert "workspace cleanup failed" in captured_logger.warning.call_args.args[0]
    assert isinstance(captured_logger.warning.call_args.args[1], OSError)
    latest = lance.dataset(config.input.episodes_lance_uri)
    assert latest.to_table(columns=[bindings.result.caption_field_name]).column(0).null_count == 0

    rerun = pipeline.run_config(config)
    assert rerun["cleanup_complete"] is False
    assert rerun["published_fragments"] == 0
    assert rerun["skipped_complete_fragments"] == 2
    assert rerun["episodes_lance_version"] == before + 1
    ray_init.assert_called_once()
    assert list(Path(workspace.results_uri).glob("*.parquet"))
    assert captured_logger.warning.call_count == 2


def _wide_staged(table: pa.Table, bindings: CaptionRunBindings) -> pa.Table:
    return table.rename_columns(
        [
            bindings.result.caption_field_name
            if name == "embedding"
            else bindings.result.metadata_field_name
            if name == "embedding_metadata"
            else name
            for name in table.column_names
        ]
    )
