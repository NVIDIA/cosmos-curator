# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Separate CPU decoding survives real Ray transport, and scene plans resume on a fresh cluster."""

from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import attrs
import pyarrow as pa
import pytest
import ray
import torch
from ray.data.context import DataContext

from cosmos_curator.next.append.attempts import reconcile_and_select_attempt
from cosmos_curator.next.append.execution import checkpointed_processing
from cosmos_curator.next.recipes.video_cut import pipeline, transnetv2
from cosmos_curator.next.recipes.video_cut.bindings import resolve_bindings
from cosmos_curator.next.recipes.video_cut.config import (
    ExecutionConfig,
    ResolvedVideoCutConfig,
    TransNetV2Config,
    resolve_config_data,
)
from cosmos_curator.next.recipes.video_cut.discovery import discover_items
from tests.cosmos_curator.next.append.fresh_cluster import assert_fresh_cluster_resume
from tests.cosmos_curator.next.core.helpers import staged_results
from tests.cosmos_curator.next.fresh_driver import run_in_fresh_driver
from tests.cosmos_curator.next.recipes.video_cut.conftest import RECIPE


class _CPUPlanner(transnetv2.TransNetV2Planner):
    """Exercise the production inference callable with deterministic CPU predictions."""

    def __init__(self, *, split: TransNetV2Config, execution: ExecutionConfig) -> None:
        self._split = split
        self._execution = execution
        self._model = lambda inputs: torch.zeros((*inputs.shape[:2], 1))

    def __call__(self, batch: pa.Table) -> pa.Table:
        resources = ray.get_runtime_context().get_assigned_resources()
        assert resources["CPU"] == 0.25
        assert "curator_io" not in resources
        assert "video_bytes" not in batch.column_names
        with patch.object(transnetv2, "predict_transitions", partial(transnetv2.predict_transitions, device="cpu")):
            return super().__call__(batch)


def _scene_config(config: ResolvedVideoCutConfig, *, uris: list[str] | None = None) -> ResolvedVideoCutConfig:
    """Cut scenes with reservations that fit the small CPU test clusters."""
    raw = config.model_dump(mode="json")
    raw["split"] = {"method": "transnetv2", "decoder": "pyav", "min_duration_s": 0.01}
    if uris is not None:
        raw["input"]["uris"] = uris
    return resolve_config_data(
        raw,
        overrides=[
            "execution.transnetv2.cpus=1",
            "execution.transnetv2.memory_gib=0.25",
            "execution.transnetv2.inference_cpus=0.25",
            "execution.transnetv2.inference_memory_gib=0.25",
        ],
    )


def _without_gpus(config: ResolvedVideoCutConfig, **execution: object) -> ResolvedVideoCutConfig:
    """Drop the GPU reservation, which config validation rejects, so inference runs on CPU-only clusters."""
    transnetv2_execution = config.execution.transnetv2.model_copy(update={"gpus": 0})
    updated = config.execution.model_copy(update={"transnetv2": transnetv2_execution, **execution})
    return config.model_copy(update={"execution": updated})


@pytest.mark.integration
def test_separate_decode_transport_and_terminal_errors(
    curator_io_cluster: None,
    config: ResolvedVideoCutConfig,
    video_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Downloads hold IO slots and decode tasks their own CPUs; terminal errors keep source recovery boundaries."""
    del curator_io_cluster
    uris = [
        "s3://test-bucket/missing.mp4",
        "s3://test-bucket/invalid.mp4",
        *[f"s3://test-bucket/video-{index}.mp4" for index in range(4)],
    ]
    # Only the model and its GPU reservation are replaced; production stage
    # assembly, decode, schemas, and sinks execute.
    config = _without_gpus(_scene_config(config, uris=uris))
    monkeypatch.setattr(transnetv2, "ray_data_gpu_runtime_env", lambda _: {})
    monkeypatch.setattr(transnetv2, "TransNetV2Planner", _CPUPlanner)
    monkeypatch.setattr(transnetv2, "validate_weights", Mock())
    monkeypatch.setattr(DataContext.get_current(), "enable_progress_bars", False)
    payload = video_path.read_bytes()

    class LocalDownloader(transnetv2.SourceDownloader):
        def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
            resources = ray.get_runtime_context().get_assigned_resources()
            assert resources["CPU"] == 0.25
            assert resources["curator_io"] == 1
            if row["source_uri"].endswith("missing.mp4"):
                return row | {
                    "video_bytes": b"",
                    "error_stage": "source-read",
                    "error_message": "S3 source unavailable: 404",
                }
            data = b"invalid video" if row["source_uri"].endswith("invalid.mp4") else payload
            return row | {"video_bytes": data, "error_stage": None, "error_message": None}

    decode = transnetv2.decode_sources

    def decode_on_cpu(batch: pa.Table, *, execution: ExecutionConfig) -> pa.Table:
        resources = ray.get_runtime_context().get_assigned_resources()
        assert resources["CPU"] == 1
        assert resources.get("GPU", 0) == 0
        assert "curator_io" not in resources
        return decode(batch, execution=execution)

    monkeypatch.setattr(transnetv2, "SourceDownloader", LocalDownloader)
    monkeypatch.setattr(transnetv2, "decode_sources", decode_on_cpu)
    attempt = reconcile_and_select_attempt(resolve_bindings(config), lambda: discover_items(config))
    assert attempt.workspace is not None
    with checkpointed_processing(attempt) as processing:
        pipeline.cut_sources(processing, config)
    assert DataContext.get_current().checkpoint_config is None
    outcomes = staged_results(attempt.workspace.results)
    assert outcomes.schema.equals(resolve_bindings(config).spec.result_fields, check_metadata=True)
    rows = {row["source_uri"]: row for row in outcomes.to_pylist()}
    assert len(rows) == len(uris)
    assert rows["s3://test-bucket/missing.mp4"]["error_stage"] == "source-read"
    assert rows["s3://test-bucket/invalid.mp4"]["error_stage"] == "source-probe"
    assert sum(row["error_stage"] is None for row in rows.values()) == 4
    assert all(len(row["spans"]) == (row["error_stage"] is None) for row in rows.values())


def assert_transnetv2_resume(config_json: str, log_dir: str, video_path: str) -> None:
    """Run the shared fresh-cluster scenario through the download, decode, and inference stages."""
    config = _without_gpus(ResolvedVideoCutConfig.model_validate_json(config_json))
    logs = Path(log_dir)
    payload = Path(video_path).read_bytes()
    steps = ("download", "decode", "infer")

    def record(step: str, source_ids: list[str]) -> None:
        with (logs / f"{step}.txt").open("a") as log:
            log.writelines(f"{source_id}\n" for source_id in source_ids)

    class RecordingDownloader(transnetv2.SourceDownloader):
        def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
            record("download", [row["source_id"]])
            if row["source_uri"].endswith("error.mp4"):
                return row | {"video_bytes": b"", "error_stage": "source-read", "error_message": "synthetic missing"}
            return row | {"video_bytes": payload, "error_stage": None, "error_message": None}

    decode = transnetv2.decode_sources

    def recording_decode(batch: pa.Table, *, execution: ExecutionConfig) -> pa.Table:
        record("decode", batch["source_id"].to_pylist())
        return decode(batch, execution=execution)

    class RecordingPlanner(_CPUPlanner):
        def __call__(self, batch: pa.Table) -> pa.Table:
            record("infer", batch["source_id"].to_pylist())
            return super().__call__(batch)

    def work_logs() -> dict[str, list[str]]:
        return {
            step: (logs / f"{step}.txt").read_text().splitlines() if (logs / f"{step}.txt").exists() else []
            for step in steps
        }

    transnetv2.SourceDownloader = RecordingDownloader
    transnetv2.decode_sources = recording_decode
    transnetv2.TransNetV2Planner = RecordingPlanner
    transnetv2.validate_weights = lambda: Path("/config/models/unused.pth")
    transnetv2.ray_data_gpu_runtime_env = lambda _: {}
    # The shared retune revalidates, which would restore the GPU reservation.
    recipe = attrs.evolve(RECIPE, retune=partial(_without_gpus, rows_per_checkpoint=3, rows_per_fragment=5))
    summary = assert_fresh_cluster_resume(recipe, config, process=pipeline.cut_sources, work_logs=work_logs, num_cpus=4)
    assert (summary["appended_rows"], summary["errors"]) == (4, 1)


@pytest.mark.integration
def test_fresh_cluster_resumes_scene_checkpoints(
    config: ResolvedVideoCutConfig, video_path: Path, tmp_path: Path
) -> None:
    """Each scene-cutting stage runs once per source across an interrupted run and a fresh cluster."""
    run_in_fresh_driver(
        "tests.cosmos_curator.next.recipes.video_cut.test_ray_transnetv2:assert_transnetv2_resume",
        _scene_config(config).model_dump_json(),
        str(tmp_path),
        str(video_path),
    )
