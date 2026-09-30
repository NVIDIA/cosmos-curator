# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Separate CPU decoding survives real Ray transport and native source checkpoints."""

from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import ray
import torch
from ray.data.context import DataContext

from cosmos_curator.next.recipes.video_cut import pipeline, transnetv2
from cosmos_curator.next.recipes.video_cut.config import (
    ExecutionConfig,
    ResolvedVideoCutConfig,
    TransNetV2Config,
    resolve_config_data,
)
from cosmos_curator.next.recipes.video_cut.contracts import SOURCE_SCHEMA
from cosmos_curator.next.recipes.video_cut.publication import resolve_workspace
from cosmos_curator.next.utils.identity import canonical_digest


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


@pytest.mark.integration
def test_separate_decode_transport_errors_and_checkpoint_reuse(
    curator_io_cluster: None,
    config: ResolvedVideoCutConfig,
    video_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decode tasks use their own CPUs; inference errors and successes retain source recovery boundaries."""
    del curator_io_cluster
    raw = config.model_dump(mode="json")
    raw["split"] = {"method": "transnetv2", "decoder": "pyav", "min_duration_s": 0.01}
    raw["input"]["uris"] = [
        "s3://test-bucket/missing.mp4",
        "s3://test-bucket/invalid.mp4",
        *[f"s3://test-bucket/video-{index}.mp4" for index in range(4)],
    ]
    config = resolve_config_data(
        raw,
        overrides=[
            "execution.transnetv2.cpus=1",
            "execution.transnetv2.memory_gib=0.25",
            "execution.transnetv2.inference_cpus=0.25",
            "execution.transnetv2.inference_memory_gib=0.25",
        ],
    )
    # The shared CPU cluster has no GPUs. Only the model and its GPU reservation
    # are replaced; production stage assembly, decode, schemas, and sinks execute.
    config = config.model_copy(
        update={
            "execution": config.execution.model_copy(
                update={"transnetv2": config.execution.transnetv2.model_copy(update={"gpus": 0})}
            )
        }
    )
    monkeypatch.setattr(pipeline, "ray_data_gpu_runtime_env", lambda _: {})
    monkeypatch.setattr(transnetv2, "TransNetV2Planner", _CPUPlanner)
    monkeypatch.setattr(DataContext.get_current(), "enable_progress_bars", False)
    forbidden = tmp_path / "forbid_downloads"
    payload = video_path.read_bytes()

    class LocalDownloader(transnetv2.SourceDownloader):
        def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
            assert not forbidden.exists(), "A checkpointed source reached downloading"
            resources = ray.get_runtime_context().get_assigned_resources()
            assert resources["CPU"] == 0.25
            assert resources["curator_io"] == 1
            if row["source_uri"].endswith("missing.mp4"):
                return row | {"video_bytes": b"", "download_error": "S3 source unavailable: 404"}
            data = b"invalid video" if row["source_uri"].endswith("invalid.mp4") else payload
            return row | {"video_bytes": data, "download_error": ""}

    decode = transnetv2.decode_sources

    def decode_on_cpu(batch: pa.Table, *, execution: ExecutionConfig) -> pa.Table:
        resources = ray.get_runtime_context().get_assigned_resources()
        assert resources["CPU"] == 1
        assert resources.get("GPU", 0) == 0
        assert "curator_io" not in resources
        return decode(batch, execution=execution)

    monkeypatch.setattr(transnetv2, "SourceDownloader", LocalDownloader)
    monkeypatch.setattr(transnetv2, "decode_sources", decode_on_cpu)
    snapshot_id = canonical_digest(config.input.uris)
    workspace = resolve_workspace(config, snapshot_id)
    pipeline.cut_sources(config, workspace, config.input.uris)
    outcomes = pq.read_table(workspace.results).sort_by("source_id")
    assert outcomes.schema.equals(SOURCE_SCHEMA)
    assert outcomes.num_rows == 6
    rows = {row["source_uri"]: row for row in outcomes.to_pylist()}
    assert rows["s3://test-bucket/missing.mp4"]["error_stage"] == "source-read"
    assert rows["s3://test-bucket/invalid.mp4"]["error_stage"] == "source-probe"
    assert sum(row["status"] == "success" for row in rows.values()) == 4
    assert all(len(row["spans"]) == (row["status"] == "success") for row in rows.values())

    forbidden.touch()
    pipeline.cut_sources(config, workspace, config.input.uris)
    assert pq.read_table(workspace.results).sort_by("source_id").equals(outcomes)
    assert DataContext.get_current().checkpoint_config is None
    published = pipeline.publish_spans(config, workspace, input_snapshot_id=snapshot_id, source_uris=config.input.uris)
    assert published.count_rows() == 4
    assert len(set(published.to_table(columns=["span_id"])["span_id"].to_pylist())) == 4
