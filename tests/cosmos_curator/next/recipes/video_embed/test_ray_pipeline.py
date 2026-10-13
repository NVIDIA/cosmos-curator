# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual Ray checkpoint/publication plan with CPU model stand-ins."""

import io
import json
from pathlib import Path
from typing import Any

import av
import lance
import pyarrow as pa
import pytest
import ray

from cosmos_curator.core.utils import environment
from cosmos_curator.next.recipes.video_embed import pipeline
from cosmos_curator.next.recipes.video_embed.config import ResolvedVideoEmbedConfig
from cosmos_curator.next.recipes.video_embed.contracts import (
    ModelSpec,
    resolve_model_spec,
    terminal_row,
)
from cosmos_curator.next.recipes.video_embed.processing import download_clip
from tests.cosmos_curator.next.enrichment.fresh_cluster import assert_fresh_cluster_resume, run_in_fresh_driver
from tests.cosmos_curator.next.recipes.video_embed.conftest import RECIPE, write_clips
from tests.cosmos_curator.next.recipes.video_embed.test_processing import video_bytes


def recording_download(row: dict[str, Any], **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
    """Record actual media reads to prove checkpoint filtering happens before download."""
    assert ray.get_runtime_context().get_assigned_resources()[environment.CURATOR_IO_RESOURCE_NAME] == 1
    result = download_clip(row, **kwargs)
    with (Path(row["clip_uri"]).parent.parent / "downloads.jsonl").open("a") as log:
        log.write(json.dumps(row["clip_id"]) + "\n")
    return result


class CpuEmbedder:
    """Stand in for GPU inference while retaining terminal schemas and row identities."""

    def __init__(self, *, spec: ModelSpec, model_dir: str, staged_schema: pa.Schema) -> None:
        """Use the test media directory for an expensive-work log."""
        self._spec = spec
        self._staged_schema = staged_schema
        self._root = Path(model_dir)

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Emit normalized synthetic vectors only for decoded clips."""
        rows = []
        for row in batch.to_pylist():
            vector = None
            if row["error_type"] is None:
                with av.open(io.BytesIO(row["clip_bytes"])) as container:
                    assert sum(1 for _ in container.decode(video=0)) > 0
                vector = [1.0, *([0.0] * (self._spec.dimension - 1))]
            rows.append(terminal_row(row, self._spec, vector))
            with (self._root / "inference.jsonl").open("a") as log:
                log.write(json.dumps(row["clip_id"]) + "\n")
        return pa.Table.from_pylist(rows, schema=self._staged_schema)


def assert_resume(config_json: str) -> None:
    """Resume on a fresh cluster after durable results, including a caption schema race."""
    config = ResolvedVideoEmbedConfig.model_validate_json(config_json)
    spec = resolve_model_spec(config.model.variant)
    write_clips(config, payload=video_bytes(3))
    root = Path(config.input.media_root)
    map_batches = ray.data.Dataset.map_batches

    def cpu_batches(self: ray.data.Dataset, fn: Any, **kwargs: Any) -> ray.data.Dataset:  # noqa: ANN401
        if fn is CpuEmbedder:
            kwargs.pop("runtime_env", None)
            kwargs["num_gpus"] = 0
            kwargs["memory"] = 64 * 1024**2
        return map_batches(self, fn, **kwargs)

    def remove_unfinished_clip() -> None:
        # One unfinished clip disappears: it still needs a terminal row and checkpoint.
        clips = lance.dataset(config.input.clips_lance_uri)
        Path(clips.get_fragment(1).to_table(columns=["clip_uri"]).column(0)[-1].as_py()).unlink()

    def work_logs() -> dict[str, list[str]]:
        return {
            name: [json.loads(line) for line in (root / name).read_text().splitlines()]
            for name in ("downloads.jsonl", "inference.jsonl")
        }

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(pipeline, "download_clip", recording_download)
        monkeypatch.setattr(pipeline, "VideoEmbedder", CpuEmbedder)
        monkeypatch.setattr(pipeline, "validate_model_directory", lambda _spec: root)
        monkeypatch.setattr(ray.data.Dataset, "map_batches", cpu_batches)
        summary = assert_fresh_cluster_resume(
            RECIPE,
            config,
            process=lambda processing, resolved: pipeline.embed_clips(processing, resolved, spec),
            work_logs=work_logs,
            # Model actors and resume's checkpoint filter leave room for reads,
            # result repartitioning, and file writes.
            num_cpus=6,
            before_resume=remove_unfinished_clip,
        )
    published = lance.dataset(config.input.clips_lance_uri)
    assert summary["clips_lance_version"] == published.version
    assert published.to_table(columns=[spec.vector_field]).column(0).null_count == 1


@pytest.mark.integration
def test_checkpoint_resume_on_fresh_cluster(config: ResolvedVideoEmbedConfig) -> None:
    """Exercise IO-limited download tasks, model actors, and checkpoint recovery."""
    run_in_fresh_driver(
        "tests.cosmos_curator.next.recipes.video_embed.test_ray_pipeline:assert_resume", config.model_dump_json()
    )
