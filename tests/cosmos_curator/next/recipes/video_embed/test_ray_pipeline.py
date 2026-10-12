# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual Ray checkpoint/publication plan with CPU model stand-ins."""

import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import attrs
import av
import lance
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import ray
from coverage import Coverage
from ray.data.context import DataContext

from cosmos_curator.core.utils import environment
from cosmos_curator.next.enrichment.attempts import reconcile_and_select_attempt
from cosmos_curator.next.enrichment.contracts import open_enrichment_input
from cosmos_curator.next.enrichment.execution import checkpointed_processing, publish_enrichment
from cosmos_curator.next.recipes.video_embed import pipeline
from cosmos_curator.next.recipes.video_embed.bindings import resolve_bindings
from cosmos_curator.next.recipes.video_embed.config import ResolvedVideoEmbedConfig
from cosmos_curator.next.recipes.video_embed.contracts import (
    ModelSpec,
    resolve_model_spec,
    terminal_row,
)
from cosmos_curator.next.recipes.video_embed.processing import download_clip
from tests.cosmos_curator.next.recipes.video_embed.conftest import write_clips
from tests.cosmos_curator.next.recipes.video_embed.test_processing import video_bytes

_DRIVER = """
import sys
import coverage
cov = coverage.Coverage(data_file=sys.argv[2], data_suffix=True) if sys.argv[2] else None
if cov:
    cov.start()
try:
    from cosmos_curator.next.recipes.video_embed.config import ResolvedVideoEmbedConfig
    from tests.cosmos_curator.next.recipes.video_embed.test_ray_pipeline import assert_resume
    assert_resume(ResolvedVideoEmbedConfig.model_validate_json(sys.argv[1]))
finally:
    if cov:
        cov.stop()
        cov.save()
"""


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


def _start_ray() -> None:
    ray.init(
        address="local",
        # Model actors and resume's checkpoint filter leave room for reads,
        # result repartitioning, and file writes.
        num_cpus=6,
        num_gpus=0,
        resources={environment.CURATOR_IO_RESOURCE_NAME: 1},
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=128 * 1024**2,
    )
    DataContext.get_current().enable_progress_bars = False


def assert_resume(config: ResolvedVideoEmbedConfig) -> None:  # noqa: PLR0915 -- one fresh-cluster recovery scenario
    """Resume on a fresh cluster after durable results, including a caption schema race."""
    spec = resolve_model_spec(config.model.variant)
    write_clips(config, payload=video_bytes(3))
    attempt = reconcile_and_select_attempt(open_enrichment_input(resolve_bindings(config)))
    workspace = attempt.workspace
    assert workspace is not None
    map_batches = ray.data.Dataset.map_batches

    def cpu_batches(self: ray.data.Dataset, fn: Any, **kwargs: Any) -> ray.data.Dataset:  # noqa: ANN401
        if fn is CpuEmbedder:
            kwargs.pop("runtime_env", None)
            kwargs["num_gpus"] = 0
            kwargs["memory"] = 64 * 1024**2
        return map_batches(self, fn, **kwargs)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(pipeline, "download_clip", recording_download)
        monkeypatch.setattr(pipeline, "VideoEmbedder", CpuEmbedder)
        monkeypatch.setattr(pipeline, "validate_model_directory", lambda _spec: Path(config.input.media_root))
        monkeypatch.setattr(ray.data.Dataset, "map_batches", cpu_batches)
        try:
            _start_ray()
            # An interrupted run leaves only the first fragment's clips durable.
            with checkpointed_processing(attrs.evolve(attempt, selected=attempt.selected[:1])) as processing:
                pipeline.embed_clips(processing, config, spec)
            assert pq.read_table(workspace.results).num_rows == 2
            assert DataContext.get_current().checkpoint_config is None
            dataset = attempt.input.dataset
            assert lance.dataset(dataset.uri).to_table(columns=[spec.vector_field]).column(0).null_count == 4
            # Simulate a crash after a subset of shards is durable, before a complete inference marker.
            Path(workspace.completion).unlink()
            ray.shutdown()

            # A disjoint schema change and execution tuning must not invalidate saved GPU work.
            latest = lance.dataset(dataset.uri)
            latest.add_columns(pa.schema([pa.field("caption_test_v1", pa.string())]))
            tuned = config.model_copy(
                update={
                    "execution": config.execution.model_copy(
                        update={"parquet_rows_per_file": 2, "inference_batch_size": 2}
                    )
                }
            )
            resumed_attempt = reconcile_and_select_attempt(open_enrichment_input(resolve_bindings(tuned)))
            assert (resumed_attempt.selected, resumed_attempt.workspace) == (attempt.selected, workspace)
            resumed = resumed_attempt.input.dataset
            # One unfinished clip disappears: it still needs a terminal row and checkpoint.
            final_uri = resumed.get_fragment(1).to_table(columns=["clip_uri"]).column(0)[-1].as_py()
            Path(final_uri).unlink()
            _start_ray()
            with checkpointed_processing(resumed_attempt) as processing:
                pipeline.embed_clips(processing, tuned, spec)
            rows = pq.read_table(workspace.results)
            assert rows.num_rows == len(set(rows["clip_id"].to_pylist())) == 4
            for name in ("downloads.jsonl", "inference.jsonl"):
                log = Path(config.input.media_root, name).read_text().splitlines()
                assert len(log) == len(set(log)) == 4
            assert resumed_attempt.processing_complete()
            outcome = publish_enrichment(resumed_attempt)
            published = outcome.dataset
            assert published.version == resumed.version + 1
            assert outcome.cleanup_complete
            assert published.to_table(columns=[spec.vector_field]).column(0).null_count == 1
            assert "caption_test_v1" in published.schema.names
        finally:
            ray.shutdown()


@pytest.mark.integration
def test_checkpoint_resume_on_fresh_cluster(config: ResolvedVideoEmbedConfig) -> None:
    """Exercise IO-limited download tasks, model actors, and checkpoint recovery."""
    active_coverage = Coverage.current()
    coverage_file = active_coverage.get_option("run:data_file") if active_coverage else ""
    result = subprocess.run(  # noqa: S603 -- fixed test-owned subprocess driver
        [sys.executable, "-c", _DRIVER, config.model_dump_json(), coverage_file],
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
