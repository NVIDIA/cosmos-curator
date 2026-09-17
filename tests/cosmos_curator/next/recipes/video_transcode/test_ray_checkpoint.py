# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A real Ray read/group/upload/Parquet plan resumed on a fresh local cluster."""

import json
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import ray
from coverage import Coverage
from ray._private.state import actors
from ray.data.context import DataContext

from cosmos_curator.core.utils import environment
from cosmos_curator.next.recipes.video_cut.contracts import make_source_id, make_span_id
from cosmos_curator.next.recipes.video_transcode import pipeline, processing
from cosmos_curator.next.recipes.video_transcode.config import ResolvedVideoTranscodeConfig
from cosmos_curator.next.recipes.video_transcode.contracts import (
    RESULT_SCHEMA,
    make_clip_id,
)
from cosmos_curator.next.recipes.video_transcode.publication import resolve_workspace

# Own both Ray clusters in a child so restarting them cannot replace the suite's
# shared cluster. Save child coverage alongside pytest-cov's data when enabled.
_DRIVER = """
import sys
import coverage

cov = coverage.Coverage(data_file=sys.argv[3], data_suffix=True) if sys.argv[3] else None
if cov is not None:
    cov.start()
try:
    import lance
    import pytest
    from cosmos_curator.next.recipes.video_transcode.config import ResolvedVideoTranscodeConfig
    from tests.cosmos_curator.next.recipes.video_transcode.test_ray_checkpoint import _assert_span_resume

    config = ResolvedVideoTranscodeConfig.model_validate_json(sys.argv[1])
    with pytest.MonkeyPatch.context() as monkeypatch:
        _assert_span_resume(config, lance.dataset(sys.argv[2]), monkeypatch)
finally:
    if cov is not None:
        cov.stop()
        cov.save()
"""


@pytest.mark.integration
def test_native_span_resume_across_source_and_fragment_boundaries(
    config: ResolvedVideoTranscodeConfig, dataset: lance.LanceDataset
) -> None:
    """A fresh cluster filters durable spans without replacing another test's cluster."""
    active_coverage = Coverage.current()
    coverage_file = active_coverage.get_option("run:data_file") if active_coverage is not None else ""
    result = subprocess.run(  # noqa: S603 -- fixed driver with test-owned configuration
        [sys.executable, "-c", _DRIVER, config.model_dump_json(), str(dataset.uri), coverage_file],
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr


class _RecordingDownloader(processing.SourceDownloader):
    """Exercise the real download actor with one source and the IO reservation."""

    def __call__(self, source: dict[str, Any]) -> dict[str, Any]:
        """Record actor work after checking that the grouping shuffle released its resources."""
        # Query GCS directly because these test clusters run without a dashboard.
        assert not any(
            actor["ActorClassName"] == "HashShuffleAggregator" and actor["State"] == "ALIVE"
            for actor in actors().values()
        )
        resources = ray.get_runtime_context().get_assigned_resources()
        assert resources["CPU"] == 0.25
        assert resources[environment.CURATOR_IO_RESOURCE_NAME] == 1
        downloaded = super().__call__(source)
        with Path(self._config.input.media_root, "downloaded.jsonl").open("a") as log:
            log.write(
                json.dumps(
                    {
                        "actor_id": ray.get_runtime_context().get_actor_id(),
                        "task_id": ray.get_runtime_context().get_task_id(),
                        "span_ids": [row["span_id"] for row in source["spans"]],
                    }
                )
                + "\n"
            )
        return downloaded


class _StreamingDownloader(_RecordingDownloader):
    """Hold later downloads until the first output reaches the next stage."""

    def __call__(self, source: dict[str, Any]) -> dict[str, Any]:
        """Require the actor task to stream before completing its input partition."""
        media_root = Path(self._config.input.media_root)
        if (media_root / "downloaded.jsonl").exists():
            deadline = time.monotonic() + 30
            while not (media_root / "processed.jsonl").exists():
                assert time.monotonic() < deadline, (
                    "Transcoding did not start while the download task was still running"
                )
                time.sleep(0.05)
        return super().__call__(source)


def _upload_without_io_slot(batch: pa.Table, *, config: ResolvedVideoTranscodeConfig) -> pa.Table:
    """Exercise real uploads while checking that only downloads reserve IO slots."""
    resources = ray.get_runtime_context().get_assigned_resources()
    assert resources["CPU"] == 0.25
    assert environment.CURATOR_IO_RESOURCE_NAME not in resources
    results = processing.upload_clips(batch, config=config)
    assert results.schema.equals(RESULT_SCHEMA, check_metadata=True)
    return results


def _synthetic_transcode(source: dict[str, Any], *, config: ResolvedVideoTranscodeConfig) -> Iterator[dict[str, Any]]:
    """Record source work and emit synthetic bytes while retaining the production DAG."""
    resources = ray.get_runtime_context().get_assigned_resources()
    assert resources["CPU"] == config.execution.transcode_cpus
    assert resources["memory"] == int(config.execution.transcode_memory_gib * 1024**3)
    assert environment.CURATOR_IO_RESOURCE_NAME not in resources
    spans = source["spans"]
    assert source["source_bytes"] == Path(source["source_uri"]).read_bytes()
    with Path(config.input.media_root, "processed.jsonl").open("a") as log:
        log.write(json.dumps([row["span_id"] for row in spans]) + "\n")
    for row in spans:
        clip_id = make_clip_id(row["span_id"], config.transcode.contract_id)
        yield {
            "fragment_id": row["fragment_id"],
            "row_offset": row["row_offset"],
            "span_id": row["span_id"],
            "transcode_contract_id": config.transcode.contract_id,
            "clip_id": clip_id,
            "clip_uri": f"{config.input.media_root}/clips/{clip_id}.mp4",
            "clip_size_bytes": 4,
            "clip_duration_ns": 1_000_000_000,
            "clip_width": 64,
            "clip_height": 64,
            "clip_frame_rate": 10.0,
            "clip_frame_count": 10,
            "clip_video_codec": "h264",
            "clip_bytes": b"clip",
        }


def _start_ray() -> None:
    ray.init(
        address="local",
        # Leave room for checkpoint actors, the persistent download actor,
        # and the sort's sampling tasks while the checkpoint pool scales down.
        num_cpus=4,
        num_gpus=0,
        resources={environment.CURATOR_IO_RESOURCE_NAME: 1},
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=128 * 1024**2,
    )
    DataContext.get_current().enable_progress_bars = False


def _assert_span_resume(
    config: ResolvedVideoTranscodeConfig, dataset: lance.LanceDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assert a fresh cluster filters durable spans before regrouping an unfinished source."""
    workspace = resolve_workspace(config, dataset)
    datasource = pipeline.SpanDatasource

    class FirstFragment(datasource):
        def __init__(self, ds: lance.LanceDataset, cfg: ResolvedVideoTranscodeConfig) -> None:
            super().__init__(ds, cfg)
            self._fragments = self._fragments[:1]

    monkeypatch.setattr(pipeline, "SourceDownloader", _RecordingDownloader)
    monkeypatch.setattr(pipeline, "transcode_source", _synthetic_transcode)
    monkeypatch.setattr(pipeline, "upload_clips", _upload_without_io_slot)
    monkeypatch.setattr(pipeline, "SpanDatasource", FirstFragment)
    try:
        _start_ray()
        shuffle_strategy = DataContext.get_current().shuffle_strategy
        scaling_threshold = DataContext.get_current().autoscaling_config.actor_pool_util_upscaling_threshold
        pipeline.transcode_spans(dataset, config, workspace)
        assert DataContext.get_current().checkpoint_config is None
        assert DataContext.get_current().shuffle_strategy == shuffle_strategy
        assert DataContext.get_current().autoscaling_config.actor_pool_util_upscaling_threshold == scaling_threshold
        assert pq.read_table(workspace.results).num_rows == 2
        assert lance.dataset(dataset.uri).to_table(columns=["clip_uri"])["clip_uri"].null_count == 4
        ray.shutdown()

        monkeypatch.setattr(pipeline, "SpanDatasource", datasource)
        _start_ray()
        resumed = config.model_copy(
            update={
                "execution": config.execution.model_copy(
                    update={"spans_per_checkpoint": 2, "transcode_memory_gib": 0.75}
                )
            }
        )
        assert resolve_workspace(resumed, dataset) == workspace
        pipeline.transcode_spans(dataset, resumed, workspace)
        assert DataContext.get_current().checkpoint_config is None
        results = pq.read_table(workspace.results)
        assert results.schema.equals(RESULT_SCHEMA, check_metadata=True)
        assert results.num_rows == len(set(results["span_id"].to_pylist())) == 4
        work = [json.loads(line) for line in Path(config.input.media_root, "processed.jsonl").read_text().splitlines()]
        assert [len(group) for group in work] == [2, 2]
        assert set(work[0]).isdisjoint(work[1])
        downloads = [
            json.loads(line) for line in Path(config.input.media_root, "downloaded.jsonl").read_text().splitlines()
        ]
        assert [download["span_ids"] for download in downloads] == work
        pipeline.transcode_spans(dataset, resumed, workspace)
        assert len(Path(config.input.media_root, "processed.jsonl").read_text().splitlines()) == 2
        assert len(Path(config.input.media_root, "downloaded.jsonl").read_text().splitlines()) == 2
        published = pipeline.publish_results(dataset, config, workspace)
        assert published.version == dataset.version + 1
        for row in published.to_table(columns=["clip_uri", "clip_size_bytes"]).to_pylist():
            assert Path(row["clip_uri"]).read_bytes() == b"clip"
            assert row["clip_size_bytes"] == 4
    finally:
        ray.shutdown()


@pytest.mark.integration
def test_sources_in_one_partition_stream_from_one_download_task(
    config: ResolvedVideoTranscodeConfig,
    dataset: lance.LanceDataset,
    curator_io_cluster: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One task streams sources to transcoding before finishing its input partition."""
    del curator_io_cluster
    # Keep metadata in one block, but make each video large enough to flush Ray's
    # output buffer without downloading hundreds of MiB in this test.
    monkeypatch.setattr(DataContext.get_current(), "target_max_block_size", 64 * 1024)
    original = dataset.to_table()
    template = original.to_pylist()[0]
    rows = []
    for index in range(3):
        source = Path(config.input.media_root, f"source-{index}.mp4")
        source.write_bytes(b"source-video".ljust(128 * 1024, b"\0"))
        source_id = make_source_id(str(source))
        rows.append(
            template
            | {
                "source_uri": str(source),
                "source_id": source_id,
                "source_size_bytes": source.stat().st_size,
                "span_id": make_span_id(source_id, template["start_ns"], template["end_ns"]),
            }
        )
    metadata = dict(original.schema.metadata or {})
    metadata[b"video_cut.publication"] = json.dumps({"publication_version": 2, "spans": len(rows)}).encode()
    dataset = lance.write_dataset(
        pa.Table.from_pylist(rows, schema=original.schema.with_metadata(metadata)),
        dataset.uri,
        mode="overwrite",
        data_storage_version="2.2",
    )
    config = config.model_copy(update={"execution": config.execution.model_copy(update={"source_partitions": 1})})
    monkeypatch.setattr(pipeline, "SourceDownloader", _StreamingDownloader)
    monkeypatch.setattr(pipeline, "transcode_source", _synthetic_transcode)
    monkeypatch.setattr(pipeline, "upload_clips", _upload_without_io_slot)
    workspace = resolve_workspace(config, dataset)
    pipeline.transcode_spans(dataset, config, workspace)
    downloads = [
        json.loads(line) for line in Path(config.input.media_root, "downloaded.jsonl").read_text().splitlines()
    ]
    assert len(downloads) == 3
    assert len({download["task_id"] for download in downloads}) == 1
    assert len({download["actor_id"] for download in downloads}) == 1
    assert all(len(download["span_ids"]) == 1 for download in downloads)
    results = pq.read_table(workspace.results)
    assert results.num_rows == 3
    assert set(results["span_id"].to_pylist()) == {row["span_id"] for row in rows}
