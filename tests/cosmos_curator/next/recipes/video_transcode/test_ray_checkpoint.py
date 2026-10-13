# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A real Ray read/group/upload/Parquet plan resumed on a fresh local cluster."""

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import ray
from ray._private.state import actors
from ray.data.context import DataContext

from cosmos_curator.core.utils import environment
from cosmos_curator.next.enrichment.attempts import reconcile_and_select_attempt
from cosmos_curator.next.enrichment.execution import ProcessingScope, checkpointed_processing
from cosmos_curator.next.enrichment.inputs import open_enrichment_input
from cosmos_curator.next.recipes.video_cut.contracts import make_source_id, make_span_id
from cosmos_curator.next.recipes.video_transcode import pipeline, processing
from cosmos_curator.next.recipes.video_transcode.bindings import resolve_bindings
from cosmos_curator.next.recipes.video_transcode.config import ResolvedVideoTranscodeConfig
from cosmos_curator.next.recipes.video_transcode.contracts import make_clip_id
from tests.cosmos_curator.next.enrichment.fresh_cluster import assert_fresh_cluster_resume, run_in_fresh_driver
from tests.cosmos_curator.next.recipes.video_transcode.conftest import RECIPE


@pytest.mark.integration
def test_native_span_resume_across_source_and_fragment_boundaries(
    config: ResolvedVideoTranscodeConfig, dataset: lance.LanceDataset
) -> None:
    """A fresh cluster filters durable spans without replacing another test's cluster."""
    del dataset
    run_in_fresh_driver(
        "tests.cosmos_curator.next.recipes.video_transcode.test_ray_checkpoint:assert_span_resume",
        config.model_dump_json(),
    )


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


def _upload_without_io_slot(
    batch: pa.Table, *, config: ResolvedVideoTranscodeConfig, staged_schema: pa.Schema
) -> pa.Table:
    """Exercise real uploads while checking that only downloads reserve IO slots."""
    resources = ray.get_runtime_context().get_assigned_resources()
    assert resources["CPU"] == 0.25
    assert environment.CURATOR_IO_RESOURCE_NAME not in resources
    results = processing.upload_clips(batch, config=config, staged_schema=staged_schema)
    assert results.schema.equals(staged_schema, check_metadata=True)
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


def _log(media_root: Path, name: str) -> list[Any]:
    return [json.loads(line) for line in (media_root / name).read_text().splitlines()]


def _transcode_preserving_context(scope: ProcessingScope, config: ResolvedVideoTranscodeConfig) -> None:
    """Run the media plan and require its shuffle and autoscaling settings to stay plan-scoped."""
    context = DataContext.get_current()
    shuffle_strategy = context.shuffle_strategy
    scaling_threshold = context.autoscaling_config.actor_pool_util_upscaling_threshold
    pipeline.transcode_spans(scope, config)
    assert DataContext.get_current().shuffle_strategy == shuffle_strategy
    assert DataContext.get_current().autoscaling_config.actor_pool_util_upscaling_threshold == scaling_threshold


def assert_span_resume(config_json: str) -> None:
    """Assert a fresh cluster filters durable spans before regrouping an unfinished source."""
    config = ResolvedVideoTranscodeConfig.model_validate_json(config_json)
    media_root = Path(config.input.media_root)

    def work_logs() -> dict[str, list[str]]:
        return {
            "download": [span_id for line in _log(media_root, "downloaded.jsonl") for span_id in line["span_ids"]],
            "transcode": [span_id for group in _log(media_root, "processed.jsonl") for span_id in group],
        }

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(pipeline, "SourceDownloader", _RecordingDownloader)
        monkeypatch.setattr(pipeline, "transcode_source", _synthetic_transcode)
        monkeypatch.setattr(pipeline, "upload_clips", _upload_without_io_slot)
        assert_fresh_cluster_resume(
            RECIPE,
            config,
            process=_transcode_preserving_context,
            work_logs=work_logs,
            # Leave room for checkpoint actors, the persistent download actor,
            # and the sort's sampling tasks while the checkpoint pool scales down.
            num_cpus=4,
        )
    # Each cluster downloaded the one source once, for just the spans it still needed.
    work = _log(media_root, "processed.jsonl")
    assert [len(group) for group in work] == [2, 2]
    assert set(work[0]).isdisjoint(work[1])
    assert [download["span_ids"] for download in _log(media_root, "downloaded.jsonl")] == work
    published = lance.dataset(config.input.spans_lance_uri)
    for row in published.to_table(columns=["clip_uri", "clip_size_bytes"]).to_pylist():
        assert Path(row["clip_uri"]).read_bytes() == b"clip"
        assert row["clip_size_bytes"] == 4


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
    dataset = lance.write_dataset(
        pa.Table.from_pylist(rows, schema=original.schema),
        dataset.uri,
        mode="overwrite",
        data_storage_version="2.2",
    )
    config = config.model_copy(update={"execution": config.execution.model_copy(update={"source_partitions": 1})})
    monkeypatch.setattr(pipeline, "SourceDownloader", _StreamingDownloader)
    monkeypatch.setattr(pipeline, "transcode_source", _synthetic_transcode)
    monkeypatch.setattr(pipeline, "upload_clips", _upload_without_io_slot)
    attempt = reconcile_and_select_attempt(open_enrichment_input(resolve_bindings(config)))
    assert attempt.workspace is not None
    with checkpointed_processing(attempt) as processing:
        pipeline.transcode_spans(processing, config)
    downloads = [
        json.loads(line) for line in Path(config.input.media_root, "downloaded.jsonl").read_text().splitlines()
    ]
    assert len(downloads) == 3
    assert len({download["task_id"] for download in downloads}) == 1
    assert len({download["actor_id"] for download in downloads}) == 1
    assert all(len(download["span_ids"]) == 1 for download in downloads)
    results = pq.read_table(attempt.workspace.results)
    assert results.num_rows == 3
    assert set(results["span_id"].to_pylist()) == {row["span_id"] for row in rows}
