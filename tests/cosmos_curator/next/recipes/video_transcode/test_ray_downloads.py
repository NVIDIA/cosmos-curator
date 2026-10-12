# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download admission and worker churn on an isolated CPU-only Ray cluster."""

import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import suppress
from pathlib import Path
from typing import Any, ClassVar

import lance
import pyarrow as pa
import pytest
import ray
from coverage import Coverage
from ray.cluster_utils import Cluster
from ray.data._internal.execution.execution_callback import ExecutionCallback
from ray.data._internal.execution.streaming_executor import StreamingExecutor
from ray.data.context import DataContext

from cosmos_curator.core.utils import environment
from cosmos_curator.next.recipes.video_cut.contracts import make_source_id, make_span_id
from cosmos_curator.next.recipes.video_transcode import pipeline, processing
from cosmos_curator.next.recipes.video_transcode.config import ResolvedVideoTranscodeConfig
from cosmos_curator.next.recipes.video_transcode.contracts import completed_publication, make_clip_id
from tests.cosmos_curator.next.recipes.video_transcode.test_ray_checkpoint import (
    _synthetic_transcode,
    _upload_without_io_slot,
)

# Own the head and every worker in a child; never replace the suite's shared Ray
# cluster. The parent captures expected node-loss diagnostics, including under -s.
_DRIVER = """
import sys
import coverage

cov = coverage.Coverage(data_file=sys.argv[2], data_suffix=True) if sys.argv[2] else None
if cov is not None:
    cov.start()
try:
    import pytest
    from cosmos_curator.next.recipes.video_transcode.config import ResolvedVideoTranscodeConfig
    from tests.cosmos_curator.next.recipes.video_transcode.test_ray_downloads import _assert_download_churn

    config = ResolvedVideoTranscodeConfig.model_validate_json(sys.argv[1])
    with pytest.MonkeyPatch.context() as monkeypatch:
        _assert_download_churn(config, monkeypatch)
finally:
    if cov is not None:
        cov.stop()
        cov.save()
"""


class _BlockedDownloader(processing.SourceDownloader):
    """Hold each actor's first task so excess submissions cannot hide behind fast IO."""

    def __call__(self, source: dict[str, Any]) -> dict[str, Any]:
        """Report the worker incarnation, then wait for the controller to release reads."""
        root = Path(self._config.input.media_root)
        context = ray.get_runtime_context()
        assert context.get_assigned_resources()[environment.CURATOR_IO_RESOURCE_NAME] == 1
        (root / f"started-{context.get_node_id()}-{context.get_actor_id()}").touch()
        deadline = time.monotonic() + 180
        while not (root / "release-downloads").exists():
            assert time.monotonic() < deadline, "Controller did not release downloads"
            time.sleep(0.05)
        return super().__call__(source)


class _DownloadObserver(ExecutionCallback):
    """Observe admission in the driver rather than merely counting started UDFs."""

    snapshots: ClassVar[list[dict[str, int]]] = []

    def on_execution_step(self, executor: StreamingExecutor) -> None:
        """Record the pinned Ray executor's submitted tasks and ready/pending actors."""
        for op, state in (executor._topology or {}).items():
            if "_BlockedDownloader" in op.name:
                actors = op.get_actor_info()
                self.snapshots.append(
                    {
                        "submitted": op.metrics.num_tasks_submitted,
                        "in_flight": op.metrics.num_tasks_running,
                        "ready": actors.running,
                        "pending": actors.pending,
                        "queued": state.total_enqueued_input_blocks(),
                    }
                )


def _wait_for(predicate: Callable[[], bool], future: Future[dict[str, object]], *, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if future.done():
            future.result()
            pytest.fail("Pipeline completed before worker-churn assertions")
        assert time.monotonic() < deadline, f"Timed out waiting for actor capacity: {_DownloadObserver.snapshots[-5:]}"
        time.sleep(0.1)


def _wait_for_capacity(capacity: int, future: Future[dict[str, object]]) -> None:
    def reached_capacity() -> bool:
        if not _DownloadObserver.snapshots:
            return False
        latest = _DownloadObserver.snapshots[-1]
        return latest["ready"] == latest["in_flight"] == capacity and latest["queued"] > 0

    _wait_for(reached_capacity, future)
    # Several executor iterations with a saturated pool must not submit work to
    # pending actors or turn the task backlog into an actor-creation backlog.
    start = len(_DownloadObserver.snapshots)
    _wait_for(lambda: len(_DownloadObserver.snapshots) >= start + 5, future)
    for sample in _DownloadObserver.snapshots[start:]:
        assert sample["submitted"] == sample["in_flight"] == capacity
        assert sample["ready"] == capacity
        assert sample["pending"] <= 1


def _run_pipeline(config: ResolvedVideoTranscodeConfig, future: Future[dict[str, object]]) -> None:
    try:
        future.set_result(pipeline.run_config(config))
    except Exception as exc:  # noqa: BLE001 -- propagate the driver's original error to the controller
        future.set_exception(exc)


def _assert_download_churn(config: ResolvedVideoTranscodeConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Late workers add capacity; a period with none preserves the eventual publication."""
    cluster = Cluster()
    root = Path(config.input.media_root)
    worker_args = {
        "num_cpus": 4,
        "num_gpus": 0,
        "resources": {environment.CURATOR_IO_RESOURCE_NAME: 1},
        # Logical capacity, not an allocation: leave room for Ray Data's default
        # per-operator heap reservations so IO is the binding resource.
        "memory": 16 * 1024**3,
        "object_store_memory": 128 * 1024**2,
    }
    monkeypatch.setattr(pipeline, "SourceDownloader", _BlockedDownloader)
    monkeypatch.setattr(pipeline, "transcode_source", _synthetic_transcode)
    monkeypatch.setattr(pipeline, "upload_clips", _upload_without_io_slot)
    future: Future[dict[str, object]] | None = None
    try:
        cluster.add_node(num_cpus=0, num_gpus=0, include_dashboard=False, object_store_memory=128 * 1024**2)
        first = cluster.add_node(**worker_args)
        ray.init(address=cluster.address, log_to_driver=False)
        context = DataContext.get_current()
        # Match the small test object store: otherwise Ray's speculative output
        # budget, not IO capacity, prevents a second actor before any task yields.
        context.target_max_block_size = 1024**2
        context.custom_execution_callback_classes = [_DownloadObserver]
        _DownloadObserver.snapshots.clear()
        future = Future()
        threading.Thread(target=_run_pipeline, args=(config, future), daemon=True).start()
        _wait_for_capacity(1, future)

        second = cluster.add_node(**worker_args)
        _wait_for_capacity(2, future)
        _wait_for(lambda: len(list(root.glob("started-*"))) == 2, future)

        cluster.remove_node(first, allow_graceful=False)
        cluster.remove_node(second, allow_graceful=False)
        _wait_for(lambda: ray.cluster_resources().get(environment.CURATOR_IO_RESOURCE_NAME, 0) == 0, future)
        start = len(_DownloadObserver.snapshots)
        _wait_for(lambda: len(_DownloadObserver.snapshots) >= start + 5, future)
        assert not future.done()
        assert len(list(root.glob("started-*"))) == 2
        assert all(sample["submitted"] == 2 for sample in _DownloadObserver.snapshots[start:])

        cluster.add_node(**worker_args)
        _wait_for(lambda: len(list(root.glob("started-*"))) > 2, future)
        (root / "release-downloads").touch()
        summary = future.result(timeout=120)
        assert summary["input_rows"] == 8
        assert summary["cleanup_complete"] is True
    finally:
        (root / "release-downloads").touch()
        if future is not None and not future.done():
            # Drain the driver before disconnecting Ray, including on assertions;
            # it must not auto-initialize another cluster during publication.
            if not cluster.worker_nodes:
                cluster.add_node(**worker_args)
            with suppress(Exception):
                future.result(timeout=30)
        ray.shutdown()
        cluster.shutdown()


@pytest.mark.integration
def test_download_submission_tracks_live_io_capacity_and_recovers(
    config: ResolvedVideoTranscodeConfig, dataset: lance.LanceDataset
) -> None:
    """A late worker contributes and replacement actors finish the same logical spans after complete worker loss."""
    table = dataset.to_table()
    template = table.to_pylist()[0]
    rows = []
    for index in range(8):
        source = Path(config.input.media_root, f"source-{index}.mp4")
        # Equal-sized sources must remain distributed so late actors can contribute.
        source.write_bytes(b"source-video")
        source_id = make_source_id(str(source))
        rows.append(
            template
            | {
                "source_id": source_id,
                "source_uri": str(source),
                "source_size_bytes": source.stat().st_size,
                "span_id": make_span_id(source_id, template["start_ns"], template["end_ns"]),
            }
        )
    dataset = lance.write_dataset(
        pa.Table.from_pylist(rows, schema=table.schema),
        dataset.uri,
        mode="overwrite",
        max_rows_per_file=2,
        data_storage_version="2.2",
    )
    config = config.model_copy(update={"execution": config.execution.model_copy(update={"source_partitions": 8})})
    active_coverage = Coverage.current()
    coverage_file = active_coverage.get_option("run:data_file") if active_coverage is not None else ""
    result = subprocess.run(  # noqa: S603 -- fixed child driver with test-owned config
        [sys.executable, "-c", _DRIVER, config.model_dump_json(), coverage_file],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    published = lance.dataset(dataset.uri)
    assert completed_publication(published) is not None
    output = published.to_table()
    assert output.num_rows == len(rows)
    assert output["span_id"].to_pylist() == [row["span_id"] for row in rows]
    assert len(set(output["clip_id"].to_pylist())) == len(rows)
    for row in output.to_pylist():
        assert row["clip_id"] == make_clip_id(row["span_id"], config.transcode.contract_id)
        assert row["clip_size_bytes"] == 4
        assert Path(row["clip_uri"]).read_bytes() == b"clip"
