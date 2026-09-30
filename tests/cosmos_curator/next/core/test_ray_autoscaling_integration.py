# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IO-limited download pools shrink under backpressure and grow when demand returns."""

import time
from pathlib import Path
from typing import Any, ClassVar

import pytest
import ray
from ray.data import ActorPoolStrategy
from ray.data._internal.execution.execution_callback import ExecutionCallback
from ray.data._internal.execution.streaming_executor import StreamingExecutor

from cosmos_curator.next.core.ray_autoscaling import configure_download_actor_autoscaling


def _wait_for_file(path: Path) -> None:
    deadline = time.monotonic() + 90
    while not path.exists():
        assert time.monotonic() < deadline, f"Controller did not release {path.name}"
        time.sleep(0.02)


class _Download:
    def __init__(self, root: str) -> None:
        self._root = Path(root)

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        _wait_for_file(self._root / "downloads")
        time.sleep(0.05)
        return row | {"video_bytes": b"x" * (256 * 1024)}


class _Consumer:
    def __init__(self, root: str) -> None:
        self._root = Path(root)

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        _wait_for_file(self._root / "consumer")
        return {"id": row["id"]}


class _ObservePool(ExecutionCallback):
    root: ClassVar[Path]
    saw_pending_with_idle: ClassVar[bool] = False
    shrank: ClassVar[bool] = False
    grew_again: ClassVar[bool] = False

    def __init__(self) -> None:
        self._deadline = time.monotonic() + 60

    def on_execution_step(self, executor: StreamingExecutor) -> None:
        for op, state in (executor._topology or {}).items():
            if op.name != "Map(_Download)":
                continue
            pool = op.get_autoscaling_actor_pools()[0]
            info = pool.get_actor_info()
            if info.active == 4 and info.pending == 1:
                self.root.joinpath("downloads").touch()
            if info.pending and info.idle and state.total_enqueued_input_blocks():
                type(self).saw_pending_with_idle = True
            if self.saw_pending_with_idle and 0 < info.running < 4 and info.pending == 0:
                type(self).shrank = True
                self.root.joinpath("consumer").touch()
            if self.shrank and info.active >= 4:
                type(self).grew_again = True
            assert self.grew_again or time.monotonic() < self._deadline, f"Pool did not shrink and regrow: {info}"


@pytest.mark.integration
def test_download_pool_shrinks_with_pending_actor_and_regrows(
    curator_io_cluster: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fifth unschedulable IO actor must not pin four live actors under backpressure."""
    del curator_io_cluster
    monkeypatch.setattr(_ObservePool, "root", tmp_path, raising=False)
    monkeypatch.setattr(_ObservePool, "saw_pending_with_idle", False)
    monkeypatch.setattr(_ObservePool, "shrank", False)
    monkeypatch.setattr(_ObservePool, "grew_again", False)
    source = ray.data.range(256, override_num_blocks=128)
    source.context.target_max_block_size = 1024**2
    source.context.enable_progress_bars = False
    source.context.downstream_capacity_backpressure_ratio = 1.0
    source.context.execution_options.resource_limits = source.context.execution_options.resource_limits.copy(
        object_store_memory=32 * 1024**2
    )
    configure_download_actor_autoscaling(source.context)
    source.context.custom_execution_callback_classes.append(_ObservePool)
    downloads = source.map(
        _Download,
        fn_constructor_args=(str(tmp_path),),
        compute=ActorPoolStrategy(min_size=1, initial_size=5, max_tasks_in_flight_per_actor=1),
        # The shared test cluster has sixteen IO slots: four actors can start,
        # while the fifth remains pending until an existing actor releases them.
        resources={"curator_io": 4},
        num_cpus=0.25,
        memory=128 * 1024**2,
        max_concurrency=1,
    )
    consumed = downloads.map(
        _Consumer,
        fn_constructor_args=(str(tmp_path),),
        compute=ActorPoolStrategy(size=1, max_tasks_in_flight_per_actor=1),
        num_cpus=0.25,
        memory=128 * 1024**2,
        max_concurrency=1,
    )
    try:
        result = consumed.materialize()
        assert sorted(row["id"] for row in result.take_all()) == list(range(256))
        assert _ObservePool.saw_pending_with_idle
        assert _ObservePool.shrank
        assert _ObservePool.grew_again
    finally:
        (tmp_path / "downloads").touch()
        (tmp_path / "consumer").touch()
