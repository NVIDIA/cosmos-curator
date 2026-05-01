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

"""Shared sync/async execution helpers for API-backed pipeline stages."""

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import TypeVar, cast

from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStage
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.pipelines.video.utils.data_model import CaptionResult

TTask = TypeVar("TTask")
TItem = TypeVar("TItem")

type GetItem[TTask, TItem] = Callable[[TTask], TItem]
type GetMajorSize[TTask] = Callable[[TTask], int | float]
type HasTargetResult[TItem] = Callable[[TItem], bool]
type AsyncGenerateWithDetail[TItem] = Callable[[TItem], Awaitable[tuple[CaptionResult, str | None]]]
type HandleResultOutcome[TTask] = Callable[[TTask, CaptionResult, str | None], None]
type WriteResult[TItem] = Callable[[TItem, CaptionResult], None]
type HandleException[TTask, TItem] = Callable[[TTask, TItem, Exception], tuple[CaptionResult, str]]
type CleanupItem[TItem] = Callable[[TItem], None]


def _is_filtered_item(item: object) -> bool:
    """Return whether the item has already been marked filtered."""
    return bool(getattr(item, "is_filtered", False))


@dataclass
class ApiTaskProcessContext[TTask, TItem]:
    """Shared state for sync/async processing of API-backed per-item tasks."""

    stage: CuratorStage
    timer: StageTimer
    log_stats: bool
    get_item: GetItem[TTask, TItem]
    get_major_size: GetMajorSize[TTask]
    has_target_result: HasTargetResult[TItem]
    async_generate_with_detail: AsyncGenerateWithDetail[TItem]
    handle_result_outcome: HandleResultOutcome[TTask]
    write_result: WriteResult[TItem]
    handle_exception: HandleException[TTask, TItem]
    cleanup_item: CleanupItem[TItem] | None = None

    def _record_task_perf(self, task: TTask, timer: StageTimer) -> None:
        """Attach timer stats to the task when profiling is enabled."""
        if not self.log_stats:
            return
        stage_perf = getattr(task, "stage_perf", None)
        if stage_perf is None:
            return
        stage_name, stage_perf_stats = timer.log_stats()
        stage_perf[stage_name] = stage_perf_stats

    async def _process_one_task_async(
        self,
        task: TTask,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Run one task through the async request path."""
        item = self.get_item(task)
        if _is_filtered_item(item):
            return
        if self.has_target_result(item):
            return

        task_timer = StageTimer(self.stage)
        task_timer.reinit(self.stage, int(self.get_major_size(task)))
        async with semaphore:
            with task_timer.time_process():
                try:
                    result, detail = await self.async_generate_with_detail(item)
                except Exception as exc:  # noqa: BLE001
                    result, detail = self.handle_exception(task, item, exc)
                else:
                    self.handle_result_outcome(task, result, detail)
        self.write_result(item, result)
        if self.cleanup_item is not None:
            self.cleanup_item(item)
        self._record_task_perf(task, task_timer)

    async def process_tasks_async(
        self,
        tasks: list[TTask],
        *,
        max_concurrent_requests: int,
    ) -> list[TTask]:
        """Run the async request function concurrently over a batch of tasks."""
        semaphore = asyncio.Semaphore(max(1, max_concurrent_requests))
        await asyncio.gather(*(self._process_one_task_async(task, semaphore) for task in tasks))
        return tasks


def close_resource_with_runner(resource: object | None, runner: asyncio.Runner | None) -> None:
    """Close a sync or async client-like resource if it exposes a close method."""

    def _run_awaitable(awaitable: Coroutine[object, object, object]) -> None:
        if runner is None:
            msg = "Async resource close requested without an asyncio runner."
            raise RuntimeError(msg)
        runner.run(awaitable)

    if resource is None:
        return
    close_method = getattr(resource, "close", None)
    if close_method is None:
        close_method = getattr(resource, "aclose", None)
    if close_method is None:
        return
    try:
        result = close_method()
        if inspect.isawaitable(result):
            _run_awaitable(cast("Coroutine[object, object, object]", result))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to close API client cleanly: {exc}")


def destroy_api_clients(
    *,
    async_client: object | None,
    runner: asyncio.Runner | None,
    sync_client: object | None,
) -> None:
    """Close async and sync API clients along with their optional runner."""
    close_resource_with_runner(async_client, runner)
    if runner is not None:
        runner.close()
    close_resource_with_runner(sync_client, None)
