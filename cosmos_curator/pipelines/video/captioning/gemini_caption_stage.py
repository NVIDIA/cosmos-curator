# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Remote API-backed caption preparation and captioning stages (Gemini)."""

import asyncio
from dataclasses import dataclass
from typing import Any, TypeVar

import nvtx  # type: ignore[import-untyped]
import tenacity
from google import genai
from google.api_core.exceptions import DeadlineExceeded
from google.genai import types as genai_types
from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource, PipelineTask
from cosmos_curator.core.utils.config.config import load_config
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.models.prompts import get_prompt
from cosmos_curator.pipelines.common.api_caption_utils import (
    gemini_error_result_from_exception,
    handle_gemini_client_exception,
    normalize_gemini_response_with_detail,
    should_retry_gemini_exception,
)
from cosmos_curator.pipelines.common.api_stage_async_utils import destroy_api_clients
from cosmos_curator.pipelines.video.utils import windowing_utils
from cosmos_curator.pipelines.video.utils.data_model import (
    CaptionOutcome,
    CaptionResult,
    Clip,
    SplitPipeTask,
    Video,
    Window,
    WindowConfig,
    get_video_from_task,
)

TTask = TypeVar("TTask", bound=PipelineTask)


@dataclass(frozen=True)
class _WindowCaptionTask:
    """One captioning work item bound to a specific window."""

    clip: Clip
    window: Window
    window_index: int


class ApiPrepStage(CuratorStage):
    """Stage that prepares windows for remote API captioning."""

    def __init__(  # noqa: PLR0913
        self,
        window_config: WindowConfig,
        *,
        model_variant: str = "gemini",
        num_cpus_for_prepare: float = 1.0,
        use_filter_windows: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the API prep stage."""
        super().__init__()
        self._timer = StageTimer(self)
        self._window_config = window_config
        self._model_variant = model_variant
        self._num_cpus_for_prepare = num_cpus_for_prepare
        self._use_filter_windows = use_filter_windows
        self._verbose = verbose
        self._log_stats = log_stats

    def secondary_name(self) -> str:
        """Return the model variant for logging."""
        return self._model_variant

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage."""
        return CuratorStageResource(cpus=self._num_cpus_for_prepare)

    @property
    def conda_env_name(self) -> str:
        """Use the unified environment for window preparation."""
        return "unified"

    def _prep_windows(self, video: Video) -> None:
        """Create windows for the provided video."""
        num_video_decode_threads = max(1, int(self.resources.cpus) + 1)
        windows, _ = windowing_utils.make_windows_for_video(
            video,
            self._window_config,
            num_video_decode_threads,
            keep_mp4=True,
            return_frames=False,
        )
        if self._verbose:
            logger.debug(f"Prepared {len(windows)} windows for {video.input_video}")
        if self._use_filter_windows:
            for clip in video.clips:
                clip.filter_windows = clip.windows[:]
                clip.windows = []

    @nvtx.annotate("ApiPrepStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[TTask]) -> list[TTask]:
        """Prepare data for API captioning."""
        for task in tasks:
            major_size = task.get_major_size()
            self._timer.reinit(self, major_size)
            video = get_video_from_task(task)
            with self._timer.time_process():
                self._prep_windows(video)

            stage_perf = getattr(task, "stage_perf", None)
            if self._log_stats and stage_perf is not None:
                stage_name, stage_perf_stats = self._timer.log_stats()
                stage_perf[stage_name] = stage_perf_stats

        return tasks


class GeminiCaptionStage(CuratorStage):
    """Caption video windows using the Google Gemini API.

    The Gemini API key must be provided in the cosmos-curator config file under the `gemini` section.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_variant: str = "gemini",
        model_name: str = "models/gemini-2.5-pro",
        prompt_variant: str = "default",
        prompt_text: str | None = None,
        max_output_tokens: int = 4096,
        max_caption_retries: int = 3,
        retry_delay_seconds: float = 1.0,
        max_video_size_bytes: int = 20 * 1024 * 1024,
        batch_size: int = 1,
        use_filter_windows: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the API caption stage.

        Args:
            model_variant: Identifier stored alongside generated captions.
            model_name: Gemini model name to invoke.
            prompt_variant: Prompt variant used to build caption instructions.
            prompt_text: Optional custom prompt text.
            max_output_tokens: Maximum output tokens requested from the API.
            max_caption_retries: Number of retries per window before giving up.
            retry_delay_seconds: Delay between retries.
            max_video_size_bytes: Maximum inline video size supported by the API.
            batch_size: Stage batch size and async concurrency limit.
            use_filter_windows: If True, iterate clip.filter_windows instead of clip.windows.
            verbose: Emit verbose logging.
            log_stats: Whether to record stage performance statistics.

        """
        super().__init__()
        self._timer = StageTimer(self)
        self._model_variant = model_variant
        self._model_name = model_name
        self._use_filter_windows = use_filter_windows
        self._prompt_variant = prompt_variant
        self._prompt_text = prompt_text
        self._prompt = get_prompt(prompt_variant, prompt_text, verbose=verbose)
        self._max_output_tokens = max_output_tokens
        self._max_caption_retries = max_caption_retries
        self._retry_delay_seconds = retry_delay_seconds
        self._max_video_size_bytes = max_video_size_bytes
        self._batch_size = max(1, batch_size)
        self._verbose = verbose
        self._log_stats = log_stats
        config = load_config()
        if config.gemini is None or not config.gemini.api_key:
            msg = "Gemini API key missing from config file."
            raise RuntimeError(msg)
        self._api_key = config.gemini.api_key
        self._client: genai.Client | None = None
        self._async_client: Any | None = None
        self._runner: asyncio.Runner | None = None

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage."""
        return CuratorStageResource(cpus=1.0)

    @property
    def stage_batch_size(self) -> int:
        """Return the batch size used for scheduling and async concurrency."""
        return self._batch_size

    def stage_setup(self) -> None:
        """Create the Gemini API client."""
        self._client = genai.Client(api_key=self._api_key)
        self._async_client = self._client.aio
        self._runner = asyncio.Runner()

    @staticmethod
    def _write_caption_result(window: Window, model_variant: str, result: CaptionResult) -> None:
        """Write a Gemini caption result onto a window."""
        if result.text is not None:
            window.caption[model_variant] = result.text
        window.caption_status = result.outcome.value
        window.caption_failure_reason = result.failure_reason if result.outcome == CaptionOutcome.ERROR else None

    async def _generate_caption_with_error_detail_async(self, window: Window) -> tuple[CaptionResult, str | None]:
        """Generate a caption result for a single window using the async client."""
        client = self._async_client
        if client is None:
            msg = "Gemini async client not initialized; call stage_setup before generating captions."
            raise RuntimeError(msg)

        instruction = self._prompt.strip()
        mp4_data = window.mp4_bytes.resolve()
        if mp4_data is None:
            msg = "Window missing mp4 bytes; _validate_window must be called before _generate_caption."
            raise RuntimeError(msg)

        inline_data = genai_types.Blob(data=bytes(mp4_data), mime_type="video/mp4")
        content = genai_types.Content(
            parts=[
                genai_types.Part(inline_data=inline_data),
                genai_types.Part(text=instruction),
            ]
        )
        generate_kwargs: dict[str, Any] = {
            "model": self._model_name,
            "contents": content,
            "config": genai_types.GenerateContentConfig(max_output_tokens=self._max_output_tokens),
        }

        async def _call() -> object:
            async for attempt in tenacity.AsyncRetrying(
                stop=tenacity.stop_after_attempt(self._max_caption_retries),
                wait=tenacity.wait_fixed(self._retry_delay_seconds),
                retry=tenacity.retry_if_exception(should_retry_gemini_exception),
                reraise=True,
            ):
                with attempt:
                    try:
                        return await client.models.generate_content(**generate_kwargs)
                    except Exception as exc:
                        new_exc = handle_gemini_client_exception(exc)
                        if new_exc is exc:
                            raise
                        raise new_exc from exc
            msg = "Gemini async retry loop exited without a result."
            raise RuntimeError(msg)

        try:
            response = await _call()
        except Exception as exc:  # noqa: BLE001
            return gemini_error_result_from_exception(exc, timeout_error_type=DeadlineExceeded)
        return normalize_gemini_response_with_detail(response)

    def _validate_window(self, window: Window) -> None:
        """Validate that the window contains data suitable for Gemini."""
        mp4_data = window.mp4_bytes.resolve()
        if mp4_data is None:
            msg = "Window missing mp4 bytes; enable keep_mp4 in the prep stage."
            raise RuntimeError(msg)
        if mp4_data.nbytes > self._max_video_size_bytes:
            size_mb = mp4_data.nbytes / (1024 * 1024)
            max_mb = self._max_video_size_bytes / (1024 * 1024)
            msg = f"Window MP4 ({size_mb:.2f} MB) exceeds Gemini inline limit ({max_mb:.2f} MB)."
            raise RuntimeError(msg)

    def _iter_window_tasks(self, task: SplitPipeTask) -> list[_WindowCaptionTask]:
        """Flatten a SplitPipeTask into caption work items."""
        window_tasks: list[_WindowCaptionTask] = []
        for clip in task.video.clips:
            window_source = clip.filter_windows if self._use_filter_windows else clip.windows
            for window_index, window in enumerate(window_source):
                window_tasks.append(_WindowCaptionTask(clip=clip, window=window, window_index=window_index))
        return window_tasks

    def _process_window_caption_result(
        self,
        window_task: _WindowCaptionTask,
        result: CaptionResult,
        error_detail: str | None,
    ) -> None:
        """Write one caption result and handle provider-specific logging."""
        clip = window_task.clip
        window = window_task.window
        window_index = window_task.window_index
        if result.outcome == CaptionOutcome.ERROR:
            clip.errors[f"{self._model_variant}_caption_{window_index}"] = error_detail or (
                f"Gemini captioning failed: {result.failure_reason}"
            )
            logger.warning(
                f"Gemini captioning failed for clip {clip.uuid} window {window_index}: "
                f"{clip.errors[f'{self._model_variant}_caption_{window_index}']}"
            )
        elif result.outcome == CaptionOutcome.BLOCKED:
            logger.warning(f"Gemini captioning blocked for clip {clip.uuid} window {window_index}")
        elif self._verbose and result.text is not None:
            logger.info(f"Gemini caption clip {clip.uuid} window {window_index}: {result.text}")
        self._write_caption_result(window, self._model_variant, result)

    async def _process_one_window_async(
        self,
        window_task: _WindowCaptionTask,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Caption one window asynchronously."""
        clip = window_task.clip
        window = window_task.window
        window_index = window_task.window_index
        try:
            self._validate_window(window)
            async with semaphore:
                result, error_detail = await self._generate_caption_with_error_detail_async(window)
        except Exception as exc:  # noqa: BLE001
            result = CaptionResult(outcome=CaptionOutcome.ERROR, failure_reason="exception")
            clip.errors[f"{self._model_variant}_caption_{window_index}"] = str(exc)
            if self._verbose:
                logger.exception(f"Gemini captioning failed for clip {clip.uuid} window {window_index}")
            else:
                logger.warning(f"Gemini captioning failed for clip {clip.uuid} window {window_index}: {exc}")
            self._write_caption_result(window, self._model_variant, result)
            return
        self._process_window_caption_result(window_task, result, error_detail)

    async def _process_task_async(
        self,
        task: SplitPipeTask,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Process one SplitPipeTask using concurrent async API requests."""
        task_timer = StageTimer(self)
        task_timer.reinit(self, task.get_major_size())
        with task_timer.time_process():
            await asyncio.gather(
                *(
                    self._process_one_window_async(window_task, semaphore)
                    for window_task in self._iter_window_tasks(task)
                )
            )
        if self._log_stats:
            stage_name, stage_perf_stats = task_timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats

    async def _process_tasks_async(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask]:
        """Process a batch of tasks using the async request path."""
        semaphore = asyncio.Semaphore(max(1, self._batch_size))
        await asyncio.gather(*(self._process_task_async(task, semaphore) for task in tasks))
        return tasks

    def destroy(self) -> None:
        """Close the async runner and any provider clients."""
        destroy_api_clients(async_client=self._async_client, runner=self._runner, sync_client=self._client)
        self._async_client = None
        self._runner = None
        self._client = None

    @nvtx.annotate("GeminiCaptionStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask]:
        """Caption each window in the provided tasks using Gemini."""
        if self._runner is None:
            msg = "Gemini async runner not initialized; call stage_setup before processing data."
            raise RuntimeError(msg)
        return self._runner.run(self._process_tasks_async(tasks))
