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
"""OpenAI-compatible API captioning stage for remote VLM inference (e.g. vLLM serving)."""

import asyncio
import base64
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import nvtx  # type: ignore[import-untyped]
import tenacity
from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStageResource
from cosmos_curator.core.utils.config.config import maybe_load_config
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.core.utils.model import pixi_utils
from cosmos_curator.models.prompts import get_prompt
from cosmos_curator.pipelines.common.api_caption_utils import (
    create_openai_client_and_resolve_model,
    normalize_openai_response_with_detail,
    openai_error_result_from_exception,
)
from cosmos_curator.pipelines.common.api_stage_async_utils import destroy_api_clients
from cosmos_curator.pipelines.video.captioning.single_inference import SingleInferenceCaptionStage
from cosmos_curator.pipelines.video.utils.data_model import (
    CaptionOutcome,
    CaptionResult,
    Clip,
    SplitPipeTask,
    Window,
)

if TYPE_CHECKING:
    import openai


if pixi_utils.is_running_in_env("unified"):
    import openai


@dataclass(frozen=True)
class _WindowCaptionTask:
    """One captioning work item bound to a specific window."""

    clip: Clip
    window: Window
    window_index: int


class OpenAICaptionStage(SingleInferenceCaptionStage):
    """Caption video windows using an OpenAI-compatible vision API.

    Sends each window's MP4 bytes as a base64-encoded video to a remote
    OpenAI-compatible endpoint (e.g. vLLM serving a VLM) and stores the
    returned caption.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_name: str,
        model_variant: str = "openai",
        prompt_variant: str = "default",
        prompt_text: str | None = None,
        max_output_tokens: int = 8192,
        max_caption_retries: int = 3,
        retry_delay_seconds: float = 1.0,
        batch_size: int = 1,
        use_filter_windows: bool = False,
        endpoint_key: Literal["caption", "enhance", "embedding", "filter", "classifier"] = "caption",
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the OpenAI-compatible API caption stage.

        Args:
            model_name: Model name to pass in the API request.
            model_variant: Identifier stored alongside generated captions.
            prompt_variant: Prompt variant used to build caption instructions.
            prompt_text: Optional custom prompt text.
            max_output_tokens: Maximum output tokens requested from the API.
            max_caption_retries: Number of retries per window before giving up.
            retry_delay_seconds: Delay between retries.
            batch_size: Stage batch size and async concurrency limit.
            use_filter_windows: If True, iterate clip.filter_windows instead of clip.windows.
            endpoint_key: Key under config.openai to read credentials from; must be one of the
                fields defined on OpenAIConfig ("caption", "enhance", "embedding", "filter", "classifier").
            verbose: Emit verbose logging.
            log_stats: Whether to record stage performance statistics.

        """
        super().__init__()
        self._timer = StageTimer(self)
        self._model_name = model_name
        self._model_variant = model_variant
        self._prompt = get_prompt(prompt_variant, prompt_text, verbose=verbose)
        self._max_output_tokens = max_output_tokens
        self._max_caption_retries = max_caption_retries
        self._retry_delay_seconds = retry_delay_seconds
        self._batch_size = max(1, batch_size)
        self._use_filter_windows = use_filter_windows
        self._endpoint_key = endpoint_key
        self._verbose = verbose
        self._log_stats = log_stats
        self._client: openai.OpenAI | None = None
        self._async_client: Any | None = None
        self._runner: asyncio.Runner | None = None

    def secondary_name(self) -> str:
        """Return the model variant for logging."""
        return self._model_variant

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage."""
        return CuratorStageResource(cpus=1.0)

    @property
    def conda_env_name(self) -> str:
        """Use the unified environment (openai package lives there)."""
        return "unified"

    @property
    def stage_batch_size(self) -> int:
        """Return the batch size used for scheduling and async concurrency."""
        return self._batch_size

    def stage_setup(self) -> None:
        """Create the OpenAI API client using credentials from the config file."""
        config = maybe_load_config()
        endpoint = (
            getattr(config.openai, self._endpoint_key, None)
            if config is not None and config.openai is not None
            else None
        )
        if endpoint is None or not endpoint.api_key:
            error_msg = (
                f"OpenAI {self._endpoint_key} configuration not found. "
                f"Provide openai.{self._endpoint_key}.api_key in ~/.config/cosmos_curator/config.yaml"
            )
            raise RuntimeError(error_msg)

        self._client, self._model_name = create_openai_client_and_resolve_model(
            openai,
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            model_name=self._model_name,
            endpoint_label=f"OpenAI {self._endpoint_key}",
        )
        client_kwargs: dict[str, Any] = {"api_key": endpoint.api_key}
        if endpoint.base_url:
            client_kwargs["base_url"] = endpoint.base_url
        self._async_client = openai.AsyncOpenAI(**client_kwargs)
        self._runner = asyncio.Runner()

    @staticmethod
    def _write_caption_result(window: Window, model_variant: str, result: CaptionResult) -> None:
        """Write an OpenAI caption result onto a window."""
        if result.text is not None:
            window.caption[model_variant] = result.text
        window.caption_status = result.outcome.value
        window.caption_failure_reason = result.failure_reason if result.outcome == CaptionOutcome.ERROR else None

    async def _generate_caption_with_error_detail_async(self, window: Window) -> tuple[CaptionResult, str | None]:
        """Generate a caption result for a single window using the async client."""
        client = self._async_client
        if client is None:
            msg = "OpenAI async client not initialized; call stage_setup before generating captions."
            raise RuntimeError(msg)

        mp4_data = window.mp4_bytes.resolve()
        if mp4_data is None:
            return (
                CaptionResult(outcome=CaptionOutcome.ERROR, failure_reason="exception"),
                "Window missing mp4 bytes; enable keep_mp4 in the prep stage.",
            )

        video_b64 = base64.b64encode(bytes(mp4_data)).decode("utf-8")
        instruction = self._prompt.strip()
        content_parts: list[dict[str, Any]] = [
            {
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
            },
            {"type": "text", "text": instruction},
        ]
        request_kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": content_parts}],
            "max_tokens": self._max_output_tokens,
        }

        async def _call() -> object:
            async for attempt in tenacity.AsyncRetrying(
                stop=tenacity.stop_after_attempt(self._max_caption_retries),
                wait=tenacity.wait_fixed(self._retry_delay_seconds),
                retry=tenacity.retry_if_not_exception_type(
                    (openai.AuthenticationError, openai.NotFoundError, openai.BadRequestError),
                ),
                reraise=True,
            ):
                with attempt:
                    return await client.chat.completions.create(**request_kwargs)
            msg = "OpenAI async retry loop exited without a result."
            raise RuntimeError(msg)

        try:
            response = await _call()
        except Exception as exc:  # noqa: BLE001
            timeout_error = getattr(openai, "APITimeoutError", None)
            return openai_error_result_from_exception(exc, timeout_error_type=timeout_error)
        return normalize_openai_response_with_detail(response)

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
                f"OpenAI API captioning failed: {result.failure_reason}"
            )
            logger.warning(
                f"OpenAI API captioning failed for clip {clip.uuid} window {window_index}: "
                f"{clip.errors[f'{self._model_variant}_caption_{window_index}']}"
            )
        elif result.outcome == CaptionOutcome.BLOCKED:
            logger.warning(f"OpenAI API captioning blocked for clip {clip.uuid} window {window_index}")
        elif self._verbose and result.text is not None:
            logger.info(f"OpenAI API caption clip {clip.uuid} window {window_index}: {result.text}")
        self._write_caption_result(window, self._model_variant, result)

    async def _process_one_window_async(
        self,
        window_task: _WindowCaptionTask,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Caption one window asynchronously."""
        clip = window_task.clip
        window_index = window_task.window_index
        try:
            async with semaphore:
                result, error_detail = await self._generate_caption_with_error_detail_async(window_task.window)
        except Exception as exc:  # noqa: BLE001
            result = CaptionResult(outcome=CaptionOutcome.ERROR, failure_reason="exception")
            clip.errors[f"{self._model_variant}_caption_{window_index}"] = str(exc)
            if self._verbose:
                logger.exception(f"OpenAI API captioning failed for clip {clip.uuid} window {window_index}")
            else:
                logger.warning(f"OpenAI API captioning failed for clip {clip.uuid} window {window_index}: {exc}")
            self._write_caption_result(window_task.window, self._model_variant, result)
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

    async def _generate_single_caption_async(
        self,
        prompt: str,
        video_bytes: bytes,
    ) -> tuple[CaptionResult, str | None]:
        """Run one chat-completions request without window-level book-keeping.

        Mirrors :meth:`_generate_caption_with_error_detail_async` (same
        retry policy, same response normalization) but takes the prompt
        and video bytes directly instead of pulling them off a ``Window``.
        Used by :meth:`caption_single` for one-shot consumers like
        ``PerEventCaptionStage``.
        """
        client = self._async_client
        if client is None:
            msg = "OpenAI async client not initialized; call stage_setup before generating captions."
            raise RuntimeError(msg)

        video_b64 = base64.b64encode(bytes(video_bytes)).decode("utf-8")
        content_parts: list[dict[str, Any]] = [
            {
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
            },
            {"type": "text", "text": prompt},
        ]
        request_kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": content_parts}],
            "max_tokens": self._max_output_tokens,
        }

        async def _call() -> object:
            async for attempt in tenacity.AsyncRetrying(
                stop=tenacity.stop_after_attempt(self._max_caption_retries),
                wait=tenacity.wait_fixed(self._retry_delay_seconds),
                retry=tenacity.retry_if_not_exception_type(
                    (openai.AuthenticationError, openai.NotFoundError, openai.BadRequestError),
                ),
                reraise=True,
            ):
                with attempt:
                    return await client.chat.completions.create(**request_kwargs)
            msg = "OpenAI async retry loop exited without a result."
            raise RuntimeError(msg)

        try:
            response = await _call()
        except Exception as exc:  # noqa: BLE001
            timeout_error = getattr(openai, "APITimeoutError", None)
            return openai_error_result_from_exception(exc, timeout_error_type=timeout_error)
        return normalize_openai_response_with_detail(response)

    def caption_single(self, prompt: str, video_bytes: bytes) -> str:
        """Implement :class:`SingleInferenceCaptionStage` for one-shot consumers.

        Drives :meth:`_generate_single_caption_async` synchronously via
        the runner created in :meth:`stage_setup`. Returns the response
        text or raises ``RuntimeError`` on blocked / empty responses.
        """
        if self._runner is None:
            msg = "OpenAI async runner not initialized; call stage_setup before generating captions."
            raise RuntimeError(msg)
        result, detail = self._runner.run(self._generate_single_caption_async(prompt, video_bytes))
        if result.outcome == CaptionOutcome.BLOCKED:
            msg = "OpenAI request blocked by content filter."
            raise RuntimeError(msg)
        if result.text is None:
            msg = detail or f"OpenAI request produced no caption text (outcome={result.outcome.value!r})."
            raise RuntimeError(msg)
        return result.text

    def destroy(self) -> None:
        """Close the async runner and any provider clients."""
        destroy_api_clients(async_client=self._async_client, runner=self._runner, sync_client=self._client)
        self._async_client = None
        self._runner = None
        self._client = None

    @nvtx.annotate("OpenAICaptionStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask]:
        """Caption each window in the provided tasks using the OpenAI-compatible API."""
        if self._runner is None:
            msg = "OpenAI async runner not initialized; call stage_setup before processing data."
            raise RuntimeError(msg)
        return self._runner.run(self._process_tasks_async(tasks))
