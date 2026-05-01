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
"""vLLM caption stages.

The VllmPrepStage and VllmCaptionStage classes are designed to be used
in any pipeline. Because they are designed to be used in any pipeline, they
are generic and not specific to any particular pipeline or task type.

For the VllmPrepStage and VllmCaptionStage to function properly, the
the tasks must have these attributes/methods:

- video: The video to process.
- stage_perf: A dictionary to store performance statistics.
- get_major_size: A method to get the major size of the task.

"""

import logging
from typing import TYPE_CHECKING, Any, TypeVar, cast

import nvtx  # type: ignore[import-untyped]
import tenacity
from loguru import logger

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource, PipelineTask
from cosmos_curator.core.utils.infra.gpu_start_helper import (
    gpu_stage_cleanup,
    gpu_stage_startup,
)
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.core.utils.model import conda_utils, model_utils
from cosmos_curator.models.all_models import get_all_models_by_id
from cosmos_curator.models.prompts import get_prompt, get_stage2_prompt
from cosmos_curator.models.vllm_model_ids import get_vllm_model_id
from cosmos_curator.models.vllm_sentinels import VLLM_UNKNOWN_CAPTION
from cosmos_curator.pipelines.video.utils import windowing_utils
from cosmos_curator.pipelines.video.utils.data_model import (
    CaptionOutcome,
    CaptionResult,
    TokenCounts,
    Video,
    VllmConfig,
    Window,
    WindowConfig,
    get_video_from_task,
)

if conda_utils.is_running_in_env("unified"):
    if TYPE_CHECKING:
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

    from cosmos_curator.models.vllm_interface import (
        VllmWindowResult,
        auto_processor,
        make_metadata,
        make_model_inputs,
        sampling_params,
        vllm_caption,
        vllm_model,
    )

    vllm_logger = logging.getLogger("vllm")
    vllm_logger.setLevel(logging.ERROR)  # Suppress warnings and info from vLLM


T = TypeVar("T", bound=PipelineTask)


def _get_windows_from_tasks[T: PipelineTask](tasks: list[T]) -> tuple[list[Window], list[str]]:
    """Get the windows from a list of tasks.

    Args:
        tasks: The tasks with video -> clips -> windows.

    Returns:
        The windows and clip uuids from the task.

    Raises:
        TypeError: If the task does not have a video attribute.

    """
    windows: list[Window] = []
    clip_uuids: list[str] = []
    for task in tasks:
        video = get_video_from_task(task)
        for clip in video.clips:
            if not clip.windows:
                logger.warning(f"Clip {clip.uuid} has no windows")
                clip.errors["clip_windowing"] = "empty"
                continue
            windows += clip.windows
            clip_uuids += [str(clip.uuid)] * len(clip.windows)

    return windows, clip_uuids


def _get_filter_windows_from_tasks[T: PipelineTask](tasks: list[T]) -> tuple[list[Window], list[str]]:
    """Get the filter_windows from a list of tasks.

    Args:
        tasks: The tasks with video -> clips -> filter_windows.

    Returns:
        The filter_windows and clip uuids from the task.

    """
    windows: list[Window] = []
    clip_uuids: list[str] = []
    for task in tasks:
        video = get_video_from_task(task)
        for clip in video.clips:
            if not clip.filter_windows:
                logger.warning(f"Clip {clip.uuid} has no filter_windows")
                clip.errors["clip_windowing"] = "empty"
                continue
            windows += clip.filter_windows
            clip_uuids += [str(clip.uuid)] * len(clip.filter_windows)

    return windows, clip_uuids


def _get_stage2_prompts(vllm_config: VllmConfig, num_windows: int) -> list[str | None]:
    """Get the stage 2 prompts for the vLLM model.

    Args:
        vllm_config: The configuration for the vLLM model.
        num_windows: The number of windows to get the stage 2 prompts for.

    Returns:
        The stage 2 prompts for the vLLM model.

    """
    if vllm_config.stage2_caption:
        return [get_stage2_prompt(vllm_config.stage2_prompt_text)] * num_windows
    return [None] * num_windows


def _scatter_captions(
    windows: list[Window],
    results: list["VllmWindowResult"],
    clip_uuids: list[str],
    model_variant: str,
    *,
    verbose: bool,
) -> None:
    """Scatter the captions and token counts back to the windows.

    Args:
        windows: The windows to scatter the captions to.
        results: The per-window vLLM results to scatter.
        clip_uuids: The clip uuids to scatter the captions to.
        model_variant: The variant of the model.
        verbose: Whether to print verbose logs.

    """
    for window, raw_result, clip_uuid in zip(windows, results, clip_uuids, strict=True):
        result = _normalize_vllm_result(raw_result)
        if result.text is not None:
            window.caption[model_variant] = result.text
        window.token_counts[model_variant] = raw_result.token_counts
        window.caption_status = result.outcome.value
        window.caption_failure_reason = result.failure_reason if result.outcome == CaptionOutcome.ERROR else None
        if verbose:
            logger.info(f"Caption for clip {clip_uuid}: {raw_result.text}")


def _normalize_vllm_result(result: "VllmWindowResult") -> CaptionResult:
    """Map a raw vLLM interface result to a caption result."""
    text = result.text.strip()
    if result.text == VLLM_UNKNOWN_CAPTION:
        return CaptionResult(outcome=CaptionOutcome.ERROR, failure_reason="exception")
    if result.finish_reason == "length":
        if text:
            return CaptionResult(outcome=CaptionOutcome.TRUNCATED, text=text)
        return CaptionResult(outcome=CaptionOutcome.ERROR, failure_reason="exception")
    if text:
        return CaptionResult(outcome=CaptionOutcome.SUCCESS, text=text)
    return CaptionResult(outcome=CaptionOutcome.ERROR, failure_reason="exception")


def _free_vllm_inputs(windows: list[Window], model_variant: str, *, keep_mp4: bool = False) -> None:
    """Free unused memory for the model variant.

    Args:
        windows: The windows to free unused memory for.
        model_variant: The variant of the model.
        keep_mp4: Whether to keep the mp4 bytes.

    """
    for window in windows:
        window.model_input.pop(model_variant, None)
        if not keep_mp4:
            window.mp4_bytes.drop()


class VllmModelInterface(ModelInterface):
    """Information about a vLLM model."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        """Initialize the vLLM model interface."""
        self._vllm_config = vllm_config

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name."""
        return "unified"

    @property
    def model_id_names(self) -> list[str]:
        """Get the model ID names."""
        model_variant = get_vllm_model_id(self._vllm_config.model_variant)
        models = get_all_models_by_id()
        model = models.get(model_variant)

        if model is None:
            msg = f"Model not found for {self._vllm_config.model_variant} -> {model_variant}"
            raise ValueError(msg)

        model_id = model.get("model_id")
        if model_id is None:
            msg = f"Model ID not found for variant {self._vllm_config.model_variant} -> {model_variant}"
            raise ValueError(msg)

        return [cast("str", model_id)]

    def setup(self) -> None:
        """Set up the vLLM model interface."""


class VllmPrepStage(CuratorStage):
    """Stage that prepares cosmos-curator video data for vLLM multimodal model processing."""

    def __init__(  # noqa: PLR0913
        self,
        vllm_config: VllmConfig,
        window_config: WindowConfig,
        *,
        keep_mp4: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
        use_filter_windows: bool = False,
    ) -> None:
        """Initialize the vLLM Preparation Stage.

        Args:
            vllm_config: Configuration for the vLLM model.
            window_config: Configuration for the windowing.
            keep_mp4: Keep mp4 bytes for the clips in memory.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.
            use_filter_windows: If True, store windows in clip.filter_windows instead
                of clip.windows. Use this when the stage is part of a filtering pipeline
                rather than a captioning pipeline.

        """
        super().__init__()

        self._timer = StageTimer(self)
        self._vllm_config = vllm_config
        self._window_config = window_config
        self._verbose = verbose
        self._log_stats = log_stats
        self._processor: AutoProcessor | None = None
        self._keep_mp4 = keep_mp4
        self._use_filter_windows = use_filter_windows
        self._model = VllmModelInterface(self._vllm_config)

    def secondary_name(self) -> str:
        """Get the secondary name of the stage.

        Returns:
            The secondary name of the stage.

        """
        # mypy is not smart enough to know that self._vllm_config.model_variant is a str
        # but mypy also thinks that this is a redundant cast
        return cast("str", self._vllm_config.model_variant)  # type: ignore[redundant-cast]

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        return CuratorStageResource(cpus=self._vllm_config.num_cpus_for_prepare)

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name.

        Returns:
            The conda environment name.

        """
        return "unified"

    def stage_setup(self) -> None:
        """Set up the model for processing."""
        self._processor = auto_processor(self._vllm_config)

    def _prep_windows(self, video: Video, prompt: str) -> None:
        """Prep the windows for the vLLM model.

        The videos are modified in-place by creating the windows
        for each clip in the videos and adding windows to each clip.

        Model inputs are added to each window.

        Args:
            video: The video to prep the windows for.
            prompt: The prompt to use for the vLLM model.

        """
        if self._processor is None:
            msg = "self._processor not initialized, call stage_setup() first"
            raise RuntimeError(msg)

        num_video_decode_threads = max(1, int(self.resources.cpus) + 1)

        windows, frames = windowing_utils.make_windows_for_video(
            video,
            self._window_config,
            num_video_decode_threads,
            keep_mp4=self._keep_mp4,
        )

        metadata = make_metadata(frames, self._window_config)

        # Create debug identifiers for frame organization
        debug_window_ids = None
        if self._vllm_config.debug_save_frames:
            debug_window_ids = []
            for window in windows:
                # Find which clip this window belongs to
                clip_uuid = "unknown_clip"
                for clip in video.clips:
                    if window in clip.windows:
                        clip_uuid = str(clip.uuid)
                        break
                debug_window_ids.append(clip_uuid)

        llm_inputs = make_model_inputs(
            frames,
            metadata,
            self._vllm_config,
            self._processor,
            prompt,
            debug_window_ids=debug_window_ids,
        )

        for window, llm_input in zip(windows, llm_inputs, strict=True):
            window.model_input[self._vllm_config.model_variant] = llm_input

        if self._use_filter_windows:
            for clip in video.clips:
                clip.filter_windows = clip.windows[:]
                clip.windows = []

    @nvtx.annotate("VllmPrepStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[T]) -> list[T]:
        """Prepare the data for the vLLM caption stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        if self._processor is None:
            msg = "self._processor not initialized, call stage_setup() first"
            raise RuntimeError(msg)

        prompt = get_prompt(
            self._vllm_config.prompt_variant,
            self._vllm_config.prompt_text,
            verbose=self._verbose,
        )

        for task in tasks:
            major_size = task.get_major_size()
            self._timer.reinit(self, major_size)

            video = get_video_from_task(task)

            with self._timer.time_process():
                self._prep_windows(video, prompt)

            stage_perf = getattr(task, "stage_perf", None)
            if self._log_stats and stage_perf is not None:
                stage_name, stage_perf_stats = self._timer.log_stats()
                stage_perf[stage_name] = stage_perf_stats

        return tasks


class VllmCaptionStage(CuratorStage):
    """Stage that prepares video windows for vLLM multimodal model processing.

    This stage handles the preparation of video windows and prompts for vLLM-based models.
    """

    def __init__(  # noqa: PLR0913
        self,
        vllm_config: VllmConfig,
        max_inflight_requests: int = 0,
        *,
        inflight_batching: bool = False,
        keep_mp4: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
        use_filter_windows: bool = False,
    ) -> None:
        """Initialize the vLLM caption stage.

        Args:
            vllm_config: Configuration for the vLLM model.
            max_inflight_requests: Maximum number of inflight requests to vLLM
               engine. Set to 0 for unlimited inflight requests. Ignored if
               inflight_batching is False.
            inflight_batching: set to True to enable inflight batching.
            keep_mp4: Whether to keep the mp4 bytes.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.
            use_filter_windows: If True, read windows from clip.filter_windows instead
                of clip.windows. Use this when paired with VllmPrepStage(use_filter_windows=True).

        """
        super().__init__()

        self._timer = StageTimer(self)
        self._vllm_config = vllm_config
        self._llm: LLM | None = None
        self._sampling_params: SamplingParams | None = None
        self._processor: AutoProcessor | None = None
        self._keep_mp4 = keep_mp4
        self._verbose = verbose
        self._log_stats = log_stats
        self._vllm_use_tqdm = False
        self._model = VllmModelInterface(self._vllm_config)
        self._max_inflight_requests = max_inflight_requests
        self._inflight_batching = inflight_batching
        self._use_filter_windows = use_filter_windows

    def stage_setup_on_node(self) -> None:
        """Set up on a node by copying model weights if configured.

        This method copies model weights from the default cache location to a
        user-configured directory (e.g., local SSD) before loading the model.

        If the copy fails, the model will use the default cache location.

        """
        if self._vllm_config.copy_weights_to is None:
            logger.debug("No custom weights directory configured, skipping weight copy")
        else:
            for model_id in self._model.model_id_names:
                source_dir = model_utils.get_local_dir_for_weights_name(model_id)

                if not source_dir.exists():
                    msg = f"Source model weights directory does not exist: {source_dir}"
                    raise FileNotFoundError(msg)

                dest_dir = self._vllm_config.copy_weights_to / model_id
                logger.info(f"Copying model weights for {model_id} from {source_dir} to {dest_dir}")

                try:
                    model_utils.copy_model_weights(source_dir, dest_dir)
                    logger.info(f"Successfully copied model weights to {dest_dir}")
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to copy model weights, will use default location")

        # Instantiate vLLM Engine involves torch.compile which produces cache
        # To avoid conflict, do it once here
        gpu_stage_startup(f"{self.__class__.__name__}-on-node", self.resources.gpus, pre_setup=True)
        self._llm = vllm_model(self._vllm_config)
        gpu_stage_startup(f"{self.__class__.__name__}-on-node", self.resources.gpus, pre_setup=False)

    def stage_setup(self) -> None:
        """Set up the model for processing."""
        if self._llm is None:
            gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)
            self._llm = vllm_model(self._vllm_config)
        self._sampling_params = sampling_params(self._vllm_config.sampling_config)
        self._processor = auto_processor(self._vllm_config)
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=False)

    def destroy(self) -> None:
        """Clean up GPU resources."""
        gpu_stage_cleanup(self.__class__.__name__)

    def _reset(self) -> None:
        """Reset the vLLM model."""
        del self._llm
        del self._sampling_params
        del self._processor
        self.destroy()
        self.stage_setup()

    def secondary_name(self) -> str:
        """Get the secondary name of the stage.

        Returns:
            The secondary name of the stage.

        """
        # mypy is not smart enough to know that self._vllm_config.model_variant is a str
        # but mypy also thinks that this is a redundant cast
        return cast("str", self._vllm_config.model_variant)  # type: ignore[redundant-cast]

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name.

        Returns:
            The conda environment name.

        """
        return "unified"

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        return CuratorStageResource(gpus=self._vllm_config.num_gpus)

    @property
    def model(self) -> VllmModelInterface:
        """Get the model for this stage.

        Returns:
            The model for this stage.

        """
        return self._model

    @nvtx.annotate("VllmCaptionStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[T]) -> list[T]:  # noqa: C901
        """Process the data for the vLLM caption stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        if self._llm is None:
            msg = "vLLM model not initialized, call stage_setup() first"
            raise RuntimeError(msg)

        if self._sampling_params is None:
            msg = "Sampling parameters not initialized, call stage_setup() first"
            raise RuntimeError(msg)

        if self._processor is None:
            msg = "Processor not initialized, call stage_setup() first"
            raise RuntimeError(msg)

        llm = self._llm
        sampling_params = self._sampling_params
        processor = self._processor

        major_size = sum(task.get_major_size() for task in tasks)
        self._timer.reinit(self, major_size)

        @tenacity.retry(stop=tenacity.stop_after_attempt(self._vllm_config.max_retries), reraise=True)
        def _vllm_caption(
            model_inputs: list[dict[str, Any]], stage2_prompts: list[str | None]
        ) -> list["VllmWindowResult"]:
            try:
                results = vllm_caption(
                    model_inputs,
                    llm,
                    processor,
                    sampling_params,
                    self._vllm_config,
                    inflight_batching=self._inflight_batching,
                    max_inflight_requests=self._max_inflight_requests,
                    stage2_prompts=stage2_prompts,
                )
            except Exception as e:
                input_videos = [str(get_video_from_task(t).input_video) for t in tasks]
                input_videos_str = ", ".join(input_videos)
                logger.exception(f"Error generating captions for video {input_videos_str}, trying again")
                for task in tasks:
                    video = get_video_from_task(task)
                    video.errors["captioning"] = f"vLLM captioning error: {e}"

                # On retry: tear down and restart vllm
                self._reset()
                raise
            else:
                for task in tasks:
                    video = get_video_from_task(task)
                    video.errors.pop("captioning", None)

                return results

        with self._timer.time_process():
            # Gather model inputs and clip uuids
            if self._use_filter_windows:
                windows, clip_uuids = _get_filter_windows_from_tasks(tasks)
            else:
                windows, clip_uuids = _get_windows_from_tasks(tasks)
            model_inputs = [window.model_input[self._vllm_config.model_variant] for window in windows]

            # Set up stage 2 prompts if enabled
            stage2_prompts = _get_stage2_prompts(self._vllm_config, len(windows))

            # Generate captions
            try:
                results = _vllm_caption(model_inputs, stage2_prompts)
            except Exception:  # noqa: BLE001
                logger.error(f"All {self._vllm_config.max_retries} retry attempts exhausted; captioning failed")
                results = [
                    VllmWindowResult(text="", finish_reason=None, token_counts=TokenCounts()) for _ in model_inputs
                ]

            # Scatter captions back to windows
            _scatter_captions(windows, results, clip_uuids, self._vllm_config.model_variant, verbose=self._verbose)

            logger.info(f"Generated {len(results)} captions for {len(tasks)} tasks")

        if self._log_stats:
            # Because there's a single call to caption all tasks, just log the first task's stage_perf.
            stage_name, stage_perf_stats = self._timer.log_stats()
            stage_perf = getattr(tasks[0], "stage_perf", None)
            if stage_perf is not None:
                stage_perf[stage_name] = stage_perf_stats

        _free_vllm_inputs(windows, self._vllm_config.model_variant, keep_mp4=self._keep_mp4)
        return tasks
