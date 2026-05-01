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
"""Captioning stage."""

from typing import cast

import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.config import operation_context
from cosmos_curator.core.utils.infra.gpu_start_helper import gpu_stage_startup
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.models import t5_encoder
from cosmos_curator.models.chat_lm import ChatLM
from cosmos_curator.models.prompts import get_enhance_prompt
from cosmos_curator.pipelines.video.utils.data_model import ShardPipeTask, SplitPipeTask


class _T5Stage(CuratorStage):
    """Stage that encodes captions using the T5 model.

    This stage processes video captions through the T5 encoder model to generate
    text embeddings for downstream tasks.
    """

    def __init__(
        self,
        caption_fields: list[str] | None = None,
        *,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the T5 caption encoding stage.

        Args:
            caption_fields: List of caption fields to encode.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        if caption_fields is None:
            caption_fields = ["qwen_caption"]
        self._timer = StageTimer(self)
        self._caption_fields = caption_fields
        self._verbose = verbose
        self._log_stats = log_stats
        self._model = t5_encoder.T5Encoder(t5_encoder.ModelVariant.T5_XXL)
        self._batch_size = 16 if operation_context.is_running_on_the_cloud() else 4

    def stage_setup(self) -> None:
        """Initialize stage resources and configuration."""
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)
        self._model.setup()
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=False)

    @property
    def model(self) -> ModelInterface:
        """Get the model.

        Returns:
            The model.

        """
        return self._model

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        return CuratorStageResource(gpus=1.0)

    def _add_prompt(self, all_prompts: list[str], caption_window: dict[str, str]) -> str | None:
        found_caption_field = None
        for field in self._caption_fields:
            if field in caption_window:
                all_prompts.append(caption_window[field])
                found_caption_field = field
                break
        return found_caption_field

    def _encode_prompts(self, all_prompts: list[str]) -> list[t5_encoder.EncodedSample]:
        return self._model.encode(all_prompts, batch_size=self._batch_size)


class T5StageForSplit(_T5Stage):
    """Stage that encodes captions using the T5 model for shard processing."""

    @nvtx.annotate("T5StageForSplit")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Process the data for the T5 caption encoding stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())

            all_prompts: list[str] = []
            mapping: list[tuple[int, int, str]] = []

            for clip_idx, clip in enumerate(task.video.clips):
                for window_idx, window in enumerate(clip.windows):
                    found_caption_field = self._add_prompt(all_prompts, window.caption)
                    if found_caption_field is None:
                        logger.error(f"Clip {clip.uuid} window {window_idx} has no caption, drop this clip!")
                        continue
                    mapping.append((clip_idx, window_idx, found_caption_field))

            encoded_results = self._encode_prompts(all_prompts)

            for result_idx, result in enumerate(encoded_results):
                clip_idx, window_idx, caption_field = mapping[result_idx]
                task.video.clips[clip_idx].windows[window_idx].t5_xxl_embedding[caption_field] = result.encoded_text

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks


class T5StageForShard(_T5Stage):
    """Stage that encodes captions using the T5 model for shard processing."""

    @nvtx.annotate("T5StageForShard")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[ShardPipeTask]) -> list[ShardPipeTask] | None:
        """Process the data for the T5 caption encoding stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())

            all_prompts: list[str] = []
            mapping: list[tuple[int, int]] = []

            for clip_idx, clip in enumerate(task.samples):
                for window_idx, window in enumerate(clip.clip_metadata["windows"]):
                    found_caption_field = self._add_prompt(all_prompts, window)
                    if found_caption_field is None:
                        logger.error(f"Clip {clip.uuid} window {window_idx} has no caption, drop this clip!")
                        clip.clip_metadata["valid"] = False
                        continue
                    mapping.append((clip_idx, window_idx))

            encoded_results = self._encode_prompts(all_prompts)

            for result_idx, result in enumerate(encoded_results):
                clip_idx, window_idx = mapping[result_idx]
                task.samples[clip_idx].t5_xxl_embeddings.append(result.encoded_text)
                assert (window_idx + 1) == len(task.samples[clip_idx].t5_xxl_embeddings)

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks


# Post-Caption-Stage to further enhance the generated Caption, either using
# additional Metadata or just 'Tuned Instructions in the Prompt'
class EnhanceCaptionStage(CuratorStage):
    """Stage that enhances video captions using a chat language model.

    This stage takes existing captions and uses a chat language model to generate
    more detailed and refined descriptions of the video content.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_variant: str = "qwen_lm",
        prompt_variant: str = "default",
        prompt_text: str | None = None,
        batch_size: int = 32,
        openai_model: str = "auto",
        *,
        fp8_enable: bool = False,
        max_output_tokens: int = 2048,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the caption enhancement stage.

        Args:
            model_variant: Language model to use for enhancement. One of
                "qwen_lm", "gpt_oss_20b", or "openai".
            prompt_variant: Type of prompt to use.
            prompt_text: Custom prompt text if provided.
            batch_size: Number of samples to process in parallel.
            openai_model: OpenAI model name (only used when model_variant is "openai").
            fp8_enable: Whether to enable FP8 precision (only for local models).
            max_output_tokens: Maximum number of tokens to generate.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        self._timer = StageTimer(self)
        self._batch_size = batch_size
        self._verbose = verbose
        self._log_stats = log_stats
        self._enhanced_key = model_variant

        quantization = "fp8" if fp8_enable and model_variant == "qwen_lm" else None
        self._raw_model = ChatLM(
            model_variant,
            max_output_tokens=max_output_tokens,
            quantization=quantization,
            openai_model=openai_model,
            verbose=verbose,
        )
        self._prompt_variant = prompt_variant
        self._prompt_text = prompt_text
        self._prompt = get_enhance_prompt(
            prompt_variant,
            prompt_text,
            verbose=verbose,
        )

    @property
    def model(self) -> ModelInterface:
        """Get the model.

        Returns:
            The model.

        """
        return cast("ModelInterface", self._raw_model)

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        gpus = 1.0 if self._raw_model.requires_gpu else 0.0
        return CuratorStageResource(cpus=1.0, gpus=gpus)

    @nvtx.annotate("EnhanceCaptionStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # noqa: C901
        """Process the data for the caption enhancement stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            inputs = []
            mapping: dict[int, tuple[int, int]] = {}
            idx = 0
            with self._timer.time_process(len(video.clips)):
                for clip_idx, clip in enumerate(video.clips):
                    if len(clip.windows) == 0:
                        logger.warning(f"Clip {clip.uuid} has no windows.")
                        clip.errors["windows"] = "empty"
                    for window_idx, window in enumerate(clip.windows):
                        if not window.caption:
                            logger.error(f"Clip {clip.uuid} window {window_idx} has no captions generated.")
                            clip.errors[f"window-{window_idx}"] = "empty"
                            continue
                        mapping[idx] = (clip_idx, window_idx)

                        caption = None
                        for caption_model_variant in window.caption:
                            caption = window.caption[caption_model_variant]
                            break

                        if caption is None:
                            logger.error(f"Clip {clip.uuid} window {window_idx} has no captions")
                            continue

                        captionInput = [
                            {"role": "system", "content": self._prompt},
                            {"role": "user", "content": caption},
                        ]
                        inputs.append(captionInput)
                        idx += 1

                if inputs:
                    captions = self._raw_model.generate(
                        inputs,
                        batch_size=self._batch_size,
                    )

                    for idx, result in enumerate(captions):
                        clip_idx, window_idx = mapping[idx]
                        video.clips[clip_idx].windows[window_idx].enhanced_caption[self._enhanced_key] = result
                        if self._verbose:
                            logger.info(
                                f"Enhanced {self._enhanced_key} Caption for clip {video.clips[clip_idx].uuid} "
                                f"window {window_idx}: {result}",
                            )

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks
