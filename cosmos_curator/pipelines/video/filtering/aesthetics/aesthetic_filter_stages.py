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
"""Aesthetic score filtering stages."""

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from loguru import logger

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.infra.gpu_start_helper import (
    gpu_stage_cleanup,
    gpu_stage_startup,
)
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.models.clip_aesthetics import CLIPAestheticScorer
from cosmos_curator.pipelines.video.utils.data_model import SplitPipeTask
from cosmos_curator.pipelines.video.utils.decoder_utils import (
    FrameExtractionPolicy,
    FrameExtractionSignature,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class AestheticFilterStage(CuratorStage):
    """Stage for filtering video clips based on aesthetic score.

    This class processes video clips through a series of steps including aesthetic score
    calculation and filtering based on thresholds.
    """

    def __init__(  # noqa: PLR0913
        self,
        score_threshold: float,
        reduction: Literal["mean", "min"] = "min",
        target_fps: float = 1.0,
        num_gpus_per_worker: float = 0.25,
        *,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Calculate aesthetic score over frames for each clip in task.

        Attributes:
            score_threshold: motion score threshold.
            target_fps: downsampling frames/s used for calculating aesthetic scores.
            reduction: method to reduce the frame-level aesthetic scores.
            verbose: whether to log aesthetic scores.
            log_stats: whether to log performance stats.

        """
        self._timer = StageTimer(self)
        self._score_threshold = score_threshold
        self._reduction = reduction
        self._reduce_fn: Callable = np.min  # type: ignore[type-arg]
        self._frame_extraction_signature = FrameExtractionSignature(
            extraction_policy=FrameExtractionPolicy.sequence,
            target_fps=target_fps,
        ).to_str()
        self._num_gpus_per_worker = num_gpus_per_worker
        self._verbose = verbose
        self._log_stats = log_stats
        self._model = CLIPAestheticScorer()
        self._process_count = 0

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            Resource configuration for the stage.

        """
        return CuratorStageResource(gpus=self._num_gpus_per_worker)

    def stage_setup(self) -> None:
        """Initialize stage resources and configuration."""
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)
        # real model setup
        self._model.setup()
        if self._reduction == "mean":
            self._reduce_fn = np.mean
        elif self._reduction == "min":
            self._reduce_fn = np.min
        else:
            error_msg = f"Reduction `{self._reduction}` not implemented."  # type: ignore[unreachable]
            raise NotImplementedError(error_msg)
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=False)

    def destroy(self) -> None:
        """Release the GPU-resident CLIP aesthetic scorer before the actor exits.

        Drops ``self._model`` so the underlying CLIP weights become unreachable, then
        lets ``gpu_stage_cleanup`` run ``gc.collect() + torch.cuda.empty_cache()`` to
        actually return the device memory. See ``InternVideo2EmbeddingStage.destroy``
        for the full rationale on why nulling the reference is required.
        """
        self._model = None  # type: ignore[assignment]
        gpu_stage_cleanup(self.__class__.__name__)

    @property
    def model(self) -> ModelInterface:
        """Get the model.

        Returns:
            The model.

        """
        return self._model

    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # type: ignore[override]  # noqa: C901, PLR0912
        """Score each clip's aesthetic quality and filter below threshold.

        Resolves ``extracted_frames``, pops this stage's frame signature from
        the shared dict, runs the aesthetic model to produce per-frame scores,
        and reduces them to a single clip score via the configured reduce
        function (mean or min).  Clips scoring below ``_score_threshold`` are
        moved to ``video.filtered_clips``.

        ::

            for each clip:
              no encoded_data? --> score = -1.0, record error
              |
              extracted_frames.resolve()
              frames missing for signature? --> score = -1.0, record error
              |
              frames = ef.pop(signature)  (removes key from shared dict)
              scores = model(frames)
              clip.aesthetic_score = reduce_fn(scores)
              |
              ef empty? --> extracted_frames.drop()  (last consumer)
              |
              score < threshold? --> filtered_clips (removed)
              score >= threshold? --> passed_clips (kept)

        Memory lifecycle:
            Uses ``ef.pop()`` instead of ``ef[]`` because multiple consumers
            may share the ``extracted_frames`` dict (keyed by different
            extraction signatures).  Each consumer pops its own key.  The
            last consumer to empty the dict triggers ``drop()`` to free the
            ``LazyData`` wrapper.

        Args:
            tasks: Tasks containing videos with clips to score and filter.

        Returns:
            Tasks with clips filtered by aesthetic score; rejected clips
            are moved to ``video.filtered_clips``.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            passed_clips = []
            for clip in video.clips:
                if not clip.encoded_data:
                    logger.warning(f"Clip {clip.uuid} has no encoded_data.")
                    clip.errors["encoded_data"] = "empty"
                    clip.aesthetic_score = -1.0
                else:
                    ef = clip.extracted_frames.resolve()
                    if ef is None or self._frame_extraction_signature not in ef:
                        clip.errors[f"frames-{self._frame_extraction_signature}"] = "missing"
                        error_msg = (
                            f"Clip {clip.uuid} has buffer but no extracted frames "
                            f"for {self._frame_extraction_signature}"
                        )
                        logger.error(error_msg)
                        clip.aesthetic_score = -1.0
                    else:
                        frames = ef.pop(self._frame_extraction_signature)
                        scores = self._model(frames).cpu().numpy()
                        clip.aesthetic_score = float(self._reduce_fn(scores))
                        if not ef:
                            clip.extracted_frames.drop()

                if clip.aesthetic_score < self._score_threshold:
                    video.filtered_clips.append(clip)
                    video.clip_stats.num_filtered_by_aesthetic += 1
                    if self._verbose:
                        logger.info(
                            f"Clip {clip.uuid} has aesthetic score {clip.aesthetic_score:.3f} below threshold "
                            f"{self._score_threshold}, skipped.",
                        )
                else:
                    passed_clips.append(clip)
                    if self._verbose:
                        logger.info(
                            f"Clip {clip.uuid} has aesthetic score {clip.aesthetic_score:.3f} above threshold "
                            f"{self._score_threshold}, kept.",
                        )

            video.clips = passed_clips

            if self._verbose:
                logger.info(
                    f"Video {video.input_video} chunk-{video.clip_chunk_index} has "
                    f"{len(video.clips)}/{len(video.filtered_clips)} clips "
                    "passed/filtered",
                )

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        # free memory periodically
        self._process_count += 1
        if self._process_count % 10 == 0:
            torch.cuda.empty_cache()

        return tasks
