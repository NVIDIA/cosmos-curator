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

"""Motion Filter Stage."""

import io
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import nvtx  # type: ignore[import-untyped]
import torch
from loguru import logger

import cosmos_curator.pipelines.video.filtering.motion.motion_vector_backend as motion
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.data.ref_resolver import prefetch, resolve_as_ready
from cosmos_curator.core.utils.infra.gpu_start_helper import (
    gpu_stage_cleanup,
    gpu_stage_startup,
)
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.pipelines.video.utils.data_model import SplitPipeTask

ScoringFunctionType = Callable[[bytes, npt.NDArray[np.uint8]], npt.NDArray[np.float32]]


class MotionVectorDecodeStage(CuratorStage):
    """Stage for decoding motion vector information from video files.

    This class processes video files through a series of steps including decoding,
    filtering by side length, and storing the results in the task.
    """

    def __init__(
        self,
        num_cpus_per_worker: float,
        *,
        verbose: bool = False,
        log_stats: bool = False,
        target_fps: float = 2.0,
        target_duration_ratio: float = 0.5,
    ) -> None:
        """Stage for decoding motion vector information from video files.

        Attributes:
            num_cpus_per_worker: number of CPUs per worker.
            target_fps: target frames per second to sample (lower is faster).
            target_duration_ratio: ratio of video duration to sample (0.5 = 50% by default).
            verbose: whether to log verbose information.
            log_stats: whether to log performance statistics.

        """
        self._timer = StageTimer(self)
        self._num_cpus_per_worker = num_cpus_per_worker
        self._num_threads = max(1, int(num_cpus_per_worker) + 1)
        self._target_fps = target_fps
        self._target_duration_ratio = target_duration_ratio
        self._verbose = verbose
        self._log_stats = log_stats

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            Resource configuration for the stage.

        """
        return CuratorStageResource(cpus=self._num_cpus_per_worker)

    @nvtx.annotate("MotionVectorDecodeStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # noqa: C901
        """Decode data for motion vector computation and filter by side length."""
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            prefetch([clip.encoded_data for clip in video.clips])
            for clip, data in resolve_as_ready([(clip, clip.encoded_data) for clip in video.clips]):
                if data is None:
                    logger.warning(f"Clip {clip.uuid} has no encoded_data.")
                    clip.errors["encoded_data"] = "empty"
                    continue
                with self._timer.time_process(), io.BytesIO(data) as fp:
                    try:
                        clip.decoded_motion_data = motion.decode_for_motion(
                            fp,
                            thread_count=int(self._num_threads),
                            target_fps=self._target_fps,
                            target_duration_ratio=self._target_duration_ratio,
                        )
                    except motion.VideoResolutionTooSmallError:
                        if self._verbose:
                            logger.warning(f"Clip {clip.uuid} has too small resolution.")
                        clip.decoded_motion_data = None
                        clip.errors["motion_decode"] = "resolution_too_small"
                    except Exception as e:  # noqa: BLE001
                        if self._verbose:
                            logger.exception(f"Clip {clip.uuid} failed to decode motion data: {e}")
                        clip.decoded_motion_data = None
                        clip.errors["motion_decode"] = "decode_failed"
                    else:
                        if clip.decoded_motion_data is None or len(clip.decoded_motion_data.frames) == 0:
                            logger.error(f"Clip {clip.uuid} has no motion frames.")
                            clip.decoded_motion_data = None
                            clip.errors["motion_decode"] = "no_motion_frames"
            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks


class MotionFilterStage(CuratorStage):
    """Stage for filtering video clips based on motion score.

    This class processes video clips through a series of steps including motion score
    computation and filtering based on thresholds.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        score_only: bool,
        global_mean_threshold: float,
        per_patch_min_256_threshold: float,
        num_gpus_per_worker: float = 0.5,
        batch_size: int = 64,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Filter clips in the task based on motion score over frames.

        Attributes:
            score_only: whether to only compute motion score without filtering.
            global_mean_threshold: global mean threshold for motion score.
            per_patch_min_256_threshold: per-patch min threshold for motion score.
            num_gpus_per_worker: number of GPUs per worker.
            batch_size: batch size for motion score computation.
            verbose: whether to log verbose information.
            log_stats: whether to log performance statistics.

        """
        self._timer = StageTimer(self)
        self._score_only = score_only
        self._global_mean_threshold = global_mean_threshold
        self._per_patch_min_256_threshold = per_patch_min_256_threshold
        self._num_gpus_per_worker = num_gpus_per_worker
        self._batch_size = batch_size
        self._verbose = verbose
        self._log_stats = log_stats
        self._process_count = 0

    def stage_setup(self) -> None:
        """Initialize stage resources and configuration."""
        gpu_stage_startup(self.__class__.__name__, self._num_gpus_per_worker, pre_setup=True)

    def destroy(self) -> None:
        """Clean up resources."""
        gpu_stage_cleanup(self.__class__.__name__)

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            Resource configuration for the stage.

        """
        return CuratorStageResource(gpus=self._num_gpus_per_worker)

    @nvtx.annotate("MotionFilterStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # noqa: C901, PLR0912
        """Process video data to filter clips based on motion score.

        Args:
            tasks: Tasks containing videos to process.

        Returns:
            Processed tasks with filtered clips.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            passing_clips = []
            for clip in video.clips:
                if not clip.decoded_motion_data:
                    if self._verbose:
                        logger.warning(f"Clip {clip.uuid} has no decoded motion data.")
                    fake_score = -1.0
                    motion_info = motion.MotionInfo(
                        fake_score < self._global_mean_threshold or fake_score < self._per_patch_min_256_threshold,
                        fake_score,
                        fake_score,
                    )
                else:
                    with self._timer.time_process():
                        motion_info = motion.check_if_small_motion(
                            clip.decoded_motion_data.frames,
                            clip.decoded_motion_data.frame_size,
                            global_mean_threshold=self._global_mean_threshold,
                            per_patch_min_256_threshold=self._per_patch_min_256_threshold,
                            use_gpu=self._num_gpus_per_worker > 0,
                            batch_size=self._batch_size,
                        )

                clip.decoded_motion_data = None
                clip.motion_score_global_mean = motion_info.global_mean
                clip.motion_score_per_patch_min_256 = motion_info.per_patch_min_256
                if motion_info.is_small_motion:
                    if self._score_only:
                        passing_clips.append(clip)
                    else:
                        video.filtered_clips.append(clip)
                        video.clip_stats.num_filtered_by_motion += 1
                    if self._verbose:
                        logger.info(
                            f"Clip {clip.uuid} has motion score global mean {clip.motion_score_global_mean:.5f}"
                            f"<{self._global_mean_threshold} or per-patch min 256 "
                            f"{clip.motion_score_per_patch_min_256:.6f}<{self._per_patch_min_256_threshold}, "
                            f"skipped.",
                        )
                else:
                    passing_clips.append(clip)
                    if self._verbose:
                        logger.info(
                            f"Clip {clip.uuid} has motion score global mean {clip.motion_score_global_mean:.5f}"
                            f">={self._global_mean_threshold} and per-patch min 256 "
                            f"{clip.motion_score_per_patch_min_256:.6f}>={self._per_patch_min_256_threshold}, "
                            f"kept.",
                        )
            video.clips = passing_clips

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
