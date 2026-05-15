# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Artificial text (overlay/post-production) filter stage."""

from typing import cast

from loguru import logger

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource, PipelineTask
from cosmos_curator.core.utils.infra.gpu_start_helper import gpu_stage_cleanup, gpu_stage_startup
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.models.paddle_ocr import (
    CORNER_X_MARGIN_NORM,
    CORNER_Y_MARGIN_NORM,
    MIN_DURATION_FRAMES,
    MIN_DURATION_FRAMES_CORNER_RATIO,
    STABILITY_IOU_CONSECUTIVE_THRESHOLD,
    ArtificialTextDetector,
    PaddleOCRModel,
)
from cosmos_curator.pipelines.video.utils.data_model import SplitPipeTask


class ArtificialTextFilterStage(CuratorStage):
    """Filter clips that contain overlay/artificial text (e.g. post-production captions).

    Uses PaddleOCR detection plus heuristics (stable text tracks, corner text) to
    classify clips with artificial text; those clips are moved to filtered_clips.
    """

    def __init__(  # noqa: PLR0913
        self,
        num_gpus_per_worker: float = 0.25,
        *,
        use_gpu: bool = True,
        use_corner_detection: bool = True,
        frame_interval: int = 3,
        min_duration_frames: int = MIN_DURATION_FRAMES,
        min_duration_frames_corner_ratio: float = MIN_DURATION_FRAMES_CORNER_RATIO,
        stability_iou_threshold: float = STABILITY_IOU_CONSECUTIVE_THRESHOLD,
        ignore_corner_region: bool = False,
        corner_x_margin_norm: float = CORNER_X_MARGIN_NORM,
        corner_y_margin_norm: float = CORNER_Y_MARGIN_NORM,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the artificial text filter stage with optional corner detection."""
        self._timer = StageTimer(self)
        self._num_gpus_per_worker = num_gpus_per_worker
        self._use_gpu = use_gpu
        self._use_corner_detection = use_corner_detection
        self._frame_interval = frame_interval
        self._ignore_corner_region = ignore_corner_region
        self._corner_x_margin_norm = corner_x_margin_norm
        self._corner_y_margin_norm = corner_y_margin_norm
        self._min_duration_frames = min_duration_frames
        self._min_duration_frames_corner_ratio = min_duration_frames_corner_ratio
        self._stability_iou_threshold = stability_iou_threshold
        self._verbose = verbose
        self._log_stats = log_stats
        self._model = PaddleOCRModel(frame_interval=frame_interval, use_gpu=use_gpu)

    @property
    def resources(self) -> CuratorStageResource:
        """Return the GPU/CPU resource requirement for this stage."""
        return CuratorStageResource(gpus=self._num_gpus_per_worker)

    def stage_setup(self) -> None:
        """Load the PaddleOCR model once per actor; use built-in GPU startup/cleanup."""
        if self._use_gpu:
            gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)
        self._model.setup()
        if self._use_gpu:
            gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=False)

    def destroy(self) -> None:
        """Release the PaddleOCR model before the actor exits.

        When running on GPU we drop ``self._model`` first so the underlying CUDA
        tensors become unreachable, then run the standard ``gpu_stage_cleanup`` to
        return device memory to the driver. See ``InternVideo2EmbeddingStage.destroy``
        for the rationale.
        """
        if self._use_gpu:
            self._model = None  # type: ignore[assignment]
            gpu_stage_cleanup(self.__class__.__name__)

    @property
    def model(self) -> ModelInterface:
        """Return the model interface for weight download."""
        return self._model

    def process_data(  # noqa: C901, PLR0912, PLR0915
        self, tasks: list[PipelineTask]
    ) -> list[PipelineTask] | None:
        """Run artificial text detection on each clip and move positive clips to filtered_clips."""
        split_tasks = cast("list[SplitPipeTask]", tasks)
        for task in split_tasks:
            self._timer.reinit(self, task.get_major_size())
            for video in task.videos:
                passed_clips = []
                for clip in video.clips:
                    if not clip.encoded_data:
                        logger.warning(f"Clip {clip.uuid} has no encoded_data; skipping artificial text check.")
                        clip.errors["artificial_text"] = "no_encoded_data"
                        clip.has_artificial_text = False
                        passed_clips.append(clip)
                        continue
                    try:
                        meta = clip.extract_metadata()
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"Clip {clip.uuid} failed to extract metadata: {e}")
                        clip.errors["artificial_text"] = str(e)
                        clip.has_artificial_text = False
                        passed_clips.append(clip)
                        continue
                    if meta is None:
                        clip.has_artificial_text = False
                        passed_clips.append(clip)
                        continue
                    height = meta.get("height") or 0
                    width = meta.get("width") or 0
                    fps = meta.get("framerate") or 30.0
                    if height <= 0 or width <= 0:
                        clip.errors["artificial_text"] = "invalid dimensions"
                        clip.has_artificial_text = False
                        passed_clips.append(clip)
                        continue
                    detector = ArtificialTextDetector(
                        frame_height=height,
                        frame_width=width,
                        fps=fps,
                        use_corner_detection=self._use_corner_detection,
                        frame_interval=self._frame_interval,
                        min_duration_frames=self._min_duration_frames,
                        min_duration_frames_corner_ratio=self._min_duration_frames_corner_ratio,
                        stability_iou_threshold=self._stability_iou_threshold,
                        ignore_corner_region=self._ignore_corner_region,
                        corner_x_margin_norm=self._corner_x_margin_norm,
                        corner_y_margin_norm=self._corner_y_margin_norm,
                    )
                    # Resolve LazyData and convert to bytes (same pattern as download_stages)
                    resolved = clip.encoded_data.resolve() if hasattr(clip.encoded_data, "resolve") else None
                    video_bytes = resolved.tobytes() if resolved is not None else b""
                    ocr_results = self._model.generate_single(video_bytes)
                    segments = detector.detect(ocr_results)
                    has_artificial = len(segments) > 0
                    clip.has_artificial_text = has_artificial
                    clip.artificial_text_segments = segments if has_artificial else None
                    if has_artificial:
                        video.filtered_clips.append(clip)
                        video.clip_stats.num_filtered_by_artificial_text += 1
                        if self._verbose:
                            logger.info(f"Clip {clip.uuid} has artificial text ({len(segments)} segment(s)), filtered.")
                    else:
                        passed_clips.append(clip)
                        if self._verbose:
                            logger.info(f"Clip {clip.uuid} has no artificial text, kept.")
                video.clips = passed_clips
                if self._verbose:
                    logger.info(
                        f"Video {video.input_video} chunk-{video.clip_chunk_index} has "
                        f"{len(video.clips)}/{len(video.filtered_clips)} clips passed/filtered after artificial text."
                    )
            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats
        return cast("list[PipelineTask]", split_tasks)
