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
"""Extraction stages using TransNetV2."""

import math
import uuid
from collections.abc import Callable, Generator
from typing import Literal

import numpy as np
import numpy.typing as npt
import nvtx  # type: ignore[import-untyped]
import torch
from loguru import logger

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.infra.gpu_start_helper import (
    gpu_stage_cleanup,
    gpu_stage_startup,
)
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.models import transnetv2
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask


class TransNetV2ClipExtractionStage(CuratorStage):
    """Stage for extracting video clips using TransNetV2.

    This class processes video clips through a series of steps including shot detection,
    scene filtering, and clip assignment.
    """

    def __init__(  # noqa: PLR0913
        self,
        threshold: float = 0.4,
        min_length_s: float | None = 2.0,
        min_length_frames: int | None = 48,
        max_length_s: float | None = 60.0,
        max_length_mode: Literal["truncate", "stride"] = "stride",
        crop_s: float | None = 0.5,
        *,
        entire_scene_as_clip: bool = True,
        num_gpus_per_worker: float = 0.25,
        limit_clips: int = 0,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """TransNetV2ClipExtractionStage constructor.

        Args:
            threshold: (float) probability threshold above which a frame is classified as a shot transition.
                Default is 0.4, which prioritizes recall over precision.
            min_length_s: (float) optional minimum length of scene, in seconds.
                If specified, will remove any scenes below this length.
            min_length_frames: (int) optional minimum length of scene, in frames.
                If specified, will remove any scenes below this length.
            max_length_s: (float) optional maximum length of scene, in seconds.
                If specified, will deal with the scene by the `max_length_mode` specified.
            max_length_mode: (str) method for dealing with scenes above the maximum specified length.
                If `truncate`, will truncate the scene to `max_length_s`.
                If `stride`, will generate a number of max_length_s scenes until the end of the scene.
                    If the end scene is less than `min_length_s`, it will drop the last scene.
            crop_s: (float) optional seconds to crop each scene at start and end.
                E.g. 0.25 will crop ~250ms from start, and ~250ms from end frame (reducing all clips by ~0.5 seconds).
                If cropped scenes result in zero-length scenes, these will be filtered.
            entire_scene_as_clip: (bool) If true, will assign the entire video as a clip if no transition is detected.
                N.B. If you are using this stage to check whether a video contains one or more transitions,
                set this to False!
            log_stats: (bool) whether to log timing statistics.
            limit_clips: (int) limit number of clips
            num_gpus_per_worker: (float) number of gpus per worker
            verbose: (bool) verbose

        """
        self._timer = StageTimer(self)
        self.threshold = threshold
        self.min_length_s = min_length_s
        self.min_length_frames = min_length_frames
        self.max_length_s = max_length_s
        if self.min_length_s and self.max_length_s and self.max_length_s < self.min_length_s:
            error_msg = "Max length is smaller than min length!"
            raise ValueError(error_msg)
        self.max_length_mode: Literal["truncate", "stride"] = max_length_mode
        self.crop_s = crop_s
        self.entire_scene_as_clip = entire_scene_as_clip
        self._num_gpus_per_worker = num_gpus_per_worker
        self._limit_clips = limit_clips
        self._verbose = verbose
        self._log_stats = log_stats
        self._model = transnetv2.TransNetV2()

    def stage_setup(self) -> None:
        """Initialize stage resources and configuration."""
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)
        self._model.setup()
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=False)

    def destroy(self) -> None:
        """Release the GPU-resident TransNetV2 model before the actor exits.

        Drops ``self._model`` first so the GPU tensors become unreachable, then lets
        ``gpu_stage_cleanup`` run ``gc.collect() + torch.cuda.empty_cache()`` to return
        the device memory to the CUDA driver. Without dropping the reference, the
        cleanup is a no-op and the worker's CUDA context lingers as ghost device
        memory after the actor exits - a problem because TransNetV2 shares the GPU
        (0.25-GPU fraction) with other stages, including the vLLM caption workers
        that need a clean GPU at startup. See ``InternVideo2EmbeddingStage.destroy``
        for the full rationale.
        """
        self._model = None  # type: ignore[assignment]
        gpu_stage_cleanup(self.__class__.__name__)

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            Resource configuration for the stage.

        """
        return CuratorStageResource(gpus=self._num_gpus_per_worker)

    @property
    def model(self) -> ModelInterface:
        """Get the TransNetV2 model.

        Returns:
            The TransNetV2 model.

        """
        return self._model

    def _get_min_length(self, framerate: float) -> int | None:
        min_length = math.ceil(self.min_length_s * framerate) if self.min_length_s is not None else None
        if self.min_length_frames is not None:
            min_length = max(min_length, self.min_length_frames) if min_length is not None else self.min_length_frames
        return min_length

    def _get_max_length(self, framerate: float) -> int | None:
        return math.ceil(self.max_length_s * framerate) if self.max_length_s is not None else None

    @nvtx.annotate("TransNetV2ClipExtractionStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # noqa: C901
        """Process video data to extract clips.

        Args:
            tasks: Tasks containing videos to process.

        Returns:
            Processed tasks with extracted clips.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            s3_file = video.input_video
            if not video.has_metadata():
                logger.warning(f"Incomplete metadata for {video.input_video}. Skipping...")
                continue
            assert video.metadata.framerate  # silence mypy
            if not video.frame_array:
                logger.warning(f"No frame array for {video.input_video}. Skipping...")
                continue

            with self._timer.time_process():
                frames = video.frame_array.resolve()
                if frames is None:
                    msg = f"frame_array resolved to None for {video.input_video}"
                    raise ValueError(msg)
                if tuple(frames.shape[1:4]) != (27, 48, 3):
                    error_msg = f"Expected frames of shape 27x48x3, got {frames.shape[1:4]}."
                    raise ValueError(error_msg)
                predictions = _get_predictions(self._model, frames, self.threshold)
                scenes = _get_scenes(predictions, entire_scene_as_clip=self.entire_scene_as_clip)
                if self._verbose:
                    logger.info(f"{video.input_video} returned {scenes.shape[0]} scenes")

                filtered_scenes = _get_filtered_scenes(
                    scenes,
                    min_length=self._get_min_length(video.metadata.framerate),
                    max_length=self._get_max_length(video.metadata.framerate),
                    max_length_mode=self.max_length_mode,
                    crop_length=(int(self.crop_s * video.metadata.framerate) if self.crop_s else None),
                )
                if self._verbose:
                    logger.info(f"{video.input_video} returned {filtered_scenes.shape[0]} filtered scenes")

                # assign information to task data struct
                for start_event, end_event in filtered_scenes:
                    clip = Clip(
                        uuid=uuid.uuid5(uuid.NAMESPACE_URL, f"{s3_file}_{start_event}_{end_event}"),
                        source_video=str(s3_file),
                        span=(
                            float(start_event) / video.metadata.framerate,
                            float(end_event) / video.metadata.framerate,
                        ),
                    )
                    video.clips.append(clip)
                    if self._limit_clips > 0 and len(video.clips) >= self._limit_clips:
                        break

                video.frame_array.drop()  # no longer needed
                if not video.clips:
                    logger.warning(f"No scene cut predicted for {s3_file}.")

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks


def _get_batches(
    frames: npt.NDArray[np.uint8],
) -> Generator[npt.NDArray[np.uint8], None, None]:
    """We fetch 100 frames, and pad the first and last batches accordingly with the first or last frame."""
    total_frames = len(frames)
    twentyfive: int = 25
    reminder = -total_frames % 50
    for i in range(0, total_frames + reminder, 50):
        start_idx = max(i - twentyfive, 0)
        end_idx = min(i + 75, total_frames)
        batch = frames[start_idx:end_idx]
        # Add padding at the beginning if necessary
        if i < twentyfive:
            padding_start = [frames[0]] * (twentyfive - i)
            batch = np.concatenate([padding_start, batch], axis=0)
        # Add padding at the end if necessary
        if end_idx > total_frames:
            padding_end = [frames[-1]] * (end_idx - total_frames)
            batch = np.concatenate([batch, padding_end], axis=0)
        yield batch


def _get_predictions(
    model: Callable[[torch.Tensor], torch.Tensor],
    frames: npt.NDArray[np.uint8],
    threshold: float,
) -> npt.NDArray[np.uint8]:
    """Get predictions from the video frame array.

    Args:
        model: shot detection model.
        frames: uint8 array of shape (# frames, height, width, 3), with RGB channels.
        threshold: probability threshold for shot detection.

    Returns:
        0/1 prediction array of shape (# frames, 1)

    """
    fndims: int = 4
    assert frames.ndim == fndims, "Expected frames tensor to have rank 4."
    predictions = []
    for batch in _get_batches(frames):
        batch_gpu = torch.from_numpy(batch.copy()).cuda()
        one_hot = model(batch_gpu.unsqueeze(0))
        predictions.append(one_hot[0, 25:75])
    predictions_ts = torch.concatenate(predictions, 0)[: len(frames)]
    return (predictions_ts > threshold).to(torch.uint8).cpu().numpy()


def _get_scenes(predictions: npt.NDArray[np.uint8], *, entire_scene_as_clip: bool) -> npt.NDArray[np.int32]:
    """Convert prediction array to scene array.

    Args:
        predictions: array of shape [# frames, 1].
            Values are 1 if frame is a shot transition, and 0 if it's not.
        entire_scene_as_clip (bool):
            If there are _no_ shot transitions found, this will make a scene spanning the whole video.

    Returns:
        scene array of shape [# scenes, 2], where the value at each row is the start and end frame of the shot.

    """
    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append((start, i))
        t_prev = t
    # If we detected at least one transition, add the trailing scene
    if scenes and t == 0:
        scenes.append((start, i))
    # If no transitions found, and entire_scene_as_clip=True, return the full video as one scene
    if not scenes and entire_scene_as_clip:
        # Use full length to match metadata.duration
        scenes.append((0, len(predictions)))
    # Return as 2D array even if empty
    return np.array(scenes, dtype=np.int32).reshape(-1, 2)


def _get_filtered_scenes(
    scenes: npt.NDArray[np.int32],
    min_length: int | None = None,
    max_length: int | None = None,
    max_length_mode: Literal["truncate", "stride"] = "truncate",
    crop_length: int | None = None,
) -> npt.NDArray[np.int32]:
    """Filter scenes.

    Args:
        scenes: integer 2D array like [[t0, t1], [t2, t3], ...]
        min_length: optional minimum length of frames a scene can have.
        max_length: optional maximum length of frames a scene can have.
        max_length_mode: how to deal with scenes that are above max length.
            If `truncate` will truncate the length of each scene by `max_length`, if specified.
            If `stride`, will generate a number of max_length scenes until the end of the scene.
                If the end scene is less than `min_length`, it will drop the last scene.
        crop_length: optional number of frames to crop from start and end of scene.
            If cropped scenes result in zero-length scenes, these will be filtered.

    Returns:
        filtered scene array.

    """
    sndims: int = 2
    if scenes.ndim != sndims:
        error_msg = "Scenes numpy array needs to be a 2D rank matrix!"
        raise ValueError(error_msg)

    if max_length is not None:
        length = scenes[:, 1] - scenes[:, 0]
        if max_length_mode == "truncate":
            scenes[:, 1] = np.minimum(scenes[:, 0] + max_length, scenes[:, 1])
        elif max_length_mode == "stride":
            new_scenes = []
            for start, end in scenes:
                new_scenes.extend(_create_spans(start, end, max_length=max_length, min_length=min_length))
            scenes = np.array(new_scenes, dtype=scenes.dtype).reshape((-1, 2))
        else:
            error_msg = f"Method `{max_length_mode}` not implemented!"  # type: ignore[unreachable]
            raise NotImplementedError(error_msg)

    if crop_length is not None:
        scenes = _crop_scenes(scenes, crop_length=crop_length)

    if min_length is not None:
        length = scenes[:, 1] - scenes[:, 0]
        scenes = scenes[length >= min_length]

    return scenes


def _crop_scenes(scenes: npt.NDArray[np.int32], crop_length: int) -> npt.NDArray[np.int32]:
    """Crop scenes by removing frames from start and end.

    Args:
        scenes: integer 2D array like [[t0, t1], [t2, t3], ...]
        crop_length: number of frames to crop from start and end of scene.

    Returns:
        cropped scene array.

    """
    cropped = np.stack([scenes[:, 0] + crop_length, scenes[:, 1] - crop_length]).T
    length = cropped[:, 1] - cropped[:, 0]
    result: npt.NDArray[np.int32] = cropped[length > 0]
    return result


def _create_spans(start: int, end: int, max_length: int, min_length: int | None) -> list[list[int]]:
    """Create spans between a start and an end point.

    Args:
        start: start point.
        end: end point.
        max_length: maximum length of span.
        min_length: minimum length of span.

    Returns:
        list of spans.

    """
    spans = []
    current_start = start

    while current_start < end:
        current_end = min(current_start + max_length, end)
        span_length = current_end - current_start

        # Check if the span meets the minimum length requirement
        if min_length and span_length < min_length and current_end == end:
            break  # Drop the span if it's the last and below min_length

        spans.append([current_start, current_end])
        current_start = current_end  # Move to the next span start

    return spans
