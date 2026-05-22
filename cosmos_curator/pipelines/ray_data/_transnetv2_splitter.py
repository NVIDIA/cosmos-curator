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

"""TransNetV2 clip splitter for Ray Data pipelines."""

import logging
import math
import uuid
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch

from cosmos_curator.core.utils.config.operation_context import make_pipeline_named_temporary_file
from cosmos_curator.core.utils.infra.gpu_start_helper import gpu_stage_cleanup
from cosmos_curator.models import transnetv2
from cosmos_curator.pipelines.video.clipping.frame_extraction_stages import get_frames_from_ffmpeg
from cosmos_curator.pipelines.video.clipping.transnetv2_extraction_stages import (
    _get_filtered_scenes,
    _get_predictions,
    _get_scenes,
)

logger = logging.getLogger(__name__)

_TRANSNETV2_FRAME_HEIGHT = 27
_TRANSNETV2_FRAME_WIDTH = 48


def transnetv2_model_ids() -> list[str]:
    """Return model IDs required by the Ray Data TransNetV2 splitter."""
    return transnetv2.transnetv2_model_id_names()


def _empty_clip_columns(row: dict[str, Any]) -> dict[str, Any]:
    """Return a row with no emitted clips."""
    return {**row, "clip_uuids": [], "clip_starts": [], "clip_ends": []}


def _min_length_frames(
    fps: float,
    min_length_s: float | None,
    min_length_frames: int | None,
) -> int | None:
    min_length = math.ceil(min_length_s * fps) if min_length_s is not None else None
    if min_length_frames is not None:
        min_length = max(min_length, min_length_frames) if min_length is not None else min_length_frames
    return min_length


def _max_length_frames(fps: float, max_length_s: float | None) -> int | None:
    return math.ceil(max_length_s * fps) if max_length_s is not None else None


def validate_transnetv2_length_bounds(min_length_s: float | None, max_length_s: float | None) -> None:
    """Reject impossible TransNetV2 length bounds."""
    if min_length_s is not None and max_length_s is not None and max_length_s < min_length_s:
        msg = "Max length is smaller than min length!"
        raise ValueError(msg)


def _scene_spans_to_clip_columns(
    video_path: str,
    scenes: npt.NDArray[np.int32],
    fps: float,
    limit_clips: int,
) -> tuple[list[str], list[float], list[float]]:
    if limit_clips > 0:
        scenes = scenes[:limit_clips]

    clip_uuids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_path}_{int(start)}_{int(end)}")) for start, end in scenes]
    clip_starts = [float(start) / fps for start, _ in scenes]
    clip_ends = [float(end) / fps for _, end in scenes]
    return clip_uuids, clip_starts, clip_ends


def split_transnetv2_frames(  # noqa: PLR0913
    *,
    frames: npt.NDArray[np.uint8],
    video_path: str,
    fps: float,
    model: Callable[[torch.Tensor], torch.Tensor],
    threshold: float,
    min_length_s: float | None,
    min_length_frames: int | None,
    max_length_s: float | None,
    max_length_mode: Literal["truncate", "stride"],
    crop_s: float | None,
    entire_scene_as_clip: bool,
    limit_clips: int,
) -> tuple[list[str], list[float], list[float]]:
    """Run TransNetV2 postprocessing on decoded frames and return clip columns."""
    validate_transnetv2_length_bounds(min_length_s, max_length_s)
    if tuple(frames.shape[1:4]) != (_TRANSNETV2_FRAME_HEIGHT, _TRANSNETV2_FRAME_WIDTH, 3):
        msg = f"Expected frames of shape 27x48x3, got {frames.shape[1:4]}."
        raise ValueError(msg)

    predictions = _get_predictions(model, frames, threshold)
    scenes = _get_scenes(predictions, entire_scene_as_clip=entire_scene_as_clip)
    filtered_scenes = _get_filtered_scenes(
        scenes,
        min_length=_min_length_frames(fps, min_length_s, min_length_frames),
        max_length=_max_length_frames(fps, max_length_s),
        max_length_mode=max_length_mode,
        crop_length=(int(crop_s * fps) if crop_s else None),
    )

    return _scene_spans_to_clip_columns(video_path, filtered_scenes, fps, limit_clips)


class TransNetV2Splitter:
    """Stateful Ray Data callable class that emits clip span list columns."""

    def __init__(  # noqa: PLR0913
        self,
        threshold: float = 0.4,
        min_length_s: float | None = 2.0,
        min_length_frames: int | None = 48,
        max_length_s: float | None = 60.0,
        max_length_mode: Literal["truncate", "stride"] = "stride",
        crop_s: float | None = 0.5,
        num_decode_cpus_per_worker: int = 3,
        *,
        entire_scene_as_clip: bool = True,
        limit_clips: int = 0,
    ) -> None:
        validate_transnetv2_length_bounds(min_length_s, max_length_s)
        self.threshold = threshold
        self.min_length_s = min_length_s
        self.min_length_frames = min_length_frames
        self.max_length_s = max_length_s
        self.max_length_mode = max_length_mode
        self.crop_s = crop_s
        self.entire_scene_as_clip = entire_scene_as_clip
        self.limit_clips = limit_clips
        self._num_cpu_threads = max(1, num_decode_cpus_per_worker)
        self._model: transnetv2.TransNetV2 = transnetv2.TransNetV2()
        self._model.setup()

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        video_path = str(row["video_path"])
        fps = float(row["fps"])
        if not math.isfinite(fps) or fps <= 0:
            logger.warning("Invalid FPS %s for %s; emitting no clips", row["fps"], video_path)
            return _empty_clip_columns(row)

        frames = self._decode_frames(row)
        if frames is None or len(frames) == 0:
            logger.warning("TransNetV2 frame extraction failed for %s; emitting no clips", video_path)
            return _empty_clip_columns(row)

        clip_uuids, clip_starts, clip_ends = split_transnetv2_frames(
            frames=frames,
            video_path=video_path,
            fps=fps,
            model=self._model,
            threshold=self.threshold,
            min_length_s=self.min_length_s,
            min_length_frames=self.min_length_frames,
            max_length_s=self.max_length_s,
            max_length_mode=self.max_length_mode,
            crop_s=self.crop_s,
            entire_scene_as_clip=self.entire_scene_as_clip,
            limit_clips=self.limit_clips,
        )

        if not clip_uuids:
            logger.warning("No scene cut predicted for %s", video_path)

        return {
            **row,
            "clip_uuids": clip_uuids,
            "clip_starts": clip_starts,
            "clip_ends": clip_ends,
        }

    def __del__(self) -> None:
        if hasattr(self, "_model"):
            del self._model
        gpu_stage_cleanup(self.__class__.__name__)

    def _decode_frames(self, row: dict[str, Any]) -> npt.NDArray[np.uint8] | None:
        video_bytes = bytes(row["video_bytes"])
        with make_pipeline_named_temporary_file(sub_dir="ray_data_transnetv2", suffix=".mp4") as video_path:
            video_path.write_bytes(video_bytes)
            return get_frames_from_ffmpeg(
                video_path,
                width=_TRANSNETV2_FRAME_WIDTH,
                height=_TRANSNETV2_FRAME_HEIGHT,
                num_cpu_threads=self._num_cpu_threads,
            )
