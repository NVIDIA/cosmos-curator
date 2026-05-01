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
"""InternVideo2 embedding stage."""

import io

import nvtx  # type: ignore[import-untyped]
import torch
from loguru import logger

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.data.ref_resolver import prefetch, resolve_as_ready
from cosmos_curator.core.utils.infra.gpu_start_helper import (
    gpu_stage_cleanup,
    gpu_stage_startup,
)
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.models.internvideo2_mm import InternVideo2MultiModality
from cosmos_curator.pipelines.video.utils.data_model import (
    Clip,
    SplitPipeTask,
)
from cosmos_curator.pipelines.video.utils.decoder_utils import (
    FrameExtractionPolicy,
    FrameExtractionSignature,
    extract_frames,
)


class InternVideo2FrameCreationStage(CuratorStage):
    """Stage for creating InternVideo2 input frames from video clips.

    This class processes video clips through a series of steps including frame extraction,
    model initialization, and input frame creation.
    """

    def __init__(
        self,
        target_fps: float = 2.0,
        *,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the InternVideo2 frame creation stage.

        Args:
            target_fps: Target frames per second for frame extraction.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        self._timer = StageTimer(self)
        self._target_fps = target_fps
        self._extraction_policy = FrameExtractionPolicy.sequence
        self._frame_extraction_signature = FrameExtractionSignature(
            extraction_policy=FrameExtractionPolicy.sequence,
            target_fps=self._target_fps,
        ).to_str()
        # utils_only set to true to skip initializing the actual model
        self._model = InternVideo2MultiModality(utils_only=True)
        self._verbose = verbose
        self._log_stats = log_stats

    @property
    def model(self) -> ModelInterface:
        """Get the InternVideo2 model.

        Returns:
            The InternVideo2 model.

        """
        return self._model

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            Resource configuration for the stage.

        """
        return CuratorStageResource(cpus=1.0)

    @nvtx.annotate("InternVideo2FrameCreationStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Resolve extracted frames and formulate InternVideo2 model input.

        For each clip, resolves the shared extracted_frames dict, retrieves
        frames for this stage's extraction signature, and passes them through
        the model's ``formulate_input_frames()``. If the clip has fewer frames
        than the model requires, re-extracts at progressively higher FPS
        (doubling until ``max_fps=20``) from the raw ``encoded_data``.

        ::

            for each clip:
              encoded_data missing? --> record error, skip
              |
              extracted_frames.resolve()
              frames missing for signature? --> record error, skip
              |
              frames.shape[0] < target?
                yes --> re-extract at 2x FPS (up to max_fps=20)
                still too few? --> log error, use what we have
              |
              model.formulate_input_frames(frames) --> clip.intern_video_2_frames
              |
              extracted_frames.drop()  (last consumer, frees heap)

        Memory lifecycle:
            ``extracted_frames`` is a ``LazyData`` wrapping a dict keyed by
            frame extraction signature.  This stage is the sole consumer of
            its key.  After formulating input frames, ``drop()`` frees the
            entire ``LazyData`` wrapper since no downstream stage needs the
            raw frames.

        Args:
            tasks: Tasks containing video clips to process.

        Returns:
            Tasks with ``intern_video_2_frames`` populated on each clip.

        """
        max_fps: int = 20

        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            prefetch([clip.encoded_data for clip in video.clips])
            for clip, data in resolve_as_ready([(clip, clip.encoded_data) for clip in video.clips]):
                if data is None:
                    clip.errors["encoded_data"] = "empty"
                    continue
                ef = clip.extracted_frames.resolve()
                if ef is None or self._frame_extraction_signature not in ef:
                    clip.errors[f"frames-{self._frame_extraction_signature}"] = "missing"
                    logger.error(
                        f"Clip {clip.uuid} has buffer but no extracted frames for {self._frame_extraction_signature}"
                    )
                    continue
                with self._timer.time_process():
                    frames = ef[self._frame_extraction_signature]
                    # check if we need re-extract
                    target_num_frames = self._model.get_target_num_frames()
                    regen_fps = self._target_fps
                    while frames.shape[0] < target_num_frames:
                        regen_fps *= 2
                        if regen_fps > max_fps:
                            logger.error(f"Clip {clip.uuid} is too short to extract enough frames.")
                            break
                        if self._verbose:
                            logger.warning(
                                f"Clip {clip.uuid} has <{target_num_frames} frames. "
                                f"Re-extracting with higher target_fps={regen_fps}. "
                                f"Current # frames={frames.shape[0]}.",
                            )
                        with io.BytesIO(data) as fp:
                            frames = extract_frames(
                                fp,
                                extraction_policy=FrameExtractionPolicy.sequence,
                                sample_rate_fps=regen_fps,
                            )
                    # create input frames for InternVideo2 model
                    clip.intern_video_2_frames = self._model.formulate_input_frames(list(frames))  # type: ignore[assignment]
                clip.extracted_frames.drop()

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks


class InternVideo2EmbeddingStage(CuratorStage):
    """Stage for generating embeddings from InternVideo2 input frames.

    This class processes video clips through a series of steps including embedding generation,
    text verification, and memory management.
    """

    def __init__(
        self,
        num_gpus_per_worker: float = 0.25,
        batch_size: int = 8,
        *,
        verbose: bool = False,
        log_stats: bool = False,
        texts_to_verify: list[str] | None = None,
    ) -> None:
        """Initialize the InternVideo2 embedding stage.

        Args:
            num_gpus_per_worker: Number of GPUs per worker.
            batch_size: Batch size for processing.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.
            texts_to_verify: Optional list of texts to verify against embeddings.

        """
        self._timer = StageTimer(self)
        self._num_gpus_per_worker = num_gpus_per_worker
        self._batch_size = batch_size
        self._verbose = verbose
        self._log_stats = log_stats
        self._texts_to_verify = texts_to_verify
        self._model = InternVideo2MultiModality()
        self._process_count = 0

    def stage_setup(self) -> None:
        """Initialize stage resources and configuration."""
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)
        self._model.setup()
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=False)

    def destroy(self) -> None:
        """Clean up resources."""
        gpu_stage_cleanup(self.__class__.__name__)

    @property
    def model(self) -> ModelInterface:
        """Get the InternVideo2 model.

        Returns:
            The InternVideo2 model.

        """
        return self._model

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            Resource configuration for the stage.

        """
        return CuratorStageResource(gpus=self._num_gpus_per_worker)

    def _verify_with_texts(self, clip: Clip) -> None:
        if self._texts_to_verify is not None and clip.intern_video_2_embedding is not None:
            text_embeddings = [self._model.get_text_embedding(x) for x in self._texts_to_verify]
            probs, idxs = self._model.evaluate(torch.from_numpy(clip.intern_video_2_embedding), text_embeddings)
            clip.intern_video_2_text_match = (self._texts_to_verify[idxs[0]], probs[0])

    @nvtx.annotate("InternVideo2EmbeddingStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Process video data to generate embeddings.

        Args:
            tasks: Tasks containing video to process.

        Returns:
            Processed task with generated embeddings.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            mapping: dict[int, int] = {}
            idx = 0
            inputs = []
            with self._timer.time_process(len(video.clips)):
                for clip_idx, clip in enumerate(video.clips):
                    iv2_frames = clip.intern_video_2_frames.resolve()
                    if iv2_frames is None:
                        clip.errors["iv2_frames"] = "none"
                        continue
                    if iv2_frames.size == 0:
                        clip.errors["iv2_frames"] = "empty"
                        continue
                    mapping[idx] = clip_idx
                    inputs.append(iv2_frames)
                    idx += 1

                if len(inputs) > 0:
                    embeddings = self._model.encode_batched_videos(inputs, self._batch_size)
                    assert len(embeddings) == len(mapping), (
                        f"Expected {len(mapping)} embeddings, but got {len(embeddings)}"
                    )
                    for idx, clip_idx in mapping.items():
                        video.clips[clip_idx].intern_video_2_embedding = embeddings[idx]

                for clip in video.clips:
                    self._verify_with_texts(clip)
                    clip.intern_video_2_frames.drop()

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        # free memory periodically
        self._process_count += 1
        if self._process_count % 10 == 0:
            torch.cuda.empty_cache()

        return tasks
