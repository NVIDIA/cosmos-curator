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
"""Aesthetic score filtering stages."""

from collections.abc import Iterable

import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.pipelines.common.filter_prompts import FILTER_CRITERIA, VIDEO_TYPE_LABELS
from cosmos_curator.pipelines.common.semantic_filter_postprocess import (
    ClassifierEvaluationConfig,
    evaluate_classifier_window_results,
    evaluate_semantic_window_results,
    parse_comma_separated_types,
)
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video


class VllmFilteringStage(CuratorStage):
    """CPU post-processing stage for semantic (criteria-based) filtering."""

    def __init__(  # noqa: PLR0913
        self,
        model_variant: str,
        user_prompt: str | None = None,
        filter_variant: str = "default",
        rejection_threshold: float = 0.5,
        *,
        score_only: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the video semantic filtering stage."""
        self._timer = StageTimer(self)
        self._model_variant = model_variant
        self._filter_variant = filter_variant
        self._rejection_threshold = rejection_threshold
        self._user_prompt = user_prompt
        self._score_only = score_only
        self._verbose = verbose
        self._log_stats = log_stats

    @property
    def resources(self) -> CuratorStageResource:
        """Return CPU resource requirements for this stage."""
        return CuratorStageResource(cpus=1.0)

    @property
    def conda_env_name(self) -> str | None:
        """Run in the default environment because this stage is CPU-only."""
        return None

    @nvtx.annotate("VllmFilteringStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Filter video clips using captions already written on filter windows."""
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            mapping: dict[int, tuple[int, int]] = {}
            results: list[str] = []
            idx = 0
            with self._timer.time_process(len(video.clips)):
                for clip_idx, clip in enumerate(video.clips):
                    if not clip.filter_windows:
                        logger.warning(f"Clip {clip.uuid} has no windows.")
                        clip.errors["windows"] = "empty"
                        continue
                    for window_idx, window in enumerate(clip.filter_windows):
                        caption = window.caption.get(self._model_variant)
                        if caption is None:
                            logger.error(
                                f"Clip {clip.uuid} window {window_idx} has no caption for {self._model_variant!r}."
                            )
                            clip.errors[f"window-{window_idx}"] = "no_caption"
                            continue
                        mapping[idx] = (clip_idx, window_idx)
                        results.append(caption)
                        idx += 1

                self._filter_clips(video, mapping, enumerate(results))

            if self._verbose:
                logger.info(
                    f"Video {video.input_video} chunk-{video.clip_chunk_index} has "
                    f"{len(video.clips)}/{len(video.filtered_clips)} clips passed/filtered"
                )

        if self._log_stats:
            stage_name, stage_perf_stats = self._timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats
        return tasks

    def _filter_clips(
        self,
        video: Video,
        mapping: dict[int, tuple[int, int]],
        captions: Iterable[tuple[int, str]],
    ) -> None:
        clip_results: dict[int, list[tuple[int, str]]] = {}
        for idx, result in captions:
            clip_idx, window_idx = mapping[idx]
            clip_results.setdefault(clip_idx, []).append((window_idx, result))

        filter_criteria = (
            FILTER_CRITERIA[self._filter_variant]
            if self._user_prompt is None
            else [s.strip() for s in self._user_prompt.split(",") if s.strip()]
        )

        passing_clips: list[Clip] = []
        for clip_idx, window_results in clip_results.items():
            self._process_semantic_clip(video, clip_idx, window_results, filter_criteria, passing_clips)

        original_clips = list(video.clips)

        for clip_idx in range(len(original_clips)):
            if clip_idx not in clip_results:
                self._handle_unmapped_semantic_clip(original_clips[clip_idx], video, passing_clips)
        video.clips = passing_clips
        for clip in video.clips + video.filtered_clips:
            for window in clip.filter_windows:
                window.model_input.clear()
                window.mp4_bytes.drop()

    def _process_semantic_clip(
        self,
        video: Video,
        clip_idx: int,
        window_results: list[tuple[int, str]],
        filter_criteria: list[str],
        passing_clips: list[Clip],
    ) -> None:
        """Evaluate semantic results for one clip and update clip/window metadata."""
        clip_should_pass, all_issues, per_window_reasons, per_window_errors = evaluate_semantic_window_results(
            window_results,
            filter_criteria=filter_criteria,
            rejection_threshold=self._rejection_threshold,
            score_only=self._score_only,
        )
        clip = video.clips[clip_idx]
        for window_idx, reasons in per_window_reasons.items():
            clip.filter_windows[window_idx].caption["qwen_rejection_reasons"] = str(reasons)
        for window_idx, error in per_window_errors.items():
            clip.filter_windows[window_idx].errors["qwen"] = error

        if clip_should_pass:
            passing_clips.append(clip)
            return

        if self._verbose:
            logger.info(f"Clip {clip.uuid} filtered out due to: {set(all_issues)}")
        clip.qwen_rejection_stage = "semantic"
        video.filtered_clips.append(clip)
        video.clip_stats.num_filtered_by_qwen_semantic += 1

    def _handle_unmapped_semantic_clip(self, clip: Clip, video: Video, passing_clips: list[Clip]) -> None:
        """Handle clips that had no successfully mapped semantic-filter windows."""
        clip.errors["qwen"] = "all_windows_failed_preparation"
        if self._score_only:
            passing_clips.append(clip)
            logger.warning(f"Clip {clip.uuid} had no successfully mapped windows during score-only mode; kept in clips")
            return

        clip.qwen_rejection_stage = "semantic"
        video.filtered_clips.append(clip)
        video.clip_stats.num_filtered_by_qwen_semantic += 1
        logger.warning(f"Clip {clip.uuid} had no successfully mapped windows; added to filtered_clips")


class VllmVideoClassifierStage(CuratorStage):
    """CPU post-processing stage for video-type (allow/block list) classification."""

    def __init__(  # noqa: PLR0913
        self,
        model_variant: str,
        rejection_threshold: float = 0.5,
        *,
        type_allow: str | None = None,
        type_block: str | None = None,
        custom_categories: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
        clear_model_input_after: bool = True,
    ) -> None:
        """Initialize the video classifier stage."""
        self._timer = StageTimer(self)
        self._model_variant = model_variant
        self._clear_model_input_after = clear_model_input_after
        self._rejection_threshold = rejection_threshold
        self._verbose = verbose
        self._log_stats = log_stats
        self._type_allow = [t.replace(" ", "_") for t in parse_comma_separated_types(type_allow)]
        self._type_block = [t.replace(" ", "_") for t in parse_comma_separated_types(type_block)]
        self._custom_categories = custom_categories
        if custom_categories:
            combined = set(self._type_allow) | set(self._type_block)
            if not combined:
                msg = "custom_categories=True requires at least one of type_allow or type_block to be set"
                raise ValueError(msg)
            self._valid_type_labels: tuple[str, ...] = tuple(sorted(combined))
        else:
            valid = set(VIDEO_TYPE_LABELS)
            for t in self._type_allow + self._type_block:
                if t not in valid:
                    msg = f"Unknown video type {t!r}; must be one of VIDEO_TYPE_LABELS: {sorted(valid)}"
                    raise ValueError(msg)
            self._valid_type_labels = VIDEO_TYPE_LABELS

    @property
    def resources(self) -> CuratorStageResource:
        """Return CPU resource requirements for this stage."""
        return CuratorStageResource(cpus=1.0)

    @property
    def conda_env_name(self) -> str | None:
        """Run in the default environment because this stage is CPU-only."""
        return None

    @nvtx.annotate("VllmVideoClassifierStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Classify and optionally filter video clips using filter-window captions."""
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            mapping: dict[int, tuple[int, int]] = {}
            results: list[str] = []
            idx = 0
            with self._timer.time_process(len(video.clips)):
                for clip_idx, clip in enumerate(video.clips):
                    if not clip.filter_windows:
                        logger.warning(f"Clip {clip.uuid} has no windows.")
                        clip.errors["windows"] = "empty"
                        continue
                    for window_idx, window in enumerate(clip.filter_windows):
                        caption = window.caption.get(self._model_variant)
                        if caption is None:
                            logger.error(
                                f"Clip {clip.uuid} window {window_idx} has no caption for {self._model_variant!r}."
                            )
                            clip.errors[f"window-{window_idx}"] = "no_caption"
                            continue
                        mapping[idx] = (clip_idx, window_idx)
                        results.append(caption)
                        idx += 1

                self._filter_clips(video, mapping, enumerate(results))

            if self._verbose:
                logger.info(
                    f"Video {video.input_video} chunk-{video.clip_chunk_index} has "
                    f"{len(video.clips)}/{len(video.filtered_clips)} clips passed/filtered"
                )

        if self._log_stats:
            stage_name, stage_perf_stats = self._timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats
        return tasks

    def _type_mode_filter_clip(
        self,
        video: Video,
        clip_idx: int,
        window_results: list[tuple[int, str]],
        all_issues: set[str],
    ) -> bool:
        clip_should_pass, issues, per_window_reasons, per_window_errors, classification = (
            evaluate_classifier_window_results(
                window_results,
                config=ClassifierEvaluationConfig(
                    type_allow=self._type_allow,
                    type_block=self._type_block,
                    custom_categories=self._custom_categories,
                    valid_type_labels=self._valid_type_labels,
                    rejection_threshold=self._rejection_threshold,
                ),
            )
        )
        all_issues.update(issues)
        for window_idx, reasons in per_window_reasons.items():
            video.clips[clip_idx].filter_windows[window_idx].caption["qwen_rejection_reasons"] = str(reasons)
        for window_idx, error in per_window_errors.items():
            video.clips[clip_idx].filter_windows[window_idx].errors["qwen"] = error
        clip = video.clips[clip_idx]
        clip.qwen_type_classification = classification
        if self._verbose and clip.qwen_type_classification:
            logger.info(f"Clip {clip.uuid} type classification: {clip.qwen_type_classification}")
        return clip_should_pass

    def _filter_clips(
        self,
        video: Video,
        mapping: dict[int, tuple[int, int]],
        captions: Iterable[tuple[int, str]],
    ) -> None:
        clip_results: dict[int, list[tuple[int, str]]] = {}
        for idx, result in captions:
            clip_idx, window_idx = mapping[idx]
            clip_results.setdefault(clip_idx, []).append((window_idx, result))

        passing_clips = []
        for clip_idx, window_results in clip_results.items():
            all_issues: set[str] = set()
            clip_should_pass = self._type_mode_filter_clip(video, clip_idx, window_results, all_issues)

            if not clip_should_pass:
                clip = video.clips[clip_idx]
                clip.qwen_rejection_stage = "classifier"
                if self._verbose:
                    logger.info(
                        f"Clip {clip.uuid} filtered out due to: {set(all_issues)} "
                        f"(classified as: {clip.qwen_type_classification or []})"
                    )
                video.filtered_clips.append(clip)
                video.clip_stats.num_filtered_by_qwen_classifier += 1
            else:
                passing_clips.append(video.clips[clip_idx])

        original_clips = list(video.clips)
        video.clips = passing_clips

        for clip_idx in range(len(original_clips)):
            if clip_idx not in clip_results:
                clip = original_clips[clip_idx]
                clip.errors["qwen"] = "all_windows_failed_preparation"
                clip.qwen_rejection_stage = "classifier"
                video.filtered_clips.append(clip)
                video.clip_stats.num_filtered_by_qwen_classifier += 1
                logger.warning(f"Clip {clip.uuid} had no successfully mapped windows; added to filtered_clips")
        if self._clear_model_input_after:
            for clip in video.clips + video.filtered_clips:
                for window in clip.filter_windows:
                    window.model_input.clear()
                    window.mp4_bytes.drop()
