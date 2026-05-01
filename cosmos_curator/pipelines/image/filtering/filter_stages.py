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

"""CPU post-processing stages for image semantic filtering and classification."""

import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.pipelines.common.filter_prompts import IMAGE_FILTER_CRITERIA, IMAGE_TYPE_LABELS
from cosmos_curator.pipelines.common.semantic_filter_postprocess import (
    ClassifierEvaluationConfig,
    evaluate_classifier_window_results,
    evaluate_semantic_window_results,
    parse_comma_separated_types,
)
from cosmos_curator.pipelines.image.utils.data_model import ImagePipeTask


def _mark_missing_filter_caption(
    task: ImagePipeTask,
    *,
    model_variant: str,
    filter_caption_key: str,
    rejection_stage: str,
    score_only: bool = False,
) -> None:
    """Handle tasks whose filter caption text is missing."""
    image = task.image
    status = image.filter_caption_status.get(filter_caption_key)
    if status in {"blocked", "error"}:
        image.qwen_rejection_stage = None if score_only else rejection_stage
        image.is_filtered = not score_only
        return

    image.errors[model_variant] = "all_windows_failed_preparation"
    image.qwen_rejection_stage = None if score_only else rejection_stage
    image.is_filtered = not score_only


class ImageSemanticFilterStage(CuratorStage):
    """CPU post-processing stage for image semantic filtering."""

    def __init__(  # noqa: PLR0913
        self,
        model_variant: str,
        user_prompt: str | None = None,
        filter_variant: str = "default",
        rejection_threshold: float = 0.5,
        *,
        filter_caption_key: str | None = None,
        criteria_by_variant: dict[str, list[str]] | None = None,
        score_only: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the image semantic filter stage."""
        self._timer = StageTimer(self)
        self._model_variant = model_variant
        self._filter_caption_key = filter_caption_key or model_variant
        self._filter_variant = filter_variant
        self._rejection_threshold = rejection_threshold
        self._user_prompt = user_prompt
        self._criteria_by_variant = criteria_by_variant or IMAGE_FILTER_CRITERIA
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

    @nvtx.annotate("ImageSemanticFilterStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[ImagePipeTask]) -> list[ImagePipeTask] | None:
        """Apply semantic filtering decisions to image tasks using generated filter captions."""
        filter_criteria = (
            self._criteria_by_variant[self._filter_variant]
            if self._user_prompt is None
            else [s.strip() for s in self._user_prompt.split(",") if s.strip()]
        )
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            image = task.image
            if image.is_filtered:
                continue
            with self._timer.time_process():
                caption = image.filter_captions.get(self._filter_caption_key)
                if caption is None:
                    _mark_missing_filter_caption(
                        task,
                        model_variant=self._model_variant,
                        filter_caption_key=self._filter_caption_key,
                        rejection_stage="semantic",
                        score_only=self._score_only,
                    )
                else:
                    clip_should_pass, all_issues, _, per_window_errors = evaluate_semantic_window_results(
                        [(0, caption)],
                        filter_criteria=filter_criteria,
                        rejection_threshold=self._rejection_threshold,
                        score_only=self._score_only,
                    )
                    if per_window_errors:
                        image.errors[self._model_variant] = next(iter(per_window_errors.values()))
                    image.is_filtered = not clip_should_pass
                    image.qwen_rejection_stage = "semantic" if not clip_should_pass else None
                    if all_issues and (self._score_only or not clip_should_pass):
                        new_reasons = dict.fromkeys(sorted(all_issues), "yes")
                        image.qwen_rejection_reasons = {**(image.qwen_rejection_reasons or {}), **new_reasons}
                    if self._verbose and not clip_should_pass:
                        logger.info(f"Image {task.session_id} filtered out due to: {set(all_issues)}")
            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats
        return tasks


class ImageClassifierStage(CuratorStage):
    """CPU post-processing stage for image type classification/filtering."""

    def __init__(  # noqa: PLR0913
        self,
        model_variant: str,
        rejection_threshold: float = 0.5,
        *,
        filter_caption_key: str | None = None,
        type_allow: str | None = None,
        type_block: str | None = None,
        custom_categories: bool = False,
        valid_type_labels: tuple[str, ...] = IMAGE_TYPE_LABELS,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the image classifier stage."""
        self._timer = StageTimer(self)
        self._model_variant = model_variant
        self._filter_caption_key = filter_caption_key or model_variant
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
            self._valid_type_labels = tuple(sorted(combined))
        else:
            valid = set(valid_type_labels)
            for t in self._type_allow + self._type_block:
                if t not in valid:
                    msg = f"Unknown image type {t!r}; must be one of {sorted(valid)}"
                    raise ValueError(msg)
            self._valid_type_labels = valid_type_labels

    @property
    def resources(self) -> CuratorStageResource:
        """Return CPU resource requirements for this stage."""
        return CuratorStageResource(cpus=1.0)

    @property
    def conda_env_name(self) -> str | None:
        """Run in the default environment because this stage is CPU-only."""
        return None

    @nvtx.annotate("ImageClassifierStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[ImagePipeTask]) -> list[ImagePipeTask] | None:
        """Apply classifier filtering decisions to image tasks using generated filter captions."""
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            image = task.image
            if image.is_filtered:
                continue
            with self._timer.time_process():
                caption = image.filter_captions.get(self._filter_caption_key)
                if caption is None:
                    _mark_missing_filter_caption(
                        task,
                        model_variant=self._model_variant,
                        filter_caption_key=self._filter_caption_key,
                        rejection_stage="classifier",
                    )
                else:
                    clip_should_pass, all_issues, per_window_reasons, per_window_errors, classification = (
                        evaluate_classifier_window_results(
                            [(0, caption)],
                            config=ClassifierEvaluationConfig(
                                type_allow=self._type_allow,
                                type_block=self._type_block,
                                custom_categories=self._custom_categories,
                                valid_type_labels=self._valid_type_labels,
                                rejection_threshold=self._rejection_threshold,
                            ),
                        )
                    )
                    if per_window_errors:
                        image.errors[self._model_variant] = next(iter(per_window_errors.values()))
                    image.qwen_type_classification = classification
                    image.is_filtered = not clip_should_pass
                    if not clip_should_pass:
                        image.qwen_rejection_stage = "classifier"
                        new_reasons = per_window_reasons.get(0) or {}
                        if new_reasons:
                            image.qwen_rejection_reasons = {**(image.qwen_rejection_reasons or {}), **new_reasons}
                    if self._verbose and image.qwen_type_classification:
                        logger.info(f"Image {task.session_id} type classification: {image.qwen_type_classification}")
                    if self._verbose and not clip_should_pass:
                        logger.info(
                            f"Image {task.session_id} filtered out due to: {set(all_issues)} "
                            f"(classified as: {image.qwen_type_classification or []})"
                        )
            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats
        return tasks
