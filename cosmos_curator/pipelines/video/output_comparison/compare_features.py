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
"""Feature comparison for split pipeline outputs."""

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, cast

import attrs
import ray
from loguru import logger
from ray.data import ActorPoolStrategy, TaskPoolStrategy

from cosmos_curator.pipelines.video.output_comparison.json_types import JsonDictObject, JsonValue
from cosmos_curator.pipelines.video.output_comparison.report import FeatureComparison, Issue
from cosmos_curator.pipelines.video.output_comparison.summary_loader import OutputRoot
from cosmos_curator.pipelines.video.output_comparison.summary_schema import OutputSummary
from cosmos_curator.pipelines.video.output_comparison.video_planning import (
    DEFAULT_PROFILE_NAME,
    VideoComparisonResult,
    build_video_comparison_specs,
)
from cosmos_curator.pipelines.video.output_comparison.video_schema import ClipComparisonSpec, VideoComparisonSpec


@attrs.define(frozen=True)
class FeatureComparisonContext:
    """Planning data passed to each feature comparison planner.

    This contains the two output roots, loaded summaries, selected video specs,
    storage profile, and optional selectors. A feature uses it to decide whether
    summary data is enough or clip-level Ray Data work is needed.
    """

    output_a: OutputRoot
    output_b: OutputRoot
    summary_a: OutputSummary
    summary_b: OutputSummary
    profile_name: str
    specs: tuple[VideoComparisonSpec, ...]
    video_limit: int | None = None
    selected_video_key: str | None = None


@attrs.define(frozen=True)
class FeatureComparisonResult:
    """Issues and report data emitted by one feature comparison planner."""

    issues: tuple[Issue, ...]
    comparison: FeatureComparison


@attrs.define(frozen=True)
class ResolvedFeaturePlan:
    """Feature comparison plan that is already resolved without artifact work."""

    feature_name: str
    result: FeatureComparisonResult


@attrs.define(frozen=True)
class ClipFeaturePlan:
    """Feature comparison plan that runs one Ray row per selected clip.

    Attributes:
        feature_name: Feature name used in report output.
        clip_specs: Clip rows that should enter the Ray Data stage.
        load_worker_class: Callable class used as an actor-backed load/normalize
            worker.
        load_worker_constructor_kwargs: Constructor kwargs for
            ``load_worker_class``.
        compare_row: Worker-side comparison callable for one loaded row.
        reduce_rows: Driver-side reducer for compact comparison rows.

    """

    feature_name: str
    clip_specs: tuple[ClipComparisonSpec, ...]
    load_worker_class: type
    load_worker_constructor_kwargs: Mapping[str, Any]
    compare_row: Callable[[Mapping[str, JsonValue]], JsonDictObject]
    reduce_rows: Callable[[Sequence[Mapping[str, JsonValue]]], FeatureComparisonResult]


type FeatureComparisonPlan = ResolvedFeaturePlan | ClipFeaturePlan


class FeatureComparisonPlanner(Protocol):
    """Feature-specific output comparison planner."""

    @property
    def name(self) -> str:
        """Return the feature name used in report output."""

    def build_plan(self, context: FeatureComparisonContext) -> FeatureComparisonPlan:
        """Build a resolved or clip-row feature comparison plan."""


def compare_features(  # noqa: PLR0913
    output_a: OutputRoot,
    output_b: OutputRoot,
    summary_a: OutputSummary,
    summary_b: OutputSummary,
    *,
    profile_name: str | None = None,
    video_limit: int | None = None,
    selected_video_key: str | None = None,
    feature_planners: Sequence[FeatureComparisonPlanner] | None = None,
    workers_per_node: int = 32,
    cpus_per_worker: float = 0.25,
) -> VideoComparisonResult:
    """Compare output features using resolved or Ray Data clip-feature plans.

    Args:
        output_a: First split pipeline output root. Used for artifact paths in
            clip-level feature plans.
        output_b: Second split pipeline output root. Used for artifact paths in
            clip-level feature plans.
        summary_a: Parsed ``summary.json`` for ``output_a``.
        summary_b: Parsed ``summary.json`` for ``output_b``.
        profile_name: Storage profile used when loading feature artifacts. If
            omitted, the default storage profile is resolved at call time.
        video_limit: Optional limit for feature comparison. When set, only the
            first N video keys from output A are planned for feature work.
        selected_video_key: Optional exact summary video key to compare. Mutually
            exclusive with ``video_limit``.
        feature_planners: Optional feature-specific planners. Each planner
            returns either a resolved result or a clip-level Ray Data plan. When
            omitted, caption structure comparison is run.
        workers_per_node: Number of Ray Data worker actors/tasks to schedule per
            Ray node for clip-level feature plans.
        cpus_per_worker: CPU reservation for each Ray Data worker actor/task.

    Returns:
        Feature comparison result containing feature-level issues and report
        summaries keyed by feature name.

    """
    if profile_name is None:
        profile_name = DEFAULT_PROFILE_NAME
    feature_planners = _default_feature_planners() if feature_planners is None else tuple(feature_planners)
    started_at = time.perf_counter()
    specs = build_video_comparison_specs(
        output_a,
        output_b,
        summary_a,
        summary_b,
        video_limit=video_limit,
        selected_video_key=selected_video_key,
    )
    logger.info(
        "Starting output feature comparison: videos={}, video_limit={}, selected_video_key={}, features={}, "
        "workers_per_node={}, cpus_per_worker={}",
        len(specs),
        video_limit,
        selected_video_key,
        [planner.name for planner in feature_planners],
        workers_per_node,
        cpus_per_worker,
    )
    context = FeatureComparisonContext(
        output_a=output_a,
        output_b=output_b,
        summary_a=summary_a,
        summary_b=summary_b,
        profile_name=profile_name,
        specs=specs,
        video_limit=video_limit,
        selected_video_key=selected_video_key,
    )
    resolved_plans: list[ResolvedFeaturePlan] = []
    clip_plans: list[ClipFeaturePlan] = []
    for planner in feature_planners:
        logger.info("Planning output feature comparison: feature={}", planner.name)
        feature_plan = planner.build_plan(context)
        match feature_plan:
            case ResolvedFeaturePlan():
                resolved_plans.append(feature_plan)
                logger.info(
                    "Resolved output feature comparison without artifact loading: feature={}",
                    feature_plan.feature_name,
                )
            case ClipFeaturePlan():
                clip_plans.append(feature_plan)
                logger.info(
                    "Queued clip-row output feature comparison: feature={}, clips={}",
                    feature_plan.feature_name,
                    len(feature_plan.clip_specs),
                )
    execution_config = _RayDataExecutionConfig(
        workers_per_node=workers_per_node,
        cpus_per_worker=cpus_per_worker,
    )
    clip_rows_by_feature = {
        plan.feature_name: _run_ray_data_clip_feature_plan(plan, execution_config) for plan in clip_plans
    }
    logger.info(
        "Reducing output feature comparisons: resolved_features={}, clip_features={}, clip_rows={}",
        len(resolved_plans),
        len(clip_plans),
        sum(len(rows) for rows in clip_rows_by_feature.values()),
    )
    comparison_result = _build_video_comparison_result(
        resolved_plans=resolved_plans,
        clip_plans=clip_plans,
        clip_rows_by_feature=clip_rows_by_feature,
    )
    logger.info(
        "Completed output feature comparison: features={}, issues={}, elapsed_sec={:.2f}",
        sorted(comparison_result.feature_comparisons),
        len(comparison_result.issues),
        time.perf_counter() - started_at,
    )
    return comparison_result


def _default_feature_planners() -> tuple[FeatureComparisonPlanner, ...]:
    from cosmos_curator.pipelines.video.output_comparison.caption_comparator import (  # noqa: PLC0415
        CaptionFeatureComparator,
    )

    return (CaptionFeatureComparator(),)


@attrs.define(frozen=True)
class _RayDataExecutionConfig:
    """Ray Data execution settings for clip feature plans."""

    workers_per_node: int
    cpus_per_worker: float


def _run_ray_data_clip_feature_plan(
    plan: ClipFeaturePlan,
    config: _RayDataExecutionConfig,
) -> list[Mapping[str, JsonValue]]:
    dataset_rows = [spec.to_json_dict() for spec in plan.clip_specs]
    if config.workers_per_node <= 0:
        error_msg = "workers_per_node must be greater than 0"
        raise ValueError(error_msg)
    if config.cpus_per_worker <= 0:
        error_msg = "cpus_per_worker must be greater than 0"
        raise ValueError(error_msg)
    if not dataset_rows:
        logger.info("Skipping Ray Data clip feature plan: feature={}, no clip specs", plan.feature_name)
        return []

    _disable_ray_data_progress_ui()
    if not ray.is_initialized():
        logger.info("Initializing Ray for clip-stage comparison")
        ray.init(ignore_reinit_error=True, include_dashboard=False)
    node_count = len(ray.nodes())  # type: ignore[no-untyped-call]
    compute_size = min(config.workers_per_node * node_count, len(dataset_rows))
    logger.info(
        "Running Ray Data clip feature plan: feature={}, clips={}, nodes={}, actors={}, cpus_per_actor={}",
        plan.feature_name,
        len(dataset_rows),
        node_count,
        compute_size,
        config.cpus_per_worker,
    )
    dataset = ray.data.from_items(dataset_rows)
    load_worker_cls = cast(
        "Callable[[dict[str, Any]], dict[str, Any]]",
        plan.load_worker_class,
    )
    view_dataset = dataset.map(
        load_worker_cls,
        num_cpus=config.cpus_per_worker,
        compute=ActorPoolStrategy(size=compute_size),
        fn_constructor_kwargs=dict(plan.load_worker_constructor_kwargs),
    )
    compare_fn = cast(
        "Callable[[dict[str, Any]], dict[str, Any]]",
        plan.compare_row,
    )
    mapped_dataset = view_dataset.map(
        compare_fn,
        num_cpus=config.cpus_per_worker,
        compute=TaskPoolStrategy(size=compute_size),
    )
    collection_started_at = time.perf_counter()
    rows = [cast("Mapping[str, JsonValue]", row) for row in mapped_dataset.iter_rows()]
    logger.info(
        "Collected Ray Data clip feature rows: feature={}, rows={}, elapsed_sec={:.2f}",
        plan.feature_name,
        len(rows),
        time.perf_counter() - collection_started_at,
    )
    return rows


def _disable_ray_data_progress_ui() -> None:
    """Disable Ray Data progress bars for this CLI-oriented comparison."""
    context = ray.data.DataContext.get_current()
    for attribute_name in (
        "enable_progress_bars",
        "enable_operator_progress_bars",
        "enable_rich_progress_bars",
        "use_ray_tqdm",
    ):
        if hasattr(context, attribute_name):
            setattr(context, attribute_name, False)
    logger.info("Disabled Ray Data progress UI for output feature comparison")


def _build_video_comparison_result(
    *,
    resolved_plans: Sequence[ResolvedFeaturePlan],
    clip_plans: Sequence[ClipFeaturePlan],
    clip_rows_by_feature: Mapping[str, Sequence[Mapping[str, JsonValue]]],
) -> VideoComparisonResult:
    results = {plan.feature_name: plan.result for plan in resolved_plans}
    for plan in clip_plans:
        results[plan.feature_name] = plan.reduce_rows(clip_rows_by_feature[plan.feature_name])
    issues: list[Issue] = []
    feature_comparisons: dict[str, FeatureComparison] = {}
    for feature_name, result in sorted(results.items()):
        issues.extend(result.issues)
        feature_comparisons[feature_name] = result.comparison
    return VideoComparisonResult(issues=tuple(issues), feature_comparisons=feature_comparisons)
