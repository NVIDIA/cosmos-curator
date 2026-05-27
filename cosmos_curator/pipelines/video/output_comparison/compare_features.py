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

import json
import time
from collections import Counter
from collections.abc import Callable, Hashable, Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from typing import Any, cast

import attrs
import ray
from loguru import logger
from ray.data import ActorPoolStrategy, TaskPoolStrategy

from cosmos_curator.pipelines.video.output_comparison.feature_plan import (
    ClipFeaturePlan,
    FeatureComparisonContext,
    FeatureComparisonPlan,
    FeatureComparisonPlanner,
    FeatureComparisonResult,
    FeatureComparisonResults,
    ResolvedFeaturePlan,
)
from cosmos_curator.pipelines.video.output_comparison.json_types import JsonValue
from cosmos_curator.pipelines.video.output_comparison.report import FeatureComparison, Issue
from cosmos_curator.pipelines.video.output_comparison.summary_loader import OutputRoot
from cosmos_curator.pipelines.video.output_comparison.summary_schema import OutputSummary
from cosmos_curator.pipelines.video.output_comparison.video_planning import (
    DEFAULT_PROFILE_NAME,
    VideoComparisonResult,
    build_video_comparison_specs,
)
from cosmos_curator.pipelines.video.output_comparison.video_schema import ClipComparisonSpec


def compare_features(  # noqa: PLR0913
    output_a: OutputRoot,
    output_b: OutputRoot,
    summary_a: OutputSummary,
    summary_b: OutputSummary,
    *,
    profile_name: str | None = None,
    video_limit: int | None = None,
    selected_video_key: str | None = None,
    feature_planners: Sequence[FeatureComparisonPlanner],
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
        feature_planners: Feature-specific planners. Each planner returns either
            a resolved result or a clip-level Ray Data plan.
        workers_per_node: Number of Ray Data worker actors/tasks to schedule per
            Ray node for clip-level feature plans.
        cpus_per_worker: CPU reservation for each Ray Data worker actor/task.

    Returns:
        Feature comparison result containing feature-level issues and report
        summaries keyed by feature name.

    """
    if profile_name is None:
        profile_name = DEFAULT_PROFILE_NAME
    feature_planners = tuple(feature_planners)
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
    _validate_unique_plan_feature_names((*resolved_plans, *clip_plans))
    execution_config = _RayDataExecutionConfig(
        workers_per_node=workers_per_node,
        cpus_per_worker=cpus_per_worker,
    )
    clip_rows_by_feature = _run_ray_data_clip_feature_plans(clip_plans, execution_config)
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


@attrs.define(frozen=True)
class _RayDataExecutionConfig:
    """Ray Data execution settings for clip feature plans."""

    workers_per_node: int
    cpus_per_worker: float


@attrs.define(frozen=True)
class _ClipFeatureLoadGroupKey:
    """Hashable identity for clip feature plans that can share loaded rows."""

    clip_specs: tuple[ClipComparisonSpec, ...]
    load_worker_class: type
    load_worker_constructor_identity: Hashable


@attrs.define(frozen=True)
class _ClipFeatureLoadGroup:
    """Clip feature plans that share the same load stage."""

    key: _ClipFeatureLoadGroupKey
    plans: tuple[ClipFeaturePlan, ...]


@attrs.define(frozen=True)
class _LoadedClipDataset:
    """Materialized Ray Dataset containing reusable loaded clip artifacts.

    ``ray.data.Dataset.__len__`` intentionally raises, and ``count()`` can
    trigger an extra distributed action.  Keep the row count captured from the
    driver-built clip specs so compare stages can size task pools and log row
    counts without collecting or recounting the loaded artifact rows.
    """

    dataset: ray.data.Dataset
    row_count: int


def _run_ray_data_clip_feature_plans(
    plans: Sequence[ClipFeaturePlan],
    config: _RayDataExecutionConfig,
) -> dict[str, list[Mapping[str, JsonValue]]]:
    rows_by_feature: dict[str, list[Mapping[str, JsonValue]]] = {}
    for group in _clip_feature_load_groups(plans):
        loaded_rows = _run_ray_data_clip_feature_load(group.plans[0], config)
        for plan in group.plans:
            if plan.feature_name in rows_by_feature:
                error_msg = f"Duplicate feature plan names are not supported: {plan.feature_name}"
                raise ValueError(error_msg)
            rows_by_feature[plan.feature_name] = _run_ray_data_clip_feature_compare(plan, loaded_rows, config)
    return rows_by_feature


def _clip_feature_load_groups(plans: Sequence[ClipFeaturePlan]) -> tuple[_ClipFeatureLoadGroup, ...]:
    plans_by_key: dict[_ClipFeatureLoadGroupKey, list[ClipFeaturePlan]] = {}
    for plan in plans:
        plans_by_key.setdefault(_clip_feature_load_group_key(plan), []).append(plan)
    return tuple(_ClipFeatureLoadGroup(key=key, plans=tuple(group_plans)) for key, group_plans in plans_by_key.items())


def _clip_feature_load_group_key(plan: ClipFeaturePlan) -> _ClipFeatureLoadGroupKey:
    return _ClipFeatureLoadGroupKey(
        clip_specs=plan.clip_specs,
        load_worker_class=plan.load_worker_class,
        load_worker_constructor_identity=plan.load_group_id
        if plan.load_group_id is not None
        else _canonical_load_worker_constructor_kwargs(plan.load_worker_constructor_kwargs),
    )


def _canonical_load_worker_constructor_kwargs(kwargs: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            _normalize_load_group_value(kwargs),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        error_msg = (
            "load_worker_constructor_kwargs must be JSON-serializable for load grouping; "
            "set load_group_id for non-serializable values"
        )
        raise TypeError(error_msg) from exc


def _normalize_load_group_value(value: object) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return cast("JsonValue", value)
    if isinstance(value, list):
        return {"__type__": "list", "items": [_normalize_load_group_value(item) for item in value]}
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_normalize_load_group_value(item) for item in value]}
    if isinstance(value, Mapping):
        normalized_items = [
            [_normalize_load_group_value(key), _normalize_load_group_value(item)] for key, item in value.items()
        ]
        normalized_items.sort(key=lambda item: json.dumps(item[0], sort_keys=True, separators=(",", ":")))
        return {"__type__": "mapping", "items": cast("list[JsonValue]", normalized_items)}
    return _load_group_json_default(value)


def _load_group_json_default(value: object) -> JsonValue:
    if isinstance(value, Enum):
        return {
            "__type__": "enum",
            "class": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "name": value.name,
        }
    if isinstance(value, datetime | date):
        return {"__type__": value.__class__.__name__, "value": value.isoformat()}
    if isinstance(value, bytes):
        return {"__type__": "bytes", "value": value.hex()}
    if isinstance(value, type):
        return {"__type__": "type", "value": f"{value.__module__}.{value.__qualname__}"}
    error_msg = f"Object of type {value.__class__.__name__} is not JSON serializable"
    raise TypeError(error_msg)


def _run_ray_data_clip_feature_load(
    plan: ClipFeaturePlan,
    config: _RayDataExecutionConfig,
) -> _LoadedClipDataset | None:
    dataset_rows = [spec.to_json_dict() for spec in plan.clip_specs]
    if config.workers_per_node <= 0:
        error_msg = "workers_per_node must be greater than 0"
        raise ValueError(error_msg)
    if config.cpus_per_worker <= 0:
        error_msg = "cpus_per_worker must be greater than 0"
        raise ValueError(error_msg)
    if not dataset_rows:
        logger.info("Skipping Ray Data clip feature plan: feature={}, no clip specs", plan.feature_name)
        return None

    _disable_ray_data_progress_ui()
    if not ray.is_initialized():
        logger.info("Initializing Ray for clip-stage comparison")
        ray.init(ignore_reinit_error=True, include_dashboard=False)
    node_count = len(ray.nodes())  # type: ignore[no-untyped-call]
    compute_size = min(config.workers_per_node * node_count, len(dataset_rows))
    logger.info(
        "Running Ray Data clip feature load stage: feature={}, clips={}, nodes={}, actors={}, cpus_per_actor={}",
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
    materialization_started_at = time.perf_counter()
    loaded_dataset = view_dataset.materialize()
    logger.info(
        "Materialized Ray Data clip feature loaded rows: feature={}, rows={}, elapsed_sec={:.2f}",
        plan.feature_name,
        len(dataset_rows),
        time.perf_counter() - materialization_started_at,
    )
    return _LoadedClipDataset(dataset=loaded_dataset, row_count=len(dataset_rows))


def _run_ray_data_clip_feature_compare(
    plan: ClipFeaturePlan,
    loaded_rows: _LoadedClipDataset | None,
    config: _RayDataExecutionConfig,
) -> list[Mapping[str, JsonValue]]:
    if loaded_rows is None or loaded_rows.row_count == 0:
        logger.info("Skipping Ray Data clip feature compare stage: feature={}, no loaded rows", plan.feature_name)
        return []
    compute_size = min(config.workers_per_node * len(ray.nodes()), loaded_rows.row_count)  # type: ignore[no-untyped-call]
    logger.info(
        "Running Ray Data clip feature compare stage: feature={}, rows={}, tasks={}, cpus_per_task={}",
        plan.feature_name,
        loaded_rows.row_count,
        compute_size,
        config.cpus_per_worker,
    )
    compare_fn = cast(
        "Callable[[dict[str, Any]], dict[str, Any]]",
        plan.compare_row,
    )
    mapped_dataset = loaded_rows.dataset.map(
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
    results: dict[str, FeatureComparisonResult] = {}
    for resolved_plan in resolved_plans:
        _add_feature_results(results, resolved_plan.feature_name, resolved_plan.result)
    for clip_plan in clip_plans:
        _add_feature_results(
            results,
            clip_plan.feature_name,
            clip_plan.reduce_rows(clip_rows_by_feature[clip_plan.feature_name]),
        )
    issues: list[Issue] = []
    feature_comparisons: dict[str, FeatureComparison] = {}
    for feature_name, result in sorted(results.items()):
        issues.extend(result.issues)
        feature_comparisons[feature_name] = result.comparison
    return VideoComparisonResult(issues=tuple(issues), feature_comparisons=feature_comparisons)


def _validate_unique_plan_feature_names(plans: Sequence[FeatureComparisonPlan]) -> None:
    duplicate_names = sorted(name for name, count in Counter(plan.feature_name for plan in plans).items() if count > 1)
    if duplicate_names:
        error_msg = f"Duplicate feature plan names are not supported: {duplicate_names}"
        raise ValueError(error_msg)


def _add_feature_results(
    results: dict[str, FeatureComparisonResult],
    feature_name: str,
    plan_results: FeatureComparisonResults,
) -> None:
    if isinstance(plan_results, FeatureComparisonResult):
        if feature_name in results:
            error_msg = f"Duplicate feature results are not supported: {feature_name}"
            raise ValueError(error_msg)
        results[feature_name] = plan_results
        return
    duplicate_names = sorted(set(results).intersection(plan_results))
    if duplicate_names:
        error_msg = f"Duplicate feature results are not supported: {duplicate_names}"
        raise ValueError(error_msg)
    results.update(plan_results)
