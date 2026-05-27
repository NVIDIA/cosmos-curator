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
"""Public API for comparing split video pipeline outputs."""

from collections.abc import Callable, Iterable
from math import isfinite
from typing import cast

import attrs

from cosmos_curator.core.utils.storage import storage_utils
from cosmos_curator.pipelines.video.output_comparison.caption_comparator import CaptionFeatureComparator
from cosmos_curator.pipelines.video.output_comparison.caption_result import CAPTIONS_FEATURE_NAME
from cosmos_curator.pipelines.video.output_comparison.compare_features import compare_features
from cosmos_curator.pipelines.video.output_comparison.feature_plan import FeatureComparisonPlanner
from cosmos_curator.pipelines.video.output_comparison.report import ComparisonReport, Issue, SummaryComparison
from cosmos_curator.pipelines.video.output_comparison.score_comparator import (
    AESTHETIC_SCORE_FEATURE_NAME,
    DEFAULT_SCORE_POLICY,
    MOTION_SCORE_FEATURE_NAME,
    ScoreComparisonPolicy,
    ScoreFeatureComparator,
)
from cosmos_curator.pipelines.video.output_comparison.summary_compare import compare_summaries
from cosmos_curator.pipelines.video.output_comparison.summary_loader import OutputRoot, load_summary
from cosmos_curator.pipelines.video.output_comparison.summary_policy import (
    DEFAULT_SUMMARY_POLICY,
    SummaryComparisonPolicy,
)
from cosmos_curator.pipelines.video.output_comparison.summary_schema import (
    InvalidSummaryFieldError,
    MissingSummaryFieldError,
    OutputSummary,
)
from cosmos_curator.pipelines.video.output_comparison.video_planning import DEFAULT_PROFILE_NAME

OUTPUT_COMPARISON_FEATURE_NAMES: frozenset[str]
DEFAULT_OUTPUT_COMPARISON_FEATURES: frozenset[str]


def _non_negative_finite_tolerance(_instance: object, attribute: "attrs.Attribute[float]", value: float) -> None:
    if not isfinite(value) or value < 0:
        error_msg = f"{attribute.name} must be a finite number greater than or equal to 0"
        raise ValueError(error_msg)


def _feature_names_converter(value: Iterable[str] | str) -> frozenset[str]:
    if isinstance(value, str):
        return frozenset((value,))
    return frozenset(value)


def _enabled_features_validator(
    _instance: object,
    attribute: "attrs.Attribute[frozenset[str]]",
    value: frozenset[str],
) -> None:
    feature_values = cast("frozenset[object]", value)
    non_str_members = [member for member in feature_values if not isinstance(member, str)]
    if non_str_members:
        error_msg = (
            f"{attribute.name} contains non-string members: {sorted(repr(member) for member in non_str_members)}"
        )
        raise ValueError(error_msg)

    feature_names = frozenset(member for member in value if isinstance(member, str))
    unknown_features = sorted(feature_names - OUTPUT_COMPARISON_FEATURE_NAMES)
    if unknown_features:
        error_msg = (
            f"{attribute.name} contains unknown output comparison features: {unknown_features}; "
            f"known features are {sorted(OUTPUT_COMPARISON_FEATURE_NAMES)}"
        )
        raise ValueError(error_msg)


def _default_output_comparison_features() -> frozenset[str]:
    return DEFAULT_OUTPUT_COMPARISON_FEATURES


@attrs.define(frozen=True)
class OutputComparisonConfig:
    """Configuration for comparing split pipeline output roots.

    ``enabled_features`` controls artifact-backed feature comparisons. Summary
    comparison always runs because it is needed to plan feature artifacts and to
    report top-level accounting differences.
    """

    summary_policy: SummaryComparisonPolicy = DEFAULT_SUMMARY_POLICY
    token_count_abs_tolerance: float = attrs.field(default=0.0, validator=_non_negative_finite_tolerance)
    token_count_rel_tolerance: float = attrs.field(default=0.0, validator=_non_negative_finite_tolerance)
    enabled_features: frozenset[str] = attrs.field(
        factory=_default_output_comparison_features,
        converter=_feature_names_converter,
        validator=_enabled_features_validator,
    )
    motion_score_policy: ScoreComparisonPolicy = DEFAULT_SCORE_POLICY
    aesthetic_score_policy: ScoreComparisonPolicy = DEFAULT_SCORE_POLICY


def _caption_feature_planner_factory(_config: OutputComparisonConfig) -> FeatureComparisonPlanner:
    return CaptionFeatureComparator()


def _aesthetic_score_feature_planner_factory(config: OutputComparisonConfig) -> FeatureComparisonPlanner:
    return ScoreFeatureComparator(AESTHETIC_SCORE_FEATURE_NAME, config.aesthetic_score_policy)


def _motion_score_feature_planner_factory(config: OutputComparisonConfig) -> FeatureComparisonPlanner:
    return ScoreFeatureComparator(MOTION_SCORE_FEATURE_NAME, config.motion_score_policy)


FEATURE_PLANNER_REGISTRY: dict[str, Callable[[OutputComparisonConfig], FeatureComparisonPlanner]] = {
    CAPTIONS_FEATURE_NAME: _caption_feature_planner_factory,
    AESTHETIC_SCORE_FEATURE_NAME: _aesthetic_score_feature_planner_factory,
    MOTION_SCORE_FEATURE_NAME: _motion_score_feature_planner_factory,
}
OUTPUT_COMPARISON_FEATURE_NAMES = frozenset(FEATURE_PLANNER_REGISTRY)
DEFAULT_OUTPUT_COMPARISON_FEATURES = OUTPUT_COMPARISON_FEATURE_NAMES
DEFAULT_OUTPUT_COMPARISON_CONFIG = OutputComparisonConfig()


@attrs.define(frozen=True)
class _LoadedSummary:
    """Successfully loaded summary for one output root."""

    summary: OutputSummary


@attrs.define(frozen=True)
class _SummaryLoadFailure:
    """Structured summary load failure for one output root."""

    issue: Issue


type _SummaryLoadResult = _LoadedSummary | _SummaryLoadFailure


def compare_split_outputs(  # noqa: PLR0913
    output_a: OutputRoot,
    output_b: OutputRoot,
    *,
    profile_name: str | None = None,
    config: OutputComparisonConfig = DEFAULT_OUTPUT_COMPARISON_CONFIG,
    video_limit: int | None = None,
    selected_video_key: str | None = None,
) -> ComparisonReport:
    """Compare split pipeline outputs for two output roots.

    Args:
        output_a: First split pipeline output root.
        output_b: Second split pipeline output root.
        profile_name: Storage profile used when reading remote summaries. If omitted,
            the default storage profile is resolved at call time.
        config: Summary policy, token tolerances, artifact feature selection,
            and per-feature score policies.
        video_limit: Optional limit for video-level feature comparisons. When set,
            only the first N video keys from ``output_a`` are matched to ``output_b``.
        selected_video_key: Optional exact video summary key for video-level feature
            comparison. Mutually exclusive with ``video_limit``.

    Returns:
        Typed report with pass/fail status, comparison counts, and issues.

    """
    if video_limit is not None and selected_video_key is not None:
        error_msg = "video_limit and selected_video_key are mutually exclusive"
        raise ValueError(error_msg)
    if profile_name is None:
        profile_name = DEFAULT_PROFILE_NAME
    summary_comparison = SummaryComparison()

    loaded_a = _load_summary(output_a, profile_name=profile_name, output_label="a")
    loaded_b = _load_summary(output_b, profile_name=profile_name, output_label="b")

    if isinstance(loaded_a, _LoadedSummary) and isinstance(loaded_b, _LoadedSummary):
        summary_a = loaded_a.summary
        summary_b = loaded_b.summary
    else:
        return ComparisonReport.from_issues(
            str(output_a),
            str(output_b),
            summary_comparison,
            _load_issues(loaded_a, loaded_b),
        )

    summary_result = compare_summaries(
        summary_a,
        summary_b,
        token_count_abs_tolerance=config.token_count_abs_tolerance,
        token_count_rel_tolerance=config.token_count_rel_tolerance,
        summary_policy=config.summary_policy,
    )
    feature_result = compare_features(
        output_a,
        output_b,
        summary_a,
        summary_b,
        profile_name=profile_name,
        video_limit=video_limit,
        selected_video_key=selected_video_key,
        feature_planners=_feature_planners_for_config(config),
    )

    return ComparisonReport.from_issues(
        str(output_a),
        str(output_b),
        summary_result.summary_comparison,
        [*summary_result.issues, *feature_result.issues],
        feature_comparisons=feature_result.feature_comparisons,
    )


def _load_summary(
    output_root: OutputRoot,
    *,
    profile_name: str,
    output_label: str,
) -> _SummaryLoadResult:
    summary_path = storage_utils.get_full_path(output_root, "summary.json")
    try:
        return _LoadedSummary(load_summary(output_root, profile_name=profile_name))
    except Exception as exc:  # noqa: BLE001
        return _SummaryLoadFailure(
            Issue.summary_load_failed(
                str(summary_path),
                output_label,
                exc.__class__.__name__,
                str(exc),
                field=_summary_error_field(exc),
            )
        )


def _load_issues(*results: _SummaryLoadResult) -> list[Issue]:
    return [result.issue for result in results if isinstance(result, _SummaryLoadFailure)]


def _feature_planners_for_config(config: OutputComparisonConfig) -> tuple[FeatureComparisonPlanner, ...]:
    return tuple(
        planner_factory(config)
        for feature_name, planner_factory in FEATURE_PLANNER_REGISTRY.items()
        if feature_name in config.enabled_features
    )


def _summary_error_field(exc: Exception) -> str | None:
    if isinstance(exc, MissingSummaryFieldError | InvalidSummaryFieldError):
        return exc.field
    return None
