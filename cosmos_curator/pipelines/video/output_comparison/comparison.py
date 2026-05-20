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

import attrs

from cosmos_curator.core.utils.storage import storage_utils
from cosmos_curator.pipelines.video.output_comparison.compare_features import compare_features
from cosmos_curator.pipelines.video.output_comparison.report import ComparisonReport, Issue, SummaryComparison
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
    token_count_abs_tolerance: float = 0,
    token_count_rel_tolerance: float = 0.0,
    summary_policy: SummaryComparisonPolicy = DEFAULT_SUMMARY_POLICY,
    video_limit: int | None = None,
    selected_video_key: str | None = None,
) -> ComparisonReport:
    """Compare split pipeline outputs for two output roots.

    Args:
        output_a: First split pipeline output root.
        output_b: Second split pipeline output root.
        profile_name: Storage profile used when reading remote summaries. If omitted,
            the default storage profile is resolved at call time.
        token_count_abs_tolerance: Absolute tolerance for token total comparisons.
        token_count_rel_tolerance: Relative tolerance for token total comparisons.
        summary_policy: Summary fields to compare and how to compare them.
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
        token_count_abs_tolerance=token_count_abs_tolerance,
        token_count_rel_tolerance=token_count_rel_tolerance,
        summary_policy=summary_policy,
    )
    feature_result = compare_features(
        output_a,
        output_b,
        summary_a,
        summary_b,
        profile_name=profile_name,
        video_limit=video_limit,
        selected_video_key=selected_video_key,
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


def _summary_error_field(exc: Exception) -> str | None:
    if isinstance(exc, MissingSummaryFieldError | InvalidSummaryFieldError):
        return exc.field
    return None
