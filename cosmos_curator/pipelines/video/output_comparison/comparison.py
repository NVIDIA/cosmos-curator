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
"""Public API for comparing split video pipeline output summaries."""

from collections.abc import Sequence

import attrs

from cosmos_curator.core.utils.storage import storage_utils
from cosmos_curator.pipelines.video.output_comparison.report import ComparisonReport, Issue, SummaryComparison
from cosmos_curator.pipelines.video.output_comparison.summary_loader import OutputRoot, load_summary
from cosmos_curator.pipelines.video.output_comparison.summary_rules import (
    DEFAULT_SUMMARY_RULES,
    SummaryComparisonRule,
    build_summary_context,
    run_summary_rules,
)
from cosmos_curator.pipelines.video.output_comparison.summary_schema import (
    InvalidSummaryFieldError,
    MissingSummaryFieldError,
    OutputSummary,
)


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
    profile_name: str = "default",
    token_count_abs_tolerance: float = 0,
    token_count_rel_tolerance: float = 0.0,
    rules: Sequence[SummaryComparisonRule] = DEFAULT_SUMMARY_RULES,
) -> ComparisonReport:
    """Compare split pipeline ``summary.json`` accounting for two output roots.

    Args:
        output_a: First split pipeline output root.
        output_b: Second split pipeline output root.
        profile_name: Storage profile used when reading remote summaries.
        token_count_abs_tolerance: Absolute tolerance for token total comparisons.
        token_count_rel_tolerance: Relative tolerance for token total comparisons.
        rules: Summary comparison rules to run.

    Returns:
        Typed report with pass/fail status, comparison counts, and issues.

    """
    summary_comparison = SummaryComparison()

    loaded_a = _load_summary_for_report(output_a, profile_name=profile_name, output_label="a")
    loaded_b = _load_summary_for_report(output_b, profile_name=profile_name, output_label="b")
    match loaded_a, loaded_b:
        case _LoadedSummary(summary=summary_a), _LoadedSummary(summary=summary_b):
            pass
        case _:
            return ComparisonReport.from_issues(
                str(output_a),
                str(output_b),
                summary_comparison,
                _load_issues(loaded_a, loaded_b),
            )

    context = build_summary_context(
        summary_a,
        summary_b,
        token_count_abs_tolerance=token_count_abs_tolerance,
        token_count_rel_tolerance=token_count_rel_tolerance,
    )
    rule_issues, summary_comparison = run_summary_rules(context, rules=rules)

    return ComparisonReport.from_issues(str(output_a), str(output_b), summary_comparison, rule_issues)


def _load_summary_for_report(
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
