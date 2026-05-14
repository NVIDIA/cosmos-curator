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
"""Summary comparison rules.

Executable comparison machinery, separate from policy data.
"""

from collections import Counter
from collections.abc import Sequence
from typing import Protocol, cast

import attrs

from cosmos_curator.pipelines.video.output_comparison.json_types import JsonDictObject, JsonValue, json_string_list
from cosmos_curator.pipelines.video.output_comparison.report import Issue, SummaryComparison
from cosmos_curator.pipelines.video.output_comparison.summary_policy import (
    DEFAULT_SUMMARY_POLICY,
    SummaryComparisonPolicy,
)
from cosmos_curator.pipelines.video.output_comparison.summary_schema import (
    OutputSummary,
    ProcessedVideoSummary,
    VideoSummary,
)


@attrs.define(frozen=True)
class SummaryComparisonContext:
    """Inputs shared by summary comparison rules."""

    summary_a: OutputSummary
    summary_b: OutputSummary
    videos_in_both: tuple[str, ...]
    videos_only_in_a: tuple[str, ...]
    videos_only_in_b: tuple[str, ...]
    token_count_abs_tolerance: float
    token_count_rel_tolerance: float


@attrs.define(frozen=True)
class SummaryRuleResult:
    """Issues and report counter updates emitted by a summary rule."""

    issues: tuple[Issue, ...] = ()
    exact_top_level_fields_compared: int | None = None
    token_fields_compared: int | None = None
    per_video_fields_compared: int | None = None


class SummaryComparisonRule(Protocol):
    """Rule interface for comparing two loaded split summaries."""

    def compare(self, context: SummaryComparisonContext) -> SummaryRuleResult:
        """Compare loaded summaries.

        Args:
            context: Summary comparison inputs.

        Returns:
            Issues and summary report counter updates.

        """


@attrs.define(frozen=True)
class VideoKeysRule:
    """Compare the set of per-video summary keys."""

    def compare(self, context: SummaryComparisonContext) -> SummaryRuleResult:
        """Compare video key sets and report key-only differences."""
        videos_only_in_a = list(context.videos_only_in_a)
        videos_only_in_b = list(context.videos_only_in_b)
        issues: list[Issue] = []
        if videos_only_in_a or videos_only_in_b:
            issues.append(
                Issue(
                    code="video_keys_mismatch",
                    message="Video summary keys differ between output A and output B",
                    details={
                        "videos_only_in_a": json_string_list(videos_only_in_a),
                        "videos_only_in_b": json_string_list(videos_only_in_b),
                    },
                )
            )
        return SummaryRuleResult(
            issues=tuple(issues),
        )


@attrs.define(frozen=True)
class ExactTopLevelFieldsRule:
    """Compare top-level summary fields exactly."""

    fields: Sequence[str]

    def compare(self, context: SummaryComparisonContext) -> SummaryRuleResult:
        """Compare configured top-level fields and count comparisons."""
        compared = 0
        issues: list[Issue] = []
        for field in self.fields:
            compared += 1
            value_a = _exact_top_level_value(context.summary_a, field)
            value_b = _exact_top_level_value(context.summary_b, field)
            if value_a == value_b:
                continue
            issues.append(
                Issue(
                    code="summary_field_mismatch",
                    message="Summary field differs between output A and output B",
                    field=field,
                    details=_required_field_mismatch_details(value_a, value_b),
                )
            )
        return SummaryRuleResult(
            issues=tuple(issues),
            exact_top_level_fields_compared=compared,
        )


@attrs.define(frozen=True)
class TokenFieldsRule:
    """Compare token total fields with configured tolerances."""

    fields: Sequence[str]

    def compare(self, context: SummaryComparisonContext) -> SummaryRuleResult:
        """Compare configured token fields and count comparisons."""
        compared = 0
        issues: list[Issue] = []
        for field in self.fields:
            compared += 1
            issue = _token_field_issue(context, field)
            if issue is not None:
                issues.append(issue)
        return SummaryRuleResult(
            issues=tuple(issues),
            token_fields_compared=compared,
        )


@attrs.define(frozen=True)
class VideoProcessedStateRule:
    """Compare whether per-video summaries are processed."""

    def compare(self, context: SummaryComparisonContext) -> SummaryRuleResult:
        """Compare processed state and count comparisons."""
        issues: list[Issue] = []
        for video_key in context.videos_in_both:
            video_a = context.summary_a.videos[video_key]
            video_b = context.summary_b.videos[video_key]
            if video_a.processed == video_b.processed:
                continue
            issues.append(
                Issue(
                    code="video_processed_state_mismatch",
                    message="Video processed state differs between output A and output B",
                    video=video_key,
                    field="processed",
                    details=_required_field_mismatch_details(video_a.processed, video_b.processed),
                )
            )
        return SummaryRuleResult(
            issues=tuple(issues),
            per_video_fields_compared=len(context.videos_in_both),
        )


@attrs.define(frozen=True)
class ExactCommonVideoFieldsRule:
    """Compare per-video fields shared by processed and unprocessed summaries."""

    fields: Sequence[str]

    def compare(self, context: SummaryComparisonContext) -> SummaryRuleResult:
        """Compare configured common per-video fields and count comparisons."""
        compared = 0
        issues: list[Issue] = []
        for video_key in context.videos_in_both:
            video_a = context.summary_a.videos[video_key]
            video_b = context.summary_b.videos[video_key]
            for field in self.fields:
                compared += 1
                value_a = _common_video_value(video_a, field)
                value_b = _common_video_value(video_b, field)
                if value_a == value_b:
                    continue
                issues.append(
                    Issue(
                        code="video_field_mismatch",
                        message="Video summary field differs between output A and output B",
                        video=video_key,
                        field=field,
                        details=_required_field_mismatch_details(value_a, value_b),
                    )
                )
        return SummaryRuleResult(
            issues=tuple(issues),
            per_video_fields_compared=compared,
        )


@attrs.define(frozen=True)
class ExactProcessedVideoFieldsRule:
    """Compare fields that only exist on processed video summaries."""

    fields: Sequence[str]

    def compare(self, context: SummaryComparisonContext) -> SummaryRuleResult:
        """Compare configured processed-video fields and count comparisons."""
        compared = 0
        issues: list[Issue] = []
        for video_key in context.videos_in_both:
            video_a = context.summary_a.videos[video_key]
            video_b = context.summary_b.videos[video_key]
            match video_a, video_b:
                case ProcessedVideoSummary() as processed_a, ProcessedVideoSummary() as processed_b:
                    pass
                case _:
                    continue
            for field in self.fields:
                compared += 1
                value_a = _processed_video_value(processed_a, field)
                value_b = _processed_video_value(processed_b, field)
                if value_a == value_b:
                    continue
                issues.append(
                    Issue(
                        code="video_field_mismatch",
                        message="Video summary field differs between output A and output B",
                        video=video_key,
                        field=field,
                        details=_required_field_mismatch_details(value_a, value_b),
                    )
                )
        return SummaryRuleResult(
            issues=tuple(issues),
            per_video_fields_compared=compared,
        )


@attrs.define(frozen=True)
class ClipUuidListsRule:
    """Compare summary-level clip UUID list accounting."""

    fields: Sequence[str]

    def compare(self, context: SummaryComparisonContext) -> SummaryRuleResult:
        """Compare configured clip UUID list fields by count and set membership.

        Args:
            context: Summary comparison inputs.

        Returns:
            Issues and summary report counter updates.

        """
        issues: list[Issue] = []
        for video_key in context.videos_in_both:
            video_a = context.summary_a.videos[video_key]
            video_b = context.summary_b.videos[video_key]
            match video_a, video_b:
                case ProcessedVideoSummary() as processed_a, ProcessedVideoSummary() as processed_b:
                    pass
                case _:
                    continue
            for field in self.fields:
                list_a = _video_string_list(processed_a, field)
                list_b = _video_string_list(processed_b, field)
                issues.extend(_clip_list_issues(video_key, field, list_a, list_b))
        return SummaryRuleResult(issues=tuple(issues))


def rules_for_policy(policy: SummaryComparisonPolicy) -> tuple[SummaryComparisonRule, ...]:
    """Build summary comparison rules from a field comparison policy.

    Args:
        policy: Field comparison policy.

    Returns:
        Summary comparison rules.

    """
    return (
        VideoKeysRule(),
        ExactTopLevelFieldsRule(policy.exact_top_level_fields),
        TokenFieldsRule(policy.token_fields),
        VideoProcessedStateRule(),
        ExactCommonVideoFieldsRule(policy.common_video_fields),
        ExactProcessedVideoFieldsRule(policy.processed_video_fields),
        ClipUuidListsRule(policy.clip_list_fields),
    )


DEFAULT_SUMMARY_RULES = rules_for_policy(DEFAULT_SUMMARY_POLICY)


def build_summary_context(
    summary_a: OutputSummary,
    summary_b: OutputSummary,
    *,
    token_count_abs_tolerance: float,
    token_count_rel_tolerance: float,
) -> SummaryComparisonContext:
    """Build rule context from loaded summaries.

    Args:
        summary_a: First loaded summary.
        summary_b: Second loaded summary.
        token_count_abs_tolerance: Absolute tolerance for token total comparisons.
        token_count_rel_tolerance: Relative tolerance for token total comparisons.

    Returns:
        Context consumed by summary comparison rules.

    """
    video_keys_a = set(summary_a.videos)
    video_keys_b = set(summary_b.videos)
    return SummaryComparisonContext(
        summary_a=summary_a,
        summary_b=summary_b,
        videos_in_both=tuple(sorted(video_keys_a & video_keys_b)),
        videos_only_in_a=tuple(sorted(video_keys_a - video_keys_b)),
        videos_only_in_b=tuple(sorted(video_keys_b - video_keys_a)),
        token_count_abs_tolerance=token_count_abs_tolerance,
        token_count_rel_tolerance=token_count_rel_tolerance,
    )


def run_summary_rules(
    context: SummaryComparisonContext,
    rules: Sequence[SummaryComparisonRule] = DEFAULT_SUMMARY_RULES,
) -> tuple[list[Issue], SummaryComparison]:
    """Run summary comparison rules.

    Args:
        context: Summary comparison inputs.
        rules: Rules to execute in order.

    Returns:
        Structured issues and summary comparison report.

    """
    issues: list[Issue] = []
    summary_comparison = SummaryComparison(
        videos_in_both=len(context.videos_in_both),
        videos_only_in_a=context.videos_only_in_a,
        videos_only_in_b=context.videos_only_in_b,
    )
    for rule in rules:
        result = rule.compare(context)
        issues.extend(result.issues)
        summary_comparison = _with_rule_counts(summary_comparison, result)
    return issues, summary_comparison


def _exact_top_level_value(summary: OutputSummary, field: str) -> JsonValue:
    return summary.field_values[field]


def _token_value(summary: OutputSummary, field: str) -> int | float:
    value = summary.field_values[field]
    if isinstance(value, bool) or not isinstance(value, int | float):
        error_msg = f"summary field {field!r} must be numeric for token comparison"
        raise TypeError(error_msg)
    return value


def _common_video_value(video: VideoSummary, field: str) -> JsonValue:
    return video.common.field_values[field]


def _processed_video_value(video: ProcessedVideoSummary, field: str) -> JsonValue:
    return video.common.field_values[field]


def _video_string_list(video: ProcessedVideoSummary, field: str) -> tuple[str, ...]:
    value = video.common.field_values[field]
    if not isinstance(value, list):
        error_msg = f"summary field {field!r} must be a list for clip UUID comparison"
        raise TypeError(error_msg)
    return tuple(str(item) for item in value)


def _token_field_issue(context: SummaryComparisonContext, field: str) -> Issue | None:
    value_a = _token_value(context.summary_a, field)
    value_b = _token_value(context.summary_b, field)
    abs_delta = abs(value_a - value_b)
    rel_delta = _relative_delta(value_a, value_b, abs_delta)
    threshold = max(
        context.token_count_abs_tolerance,
        context.token_count_rel_tolerance * max(abs(value_a), abs(value_b)),
    )
    if abs_delta <= threshold:
        return None
    details: JsonDictObject = {
        "a": value_a,
        "b": value_b,
        "abs_delta": abs_delta,
        "rel_delta": rel_delta,
        "token_count_abs_tolerance": context.token_count_abs_tolerance,
        "token_count_rel_tolerance": context.token_count_rel_tolerance,
    }
    return Issue(
        code="token_field_mismatch",
        message="Token summary field differs beyond configured tolerance",
        field=field,
        details=details,
    )


def _with_rule_counts(summary_comparison: SummaryComparison, result: SummaryRuleResult) -> SummaryComparison:
    if result.exact_top_level_fields_compared is not None:
        summary_comparison = attrs.evolve(
            summary_comparison,
            exact_top_level_fields_compared=(
                summary_comparison.exact_top_level_fields_compared + result.exact_top_level_fields_compared
            ),
        )
    if result.token_fields_compared is not None:
        summary_comparison = attrs.evolve(
            summary_comparison,
            token_fields_compared=summary_comparison.token_fields_compared + result.token_fields_compared,
        )
    if result.per_video_fields_compared is not None:
        summary_comparison = attrs.evolve(
            summary_comparison,
            per_video_fields_compared=summary_comparison.per_video_fields_compared + result.per_video_fields_compared,
        )
    return summary_comparison


def _clip_list_issues(video_key: str, field: str, list_a: Sequence[str], list_b: Sequence[str]) -> list[Issue]:
    issues: list[Issue] = []
    if len(list_a) != len(list_b):
        issues.append(
            Issue(
                code="clip_list_length_mismatch",
                message="Video summary clip list length differs between output A and output B",
                video=video_key,
                field=field,
                details={
                    "a_count": len(list_a),
                    "b_count": len(list_b),
                },
            )
        )
    counts_a = Counter(list_a)
    counts_b = Counter(list_b)
    only_in_a = sorted(counts_a.keys() - counts_b.keys())
    only_in_b = sorted(counts_b.keys() - counts_a.keys())
    count_mismatches = [
        {
            "clip_uuid": clip_uuid,
            "a_count": counts_a[clip_uuid],
            "b_count": counts_b[clip_uuid],
        }
        for clip_uuid in sorted(counts_a.keys() & counts_b.keys())
        if counts_a[clip_uuid] != counts_b[clip_uuid]
    ]
    if only_in_a or only_in_b or count_mismatches:
        details: JsonDictObject = {
            "only_in_a": json_string_list(only_in_a),
            "only_in_b": json_string_list(only_in_b),
        }
        if count_mismatches:
            details["count_mismatches"] = cast("JsonValue", count_mismatches)
        issues.append(
            Issue(
                code="clip_uuid_set_mismatch",
                message="Video summary clip UUID set differs between output A and output B",
                video=video_key,
                field=field,
                details=details,
            )
        )
    return issues


def _required_field_mismatch_details(value_a: JsonValue, value_b: JsonValue) -> JsonDictObject:
    return {
        "a": value_a,
        "b": value_b,
        "a_present": True,
        "b_present": True,
    }


def _relative_delta(value_a: float, value_b: float, abs_delta: float) -> float:
    denominator = max(abs(value_a), abs(value_b))
    if denominator == 0:
        return 0.0
    return abs_delta / denominator
