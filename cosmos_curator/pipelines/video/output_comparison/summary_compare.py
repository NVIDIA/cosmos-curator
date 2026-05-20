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
"""Summary accounting comparison for split pipeline outputs."""

from collections import Counter
from collections.abc import Sequence
from typing import cast

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
class SummaryComparisonResult:
    """Issues and counters emitted by summary accounting comparison."""

    issues: tuple[Issue, ...]
    summary_comparison: SummaryComparison


@attrs.define(frozen=True)
class _SummaryComparisonContext:
    summary_a: OutputSummary
    summary_b: OutputSummary
    videos_in_both: tuple[str, ...]
    videos_only_in_a: tuple[str, ...]
    videos_only_in_b: tuple[str, ...]
    token_count_abs_tolerance: float
    token_count_rel_tolerance: float


def compare_summaries(
    summary_a: OutputSummary,
    summary_b: OutputSummary,
    *,
    token_count_abs_tolerance: float,
    token_count_rel_tolerance: float,
    summary_policy: SummaryComparisonPolicy = DEFAULT_SUMMARY_POLICY,
) -> SummaryComparisonResult:
    """Compare loaded split-output summaries using the configured accounting policy."""
    context = _build_summary_context(
        summary_a,
        summary_b,
        token_count_abs_tolerance=token_count_abs_tolerance,
        token_count_rel_tolerance=token_count_rel_tolerance,
    )

    issues: list[Issue] = []
    issues.extend(_video_key_issues(context))

    exact_field_count, exact_field_issues = _compare_exact_top_level_fields(
        context,
        summary_policy.exact_top_level_fields,
    )
    issues.extend(exact_field_issues)

    token_field_count, token_field_issues = _compare_token_fields(context, summary_policy.token_fields)
    issues.extend(token_field_issues)

    processed_state_count, processed_state_issues = _compare_video_processed_state(context)
    issues.extend(processed_state_issues)

    common_video_field_count, common_video_field_issues = _compare_common_video_fields(
        context,
        summary_policy.common_video_fields,
    )
    issues.extend(common_video_field_issues)

    processed_video_field_count, processed_video_field_issues = _compare_processed_video_fields(
        context,
        summary_policy.processed_video_fields,
    )
    issues.extend(processed_video_field_issues)
    clip_list_count, clip_list_issues = _compare_clip_uuid_lists(context, summary_policy.clip_list_fields)
    issues.extend(clip_list_issues)

    return SummaryComparisonResult(
        issues=tuple(issues),
        summary_comparison=SummaryComparison(
            videos_in_both=len(context.videos_in_both),
            videos_only_in_a=context.videos_only_in_a,
            videos_only_in_b=context.videos_only_in_b,
            exact_top_level_fields_compared=exact_field_count,
            token_fields_compared=token_field_count,
            per_video_fields_compared=(
                processed_state_count + common_video_field_count + processed_video_field_count + clip_list_count
            ),
        ),
    )


def _build_summary_context(
    summary_a: OutputSummary,
    summary_b: OutputSummary,
    *,
    token_count_abs_tolerance: float,
    token_count_rel_tolerance: float,
) -> _SummaryComparisonContext:
    video_keys_a = set(summary_a.videos)
    video_keys_b = set(summary_b.videos)
    return _SummaryComparisonContext(
        summary_a=summary_a,
        summary_b=summary_b,
        videos_in_both=tuple(sorted(video_keys_a & video_keys_b)),
        videos_only_in_a=tuple(sorted(video_keys_a - video_keys_b)),
        videos_only_in_b=tuple(sorted(video_keys_b - video_keys_a)),
        token_count_abs_tolerance=token_count_abs_tolerance,
        token_count_rel_tolerance=token_count_rel_tolerance,
    )


def _video_key_issues(context: _SummaryComparisonContext) -> list[Issue]:
    if not context.videos_only_in_a and not context.videos_only_in_b:
        return []
    return [
        Issue(
            code="video_keys_mismatch",
            message="Video summary keys differ between output A and output B",
            details={
                "videos_only_in_a": json_string_list(context.videos_only_in_a),
                "videos_only_in_b": json_string_list(context.videos_only_in_b),
            },
        )
    ]


def _compare_exact_top_level_fields(
    context: _SummaryComparisonContext,
    fields: Sequence[str],
) -> tuple[int, list[Issue]]:
    issues: list[Issue] = []
    for field in fields:
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
    return len(fields), issues


def _compare_token_fields(context: _SummaryComparisonContext, fields: Sequence[str]) -> tuple[int, list[Issue]]:
    issues = [issue for field in fields if (issue := _token_field_issue(context, field)) is not None]
    return len(fields), issues


def _compare_video_processed_state(context: _SummaryComparisonContext) -> tuple[int, list[Issue]]:
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
    return len(context.videos_in_both), issues


def _compare_common_video_fields(
    context: _SummaryComparisonContext,
    fields: Sequence[str],
) -> tuple[int, list[Issue]]:
    compared = 0
    issues: list[Issue] = []
    for video_key in context.videos_in_both:
        video_a = context.summary_a.videos[video_key]
        video_b = context.summary_b.videos[video_key]
        for field in fields:
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
    return compared, issues


def _compare_processed_video_fields(
    context: _SummaryComparisonContext,
    fields: Sequence[str],
) -> tuple[int, list[Issue]]:
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
        for field in fields:
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
    return compared, issues


def _compare_clip_uuid_lists(context: _SummaryComparisonContext, fields: Sequence[str]) -> tuple[int, list[Issue]]:
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
        for field in fields:
            compared += 1
            list_a = _video_string_list(processed_a, field)
            list_b = _video_string_list(processed_b, field)
            issues.extend(_clip_list_issues(video_key, field, list_a, list_b))
    return compared, issues


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


def _token_field_issue(context: _SummaryComparisonContext, field: str) -> Issue | None:
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
