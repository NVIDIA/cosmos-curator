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
"""A/B compare of ``summary.json`` for the split pipeline.

Produces a ``pa.Table`` of summary-level issues. Comparison is strict-equality
on the typed top-level and per-video fields, with one exception:
``total_prompt_tokens`` / ``total_output_tokens`` honor
``SummaryPolicy.token_count_abs_tolerance`` / ``rel_tolerance`` (LLM token
counts are weakly reproducible across runs even at fixed inputs).
"""

from typing import TYPE_CHECKING

import pyarrow as pa

from cosmos_curator.pipelines.video.split_comparison.config import SummaryPolicy
from cosmos_curator.pipelines.video.split_comparison.result_model import (
    ISSUE_SCHEMA,
    Issue,
    make_issue,
)

if TYPE_CHECKING:
    from cosmos_curator.pipelines.video.split_comparison.summary_schema import (
        OutputSummary,
        ProcessedVideoSummary,
        VideoSummary,
    )

_FEATURE = "summary"

_TOKEN_COUNT_FIELDS: frozenset[str] = frozenset({"total_prompt_tokens", "total_output_tokens"})

_TOP_LEVEL_FIELDS: tuple[str, ...] = (
    "num_input_videos",
    "num_input_videos_selected",
    "num_processed_videos",
    "num_remuxed_videos",
    "embedding_algorithm",
    "total_video_bytes",
    "total_video_duration",
    "total_clip_duration",
    "max_clip_duration",
    "total_num_clips_filtered_by_motion",
    "total_num_clips_filtered_by_aesthetic",
    "total_num_clips_filtered_by_qwen_classifier",
    "total_num_clips_filtered_by_qwen_semantic",
    "total_num_clips_filtered_by_artificial_text",
    "total_num_clips_passed",
    "total_num_clips_transcoded",
    "total_num_clips_with_embeddings",
    "total_num_clips_with_caption",
    "total_num_caption_windows",
    "total_num_clips_with_webp",
    "total_prompt_tokens",
    "total_output_tokens",
)

_PROCESSED_VIDEO_FIELDS: tuple[str, ...] = (
    "video_uuid",
    "num_clip_chunks",
    "num_total_clips",
    "num_clips_filtered_by_motion",
    "num_clips_filtered_by_aesthetic",
    "num_clips_filtered_by_qwen_classifier",
    "num_clips_filtered_by_qwen_semantic",
    "num_clips_filtered_by_artificial_text",
    "num_clips_passed",
    "num_clips_transcoded",
    "num_clips_with_embeddings",
    "num_clips_with_caption",
    "num_caption_windows",
    "num_clips_with_webp",
)


def compare_summaries(
    summary_a: "OutputSummary",
    summary_b: "OutputSummary",
    policy: SummaryPolicy,
) -> pa.Table:
    """Return a ``pa.Table`` of summary-level issues."""
    issues: list[Issue] = []
    issues.extend(_top_level_field_issues(summary_a, summary_b, policy))
    issues.extend(_video_key_issues(summary_a, summary_b))
    for video_key in sorted(set(summary_a.videos) & set(summary_b.videos)):
        issues.extend(_video_issues(video_key, summary_a.videos[video_key], summary_b.videos[video_key]))
    return pa.Table.from_pylist(issues, schema=ISSUE_SCHEMA)  # type: ignore[arg-type]


def _top_level_field_issues(
    summary_a: "OutputSummary",
    summary_b: "OutputSummary",
    policy: SummaryPolicy,
) -> list[Issue]:
    issues: list[Issue] = []
    for field in _TOP_LEVEL_FIELDS:
        value_a = getattr(summary_a, field)
        value_b = getattr(summary_b, field)
        if _values_match_for_field(field, value_a, value_b, policy):
            continue
        issues.append(
            make_issue(
                code="summary_field_mismatch",
                message=f"Top-level summary field {field!r} differs",
                feature=_FEATURE,
                field=field,
                details={"a": value_a, "b": value_b},
            ),
        )
    return issues


def _values_match_for_field(field: str, value_a: object, value_b: object, policy: SummaryPolicy) -> bool:
    """Strict equality for most fields; abs/rel tolerance for the token-count pair."""
    if value_a == value_b:
        return True
    if field not in _TOKEN_COUNT_FIELDS:
        return False
    if not (isinstance(value_a, (int, float)) and isinstance(value_b, (int, float))):
        return False
    return _within_token_tolerance(float(value_a), float(value_b), policy)


def _within_token_tolerance(value_a: float, value_b: float, policy: SummaryPolicy) -> bool:
    diff = abs(value_a - value_b)
    if diff <= policy.token_count_abs_tolerance:
        return True
    larger = max(abs(value_a), abs(value_b))
    return larger > 0 and diff / larger <= policy.token_count_rel_tolerance


def _video_key_issues(summary_a: "OutputSummary", summary_b: "OutputSummary") -> list[Issue]:
    issues: list[Issue] = []
    only_in_a = sorted(set(summary_a.videos) - set(summary_b.videos))
    only_in_b = sorted(set(summary_b.videos) - set(summary_a.videos))
    issues.extend(
        make_issue(
            code="summary_video_only_in_a",
            message="Video key present only in output A",
            feature=_FEATURE,
            video=video_key,
        )
        for video_key in only_in_a
    )
    issues.extend(
        make_issue(
            code="summary_video_only_in_b",
            message="Video key present only in output B",
            feature=_FEATURE,
            video=video_key,
        )
        for video_key in only_in_b
    )
    return issues


def _video_issues(video_key: str, video_a: "VideoSummary", video_b: "VideoSummary") -> list[Issue]:
    if video_a.processed != video_b.processed:
        return [
            make_issue(
                code="summary_video_processed_state_mismatch",
                message="Video processed state differs between outputs",
                feature=_FEATURE,
                video=video_key,
                details={"a_processed": video_a.processed, "b_processed": video_b.processed},
            ),
        ]
    if not (video_a.processed and video_b.processed):
        # Both unprocessed: nothing to compare at the field level.
        return []
    return _processed_video_issues(video_key, video_a, video_b)  # type: ignore[arg-type]


def _processed_video_issues(
    video_key: str,
    video_a: "ProcessedVideoSummary",
    video_b: "ProcessedVideoSummary",
) -> list[Issue]:
    issues: list[Issue] = []
    for field in _PROCESSED_VIDEO_FIELDS:
        value_a = getattr(video_a, field)
        value_b = getattr(video_b, field)
        if value_a == value_b:
            continue
        issues.append(
            make_issue(
                code="summary_video_field_mismatch",
                message=f"Per-video summary field {field!r} differs",
                feature=_FEATURE,
                video=video_key,
                field=field,
                details={"a": value_a, "b": value_b},
            ),
        )
    issues.extend(
        _clip_uuid_set_issues(video_key, "clips", video_a.clips, video_b.clips),
    )
    issues.extend(
        _clip_uuid_set_issues(video_key, "filtered_clips", video_a.filtered_clips, video_b.filtered_clips),
    )
    return issues


def _clip_uuid_set_issues(
    video_key: str,
    field: str,
    output_a: tuple[str, ...],
    output_b: tuple[str, ...],
) -> list[Issue]:
    set_a = set(output_a)
    set_b = set(output_b)
    if set_a == set_b:
        return []
    return [
        make_issue(
            code="summary_clip_uuid_set_mismatch",
            message=f"Per-video {field!r} clip uuid set differs",
            feature=_FEATURE,
            video=video_key,
            field=field,
            details={
                "only_in_a": sorted(set_a - set_b),
                "only_in_b": sorted(set_b - set_a),
            },
        ),
    ]
