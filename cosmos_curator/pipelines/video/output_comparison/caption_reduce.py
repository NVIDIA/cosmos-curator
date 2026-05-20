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
"""Driver-side reduction for caption comparison rows."""

from collections.abc import Iterable, Sequence

import attrs

from cosmos_curator.pipelines.video.output_comparison.caption_result import (
    CAPTIONS_FEATURE_NAME,
    CaptionClipCompareResult,
    CaptionComparisonResult,
)
from cosmos_curator.pipelines.video.output_comparison.caption_schema import CaptionComparisonCounts
from cosmos_curator.pipelines.video.output_comparison.json_types import JsonDictObject, json_string_list
from cosmos_curator.pipelines.video.output_comparison.report import FeatureComparison, FeatureComparisonStatus, Issue


def reduce_caption_clip_results(
    *,
    expected_counts: CaptionComparisonCounts,
    videos_only_in_a: Sequence[str],
    videos_only_in_b: Sequence[str],
    clip_results: Sequence[CaptionClipCompareResult],
) -> CaptionComparisonResult:
    """Reduce compact per-clip results into final caption comparison output."""
    counts = _clip_result_counts(clip_results)
    issues: list[Issue] = _clip_data_missing_issues(
        expected_counts=expected_counts,
        counts=counts,
        clip_results=clip_results,
    )
    if videos_only_in_a or videos_only_in_b:
        issues.append(
            Issue(
                code="caption_video_set_mismatch",
                message="Captioned video set differs between output A and output B",
                feature=CAPTIONS_FEATURE_NAME,
                details={
                    "videos_only_in_a": json_string_list(videos_only_in_a),
                    "videos_only_in_b": json_string_list(videos_only_in_b),
                },
            )
        )
    issues.extend(_clip_set_mismatch_issues(clip_results, videos_only_in_a, videos_only_in_b))
    for result in clip_results:
        issues.extend(result.issues)
    status: FeatureComparisonStatus = "failed" if issues else "passed"
    return CaptionComparisonResult(issues=tuple(issues), comparison=_caption_comparison(counts, status=status))


def caption_presence_mismatch_result(
    *,
    a_has_caption_records: bool,
    b_has_caption_records: bool,
    counts: CaptionComparisonCounts,
) -> CaptionComparisonResult:
    """Build the result for output-level caption presence mismatches."""
    return CaptionComparisonResult(
        issues=(
            Issue(
                code="caption_presence_mismatch",
                message="Caption record presence differs between output A and output B",
                feature=CAPTIONS_FEATURE_NAME,
                details={
                    "a_has_caption_records": a_has_caption_records,
                    "b_has_caption_records": b_has_caption_records,
                },
            ),
        ),
        comparison=_caption_comparison(counts, status="failed"),
    )


def empty_caption_result() -> CaptionComparisonResult:
    """Build the result for outputs with zero caption counts."""
    return CaptionComparisonResult(issues=(), comparison=_caption_comparison(CaptionComparisonCounts(), "passed"))


def _clip_result_counts(clip_results: Sequence[CaptionClipCompareResult]) -> CaptionComparisonCounts:
    video_keys_with_captions_a = {result.video_key for result in clip_results if result.counts.clips_with_captions_a}
    video_keys_with_captions_b = {result.video_key for result in clip_results if result.counts.clips_with_captions_b}
    video_keys_compared = {
        result.video_key
        for result in clip_results
        if result.counts.clips_with_captions_a and result.counts.clips_with_captions_b
    }
    summed_counts = _sum_counts(result.counts for result in clip_results)
    return attrs.evolve(
        summed_counts,
        videos_with_captions_a=len(video_keys_with_captions_a),
        videos_with_captions_b=len(video_keys_with_captions_b),
        videos_compared=len(video_keys_compared),
    )


def _clip_data_missing_issues(
    *,
    expected_counts: CaptionComparisonCounts,
    counts: CaptionComparisonCounts,
    clip_results: Sequence[CaptionClipCompareResult],
) -> list[Issue]:
    issues: list[Issue] = []
    if _loaded_counts_are_below_expected("a", counts, expected_counts):
        issues.append(_caption_data_missing_issue_from_clip_results("a", counts, expected_counts, clip_results))
    if _loaded_counts_are_below_expected("b", counts, expected_counts):
        issues.append(_caption_data_missing_issue_from_clip_results("b", counts, expected_counts, clip_results))
    return issues


def _caption_data_missing_issue_from_clip_results(
    output_label: str,
    counts: CaptionComparisonCounts,
    expected_counts: CaptionComparisonCounts,
    clip_results: Sequence[CaptionClipCompareResult],
) -> Issue:
    output_name = output_label.upper()
    missing_paths: list[str] = []
    invalid_paths: list[str] = []
    for result in clip_results:
        if output_label == "a":
            if result.missing_path_a is not None:
                missing_paths.append(result.missing_path_a)
            if result.invalid_path_a is not None:
                invalid_paths.append(result.invalid_path_a)
        else:
            if result.missing_path_b is not None:
                missing_paths.append(result.missing_path_b)
            if result.invalid_path_b is not None:
                invalid_paths.append(result.invalid_path_b)
    details = _missing_count_details(output_label, counts, expected_counts)
    if missing_paths:
        details["missing_paths"] = json_string_list(sorted(set(missing_paths)))
    if invalid_paths:
        details["invalid_paths"] = json_string_list(sorted(set(invalid_paths)))
    return Issue(
        code="caption_data_missing",
        message=f"Caption evidence exists for output {output_name}, but fewer caption records could be loaded",
        output=output_label,
        feature=CAPTIONS_FEATURE_NAME,
        details=details,
    )


def _clip_set_mismatch_issues(
    clip_results: Sequence[CaptionClipCompareResult],
    videos_only_in_a: Sequence[str],
    videos_only_in_b: Sequence[str],
) -> list[Issue]:
    non_shared_video_keys = set(videos_only_in_a) | set(videos_only_in_b)
    mismatches_by_video: dict[str, tuple[list[str], list[str]]] = {}
    for result in clip_results:
        if result.video_key in non_shared_video_keys or result.in_a == result.in_b:
            continue
        clips_only_in_a, clips_only_in_b = mismatches_by_video.setdefault(result.video_key, ([], []))
        if result.in_a:
            clips_only_in_a.append(result.clip_id)
        else:
            clips_only_in_b.append(result.clip_id)
    return [
        Issue(
            code="caption_clip_set_mismatch",
            message="Captioned clip UUID set differs between output A and output B",
            feature=CAPTIONS_FEATURE_NAME,
            video=video_key,
            details={
                "clips_only_in_a": json_string_list(sorted(clips_only_in_a)),
                "clips_only_in_b": json_string_list(sorted(clips_only_in_b)),
            },
        )
        for video_key, (clips_only_in_a, clips_only_in_b) in sorted(mismatches_by_video.items())
    ]


def _loaded_counts_are_below_expected(
    output_label: str,
    counts: CaptionComparisonCounts,
    expected_counts: CaptionComparisonCounts,
) -> bool:
    expected_videos, loaded_videos = _expected_and_loaded_videos(output_label, counts, expected_counts)
    expected_clips, loaded_clips = _expected_and_loaded_clips(output_label, counts, expected_counts)
    expected_windows, loaded_windows = _expected_and_loaded_windows(output_label, counts, expected_counts)
    return loaded_videos < expected_videos or loaded_clips < expected_clips or loaded_windows < expected_windows


def _missing_count_details(
    output_label: str,
    counts: CaptionComparisonCounts,
    expected_counts: CaptionComparisonCounts,
) -> JsonDictObject:
    details: JsonDictObject = {}
    expected_videos, loaded_videos = _expected_and_loaded_videos(output_label, counts, expected_counts)
    if loaded_videos < expected_videos:
        details["expected_videos_with_captions"] = expected_videos
        details["loaded_videos_with_captions"] = loaded_videos
        details["missing_videos_with_captions"] = expected_videos - loaded_videos

    expected_clips, loaded_clips = _expected_and_loaded_clips(output_label, counts, expected_counts)
    if loaded_clips < expected_clips:
        details["expected_clips_with_captions"] = expected_clips
        details["loaded_clips_with_captions"] = loaded_clips
        details["missing_clips_with_captions"] = expected_clips - loaded_clips

    expected_windows, loaded_windows = _expected_and_loaded_windows(output_label, counts, expected_counts)
    if loaded_windows < expected_windows:
        details["expected_caption_windows"] = expected_windows
        details["loaded_caption_windows"] = loaded_windows
        details["missing_caption_windows"] = expected_windows - loaded_windows
    return details


def _expected_and_loaded_videos(
    output_label: str,
    counts: CaptionComparisonCounts,
    expected_counts: CaptionComparisonCounts,
) -> tuple[int, int]:
    if output_label == "a":
        return expected_counts.videos_with_captions_a, counts.videos_with_captions_a
    return expected_counts.videos_with_captions_b, counts.videos_with_captions_b


def _expected_and_loaded_clips(
    output_label: str,
    counts: CaptionComparisonCounts,
    expected_counts: CaptionComparisonCounts,
) -> tuple[int, int]:
    if output_label == "a":
        return expected_counts.clips_with_captions_a, counts.clips_with_captions_a
    return expected_counts.clips_with_captions_b, counts.clips_with_captions_b


def _expected_and_loaded_windows(
    output_label: str,
    counts: CaptionComparisonCounts,
    expected_counts: CaptionComparisonCounts,
) -> tuple[int, int]:
    if output_label == "a":
        return expected_counts.caption_windows_a, counts.caption_windows_a
    return expected_counts.caption_windows_b, counts.caption_windows_b


def _sum_counts(counts: Iterable[CaptionComparisonCounts]) -> CaptionComparisonCounts:
    counts = tuple(counts)
    return CaptionComparisonCounts(
        videos_with_captions_a=sum(count.videos_with_captions_a for count in counts),
        videos_with_captions_b=sum(count.videos_with_captions_b for count in counts),
        clips_with_captions_a=sum(count.clips_with_captions_a for count in counts),
        clips_with_captions_b=sum(count.clips_with_captions_b for count in counts),
        caption_windows_a=sum(count.caption_windows_a for count in counts),
        caption_windows_b=sum(count.caption_windows_b for count in counts),
        videos_compared=sum(count.videos_compared for count in counts),
        clips_compared=sum(count.clips_compared for count in counts),
        windows_compared=sum(count.windows_compared for count in counts),
    )


def _caption_comparison(counts: CaptionComparisonCounts, status: FeatureComparisonStatus) -> FeatureComparison:
    return FeatureComparison(
        status=status,
        metrics=counts.to_json_dict(),
    )
