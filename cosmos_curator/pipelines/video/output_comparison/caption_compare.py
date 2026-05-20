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
"""Per-clip caption structure comparison."""

from cosmos_curator.pipelines.video.output_comparison.caption_result import (
    CAPTIONS_FEATURE_NAME,
    CaptionClipCompareResult,
)
from cosmos_curator.pipelines.video.output_comparison.caption_schema import (
    CaptionComparisonCounts,
    CaptionWindowRange,
    ClipCaptionView,
)
from cosmos_curator.pipelines.video.output_comparison.json_types import json_string_list
from cosmos_curator.pipelines.video.output_comparison.report import Issue


def compare_caption_clip_view(
    view: ClipCaptionView,
    *,
    a_has_caption_records: bool,
    b_has_caption_records: bool,
) -> CaptionClipCompareResult:
    """Compare caption clip/window structure for one normalized clip view."""
    issues: list[Issue] = []
    if (
        a_has_caption_records
        and b_has_caption_records
        and view.in_a
        and view.in_b
        and not _has_metadata_load_error(view)
    ):
        issues.extend(_window_set_issues(view.video_key, view.clip_id, view.windows_a, view.windows_b))
    return CaptionClipCompareResult(
        video_key=view.video_key,
        clip_id=view.clip_id,
        in_a=view.in_a,
        in_b=view.in_b,
        issues=tuple(issues),
        counts=_clip_counts(
            view,
            a_has_caption_records=a_has_caption_records,
            b_has_caption_records=b_has_caption_records,
        ),
        missing_path_a=view.metadata_path_a if view.missing_metadata_a else None,
        missing_path_b=view.metadata_path_b if view.missing_metadata_b else None,
        invalid_path_a=view.invalid_metadata_a,
        invalid_path_b=view.invalid_metadata_b,
    )


def _has_metadata_load_error(view: ClipCaptionView) -> bool:
    return (
        view.missing_metadata_a
        or view.missing_metadata_b
        or view.invalid_metadata_a is not None
        or view.invalid_metadata_b is not None
    )


def _window_set_issues(
    video_key: str,
    clip_uuid: str,
    windows_a: frozenset[CaptionWindowRange],
    windows_b: frozenset[CaptionWindowRange],
) -> list[Issue]:
    windows_only_in_a = _window_labels(windows_a - windows_b)
    windows_only_in_b = _window_labels(windows_b - windows_a)
    if not windows_only_in_a and not windows_only_in_b:
        return []
    return [
        Issue(
            code="caption_window_set_mismatch",
            message="Caption window frame ranges differ between output A and output B",
            feature=CAPTIONS_FEATURE_NAME,
            video=video_key,
            clip=clip_uuid,
            details={
                "windows_only_in_a": json_string_list(windows_only_in_a),
                "windows_only_in_b": json_string_list(windows_only_in_b),
            },
        )
    ]


def _clip_counts(
    view: ClipCaptionView,
    *,
    a_has_caption_records: bool,
    b_has_caption_records: bool,
) -> CaptionComparisonCounts:
    both_sides_have_caption_records = a_has_caption_records and b_has_caption_records
    both_sides_have_windows = bool(view.windows_a) and bool(view.windows_b)
    return CaptionComparisonCounts(
        clips_with_captions_a=1 if a_has_caption_records and view.windows_a else 0,
        clips_with_captions_b=1 if b_has_caption_records and view.windows_b else 0,
        caption_windows_a=len(view.windows_a) if a_has_caption_records else 0,
        caption_windows_b=len(view.windows_b) if b_has_caption_records else 0,
        clips_compared=1 if both_sides_have_caption_records and both_sides_have_windows else 0,
        windows_compared=len(view.windows_a & view.windows_b) if both_sides_have_caption_records else 0,
    )


def _window_labels(windows: frozenset[CaptionWindowRange]) -> list[str]:
    return [window.label for window in sorted(windows)]
