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
"""Tests for caption comparison rules."""

from cosmos_curator.pipelines.video.output_comparison.caption_compare import compare_caption_clip_view
from cosmos_curator.pipelines.video.output_comparison.caption_reduce import reduce_caption_clip_results
from cosmos_curator.pipelines.video.output_comparison.caption_result import CaptionClipCompareResult
from cosmos_curator.pipelines.video.output_comparison.caption_schema import (
    CaptionComparisonCounts,
    CaptionWindowRange,
    ClipCaptionView,
)

_DEFAULT_WINDOWS = frozenset({CaptionWindowRange(0, 30)})


def _caption_view(
    **overrides: object,
) -> ClipCaptionView:
    values: dict[str, object] = {
        "video_key": "video.mp4",
        "clip_id": "clip-a",
        "in_a": True,
        "in_b": True,
        "windows_a": _DEFAULT_WINDOWS,
        "windows_b": _DEFAULT_WINDOWS,
        "metadata_path_a": "output-a/metas/v0/clip-a.json",
        "metadata_path_b": "output-b/metas/v0/clip-a.json",
        "missing_metadata_a": False,
        "missing_metadata_b": False,
        "invalid_metadata_a": None,
        "invalid_metadata_b": None,
    }
    values.update(overrides)
    return ClipCaptionView(**values)


def test_compare_caption_clip_view_reports_window_mismatch() -> None:
    """Clip caption comparison reports window mismatches for shared caption clips."""
    result = compare_caption_clip_view(
        _caption_view(
            windows_a=frozenset({CaptionWindowRange(0, 30)}),
            windows_b=frozenset({CaptionWindowRange(15, 45)}),
        ),
        a_has_caption_records=True,
        b_has_caption_records=True,
    )

    assert result.counts == CaptionComparisonCounts(
        clips_with_captions_a=1,
        clips_with_captions_b=1,
        caption_windows_a=1,
        caption_windows_b=1,
        clips_compared=1,
        windows_compared=0,
    )
    assert [issue.code for issue in result.issues] == ["caption_window_set_mismatch"]
    assert result.issues[0].clip == "clip-a"


def test_reduce_caption_clip_results_counts_videos_once() -> None:
    """Reducing clip rows counts a multi-clip video once at video level."""
    rows = (
        compare_caption_clip_view(
            _caption_view(clip_id="clip-a"),
            a_has_caption_records=True,
            b_has_caption_records=True,
        ),
        compare_caption_clip_view(
            _caption_view(
                clip_id="clip-b",
                metadata_path_a="output-a/metas/v0/clip-b.json",
                metadata_path_b="output-b/metas/v0/clip-b.json",
            ),
            a_has_caption_records=True,
            b_has_caption_records=True,
        ),
    )

    result = reduce_caption_clip_results(
        expected_counts=CaptionComparisonCounts(
            videos_with_captions_a=1,
            videos_with_captions_b=1,
            clips_with_captions_a=2,
            clips_with_captions_b=2,
            caption_windows_a=2,
            caption_windows_b=2,
        ),
        videos_only_in_a=(),
        videos_only_in_b=(),
        clip_results=rows,
    )

    assert result.issues == ()
    assert result.comparison.metrics == {
        "videos_with_captions_a": 1,
        "videos_with_captions_b": 1,
        "clips_with_captions_a": 2,
        "clips_with_captions_b": 2,
        "caption_windows_a": 2,
        "caption_windows_b": 2,
        "videos_compared": 1,
        "clips_compared": 2,
        "windows_compared": 2,
    }


def test_reduce_caption_clip_results_groups_clip_set_mismatch() -> None:
    """Clip-side mismatches reduce into one video-level clip-set issue."""
    rows = (
        compare_caption_clip_view(
            _caption_view(clip_id="clip-a", in_a=True, in_b=False, windows_b=frozenset(), metadata_path_b=None),
            a_has_caption_records=True,
            b_has_caption_records=True,
        ),
        compare_caption_clip_view(
            _caption_view(clip_id="clip-b", in_a=False, in_b=True, windows_a=frozenset(), metadata_path_a=None),
            a_has_caption_records=True,
            b_has_caption_records=True,
        ),
    )

    result = reduce_caption_clip_results(
        expected_counts=CaptionComparisonCounts(clips_with_captions_a=1, clips_with_captions_b=1),
        videos_only_in_a=(),
        videos_only_in_b=(),
        clip_results=rows,
    )

    assert [issue.code for issue in result.issues] == ["caption_clip_set_mismatch"]
    assert result.issues[0].details == {
        "clips_only_in_a": ["clip-a"],
        "clips_only_in_b": ["clip-b"],
    }


def test_reduce_caption_clip_results_reports_missing_metadata_paths() -> None:
    """Missing metadata on clip rows appears in count-mismatch issue details."""
    row = compare_caption_clip_view(
        _caption_view(windows_a=frozenset(), missing_metadata_a=True),
        a_has_caption_records=True,
        b_has_caption_records=True,
    )

    result = reduce_caption_clip_results(
        expected_counts=CaptionComparisonCounts(clips_with_captions_a=1, caption_windows_a=1),
        videos_only_in_a=(),
        videos_only_in_b=(),
        clip_results=(row,),
    )

    assert [issue.code for issue in result.issues] == ["caption_data_missing"]
    assert result.issues[0].output == "a"
    assert result.issues[0].details == {
        "expected_clips_with_captions": 1,
        "loaded_clips_with_captions": 0,
        "missing_clips_with_captions": 1,
        "expected_caption_windows": 1,
        "loaded_caption_windows": 0,
        "missing_caption_windows": 1,
        "missing_paths": ["output-a/metas/v0/clip-a.json"],
    }


def test_compare_caption_clip_view_ignores_side_without_caption_records() -> None:
    """Loaded windows do not count for an output side without summary caption evidence."""
    result = compare_caption_clip_view(
        _caption_view(),
        a_has_caption_records=True,
        b_has_caption_records=False,
    )

    assert result.counts == CaptionComparisonCounts(
        clips_with_captions_a=1,
        clips_with_captions_b=0,
        caption_windows_a=1,
        caption_windows_b=0,
    )
    assert result.issues == ()


def test_caption_clip_compare_result_json_round_trip() -> None:
    """Caption clip results round-trip through JSON-compatible rows."""
    result = compare_caption_clip_view(
        _caption_view(
            windows_a=frozenset({CaptionWindowRange(0, 30)}),
            windows_b=frozenset({CaptionWindowRange(15, 45)}),
        ),
        a_has_caption_records=True,
        b_has_caption_records=True,
    )

    assert CaptionClipCompareResult.from_json_dict(result.to_json_dict()) == result
