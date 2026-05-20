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
"""Tests for split output caption structure comparison."""

import json
from pathlib import Path
from typing import Any, cast

from cosmos_curator.pipelines.video.output_comparison.comparison import compare_split_outputs
from cosmos_curator.pipelines.video.output_comparison.report import ComparisonReport

from .conftest import summary, video_summary, write_summary


def _json_report(report: ComparisonReport) -> dict[str, Any]:
    return cast("dict[str, Any]", report.to_json_dict())


def _caption_summary(video_clips: dict[str, list[str]], *, caption_windows: int) -> dict[str, Any]:
    summary_overrides: dict[str, Any] = {
        "num_input_videos": len(video_clips),
        "num_input_videos_selected": len(video_clips),
        "num_processed_videos": len(video_clips),
        "total_num_clips_passed": sum(len(clips) for clips in video_clips.values()),
        "total_num_clips_transcoded": sum(len(clips) for clips in video_clips.values()),
        "total_num_clips_with_embeddings": sum(len(clips) for clips in video_clips.values()),
        "total_num_clips_with_caption": sum(1 for clips in video_clips.values() for _clip in clips)
        if caption_windows
        else 0,
        "total_num_caption_windows": caption_windows,
        "total_num_clips_with_webp": sum(len(clips) for clips in video_clips.values()),
    }
    for video_key, clips in video_clips.items():
        summary_overrides[video_key] = video_summary(clips=clips, filtered_clips=[], num_total_clips=len(clips)) | {
            "source_video": f"/inputs/{video_key}",
            "num_clips_passed": len(clips),
            "num_clips_transcoded": len(clips),
            "num_clips_with_embeddings": len(clips),
            "num_clips_with_caption": len(clips) if caption_windows else 0,
            "num_caption_windows": caption_windows,
            "num_clips_with_webp": len(clips),
        }
    return summary(**summary_overrides)


def _write_caption_meta(
    output_root: Path,
    clip_uuid: str,
    *,
    windows: list[dict[str, Any]],
    source_video: str = "/inputs/video.mp4",
) -> None:
    meta_path = output_root / "metas" / "v0" / f"{clip_uuid}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {
                "span_uuid": clip_uuid,
                "source_video": source_video,
                "has_caption": bool(windows),
                "num_caption_windows": len(windows),
                "windows": windows,
            }
        ),
        encoding="utf-8",
    )


def _window(start_frame: int, end_frame: int, caption: str) -> dict[str, Any]:
    return {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "caption_status": "success",
        "caption_failure_reason": None,
        "qwen_caption": caption,
    }


def _issue_codes(report: dict[str, Any]) -> list[str]:
    return [issue["code"] for issue in report["issues"]]


def _caption_issue_codes(report: dict[str, Any]) -> list[str]:
    return [issue["code"] for issue in report["issues"] if issue["code"].startswith("caption_")]


def _caption_comparison(report: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", report["feature_comparisons"]["captions"])


def test_no_caption_records_pass_with_zero_count_caption_comparison(tmp_path: Path) -> None:
    """Outputs with no caption evidence produce a passed zero-count caption comparison."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, _caption_summary({"video.mp4": ["clip-a"]}, caption_windows=0))
    write_summary(output_b, _caption_summary({"video.mp4": ["clip-a"]}, caption_windows=0))

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is True
    assert _caption_comparison(report) == {
        "status": "passed",
        "metrics": {
            "videos_with_captions_a": 0,
            "videos_with_captions_b": 0,
            "clips_with_captions_a": 0,
            "clips_with_captions_b": 0,
            "caption_windows_a": 0,
            "caption_windows_b": 0,
            "videos_compared": 0,
            "clips_compared": 0,
            "windows_compared": 0,
        },
    }


def test_matching_caption_structure_passes_regardless_of_caption_text(tmp_path: Path) -> None:
    """Caption text differences do not fail structural caption comparison."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, _caption_summary({"video.mp4": ["clip-a"]}, caption_windows=1))
    write_summary(output_b, _caption_summary({"video.mp4": ["clip-a"]}, caption_windows=1))
    _write_caption_meta(output_a, "clip-a", windows=[_window(0, 30, "a caption")])
    _write_caption_meta(output_b, "clip-a", windows=[_window(0, 30, "different text")])

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is True
    assert report["issues"] == []
    assert _caption_comparison(report) == {
        "status": "passed",
        "metrics": {
            "videos_with_captions_a": 1,
            "videos_with_captions_b": 1,
            "clips_with_captions_a": 1,
            "clips_with_captions_b": 1,
            "caption_windows_a": 1,
            "caption_windows_b": 1,
            "videos_compared": 1,
            "clips_compared": 1,
            "windows_compared": 1,
        },
    }


def test_caption_presence_mismatch_is_reported(tmp_path: Path) -> None:
    """If only one output has caption evidence, comparison emits a caption presence issue."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, _caption_summary({"video.mp4": ["clip-a"]}, caption_windows=1))
    write_summary(output_b, _caption_summary({"video.mp4": ["clip-a"]}, caption_windows=0))
    _write_caption_meta(output_a, "clip-a", windows=[_window(0, 30, "caption")])

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    assert "caption_presence_mismatch" in _issue_codes(report)
    caption_issue = next(issue for issue in report["issues"] if issue["code"] == "caption_presence_mismatch")
    assert caption_issue["feature"] == "captions"
    assert _caption_comparison(report)["status"] == "failed"


def test_caption_data_missing_is_reported_when_evidence_cannot_be_loaded(tmp_path: Path) -> None:
    """Caption evidence without loadable per-clip window records is reported directly."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, _caption_summary({"video.mp4": ["clip-a"]}, caption_windows=1))
    write_summary(output_b, _caption_summary({"video.mp4": ["clip-a"]}, caption_windows=1))

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    assert _issue_codes(report) == ["caption_data_missing", "caption_data_missing"]
    assert report["issues"][0]["output"] == "a"
    assert report["issues"][0]["details"]["missing_paths"] == [str(output_a / "metas" / "v0" / "clip-a.json")]
    assert report["issues"][1]["output"] == "b"
    assert report["issues"][1]["details"]["missing_paths"] == [str(output_b / "metas" / "v0" / "clip-a.json")]
    assert _caption_comparison(report)["status"] == "failed"


def test_partial_caption_load_below_summary_counts_is_reported(tmp_path: Path) -> None:
    """Loaded caption counts below summary counts fail even when some windows overlap."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, _caption_summary({"video.mp4": ["clip-a", "clip-b"]}, caption_windows=2))
    write_summary(output_b, _caption_summary({"video.mp4": ["clip-a", "clip-b"]}, caption_windows=2))
    _write_caption_meta(output_a, "clip-a", windows=[_window(0, 30, "caption")])
    _write_caption_meta(output_b, "clip-a", windows=[_window(0, 30, "caption")])

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    assert _issue_codes(report) == ["caption_data_missing", "caption_data_missing"]
    assert report["issues"][0]["output"] == "a"
    assert report["issues"][0]["details"] == {
        "expected_clips_with_captions": 2,
        "loaded_clips_with_captions": 1,
        "missing_clips_with_captions": 1,
        "expected_caption_windows": 2,
        "loaded_caption_windows": 1,
        "missing_caption_windows": 1,
        "missing_paths": [str(output_a / "metas" / "v0" / "clip-b.json")],
    }
    assert report["issues"][1]["output"] == "b"
    assert report["issues"][1]["details"] == {
        "expected_clips_with_captions": 2,
        "loaded_clips_with_captions": 1,
        "missing_clips_with_captions": 1,
        "expected_caption_windows": 2,
        "loaded_caption_windows": 1,
        "missing_caption_windows": 1,
        "missing_paths": [str(output_b / "metas" / "v0" / "clip-b.json")],
    }
    assert _caption_comparison(report)["status"] == "failed"


def test_caption_video_set_mismatch_is_reported(tmp_path: Path) -> None:
    """Captioned video set differences produce a caption-specific issue."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, _caption_summary({"a.mp4": ["clip-a"]}, caption_windows=1))
    write_summary(output_b, _caption_summary({"b.mp4": ["clip-b"]}, caption_windows=1))
    _write_caption_meta(output_a, "clip-a", windows=[_window(0, 30, "caption")], source_video="/inputs/a.mp4")
    _write_caption_meta(output_b, "clip-b", windows=[_window(0, 30, "caption")], source_video="/inputs/b.mp4")

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert "caption_video_set_mismatch" in _issue_codes(report)
    caption_issue = next(issue for issue in report["issues"] if issue["code"] == "caption_video_set_mismatch")
    assert caption_issue["details"] == {
        "videos_only_in_a": ["a.mp4"],
        "videos_only_in_b": ["b.mp4"],
    }


def test_caption_clip_set_mismatch_is_reported(tmp_path: Path) -> None:
    """Captioned clip UUID set differences are reported per shared video."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, _caption_summary({"video.mp4": ["clip-a"]}, caption_windows=1))
    write_summary(output_b, _caption_summary({"video.mp4": ["clip-b"]}, caption_windows=1))
    _write_caption_meta(output_a, "clip-a", windows=[_window(0, 30, "caption")])
    _write_caption_meta(output_b, "clip-b", windows=[_window(0, 30, "caption")])

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert "caption_clip_set_mismatch" in _issue_codes(report)
    caption_issue = next(issue for issue in report["issues"] if issue["code"] == "caption_clip_set_mismatch")
    assert caption_issue["video"] == "video.mp4"
    assert caption_issue["details"] == {
        "clips_only_in_a": ["clip-a"],
        "clips_only_in_b": ["clip-b"],
    }


def test_caption_clip_set_mismatch_still_compares_overlapping_clips(tmp_path: Path) -> None:
    """Clip set mismatches do not prevent window comparison for shared clips."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, _caption_summary({"video.mp4": ["clip-a", "shared"]}, caption_windows=2))
    write_summary(output_b, _caption_summary({"video.mp4": ["clip-b", "shared"]}, caption_windows=2))
    _write_caption_meta(output_a, "clip-a", windows=[_window(0, 30, "caption")])
    _write_caption_meta(output_a, "shared", windows=[_window(0, 30, "caption")])
    _write_caption_meta(output_b, "clip-b", windows=[_window(0, 30, "caption")])
    _write_caption_meta(output_b, "shared", windows=[_window(15, 45, "caption")])

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert _caption_issue_codes(report) == ["caption_clip_set_mismatch", "caption_window_set_mismatch"]
    caption_issues = [issue for issue in report["issues"] if issue["code"].startswith("caption_")]
    assert caption_issues[0]["details"] == {
        "clips_only_in_a": ["clip-a"],
        "clips_only_in_b": ["clip-b"],
    }
    assert caption_issues[1]["clip"] == "shared"
    assert caption_issues[1]["details"] == {
        "windows_only_in_a": ["0_30"],
        "windows_only_in_b": ["15_45"],
    }


def test_caption_window_set_mismatch_is_reported(tmp_path: Path) -> None:
    """Caption window frame range differences are reported per shared clip."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, _caption_summary({"video.mp4": ["clip-a"]}, caption_windows=1))
    write_summary(output_b, _caption_summary({"video.mp4": ["clip-a"]}, caption_windows=1))
    _write_caption_meta(output_a, "clip-a", windows=[_window(0, 30, "caption")])
    _write_caption_meta(output_b, "clip-a", windows=[_window(15, 45, "caption")])

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    assert report["issues"] == [
        {
            "code": "caption_window_set_mismatch",
            "message": "Caption window frame ranges differ between output A and output B",
            "feature": "captions",
            "video": "video.mp4",
            "clip": "clip-a",
            "details": {
                "windows_only_in_a": ["0_30"],
                "windows_only_in_b": ["15_45"],
            },
        }
    ]
    assert _caption_comparison(report)["status"] == "failed"
