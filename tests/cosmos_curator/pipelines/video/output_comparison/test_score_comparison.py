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
"""Tests for split output motion and aesthetic score comparison."""

import json
from pathlib import Path
from typing import Any, cast

import pytest

from cosmos_curator.pipelines.video.output_comparison.caption_comparator import CaptionFeatureComparator
from cosmos_curator.pipelines.video.output_comparison.compare_features import compare_features
from cosmos_curator.pipelines.video.output_comparison.comparison import OutputComparisonConfig, compare_split_outputs
from cosmos_curator.pipelines.video.output_comparison.report import ComparisonReport
from cosmos_curator.pipelines.video.output_comparison.score_comparator import (
    AESTHETIC_SCORE_FEATURE_NAME,
    MOTION_SCORE_FEATURE_NAME,
    ScoreComparisonPolicy,
    ScoreFeatureComparator,
)
from cosmos_curator.pipelines.video.output_comparison.summary_schema import OutputSummary

from .conftest import summary, video_summary, write_summary

_MISSING = object()


def _json_report(report: ComparisonReport) -> dict[str, Any]:
    return cast("dict[str, Any]", report.to_json_dict())


def _score_summary(video_clips: dict[str, list[str]]) -> dict[str, Any]:
    summary_overrides: dict[str, Any] = {
        "num_input_videos": len(video_clips),
        "num_input_videos_selected": len(video_clips),
        "num_processed_videos": len(video_clips),
        "total_num_clips_passed": sum(len(clips) for clips in video_clips.values()),
        "total_num_clips_transcoded": sum(len(clips) for clips in video_clips.values()),
        "total_num_clips_with_embeddings": sum(len(clips) for clips in video_clips.values()),
        "total_num_clips_with_webp": sum(len(clips) for clips in video_clips.values()),
    }
    for video_key, clips in video_clips.items():
        summary_overrides[video_key] = video_summary(clips=clips, filtered_clips=[], num_total_clips=len(clips)) | {
            "source_video": f"/inputs/{video_key}",
            "num_clips_passed": len(clips),
            "num_clips_transcoded": len(clips),
            "num_clips_with_embeddings": len(clips),
            "num_clips_with_webp": len(clips),
        }
    summary_data = summary(**summary_overrides)
    if "video.mp4" not in video_clips:
        del summary_data["video.mp4"]
    return summary_data


def _write_score_meta(
    output_root: Path,
    clip_uuid: str,
    *,
    motion_score: object = _MISSING,
    aesthetic_score: object = _MISSING,
) -> None:
    meta: dict[str, Any] = {
        "span_uuid": clip_uuid,
        "source_video": "/inputs/video.mp4",
        "windows": [],
    }
    if motion_score is not _MISSING:
        meta["motion_score"] = motion_score
    if aesthetic_score is not _MISSING:
        meta["aesthetic_score"] = aesthetic_score
    meta_path = output_root / "metas" / "v0" / f"{clip_uuid}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


def _write_invalid_meta(output_root: Path, clip_uuid: str) -> None:
    meta_path = output_root / "metas" / "v0" / f"{clip_uuid}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text("{not json", encoding="utf-8")


def _matching_motion_score() -> dict[str, float]:
    return {"global_mean": 0.25, "per_patch_min_256": 0.125}


def _feature(report: dict[str, Any], name: str) -> dict[str, Any]:
    return cast("dict[str, Any]", report["feature_comparisons"][name])


def _score_issue_codes(report: dict[str, Any]) -> list[str]:
    return [issue["code"] for issue in report["issues"] if issue.get("feature") in {"motion_score", "aesthetic_score"}]


@pytest.mark.parametrize(
    "policy_kwargs",
    [
        pytest.param({"abs_tolerance": -1e-6}, id="abs-negative"),
        pytest.param({"rel_tolerance": float("nan")}, id="rel-nan"),
        pytest.param({"abs_tolerance": float("inf")}, id="abs-inf"),
        pytest.param({"rel_tolerance": -1e-6}, id="rel-negative"),
    ],
)
def test_score_comparison_policy_rejects_invalid_tolerances(policy_kwargs: dict[str, float]) -> None:
    """Invalid tolerances fail before reaching math.isclose."""
    with pytest.raises(ValueError, match="finite number greater than or equal to 0"):
        ScoreComparisonPolicy(**policy_kwargs)


def test_matching_motion_and_aesthetic_scores_pass_with_metrics(tmp_path: Path) -> None:
    """Matching score metadata passes and records per-feature comparison metrics."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, _score_summary({"video.mp4": ["clip-a"]}))
    write_summary(output_b, _score_summary({"video.mp4": ["clip-a"]}))
    _write_score_meta(output_a, "clip-a", motion_score=_matching_motion_score(), aesthetic_score=0.75)
    _write_score_meta(output_b, "clip-a", motion_score=_matching_motion_score(), aesthetic_score=0.75)

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is True
    assert report["issues"] == []
    assert _feature(report, "motion_score") == {
        "status": "passed",
        "metrics": {
            "clips_with_scores_a": 1,
            "clips_with_scores_b": 1,
            "clips_compared": 1,
            "fields_compared": 2,
            "score_abs_tolerance": pytest.approx(1e-6),
            "score_rel_tolerance": pytest.approx(1e-6),
        },
    }
    assert _feature(report, "aesthetic_score") == {
        "status": "passed",
        "metrics": {
            "clips_with_scores_a": 1,
            "clips_with_scores_b": 1,
            "clips_compared": 1,
            "fields_compared": 1,
            "score_abs_tolerance": pytest.approx(1e-6),
            "score_rel_tolerance": pytest.approx(1e-6),
        },
    }


def test_score_tolerances_are_configured_per_feature(tmp_path: Path) -> None:
    """Motion and aesthetic score tolerances are controlled independently."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, _score_summary({"video.mp4": ["clip-a"]}))
    write_summary(output_b, _score_summary({"video.mp4": ["clip-a"]}))
    _write_score_meta(output_a, "clip-a", motion_score=_matching_motion_score(), aesthetic_score=0.75)
    _write_score_meta(
        output_b,
        "clip-a",
        motion_score={"global_mean": 0.251, "per_patch_min_256": 0.125},
        aesthetic_score=0.751,
    )

    motion_tolerant_report = _json_report(
        compare_split_outputs(
            output_a,
            output_b,
            config=OutputComparisonConfig(
                motion_score_policy=ScoreComparisonPolicy(abs_tolerance=0.01, rel_tolerance=0.0),
                aesthetic_score_policy=ScoreComparisonPolicy(abs_tolerance=0.0001, rel_tolerance=0.0),
            ),
        )
    )
    aesthetic_tolerant_report = _json_report(
        compare_split_outputs(
            output_a,
            output_b,
            config=OutputComparisonConfig(
                motion_score_policy=ScoreComparisonPolicy(abs_tolerance=0.0001, rel_tolerance=0.0),
                aesthetic_score_policy=ScoreComparisonPolicy(abs_tolerance=0.01, rel_tolerance=0.0),
            ),
        )
    )
    tolerant_report = _json_report(
        compare_split_outputs(
            output_a,
            output_b,
            config=OutputComparisonConfig(
                motion_score_policy=ScoreComparisonPolicy(abs_tolerance=0.01, rel_tolerance=0.0),
                aesthetic_score_policy=ScoreComparisonPolicy(abs_tolerance=0.01, rel_tolerance=0.0),
            ),
        )
    )

    assert tolerant_report["passed"] is True
    assert motion_tolerant_report["passed"] is False
    assert _score_issue_codes(motion_tolerant_report) == ["aesthetic_score_value_mismatch"]
    assert aesthetic_tolerant_report["passed"] is False
    assert _score_issue_codes(aesthetic_tolerant_report) == ["motion_score_value_mismatch"]
    motion_issue = next(issue for issue in aesthetic_tolerant_report["issues"] if issue["feature"] == "motion_score")
    assert motion_issue["video"] == "video.mp4"
    assert motion_issue["clip"] == "clip-a"
    assert motion_issue["field"] == "motion_score.global_mean"
    assert motion_issue["details"]["a"] == pytest.approx(0.25)
    assert motion_issue["details"]["b"] == pytest.approx(0.251)
    assert motion_issue["details"]["score_abs_tolerance"] == pytest.approx(0.0001)


def test_missing_score_field_on_one_side_reports_feature_issue(tmp_path: Path) -> None:
    """One-sided score metadata is reported instead of being silently ignored."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, _score_summary({"video.mp4": ["clip-a"]}))
    write_summary(output_b, _score_summary({"video.mp4": ["clip-a"]}))
    _write_score_meta(output_a, "clip-a", motion_score=_matching_motion_score(), aesthetic_score=0.75)
    _write_score_meta(output_b, "clip-a")

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    assert _score_issue_codes(report) == ["aesthetic_score_field_missing", "motion_score_field_missing"]
    assert report["issues"][0] == {
        "code": "aesthetic_score_field_missing",
        "message": "Score field is present on only one output",
        "feature": "aesthetic_score",
        "video": "video.mp4",
        "clip": "clip-a",
        "field": "aesthetic_score",
        "output": "b",
        "details": {"a_present": True, "b_present": False},
    }


def test_invalid_score_field_shapes_report_feature_issues(tmp_path: Path) -> None:
    """Invalid score metadata shape produces structured issues and does not crash."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, _score_summary({"video.mp4": ["clip-a"]}))
    write_summary(output_b, _score_summary({"video.mp4": ["clip-a"]}))
    _write_score_meta(
        output_a,
        "clip-a",
        motion_score={"global_mean": "bad", "per_patch_min_256": 0.125},
        aesthetic_score="bad",
    )
    _write_score_meta(output_b, "clip-a", motion_score=_matching_motion_score(), aesthetic_score=0.75)

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    invalid_issues = [issue for issue in report["issues"] if issue["code"].endswith("_field_invalid")]
    assert [issue["field"] for issue in invalid_issues] == ["aesthetic_score", "motion_score.global_mean"]
    assert invalid_issues[0]["feature"] == "aesthetic_score"
    assert invalid_issues[0]["output"] == "a"
    assert invalid_issues[0]["details"]["reason"] == "must be numeric"


def test_invalid_score_field_is_reported_when_other_side_is_missing(tmp_path: Path) -> None:
    """Malformed score metadata is reported even when the other output lacks that score."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, _score_summary({"video.mp4": ["clip-a"]}))
    write_summary(output_b, _score_summary({"video.mp4": ["clip-a"]}))
    _write_score_meta(
        output_a,
        "clip-a",
        motion_score={"global_mean": "bad"},
    )
    _write_score_meta(output_b, "clip-a")

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    motion_issues = [
        (issue["code"], issue.get("field"), issue.get("output"))
        for issue in report["issues"]
        if issue.get("feature") == "motion_score"
    ]
    assert motion_issues == [
        ("motion_score_field_invalid", "motion_score.global_mean", "a"),
        ("motion_score_field_invalid", "motion_score.per_patch_min_256", "a"),
        ("motion_score_field_missing", "motion_score", "b"),
    ]


def test_non_finite_score_values_report_invalid_fields(tmp_path: Path) -> None:
    """NaN and infinite score values are invalid, not comparable score values."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, _score_summary({"video.mp4": ["clip-a"]}))
    write_summary(output_b, _score_summary({"video.mp4": ["clip-a"]}))
    _write_score_meta(
        output_a,
        "clip-a",
        motion_score={"global_mean": float("nan"), "per_patch_min_256": float("inf")},
        aesthetic_score=float("-inf"),
    )
    _write_score_meta(output_b, "clip-a", motion_score=_matching_motion_score(), aesthetic_score=0.75)

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    invalid_issues = [issue for issue in report["issues"] if issue["code"].endswith("_field_invalid")]
    assert [(issue["feature"], issue["field"], issue["details"]["reason"]) for issue in invalid_issues] == [
        ("aesthetic_score", "aesthetic_score", "must be numeric"),
        ("motion_score", "motion_score.global_mean", "must be numeric"),
        ("motion_score", "motion_score.per_patch_min_256", "must be numeric"),
    ]


def test_invalid_score_metadata_reports_unavailable_artifact(tmp_path: Path) -> None:
    """Unreadable metadata is reported distinctly from absent score fields."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, _score_summary({"video.mp4": ["clip-a"]}))
    write_summary(output_b, _score_summary({"video.mp4": ["clip-a"]}))
    _write_score_meta(output_a, "clip-a", motion_score=_matching_motion_score(), aesthetic_score=0.75)
    _write_invalid_meta(output_b, "clip-a")

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    assert _score_issue_codes(report) == [
        "aesthetic_score_metadata_unavailable",
        "motion_score_metadata_unavailable",
    ]
    metadata_issues = [issue for issue in report["issues"] if issue["code"].endswith("_metadata_unavailable")]
    assert [(issue["feature"], issue["output"], issue["field"]) for issue in metadata_issues] == [
        ("aesthetic_score", "b", "aesthetic_score"),
        ("motion_score", "b", "motion_score"),
    ]
    assert metadata_issues[0]["details"]["reason"] == "invalid"
    assert metadata_issues[0]["details"]["metadata_path"] == str(output_b / "metas" / "v0" / "clip-a.json")
    assert "JSONDecodeError" in metadata_issues[0]["details"]["error"]
    assert _feature(report, "motion_score")["status"] == "failed"
    assert _feature(report, "aesthetic_score")["status"] == "failed"


def test_missing_score_metadata_reports_unavailable_artifact_when_counterpart_has_scores(tmp_path: Path) -> None:
    """Missing metadata is reported when the other output has score evidence for that feature."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, _score_summary({"video.mp4": ["clip-a"]}))
    write_summary(output_b, _score_summary({"video.mp4": ["clip-a"]}))
    _write_score_meta(output_a, "clip-a", motion_score=_matching_motion_score(), aesthetic_score=0.75)

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    assert _score_issue_codes(report) == [
        "aesthetic_score_metadata_unavailable",
        "motion_score_metadata_unavailable",
    ]
    metadata_issues = [issue for issue in report["issues"] if issue["code"].endswith("_metadata_unavailable")]
    assert [(issue["feature"], issue["output"], issue["details"]["reason"]) for issue in metadata_issues] == [
        ("aesthetic_score", "b", "missing"),
        ("motion_score", "b", "missing"),
    ]
    assert _feature(report, "motion_score")["status"] == "failed"
    assert _feature(report, "motion_score")["metrics"]["clips_with_scores_a"] == 1
    assert _feature(report, "motion_score")["metrics"]["clips_with_scores_b"] == 0
    assert _feature(report, "aesthetic_score")["status"] == "failed"
    assert _feature(report, "aesthetic_score")["metrics"]["clips_with_scores_a"] == 1
    assert _feature(report, "aesthetic_score")["metrics"]["clips_with_scores_b"] == 0


def test_disabled_score_metadata_is_skipped(tmp_path: Path) -> None:
    """Outputs with no score fields on either side do not fail score comparison."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, _score_summary({"video.mp4": ["clip-a"]}))
    write_summary(output_b, _score_summary({"video.mp4": ["clip-a"]}))
    _write_score_meta(output_a, "clip-a")
    _write_score_meta(output_b, "clip-a")

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is True
    assert _feature(report, "motion_score")["status"] == "skipped"
    assert _feature(report, "motion_score")["metrics"]["fields_compared"] == 0
    assert _feature(report, "aesthetic_score")["status"] == "skipped"
    assert _feature(report, "aesthetic_score")["metrics"]["fields_compared"] == 0


def test_side_only_clips_do_not_emit_score_value_mismatches(tmp_path: Path) -> None:
    """Score comparison loads side-only clips but only compares clips present on both sides."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, _score_summary({"video.mp4": ["clip-a-only", "shared"]}))
    write_summary(output_b, _score_summary({"video.mp4": ["clip-b-only", "shared"]}))
    _write_score_meta(output_a, "clip-a-only", motion_score=_matching_motion_score(), aesthetic_score=0.75)
    _write_score_meta(output_a, "shared", motion_score=_matching_motion_score(), aesthetic_score=0.75)
    _write_score_meta(
        output_b,
        "clip-b-only",
        motion_score={"global_mean": 0.99, "per_patch_min_256": 0.88},
        aesthetic_score=0.01,
    )
    _write_score_meta(output_b, "shared", motion_score=_matching_motion_score(), aesthetic_score=0.75)

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    assert any(issue["code"] == "clip_uuid_set_mismatch" for issue in report["issues"])
    assert not any(issue["code"].endswith("_value_mismatch") for issue in report["issues"])
    assert _feature(report, "motion_score")["metrics"]["clips_compared"] == 1
    assert _feature(report, "aesthetic_score")["metrics"]["clips_compared"] == 1


@pytest.mark.parametrize(
    ("selector_kwargs", "expected_compared_clip"),
    [
        pytest.param({"video_limit": 1}, "clip-first", id="limit"),
        pytest.param({"selected_video_key": "second.mp4"}, "clip-second", id="selected-video-key"),
    ],
)
def test_score_comparison_respects_video_selectors(
    tmp_path: Path,
    selector_kwargs: dict[str, int | str],
    expected_compared_clip: str,
) -> None:
    """Video selectors scope score comparison to the selected clip set."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, _score_summary({"first.mp4": ["clip-first"], "second.mp4": ["clip-second"]}))
    write_summary(output_b, _score_summary({"first.mp4": ["clip-first"], "second.mp4": ["clip-second"]}))
    for output_root in (output_a, output_b):
        _write_score_meta(
            output_root, expected_compared_clip, motion_score=_matching_motion_score(), aesthetic_score=0.75
        )
    unselected_clip = "clip-second" if expected_compared_clip == "clip-first" else "clip-first"
    _write_score_meta(output_a, unselected_clip, motion_score=_matching_motion_score(), aesthetic_score=0.75)
    _write_score_meta(
        output_b,
        unselected_clip,
        motion_score={"global_mean": 0.99, "per_patch_min_256": 0.88},
        aesthetic_score=0.01,
    )

    report = _json_report(compare_split_outputs(output_a, output_b, **selector_kwargs))

    assert report["passed"] is True
    assert _feature(report, "motion_score")["metrics"]["clips_compared"] == 1
    assert _feature(report, "aesthetic_score")["metrics"]["clips_compared"] == 1


def test_default_metadata_backed_features_share_clip_metadata_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caption and score features share one metadata load for each selected clip side."""
    read_paths: list[str] = []

    def fake_read_json_object(path: object, *, client_params: object) -> dict[str, object]:
        _ = client_params
        read_paths.append(str(path))
        return {
            "span_uuid": Path(str(path)).stem,
            "motion_score": _matching_motion_score(),
            "aesthetic_score": 0.75,
            "windows": [
                {
                    "start_frame": 0,
                    "end_frame": 30,
                    "caption_status": "success",
                    "qwen_caption": "caption",
                }
            ],
        }

    summary_value = OutputSummary.from_json_dict(
        summary(
            **{
                "video.mp4": video_summary(clips=["clip-a"])
                | {
                    "num_clips_with_caption": 1,
                    "num_caption_windows": 1,
                },
                "total_num_clips_with_caption": 1,
                "total_num_caption_windows": 1,
            }
        )
    )
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.output_comparison.video_artifacts._read_json_object",
        fake_read_json_object,
    )

    result = compare_features(
        tmp_path / "output-a",
        tmp_path / "output-b",
        summary_value,
        summary_value,
        feature_planners=(
            CaptionFeatureComparator(),
            ScoreFeatureComparator(AESTHETIC_SCORE_FEATURE_NAME),
            ScoreFeatureComparator(MOTION_SCORE_FEATURE_NAME),
        ),
    )

    assert result.issues == ()
    assert sorted(result.feature_comparisons) == ["aesthetic_score", "captions", "motion_score"]
    assert read_paths == [
        str(tmp_path / "output-a" / "metas" / "v0" / "clip-a.json"),
        str(tmp_path / "output-b" / "metas" / "v0" / "clip-a.json"),
    ]
