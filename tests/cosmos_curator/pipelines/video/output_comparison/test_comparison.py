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
"""Tests for split output summary comparison."""

from collections import Counter
from pathlib import Path
from typing import Any, cast

import attrs
import pytest

from cosmos_curator.pipelines.video.output_comparison.comparison import compare_split_outputs
from cosmos_curator.pipelines.video.output_comparison.report import ComparisonReport, Issue, SummaryComparison
from cosmos_curator.pipelines.video.output_comparison.summary_policy import DEFAULT_SUMMARY_POLICY
from cosmos_curator.pipelines.video.output_comparison.summary_rules import (
    SummaryComparisonContext,
    SummaryRuleResult,
    rules_for_policy,
)

from .conftest import summary, video_summary, write_summary


def _json_report(report: ComparisonReport) -> dict[str, Any]:
    return cast("dict[str, Any]", report.to_json_dict())


def _issue_codes(report: dict[str, Any]) -> list[str]:
    return [issue["code"] for issue in report["issues"]]


def _issue_for(report: dict[str, Any], code: str, field: str) -> dict[str, Any]:
    return next(issue for issue in report["issues"] if issue["code"] == code and issue["field"] == field)


class _CountingRule:
    def __init__(self, exact_count: int) -> None:
        self._exact_count = exact_count

    def compare(self, context: SummaryComparisonContext) -> SummaryRuleResult:
        assert context.videos_in_both == ("video.mp4",)
        return SummaryRuleResult(exact_top_level_fields_compared=self._exact_count)


def test_comparison_report_passed_is_derived_from_issues() -> None:
    """Report pass/fail state is derived from the current issue set."""
    summary_comparison = SummaryComparison()
    passing_report = ComparisonReport.from_issues("output-a", "output-b", summary_comparison, [])
    issue = Issue(code="example", message="Example issue")
    failing_report = ComparisonReport.from_issues("output-a", "output-b", summary_comparison, [issue])

    assert passing_report.passed is True
    assert passing_report.to_json_dict()["passed"] is True
    assert failing_report.passed is False
    assert failing_report.issues == (issue,)
    assert failing_report.to_json_dict()["passed"] is False


def test_issue_summary_load_failed_builds_structured_issue() -> None:
    """Summary load failures use the canonical issue shape."""
    issue = Issue.summary_load_failed(
        "/path/summary.json",
        "a",
        "MissingSummaryFieldError",
        "summary.json missing required field 'num_processed_videos'",
        field="num_processed_videos",
    )

    assert issue.to_json_dict() == {
        "code": "summary_load_failed",
        "message": (
            "Failed to load output A summary at /path/summary.json: "
            "summary.json missing required field 'num_processed_videos'"
        ),
        "details": {
            "path": "/path/summary.json",
            "error_type": "MissingSummaryFieldError",
            "error": "summary.json missing required field 'num_processed_videos'",
        },
        "output": "a",
        "field": "num_processed_videos",
    }


def test_matching_summary_accounting_passes(tmp_path: Path) -> None:
    """Matching summary accounting produces a passing report with comparison counts."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, summary())
    write_summary(output_b, summary())

    typed_report = compare_split_outputs(output_a, output_b)
    report = _json_report(typed_report)

    assert typed_report.summary_comparison.videos_in_both == 1
    assert report["passed"] is True
    assert report["output_a"] == str(output_a)
    assert report["output_b"] == str(output_b)
    assert report["issues"] == []
    assert report["summary_comparison"] == {
        "videos_in_both": 1,
        "videos_only_in_a": [],
        "videos_only_in_b": [],
        "exact_top_level_fields_compared": len(DEFAULT_SUMMARY_POLICY.exact_top_level_fields),
        "token_fields_compared": len(DEFAULT_SUMMARY_POLICY.token_fields),
        "per_video_fields_compared": (
            1 + len(DEFAULT_SUMMARY_POLICY.common_video_fields) + len(DEFAULT_SUMMARY_POLICY.processed_video_fields)
        ),
    }


def test_compare_split_outputs_accepts_custom_rules_and_adds_rule_counts(tmp_path: Path) -> None:
    """Public comparison API runs injected rules and accumulates their count updates."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, summary())
    write_summary(output_b, summary())

    report = _json_report(compare_split_outputs(output_a, output_b, rules=[_CountingRule(2), _CountingRule(3)]))

    assert report["passed"] is True
    assert report["issues"] == []
    assert report["summary_comparison"] == {
        "videos_in_both": 1,
        "videos_only_in_a": [],
        "videos_only_in_b": [],
        "exact_top_level_fields_compared": 5,
        "token_fields_compared": 0,
        "per_video_fields_compared": 0,
    }


def test_compare_split_outputs_accepts_empty_rules(tmp_path: Path) -> None:
    """An explicit empty rule sequence skips default summary rules."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, summary(total_num_clips_passed=2))
    write_summary(output_b, summary(total_num_clips_passed=1))

    report = _json_report(compare_split_outputs(output_a, output_b, rules=[]))

    assert report["passed"] is True
    assert report["issues"] == []
    assert report["summary_comparison"] == {
        "videos_in_both": 1,
        "videos_only_in_a": [],
        "videos_only_in_b": [],
        "exact_top_level_fields_compared": 0,
        "token_fields_compared": 0,
        "per_video_fields_compared": 0,
    }


def test_missing_summary_produces_structured_failure(tmp_path: Path) -> None:
    """Missing summary.json is captured as a structured load failure."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    output_a.mkdir()
    write_summary(output_b, summary())

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    assert report["summary_comparison"]["videos_in_both"] == 0
    assert report["issues"][0]["code"] == "summary_load_failed"
    assert report["issues"][0]["output"] == "a"
    assert report["issues"][0]["details"]["path"] == str(output_a / "summary.json")


def test_missing_second_summary_produces_structured_failure(tmp_path: Path) -> None:
    """Missing summary.json from output B is captured as a structured load failure."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, summary())
    output_b.mkdir()

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    assert report["summary_comparison"]["videos_in_both"] == 0
    assert report["issues"][0]["code"] == "summary_load_failed"
    assert report["issues"][0]["output"] == "b"
    assert report["issues"][0]["details"]["path"] == str(output_b / "summary.json")


def test_missing_required_top_level_field_load_failure_identifies_output_path_and_field(tmp_path: Path) -> None:
    """Missing required summary fields report the output, path, error, and field."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    summary_a = summary()
    del summary_a["num_processed_videos"]
    write_summary(output_a, summary_a)
    write_summary(output_b, summary())

    report = _json_report(compare_split_outputs(output_a, output_b))

    expected_error = "summary.json missing required field 'num_processed_videos'"
    assert report["passed"] is False
    assert report["issues"][0] == {
        "code": "summary_load_failed",
        "message": f"Failed to load output A summary at {output_a / 'summary.json'}: {expected_error}",
        "output": "a",
        "field": "num_processed_videos",
        "details": {
            "path": str(output_a / "summary.json"),
            "error_type": "MissingSummaryFieldError",
            "error": expected_error,
        },
    }


def test_invalid_summary_produces_structured_failure(tmp_path: Path) -> None:
    """Invalid summary.json is captured as a structured load failure."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    output_a.mkdir()
    (output_a / "summary.json").write_text("{invalid", encoding="utf-8")
    write_summary(output_b, summary())

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    assert report["issues"][0]["code"] == "summary_load_failed"
    assert report["issues"][0]["output"] == "a"
    assert report["issues"][0]["details"]["error_type"] == "JSONDecodeError"


def test_non_object_summary_produces_structured_failure(tmp_path: Path) -> None:
    """Valid JSON with the wrong top-level shape is captured as a structured load failure."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    output_a.mkdir()
    (output_a / "summary.json").write_text("[]", encoding="utf-8")
    write_summary(output_b, summary())

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    assert report["issues"][0]["code"] == "summary_load_failed"
    assert report["issues"][0]["output"] == "a"
    assert report["issues"][0]["details"]["error_type"] == "ValueError"


def test_video_keys_only_in_one_output_are_reported(tmp_path: Path) -> None:
    """Video keys present in only one output are listed and reported."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, summary(**{"a-only.mp4": video_summary(video_uuid="a-only")}))
    write_summary(output_b, summary(**{"b-only.mp4": video_summary(video_uuid="b-only")}))

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    assert report["summary_comparison"]["videos_in_both"] == 1
    assert report["summary_comparison"]["videos_only_in_a"] == ["a-only.mp4"]
    assert report["summary_comparison"]["videos_only_in_b"] == ["b-only.mp4"]
    assert "video_keys_mismatch" in _issue_codes(report)


def test_exact_top_level_field_mismatch_is_reported(tmp_path: Path) -> None:
    """Exact top-level accounting field mismatches include the field and values."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, summary(total_num_clips_passed=2))
    write_summary(output_b, summary(total_num_clips_passed=1))

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    assert report["issues"][0] == {
        "code": "summary_field_mismatch",
        "message": "Summary field differs between output A and output B",
        "field": "total_num_clips_passed",
        "details": {
            "a": 2,
            "b": 1,
            "a_present": True,
            "b_present": True,
        },
    }


def test_deterministic_top_level_accounting_mismatches_are_reported(tmp_path: Path) -> None:
    """Deterministic media duration and byte accounting fields are exact comparisons."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(
        output_a,
        summary(total_video_duration=10.0, total_clip_duration=8.0, max_clip_duration=4.0, total_video_bytes=12345),
    )
    write_summary(
        output_b,
        summary(total_video_duration=11.0, total_clip_duration=9.0, max_clip_duration=5.0, total_video_bytes=23456),
    )

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    assert Counter((issue["code"], issue["field"]) for issue in report["issues"]) >= Counter(
        {
            ("summary_field_mismatch", "total_video_duration"): 1,
            ("summary_field_mismatch", "total_clip_duration"): 1,
            ("summary_field_mismatch", "max_clip_duration"): 1,
            ("summary_field_mismatch", "total_video_bytes"): 1,
        }
    )


def test_invalid_required_top_level_fields_produce_load_failures(tmp_path: Path) -> None:
    """Invalid required top-level fields fail summary loading before comparison rules run."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    summary_b = summary()
    del summary_b["total_num_clips_passed"]
    write_summary(output_a, summary(total_num_clips_passed=None))
    write_summary(output_b, summary_b)

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    assert report["summary_comparison"]["videos_in_both"] == 0
    assert [issue["code"] for issue in report["issues"]] == ["summary_load_failed", "summary_load_failed"]
    assert report["issues"][0]["output"] == "a"
    assert report["issues"][0]["field"] == "total_num_clips_passed"
    assert report["issues"][0]["details"]["error_type"] == "InvalidSummaryFieldError"
    assert "total_num_clips_passed" in report["issues"][0]["details"]["error"]
    assert report["issues"][1]["output"] == "b"
    assert report["issues"][1]["field"] == "total_num_clips_passed"
    assert report["issues"][1]["details"]["error_type"] == "MissingSummaryFieldError"
    assert "total_num_clips_passed" in report["issues"][1]["details"]["error"]


def test_token_totals_use_configured_tolerances(tmp_path: Path) -> None:
    """Token total fields pass or fail according to the configured tolerances."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, summary(total_prompt_tokens=100, total_output_tokens=50))
    write_summary(output_b, summary(total_prompt_tokens=105, total_output_tokens=50))

    passing_report = _json_report(
        compare_split_outputs(
            output_a,
            output_b,
            token_count_abs_tolerance=5,
            token_count_rel_tolerance=0.0,
        )
    )
    failing_report = _json_report(
        compare_split_outputs(
            output_a,
            output_b,
            token_count_abs_tolerance=4,
            token_count_rel_tolerance=0.0,
        )
    )

    assert passing_report["passed"] is True
    assert failing_report["passed"] is False
    assert failing_report["issues"][0]["code"] == "token_field_mismatch"
    assert failing_report["issues"][0]["field"] == "total_prompt_tokens"
    assert failing_report["issues"][0]["details"] == {
        "a": 100,
        "b": 105,
        "abs_delta": 5,
        "rel_delta": 5 / 105,
        "token_count_abs_tolerance": 4,
        "token_count_rel_tolerance": 0.0,
    }


def test_zero_token_totals_compare_without_relative_delta_error(tmp_path: Path) -> None:
    """Zero-valued token totals compare without dividing by zero."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, summary(total_prompt_tokens=0, total_output_tokens=0))
    write_summary(output_b, summary(total_prompt_tokens=0, total_output_tokens=0))

    report = _json_report(compare_split_outputs(output_a, output_b, token_count_rel_tolerance=0.1))

    assert report["passed"] is True


def test_custom_tolerant_policy_field_requires_numeric_value(tmp_path: Path) -> None:
    """Custom tolerance-compared policy fields must point to numeric summary values."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, summary(custom_total="N/A"))
    write_summary(output_b, summary(custom_total=1))
    policy = attrs.evolve(
        DEFAULT_SUMMARY_POLICY,
        exact_top_level_fields=(),
        token_fields=("custom_total",),
        common_video_fields=(),
        processed_video_fields=(),
        clip_list_fields=(),
    )

    with pytest.raises(TypeError, match=r"summary field 'custom_total' must be numeric for token comparison"):
        compare_split_outputs(output_a, output_b, rules=rules_for_policy(policy))


def test_nonnumeric_token_totals_produce_load_failure(tmp_path: Path) -> None:
    """Required token totals must be numeric."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, summary(total_prompt_tokens=None, total_output_tokens="N/A"))
    write_summary(output_b, summary())

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    assert report["issues"] == [
        {
            "code": "summary_load_failed",
            "message": (
                f"Failed to load output A summary at {output_a / 'summary.json'}: "
                "summary.json field 'total_prompt_tokens' must be a number"
            ),
            "output": "a",
            "field": "total_prompt_tokens",
            "details": {
                "path": str(output_a / "summary.json"),
                "error_type": "InvalidSummaryFieldError",
                "error": "summary.json field 'total_prompt_tokens' must be a number",
            },
        }
    ]


def test_bool_token_total_produces_load_failure(tmp_path: Path) -> None:
    """Boolean token totals do not pass summary validation as numbers."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, summary(total_prompt_tokens=True))
    write_summary(output_b, summary(total_prompt_tokens=1))

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    assert report["issues"][0]["code"] == "summary_load_failed"
    assert report["issues"][0]["output"] == "a"
    assert report["issues"][0]["field"] == "total_prompt_tokens"
    assert report["issues"][0]["details"]["error_type"] == "InvalidSummaryFieldError"
    assert report["issues"][0]["details"]["error"] == "summary.json field 'total_prompt_tokens' must be a number"


def test_processed_state_mismatch_is_reported_without_processed_field_noise(tmp_path: Path) -> None:
    """Processed/unprocessed mismatches report a single state issue for the shared video."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, summary(**{"video.mp4": video_summary()}))
    write_summary(
        output_b,
        summary(
            **{
                "video.mp4": {
                    "source_video": "/inputs/video.mp4",
                    "processed": False,
                }
            }
        ),
    )

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    assert report["issues"] == [
        {
            "code": "video_processed_state_mismatch",
            "message": "Video processed state differs between output A and output B",
            "video": "video.mp4",
            "field": "processed",
            "details": {
                "a": True,
                "b": False,
                "a_present": True,
                "b_present": True,
            },
        }
    ]


def test_exact_per_video_field_mismatch_is_reported(tmp_path: Path) -> None:
    """Exact per-video accounting field mismatches include the video, field, and values."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, summary(**{"video.mp4": video_summary()}))
    write_summary(output_b, summary(**{"video.mp4": video_summary(video_uuid="different-video")}))

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    assert report["issues"][0] == {
        "code": "video_field_mismatch",
        "message": "Video summary field differs between output A and output B",
        "video": "video.mp4",
        "field": "video_uuid",
        "details": {
            "a": "video-uuid",
            "b": "different-video",
            "a_present": True,
            "b_present": True,
        },
    }


def test_common_per_video_field_mismatch_is_reported(tmp_path: Path) -> None:
    """Common per-video field mismatches include the video, field, and values."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(output_a, summary(**{"video.mp4": video_summary() | {"source_video": "/inputs/a.mp4"}}))
    write_summary(output_b, summary(**{"video.mp4": video_summary() | {"source_video": "/inputs/b.mp4"}}))

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    assert _issue_for(report, "video_field_mismatch", "source_video") == {
        "code": "video_field_mismatch",
        "message": "Video summary field differs between output A and output B",
        "video": "video.mp4",
        "field": "source_video",
        "details": {
            "a": "/inputs/a.mp4",
            "b": "/inputs/b.mp4",
            "a_present": True,
            "b_present": True,
        },
    }


def test_clip_uuid_accounting_mismatches_are_reported(tmp_path: Path) -> None:
    """Clip and filtered clip UUID list accounting is compared without reading artifacts."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(
        output_a,
        summary(
            **{
                "video.mp4": video_summary(
                    clips=["clip-a", "clip-b"],
                    filtered_clips=["filtered-a"],
                )
            }
        ),
    )
    write_summary(
        output_b,
        summary(
            **{
                "video.mp4": video_summary(
                    clips=["clip-a", "clip-c"],
                    filtered_clips=["filtered-a", "filtered-b"],
                )
            }
        ),
    )

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    assert Counter(_issue_codes(report)) == Counter(
        {
            "clip_uuid_set_mismatch": 2,
            "clip_list_length_mismatch": 1,
        }
    )
    assert _issue_for(report, "clip_uuid_set_mismatch", "clips") == {
        "code": "clip_uuid_set_mismatch",
        "message": "Video summary clip UUID set differs between output A and output B",
        "video": "video.mp4",
        "field": "clips",
        "details": {
            "only_in_a": ["clip-b"],
            "only_in_b": ["clip-c"],
        },
    }
    filtered_length_issue = _issue_for(report, "clip_list_length_mismatch", "filtered_clips")
    assert filtered_length_issue["details"] == {
        "a_count": 1,
        "b_count": 2,
    }


def test_clip_uuid_duplicate_accounting_mismatches_are_reported(tmp_path: Path) -> None:
    """Clip UUID accounting compares duplicate counts, not just unique UUID membership."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(
        output_a,
        summary(
            **{
                "video.mp4": video_summary(clips=["clip-a", "clip-a", "clip-b"]),
            }
        ),
    )
    write_summary(
        output_b,
        summary(
            **{
                "video.mp4": video_summary(clips=["clip-a", "clip-b", "clip-b"]),
            }
        ),
    )

    report = _json_report(compare_split_outputs(output_a, output_b))

    assert report["passed"] is False
    assert _issue_for(report, "clip_uuid_set_mismatch", "clips") == {
        "code": "clip_uuid_set_mismatch",
        "message": "Video summary clip UUID set differs between output A and output B",
        "video": "video.mp4",
        "field": "clips",
        "details": {
            "only_in_a": [],
            "only_in_b": [],
            "count_mismatches": [
                {
                    "clip_uuid": "clip-a",
                    "a_count": 2,
                    "b_count": 1,
                },
                {
                    "clip_uuid": "clip-b",
                    "a_count": 1,
                    "b_count": 2,
                },
            ],
        },
    }


def test_custom_clip_list_policy_fields_are_compared(tmp_path: Path) -> None:
    """Clip-list policy fields are real extension points, not only the default fields."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(
        output_a,
        summary(
            **{
                "video.mp4": video_summary() | {"kept_clips": ["clip-a", "clip-b"]},
            }
        ),
    )
    write_summary(
        output_b,
        summary(
            **{
                "video.mp4": video_summary() | {"kept_clips": ["clip-a", "clip-c"]},
            }
        ),
    )
    policy = attrs.evolve(DEFAULT_SUMMARY_POLICY, clip_list_fields=("kept_clips",))

    report = _json_report(compare_split_outputs(output_a, output_b, rules=rules_for_policy(policy)))

    assert report["passed"] is False
    assert _issue_for(report, "clip_uuid_set_mismatch", "kept_clips") == {
        "code": "clip_uuid_set_mismatch",
        "message": "Video summary clip UUID set differs between output A and output B",
        "video": "video.mp4",
        "field": "kept_clips",
        "details": {
            "only_in_a": ["clip-b"],
            "only_in_b": ["clip-c"],
        },
    }


def test_custom_clip_list_policy_field_requires_list_value(tmp_path: Path) -> None:
    """Custom clip-list policy fields must point to list-shaped summary values."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    write_summary(
        output_a,
        summary(
            **{
                "video.mp4": video_summary() | {"kept_clips": "clip-a"},
            }
        ),
    )
    write_summary(
        output_b,
        summary(
            **{
                "video.mp4": video_summary() | {"kept_clips": ["clip-a"]},
            }
        ),
    )
    policy = attrs.evolve(DEFAULT_SUMMARY_POLICY, clip_list_fields=("kept_clips",))

    with pytest.raises(TypeError, match=r"summary field 'kept_clips' must be a list for clip UUID comparison"):
        compare_split_outputs(output_a, output_b, rules=rules_for_policy(policy))
