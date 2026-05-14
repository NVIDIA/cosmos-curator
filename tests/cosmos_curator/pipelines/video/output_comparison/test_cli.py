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
"""Tests for split output comparison CLI."""

import json
from pathlib import Path

import pytest

from cosmos_curator.pipelines.video.output_comparison.cli import _format_stdout_summary, main
from cosmos_curator.pipelines.video.output_comparison.report import ComparisonReport, Issue, SummaryComparison

from .conftest import summary, write_summary


def test_cli_writes_report_and_returns_zero_for_matching_summaries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI writes the full report, prints a compact summary, and returns zero on pass."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    report_path = tmp_path / "reports" / "comparison.json"
    write_summary(output_a, summary())
    write_summary(output_b, summary())

    exit_code = main([str(output_a), str(output_b), "--report-path", str(report_path)])

    stdout = capsys.readouterr().out
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["passed"] is True
    assert report["summary_comparison"]["videos_in_both"] == 1
    assert stdout == (
        "PASSED split output comparison\n"
        "videos in both: 1, only in A: 0, only in B: 0, issues: 0\n"
        f"report: {report_path}\n"
    )


def test_cli_writes_report_and_returns_nonzero_for_failed_comparison(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI writes the full report, prints bounded issue details, and returns nonzero on failure."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    report_path = tmp_path / "comparison.json"
    write_summary(output_a, summary(total_num_clips_passed=2))
    write_summary(output_b, summary(total_num_clips_passed=1))

    exit_code = main([str(output_a), str(output_b), "--report-path", str(report_path)])

    stdout = capsys.readouterr().out
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["passed"] is False
    assert report["issues"][0]["code"] == "summary_field_mismatch"
    assert stdout == (
        "FAILED split output comparison\n"
        "videos in both: 1, only in A: 0, only in B: 0, issues: 1\n"
        "first issues:\n"
        "- summary_field_mismatch: Summary field differs between output A and output B "
        "(field=total_num_clips_passed)\n"
        f"report: {report_path}\n"
    )


def test_cli_load_failure_names_output_path_error_and_field(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI and report output make summary schema load failures easy to locate."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    report_path = tmp_path / "comparison.json"
    summary_a = summary()
    del summary_a["num_processed_videos"]
    write_summary(output_a, summary_a)
    write_summary(output_b, summary())

    exit_code = main([str(output_a), str(output_b), "--report-path", str(report_path)])

    stdout = capsys.readouterr().out
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_error = "summary.json missing required field 'num_processed_videos'"
    expected_message = f"Failed to load output A summary at {output_a / 'summary.json'}: {expected_error}"
    assert exit_code == 1
    assert report["issues"][0]["message"] == expected_message
    assert report["issues"][0]["output"] == "a"
    assert report["issues"][0]["field"] == "num_processed_videos"
    assert stdout == (
        "FAILED split output comparison\n"
        "videos in both: 0, only in A: 0, only in B: 0, issues: 1\n"
        "first issues:\n"
        f"- summary_load_failed: {expected_message} (field=num_processed_videos, "
        "error_type=MissingSummaryFieldError)\n"
        f"report: {report_path}\n"
    )


def test_format_stdout_summary_truncates_issues_and_formats_video_suffix() -> None:
    """Stdout summary bounds issue output and formats video-only suffixes."""
    issues = [
        Issue(code="video_only", message="Video-only issue", video="video.mp4"),
        *(Issue(code=f"issue_{index}", message="Issue") for index in range(5)),
    ]
    report = ComparisonReport.from_issues("output-a", "output-b", SummaryComparison(), issues)

    assert _format_stdout_summary(report, "report.json") == (
        "FAILED split output comparison\n"
        "videos in both: 0, only in A: 0, only in B: 0, issues: 6\n"
        "first issues:\n"
        "- video_only: Video-only issue (video=video.mp4)\n"
        "- issue_0: Issue\n"
        "- issue_1: Issue\n"
        "- issue_2: Issue\n"
        "- issue_3: Issue\n"
        "- 1 more issues omitted from stdout\n"
        "report: report.json\n"
    )


def test_format_stdout_summary_surfaces_error_type_from_issue_details() -> None:
    """Stdout summary distinguishes load-failure causes by surfacing error_type from details."""
    issues = [
        Issue.summary_load_failed("/path/a/summary.json", "a", "FileNotFoundError", "[Errno 2] No such file"),
        Issue.summary_load_failed("/path/b/summary.json", "b", "JSONDecodeError", "Expecting value: line 1 column 1"),
    ]
    report = ComparisonReport.from_issues("output-a", "output-b", SummaryComparison(), issues)

    stdout = _format_stdout_summary(report, "report.json")

    assert "- summary_load_failed: Failed to load output A summary at /path/a/summary.json: " in stdout
    assert "(error_type=FileNotFoundError)" in stdout
    assert "(error_type=JSONDecodeError)" in stdout
