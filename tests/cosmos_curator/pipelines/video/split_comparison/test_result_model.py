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
"""Tests for the Issue / Report types and Arrow schema round-trips."""

import json

import pyarrow as pa

from cosmos_curator.pipelines.video.split_comparison.result_model import (
    ISSUE_SCHEMA,
    Report,
    empty_issues,
    make_issue,
)


def test_make_issue_produces_row_that_fits_issue_schema() -> None:
    """A fully-populated row from make_issue lands in the Arrow table cleanly."""
    row = make_issue(
        code="caption_similarity_below_threshold",
        message="similarity 0.71 below 0.85",
        feature="captions",
        video="video.mp4",
        clip="clip-a",
        output=None,
        field="caption",
        details={"start_frame": 0, "similarity": 0.71},
    )
    table = pa.Table.from_pylist([row], schema=ISSUE_SCHEMA)

    assert table.num_rows == 1
    assert table["code"][0].as_py() == "caption_similarity_below_threshold"
    assert table["field"][0].as_py() == "caption"
    parsed = json.loads(table["details"][0].as_py())
    assert parsed == {"start_frame": 0, "similarity": 0.71}


def test_make_issue_leaves_unset_fields_null_with_no_details() -> None:
    """Optional kwargs default to None; details stays null when no dict is provided."""
    row = make_issue(code="summary_field_mismatch", message="num_input_videos differs")
    table = pa.Table.from_pylist([row], schema=ISSUE_SCHEMA)

    assert table["clip"][0].as_py() is None
    assert table["video"][0].as_py() is None
    assert table["details"][0].as_py() is None


def test_make_issue_serializes_details_with_sorted_keys() -> None:
    """Details JSON is sorted-key so identical content produces identical bytes."""
    row_a = make_issue(code="aesthetic_score_mismatch", message="x", details={"a": 1, "b": 2})
    row_b = make_issue(code="aesthetic_score_mismatch", message="x", details={"b": 2, "a": 1})

    assert row_a["details"] == row_b["details"]


def test_empty_issues_carries_the_schema() -> None:
    """A zero-row issue table still advertises ISSUE_SCHEMA so concat_tables stays happy."""
    table = empty_issues()

    assert table.num_rows == 0
    assert table.schema == ISSUE_SCHEMA


def test_concat_issue_tables_preserves_schema() -> None:
    """Concatenating empty + populated tables produces a uniform issue table."""
    populated = pa.Table.from_pylist(
        [make_issue(code="summary_field_mismatch", message="x")],
        schema=ISSUE_SCHEMA,
    )
    combined = pa.concat_tables([empty_issues(), populated, empty_issues()])

    assert combined.num_rows == 1
    assert combined.schema == ISSUE_SCHEMA


def test_report_passed_flag_is_independent_of_issue_count() -> None:
    """Report carries its own passed flag rather than recomputing; callers control truth."""
    issues = pa.Table.from_pylist(
        [make_issue(code="summary_field_mismatch", message="x")],
        schema=ISSUE_SCHEMA,
    )
    report = Report(issues=issues, passed=False, stages_run=frozenset({"summary"}))

    assert not report.passed
    assert report.stages_run == frozenset({"summary"})


def test_report_with_empty_issues_can_be_marked_passed() -> None:
    """A clean run -- empty issues, all enabled stages ran -- reads as passed=True."""
    report = Report(
        issues=empty_issues(),
        passed=True,
        stages_run=frozenset({"summary", "metadata", "video_index"}),
    )

    assert report.passed
    assert report.issues.num_rows == 0
