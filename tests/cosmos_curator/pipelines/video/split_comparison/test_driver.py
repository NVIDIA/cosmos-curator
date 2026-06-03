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
"""Tests for the compare_split_outputs driver: stage fan-in into Report."""

from collections.abc import Mapping

import pyarrow as pa
import pytest

from cosmos_curator.pipelines.video.split_comparison import driver as driver_module
from cosmos_curator.pipelines.video.split_comparison.config import SplitComparisonConfig
from cosmos_curator.pipelines.video.split_comparison.driver import compare_split_outputs
from cosmos_curator.pipelines.video.split_comparison.result_model import ISSUE_SCHEMA, empty_issues, make_issue
from tests.cosmos_curator.pipelines.video.split_comparison.test_clip_discovery import _processed_video, _summary


def _config(**overrides: object) -> SplitComparisonConfig:
    """Build a config with placeholder targets so tests only spell out the override under test."""
    return SplitComparisonConfig(output_a="/a", output_b="/b", **overrides)  # type: ignore[arg-type]


def _stub_load_summary(monkeypatch: pytest.MonkeyPatch, by_root: Mapping[str, object]) -> None:
    monkeypatch.setattr(
        driver_module,
        "load_summary",
        lambda output_root, **_kwargs: by_root[str(output_root)],
    )


def _stub_stages(
    monkeypatch: pytest.MonkeyPatch,
    *,
    metadata_issues: pa.Table | None = None,
    video_index_issues: pa.Table | None = None,
) -> None:
    monkeypatch.setattr(
        driver_module,
        "run_metadata_stage",
        lambda _clips, **_kwargs: metadata_issues if metadata_issues is not None else empty_issues(),
    )
    monkeypatch.setattr(
        driver_module,
        "run_video_index_stage",
        lambda _clips, **_kwargs: video_index_issues if video_index_issues is not None else empty_issues(),
    )


def test_driver_returns_passed_report_when_summaries_and_stages_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """All-clean run: summary matches, no clip issues, passed=True."""
    summary = _summary({"video.mp4": _processed_video("video.mp4", clips=("c1",))})
    _stub_load_summary(monkeypatch, {"/a": summary, "/b": summary})
    _stub_stages(monkeypatch)

    report = compare_split_outputs(config=_config())

    assert report.passed
    assert report.issues.num_rows == 0
    assert report.stages_run == frozenset({"summary", "metadata", "video_index"})


def test_driver_populates_videos_table_and_derives_source_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    """Videos table is keyed on video_key with per-side presence; source roots derive from uniform layout."""
    summary_a = _summary(
        {
            "shared.mp4": _processed_video("shared.mp4", clips=("c1",)),
            "a-only.mp4": _processed_video("a-only.mp4", clips=("c2",)),
        },
    )
    summary_b = _summary(
        {
            "shared.mp4": _processed_video("shared.mp4", clips=("c1",)),
            "b-only.mp4": _processed_video("b-only.mp4", clips=("c3",)),
        },
    )
    _stub_load_summary(monkeypatch, {"/a": summary_a, "/b": summary_b})
    _stub_stages(monkeypatch)

    report = compare_split_outputs(config=_config())

    # _processed_video sets source_video=f"/inputs/{key}", so the derived root is "/inputs/" for both sides.
    assert report.source_a == "/inputs/"
    assert report.source_b == "/inputs/"
    assert report.videos.to_pylist() == [
        {"video_key": "a-only.mp4", "in_a": True, "in_b": False},
        {"video_key": "b-only.mp4", "in_a": False, "in_b": True},
        {"video_key": "shared.mp4", "in_a": True, "in_b": True},
    ]


def test_driver_emits_layout_inconsistent_issue_when_source_root_is_not_uniform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-uniform source layout on a side surfaces a structured issue and leaves source_X empty."""
    weird_video = _processed_video("weird.mp4", clips=("c2",)).model_copy(
        update={"source_video": "s3://other-bucket/extras/weird.mp4"},
    )
    summary_a = _summary(
        {
            "v1.mp4": _processed_video("v1.mp4", clips=("c1",)),
            "weird.mp4": weird_video,
        },
    )
    summary_b = _summary({"v1.mp4": _processed_video("v1.mp4", clips=("c1",))})
    _stub_load_summary(monkeypatch, {"/a": summary_a, "/b": summary_b})
    _stub_stages(monkeypatch)

    report = compare_split_outputs(config=_config())

    layout_codes = [code for code in report.issues["code"].to_pylist() if code == "summary_source_layout_inconsistent"]
    assert layout_codes == ["summary_source_layout_inconsistent"]
    assert report.source_a == ""
    assert report.source_b == "/inputs/"
    # Videos table still lists every key with presence booleans even when one side's root failed.
    assert report.videos["video_key"].to_pylist() == ["v1.mp4", "weird.mp4"]


def test_driver_emits_empty_videos_table_when_either_summary_fails_to_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load-failure short-circuit returns the structured failure issue + an empty videos table."""

    def fake_load(output_root: str, **_kwargs: object) -> object:
        if output_root == "/a":
            msg = "summary.json missing"
            raise FileNotFoundError(msg)
        return _summary({"video.mp4": _processed_video("video.mp4", clips=("c1",))})

    monkeypatch.setattr(driver_module, "load_summary", fake_load)
    _stub_stages(monkeypatch)

    report = compare_split_outputs(config=_config())

    assert not report.passed
    assert report.issues["code"].to_pylist() == ["summary_load_failed"]
    assert report.videos.num_rows == 0
    assert report.videos.schema.names == ["video_key", "in_a", "in_b"]
    assert report.source_a == ""
    assert report.source_b == ""


def test_driver_reports_summary_issues_without_running_clip_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no shared processed videos exist, stages aren't called and stages_run shows it."""
    summary_a = _summary({"a-only.mp4": _processed_video("a-only.mp4")})
    summary_b = _summary({"b-only.mp4": _processed_video("b-only.mp4")})
    _stub_load_summary(monkeypatch, {"/a": summary_a, "/b": summary_b})
    metadata_called = {"hit": False}
    video_index_called = {"hit": False}

    def fake_metadata(_clips: pa.Table, **_kwargs: object) -> pa.Table:
        metadata_called["hit"] = True
        return empty_issues()

    def fake_video_index(_clips: pa.Table, **_kwargs: object) -> pa.Table:
        video_index_called["hit"] = True
        return empty_issues()

    monkeypatch.setattr(driver_module, "run_metadata_stage", fake_metadata)
    monkeypatch.setattr(driver_module, "run_video_index_stage", fake_video_index)

    report = compare_split_outputs(config=_config())

    assert not report.passed  # summary_video_only_in_a / only_in_b issues
    assert report.stages_run == frozenset({"summary"})
    assert not metadata_called["hit"]
    assert not video_index_called["hit"]


def test_driver_skips_video_index_when_flag_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """compare_video_index=False keeps Stage 2 out of stages_run and out of the issue stream."""
    summary = _summary({"video.mp4": _processed_video("video.mp4", clips=("c1",))})
    _stub_load_summary(monkeypatch, {"/a": summary, "/b": summary})
    video_index_called = {"hit": False}

    def fake_video_index(_clips: pa.Table, **_kwargs: object) -> pa.Table:
        video_index_called["hit"] = True
        return pa.Table.from_pylist(
            [make_issue(code="clip_mp4_missing", message="x")],
            schema=ISSUE_SCHEMA,
        )

    monkeypatch.setattr(driver_module, "run_video_index_stage", fake_video_index)
    monkeypatch.setattr(
        driver_module,
        "run_metadata_stage",
        lambda _clips, **_kwargs: empty_issues(),
    )

    report = compare_split_outputs(config=_config(compare_video_index=False))

    assert report.passed
    assert report.stages_run == frozenset({"summary", "metadata"})
    assert not video_index_called["hit"]


def test_driver_concatenates_issues_from_all_three_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Summary + metadata + video_index issue tables all end up in Report.issues."""
    summary_a = _summary({"video.mp4": _processed_video("video.mp4", clips=("c1",))}, num_processed_videos=2)
    summary_b = _summary({"video.mp4": _processed_video("video.mp4", clips=("c1",))}, num_processed_videos=3)
    _stub_load_summary(monkeypatch, {"/a": summary_a, "/b": summary_b})

    metadata_issues = pa.Table.from_pylist(
        [make_issue(code="aesthetic_score_mismatch", message="x", clip="c1", video="video.mp4")],
        schema=ISSUE_SCHEMA,
    )
    video_index_issues = pa.Table.from_pylist(
        [make_issue(code="clip_mp4_index_mismatch", message="y", clip="c1", video="video.mp4")],
        schema=ISSUE_SCHEMA,
    )
    _stub_stages(monkeypatch, metadata_issues=metadata_issues, video_index_issues=video_index_issues)

    report = compare_split_outputs(config=_config())

    codes = sorted(report.issues["code"].to_pylist())
    assert codes == ["aesthetic_score_mismatch", "clip_mp4_index_mismatch", "summary_field_mismatch"]
    assert not report.passed
    assert report.stages_run == frozenset({"summary", "metadata", "video_index"})


def test_driver_applies_clip_limit_to_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    """clip_limit caps the table handed to Stage 1/2 to the first N rows."""
    summary = _summary({"video.mp4": _processed_video("video.mp4", clips=("c1", "c2", "c3", "c4", "c5"))})
    _stub_load_summary(monkeypatch, {"/a": summary, "/b": summary})
    received: dict[str, pa.Table] = {}

    def fake_metadata(clips: pa.Table, **_kwargs: object) -> pa.Table:
        received["metadata"] = clips
        return empty_issues()

    def fake_video_index(clips: pa.Table, **_kwargs: object) -> pa.Table:
        received["video_index"] = clips
        return empty_issues()

    monkeypatch.setattr(driver_module, "run_metadata_stage", fake_metadata)
    monkeypatch.setattr(driver_module, "run_video_index_stage", fake_video_index)

    report = compare_split_outputs(config=_config(clip_limit=2))

    assert received["metadata"].num_rows == 2
    assert received["video_index"].num_rows == 2
    assert report.passed
    assert report.stages_run == frozenset({"summary", "metadata", "video_index"})


def test_driver_clip_limit_above_total_keeps_all_clips(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clip_limit larger than the discovered clip count leaves the table untouched."""
    summary = _summary({"video.mp4": _processed_video("video.mp4", clips=("c1", "c2"))})
    _stub_load_summary(monkeypatch, {"/a": summary, "/b": summary})
    received: dict[str, pa.Table] = {}

    def fake_metadata(clips: pa.Table, **_kwargs: object) -> pa.Table:
        received["metadata"] = clips
        return empty_issues()

    monkeypatch.setattr(driver_module, "run_metadata_stage", fake_metadata)
    monkeypatch.setattr(driver_module, "run_video_index_stage", lambda _clips, **_kwargs: empty_issues())

    compare_split_outputs(config=_config(clip_limit=999))

    assert received["metadata"].num_rows == 2


def test_driver_filters_clips_to_selected_video_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """config.video_key restricts the clip table to that video before stage execution."""
    summary = _summary(
        {
            "video-a.mp4": _processed_video("video-a.mp4", clips=("c1", "c2")),
            "video-b.mp4": _processed_video("video-b.mp4", clips=("c3", "c4")),
        },
    )
    _stub_load_summary(monkeypatch, {"/a": summary, "/b": summary})
    received: dict[str, pa.Table] = {}

    def fake_metadata(clips: pa.Table, **_kwargs: object) -> pa.Table:
        received["metadata"] = clips
        return empty_issues()

    monkeypatch.setattr(driver_module, "run_metadata_stage", fake_metadata)
    monkeypatch.setattr(driver_module, "run_video_index_stage", lambda _clips, **_kwargs: empty_issues())

    compare_split_outputs(config=_config(video_key="video-a.mp4"))

    metadata_clips = received["metadata"]
    assert metadata_clips.num_rows == 2
    assert set(metadata_clips["video_key"].to_pylist()) == {"video-a.mp4"}


def test_driver_video_key_with_no_match_skips_clip_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    """A video_key that matches nothing in the discovered table zeros out the clip stages."""
    summary = _summary({"video-a.mp4": _processed_video("video-a.mp4", clips=("c1",))})
    _stub_load_summary(monkeypatch, {"/a": summary, "/b": summary})
    metadata_called = {"hit": False}

    def fake_metadata(_clips: pa.Table, **_kwargs: object) -> pa.Table:
        metadata_called["hit"] = True
        return empty_issues()

    monkeypatch.setattr(driver_module, "run_metadata_stage", fake_metadata)
    monkeypatch.setattr(driver_module, "run_video_index_stage", lambda _clips, **_kwargs: empty_issues())

    report = compare_split_outputs(config=_config(video_key="missing.mp4"))

    assert not metadata_called["hit"]
    assert report.stages_run == frozenset({"summary"})


def test_driver_video_key_and_clip_limit_compose(monkeypatch: pytest.MonkeyPatch) -> None:
    """video_key filters first, then clip_limit head-slices the filtered set."""
    summary = _summary(
        {
            "video-a.mp4": _processed_video("video-a.mp4", clips=("c1", "c2", "c3")),
            "video-b.mp4": _processed_video("video-b.mp4", clips=("c4", "c5", "c6")),
        },
    )
    _stub_load_summary(monkeypatch, {"/a": summary, "/b": summary})
    received: dict[str, pa.Table] = {}

    def fake_metadata(clips: pa.Table, **_kwargs: object) -> pa.Table:
        received["metadata"] = clips
        return empty_issues()

    monkeypatch.setattr(driver_module, "run_metadata_stage", fake_metadata)
    monkeypatch.setattr(driver_module, "run_video_index_stage", lambda _clips, **_kwargs: empty_issues())

    compare_split_outputs(config=_config(video_key="video-a.mp4", clip_limit=2))

    metadata_clips = received["metadata"]
    assert metadata_clips.num_rows == 2
    assert set(metadata_clips["video_key"].to_pylist()) == {"video-a.mp4"}


def test_driver_records_per_side_clip_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report carries clips_in_a / clips_in_b / clips_in_both derived from the clip table."""
    summary_a = _summary({"v.mp4": _processed_video("v.mp4", clips=("c1", "c2", "c3"))})
    summary_b = _summary({"v.mp4": _processed_video("v.mp4", clips=("c2", "c3", "c4"))})
    _stub_load_summary(monkeypatch, {"/a": summary_a, "/b": summary_b})
    _stub_stages(monkeypatch)

    report = compare_split_outputs(config=_config())

    assert report.clip_count == 4  # union: c1, c2, c3, c4
    assert report.clips_in_a == 3  # c1, c2, c3
    assert report.clips_in_b == 3  # c2, c3, c4
    assert report.clips_in_both == 2  # c2, c3


def test_driver_per_side_counts_zero_when_no_clips_discovered(monkeypatch: pytest.MonkeyPatch) -> None:
    """No shared processed videos -> empty clip table -> all per-output counts are zero."""
    summary_a = _summary({"a-only.mp4": _processed_video("a-only.mp4")})
    summary_b = _summary({"b-only.mp4": _processed_video("b-only.mp4")})
    _stub_load_summary(monkeypatch, {"/a": summary_a, "/b": summary_b})
    _stub_stages(monkeypatch)

    report = compare_split_outputs(config=_config())

    assert report.clip_count == 0
    assert report.clips_in_a == 0
    assert report.clips_in_b == 0
    assert report.clips_in_both == 0


def test_driver_emits_summary_load_failed_and_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed summary load yields a failing Report with a summary_load_failed issue, not a crash."""
    summary = _summary({"video.mp4": _processed_video("video.mp4", clips=("c1",))})

    def fake_load(output_root: object, **_kwargs: object) -> object:
        if str(output_root) == "/b":
            msg = "no such key: /b/summary.json"
            raise FileNotFoundError(msg)
        return summary

    monkeypatch.setattr(driver_module, "load_summary", fake_load)
    stage_called = {"hit": False}

    def fake_stage(_clips: pa.Table, **_kwargs: object) -> pa.Table:
        stage_called["hit"] = True
        return empty_issues()

    monkeypatch.setattr(driver_module, "run_metadata_stage", fake_stage)
    monkeypatch.setattr(driver_module, "run_video_index_stage", fake_stage)

    report = compare_split_outputs(config=_config())

    assert not report.passed
    assert report.stages_run == frozenset({"summary"})
    assert not stage_called["hit"]  # clip discovery never reached
    assert report.clip_count == 0
    codes = report.issues["code"].to_pylist()
    assert codes == ["summary_load_failed"]
    failed = report.issues.to_pylist()[0]
    assert failed["output"] == "b"
    assert "summary.json" in failed["message"]


def test_driver_emits_summary_load_failed_for_both_sides(monkeypatch: pytest.MonkeyPatch) -> None:
    """When both summaries fail to load, both load failures are reported."""

    def fake_load(output_root: object, **_kwargs: object) -> object:
        msg = f"bad summary at {output_root}"
        raise ValueError(msg)

    monkeypatch.setattr(driver_module, "load_summary", fake_load)
    _stub_stages(monkeypatch)

    report = compare_split_outputs(config=_config())

    assert not report.passed
    assert report.issues["code"].to_pylist() == ["summary_load_failed", "summary_load_failed"]
    assert sorted(row["output"] for row in report.issues.to_pylist()) == ["a", "b"]


def test_driver_handles_empty_summaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty summaries (zero videos) yield a clean pass; clip stages don't run."""
    empty = _summary({})
    _stub_load_summary(monkeypatch, {"/a": empty, "/b": empty})
    _stub_stages(monkeypatch)

    report = compare_split_outputs(config=_config())

    assert report.passed
    assert report.issues.num_rows == 0
    assert report.stages_run == frozenset({"summary"})
