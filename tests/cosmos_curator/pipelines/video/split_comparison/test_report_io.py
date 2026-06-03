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
"""Tests for report_io: extension-driven dispatch, JSON via smart_open, Lance native."""

import json
from pathlib import Path

import lance
import pyarrow as pa
import pytest

from cosmos_curator.pipelines.video.split_comparison.config import (
    CaptionPolicy,
    ScoreTolerance,
    SplitComparisonConfig,
)
from cosmos_curator.pipelines.video.split_comparison.report_io import build_summary, write_report
from cosmos_curator.pipelines.video.split_comparison.result_model import (
    ISSUE_SCHEMA,
    VIDEOS_SCHEMA,
    Report,
    empty_issues,
    make_issue,
)


def _sample_report() -> Report:
    issues = pa.Table.from_pylist(
        [
            make_issue(
                code="aesthetic_score_mismatch",
                message="Aesthetic score differs",
                feature="aesthetic_score",
                video="video.mp4",
                clip="clip-a",
                details={"a": 0.5, "b": 0.6},
            ),
            make_issue(
                code="summary_field_mismatch",
                message="num_input_videos differs",
                feature=None,
            ),
        ],
        schema=ISSUE_SCHEMA,
    )
    videos = pa.Table.from_pylist(
        [
            {"video_key": "video.mp4", "in_a": True, "in_b": True},
            {"video_key": "a-only.mp4", "in_a": True, "in_b": False},
        ],
        schema=VIDEOS_SCHEMA,
    )
    return Report(
        issues=issues,
        videos=videos,
        passed=False,
        stages_run=frozenset({"summary", "metadata", "video_index"}),
        clip_count=42,
        output_a="s3://bucket/run-a/",
        output_b="s3://bucket/run-b/",
        source_a="s3://bucket/inputs/",
        source_b="s3://bucket/inputs/",
        config=SplitComparisonConfig(
            output_a="s3://bucket/run-a/",
            output_b="s3://bucket/run-b/",
            profile_name="default",
            aesthetic=ScoreTolerance(abs_tolerance=0.01, rel_tolerance=0.02),
            caption=CaptionPolicy(min_similarity=0.85),
            clip_limit=100,
            video_key="video.mp4",
        ),
    )


# --- JSON writer (via smart_open) ---------------------------------------------------


def test_write_report_json_round_trips_details_as_nested_dict(tmp_path: Path) -> None:
    """JSON writer parses details back to a nested object for human readability."""
    report = _sample_report()
    target = str(tmp_path / "audit.json")

    result = write_report(report, target, report_format="json")

    assert result == target
    payload = json.loads(Path(target).read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["output_a"] == "s3://bucket/run-a/"
    assert payload["output_b"] == "s3://bucket/run-b/"
    assert payload["summary"]["stages_run"] == ["metadata", "summary", "video_index"]
    assert payload["config"]["profile_name"] == "default"
    assert payload["config"]["aesthetic"] == {"abs_tolerance": 0.01, "rel_tolerance": 0.02}
    assert payload["config"]["video_key"] == "video.mp4"
    assert len(payload["issues"]) == 2
    aesthetic = next(issue for issue in payload["issues"] if issue["code"] == "aesthetic_score_mismatch")
    assert aesthetic["details"] == {"a": 0.5, "b": 0.6}


def test_write_report_json_includes_videos_table_and_source_roots(tmp_path: Path) -> None:
    """JSON payload carries source_a / source_b plus a per-video presence table."""
    target = str(tmp_path / "audit.json")

    write_report(_sample_report(), target, report_format="json")

    payload = json.loads(Path(target).read_text(encoding="utf-8"))
    assert payload["source_a"] == "s3://bucket/inputs/"
    assert payload["source_b"] == "s3://bucket/inputs/"
    assert payload["videos"] == [
        {"video_key": "video.mp4", "in_a": True, "in_b": True},
        {"video_key": "a-only.mp4", "in_a": True, "in_b": False},
    ]


def test_write_report_json_drops_null_fields_for_readability(tmp_path: Path) -> None:
    """Optional fields that are None on a row don't clutter the JSON output."""
    report = _sample_report()
    target = str(tmp_path / "audit.json")

    write_report(report, target, report_format="json")

    payload = json.loads(Path(target).read_text(encoding="utf-8"))
    summary_issue = next(issue for issue in payload["issues"] if issue["code"] == "summary_field_mismatch")
    for absent_key in ("clip", "video", "output", "field", "details", "feature"):
        assert absent_key not in summary_issue


def test_write_report_json_omits_config_when_report_has_no_config(tmp_path: Path) -> None:
    """A Report constructed without config (default None) writes no "config" key in JSON."""
    report = Report(
        issues=empty_issues(),
        passed=True,
        stages_run=frozenset({"summary"}),
    )
    target = str(tmp_path / "audit.json")

    write_report(report, target, report_format="json")

    payload = json.loads(Path(target).read_text(encoding="utf-8"))
    assert "config" not in payload


def test_write_report_json_routes_through_smart_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """JSON writer calls smart_open.open with transport_params from the project's storage profile.

    Pins the contract: cloud URLs (s3://, az://, ...) flow through the same plumbing
    the stage actors use; the writer is path-scheme agnostic.
    """
    captured: dict[str, object] = {}
    sentinel_transport = {"client": "sentinel-s3-client"}

    # Inject non-empty transport params so the flow-through is observable -- a local
    # profile resolves to {}, in which case smart_open.open gets no transport_params
    # kwarg at all.
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.split_comparison.report_io.storage_utils.get_smart_open_params",
        lambda _path, **_kwargs: {"transport_params": sentinel_transport},
    )

    def fake_open(path: str, mode: str, *, encoding: str, transport_params: object) -> object:
        captured["path"] = path
        captured["mode"] = mode
        captured["encoding"] = encoding
        captured["transport_params"] = transport_params
        # Test paths are always local tmp_path; delegate to stdlib open() to satisfy
        # the writer's context-manager usage without recursing through the patched module.
        return Path(path).open(mode, encoding=encoding)

    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.split_comparison.report_io.smart_open.open",
        fake_open,
    )

    target = str(tmp_path / "audit.json")
    write_report(_sample_report(), target, report_format="json")

    assert captured["path"] == target
    assert captured["mode"] == "w"
    assert captured["encoding"] == "utf-8"
    assert captured["transport_params"] == sentinel_transport


# --- Lance writer (native multi-backend) --------------------------------------------


def test_write_report_lance_persists_issue_table_and_sidecar(tmp_path: Path) -> None:
    """Lance writer produces a dataset of issues plus a one-row metadata sidecar."""
    report = _sample_report()
    target = str(tmp_path / "audit.lance")

    result = write_report(report, target, report_format="lance")

    assert result == target
    issues_back = lance.dataset(target).to_table()
    assert issues_back.num_rows == 2
    codes = sorted(issues_back["code"].to_pylist())
    assert codes == ["aesthetic_score_mismatch", "summary_field_mismatch"]

    meta_back = lance.dataset(str(tmp_path / "audit.meta.lance")).to_table().to_pylist()
    assert len(meta_back) == 1
    assert meta_back[0]["passed"] is False
    assert meta_back[0]["output_a"] == "s3://bucket/run-a/"
    assert meta_back[0]["output_b"] == "s3://bucket/run-b/"
    sidecar_summary = json.loads(meta_back[0]["summary"])
    assert sidecar_summary["stages_run"] == ["metadata", "summary", "video_index"]
    assert sidecar_summary["total_clips_compared"] == 42
    assert sidecar_summary["total_issues"] == 2
    sidecar_config = json.loads(meta_back[0]["config"])
    assert sidecar_config["profile_name"] == "default"
    assert sidecar_config["aesthetic"] == {"abs_tolerance": 0.01, "rel_tolerance": 0.02}
    assert sidecar_config["video_key"] == "video.mp4"
    assert sidecar_config["clip_limit"] == 100


def test_write_report_lance_persists_videos_sidecar_and_source_roots(tmp_path: Path) -> None:
    """Lance writer materializes the videos sidecar dataset and embeds source roots in the meta sidecar."""
    target = str(tmp_path / "audit.lance")

    write_report(_sample_report(), target, report_format="lance")

    meta_back = lance.dataset(str(tmp_path / "audit.meta.lance")).to_table().to_pylist()
    assert meta_back[0]["source_a"] == "s3://bucket/inputs/"
    assert meta_back[0]["source_b"] == "s3://bucket/inputs/"
    videos_back = lance.dataset(str(tmp_path / "audit.videos.lance")).to_table().to_pylist()
    assert videos_back == [
        {"video_key": "video.mp4", "in_a": True, "in_b": True},
        {"video_key": "a-only.mp4", "in_a": True, "in_b": False},
    ]


def test_write_report_lance_writes_null_config_when_report_has_no_config(tmp_path: Path) -> None:
    """Lance sidecar's config column is null when the report has no config attached."""
    report = Report(
        issues=empty_issues(),
        passed=True,
        stages_run=frozenset({"summary"}),
    )
    target = str(tmp_path / "audit.lance")

    write_report(report, target, report_format="lance")

    meta_back = lance.dataset(str(tmp_path / "audit.meta.lance")).to_table().to_pylist()
    assert meta_back[0]["config"] is None


# --- Empty / parent-dir / overwrite (both writers) ----------------------------------


def test_write_report_handles_passed_empty_report(tmp_path: Path) -> None:
    """A passed run with no issues writes cleanly in either format."""
    report = Report(
        issues=empty_issues(),
        passed=True,
        stages_run=frozenset({"summary", "metadata"}),
        clip_count=10,
    )
    json_path = str(tmp_path / "clean.json")
    lance_path = str(tmp_path / "clean.lance")

    write_report(report, json_path, report_format="json")
    write_report(report, lance_path, report_format="lance")

    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["issues"] == []
    issues_back = lance.dataset(lance_path).to_table()
    assert issues_back.num_rows == 0


def test_write_report_creates_parent_directories(tmp_path: Path) -> None:
    """Writer materializes missing parent directories for local paths, for both formats."""
    report = _sample_report()
    target_dir = tmp_path / "deep" / "nested" / "reports"

    write_report(report, str(target_dir / "audit.json"), report_format="json")
    write_report(report, str(target_dir / "audit.lance"), report_format="lance")

    assert (target_dir / "audit.json").is_file()
    assert (target_dir / "audit.lance").is_dir()


def test_write_report_overwrites_existing_outputs(tmp_path: Path) -> None:
    """Re-running with the same path replaces previous output, not appends -- both formats."""
    first = Report(
        issues=pa.Table.from_pylist(
            [make_issue(code="summary_field_mismatch", message="first")],
            schema=ISSUE_SCHEMA,
        ),
        passed=False,
        stages_run=frozenset({"summary"}),
    )
    second = Report(
        issues=pa.Table.from_pylist(
            [make_issue(code="summary_field_mismatch", message="second")],
            schema=ISSUE_SCHEMA,
        ),
        passed=False,
        stages_run=frozenset({"summary"}),
    )
    json_path = str(tmp_path / "audit.json")
    lance_path = str(tmp_path / "audit.lance")

    write_report(first, json_path, report_format="json")
    write_report(second, json_path, report_format="json")
    write_report(first, lance_path, report_format="lance")
    write_report(second, lance_path, report_format="lance")

    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert payload["issues"][0]["message"] == "second"
    issues_back = lance.dataset(lance_path).to_table().to_pylist()
    assert len(issues_back) == 1
    assert issues_back[0]["message"] == "second"


# --- build_summary ------------------------------------------------------------------


def test_build_summary_surfaces_runtime_sec() -> None:
    """``runtime_sec`` rides through from Report straight into the summary block."""
    report = Report(
        issues=empty_issues(),
        passed=True,
        stages_run=frozenset({"summary"}),
        runtime_sec=12.5,
    )

    summary = build_summary(report)

    assert summary["runtime_sec"] == 12.5


def test_build_summary_includes_per_output_clip_breakdown() -> None:
    """clips_in_a / clips_in_b / clips_in_both come straight off the Report fields."""
    report = Report(
        issues=empty_issues(),
        passed=True,
        stages_run=frozenset({"summary", "metadata"}),
        clip_count=5100,
        clips_in_a=5000,
        clips_in_b=5050,
        clips_in_both=4950,
    )

    summary = build_summary(report)

    assert summary["total_clips_compared"] == 5100
    assert summary["clips_in_a"] == 5000
    assert summary["clips_in_b"] == 5050
    assert summary["clips_in_both"] == 4950


def test_build_summary_buckets_issues_by_feature_and_code() -> None:
    """build_summary aggregates the issue table into the published summary block."""
    issues = pa.Table.from_pylist(
        [
            make_issue(
                code="aesthetic_score_mismatch",
                message="aesthetic differs",
                feature="aesthetic_score",
                video="video-a.mp4",
                clip="clip-1",
            ),
            make_issue(
                code="motion_score_mismatch",
                message="motion differs",
                feature="motion_score",
                video="video-a.mp4",
                clip="clip-1",
            ),
            make_issue(
                code="motion_score_mismatch",
                message="motion differs",
                feature="motion_score",
                video="video-a.mp4",
                clip="clip-2",
            ),
            make_issue(
                code="clip_mp4_index_mismatch",
                message="vix differs",
                feature="video_indexes",
                video="video-b.mp4",
                clip="clip-3",
            ),
            make_issue(code="summary_field_mismatch", message="totals differ", feature="summary"),
        ],
        schema=ISSUE_SCHEMA,
    )
    report = Report(
        issues=issues,
        passed=False,
        stages_run=frozenset({"summary", "metadata", "video_index"}),
        clip_count=100,
    )

    summary = build_summary(report)

    assert summary["stages_run"] == ["metadata", "summary", "video_index"]
    assert summary["total_clips_compared"] == 100
    assert summary["total_issues"] == 5
    assert summary["clips_with_issues"] == 3
    assert summary["issues_by_feature"] == {
        "aesthetic_score": 1,
        "motion_score": 2,
        "video_indexes": 1,
        "summary": 1,
    }
    assert summary["clips_with_issues_by_feature"] == {
        "aesthetic_score": 1,
        "motion_score": 2,
        "video_indexes": 1,
    }
    assert summary["issues_by_code"] == {
        "aesthetic_score_mismatch": 1,
        "motion_score_mismatch": 2,
        "clip_mp4_index_mismatch": 1,
        "summary_field_mismatch": 1,
    }


def test_build_summary_dedupes_clip_with_repeat_issues() -> None:
    """A clip that fails the same feature twice still counts once in clips_with_issues_by_feature."""
    issues = pa.Table.from_pylist(
        [
            make_issue(
                code="clip_mp4_metadata_mismatch",
                message="m1",
                feature="video_indexes",
                video="video.mp4",
                clip="clip-x",
                field="codec_name",
            ),
            make_issue(
                code="clip_mp4_metadata_mismatch",
                message="m2",
                feature="video_indexes",
                video="video.mp4",
                clip="clip-x",
                field="height",
            ),
        ],
        schema=ISSUE_SCHEMA,
    )
    report = Report(issues=issues, passed=False, stages_run=frozenset({"video_index"}), clip_count=1)

    summary = build_summary(report)

    assert summary["total_issues"] == 2
    assert summary["clips_with_issues"] == 1
    assert summary["clips_with_issues_by_feature"] == {"video_indexes": 1}
    assert summary["issues_by_feature"] == {"video_indexes": 2}
