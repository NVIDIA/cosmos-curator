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
"""Tests for the split-comparison CLI: `--config` + `--print-default-config`."""

import json
from pathlib import Path

import pyarrow as pa
import pytest

from cosmos_curator.pipelines.video.split_comparison import cli as cli_module
from cosmos_curator.pipelines.video.split_comparison.cli import main
from cosmos_curator.pipelines.video.split_comparison.config import SplitComparisonConfig
from cosmos_curator.pipelines.video.split_comparison.result_model import (
    ISSUE_SCHEMA,
    Report,
    empty_issues,
    make_issue,
)


def _write_config(path: Path, **overrides: object) -> Path:
    """Drop a config JSON file at ``path`` with placeholder targets + per-test overrides."""
    config = SplitComparisonConfig(output_a="/a", output_b="/b", **overrides)  # type: ignore[arg-type]
    path.write_text(config.model_dump_json(), encoding="utf-8")
    return path


def _stub_compare(monkeypatch: pytest.MonkeyPatch, *, report: Report) -> dict[str, object]:
    """Replace compare_split_outputs with a stub that records the config it was called with."""
    captured: dict[str, object] = {}

    def fake(*, config: object) -> Report:
        captured["config"] = config
        return report

    monkeypatch.setattr(cli_module, "compare_split_outputs", fake)
    return captured


def _clean_report() -> Report:
    return Report(
        issues=empty_issues(),
        passed=True,
        stages_run=frozenset({"summary", "metadata", "video_index"}),
    )


def _failing_report() -> Report:
    issues = pa.Table.from_pylist(
        [
            make_issue(
                code="aesthetic_score_mismatch",
                message="aesthetic differs",
                feature="aesthetic_score",
                video="video.mp4",
                clip="clip-a",
                details={"a": 0.5, "b": 0.6},
            ),
        ],
        schema=ISSUE_SCHEMA,
    )
    return Report(
        issues=issues,
        passed=False,
        stages_run=frozenset({"summary", "metadata", "video_index"}),
    )


# --- --print-default-config --------------------------------------------------------


def test_print_default_config_emits_valid_json(capsys: pytest.CaptureFixture[str]) -> None:
    """--print-default-config writes indented JSON to stdout and exits 0."""
    exit_code = main(["--print-default-config"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["output_a"] == "REPLACE_WITH_OUTPUT_A_PATH"
    assert payload["output_b"] == "REPLACE_WITH_OUTPUT_B_PATH"
    assert payload["compare_video_index"] is True
    assert payload["caption"]["model_id"] == "BAAI/bge-small-en-v1.5"


def test_print_default_config_output_round_trips_through_pydantic(capsys: pytest.CaptureFixture[str]) -> None:
    """Output of --print-default-config validates back to an SplitComparisonConfig."""
    main(["--print-default-config"])
    payload = capsys.readouterr().out

    reloaded = SplitComparisonConfig.model_validate_json(payload)
    assert reloaded.output_a == "REPLACE_WITH_OUTPUT_A_PATH"


# --- --config: happy path ----------------------------------------------------------


def test_cli_returns_zero_and_writes_json_report_on_passed_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A clean comparison: exit 0, JSON report on disk, PASSED message on stdout."""
    captured = _stub_compare(monkeypatch, report=_clean_report())
    config_path = _write_config(tmp_path / "config.json", report_path=str(tmp_path / "audit.json"))

    exit_code = main(["--config", str(config_path)])

    assert exit_code == 0
    assert captured["config"].output_a == "/a"  # type: ignore[union-attr]
    stdout = capsys.readouterr().out
    assert "PASSED split output comparison" in stdout
    assert "issues: 0" in stdout
    payload = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["summary"]["stages_run"] == ["metadata", "summary", "video_index"]


def test_cli_returns_one_and_emits_first_issues_on_failed_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A failing comparison: exit 1, FAILED + head of issues on stdout."""
    _stub_compare(monkeypatch, report=_failing_report())
    config_path = _write_config(tmp_path / "config.json", report_path=str(tmp_path / "audit.json"))

    exit_code = main(["--config", str(config_path)])

    assert exit_code == 1
    stdout = capsys.readouterr().out
    assert "FAILED split output comparison" in stdout
    assert "aesthetic_score_mismatch: aesthetic differs" in stdout
    assert "video=video.mp4" in stdout


def test_cli_writes_lance_report_when_config_selects_lance_format(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """report_format='lance' routes the single report to the Lance writer."""
    _stub_compare(monkeypatch, report=_clean_report())
    lance_target = str(tmp_path / "audit.lance")
    config_path = _write_config(
        tmp_path / "config.json",
        report_path=lance_target,
        report_format="lance",
    )

    exit_code = main(["--config", str(config_path)])

    assert exit_code == 0
    assert (tmp_path / "audit.lance").is_dir()
    stdout = capsys.readouterr().out
    assert f"report: {lance_target}" in stdout


def test_cli_propagates_overrides_from_config_into_compare_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Config overrides land on the SplitComparisonConfig passed to compare_split_outputs."""
    captured = _stub_compare(monkeypatch, report=_clean_report())
    config_path = _write_config(
        tmp_path / "config.json",
        report_path=str(tmp_path / "audit.json"),
        compare_video_index=False,
        compare_captions=False,
        clip_limit=25,
        video_key="video-a.mp4",
        metadata_workers=64,
        metadata_cpus_per_worker=0.25,
        metadata_batch_size=8,
        video_index_batch_size=4,
    )

    main(["--config", str(config_path)])

    config = captured["config"]
    assert config.compare_video_index is False  # type: ignore[union-attr]
    assert config.compare_captions is False  # type: ignore[union-attr]
    assert config.clip_limit == 25  # type: ignore[union-attr]
    assert config.video_key == "video-a.mp4"  # type: ignore[union-attr]
    assert config.metadata_workers == 64  # type: ignore[union-attr]
    assert config.metadata_cpus_per_worker == 0.25  # type: ignore[union-attr]
    assert config.metadata_batch_size == 8  # type: ignore[union-attr]
    assert config.video_index_batch_size == 4  # type: ignore[union-attr]


# --- --config: error paths ---------------------------------------------------------


def test_cli_returns_two_when_config_file_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing config file -> exit 2 with stderr explanation; compare_split_outputs never runs."""
    _stub_compare(monkeypatch, report=_clean_report())

    exit_code = main(["--config", str(tmp_path / "does-not-exist.json")])

    assert exit_code == 2
    assert "Failed to load config" in capsys.readouterr().err


def test_cli_returns_two_on_malformed_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Malformed JSON -> exit 2 with stderr explanation."""
    _stub_compare(monkeypatch, report=_clean_report())
    path = tmp_path / "config.json"
    path.write_text("{ not valid json", encoding="utf-8")

    exit_code = main(["--config", str(path)])

    assert exit_code == 2
    assert "Failed to load config" in capsys.readouterr().err


def test_cli_returns_two_on_validation_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A config with an invalid value (negative tolerance) is rejected at load."""
    _stub_compare(monkeypatch, report=_clean_report())
    path = tmp_path / "config.json"
    # Hand-write JSON since we can't construct an invalid SplitComparisonConfig at this layer.
    path.write_text(
        json.dumps(
            {
                "output_a": "/a",
                "output_b": "/b",
                "aesthetic": {"abs_tolerance": -1.0, "rel_tolerance": 1e-6},
            },
        ),
        encoding="utf-8",
    )

    exit_code = main(["--config", str(path)])

    assert exit_code == 2
    assert "Failed to load config" in capsys.readouterr().err


def test_cli_returns_two_on_unknown_field_in_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """extra='forbid' in the config model surfaces a typo'd field as a load error."""
    _stub_compare(monkeypatch, report=_clean_report())
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"output_a": "/a", "output_b": "/b", "metadta_workers": 8}),
        encoding="utf-8",
    )

    exit_code = main(["--config", str(path)])

    assert exit_code == 2
    assert "Failed to load config" in capsys.readouterr().err


# --- argparse top-level ------------------------------------------------------------


def test_cli_requires_one_of_the_two_flags() -> None:
    """No flags at all -> argparse exits non-zero (mutually exclusive group is required)."""
    with pytest.raises(SystemExit):
        main([])


def test_cli_rejects_both_flags_together(tmp_path: Path) -> None:
    """--config and --print-default-config are mutually exclusive."""
    config_path = _write_config(tmp_path / "config.json")
    with pytest.raises(SystemExit):
        main(["--config", str(config_path), "--print-default-config"])
