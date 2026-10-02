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

"""Tests for the run-only pipeline runtime entrypoint."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

from cosmos_curator.next.recipes.video_transcode import pipeline as transcoding_pipeline
from cosmos_curator.next.recipes.video_transcode.config import ResolvedVideoTranscodeConfig
from cosmos_curator.pipelines import pipeline_runtime
from cosmos_curator.pipelines.ray_data.caption_judge import driver as caption_judge_driver
from cosmos_curator.pipelines.ray_data.caption_judge import report_io as caption_judge_report_io
from cosmos_curator.pipelines.ray_data.caption_judge.config import CaptionJudgePipelineConfig
from cosmos_curator.pipelines.ray_data.video_split import pipeline as splitting_pipeline
from cosmos_curator.pipelines.ray_data.video_split.config import ResolvedVideoSplitConfig


def _write_config(path: Path, *, extra: str = "") -> Path:
    path.write_text(
        f"""schema_version: 1
kind: video_split
input:
  video_path: /videos
output:
  clip_path: /clips
{extra}
""",
        encoding="utf-8",
    )
    return path


def _write_caption_judge_config(path: Path, *, extra: str = "") -> Path:
    path.write_text(
        f"""schema_version: 1
kind: caption_judge
input:
  baseline: /baseline
  candidate: /candidate
output:
  report_path: /reports/caption_judge_report.json
{extra}
""",
        encoding="utf-8",
    )
    return path


def test_pipeline_runtime_hands_resolved_config_to_ray_data_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Runtime entrypoint resolves config and delegates to the Ray Data typed execution entrypoint."""
    captured: dict[str, ResolvedVideoSplitConfig] = {}

    def fake_run_config(config: ResolvedVideoSplitConfig) -> int:
        captured["config"] = config
        return 12

    monkeypatch.setattr(splitting_pipeline, "run_config", fake_run_config)
    config_path = _write_config(tmp_path / "split.yaml", extra="split:\n  preset: fixed_stride_10s")

    pipeline_runtime.main(config_path, set_overrides=None, json_output=True)

    assert json.loads(capsys.readouterr().out) == {"clips_written": 12}
    assert captured["config"].split.method == "fixed_stride"
    assert captured["config"].input.video_path == "/videos"


def test_pipeline_runtime_hands_caption_judge_config_to_ray_data_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Runtime entrypoint dispatches caption_judge configs by kind."""
    captured: dict[str, object] = {}

    def fake_run_caption_judge_pipeline(*, config: CaptionJudgePipelineConfig) -> SimpleNamespace:
        captured["config"] = config
        return SimpleNamespace(
            passed=False,
            issues=SimpleNamespace(num_rows=2),
            stats=SimpleNamespace(windows_judged=5),
        )

    def fake_write_report(report: SimpleNamespace, path: str, *, report_format: str) -> str:
        captured["report"] = report
        captured["report_format"] = report_format
        return path

    monkeypatch.setattr(caption_judge_driver, "run_caption_judge_pipeline", fake_run_caption_judge_pipeline)
    monkeypatch.setattr(caption_judge_report_io, "write_report", fake_write_report)
    config_path = _write_caption_judge_config(tmp_path / "caption_judge.yaml")

    pipeline_runtime.main(
        config_path,
        set_overrides=["judge.max_output_tokens=1024"],
        json_output=True,
    )

    assert json.loads(capsys.readouterr().out) == {
        "passed": False,
        "issues": 2,
        "windows_judged": 5,
        "report_path": "/reports/caption_judge_report.json",
    }
    assert isinstance(captured["config"], CaptionJudgePipelineConfig)
    config = captured["config"]
    assert isinstance(config, CaptionJudgePipelineConfig)
    assert config.input.baseline == "/baseline"
    assert config.baseline_metadata == "/baseline/lance/v0"
    assert config.judge.max_output_tokens == 1024
    assert captured["report_format"] == "json"


@pytest.mark.parametrize("json_output", [True, False])
def test_pipeline_runtime_hands_resolved_config_to_video_transcode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    *,
    json_output: bool,
) -> None:
    """The packaged runner dispatches video-transcode configs with typed overrides."""
    configs: list[ResolvedVideoTranscodeConfig] = []
    summary: dict[str, object] = {"spans": 12, "transcoded_spans": 12}

    def fake_run_config(config: ResolvedVideoTranscodeConfig) -> dict[str, object]:
        configs.append(config)
        return summary

    monkeypatch.setattr(transcoding_pipeline, "run_config", fake_run_config)
    config_path = tmp_path / "video-transcode.json"
    config_path.write_text(
        json.dumps({"schema_version": 1, "kind": "video-transcode", "input": {"media_root": "/media"}})
    )

    pipeline_runtime.main(config_path, set_overrides=["execution.transcode_cpus=2"], json_output=json_output)

    assert len(configs) == 1
    assert configs[0].input.spans_lance_uri == "/media/lance"
    assert configs[0].execution.transcode_cpus == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    if json_output:
        assert json.loads(captured.out) == summary
    else:
        assert captured.out == "video-transcode: 12 transcoded clip(s)\n"


def test_pipeline_runtime_reports_runtime_errors_as_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Runtime errors are machine-readable when --json is requested."""

    def fake_run_config(config: ResolvedVideoSplitConfig) -> int:
        del config
        msg = "runtime boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(splitting_pipeline, "run_config", fake_run_config)
    config_path = _write_config(tmp_path / "split.yaml")

    with pytest.raises(typer.Exit) as exc_info:
        pipeline_runtime.main(config_path, set_overrides=None, json_output=True)

    captured = capsys.readouterr()
    assert exc_info.value.exit_code == 2
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "ok": False,
        "error": "runtime",
        "message": "runtime boom",
    }


def test_pipeline_runtime_reraises_runtime_errors_without_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Interactive runs keep normal tracebacks for runtime failures."""

    def fake_run_config(config: ResolvedVideoSplitConfig) -> int:
        del config
        msg = "runtime boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(splitting_pipeline, "run_config", fake_run_config)
    config_path = _write_config(tmp_path / "split.yaml")

    with pytest.raises(RuntimeError, match="runtime boom"):
        pipeline_runtime.main(config_path, set_overrides=None, json_output=False)
