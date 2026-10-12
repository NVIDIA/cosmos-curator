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
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import typer

from cosmos_curator.core.utils import environment
from cosmos_curator.core.utils.infra import metrics_push
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
    summary: dict[str, object] = {"input_rows": 12, "selected_rows": 4, "remaining_fragments": 2}

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
        assert captured.out == "video-transcode: published results for 4 of 12 input row(s); 2 fragment(s) pending\n"


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


@pytest.fixture
def metrics_push_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Enable the push via env and record the enable/flush/disable calls, in order."""
    calls: list[str] = []

    def fake_enable(config: object) -> bool:
        del config
        calls.append("enable")
        return True

    monkeypatch.setenv(environment.OTLP_METRICS_PUSH_ENV_VAR, "true")
    monkeypatch.setattr(metrics_push, "enable_metrics_push", fake_enable)
    monkeypatch.setattr(metrics_push, "flush_metrics_push", lambda: calls.append("flush"))
    monkeypatch.setattr(metrics_push, "disable_metrics_push", lambda: calls.append("disable"))
    return calls


def test_pipeline_runtime_wraps_the_run_in_the_metrics_push_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    metrics_push_calls: list[str],
) -> None:
    """The run is bracketed by enable and flush -> disable."""

    def fake_run_config(config: ResolvedVideoSplitConfig) -> int:
        del config
        metrics_push_calls.append("run")
        return 3

    monkeypatch.setattr(splitting_pipeline, "run_config", fake_run_config)

    pipeline_runtime.main(_write_config(tmp_path / "split.yaml"), set_overrides=None, json_output=True)

    assert metrics_push_calls == ["enable", "run", "flush", "disable"]
    assert json.loads(capsys.readouterr().out) == {"clips_written": 3}


def test_pipeline_runtime_still_flushes_the_metrics_push_when_the_run_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    metrics_push_calls: list[str],
) -> None:
    """A failed run exports the metrics it produced before the fault."""

    def fake_run_config(config: ResolvedVideoSplitConfig) -> int:
        del config
        msg = "runtime boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(splitting_pipeline, "run_config", fake_run_config)

    with pytest.raises(RuntimeError, match="runtime boom"):
        pipeline_runtime.main(_write_config(tmp_path / "split.yaml"), set_overrides=None, json_output=False)

    assert metrics_push_calls == ["enable", "flush", "disable"]


@pytest.mark.parametrize("value", [None, "false", "enabled"])
def test_metrics_push_is_not_enabled_without_a_truthy_switch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    value: str | None,
) -> None:
    """Unset, false, or an invalid value (warned, not fatal) leaves the run plain."""

    def fake_run_config(config: ResolvedVideoSplitConfig) -> int:
        del config
        return 0

    if value is None:
        monkeypatch.delenv(environment.OTLP_METRICS_PUSH_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(environment.OTLP_METRICS_PUSH_ENV_VAR, value)
    monkeypatch.setattr(splitting_pipeline, "run_config", fake_run_config)
    enable = Mock()
    monkeypatch.setattr(metrics_push, "enable_metrics_push", enable)

    pipeline_runtime.main(_write_config(tmp_path / "split.yaml"), set_overrides=None, json_output=True)

    enable.assert_not_called()


def test_a_broken_metrics_push_install_degrades_to_a_plain_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A failed import of the pusher warns instead of failing the run."""

    def fake_run_config(config: ResolvedVideoSplitConfig) -> int:
        del config
        return 7

    monkeypatch.setattr(splitting_pipeline, "run_config", fake_run_config)
    monkeypatch.setenv(environment.OTLP_METRICS_PUSH_ENV_VAR, "true")
    monkeypatch.setitem(sys.modules, "cosmos_curator.core.utils.infra.metrics_push", None)

    pipeline_runtime.main(_write_config(tmp_path / "split.yaml"), set_overrides=None, json_output=True)

    assert json.loads(capsys.readouterr().out) == {"clips_written": 7}
