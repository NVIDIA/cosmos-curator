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
"""Tests for split pipeline CLI argument wiring."""

import argparse
from pathlib import Path

import pytest

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.pipelines.video.captioning.captioning_builders import CaptioningConfig
from cosmos_curator.pipelines.video.read_write.metadata_writer_stage import ClipWriterStage
from cosmos_curator.pipelines.video.splitting_pipeline import _assemble_stages, _setup_parser
from cosmos_curator.pipelines.video.utils.data_model import VllmConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _setup_parser(parser)
    return parser


def _stage_object(stage: CuratorStage | CuratorStageSpec) -> CuratorStage:
    if isinstance(stage, CuratorStageSpec):
        return stage.stage
    return stage


def _caption_args(extra_args: list[str]) -> argparse.Namespace:
    input_path = Path.cwd() / "tmp-input"
    output_path = Path.cwd() / "tmp-output"
    return _parser().parse_args(
        [
            "--input-video-path",
            input_path.as_posix(),
            "--output-clip-path",
            output_path.as_posix(),
            "--no-generate-embeddings",
            *extra_args,
        ]
    )


def _capture_captioning_config(monkeypatch: pytest.MonkeyPatch) -> dict[str, CaptioningConfig]:
    captured: dict[str, CaptioningConfig] = {}

    def fake_build_captioning_stages(config: CaptioningConfig) -> list[CuratorStage | CuratorStageSpec]:
        captured["config"] = config
        return []

    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.splitting_pipeline.build_captioning_stages", fake_build_captioning_stages
    )
    return captured


def test_caption_quality_flags_default_enabled() -> None:
    """Caption quality flags should default to enabled."""
    args = _parser().parse_args([])

    assert args.caption_quality_flags_enabled is True


def test_no_caption_quality_flags_disables_flags() -> None:
    """The disable flag should set caption_quality_flags_enabled to False."""
    args = _parser().parse_args(["--no-caption-quality-flags"])

    assert args.caption_quality_flags_enabled is False


def test_caption_quality_stats_default_enabled() -> None:
    """Run-level caption quality stats should default to enabled."""
    args = _parser().parse_args([])

    assert args.caption_quality_stats_enabled is True


def test_no_caption_quality_stats_disables_artifact() -> None:
    """The disable flag should set caption_quality_stats_enabled to False."""
    args = _parser().parse_args(["--no-caption-quality-stats"])

    assert args.caption_quality_stats_enabled is False


def test_no_caption_quality_stats_reaches_clip_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage assembly should pass the disable flag to ClipWriterStage."""
    monkeypatch.setattr("cosmos_curator.pipelines.video.splitting_pipeline.build_captioning_stages", lambda _: [])
    input_path = Path.cwd() / "tmp-input"
    output_path = Path.cwd() / "tmp-output"
    args = _parser().parse_args(
        [
            "--input-video-path",
            input_path.as_posix(),
            "--output-clip-path",
            output_path.as_posix(),
            "--no-generate-embeddings",
            "--no-caption-quality-stats",
        ]
    )

    stages = _assemble_stages(args)
    writers = [stage for stage in map(_stage_object, stages) if isinstance(stage, ClipWriterStage)]

    assert len(writers) == 1
    assert writers[0]._caption_quality_stats_enabled is False


def test_write_all_caption_json_default_disabled() -> None:
    """Aggregate caption JSON should be opt-in."""
    args = _parser().parse_args([])

    assert args.write_all_caption_json is False


def test_write_all_caption_json_opt_in() -> None:
    """The positive flag should enable aggregate caption JSON."""
    args = _parser().parse_args(["--write-all-caption-json"])

    assert args.write_all_caption_json is True


def test_no_write_all_caption_json_flag_removed() -> None:
    """The old negative flag should no longer be accepted."""
    with pytest.raises(SystemExit):
        _parser().parse_args(["--no-write-all-caption-json"])


@pytest.mark.parametrize(
    "caption_algo",
    ["qwen", "qwen3_6_27b", "qwen3_vl_30b", "cosmos_r1", "cosmos_r2", "nemotron"],
)
def test_vllm_video_max_pixels_reaches_sync_vllm_configs(
    monkeypatch: pytest.MonkeyPatch,
    caption_algo: str,
) -> None:
    """Accepted regular sync vLLM backends receive both resize-budget carriers."""
    captured = _capture_captioning_config(monkeypatch)
    args = _caption_args(
        [
            "--captioning-algorithm",
            caption_algo,
            "--vllm-video-max-pixels-per-frame",
            "100500",
        ]
    )

    _assemble_stages(args)

    config = captured["config"]
    assert config.window_config.video_max_pixels_per_frame == 100500
    assert isinstance(config.backend, VllmConfig)
    assert config.backend.video_max_pixels_per_frame == 100500


@pytest.mark.parametrize("caption_algo", ["vllm_async", "gemini", "openai"])
def test_vllm_video_max_pixels_rejects_non_sync_vllm(
    monkeypatch: pytest.MonkeyPatch,
    caption_algo: str,
) -> None:
    """The sync-only flag is rejected for async and API captioning paths."""
    _capture_captioning_config(monkeypatch)
    args = _caption_args(
        [
            "--captioning-algorithm",
            caption_algo,
            "--vllm-video-max-pixels-per-frame",
            "100500",
        ]
    )

    with pytest.raises(ValueError, match="regular windowed sync vLLM"):
        _assemble_stages(args)


@pytest.mark.parametrize("value", ["100351", "602113"])
def test_vllm_video_max_pixels_rejects_values_outside_bounds(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    """The flag is rejected outside the accepted upper-bound domain."""
    _capture_captioning_config(monkeypatch)
    args = _caption_args(["--captioning-algorithm", "qwen", "--vllm-video-max-pixels-per-frame", value])

    with pytest.raises(ValueError, match=r"integer in \[100352, 602112\]"):
        _assemble_stages(args)


def test_vllm_video_max_pixels_rejects_unsupported_caption_algorithm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage assembly still rejects future unsupported caption algorithms."""
    _capture_captioning_config(monkeypatch)
    args = _caption_args(["--captioning-algorithm", "qwen", "--vllm-video-max-pixels-per-frame", "100500"])
    args.captioning_algorithm = "future_backend"

    with pytest.raises(RuntimeError, match="Unsupported captioning algorithm"):
        _assemble_stages(args)


def test_nemotron_forces_model_does_preprocess_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting nemotron must force both preprocess flags to True at the call site.

    Nemotron's intended mode is ``model_does_preprocess=True``: vLLM owns
    ``do_resize``/``do_rescale``/``do_normalize`` rather than the CPU prep stage.
    Both ``window_config.model_does_preprocess`` and ``backend.preprocess`` must
    be True regardless of the dataclass defaults so the contract is visible at
    the call site and resistant to default drift.
    """
    captured = _capture_captioning_config(monkeypatch)
    args = _caption_args(["--captioning-algorithm", "nemotron"])

    _assemble_stages(args)

    config = captured["config"]
    assert config.window_config.model_does_preprocess is True
    assert isinstance(config.backend, VllmConfig)
    assert config.backend.preprocess is True


def test_vllm_video_max_pixels_help_text_names_scope_bounds_and_grid() -> None:
    """Help text should describe the upper-bound scope, bounds, and grid quantization."""
    help_text = _parser().format_help()

    assert "--vllm-video-max-pixels-per-frame" in help_text
    assert "regular" in help_text
    assert "windowed sync vLLM" in help_text
    assert "[100352, 602112]" in help_text
    assert "28 for CPU prep" in help_text
    assert "32 for Qwen3" in help_text
