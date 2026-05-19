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
from cosmos_curator.pipelines.video.read_write.metadata_writer_stage import ClipWriterStage
from cosmos_curator.pipelines.video.splitting_pipeline import _assemble_stages, _setup_parser


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _setup_parser(parser)
    return parser


def _stage_object(stage: CuratorStage | CuratorStageSpec) -> CuratorStage:
    if isinstance(stage, CuratorStageSpec):
        return stage.stage
    return stage


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
