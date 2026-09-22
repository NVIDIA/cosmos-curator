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
"""Tests for stage compare wiring in the video splitting pipeline."""

import argparse
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.core.utils.misc.stage_compare import get_stage_name_after, get_stages_to_compare
from cosmos_curator.pipelines.video.splitting_pipeline import _split
from tests.cosmos_curator.pipelines.stage_test_utils import stage_object


class StageA(CuratorStage):
    """First test stage."""

    def process_data(self, tasks: list) -> list:
        """Return tasks unchanged for wiring tests."""
        return tasks


class StageB(CuratorStage):
    """Second test stage."""

    def process_data(self, tasks: list) -> list:
        """Return tasks unchanged for wiring tests."""
        return tasks


class StageC(CuratorStage):
    """Third test stage."""

    def process_data(self, tasks: list) -> list:
        """Return tasks unchanged for wiring tests."""
        return tasks


@pytest.fixture
def stub_ffmpeg_h264_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the startup FFmpeg check for wiring tests that do not process media."""
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.splitting_pipeline.assert_ffmpeg_supports_h264",
        lambda: None,
    )


def test_get_stage_name_after() -> None:
    """The helper should return the immediate successor stage name."""
    assert get_stage_name_after([StageA(), StageB(), StageC()], "StageA") == "StageB"


def test_get_stage_name_after_last_stage() -> None:
    """Comparing the last stage without override should fail clearly."""
    with pytest.raises(ValueError, match="cannot infer golden for last stage"):
        get_stage_name_after([StageA(), StageB()], "StageB")


def test_get_stages_to_compare_uses_half_open_interval_and_preserves_specs() -> None:
    """The compare range should be [start, end) and preserve original stage specs."""
    stage_b_spec = CuratorStageSpec(StageB(), num_workers_per_node=2)
    stages = [StageA(), stage_b_spec, StageC()]

    compare_stages = get_stages_to_compare(stages, "StageA", "StageC")

    assert [type(stage_object(stage)) for stage in compare_stages] == [StageA, StageB]
    assert compare_stages[1] is stage_b_spec


@pytest.mark.usefixtures("stub_ffmpeg_h264_preflight")
def test_split_compare_branch_uses_stage_compare(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The compare branch should invoke run_stage_compare and return early."""
    args = argparse.Namespace(
        stage_save=[],
        stage_save_sample_rate=0.0,
        stage_replay=[],
        stage_compare=["StageA"],
        stage_compare_path=None,
        stage_compare_atol=2.0,
        stage_compare_pass_threshold=1.0,
        output_clip_path=tmp_path / "output",
        model_weights_path=tmp_path / "weights",
        limit=5,
        upload_clip_info_in_lance=False,
        output_s3_profile_name=None,
        stage_compare_backend="xenna",
    )

    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.splitting_pipeline.build_input_data",
        lambda _args: ([], [], 0, 0),
    )
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.splitting_pipeline._assemble_stages",
        lambda _args: [StageA(), StageB()],
    )
    compare_mock = MagicMock()
    compare_mock.return_value = MagicMock(passed=True, report=MagicMock(pass_rate=1.0))
    monkeypatch.setattr("cosmos_curator.pipelines.video.splitting_pipeline.run_stage_compare", compare_mock)
    run_pipeline_mock = MagicMock()
    monkeypatch.setattr("cosmos_curator.pipelines.video.splitting_pipeline.run_pipeline", run_pipeline_mock)

    _split(args)

    run_pipeline_mock.assert_not_called()
    compare_mock.assert_called_once()
    call_args = compare_mock.call_args
    assert call_args.args[1] == args.output_clip_path / "tasks" / "StageA"
    assert call_args.args[2] == args.output_clip_path / "tasks" / "StageB"
    assert call_args.args[3] == args.stage_compare_atol
    assert call_args.args[5] == args.stage_compare_pass_threshold
    assert call_args.kwargs["profile_name"] == args.output_s3_profile_name
    assert call_args.kwargs["report_path"] == args.output_clip_path / "compare" / "StageA" / "report.json"
    assert call_args.kwargs["backend"] == "xenna"
    assert call_args.kwargs["args"] is args
    assert call_args.kwargs["model_weights_prefix"] == args.model_weights_path


@pytest.mark.usefixtures("stub_ffmpeg_h264_preflight")
def test_split_compare_branch_preserves_remote_stage_compare_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Remote golden override paths should remain remote when passed into compare."""
    args = argparse.Namespace(
        stage_save=[],
        stage_save_sample_rate=0.0,
        stage_replay=[],
        stage_compare=["StageA"],
        stage_compare_path="s3://bucket/golden",
        stage_compare_atol=2.0,
        stage_compare_pass_threshold=1.0,
        output_clip_path=tmp_path / "output",
        model_weights_path=tmp_path / "weights",
        limit=5,
        upload_clip_info_in_lance=False,
        output_s3_profile_name="default",
        stage_compare_backend="xenna",
    )

    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.splitting_pipeline.build_input_data",
        lambda _args: ([], [], 0, 0),
    )
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.splitting_pipeline._assemble_stages",
        lambda _args: [StageA(), StageB()],
    )
    compare_mock = MagicMock()
    compare_mock.return_value = MagicMock(passed=True, report=MagicMock(pass_rate=1.0))
    monkeypatch.setattr("cosmos_curator.pipelines.video.splitting_pipeline.run_stage_compare", compare_mock)
    monkeypatch.setattr("cosmos_curator.pipelines.video.splitting_pipeline.run_pipeline", MagicMock())

    _split(args)

    compare_mock.assert_called_once()
    call_args = compare_mock.call_args
    assert str(call_args.args[2]) == "s3://bucket/golden/tasks/StageB"


@pytest.mark.usefixtures("stub_ffmpeg_h264_preflight")
def test_split_compare_branch_uses_explicit_half_open_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Two-stage compare should run [start, end) and compare against input to end."""
    args = argparse.Namespace(
        stage_save=[],
        stage_save_sample_rate=0.0,
        stage_replay=[],
        stage_compare=["StageA", "StageC"],
        stage_compare_path=None,
        stage_compare_atol=2.0,
        stage_compare_pass_threshold=1.0,
        output_clip_path=tmp_path / "output",
        model_weights_path=tmp_path / "weights",
        limit=5,
        upload_clip_info_in_lance=False,
        output_s3_profile_name=None,
        stage_compare_backend="xenna",
    )

    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.splitting_pipeline.build_input_data",
        lambda _args: ([], [], 0, 0),
    )
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.splitting_pipeline._assemble_stages",
        lambda _args: [StageA(), StageB(), StageC()],
    )
    compare_mock = MagicMock()
    compare_mock.return_value = MagicMock(passed=True, report=MagicMock(pass_rate=1.0))
    monkeypatch.setattr("cosmos_curator.pipelines.video.splitting_pipeline.run_stage_compare", compare_mock)
    monkeypatch.setattr("cosmos_curator.pipelines.video.splitting_pipeline.run_pipeline", MagicMock())

    _split(args)

    compare_mock.assert_called_once()
    call_args = compare_mock.call_args
    compared_stage_names = [stage.__class__.__name__ for stage in call_args.args[0]]
    assert compared_stage_names == ["StageA", "StageB"]
    assert call_args.args[1] == args.output_clip_path / "tasks" / "StageA"
    assert call_args.args[2] == args.output_clip_path / "tasks" / "StageC"
