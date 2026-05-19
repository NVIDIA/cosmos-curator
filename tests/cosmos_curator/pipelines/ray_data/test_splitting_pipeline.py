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

"""Tests for the Ray Data splitting pipeline CLI helpers."""

import argparse

import pytest
import ray

from cosmos_curator.pipelines.ray_data.splitting_pipeline import (
    _configure_ray_data_progress,
    _download_slots_for_video_count,
    _setup_parser,
)


def _parse_args(*args: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    _setup_parser(parser)
    return parser.parse_args(
        [
            "--input-video-path",
            "/input",
            "--output-clip-path",
            "/output",
            *args,
        ]
    )


def test_progress_cli_defaults_to_disabled_with_boolean_overrides() -> None:
    """The CLI exposes explicit --progress/--no-progress controls."""
    assert _parse_args().progress is False
    assert _parse_args("--progress").progress is True
    assert _parse_args("--no-progress").progress is False


def test_vllm_max_num_seqs_cli_defaults_to_qwen_ray_data_default() -> None:
    """Ray Data captioning exposes the vLLM scheduler capacity tuning knob."""
    assert _parse_args().vllm_max_num_seqs == 64
    assert _parse_args("--vllm-max-num-seqs", "256").vllm_max_num_seqs == 256


def test_vllm_max_num_seqs_cli_rejects_non_positive_values() -> None:
    """VLLM max_num_seqs must be positive."""
    with pytest.raises(SystemExit):
        _parse_args("--vllm-max-num-seqs", "0")
    with pytest.raises(SystemExit):
        _parse_args("--vllm-max-num-seqs", "-1")


def test_configure_ray_data_progress_sets_all_progress_flags() -> None:
    """Progress disabling should suppress bars and execution-start banners."""
    ctx = ray.data.DataContext.get_current()
    previous = {
        "enable_progress_bars": ctx.enable_progress_bars,
        "enable_operator_progress_bars": ctx.enable_operator_progress_bars,
        "enable_rich_progress_bars": ctx.enable_rich_progress_bars,
        "print_on_execution_start": ctx.print_on_execution_start,
        "use_ray_tqdm": ctx.use_ray_tqdm,
    }
    try:
        _configure_ray_data_progress(progress=False)
        assert ctx.enable_progress_bars is False
        assert ctx.enable_operator_progress_bars is False
        assert ctx.enable_rich_progress_bars is False
        assert ctx.print_on_execution_start is False
        assert ctx.use_ray_tqdm is False

        _configure_ray_data_progress(progress=True)
        assert ctx.enable_progress_bars is True
        assert ctx.enable_operator_progress_bars is True
        assert ctx.enable_rich_progress_bars is True
        assert ctx.print_on_execution_start is True
        assert ctx.use_ray_tqdm is False
    finally:
        for name, value in previous.items():
            setattr(ctx, name, value)


def test_download_slots_are_capped_by_input_video_count() -> None:
    """Small runs should not ask Ray for more read/split tasks than inputs."""
    assert _download_slots_for_video_count(num_videos=0, num_nodes=8) == 0
    assert _download_slots_for_video_count(num_videos=1, num_nodes=8) == 1
    assert _download_slots_for_video_count(num_videos=10, num_nodes=1) == 10
    assert _download_slots_for_video_count(num_videos=100, num_nodes=2) == 32
