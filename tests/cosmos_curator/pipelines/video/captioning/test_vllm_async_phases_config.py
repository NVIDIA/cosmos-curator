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

"""Focused tests for vLLM async builder-level config validation."""

import pytest

from cosmos_curator.core.interfaces.stage_interface import CuratorStageSpec
from cosmos_curator.pipelines.video.captioning.captioning_builders import (
    CaptioningConfig,
    VllmAsyncCaptionConfig,
    build_captioning_stages,
)
from cosmos_curator.pipelines.video.captioning.gemini_caption_stage import ApiPrepStage
from cosmos_curator.pipelines.video.captioning.vllm_async_config import VllmAsyncConfig
from cosmos_curator.pipelines.video.captioning.vllm_async_stage import (
    VllmAsyncCaptionStage,
    VllmAsyncPrepStage,
)
from cosmos_curator.pipelines.video.preview.preview_stages import PreviewStage
from cosmos_curator.pipelines.video.utils.data_model import WindowConfig


def test_vllm_async_caption_config_default_uses_auto_mode_sentinel() -> None:
    """Default config should use `0` as the auto-derive sentinel."""
    cfg = VllmAsyncCaptionConfig()
    assert cfg.max_concurrent_requests == 0


def test_vllm_async_caption_config_accepts_zero_concurrency_for_auto_mode() -> None:
    """`max_concurrent_requests=0` should be valid (auto-derive mode)."""
    cfg = VllmAsyncCaptionConfig(max_concurrent_requests=0)
    assert cfg.max_concurrent_requests == 0


def test_vllm_async_caption_config_rejects_negative_concurrency() -> None:
    """Negative concurrency should fail validation."""
    with pytest.raises(ValueError, match="must be >= 0"):
        VllmAsyncCaptionConfig(max_concurrent_requests=-1)


def test_num_workers_per_node_default_is_zero() -> None:
    """Default num_workers_per_node should be 0 (autoscale)."""
    cfg = VllmAsyncCaptionConfig()
    assert cfg.num_workers_per_node == 0


def _make_config(
    data_parallel_size: int = 1,
    num_gpus: int = 1,
    num_workers_per_node: int = 0,
    *,
    generate_previews: bool = False,
) -> CaptioningConfig:
    serve_config = VllmAsyncConfig(
        model_variant="qwen",
        num_gpus=num_gpus,
        data_parallel_size=data_parallel_size,
    )
    backend = VllmAsyncCaptionConfig(
        serve_config=serve_config,
        num_workers_per_node=num_workers_per_node,
    )
    return CaptioningConfig(
        backend=backend,
        window_config=WindowConfig(),
        generate_previews=generate_previews,
    )


class TestBuildersNumWorkersPerNode:
    """Verify builders set num_workers_per_node based on mode and config."""

    def test_explicit_workers(self) -> None:
        """Explicit positive value -> exact worker count."""
        cfg = _make_config(num_workers_per_node=7)
        stages = build_captioning_stages(cfg)
        caption_stage = stages[-1]
        assert isinstance(caption_stage, CuratorStageSpec)
        assert caption_stage.num_workers_per_node == 7

    def test_dp_mode_ignores_num_workers(self) -> None:
        """DP mode (dp=7): always 1 worker, num_workers_per_node ignored."""
        cfg = _make_config(data_parallel_size=7, num_workers_per_node=5)
        stages = build_captioning_stages(cfg)
        caption_stage = stages[-1]
        assert isinstance(caption_stage, CuratorStageSpec)
        assert caption_stage.num_workers_per_node == 1

    def test_prep_stage_receives_windowing_fields(self) -> None:
        """VllmAsyncPrepConfig should receive windowing fields from CaptioningConfig."""
        cfg = _make_config()
        stages = build_captioning_stages(cfg)
        prep_spec = stages[0]
        assert isinstance(prep_spec, CuratorStageSpec)
        stage = prep_spec.stage
        assert isinstance(stage, VllmAsyncPrepStage)
        assert stage._prep_config.window_size == 256
        assert stage._prep_config.remainder_threshold == 128
        assert stage._prep_config.keep_mp4 is False

    def test_build_stages_vllm_async_prep_is_first(self) -> None:
        """``vllm_async`` ``build_stages`` produces exactly Prep + Caption (2 stages)."""
        cfg = _make_config()
        stages = build_captioning_stages(cfg)
        assert len(stages) == 2
        assert isinstance(stages[0], CuratorStageSpec)
        assert isinstance(stages[0].stage, VllmAsyncPrepStage)
        assert isinstance(stages[1], CuratorStageSpec)
        assert isinstance(stages[1].stage, VllmAsyncCaptionStage)

    def test_build_stages_vllm_async_no_api_prep_stage(self) -> None:
        """vllm_async pipeline should NOT include ApiPrepStage."""
        cfg = _make_config()
        stages = build_captioning_stages(cfg)
        for s in stages:
            stage_obj = s.stage if isinstance(s, CuratorStageSpec) else s
            assert not isinstance(stage_obj, ApiPrepStage)

    def test_build_stages_vllm_async_with_previews(self) -> None:
        """Enabling previews produces Prep + Preview + Caption (3 stages)."""
        cfg = _make_config(generate_previews=True)
        stages = build_captioning_stages(cfg)
        assert len(stages) == 3
        assert isinstance(stages[0], CuratorStageSpec)
        assert isinstance(stages[0].stage, VllmAsyncPrepStage)
        assert isinstance(stages[1], CuratorStageSpec)
        assert isinstance(stages[1].stage, PreviewStage)
        assert isinstance(stages[2], CuratorStageSpec)
        assert isinstance(stages[2].stage, VllmAsyncCaptionStage)

    def test_prep_config_keep_mp4_from_generate_previews(self) -> None:
        """keep_mp4 should be True when generate_previews is enabled."""
        cfg = _make_config(generate_previews=True)
        stages = build_captioning_stages(cfg)
        prep_spec = stages[0]
        assert isinstance(prep_spec, CuratorStageSpec)
        stage = prep_spec.stage
        assert isinstance(stage, VllmAsyncPrepStage)
        assert stage._prep_config.keep_mp4 is True
