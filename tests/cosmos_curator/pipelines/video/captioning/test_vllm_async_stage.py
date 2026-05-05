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

"""Tests for vllm_async_stage: config, utilities, and VllmAsyncCaptionStage.

Covers the in-process AsyncLLM architecture.  All vLLM and transformers
imports are mocked since these tests run on CPU without the ``vllm``
pixi environment.
"""

import argparse
import asyncio
import collections
import contextlib
import json
import os
import pickle
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Generator
from pathlib import Path
from unittest.mock import MagicMock, call, patch
from uuid import uuid4

import attrs
import numpy as np
import pytest

from cosmos_curator.pipelines.video.captioning import vllm_async_stage
from cosmos_curator.pipelines.video.captioning.vllm_async_config import (
    VllmAsyncConfig,
    VllmAsyncPrepConfig,
    build_vllm_async_config,
)
from cosmos_curator.pipelines.video.captioning.vllm_async_stage import (
    VllmAsyncCaptionStage,
    VllmAsyncPrepStage,
    _build_engine_args,
    _build_render_payload,
    _ContinuousTaskTracker,
    _PreparedWindow,
    _resolve_mode,
    _VllmAsyncModel,
    _VllmAsyncStageMode,
    resolve_model_path,
)
from cosmos_curator.pipelines.video.utils.data_model import (
    Clip,
    SplitPipeTask,
    TokenCounts,
    Video,
    VllmSamplingConfig,
    Window,
)
from cosmos_curator.pipelines.video.utils.vision_process import VIDEO_MAX_PIXELS, VIDEO_MIN_PIXELS
from cosmos_curator.pipelines.video.utils.windowing_utils import WindowFrameInfo
from cosmos_xenna.ray_utils.continuous_stage import (
    ContinuousInterface,
    ContinuousTaskInput,
    ContinuousTaskOutput,
)


def _make_task(mp4_bytes: bytes | None, *, num_windows: int = 1) -> SplitPipeTask:
    """Create a minimal SplitPipeTask with one clip and the given windows."""
    clip = Clip(uuid=uuid4(), source_video="source.mp4", span=(0.0, 1.0))
    for i in range(num_windows):
        clip.windows.append(Window(start_frame=i * 10, end_frame=(i + 1) * 10, mp4_bytes=mp4_bytes))
    video = Video(input_video=Path("source.mp4"))
    video.clips.append(clip)
    return SplitPipeTask(session_id="test-session", video=video)


def _make_task_with_encoded_data(encoded_data: bytes | None) -> SplitPipeTask:
    """Create a minimal SplitPipeTask with one clip carrying encoded_data (no pre-existing windows)."""
    clip = Clip(uuid=uuid4(), source_video="source.mp4", span=(0.0, 1.0), encoded_data=encoded_data)
    video = Video(input_video=Path("source.mp4"))
    video.clips.append(clip)
    return SplitPipeTask(session_id="test-session", video=video)


def _mock_request_output(text: str = "A cat video") -> MagicMock:
    """Build a mock vLLM RequestOutput with one output containing the given text."""
    output = MagicMock()
    output.text = text
    output.finish_reason = "stop"
    result = MagicMock()
    result.outputs = [output]
    return result


def _async_gen_side_effect(request_output: MagicMock) -> Callable[..., AsyncGenerator[MagicMock, None]]:
    """Return a side_effect callable that produces an async generator yielding *request_output*.

    ``AsyncLLM.generate()`` returns an ``AsyncGenerator[RequestOutput, None]``,
    so the mock must also produce an async iterable rather than a plain coroutine.
    """

    async def _generate(**_kwargs: object) -> AsyncGenerator[MagicMock, None]:
        yield request_output

    return _generate


def _async_gen_side_effect_per_request(text_prefix: str = "CAP") -> Callable[..., AsyncGenerator[MagicMock, None]]:
    """Return a side_effect that encodes the per-call ``request_id`` into the caption text.

    Each call to ``engine.generate(..., request_id=...)`` produces a unique
    ``RequestOutput`` whose ``.outputs[0].text`` is ``f"{text_prefix}-{request_id}"``.
    Tests use this to detect cross-window / cross-task mis-routing -- a swap
    would surface as the wrong window holding another window's caption text.
    """

    async def _generate(**kwargs: object) -> AsyncGenerator[MagicMock, None]:
        request_id = kwargs.get("request_id", "<missing>")
        yield _mock_request_output(f"{text_prefix}-{request_id}")

    return _generate


def _mock_renderer(mock_engine: MagicMock) -> None:
    """Set up ``mock_engine.renderer.render_cmpl`` as a sync passthrough.

    ``VllmAsyncCaptionStage._render_payload`` invokes the **sync**
    ``engine.renderer.render_cmpl`` API via ``asyncio.to_thread`` under
    ``self._render_lock``.  This helper makes the mock return its input
    unchanged so generation-path tests work without a real Renderer.
    """
    mock_engine.renderer.render_cmpl = MagicMock(side_effect=lambda prompts: prompts)


def _populate_window_input(window: Window, prompt_text: str, frames: np.ndarray) -> None:
    """Pre-populate ``window.model_input['vllm_async']`` as the prep stage would.

    Mirrors the flat shape the producer writes today: a prompt string, the
    raw decoded frames buffer, and the frames shape tuple.  The renderer-
    shaped ``{"video": [...]}`` dict is built per render call by
    ``_build_render_payload`` and is never cached on the window.
    """
    window.model_input["vllm_async"] = {
        "prompt": prompt_text,
        "video_frames": frames,
        "frames_shape": tuple(frames.shape),
    }


def _make_prepared_window(
    *,
    prompt_text: str = "describe",
    frames: np.ndarray | None = None,
    window_index: int = 0,
) -> _PreparedWindow:
    """Build a minimal ``_PreparedWindow`` for unit tests."""
    if frames is None:
        frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
    window = Window(start_frame=window_index * 10, end_frame=(window_index + 1) * 10)
    clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 1.0))
    clip.windows = [window]
    return _PreparedWindow(
        clip=clip,
        window_index=window_index,
        window=window,
        prompt_text=prompt_text,
        decoded_rgb_frames=frames,
        sampling_params=MagicMock(),
        frames_shape=tuple(frames.shape),
    )


class TestResolveModelPath:
    """Tests for resolve_model_path() -- local weight cache resolution."""

    @patch("cosmos_curator.core.utils.model.model_utils.get_local_dir_for_weights_name")
    def test_cached_weights_returns_local_path(self, mock_local_dir: MagicMock) -> None:
        """When weights are cached locally, should return the local path."""
        expected = "/config/models/Qwen/Qwen2.5-VL-7B-Instruct"
        mock_dir = MagicMock(spec=Path)
        mock_dir.exists.return_value = True
        mock_dir.__str__ = MagicMock(return_value=expected)
        mock_local_dir.return_value = mock_dir

        assert resolve_model_path("Qwen/Qwen2.5-VL-7B-Instruct") == expected

    @patch("cosmos_curator.core.utils.model.model_utils.get_local_dir_for_weights_name")
    def test_no_cache_raises_error(self, mock_local_dir: MagicMock) -> None:
        """When weights are not cached, should raise FileNotFoundError."""
        mock_dir = MagicMock(spec=Path)
        mock_dir.exists.return_value = False
        mock_local_dir.return_value = mock_dir

        with pytest.raises(FileNotFoundError, match="Pre-downloaded model weights not found"):
            resolve_model_path("Qwen/Qwen2.5-VL-7B-Instruct")


class TestVllmAsyncModel:
    """Tests for _VllmAsyncModel -- lightweight ModelInterface for weight download registration."""

    def test_model_id_names_resolves_known_variant(self) -> None:
        """Known variant 'qwen' should resolve to the Qwen HuggingFace model ID."""
        model = _VllmAsyncModel("qwen")
        assert model.model_id_names == ["Qwen/Qwen2.5-VL-7B-Instruct"]

    def test_unknown_variant_raises(self) -> None:
        """Unregistered variant should raise ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            _VllmAsyncModel("custom-org/my-model")

    def test_conda_env_name_is_unified(self) -> None:
        """conda_env_name should return 'unified' where vLLM is installed."""
        model = _VllmAsyncModel("qwen")
        assert model.conda_env_name == "unified"

    def test_setup_is_noop(self) -> None:
        """setup() should succeed without side effects (engine loads model weights)."""
        model = _VllmAsyncModel("qwen")
        model.setup()

    def test_each_variant_resolves_correctly(self) -> None:
        """All registered vLLM variants should produce the expected model IDs."""
        expected = {
            "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
            "nemotron": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
            "cosmos_r1": "nvidia/Cosmos-Reason1-7B",
            "cosmos_r2": "nvidia/Cosmos-Reason2-8B",
        }
        for variant, hf_id in expected.items():
            model = _VllmAsyncModel(variant)
            assert model.model_id_names == [hf_id], f"variant={variant}"


class TestBuildEngineArgs:
    """Tests for _build_engine_args() -- VllmAsyncConfig to AsyncEngineArgs conversion.

    Both ``AsyncEngineArgs`` and ``CompilationConfig`` are conditionally
    imported in the main module (only available inside the ``vllm``
    pixi environment).  The :meth:`_patch_engine` helper patches both
    onto the module with ``create=True`` so tests run on plain CPU.
    """

    @contextlib.contextmanager
    def _patch_engine(self) -> Generator[tuple[MagicMock, MagicMock], None, None]:
        """Patch ``AsyncEngineArgs`` and ``CompilationConfig`` on the module."""
        mock_engine_args_cls = MagicMock()
        mock_comp_config_cls = MagicMock()
        with (
            patch.object(vllm_async_stage, "AsyncEngineArgs", mock_engine_args_cls, create=True),
            patch.object(vllm_async_stage, "CompilationConfig", mock_comp_config_cls, create=True),
        ):
            yield mock_engine_args_cls, mock_comp_config_cls

    def test_basic_mapping(self) -> None:
        """Core fields should map to AsyncEngineArgs attributes."""
        with self._patch_engine() as (mock_engine_args_cls, _):
            config = VllmAsyncConfig(model_variant="qwen", num_gpus=4, gpu_memory_utilization=0.9)
            _build_engine_args(config, "/config/models/Qwen")

        call_kwargs = mock_engine_args_cls.call_args.kwargs
        assert call_kwargs["model"] == "/config/models/Qwen"
        assert call_kwargs["tensor_parallel_size"] == 4
        assert call_kwargs["gpu_memory_utilization"] == 0.9
        assert call_kwargs["served_model_name"] == ["qwen"]

    def test_max_model_len_zero_maps_to_none(self) -> None:
        """max_model_len=0 should pass None to let the engine auto-detect."""
        with self._patch_engine() as (mock_engine_args_cls, _):
            config = VllmAsyncConfig(model_variant="test", max_model_len=0)
            _build_engine_args(config, "/model")

        assert mock_engine_args_cls.call_args.kwargs["max_model_len"] is None

    def test_max_model_len_nonzero_passed_through(self) -> None:
        """max_model_len > 0 should be passed as-is."""
        with self._patch_engine() as (mock_engine_args_cls, _):
            config = VllmAsyncConfig(model_variant="test", max_model_len=32768)
            _build_engine_args(config, "/model")

        assert mock_engine_args_cls.call_args.kwargs["max_model_len"] == 32768

    def test_cudagraph_mode_piecewise_emits_compilation_config(self) -> None:
        """Default cudagraph_mode='piecewise' should build CompilationConfig."""
        with self._patch_engine() as (mock_engine_args_cls, mock_comp_config_cls):
            config = VllmAsyncConfig(model_variant="test")
            _build_engine_args(config, "/model")

        mock_comp_config_cls.assert_called_once_with(cudagraph_mode="piecewise")
        assert mock_engine_args_cls.call_args.kwargs["compilation_config"] is mock_comp_config_cls.return_value

    def test_cudagraph_mode_empty_omits_compilation_config(self) -> None:
        """Empty cudagraph_mode should produce None compilation_config."""
        with self._patch_engine() as (mock_engine_args_cls, mock_comp_config_cls):
            config = VllmAsyncConfig(model_variant="test", cudagraph_mode="")
            _build_engine_args(config, "/model")

        mock_comp_config_cls.assert_not_called()
        assert mock_engine_args_cls.call_args.kwargs["compilation_config"] is None

    def test_limit_mm_per_prompt_parsed_as_json(self) -> None:
        """limit_mm_per_prompt should be parsed from JSON string to dict."""
        with self._patch_engine() as (mock_engine_args_cls, _):
            config = VllmAsyncConfig(model_variant="test", limit_mm_per_prompt='{"video": 1}')
            _build_engine_args(config, "/model")

        assert mock_engine_args_cls.call_args.kwargs["limit_mm_per_prompt"] == {"video": 1}

    def test_data_parallel_size_greater_than_one(self) -> None:
        """data_parallel_size > 1 should pass through."""
        with self._patch_engine() as (mock_engine_args_cls, _):
            config = VllmAsyncConfig(model_variant="test", data_parallel_size=4)
            _build_engine_args(config, "/model")

        assert mock_engine_args_cls.call_args.kwargs["data_parallel_size"] == 4

    def test_enforce_eager_passed_through(self) -> None:
        """enforce_eager should be passed directly to AsyncEngineArgs."""
        with self._patch_engine() as (mock_engine_args_cls, _):
            config = VllmAsyncConfig(model_variant="test", enforce_eager=True)
            _build_engine_args(config, "/model")

        assert mock_engine_args_cls.call_args.kwargs["enforce_eager"] is True

    def test_enable_prefix_caching_always_true(self) -> None:
        """enable_prefix_caching should always be True for KV cache reuse."""
        with self._patch_engine() as (mock_engine_args_cls, _):
            config = VllmAsyncConfig(model_variant="test")
            _build_engine_args(config, "/model")

        assert mock_engine_args_cls.call_args.kwargs["enable_prefix_caching"] is True

    def test_mm_processor_cache_fields(self) -> None:
        """mm_processor_cache_gb and _type should be passed through."""
        with self._patch_engine() as (mock_engine_args_cls, _):
            config = VllmAsyncConfig(
                model_variant="test",
                mm_processor_cache_gb=32.0,
                mm_processor_cache_type="shm",
            )
            _build_engine_args(config, "/model")

        call_kwargs = mock_engine_args_cls.call_args.kwargs
        assert call_kwargs["mm_processor_cache_gb"] == 32.0
        assert call_kwargs["mm_processor_cache_type"] == "shm"

    def test_long_prefill_threshold_zero_passes_disabled(self) -> None:
        """Explicit 0 should pass through as disabled (no clamping)."""
        with self._patch_engine() as (mock_engine_args_cls, _):
            config = VllmAsyncConfig(model_variant="test", long_prefill_token_threshold=0)
            _build_engine_args(config, "/model")
        assert mock_engine_args_cls.call_args.kwargs["long_prefill_token_threshold"] == 0

    def test_long_prefill_threshold_explicit_passes_through(self) -> None:
        """Explicit positive value should pass through unchanged."""
        with self._patch_engine() as (mock_engine_args_cls, _):
            config = VllmAsyncConfig(model_variant="test", long_prefill_token_threshold=4096)
            _build_engine_args(config, "/model")
        assert mock_engine_args_cls.call_args.kwargs["long_prefill_token_threshold"] == 4096

    def test_distributed_executor_backend_passed_through(self) -> None:
        """distributed_executor_backend should be passed to AsyncEngineArgs."""
        with self._patch_engine() as (mock_engine_args_cls, _):
            config = VllmAsyncConfig(model_variant="test", distributed_executor_backend="ray")
            _build_engine_args(config, "/model")
        assert mock_engine_args_cls.call_args.kwargs["distributed_executor_backend"] == "ray"

    def test_distributed_executor_backend_mp(self) -> None:
        """distributed_executor_backend='mp' should pass through."""
        with self._patch_engine() as (mock_engine_args_cls, _):
            config = VllmAsyncConfig(model_variant="test", distributed_executor_backend="mp")
            _build_engine_args(config, "/model")
        assert mock_engine_args_cls.call_args.kwargs["distributed_executor_backend"] == "mp"

    def test_async_scheduling_passed_through(self) -> None:
        """async_scheduling should be passed to AsyncEngineArgs."""
        with self._patch_engine() as (mock_engine_args_cls, _):
            config = VllmAsyncConfig(model_variant="test", async_scheduling=False)
            _build_engine_args(config, "/model")
        assert mock_engine_args_cls.call_args.kwargs["async_scheduling"] is False

    def test_enable_chunked_prefill_none_passed_through(self) -> None:
        """enable_chunked_prefill=None (auto-detect) should pass to AsyncEngineArgs."""
        with self._patch_engine() as (mock_engine_args_cls, _):
            config = VllmAsyncConfig(model_variant="test", enable_chunked_prefill=None)
            _build_engine_args(config, "/model")
        assert mock_engine_args_cls.call_args.kwargs["enable_chunked_prefill"] is None

    def test_enable_chunked_prefill_explicit_passed_through(self) -> None:
        """enable_chunked_prefill=True should pass to AsyncEngineArgs."""
        with self._patch_engine() as (mock_engine_args_cls, _):
            config = VllmAsyncConfig(model_variant="test", enable_chunked_prefill=True)
            _build_engine_args(config, "/model")
        assert mock_engine_args_cls.call_args.kwargs["enable_chunked_prefill"] is True

    def test_enable_chunked_prefill_false_passed_through(self) -> None:
        """enable_chunked_prefill=False should pass to AsyncEngineArgs."""
        with self._patch_engine() as (mock_engine_args_cls, _):
            config = VllmAsyncConfig(model_variant="test", enable_chunked_prefill=False)
            _build_engine_args(config, "/model")
        assert mock_engine_args_cls.call_args.kwargs["enable_chunked_prefill"] is False

    def test_quantization_none_passes_none(self) -> None:
        """quantization=None should pass None to AsyncEngineArgs (no quantization)."""
        with self._patch_engine() as (mock_engine_args_cls, _):
            config = VllmAsyncConfig(model_variant="test", quantization=None)
            _build_engine_args(config, "/model")
        assert mock_engine_args_cls.call_args.kwargs["quantization"] is None

    def test_quantization_empty_string_passes_none(self) -> None:
        """quantization="" should be normalized to None before reaching AsyncEngineArgs."""
        with self._patch_engine() as (mock_engine_args_cls, _):
            config = VllmAsyncConfig(model_variant="test", quantization="")
            _build_engine_args(config, "/model")
        assert mock_engine_args_cls.call_args.kwargs["quantization"] is None

    def test_quantization_explicit_value_passes_through(self) -> None:
        """quantization="fp8" should pass "fp8" to AsyncEngineArgs."""
        with self._patch_engine() as (mock_engine_args_cls, _):
            config = VllmAsyncConfig(model_variant="test", quantization="fp8")
            _build_engine_args(config, "/model")
        assert mock_engine_args_cls.call_args.kwargs["quantization"] == "fp8"

    def test_attention_backend_not_passed_to_engine_args(self) -> None:
        """attention_backend should not be passed, letting vLLM auto-select."""
        with self._patch_engine() as (mock_engine_args_cls, _):
            config = VllmAsyncConfig(model_variant="test")
            _build_engine_args(config, "/model")
        assert "attention_backend" not in mock_engine_args_cls.call_args.kwargs


class TestVllmAsyncGpuTraceAttributes:
    """Tests for _vllm_async_collect_gpu_trace_attributes (OTel GPU metadata)."""

    def test_visible_gpu_ids_from_cuda_visible_devices(self) -> None:
        """Xenna env parser should populate visible_gpu_ids and cuda_visible_devices."""
        stage = MagicMock()
        stage.resources.gpus = 2.0
        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}),
            patch.object(vllm_async_stage.ray, "is_initialized", return_value=False),
        ):
            d = vllm_async_stage._vllm_async_collect_gpu_trace_attributes(stage)
        assert d["stage.requested_gpus"] == 2.0
        assert d["stage.cuda_visible_devices"] == "0,1"
        assert d["stage.visible_gpu_ids"] == "0,1"

    def test_ray_fallback_when_no_visible_ids_from_env(self) -> None:
        """When CUDA env parses to no IDs, use ray.get_gpu_ids() if Ray is up."""
        stage = MagicMock()
        stage.resources.gpus = 1.0
        rt = MagicMock()
        rt.get_node_id.return_value = None
        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}),
            patch.object(vllm_async_stage.ray, "is_initialized", return_value=True),
            patch.object(vllm_async_stage.ray, "get_gpu_ids", return_value=[7]),
            patch.object(vllm_async_stage.ray, "get_runtime_context", return_value=rt),
        ):
            d = vllm_async_stage._vllm_async_collect_gpu_trace_attributes(stage)
        assert d["stage.requested_gpus"] == 1.0
        assert "stage.visible_gpu_ids" not in d
        assert d["stage.ray_gpu_ids"] == "7"


class TestVllmAsyncCaptionStage:
    """Tests for VllmAsyncCaptionStage resource and config declarations."""

    def _make_config(self, **overrides: object) -> VllmAsyncConfig:
        defaults: dict[str, object] = {
            "model_variant": "qwen",
            "num_gpus": 2,
        }
        defaults.update(overrides)
        return VllmAsyncConfig(**defaults)

    def test_model_property_returns_vllm_async_model(self) -> None:
        """Model property should return a _VllmAsyncModel instance for weight download."""
        config = self._make_config()
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
        )
        assert isinstance(stage.model, _VllmAsyncModel)

    def test_model_id_names_matches_configured_variant(self) -> None:
        """model.model_id_names should contain the resolved HF ID for the configured variant."""
        config = self._make_config(model_variant="nemotron")
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="nemotron",
        )
        assert stage.model.model_id_names == ["nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"]

    def test_resources_declare_gpus(self) -> None:
        """N-actors mode: resources should declare 1.0 CPU + num_gpus GPUs."""
        config = self._make_config(num_gpus=4)
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
        )
        res = stage.resources
        assert res.gpus == 4
        assert res.cpus == 1.0

    def test_resources_include_data_parallel_gpus(self) -> None:
        """Resources should multiply num_gpus by data_parallel_size, with 1.0 CPU."""
        config = self._make_config(num_gpus=2, data_parallel_size=3)
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
        )
        res = stage.resources
        assert res.gpus == 6
        assert res.cpus == 1.0

    def test_resources_single_gpu_single_dp(self) -> None:
        """When data_parallel_size is 1, resources equal num_gpus with 1.0 CPU."""
        config = self._make_config(num_gpus=4, data_parallel_size=1)
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
        )
        res = stage.resources
        assert res.gpus == 4
        assert res.cpus == 1.0

    def test_resources_cpu_always_1(self) -> None:
        """CPU request should always be 1.0 regardless of decode pool size."""
        with patch("os.cpu_count", return_value=200):
            config = self._make_config(num_gpus=8, data_parallel_size=1)
            stage = VllmAsyncCaptionStage(
                serve_config=config,
                model_name="qwen",
            )
            res = stage.resources
            assert res.cpus == 1.0
            assert res.gpus == 8.0

    def test_stage_batch_size_auto_multi_gpu(self) -> None:
        """Auto-derived stage_batch_size should be max(3*4, 8) = 12 for 4 GPUs."""
        config = self._make_config(num_gpus=2, data_parallel_size=2)
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
        )
        assert stage.stage_batch_size == 12

    def test_stage_batch_size_auto_single_gpu(self) -> None:
        """N-actors mode (dp=1): stage_batch_size should be 1."""
        config = self._make_config(num_gpus=1, data_parallel_size=1)
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
        )
        assert stage.stage_batch_size == 1

    def test_stage_batch_size_auto_three_gpu(self) -> None:
        """Auto-derived stage_batch_size for 3 GPUs: max(9, 8) = 9."""
        config = self._make_config(num_gpus=1, data_parallel_size=3)
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
        )
        assert stage.stage_batch_size == 9

    def test_stage_batch_size_auto_seven_gpu(self) -> None:
        """Auto-derived stage_batch_size for 7 GPUs: max(21, 8) = 21."""
        config = self._make_config(num_gpus=1, data_parallel_size=7)
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
        )
        assert stage.stage_batch_size == 21

    def test_effective_max_concurrent_requests_auto(self) -> None:
        """Auto-derived concurrency should be 256 * total_gpus in DP mode."""
        config = self._make_config(num_gpus=2, data_parallel_size=2)
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
        )
        assert stage._effective_max_concurrent_requests == 256 * 4

    def test_effective_max_concurrent_requests_single_gpu(self) -> None:
        """Single GPU should get N_ACTORS_SEMAPHORE_LIMIT (256)."""
        config = self._make_config(num_gpus=1, data_parallel_size=1)
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
        )
        assert stage._effective_max_concurrent_requests == 256

    def test_effective_max_concurrent_requests_explicit(self) -> None:
        """Explicit positive value should override auto-derivation."""
        config = self._make_config(num_gpus=4, data_parallel_size=2)
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
            max_concurrent_requests=32,
        )
        assert stage._effective_max_concurrent_requests == 32

    def test_stage_batch_size_explicit_value_used(self) -> None:
        """When stage_batch_size > 0, it should be returned as-is."""
        config = self._make_config(num_gpus=2, data_parallel_size=3)
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
            stage_batch_size=10,
        )
        assert stage.stage_batch_size == 10

    def test_conda_env_is_unified(self) -> None:
        """Conda_env_name should return 'unified'."""
        config = self._make_config()
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
        )
        assert stage.conda_env_name == "unified"

    def test_env_info_returns_pixi_runtime(self) -> None:
        """env_info should return a RuntimeEnv with conda_env_name='unified'."""
        config = self._make_config()
        stage = VllmAsyncCaptionStage(serve_config=config, model_name="qwen")
        runtime = stage.env_info
        assert runtime is not None
        assert runtime.conda is not None
        assert runtime.conda.name == "unified"

    def test_secondary_name(self) -> None:
        """Secondary_name should return 'vllm_async'."""
        config = self._make_config()
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
        )
        assert stage.secondary_name() == "vllm_async"

    def test_destroy_shuts_down_engine(self) -> None:
        """Destroy should call shutdown() on the AsyncLLM engine and gpu_stage_cleanup."""
        config = self._make_config()
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
        )

        mock_engine = MagicMock()
        stage._engine = mock_engine

        with patch.object(vllm_async_stage, "gpu_stage_cleanup") as mock_cleanup:
            stage.destroy()

        mock_engine.shutdown.assert_called_once()
        assert stage._engine is None
        mock_cleanup.assert_called_once_with("VllmAsyncCaptionStage")

    def test_destroy_noop_when_no_engine(self) -> None:
        """Destroy should skip shutdown when no engine exists but still call gpu_stage_cleanup."""
        config = self._make_config()
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
        )

        with patch.object(vllm_async_stage, "gpu_stage_cleanup") as mock_cleanup:
            stage.destroy()

        mock_cleanup.assert_called_once_with("VllmAsyncCaptionStage")

    def test_stage_setup_creates_engine(self) -> None:
        """stage_setup should create AsyncLLM engine."""
        config = self._make_config()
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
        )

        mock_engine = MagicMock()
        mock_async_llm_cls = MagicMock()
        mock_async_llm_cls.from_engine_args.return_value = mock_engine

        with (
            patch.object(vllm_async_stage, "get_vllm_model_id", return_value="Qwen/Qwen2.5-VL-7B-Instruct"),
            patch.object(vllm_async_stage, "resolve_model_path", return_value="/config/models/Qwen"),
            patch.object(vllm_async_stage, "_build_engine_args", return_value=MagicMock()) as mock_build,
            patch.object(vllm_async_stage, "AsyncLLM", mock_async_llm_cls, create=True),
            patch.object(vllm_async_stage, "gpu_stage_startup") as mock_startup,
            patch.object(vllm_async_stage, "build_sampling_params", return_value=MagicMock(), create=True),
        ):
            stage.stage_setup()

        mock_build.assert_called_once()
        mock_async_llm_cls.from_engine_args.assert_called_once()
        assert stage._engine is mock_engine
        assert mock_startup.call_args_list == [
            call("VllmAsyncCaptionStage", 2.0, pre_setup=True),
            call("VllmAsyncCaptionStage", 2.0, pre_setup=False),
        ]

    def test_stage_setup_engine_init_failure_propagates(self) -> None:
        """stage_setup should propagate exception when AsyncLLM.from_engine_args() raises."""
        config = self._make_config()
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
        )

        mock_async_llm_cls = MagicMock()
        mock_async_llm_cls.from_engine_args.side_effect = RuntimeError("CUBLAS_STATUS_INVALID_VALUE")

        with (
            patch.object(vllm_async_stage, "get_vllm_model_id", return_value="Qwen/Qwen2.5-VL-7B-Instruct"),
            patch.object(vllm_async_stage, "resolve_model_path", return_value="/config/models/Qwen"),
            patch.object(vllm_async_stage, "_build_engine_args", return_value=MagicMock()),
            patch.object(vllm_async_stage, "AsyncLLM", mock_async_llm_cls, create=True),
            patch.object(vllm_async_stage, "gpu_stage_startup"),
            pytest.raises(RuntimeError, match="CUBLAS_STATUS_INVALID_VALUE"),
        ):
            stage.stage_setup()

    def test_generate_caption_async_raises_on_no_engine(self) -> None:
        """_generate_caption_async should raise RuntimeError if engine is None."""
        config = self._make_config()
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
        )

        with pytest.raises(RuntimeError, match="AsyncLLM engine not initialized"):
            asyncio.run(
                stage._generate_caption_async(
                    rendered_prompt=MagicMock(),
                    sampling_params=MagicMock(),
                    frames_shape=(4, 224, 224, 3),
                    clip_source="test.mp4",
                    window_index=0,
                )
            )

    def test_generate_caption_async_raises_on_no_outputs(self) -> None:
        """_generate_caption_async should raise RuntimeError when engine yields nothing."""
        config = self._make_config()
        stage = VllmAsyncCaptionStage(serve_config=config, model_name="qwen")

        mock_engine = MagicMock()

        async def _empty_generate(**_kwargs: object) -> AsyncIterator:
            return
            yield

        mock_engine.generate = _empty_generate
        stage._engine = mock_engine

        with pytest.raises(RuntimeError, match="AsyncLLM engine returned no outputs"):
            asyncio.run(
                stage._generate_caption_async(
                    rendered_prompt=MagicMock(),
                    sampling_params=MagicMock(),
                    frames_shape=(4, 224, 224, 3),
                    clip_source="test.mp4",
                    window_index=0,
                )
            )

    def test_generate_caption_async_raises_on_empty_outputs_list(self) -> None:
        """_generate_caption_async should raise RuntimeError when outputs list is empty."""
        config = self._make_config()
        stage = VllmAsyncCaptionStage(serve_config=config, model_name="qwen")

        mock_engine = MagicMock()
        final_output = MagicMock()
        final_output.outputs = []

        async def _generate_empty_outputs(**_kwargs: object) -> AsyncIterator:
            yield final_output

        mock_engine.generate = _generate_empty_outputs
        stage._engine = mock_engine

        with pytest.raises(RuntimeError, match="AsyncLLM engine returned no outputs"):
            asyncio.run(
                stage._generate_caption_async(
                    rendered_prompt=MagicMock(),
                    sampling_params=MagicMock(),
                    frames_shape=(4, 224, 224, 3),
                    clip_source="test.mp4",
                    window_index=0,
                )
            )

    def test_generate_caption_async_raises_on_empty_caption(self) -> None:
        """_generate_caption_async should raise RuntimeError when caption text is whitespace-only."""
        config = self._make_config()
        stage = VllmAsyncCaptionStage(serve_config=config, model_name="qwen")

        mock_engine = MagicMock()
        out0 = MagicMock()
        out0.text = "   "
        out0.finish_reason = "length"
        out0.token_ids = [1, 2, 3]
        out0.cumulative_logprob = -0.5
        final_output = MagicMock()
        final_output.outputs = [out0]
        final_output.prompt_token_ids = [10, 20, 30, 40]

        async def _generate_whitespace(**_kwargs: object) -> AsyncIterator:
            yield final_output

        mock_engine.generate = _generate_whitespace
        stage._engine = mock_engine

        sampling = MagicMock()
        sampling.min_tokens = 1

        with pytest.raises(RuntimeError, match="AsyncLLM engine returned empty caption"):
            asyncio.run(
                stage._generate_caption_async(
                    rendered_prompt=MagicMock(),
                    sampling_params=sampling,
                    frames_shape=(4, 224, 224, 3),
                    clip_source="test.mp4",
                    window_index=0,
                )
            )

    def test_generate_caption_async_returns_stripped_text(self) -> None:
        """_generate_caption_async should return the stripped caption text on success."""
        config = self._make_config()
        stage = VllmAsyncCaptionStage(serve_config=config, model_name="qwen")

        mock_engine = MagicMock()
        out0 = MagicMock()
        out0.text = "  A beautiful sunset over the ocean.  \n"
        out0.token_ids = [1, 2, 3, 4, 5]
        final_output = MagicMock()
        final_output.outputs = [out0]
        final_output.prompt_token_ids = [10, 20, 30]

        async def _generate_ok(**_kwargs: object) -> AsyncIterator:
            yield final_output

        mock_engine.generate = _generate_ok
        stage._engine = mock_engine

        caption, tc = asyncio.run(
            stage._generate_caption_async(
                rendered_prompt=MagicMock(),
                sampling_params=MagicMock(),
                frames_shape=(4, 224, 224, 3),
                clip_source="test.mp4",
                window_index=0,
            )
        )
        assert caption == "A beautiful sunset over the ocean."
        assert tc.prompt_tokens == 3
        assert tc.output_tokens == 5


class TestVllmAsyncConfig:
    """Tests for VllmAsyncConfig data class."""

    def test_frozen(self) -> None:
        """Config should be immutable (frozen=True)."""
        config = VllmAsyncConfig(model_variant="test/model")
        with pytest.raises(AttributeError):
            config.model_variant = "other/model"  # type: ignore[misc]

    def test_defaults(self) -> None:
        """Verify default values for the in-process engine config."""
        config = VllmAsyncConfig(model_variant="test/model")
        assert config.num_gpus == 1.0
        assert config.gpu_memory_utilization == 0.85
        assert config.max_model_len == 0
        assert config.dtype == "auto"
        assert config.quantization is None
        assert config.max_num_batched_tokens == 0
        assert config.max_num_seqs == 0
        assert config.enforce_eager is False
        assert config.cudagraph_mode == "piecewise"
        limit_mm = json.loads(config.limit_mm_per_prompt)
        assert limit_mm == {"image": 0, "video": 1}
        assert config.mm_encoder_tp_mode == "data"
        assert config.kv_cache_dtype == "auto"
        assert config.mm_processor_cache_gb == 4.0
        assert config.mm_processor_cache_type == ""
        assert config.trust_remote_code is True
        assert config.data_parallel_size == 1
        assert config.enable_log_requests is False
        assert config.sampling_config == VllmSamplingConfig()
        assert config.async_scheduling is None
        assert config.enable_chunked_prefill is None
        assert config.disable_chunked_mm_input is False
        assert config.long_prefill_token_threshold == 0
        assert config.stream_interval == 9999
        assert config.distributed_executor_backend == "ray"
        assert config.skip_mm_profiling is True
        mm_kwargs = json.loads(config.mm_processor_kwargs)
        assert mm_kwargs == {"max_pixels": 602112}
        assert config.extra_env_vars == ""

    def test_total_gpus_single_gpu(self) -> None:
        """total_gpus should equal num_gpus when data_parallel_size is 1."""
        config = VllmAsyncConfig(model_variant="test/model", num_gpus=4.0)
        assert config.total_gpus == 4.0

    def test_total_gpus_with_data_parallel(self) -> None:
        """total_gpus should multiply num_gpus by data_parallel_size."""
        config = VllmAsyncConfig(model_variant="test/model", num_gpus=2.0, data_parallel_size=3)
        assert config.total_gpus == 6.0

    def test_validation_async_scheduling_with_ray_raises(self) -> None:
        """async_scheduling=True + distributed_executor_backend='ray' should raise ValueError."""
        with pytest.raises(ValueError, match="async_scheduling=True requires"):
            VllmAsyncConfig(
                model_variant="test/model",
                async_scheduling=True,
                distributed_executor_backend="ray",
            )

    def test_validation_async_scheduling_with_mp_ok(self) -> None:
        """async_scheduling=True + distributed_executor_backend='mp' should succeed."""
        config = VllmAsyncConfig(
            model_variant="test/model",
            async_scheduling=True,
            distributed_executor_backend="mp",
        )
        assert config.async_scheduling is True

    def test_validation_async_scheduling_with_uni_ok(self) -> None:
        """async_scheduling=True + distributed_executor_backend='uni' should succeed."""
        config = VllmAsyncConfig(
            model_variant="test/model",
            async_scheduling=True,
            distributed_executor_backend="uni",
        )
        assert config.async_scheduling is True

    def test_async_scheduling_none_is_valid_with_any_backend(self) -> None:
        """async_scheduling=None (auto-detect) should not raise with any backend."""
        for backend in ("ray", "mp", "uni"):
            config = VllmAsyncConfig(
                model_variant="test/model",
                async_scheduling=None,
                distributed_executor_backend=backend,
            )
            assert config.async_scheduling is None

    def test_validation_num_gpus_below_one_raises(self) -> None:
        """num_gpus < 1.0 should raise ValueError at construction time."""
        with pytest.raises(ValueError, match=r"num_gpus must be >= 1\.0"):
            VllmAsyncConfig(model_variant="test/model", num_gpus=0.5)

    def test_validation_invalid_json_limit_mm_raises(self) -> None:
        """Invalid JSON in limit_mm_per_prompt should raise ValueError."""
        with pytest.raises(ValueError, match="limit_mm_per_prompt must be valid JSON"):
            VllmAsyncConfig(model_variant="test/model", limit_mm_per_prompt="not-json")

    def test_validation_empty_limit_mm_ok(self) -> None:
        """Empty string for limit_mm_per_prompt should pass validation (falsy skip)."""
        config = VllmAsyncConfig(model_variant="test/model", limit_mm_per_prompt="")
        assert config.limit_mm_per_prompt == ""

    @contextlib.contextmanager
    def _patch_engine(self) -> Generator[tuple[MagicMock, MagicMock], None, None]:
        """Patch ``AsyncEngineArgs`` and ``CompilationConfig`` on the module.

        Mirrors :meth:`TestBuildEngineArgs._patch_engine` so that
        tests calling ``_build_engine_args`` can run on plain CPU.
        """
        mock_engine_args_cls = MagicMock()
        mock_comp_config_cls = MagicMock()
        with (
            patch.object(vllm_async_stage, "AsyncEngineArgs", mock_engine_args_cls, create=True),
            patch.object(vllm_async_stage, "CompilationConfig", mock_comp_config_cls, create=True),
        ):
            yield mock_engine_args_cls, mock_comp_config_cls

    def test_skip_mm_profiling_wired_to_engine_args(self) -> None:
        """skip_mm_profiling should be passed through to AsyncEngineArgs."""
        with self._patch_engine() as (mock_engine_args_cls, _):
            config = VllmAsyncConfig(model_variant="test/model", skip_mm_profiling=True)
            _build_engine_args(config, "/tmp/model")  # noqa: S108
        assert mock_engine_args_cls.call_args.kwargs["skip_mm_profiling"] is True

    def test_mm_processor_kwargs_valid_json(self) -> None:
        """Valid JSON for mm_processor_kwargs should pass validation."""
        config = VllmAsyncConfig(model_variant="test/model", mm_processor_kwargs='{"max_pixels": 100000}')
        parsed = json.loads(config.mm_processor_kwargs)
        assert parsed == {"max_pixels": 100000}

    def test_mm_processor_kwargs_invalid_json_raises(self) -> None:
        """Invalid JSON in mm_processor_kwargs should raise ValueError."""
        with pytest.raises(ValueError, match="mm_processor_kwargs must be valid JSON"):
            VllmAsyncConfig(model_variant="test/model", mm_processor_kwargs="not-json")

    def test_mm_processor_kwargs_empty_string_ok(self) -> None:
        """Empty string for mm_processor_kwargs should pass validation (falsy skip)."""
        config = VllmAsyncConfig(model_variant="test/model", mm_processor_kwargs="")
        assert config.mm_processor_kwargs == ""

    def test_mm_processor_kwargs_wired_to_engine_args(self) -> None:
        """mm_processor_kwargs JSON should be parsed to a dict in AsyncEngineArgs."""
        with self._patch_engine() as (mock_engine_args_cls, _):
            config = VllmAsyncConfig(model_variant="test/model", mm_processor_kwargs='{"max_pixels": 602112}')
            _build_engine_args(config, "/tmp/model")  # noqa: S108
        assert mock_engine_args_cls.call_args.kwargs["mm_processor_kwargs"] == {"max_pixels": 602112}

    def test_mm_processor_kwargs_empty_wired_as_none(self) -> None:
        """Empty mm_processor_kwargs should be wired as None to AsyncEngineArgs."""
        with self._patch_engine() as (mock_engine_args_cls, _):
            config = VllmAsyncConfig(model_variant="test/model", mm_processor_kwargs="")
            _build_engine_args(config, "/tmp/model")  # noqa: S108
        assert mock_engine_args_cls.call_args.kwargs["mm_processor_kwargs"] is None

    def test_limit_mm_per_prompt_configurable_format(self) -> None:
        """The resolution-constrained limit_mm_per_prompt format should be accepted."""
        constrained = '{"image": 0, "video": {"count": 1, "num_frames": 768, "width": 784, "height": 784}}'
        config = VllmAsyncConfig(model_variant="test/model", limit_mm_per_prompt=constrained)
        parsed = json.loads(config.limit_mm_per_prompt)
        assert parsed["video"]["count"] == 1
        assert parsed["video"]["num_frames"] == 768
        assert parsed["video"]["width"] == 784

    def test_extra_env_vars_default_empty(self) -> None:
        """Default extra_env_vars should be an empty string."""
        config = VllmAsyncConfig(model_variant="test/model")
        assert config.extra_env_vars == ""

    def test_extra_env_vars_valid_json(self) -> None:
        """Valid JSON dict with string keys and values should pass validation."""
        env_json = '{"CUDA_LAUNCH_BLOCKING": "1", "NCCL_DEBUG": "TRACE"}'
        config = VllmAsyncConfig(model_variant="test/model", extra_env_vars=env_json)
        parsed = json.loads(config.extra_env_vars)
        assert parsed == {"CUDA_LAUNCH_BLOCKING": "1", "NCCL_DEBUG": "TRACE"}

    def test_extra_env_vars_empty_string_ok(self) -> None:
        """Empty string for extra_env_vars should pass validation (falsy skip)."""
        config = VllmAsyncConfig(model_variant="test/model", extra_env_vars="")
        assert config.extra_env_vars == ""

    def test_extra_env_vars_invalid_json_raises(self) -> None:
        """Invalid JSON in extra_env_vars should raise ValueError."""
        with pytest.raises(ValueError, match="extra_env_vars must be valid JSON"):
            VllmAsyncConfig(model_variant="test/model", extra_env_vars="not-json")

    def test_extra_env_vars_non_dict_raises(self) -> None:
        """A JSON array instead of an object should raise TypeError."""
        with pytest.raises(TypeError, match="extra_env_vars must be a JSON object"):
            VllmAsyncConfig(model_variant="test/model", extra_env_vars='["a", "b"]')

    def test_extra_env_vars_non_string_value_raises(self) -> None:
        """Non-string values in the JSON dict should raise TypeError."""
        with pytest.raises(TypeError, match="extra_env_vars values must be strings"):
            VllmAsyncConfig(model_variant="test/model", extra_env_vars='{"KEY": 123}')

    def test_no_subprocess_fields(self) -> None:
        """Verify that legacy subprocess fields are not present."""
        config = VllmAsyncConfig(model_variant="test/model")
        assert not hasattr(config, "startup_timeout_s")
        assert not hasattr(config, "api_key")
        assert not hasattr(config, "enable_tracing")
        assert not hasattr(config, "otlp_traces_endpoint")
        assert not hasattr(config, "extra_args")
        assert not hasattr(config, "api_server_count")
        assert not hasattr(config, "video_pruning_rate")


class TestBuildVllmAsyncConfig:
    """Tests for build_vllm_async_config() CLI-to-config builder."""

    def _make_args(self, **overrides: object) -> argparse.Namespace:
        """Build a minimal argparse.Namespace mimicking real CLI behaviour.

        All ``--vllm-async-*`` optional fields default to ``None`` (sentinel),
        matching the argparse definitions in ``add_vllm_async_cli_args``.
        Tests that need explicit CLI values should pass them as overrides.
        """
        defaults: dict[str, object] = {
            "generate_captions": True,
            "captioning_algorithm": "vllm_async",
            "vllm_async_model_name": "qwen",
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def _default_sampling_config(self) -> VllmSamplingConfig:
        return VllmSamplingConfig()

    def test_returns_none_for_non_vllm_async(self) -> None:
        """Should return None when captioning_algorithm is not vllm_async."""
        args = self._make_args(captioning_algorithm="qwen")
        assert build_vllm_async_config(args, sampling_config=self._default_sampling_config()) is None

    def test_returns_config_for_vllm_async(self) -> None:
        """Should return a VllmAsyncConfig with explicit CLI values."""
        args = self._make_args(
            vllm_async_num_gpus=2.0,
            vllm_async_gpu_memory_utilization=0.9,
            vllm_async_max_model_len=4096,
        )
        config = build_vllm_async_config(args, sampling_config=self._default_sampling_config())

        assert config is not None
        assert config.model_variant == "qwen"
        assert config.num_gpus == 2.0
        assert config.gpu_memory_utilization == 0.9
        assert config.max_model_len == 4096

    def test_model_defaults_applied_when_no_cli_override(self) -> None:
        """When CLI does not set a field, _MODEL_DEFAULTS for that model apply."""
        args = self._make_args(vllm_async_model_name="qwen")
        config = build_vllm_async_config(args, sampling_config=self._default_sampling_config())

        assert config is not None
        assert config.model_variant == "qwen"
        # Qwen model defaults: max_model_len=32768, max_num_batched_tokens=32768.
        # gpu_memory_utilization, kv_cache_dtype, quantization use VllmAsyncConfig
        # field defaults (0.85, "auto", None).
        assert config.gpu_memory_utilization == 0.85
        assert config.kv_cache_dtype == "auto"
        assert config.quantization is None
        assert config.max_num_batched_tokens == 32768
        assert config.skip_mm_profiling is True
        assert json.loads(config.mm_processor_kwargs) == {
            "videos_kwargs": {
                "size": {
                    "shortest_edge": VIDEO_MIN_PIXELS,
                    "longest_edge": VIDEO_MAX_PIXELS,
                },
            },
        }

    def test_non_qwen_uses_base_mm_processor_kwargs_default(self) -> None:
        """Non-qwen variants get the base flat max_pixels default, not qwen's nested videos_kwargs.size."""
        args = self._make_args(vllm_async_model_name="nemotron")
        config = build_vllm_async_config(args, sampling_config=self._default_sampling_config())

        assert config is not None
        assert json.loads(config.mm_processor_kwargs) == {"max_pixels": VIDEO_MAX_PIXELS}

    def test_cli_overrides_model_default(self) -> None:
        """Explicit CLI value should override _MODEL_DEFAULTS for that model."""
        args = self._make_args(
            vllm_async_model_name="nemotron",
            vllm_async_gpu_memory_utilization=0.8,
        )
        config = build_vllm_async_config(args, sampling_config=self._default_sampling_config())

        assert config is not None
        assert config.gpu_memory_utilization == 0.8

    def test_case_insensitive_algorithm_match(self) -> None:
        """Should match 'Vllm_Async' (case-insensitive)."""
        args = self._make_args(captioning_algorithm="Vllm_Async")
        assert build_vllm_async_config(args, sampling_config=self._default_sampling_config()) is not None

    def test_returns_none_when_captions_disabled(self) -> None:
        """Should return None when generate_captions is False, even if algo is vllm_async."""
        args = self._make_args(generate_captions=False)
        assert build_vllm_async_config(args, sampling_config=self._default_sampling_config()) is None

    def test_new_tier1_fields_wired(self) -> None:
        """New Tier 1 fields should be wired from CLI args."""
        args = self._make_args(
            vllm_async_dtype="bfloat16",
            vllm_async_quantization="fp8",
            vllm_async_max_num_batched_tokens=32768,
            vllm_async_max_num_seqs=32,
            vllm_async_enforce_eager=True,
            vllm_async_limit_mm_per_prompt='{"video": 1}',
        )
        config = build_vllm_async_config(args, sampling_config=self._default_sampling_config())

        assert config is not None
        assert config.dtype == "bfloat16"
        assert config.quantization == "fp8"
        assert config.max_num_batched_tokens == 32768
        assert config.max_num_seqs == 32
        assert config.enforce_eager is True
        assert config.limit_mm_per_prompt == '{"video": 1}'

    def test_data_parallel_size_wired(self) -> None:
        """data_parallel_size should be wired from CLI args."""
        args = self._make_args(vllm_async_data_parallel_size=4)
        config = build_vllm_async_config(args, sampling_config=self._default_sampling_config())

        assert config is not None
        assert config.data_parallel_size == 4

    def test_data_parallel_size_defaults_to_one(self) -> None:
        """When CLI arg is absent, data_parallel_size should default to 1."""
        args = self._make_args()
        config = build_vllm_async_config(args, sampling_config=self._default_sampling_config())

        assert config is not None
        assert config.data_parallel_size == 1

    def test_cudagraph_mode_wired(self) -> None:
        """cudagraph_mode should be wired from CLI args."""
        args = self._make_args(vllm_async_cudagraph_mode="full")
        config = build_vllm_async_config(args, sampling_config=self._default_sampling_config())

        assert config is not None
        assert config.cudagraph_mode == "full"

    def test_cudagraph_mode_defaults_to_piecewise(self) -> None:
        """When CLI arg is absent, cudagraph_mode should default to 'piecewise'."""
        args = self._make_args()
        config = build_vllm_async_config(args, sampling_config=self._default_sampling_config())

        assert config is not None
        assert config.cudagraph_mode == "piecewise"

    def test_enable_log_requests_wired_from_verbose(self) -> None:
        """enable_log_requests should mirror args.verbose."""
        args = self._make_args(verbose=True)
        config = build_vllm_async_config(args, sampling_config=self._default_sampling_config())

        assert config is not None
        assert config.enable_log_requests is True

    def test_enable_log_requests_false_when_not_verbose(self) -> None:
        """enable_log_requests should be False when verbose is absent."""
        args = self._make_args()
        config = build_vllm_async_config(args, sampling_config=self._default_sampling_config())

        assert config is not None
        assert config.enable_log_requests is False

    def test_extra_env_vars_wired_from_cli(self) -> None:
        """--vllm-async-extra-env-vars should flow through to VllmAsyncConfig."""
        env_json = '{"CUDA_LAUNCH_BLOCKING": "1"}'
        args = self._make_args(vllm_async_extra_env_vars=env_json)
        config = build_vllm_async_config(args, sampling_config=self._default_sampling_config())

        assert config is not None
        assert config.extra_env_vars == env_json

    def test_extra_env_vars_defaults_to_empty(self) -> None:
        """When CLI does not set extra_env_vars, it should default to empty string."""
        args = self._make_args()
        config = build_vllm_async_config(args, sampling_config=self._default_sampling_config())

        assert config is not None
        assert config.extra_env_vars == ""

    def test_empty_string_quantization_treated_as_none(self) -> None:
        """Empty string quantization should resolve to None (attrs default)."""
        args = self._make_args(
            vllm_async_model_name="qwen",
            vllm_async_quantization="",
        )
        config = build_vllm_async_config(args, sampling_config=self._default_sampling_config())

        assert config is not None
        assert config.quantization is None

    def test_empty_string_kv_cache_dtype_uses_attrs_default(self) -> None:
        """Empty string kv_cache_dtype should fall through to attrs default 'auto'."""
        args = self._make_args(
            vllm_async_model_name="qwen",
            vllm_async_kv_cache_dtype="",
        )
        config = build_vllm_async_config(args, sampling_config=self._default_sampling_config())

        assert config is not None
        assert config.kv_cache_dtype == "auto"

    def test_none_quantization_uses_model_default(self) -> None:
        """None (not provided) quantization should use qwen model default (None)."""
        args = self._make_args(vllm_async_model_name="qwen")
        config = build_vllm_async_config(args, sampling_config=self._default_sampling_config())

        assert config is not None
        assert config.quantization is None

    def test_sampling_config_passed_through(self) -> None:
        """sampling_config should be embedded in the resulting VllmAsyncConfig."""
        custom_sc = VllmSamplingConfig(temperature=0.3, min_tokens=16)
        args = self._make_args()
        config = build_vllm_async_config(args, sampling_config=custom_sc)

        assert config is not None
        assert config.sampling_config == custom_sc
        assert config.sampling_config.temperature == 0.3
        assert config.sampling_config.min_tokens == 16


class TestConfigureVllmEnvironment:
    """Tests for _configure_vllm_environment() -- mirrors env_info vars into os.environ.

    Uses ``patch.dict(os.environ)`` to automatically restore the full
    environment after each test.
    """

    def _make_stage(self, **config_overrides: object) -> VllmAsyncCaptionStage:
        defaults: dict[str, object] = {
            "model_variant": "qwen",
            "num_gpus": 2,
        }
        defaults.update(config_overrides)
        config = VllmAsyncConfig(**defaults)
        return VllmAsyncCaptionStage(serve_config=config, model_name="qwen")

    def test_mirrors_env_info_vars_into_os_environ(self) -> None:
        """All non-empty env_info vars should be mirrored into os.environ."""
        stage = self._make_stage()

        with patch.dict(os.environ, clear=False):
            stage._configure_vllm_environment()
            assert os.environ["VLLM_LOGGING_LEVEL"] == "INFO"

    def test_extra_env_vars_applied_to_os_environ(self) -> None:
        """Extra env vars from config should be set in os.environ."""
        env_json = '{"MY_TEST_VAR_A": "hello", "MY_TEST_VAR_B": "world"}'
        stage = self._make_stage(extra_env_vars=env_json)

        with patch.dict(os.environ, clear=False):
            stage._configure_vllm_environment()
            assert os.environ["MY_TEST_VAR_A"] == "hello"
            assert os.environ["MY_TEST_VAR_B"] == "world"

    def test_extra_env_vars_override_builtin(self) -> None:
        """Extra env vars should override built-in defaults."""
        env_json = '{"VLLM_LOGGING_LEVEL": "DEBUG"}'
        stage = self._make_stage(extra_env_vars=env_json)
        stage._verbose = False

        with patch.dict(os.environ, clear=False):
            stage._configure_vllm_environment()
            assert os.environ["VLLM_LOGGING_LEVEL"] == "DEBUG"

    def test_stale_vars_popped_from_os_environ(self) -> None:
        """Stale env vars inherited from the Dockerfile should be removed."""
        stage = self._make_stage()

        with patch.dict(os.environ, {"VLLM_USE_V1": "1", "VLLM_ATTENTION_BACKEND": "FLASHINFER"}, clear=False):
            stage._configure_vllm_environment()
            assert "VLLM_USE_V1" not in os.environ
            assert "VLLM_ATTENTION_BACKEND" not in os.environ


class TestEnvInfoEnvVars:
    """Tests for env var dict built inside env_info -- complete env var set for the Ray worker."""

    def _make_stage(self, *, verbose: bool = False, **config_overrides: object) -> VllmAsyncCaptionStage:
        defaults: dict[str, object] = {
            "model_variant": "qwen",
            "num_gpus": 2,
        }
        defaults.update(config_overrides)
        config = VllmAsyncConfig(**defaults)
        return VllmAsyncCaptionStage(serve_config=config, model_name="qwen", verbose=verbose)

    def _env_vars(self, *, verbose: bool = False, **config_overrides: object) -> dict[str, str]:
        stage = self._make_stage(verbose=verbose, **config_overrides)
        env = stage.env_info
        assert env is not None
        return dict(env.extra_env_vars)

    def test_verbose_sets_debug_level(self) -> None:
        """verbose=True should set VLLM_LOGGING_LEVEL=DEBUG."""
        env = self._env_vars(verbose=True)
        assert env["VLLM_LOGGING_LEVEL"] == "DEBUG"

    def test_non_verbose_sets_info_level(self) -> None:
        """verbose=False should set VLLM_LOGGING_LEVEL=INFO."""
        env = self._env_vars(verbose=False)
        assert env["VLLM_LOGGING_LEVEL"] == "INFO"

    def test_logging_prefix_set(self) -> None:
        """VLLM_LOGGING_PREFIX should be set from log_tag."""
        env = self._env_vars()
        assert env["VLLM_LOGGING_PREFIX"] == "[asyncvLLM:qwen] "

    def test_tqdm_suppressed_when_not_verbose(self) -> None:
        """TQDM_DISABLE=1 when verbose=False."""
        env = self._env_vars(verbose=False)
        assert env["TQDM_DISABLE"] == "1"

    def test_tqdm_not_set_when_verbose(self) -> None:
        """TQDM_DISABLE should not be set when verbose=True."""
        env = self._env_vars(verbose=True)
        assert "TQDM_DISABLE" not in env

    def test_cache_root_set_to_tmp(self) -> None:
        """VLLM_CACHE_ROOT should default to /tmp/vllm for fast local storage."""
        env = self._env_vars()
        assert env["VLLM_CACHE_ROOT"] == "/tmp/vllm"  # noqa: S108

    def test_cache_root_overridable_by_extra_env_vars(self) -> None:
        """User extra_env_vars should be able to override VLLM_CACHE_ROOT."""
        env = self._env_vars(extra_env_vars='{"VLLM_CACHE_ROOT": "/mnt/fast/vllm"}')
        assert env["VLLM_CACHE_ROOT"] == "/mnt/fast/vllm"

    def test_otel_not_in_env_info(self) -> None:
        """OTEL_SDK_DISABLED is set globally in profiling_scope, not env_info."""
        env = self._env_vars()
        assert "OTEL_SDK_DISABLED" not in env

    def test_stale_vars_set_to_empty_string(self) -> None:
        """Stale VLLM vars should be set to empty string to unset them."""
        env = self._env_vars()
        for var in VllmAsyncCaptionStage._UNSET_VLLM_ENV_VARS:
            assert env[var] == "", f"{var} should be empty string"

    def test_extra_env_vars_all_included(self) -> None:
        """All extra_env_vars (VLLM_* and non-VLLM_*) should be included."""
        env = self._env_vars(
            extra_env_vars='{"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "CUDA_LAUNCH_BLOCKING": "1"}',
        )
        assert env["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
        assert env["CUDA_LAUNCH_BLOCKING"] == "1"

    def test_extra_env_vars_override_builtins(self) -> None:
        """User extra_env_vars should override built-in defaults."""
        env = self._env_vars(verbose=False, extra_env_vars='{"VLLM_LOGGING_LEVEL": "DEBUG"}')
        assert env["VLLM_LOGGING_LEVEL"] == "DEBUG"

    def test_no_extra_env_vars(self) -> None:
        """Empty extra_env_vars should not add any user keys."""
        env = self._env_vars(extra_env_vars="")
        assert "VLLM_ENABLE_V1_MULTIPROCESSING" not in env
        assert "CUDA_LAUNCH_BLOCKING" not in env


class TestEnvInfoProperty:
    """Tests for env_info property -- env var propagation via Ray runtime env."""

    def _make_stage(self, **config_overrides: object) -> VllmAsyncCaptionStage:
        defaults: dict[str, object] = {
            "model_variant": "qwen",
            "num_gpus": 2,
        }
        defaults.update(config_overrides)
        config = VllmAsyncConfig(**defaults)
        return VllmAsyncCaptionStage(serve_config=config, model_name="qwen")

    def test_env_info_returns_runtime_env(self) -> None:
        """env_info should return a RuntimeEnv (not None)."""
        stage = self._make_stage()
        env = stage.env_info
        assert env is not None

    def test_env_info_propagates_all_extra_env_vars(self) -> None:
        """All extra_env_vars should appear in env_info.extra_env_vars."""
        stage = self._make_stage(
            extra_env_vars='{"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "CUDA_LAUNCH_BLOCKING": "1"}',
        )
        env = stage.env_info
        assert env is not None
        assert env.extra_env_vars.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0"
        assert env.extra_env_vars.get("CUDA_LAUNCH_BLOCKING") == "1"

    def test_env_info_has_conda_env(self) -> None:
        """env_info should have the 'unified' conda env."""
        stage = self._make_stage()
        env = stage.env_info
        assert env is not None
        assert env.conda is not None
        assert env.conda.name == "unified"

    def test_env_info_includes_stale_var_removal(self) -> None:
        """Stale VLLM vars should be present as empty strings."""
        stage = self._make_stage()
        env = stage.env_info
        assert env is not None
        assert env.extra_env_vars.get("VLLM_USE_V1") == ""
        assert env.extra_env_vars.get("VLLM_ATTENTION_BACKEND") == ""


class TestVllmAsyncPrepConfig:
    """Tests for VllmAsyncPrepConfig -- config ownership verification."""

    def test_contains_only_prep_fields(self) -> None:
        """VllmAsyncPrepConfig should have only prep-relevant fields."""
        field_names = {f.name for f in attrs.fields(VllmAsyncPrepConfig)}
        expected = {
            "model_variant",
            "sampling_config",
            "prompt_variant",
            "prompt_text",
            "sample_fps",
            "window_size",
            "remainder_threshold",
            "keep_mp4",
            "use_input_bit_rate",
            "decode_workers",
        }
        assert field_names == expected

    def test_no_engine_gpu_fields(self) -> None:
        """VllmAsyncPrepConfig should not have any engine/GPU fields."""
        field_names = {f.name for f in attrs.fields(VllmAsyncPrepConfig)}
        engine_fields = {
            "num_gpus",
            "data_parallel_size",
            "gpu_memory_utilization",
            "max_concurrent_requests",
            "stage_batch_size",
            "max_model_len",
            "distributed_executor_backend",
            "enforce_eager",
        }
        assert field_names.isdisjoint(engine_fields), f"Leaked engine fields: {field_names & engine_fields}"

    def test_frozen(self) -> None:
        """VllmAsyncPrepConfig should be immutable."""
        cfg = VllmAsyncPrepConfig(model_variant="qwen")
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            cfg.model_variant = "nemotron"  # type: ignore[misc]

    def test_defaults(self) -> None:
        """VllmAsyncPrepConfig should have sensible defaults."""
        cfg = VllmAsyncPrepConfig(model_variant="qwen")
        assert cfg.prompt_variant == "default"
        assert cfg.prompt_text is None
        assert cfg.sample_fps == 2.0
        assert cfg.window_size == 256
        assert cfg.remainder_threshold == 128
        assert cfg.keep_mp4 is False
        assert cfg.use_input_bit_rate is False
        assert cfg.decode_workers == 0


class TestVllmAsyncPrepStage:
    """Tests for VllmAsyncPrepStage -- CPU-only prep stage."""

    def _make_stage(self, **config_overrides: object) -> VllmAsyncPrepStage:
        defaults: dict[str, object] = {"model_variant": "qwen"}
        defaults.update(config_overrides)
        prep_config = VllmAsyncPrepConfig(**defaults)
        return VllmAsyncPrepStage(
            prep_config=prep_config,
        )

    def test_resources_cpu_only(self) -> None:
        """Prep stage should request 0.5 CPU and 0 GPUs."""
        stage = self._make_stage()
        assert stage.resources.cpus == 0.5
        assert stage.resources.gpus == 0

    def test_conda_env_name(self) -> None:
        """Prep stage should use the 'unified' environment."""
        stage = self._make_stage()
        assert stage.conda_env_name == "unified"

    def test_secondary_name(self) -> None:
        """Prep stage should return 'vllm_async' as secondary name."""
        stage = self._make_stage()
        assert stage.secondary_name() == "vllm_async"

    def test_model_returns_vllm_async_model(self) -> None:
        """Prep stage should expose a model for weight download."""
        stage = self._make_stage()
        model = stage.model
        assert model is not None
        assert model.conda_env_name == "unified"
        assert "Qwen/Qwen2.5-VL-7B-Instruct" in model.model_id_names

    def test_create_windows_single_window(self) -> None:
        """_create_windows_and_decode should produce one window for a short clip."""
        stage = self._make_stage()
        stage._prompt_template = "<prompt>describe</prompt>"

        clip = Clip(uuid=uuid4(), source_video="s.mp4", span=(0.0, 1.0), encoded_data=b"\x00\x01\x02")
        fake_frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        window_info = WindowFrameInfo(start=0, end=99)

        with (
            patch.object(vllm_async_stage, "buffer_as_memfd_path") as mock_memfd,
            patch.object(vllm_async_stage, "get_avg_frame_rate", return_value=30.0),
            patch.object(vllm_async_stage, "get_frame_count", return_value=100),
            patch.object(vllm_async_stage, "compute_windows", return_value=[window_info]),
            patch.object(vllm_async_stage, "smart_nframes", return_value=4),
            patch.object(vllm_async_stage, "decode_video_cpu_frame_ids", return_value=fake_frames),
        ):
            mock_memfd.return_value.__enter__ = MagicMock(return_value="/fake/memfd")
            mock_memfd.return_value.__exit__ = MagicMock(return_value=False)
            windows = stage._create_windows_and_decode(clip)

        assert len(windows) == 1
        assert len(clip.windows) == 1
        assert windows[0].start_frame == 0
        assert windows[0].end_frame == 99
        stored = windows[0].model_input["vllm_async"]
        assert stored["prompt"] == "<prompt>describe</prompt>"
        assert stored["video_frames"].shape == (4, 224, 224, 3)
        assert stored["frames_shape"] == (4, 224, 224, 3)
        assert isinstance(stored["frames_shape"], tuple)
        assert "sampling_params" not in stored

    def test_create_windows_multi_window(self) -> None:
        """_create_windows_and_decode should split frames across multiple windows."""
        stage = self._make_stage(window_size=128)
        stage._prompt_template = "<prompt>describe</prompt>"

        clip = Clip(uuid=uuid4(), source_video="s.mp4", span=(0.0, 5.0), encoded_data=b"\x00" * 100)
        n_sampled_per_window = 4
        total_sampled = n_sampled_per_window * 2
        fake_frames = np.arange(total_sampled * 224 * 224 * 3, dtype=np.uint8).reshape(total_sampled, 224, 224, 3)
        window_infos = [WindowFrameInfo(start=0, end=127), WindowFrameInfo(start=128, end=255)]

        with (
            patch.object(vllm_async_stage, "buffer_as_memfd_path") as mock_memfd,
            patch.object(vllm_async_stage, "get_avg_frame_rate", return_value=30.0),
            patch.object(vllm_async_stage, "get_frame_count", return_value=256),
            patch.object(vllm_async_stage, "compute_windows", return_value=window_infos),
            patch.object(vllm_async_stage, "smart_nframes", return_value=n_sampled_per_window),
            patch.object(vllm_async_stage, "decode_video_cpu_frame_ids", return_value=fake_frames),
        ):
            mock_memfd.return_value.__enter__ = MagicMock(return_value="/fake/memfd")
            mock_memfd.return_value.__exit__ = MagicMock(return_value=False)
            windows = stage._create_windows_and_decode(clip)

        assert len(windows) == 2
        assert len(clip.windows) == 2
        assert windows[0].start_frame == 0
        assert windows[1].start_frame == 128
        for w in windows:
            assert "vllm_async" in w.model_input
            assert w.model_input["vllm_async"]["frames_shape"] == (n_sampled_per_window, 224, 224, 3)

    def test_create_windows_empty_encoded_data(self) -> None:
        """_create_windows_and_decode should skip clip with None encoded_data."""
        stage = self._make_stage()
        clip = Clip(uuid=uuid4(), source_video="s.mp4", span=(0.0, 1.0), encoded_data=None)
        windows = stage._create_windows_and_decode(clip)
        assert len(windows) == 0
        assert "encoded_data" in clip.errors

    def test_keep_mp4_extracts_bytes(self) -> None:
        """When keep_mp4=True, MP4 bytes should be extracted per window."""
        stage = self._make_stage(keep_mp4=True)
        stage._prompt_template = "<prompt>describe</prompt>"

        clip = Clip(uuid=uuid4(), source_video="s.mp4", span=(0.0, 1.0), encoded_data=b"\x00\x01\x02")
        fake_frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        window_info = WindowFrameInfo(start=0, end=99)

        fake_mp4_bytes = [b"\xff\x00\x01"]

        with (
            patch.object(vllm_async_stage, "buffer_as_memfd_path") as mock_memfd,
            patch.object(vllm_async_stage, "get_avg_frame_rate", return_value=30.0),
            patch.object(vllm_async_stage, "get_frame_count", return_value=100),
            patch.object(vllm_async_stage, "compute_windows", return_value=[window_info]),
            patch.object(vllm_async_stage, "smart_nframes", return_value=4),
            patch.object(vllm_async_stage, "decode_video_cpu_frame_ids", return_value=fake_frames),
            patch(
                "cosmos_curator.pipelines.video.utils.windowing_utils.split_video_into_windows",
                return_value=(fake_mp4_bytes, [None], [window_info]),
            ),
        ):
            mock_memfd.return_value.__enter__ = MagicMock(return_value="/fake/memfd")
            mock_memfd.return_value.__exit__ = MagicMock(return_value=False)
            windows = stage._create_windows_and_decode(clip)

        assert len(windows) == 1
        assert windows[0].mp4_bytes.resolve().tobytes() == b"\xff\x00\x01"

    def test_create_windows_frame_drop_partial(self) -> None:
        """When PyAV returns fewer frames than expected, windows should get best-effort slices."""
        stage = self._make_stage(window_size=128)
        stage._prompt_template = "<prompt>describe</prompt>"

        clip = Clip(uuid=uuid4(), source_video="s.mp4", span=(0.0, 5.0), encoded_data=b"\x00" * 100)
        n_sampled_per_window = 4
        # PyAV returns only 6 frames instead of 8 (2 windows x 4 each)
        returned_frames = 6
        fake_frames = np.zeros((returned_frames, 224, 224, 3), dtype=np.uint8)
        window_infos = [WindowFrameInfo(start=0, end=127), WindowFrameInfo(start=128, end=255)]

        with (
            patch.object(vllm_async_stage, "buffer_as_memfd_path") as mock_memfd,
            patch.object(vllm_async_stage, "get_avg_frame_rate", return_value=30.0),
            patch.object(vllm_async_stage, "get_frame_count", return_value=256),
            patch.object(vllm_async_stage, "compute_windows", return_value=window_infos),
            patch.object(vllm_async_stage, "smart_nframes", return_value=n_sampled_per_window),
            patch.object(vllm_async_stage, "decode_video_cpu_frame_ids", return_value=fake_frames),
        ):
            mock_memfd.return_value.__enter__ = MagicMock(return_value="/fake/memfd")
            mock_memfd.return_value.__exit__ = MagicMock(return_value=False)
            windows = stage._create_windows_and_decode(clip)

        # First window gets full 4 frames, second window gets remaining 2
        assert len(windows) == 2
        assert windows[0].model_input["vllm_async"]["frames_shape"] == (4, 224, 224, 3)
        assert windows[1].model_input["vllm_async"]["frames_shape"] == (2, 224, 224, 3)

    def test_create_windows_frame_drop_exhausted(self) -> None:
        """When PyAV returns too few frames, exhausted windows should be skipped."""
        stage = self._make_stage(window_size=128)
        stage._prompt_template = "<prompt>describe</prompt>"

        clip = Clip(uuid=uuid4(), source_video="s.mp4", span=(0.0, 5.0), encoded_data=b"\x00" * 100)
        n_sampled_per_window = 4
        # PyAV returns only 4 frames -- enough for window 0 but nothing for window 1
        fake_frames = np.zeros((n_sampled_per_window, 224, 224, 3), dtype=np.uint8)
        window_infos = [WindowFrameInfo(start=0, end=127), WindowFrameInfo(start=128, end=255)]

        with (
            patch.object(vllm_async_stage, "buffer_as_memfd_path") as mock_memfd,
            patch.object(vllm_async_stage, "get_avg_frame_rate", return_value=30.0),
            patch.object(vllm_async_stage, "get_frame_count", return_value=256),
            patch.object(vllm_async_stage, "compute_windows", return_value=window_infos),
            patch.object(vllm_async_stage, "smart_nframes", return_value=n_sampled_per_window),
            patch.object(vllm_async_stage, "decode_video_cpu_frame_ids", return_value=fake_frames),
        ):
            mock_memfd.return_value.__enter__ = MagicMock(return_value="/fake/memfd")
            mock_memfd.return_value.__exit__ = MagicMock(return_value=False)
            windows = stage._create_windows_and_decode(clip)

        # Only the first window should be created; second is skipped (0 frames remaining)
        assert len(windows) == 1
        assert len(clip.windows) == 1
        assert windows[0].start_frame == 0
        assert windows[0].end_frame == 127
        assert windows[0].model_input["vllm_async"]["frames_shape"] == (4, 224, 224, 3)

    def test_keep_mp4_with_frame_drop_exhausted(self) -> None:
        """When keep_mp4=True and a window is skipped due to frame drop, MP4 extraction must not crash."""
        stage = self._make_stage(keep_mp4=True, window_size=128)
        stage._prompt_template = "<prompt>describe</prompt>"

        clip = Clip(uuid=uuid4(), source_video="s.mp4", span=(0.0, 5.0), encoded_data=b"\x00" * 100)
        n_sampled_per_window = 4
        # Only enough frames for window 0; window 1 is skipped
        fake_frames = np.zeros((n_sampled_per_window, 224, 224, 3), dtype=np.uint8)
        window_infos = [WindowFrameInfo(start=0, end=127), WindowFrameInfo(start=128, end=255)]

        # split_video_into_windows returns mp4 bytes for BOTH windows
        fake_mp4_bytes = [b"\xaa\xbb", b"\xcc\xdd"]

        with (
            patch.object(vllm_async_stage, "buffer_as_memfd_path") as mock_memfd,
            patch.object(vllm_async_stage, "get_avg_frame_rate", return_value=30.0),
            patch.object(vllm_async_stage, "get_frame_count", return_value=256),
            patch.object(vllm_async_stage, "compute_windows", return_value=window_infos),
            patch.object(vllm_async_stage, "smart_nframes", return_value=n_sampled_per_window),
            patch.object(vllm_async_stage, "decode_video_cpu_frame_ids", return_value=fake_frames),
            patch(
                "cosmos_curator.pipelines.video.utils.windowing_utils.split_video_into_windows",
                return_value=(fake_mp4_bytes, [None, None], window_infos),
            ),
        ):
            mock_memfd.return_value.__enter__ = MagicMock(return_value="/fake/memfd")
            mock_memfd.return_value.__exit__ = MagicMock(return_value=False)
            windows = stage._create_windows_and_decode(clip)

        # Only 1 window created (window 1 skipped due to exhausted frames)
        assert len(windows) == 1
        assert windows[0].start_frame == 0
        assert windows[0].end_frame == 127
        # MP4 bytes should be matched by (start, end) range, not by index.
        # LazyData.coerce converts bytes to numpy array; use .tobytes() to compare.
        assert windows[0].mp4_bytes.resolve().tobytes() == b"\xaa\xbb"

    def test_process_data_creates_windows_from_encoded_data(self) -> None:
        """process_data should create windows from clip.encoded_data (no pre-existing windows)."""
        stage = self._make_stage()
        stage._prompt_template = "<prompt>describe</prompt>"

        task = _make_task_with_encoded_data(b"\x00\x01\x02")
        fake_frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        window_info = WindowFrameInfo(start=0, end=99)

        with (
            patch.object(vllm_async_stage, "buffer_as_memfd_path") as mock_memfd,
            patch.object(vllm_async_stage, "get_avg_frame_rate", return_value=30.0),
            patch.object(vllm_async_stage, "get_frame_count", return_value=100),
            patch.object(vllm_async_stage, "compute_windows", return_value=[window_info]),
            patch.object(vllm_async_stage, "smart_nframes", return_value=4),
            patch.object(vllm_async_stage, "decode_video_cpu_frame_ids", return_value=fake_frames),
        ):
            mock_memfd.return_value.__enter__ = MagicMock(return_value="/fake/memfd")
            mock_memfd.return_value.__exit__ = MagicMock(return_value=False)
            result = stage.process_data([task])

        clip = result[0].video.clips[0]
        assert len(clip.windows) == 1
        assert "vllm_async" in clip.windows[0].model_input

    def test_process_data_fault_isolation(self) -> None:
        """process_data should record error on clip when _create_windows_and_decode fails."""
        stage = self._make_stage()
        stage._prompt_template = "<prompt>describe</prompt>"

        task = _make_task_with_encoded_data(b"\x00\x01")

        with (
            patch.object(vllm_async_stage, "buffer_as_memfd_path") as mock_memfd,
            patch.object(vllm_async_stage, "get_avg_frame_rate", side_effect=RuntimeError("probe error")),
            patch.object(vllm_async_stage, "get_frame_count", return_value=100),
        ):
            mock_memfd.return_value.__enter__ = MagicMock(return_value="/fake/memfd")
            mock_memfd.return_value.__exit__ = MagicMock(return_value=False)
            result = stage.process_data([task])

        clip = result[0].video.clips[0]
        assert "vllm_async_prep" in clip.errors
        assert "windowing+decode failed" in clip.errors["vllm_async_prep"]

    def test_getstate_excludes_non_serializable(self) -> None:
        """__getstate__ should exclude _logger for pickling."""
        stage = self._make_stage()
        state = stage.__getstate__()
        assert "_logger" not in state

    def test_setstate_restores_logger(self) -> None:
        """__setstate__ should recreate the _logger from the persisted _log_tag."""
        stage = self._make_stage()
        state = stage.__getstate__()
        assert "_logger" not in state

        stage.__setstate__(state)
        assert stage._logger is not None
        assert stage._log_tag == "[asyncvLLM-prep:qwen]"

    def test_pickle_roundtrip_restores_logger(self) -> None:
        """Full pickle roundtrip should produce a usable stage with a working logger."""
        stage = self._make_stage()
        restored = pickle.loads(pickle.dumps(stage))  # noqa: S301
        assert hasattr(restored, "_logger")
        assert restored._logger is not None
        assert restored._model_variant == "vllm_async"

    def test_build_prompt_raises_without_prompt_template(self) -> None:
        """_build_prompt should raise RuntimeError when prompt_template is not set."""
        stage = self._make_stage()
        assert stage._prompt_template is None
        with pytest.raises(RuntimeError, match="Prompt template not initialized"):
            stage._build_prompt()

    def test_create_windows_returns_empty_when_compute_windows_empty(self) -> None:
        """_create_windows_and_decode should return [] when compute_windows yields no windows."""
        stage = self._make_stage()
        stage._prompt_template = "<prompt>describe</prompt>"

        clip = Clip(uuid=uuid4(), source_video="s.mp4", span=(0.0, 1.0), encoded_data=b"\x00\x01\x02")

        with (
            patch.object(vllm_async_stage, "buffer_as_memfd_path") as mock_memfd,
            patch.object(vllm_async_stage, "get_avg_frame_rate", return_value=30.0),
            patch.object(vllm_async_stage, "get_frame_count", return_value=0),
            patch.object(vllm_async_stage, "compute_windows", return_value=[]),
        ):
            mock_memfd.return_value.__enter__ = MagicMock(return_value="/fake/memfd")
            mock_memfd.return_value.__exit__ = MagicMock(return_value=False)
            windows = stage._create_windows_and_decode(clip)

        assert len(windows) == 0
        assert len(clip.windows) == 0

    def test_per_window_failure_does_not_block_siblings(self) -> None:
        """A failed window does not block its healthy siblings; the loop continues."""
        stage = self._make_stage(window_size=128)
        stage._prompt_template = "<prompt>describe</prompt>"

        clip = Clip(uuid=uuid4(), source_video="s.mp4", span=(0.0, 5.0), encoded_data=b"\x00" * 100)
        n_sampled_per_window = 4
        total_sampled = n_sampled_per_window * 2
        fake_frames = np.zeros((total_sampled, 224, 224, 3), dtype=np.uint8)
        window_infos = [WindowFrameInfo(start=0, end=127), WindowFrameInfo(start=128, end=255)]

        with (
            patch.object(vllm_async_stage, "buffer_as_memfd_path") as mock_memfd,
            patch.object(vllm_async_stage, "get_avg_frame_rate", return_value=30.0),
            patch.object(vllm_async_stage, "get_frame_count", return_value=256),
            patch.object(vllm_async_stage, "compute_windows", return_value=window_infos),
            patch.object(vllm_async_stage, "smart_nframes", return_value=n_sampled_per_window),
            patch.object(vllm_async_stage, "decode_video_cpu_frame_ids", return_value=fake_frames),
            patch.object(stage, "_build_prompt", side_effect=["<prompt>describe</prompt>", RuntimeError("boom")]),
        ):
            mock_memfd.return_value.__enter__ = MagicMock(return_value="/fake/memfd")
            mock_memfd.return_value.__exit__ = MagicMock(return_value=False)
            windows = stage._create_windows_and_decode(clip)

        assert len(windows) == 1
        assert windows[0].start_frame == 0
        assert windows[0].end_frame == 127
        assert len(clip.windows) == 1
        assert clip.windows[0] is windows[0]
        assert "vllm_async_prep_window_128_255" in clip.errors
        assert "window assembly failed" in clip.errors["vllm_async_prep_window_128_255"]
        assert "boom" in clip.errors["vllm_async_prep_window_128_255"]

    def test_per_window_failure_records_error_and_skips_orphan(self) -> None:
        """A failed window leaves no orphan on clip.windows and only a per-window error key."""
        stage = self._make_stage()
        stage._prompt_template = "<prompt>describe</prompt>"

        clip = Clip(uuid=uuid4(), source_video="s.mp4", span=(0.0, 1.0), encoded_data=b"\x00\x01\x02")
        fake_frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        window_info = WindowFrameInfo(start=0, end=99)

        with (
            patch.object(vllm_async_stage, "buffer_as_memfd_path") as mock_memfd,
            patch.object(vllm_async_stage, "get_avg_frame_rate", return_value=30.0),
            patch.object(vllm_async_stage, "get_frame_count", return_value=100),
            patch.object(vllm_async_stage, "compute_windows", return_value=[window_info]),
            patch.object(vllm_async_stage, "smart_nframes", return_value=4),
            patch.object(vllm_async_stage, "decode_video_cpu_frame_ids", return_value=fake_frames),
            patch.object(stage, "_build_prompt", side_effect=RuntimeError("boom")),
        ):
            mock_memfd.return_value.__enter__ = MagicMock(return_value="/fake/memfd")
            mock_memfd.return_value.__exit__ = MagicMock(return_value=False)
            windows = stage._create_windows_and_decode(clip)

        assert windows == []
        assert len(clip.windows) == 0
        assert "vllm_async_prep_window_0_99" in clip.errors
        # Per-window data errors must NOT escalate to the outer per-clip catchall.
        assert "vllm_async_prep" not in clip.errors


class TestTextPromptSerialization:
    """Verify that raw TextPrompt dicts survive pickle roundtrip.

    Ray uses pickle for inter-actor serialization. The raw TextPrompt
    contains numpy uint8 frame arrays which must survive the CPU prep
    stage -> GPU caption stage transfer.
    """

    def test_text_prompt_pickle_roundtrip(self) -> None:
        """A TextPrompt-shaped dict with numpy arrays should survive pickle."""
        rng = np.random.default_rng(42)
        text_prompt = {
            "prompt": "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>Describe.",
            "multi_modal_data": {
                "video": [rng.integers(0, 255, size=(10, 384, 672, 3), dtype=np.uint8)],
            },
        }
        roundtripped = pickle.loads(pickle.dumps(text_prompt))  # noqa: S301

        assert roundtripped["prompt"] == text_prompt["prompt"]
        np.testing.assert_array_equal(
            roundtripped["multi_modal_data"]["video"][0],
            text_prompt["multi_modal_data"]["video"][0],
        )


class TestCaptionStageInputExtraction:
    """Tests for ``_extract_prepared_windows`` reading raw prompt + frames."""

    def _make_stage(self) -> VllmAsyncCaptionStage:
        config = VllmAsyncConfig(model_variant="qwen", num_gpus=1)
        return VllmAsyncCaptionStage(serve_config=config, model_name="qwen")

    def test_extracts_prepared_windows(self) -> None:
        """Build a ``_PreparedWindow`` when ``model_input[variant]`` carries the flat prep shape."""
        stage = self._make_stage()
        stage._sampling_params = MagicMock()

        frames = np.zeros((10, 2, 2, 3), dtype=np.uint8)
        task = _make_task(b"\x00", num_windows=1)
        window = task.video.clips[0].windows[0]
        _populate_window_input(window, "describe", frames)

        result = stage._extract_prepared_windows(task)

        assert len(result) == 1
        pw = result[0]
        assert isinstance(pw, _PreparedWindow)
        assert pw.prompt_text == "describe"
        assert pw.decoded_rgb_frames is frames
        assert pw.frames_shape == frames.shape
        assert "vllm_async" not in window.model_input  # cache evicted

    def test_records_error_on_missing_required_field(self) -> None:
        """A producer-side bug (missing flat-cache key) trips the KeyError path."""
        stage = self._make_stage()
        stage._sampling_params = MagicMock()

        task = _make_task(b"\x00", num_windows=1)
        clip = task.video.clips[0]
        # Missing ``"prompt"``, ``"video_frames"`` -- only ``frames_shape`` set.
        # ``cached["prompt"]`` raises ``KeyError`` inside the try-block,
        # which the ``except Exception`` branch records as
        # ``"input extraction failed: ..."`` on ``clip.errors``.
        clip.windows[0].model_input["vllm_async"] = {"frames_shape": (4, 224, 224, 3)}

        result = stage._extract_prepared_windows(task)

        assert result == []
        assert "vllm_async_caption_0" in clip.errors
        assert "input extraction failed" in clip.errors["vllm_async_caption_0"]
        assert clip.windows[0].caption_status == "error"
        assert "vllm_async" not in clip.windows[0].model_input

    def test_skips_missing_model_input(self) -> None:
        """Windows without ``model_input[variant]`` should be silently skipped."""
        stage = self._make_stage()
        stage._sampling_params = MagicMock()

        task = _make_task(b"\x00", num_windows=1)
        assert stage._extract_prepared_windows(task) == []

    def test_isolates_failure_across_windows(self) -> None:
        """A bad window does not block its healthy siblings; the loop continues."""
        stage = self._make_stage()
        stage._sampling_params = MagicMock()

        task = _make_task(b"\x00", num_windows=2)
        clip = task.video.clips[0]
        # Window 0: triggers ``KeyError`` on ``cached["frames_shape"]``.
        bad_frames = np.zeros((4, 4, 4, 3), dtype=np.uint8)
        clip.windows[0].model_input["vllm_async"] = {
            "prompt": "describe-0",
            "video_frames": bad_frames,
        }
        # Window 1: well-formed; must still be extracted.
        good_frames = np.zeros((4, 4, 4, 3), dtype=np.uint8)
        _populate_window_input(clip.windows[1], "describe-1", good_frames)

        result = stage._extract_prepared_windows(task)

        assert len(result) == 1
        assert result[0].window is clip.windows[1]
        assert result[0].prompt_text == "describe-1"
        assert result[0].decoded_rgb_frames is good_frames

        assert clip.windows[0].caption_status == "error"
        assert "vllm_async_caption_0" in clip.errors
        assert "vllm_async_caption_1" not in clip.errors
        assert "vllm_async" not in clip.windows[0].model_input
        assert "vllm_async" not in clip.windows[1].model_input


class TestStage2CaptionRefinement:
    """Tests for stage-2 caption refinement in ``VllmAsyncCaptionStage``."""

    def _make_stage(
        self, *, stage2_caption: bool = False, stage2_prompt_text: str | None = None
    ) -> VllmAsyncCaptionStage:
        config = VllmAsyncConfig(model_variant="qwen", num_gpus=1)
        return VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
            stage2_caption=stage2_caption,
            stage2_prompt_text=stage2_prompt_text,
        )

    def test_stage2_defaults_disabled(self) -> None:
        """``stage2_caption`` should default to ``False``."""
        stage = self._make_stage()
        assert stage._stage2_caption is False
        assert stage._stage2_prompt_text is None
        assert stage._stage2_processor is None

    def test_stage2_enabled_stores_params(self) -> None:
        """When ``stage2_caption=True``, init stores both flag and prompt text."""
        stage = self._make_stage(stage2_caption=True, stage2_prompt_text="Refine this.")
        assert stage._stage2_caption is True
        assert stage._stage2_prompt_text == "Refine this."

    def test_getstate_excludes_stage2_processor(self) -> None:
        """``__getstate__`` excludes ``_stage2_processor`` from pickle state."""
        stage = self._make_stage(stage2_caption=True)
        stage._stage2_processor = MagicMock()
        assert "_stage2_processor" not in stage.__getstate__()

    def test_setstate_restores_stage2_processor_as_none(self) -> None:
        """``__setstate__`` restores ``_stage2_processor`` as ``None``."""
        stage = self._make_stage(stage2_caption=True)
        stage.__setstate__(stage.__getstate__())
        assert stage._stage2_processor is None

    def test_generate_and_assign_stage1_only(self) -> None:
        """Without stage 2, ``_generate_and_assign`` assigns the caption directly."""
        stage = self._make_stage(stage2_caption=False)
        mock_engine = MagicMock()
        _mock_renderer(mock_engine)
        stage._engine = mock_engine
        stage._sampling_params = MagicMock()

        pw = _make_prepared_window()
        stage2_queue: collections.deque[tuple[_PreparedWindow, str]] = collections.deque()

        with patch.object(stage, "_generate_caption_async", return_value=("A test caption", TokenCounts(10, 5))):
            asyncio.run(stage._generate_and_assign(pw, asyncio.Semaphore(1), stage2_queue))

        assert pw.window.caption["vllm_async"] == "A test caption"
        assert pw.window.caption_status == "success"
        assert pw.window.caption_failure_reason is None
        assert not stage2_queue

    def test_generate_and_assign_stage2_enqueues(self) -> None:
        """With stage 2, the result is enqueued for refinement instead of assigned."""
        stage = self._make_stage(stage2_caption=True, stage2_prompt_text="Refine it.")
        mock_engine = MagicMock()
        _mock_renderer(mock_engine)
        stage._engine = mock_engine
        stage._sampling_params = MagicMock()
        stage._stage2_processor = MagicMock()

        pw = _make_prepared_window()
        stage2_queue: collections.deque[tuple[_PreparedWindow, str]] = collections.deque()

        with patch.object(stage, "_generate_caption_async", return_value=("stage1 caption", TokenCounts(10, 5))):
            asyncio.run(stage._generate_and_assign(pw, asyncio.Semaphore(1), stage2_queue))

        assert "vllm_async" not in pw.window.caption
        assert len(stage2_queue) == 1
        queued_pw, queued_caption = stage2_queue[0]
        assert queued_pw is pw
        assert queued_caption == "stage1 caption"

    def test_generate_and_assign_skips_stage2_when_processor_none(self) -> None:
        """If ``_stage2_processor`` is ``None``, no stage-2 enqueue happens."""
        stage = self._make_stage(stage2_caption=True)
        mock_engine = MagicMock()
        _mock_renderer(mock_engine)
        stage._engine = mock_engine
        stage._sampling_params = MagicMock()

        pw = _make_prepared_window()
        stage2_queue: collections.deque[tuple[_PreparedWindow, str]] = collections.deque()

        with patch.object(stage, "_generate_caption_async", return_value=("Only caption", TokenCounts(10, 5))):
            asyncio.run(stage._generate_and_assign(pw, asyncio.Semaphore(1), stage2_queue))

        assert pw.window.caption["vllm_async"] == "Only caption"
        assert not stage2_queue

    def test_stage2_refine_assigns_final_caption(self) -> None:
        """``_stage2_refine_and_assign`` renders refinement prompt and writes final caption."""
        stage = self._make_stage(stage2_caption=True, stage2_prompt_text="Refine it.")
        mock_engine = MagicMock()
        _mock_renderer(mock_engine)
        stage._engine = mock_engine
        stage._sampling_params = MagicMock()
        stage._stage2_processor = MagicMock()

        pw = _make_prepared_window()

        with (
            patch.object(stage, "_generate_caption_async", return_value=("Refined caption", TokenCounts(2, 3))),
            patch(
                "cosmos_curator.pipelines.video.captioning.vllm_async_stage.build_refinement_prompt_text",
                return_value="<refined>",
            ) as mock_build,
        ):
            asyncio.run(stage._stage2_refine_and_assign(pw, "Stage-1 text", asyncio.Semaphore(1)))

        mock_build.assert_called_once_with(stage._stage2_processor, "Stage-1 text", "Refine it.")
        mock_engine.renderer.render_cmpl.assert_called_once()
        assert pw.window.caption["vllm_async"] == "Refined caption"
        assert pw.window.caption_status == "success"
        assert pw.window.caption_failure_reason is None

    def test_default_stage2_prompt_text_forwarded_as_none(self) -> None:
        """When no custom prompt is set, the builder receives ``None`` (uses default)."""
        stage = self._make_stage(stage2_caption=True, stage2_prompt_text=None)
        mock_engine = MagicMock()
        _mock_renderer(mock_engine)
        stage._engine = mock_engine
        stage._sampling_params = MagicMock()
        stage._stage2_processor = MagicMock()

        pw = _make_prepared_window()

        with (
            patch.object(stage, "_generate_caption_async", return_value=("Refined caption", TokenCounts(0, 0))),
            patch(
                "cosmos_curator.pipelines.video.captioning.vllm_async_stage.build_refinement_prompt_text",
                return_value="<rendered>",
            ) as mock_build,
        ):
            asyncio.run(stage._stage2_refine_and_assign(pw, "Stage-1 text", asyncio.Semaphore(1)))

        mock_build.assert_called_once_with(stage._stage2_processor, "Stage-1 text", None)


class TestRenderStageRemoval:
    """Guard test confirming the legacy render stage has been removed."""

    def test_render_stage_class_removed(self) -> None:
        """``VllmAsyncPromptRenderStage`` is no longer importable."""
        assert not hasattr(vllm_async_stage, "VllmAsyncPromptRenderStage")


class TestCaptionStageResources:
    """Tests for VllmAsyncCaptionStage resource allocation."""

    def test_resources_1_cpu_with_gpus(self) -> None:
        """DP mode: caption stage should request total_gpus (num_gpus * dp)."""
        config = VllmAsyncConfig(model_variant="qwen", num_gpus=2, data_parallel_size=2)
        stage = VllmAsyncCaptionStage(serve_config=config, model_name="qwen")
        assert stage.resources.cpus == 1.0
        assert stage.resources.gpus == 4.0

    def test_resources_single_gpu(self) -> None:
        """N-actors mode: single GPU should request num_gpus."""
        config = VllmAsyncConfig(model_variant="qwen", num_gpus=1)
        stage = VllmAsyncCaptionStage(serve_config=config, model_name="qwen")
        assert stage.resources.cpus == 1.0
        assert stage.resources.gpus == 1.0


class TestResolveMode:
    """Unit tests for _resolve_mode() -- pure function, easy to test."""

    def test_mode_n_actors_tp1(self) -> None:
        """num_gpus=1, dp=1 -> N-actors: gpus=1, batch=1, sem=128, backend='mp'."""
        config = VllmAsyncConfig(model_variant="qwen", num_gpus=1, data_parallel_size=1)
        mode = _resolve_mode(config)
        assert mode.gpus_per_actor == 1.0
        assert mode.stage_batch_size == 1
        assert mode.semaphore_limit == _VllmAsyncStageMode.N_ACTORS_SEMAPHORE_LIMIT
        assert mode.executor_backend == "mp"
        assert mode.is_dp_mode is False

    def test_mode_n_actors_tp2(self) -> None:
        """num_gpus=2, dp=1 -> N-actors: gpus=2, batch=1, sem=128, backend='ray'."""
        config = VllmAsyncConfig(model_variant="qwen", num_gpus=2, data_parallel_size=1)
        mode = _resolve_mode(config)
        assert mode.gpus_per_actor == 2.0
        assert mode.stage_batch_size == 1
        assert mode.semaphore_limit == _VllmAsyncStageMode.N_ACTORS_SEMAPHORE_LIMIT
        assert mode.executor_backend == "ray"
        assert mode.is_dp_mode is False

    def test_mode_n_actors_tp4(self) -> None:
        """num_gpus=4, dp=1 -> N-actors: gpus=4, batch=1, sem=128, backend='ray'."""
        config = VllmAsyncConfig(model_variant="qwen", num_gpus=4, data_parallel_size=1)
        mode = _resolve_mode(config)
        assert mode.gpus_per_actor == 4.0
        assert mode.stage_batch_size == 1
        assert mode.semaphore_limit == _VllmAsyncStageMode.N_ACTORS_SEMAPHORE_LIMIT
        assert mode.executor_backend == "ray"
        assert mode.is_dp_mode is False

    def test_mode_dp_tp1(self) -> None:
        """num_gpus=1, dp=7 -> DP: gpus=7, batch=21, sem=896, backend='ray'."""
        config = VllmAsyncConfig(model_variant="qwen", num_gpus=1, data_parallel_size=7)
        mode = _resolve_mode(config)
        assert mode.gpus_per_actor == 7.0
        assert mode.stage_batch_size == max(
            _VllmAsyncStageMode.DP_BATCH_MULTIPLIER * 7,
            _VllmAsyncStageMode.DP_BATCH_FLOOR,
        )
        assert mode.semaphore_limit == _VllmAsyncStageMode.N_ACTORS_SEMAPHORE_LIMIT * 7
        assert mode.executor_backend == "ray"
        assert mode.is_dp_mode is True

    def test_mode_dp_tp2(self) -> None:
        """num_gpus=2, dp=2 -> DP: gpus=4, batch=12, sem=512, backend='ray'."""
        config = VllmAsyncConfig(model_variant="qwen", num_gpus=2, data_parallel_size=2)
        mode = _resolve_mode(config)
        assert mode.gpus_per_actor == 4.0
        assert mode.stage_batch_size == max(
            _VllmAsyncStageMode.DP_BATCH_MULTIPLIER * 4,
            _VllmAsyncStageMode.DP_BATCH_FLOOR,
        )
        assert mode.semaphore_limit == _VllmAsyncStageMode.N_ACTORS_SEMAPHORE_LIMIT * 4
        assert mode.executor_backend == "ray"
        assert mode.is_dp_mode is True

    def test_mode_dp_tp2_dp3(self) -> None:
        """num_gpus=2, dp=3 -> DP: gpus=6, batch=18, sem=768, backend='ray'."""
        config = VllmAsyncConfig(model_variant="qwen", num_gpus=2, data_parallel_size=3)
        mode = _resolve_mode(config)
        assert mode.gpus_per_actor == 6.0
        assert mode.stage_batch_size == max(
            _VllmAsyncStageMode.DP_BATCH_MULTIPLIER * 6,
            _VllmAsyncStageMode.DP_BATCH_FLOOR,
        )
        assert mode.semaphore_limit == _VllmAsyncStageMode.N_ACTORS_SEMAPHORE_LIMIT * 6
        assert mode.executor_backend == "ray"
        assert mode.is_dp_mode is True

    def test_mode_frozen(self) -> None:
        """_VllmAsyncStageMode should be immutable after creation."""
        config = VllmAsyncConfig(model_variant="qwen", num_gpus=1)
        mode = _resolve_mode(config)
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            mode.gpus_per_actor = 99  # type: ignore[misc]

    def test_mode_n_actors_default_dp(self) -> None:
        """Default data_parallel_size (1) should select N-actors mode."""
        config = VllmAsyncConfig(model_variant="qwen", num_gpus=2)
        mode = _resolve_mode(config)
        assert mode.is_dp_mode is False
        assert mode.stage_batch_size == 1


class TestStageModePropertyDelegation:
    """Verify stage properties delegate to _VllmAsyncStageMode correctly."""

    def _make_config(self, **overrides: object) -> VllmAsyncConfig:
        defaults: dict[str, object] = {"model_variant": "qwen", "num_gpus": 2}
        defaults.update(overrides)
        return VllmAsyncConfig(**defaults)

    def test_resources_uses_mode(self) -> None:
        """resources.gpus should match _mode.gpus_per_actor."""
        config = self._make_config(num_gpus=3, data_parallel_size=1)
        stage = VllmAsyncCaptionStage(serve_config=config, model_name="qwen")
        assert stage.resources.gpus == stage._mode.gpus_per_actor

    def test_batch_size_uses_mode(self) -> None:
        """stage_batch_size should return _mode.stage_batch_size when no override."""
        config = self._make_config(num_gpus=1, data_parallel_size=1)
        stage = VllmAsyncCaptionStage(serve_config=config, model_name="qwen")
        assert stage.stage_batch_size == stage._mode.stage_batch_size

    def test_batch_size_explicit_override(self) -> None:
        """Explicit stage_batch_size > 0 should override _mode."""
        config = self._make_config(num_gpus=1, data_parallel_size=1)
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
            stage_batch_size=42,
        )
        assert stage.stage_batch_size == 42
        assert stage._mode.stage_batch_size == 1

    def test_semaphore_uses_mode(self) -> None:
        """_effective_max_concurrent_requests should return _mode.semaphore_limit."""
        config = self._make_config(num_gpus=2, data_parallel_size=1)
        stage = VllmAsyncCaptionStage(serve_config=config, model_name="qwen")
        assert stage._effective_max_concurrent_requests == stage._mode.semaphore_limit

    def test_semaphore_explicit_override(self) -> None:
        """Explicit max_concurrent_requests > 0 should override _mode."""
        config = self._make_config(num_gpus=2, data_parallel_size=1)
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
            max_concurrent_requests=99,
        )
        assert stage._effective_max_concurrent_requests == 99
        assert stage._mode.semaphore_limit == _VllmAsyncStageMode.N_ACTORS_SEMAPHORE_LIMIT

    def test_mode_survives_pickle(self) -> None:
        """_mode should be recomputed after pickle round-trip."""
        config = self._make_config(num_gpus=1, data_parallel_size=1)
        stage = VllmAsyncCaptionStage(serve_config=config, model_name="qwen")
        restored = pickle.loads(pickle.dumps(stage))  # noqa: S301
        assert restored._mode.gpus_per_actor == 1.0
        assert restored._mode.executor_backend == "mp"
        assert restored._mode.is_dp_mode is False


class TestPreparedWindow:
    """Tests for the ``_PreparedWindow`` attrs class."""

    def test_attrs_round_trip(self) -> None:
        """All fields are stored and retrievable as supplied."""
        frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        sampling = MagicMock()
        clip = Clip(uuid=uuid4(), source_video="v.mp4", span=(0.0, 1.0))
        window = Window(start_frame=0, end_frame=10)
        clip.windows = [window]
        pw = vllm_async_stage._PreparedWindow(
            clip=clip,
            window_index=3,
            window=window,
            prompt_text="hello",
            decoded_rgb_frames=frames,
            sampling_params=sampling,
            frames_shape=tuple(frames.shape),
        )
        assert pw.clip is clip
        assert pw.window_index == 3
        assert pw.window is window
        assert pw.prompt_text == "hello"
        assert pw.decoded_rgb_frames is frames
        assert pw.sampling_params is sampling
        assert pw.frames_shape == (2, 4, 4, 3)

    def test_eq_false_identity_equality(self) -> None:
        """Two structurally-identical instances are NOT equal (eq=False)."""
        frames = np.zeros((1, 1, 1, 3), dtype=np.uint8)
        sampling = MagicMock()
        clip = Clip(uuid=uuid4(), source_video="v.mp4", span=(0.0, 1.0))
        window = Window(start_frame=0, end_frame=1)
        clip.windows = [window]
        a = vllm_async_stage._PreparedWindow(
            clip=clip,
            window_index=0,
            window=window,
            prompt_text="x",
            decoded_rgb_frames=frames,
            sampling_params=sampling,
            frames_shape=tuple(frames.shape),
        )
        b = vllm_async_stage._PreparedWindow(
            clip=clip,
            window_index=0,
            window=window,
            prompt_text="x",
            decoded_rgb_frames=frames,
            sampling_params=sampling,
            frames_shape=tuple(frames.shape),
        )
        assert a != b
        assert hash(a) != hash(b)


class TestBuildRenderPayload:
    """Tests for the module-level ``_build_render_payload`` helper."""

    def test_returns_expected_keys(self) -> None:
        """Payload contains ``prompt`` and ``multi_modal_data.video``."""
        frames = np.zeros((4, 4, 4, 3), dtype=np.uint8)
        payload = _build_render_payload("describe", frames, "")
        assert payload["prompt"] == "describe"
        assert payload["multi_modal_data"]["video"][0] is frames

    def test_render_payload_is_fresh_per_call(self) -> None:
        """Section-30 guard: outer dict and ``multi_modal_data`` are fresh per call."""
        frames = np.zeros((4, 4, 4, 3), dtype=np.uint8)
        payload_a = _build_render_payload("describe", frames, "")
        payload_b = _build_render_payload("describe", frames, "")
        assert payload_a is not payload_b
        assert payload_a["multi_modal_data"] is not payload_b["multi_modal_data"]
        # Inner ndarray is intentionally shared (zero-copy).
        assert payload_a["multi_modal_data"]["video"][0] is payload_b["multi_modal_data"]["video"][0]

    def test_includes_mm_processor_kwargs_when_non_empty(self) -> None:
        """Non-empty JSON is parsed and attached as ``mm_processor_kwargs``."""
        frames = np.zeros((4, 4, 4, 3), dtype=np.uint8)
        payload = _build_render_payload("describe", frames, '{"max_pixels": 12345}')
        assert payload["mm_processor_kwargs"] == {"max_pixels": 12345}

    def test_omits_mm_processor_kwargs_when_empty(self) -> None:
        """Empty string suppresses the per-prompt ``mm_processor_kwargs`` key."""
        frames = np.zeros((4, 4, 4, 3), dtype=np.uint8)
        payload = _build_render_payload("describe", frames, "")
        assert "mm_processor_kwargs" not in payload

    def test_mm_processor_kwargs_is_fresh_tree_per_call(self) -> None:
        """Each call yields a freshly parsed tree -- HF in-place mutation cannot bleed."""
        frames = np.zeros((4, 4, 4, 3), dtype=np.uint8)
        kwargs_json = '{"videos_kwargs": {"size": {"longest_edge": 100}}}'
        payload_a = _build_render_payload("describe", frames, kwargs_json)
        payload_b = _build_render_payload("describe", frames, kwargs_json)
        # Outer dicts and nested dicts are distinct objects; mutating one
        # must not poison the next.
        assert payload_a["mm_processor_kwargs"] is not payload_b["mm_processor_kwargs"]
        assert (
            payload_a["mm_processor_kwargs"]["videos_kwargs"] is not payload_b["mm_processor_kwargs"]["videos_kwargs"]
        )
        payload_a["mm_processor_kwargs"]["videos_kwargs"].pop("size")
        assert "size" in payload_b["mm_processor_kwargs"]["videos_kwargs"]


class TestRenderPayloadWiring:
    """Confirms ``_render_payload`` forwards ``_serve_config.mm_processor_kwargs``."""

    def test_forwards_mm_processor_kwargs_from_serve_config(self) -> None:
        """The payload sent to ``renderer.render_cmpl`` carries the configured kwargs."""
        config = VllmAsyncConfig(
            model_variant="qwen",
            num_gpus=1,
            mm_processor_kwargs='{"max_pixels": 99999}',
        )
        stage = VllmAsyncCaptionStage(serve_config=config, model_name="qwen")
        mock_engine = MagicMock()
        _mock_renderer(mock_engine)
        stage._engine = mock_engine

        frames = np.zeros((4, 4, 4, 3), dtype=np.uint8)
        asyncio.run(stage._render_payload("describe", frames))

        called_payload = mock_engine.renderer.render_cmpl.call_args[0][0][0]
        assert called_payload["mm_processor_kwargs"] == {"max_pixels": 99999}


class TestContinuousTaskTracker:
    """Tests for the ``_ContinuousTaskTracker`` attrs class."""

    def _make_input(self) -> ContinuousTaskInput:
        return ContinuousTaskInput(task_id="t1", data=[MagicMock()], timing=None, object_sizes=None)

    def test_factory_defaults_isolated(self) -> None:
        """``pending`` and ``stage2_queue`` defaults must be per-instance."""
        a = _ContinuousTaskTracker(task_input=self._make_input())
        b = _ContinuousTaskTracker(task_input=self._make_input())
        assert a.pending is not b.pending
        assert a.stage2_queue is not b.stage2_queue

    def test_all_done_initial(self) -> None:
        """A freshly constructed tracker reports ``all_done`` == True."""
        tr = _ContinuousTaskTracker(task_input=self._make_input())
        assert tr.all_done() is True

    def test_all_done_after_pending_added(self) -> None:
        """``all_done`` is False while ``pending`` or ``stage2_queue`` is non-empty."""

        async def _runner() -> None:
            tr = _ContinuousTaskTracker(task_input=self._make_input())
            task = asyncio.create_task(asyncio.sleep(0))
            tr.pending.add(task)
            assert tr.all_done() is False
            await task
            tr.pending.discard(task)
            assert tr.all_done() is True

        asyncio.run(_runner())


class TestRenderLockPickle:
    """Tests for ``_render_lock`` exclusion from pickle and re-creation."""

    def _make_stage(self) -> VllmAsyncCaptionStage:
        config = VllmAsyncConfig(model_variant="qwen", num_gpus=1)
        return VllmAsyncCaptionStage(serve_config=config, model_name="qwen")

    def test_getstate_excludes_render_lock(self) -> None:
        """``__getstate__`` must omit the un-picklable ``asyncio.Lock``."""
        stage = self._make_stage()
        assert "_render_lock" not in stage.__getstate__()

    def test_pickle_roundtrip_recreates_render_lock(self) -> None:
        """After pickle round-trip, ``_render_lock`` is a fresh ``asyncio.Lock``."""
        stage = self._make_stage()
        restored = pickle.loads(pickle.dumps(stage))  # noqa: S301
        assert isinstance(restored._render_lock, asyncio.Lock)


class TestContinuousInterfaceMixin:
    """Smoke test for the ``ContinuousInterface`` mixin."""

    def test_caption_stage_implements_continuous_interface(self) -> None:
        """``VllmAsyncCaptionStage`` must satisfy ``isinstance(..., ContinuousInterface)``."""
        config = VllmAsyncConfig(model_variant="qwen", num_gpus=1)
        stage = VllmAsyncCaptionStage(serve_config=config, model_name="qwen")
        assert isinstance(stage, ContinuousInterface)


class TestVllmAsyncCaptionStageContinuous:
    """Async tests for ``VllmAsyncCaptionStage.run_continuous``."""

    def _make_stage(self) -> VllmAsyncCaptionStage:
        config = VllmAsyncConfig(model_variant="qwen", num_gpus=1)
        stage = VllmAsyncCaptionStage(serve_config=config, model_name="qwen", max_concurrent_requests=2)
        stage._sampling_params = MagicMock()
        mock_engine = MagicMock()
        _mock_renderer(mock_engine)
        mock_engine.generate = MagicMock(side_effect=_async_gen_side_effect(_mock_request_output("CAP")))
        stage._engine = mock_engine
        return stage

    def _make_task_input(self, *, task_id: str, num_windows: int) -> ContinuousTaskInput:
        task = _make_task(b"\x00", num_windows=num_windows)
        frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        for window in task.video.clips[0].windows:
            _populate_window_input(window, "describe", frames)
        return ContinuousTaskInput(task_id=task_id, data=[task], timing=None, object_sizes=None)

    def _make_zero_window_task_input(self, *, task_id: str, num_windows: int = 1) -> ContinuousTaskInput:
        """Build a ``ContinuousTaskInput`` whose windows have no ``model_input`` populated.

        ``_extract_prepared_windows`` skips windows whose ``model_input[variant]``
        is missing, so the returned task yields zero ``_PreparedWindow`` items
        - exercising the synchronous-emit path in ``_register_task``.
        """
        task = _make_task(b"\x00", num_windows=num_windows)
        return ContinuousTaskInput(task_id=task_id, data=[task], timing=None, object_sizes=None)

    async def _run_until_outputs(  # noqa: PLR0913
        self,
        stage: VllmAsyncCaptionStage,
        input_q: "asyncio.Queue[ContinuousTaskInput]",
        output_q: "asyncio.Queue[ContinuousTaskOutput]",
        stop: asyncio.Event,
        expected: int,
        deadline_s: float = 5.0,
    ) -> list[ContinuousTaskOutput]:
        """Run ``run_continuous`` until ``expected`` outputs land on ``output_q``.

        Event-driven: ``await output_q.get()`` is the synchronisation primitive,
        so the helper fails fast with ``TimeoutError`` carrying the actual
        cause instead of swallowing the failure inside a polling loop.
        Mirrors the framework contract where shutdown is driven exclusively
        by ``stop_event`` (no in-band sentinel).

        The cleanup path bounds the runner await with a hard timeout and
        cancels on hang so a stuck ``run_continuous`` never masks the
        original ``TimeoutError`` (or other inner failure) by blocking
        the test indefinitely in ``finally``.
        """
        runner = asyncio.create_task(stage.run_continuous(input_q, output_q, stop))
        try:
            async with asyncio.timeout(deadline_s):
                received = [await output_q.get() for _ in range(expected)]
        finally:
            stop.set()
            try:
                await asyncio.wait_for(runner, timeout=2.0)
            except TimeoutError:
                runner.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await runner
        return received

    @pytest.mark.asyncio
    async def test_run_continuous_processes_single_task_single_window(self) -> None:
        """A single task / single window emits one ``ContinuousTaskOutput``."""
        stage = self._make_stage()
        input_q: asyncio.Queue[ContinuousTaskInput] = asyncio.Queue()
        output_q: asyncio.Queue[ContinuousTaskOutput] = asyncio.Queue()
        stop = asyncio.Event()

        await input_q.put(self._make_task_input(task_id="t1", num_windows=1))

        results = await self._run_until_outputs(stage, input_q, output_q, stop, expected=1)

        assert len(results) == 1
        out = results[0]
        assert out.task_id == "t1"
        assert len(out.out_data) == 1
        clip = out.out_data[0].video.clips[0]
        assert clip.windows[0].caption["vllm_async"] == "CAP"

    @pytest.mark.asyncio
    async def test_run_continuous_overlaps_multiple_tasks(self) -> None:
        """Multiple overlapping tasks emit unique captions per window with no mis-routing.

        Each ``engine.generate`` call gets a unique ``request_id``; the mock
        encodes that id into the caption text, so any window-to-window swap
        would surface as a duplicate caption or as a window holding a caption
        that does not match its own request_id pattern.
        """
        stage = self._make_stage()
        stage._engine.generate = MagicMock(side_effect=_async_gen_side_effect_per_request("CAP"))

        input_q: asyncio.Queue[ContinuousTaskInput] = asyncio.Queue()
        output_q: asyncio.Queue[ContinuousTaskOutput] = asyncio.Queue()
        stop = asyncio.Event()

        for i in range(3):
            await input_q.put(self._make_task_input(task_id=f"t{i}", num_windows=2))

        results = await self._run_until_outputs(stage, input_q, output_q, stop, expected=3)

        assert output_q.empty(), "no straggler outputs should remain after the helper drains"
        assert {r.task_id for r in results} == {"t0", "t1", "t2"}

        all_captions: list[str] = []
        for r in results:
            assert len(r.out_data) == 1
            for w in r.out_data[0].video.clips[0].windows:
                caption = w.caption["vllm_async"]
                assert caption.startswith("CAP-caption-"), f"unexpected caption shape: {caption!r}"
                all_captions.append(caption)

        assert len(all_captions) == 6, "expected 6 windows total (3 tasks x 2 windows)"
        assert len(set(all_captions)) == 6, f"expected 6 unique captions, got duplicates: {all_captions}"

    @pytest.mark.asyncio
    async def test_run_continuous_emits_task_with_no_prepared_windows(self) -> None:
        """A task that yields zero prepared windows is emitted synchronously.

        Regression: previously the tracker was inserted into ``trackers`` with
        ``pending == set()``, but ``_emit_completed_tasks`` ran only inside
        the ``if has_pending:`` branch -- so an all-zero-window stream would
        stall until ``stop_event`` fired.  After the fix the zero-window task
        is emitted directly to ``output_queue`` from ``_register_task``.
        """
        stage = self._make_stage()
        input_q: asyncio.Queue[ContinuousTaskInput] = asyncio.Queue()
        output_q: asyncio.Queue[ContinuousTaskOutput] = asyncio.Queue()
        stop = asyncio.Event()

        await input_q.put(self._make_zero_window_task_input(task_id="zero-1"))

        results = await self._run_until_outputs(stage, input_q, output_q, stop, expected=1)

        assert len(results) == 1
        assert results[0].task_id == "zero-1"
        stage._engine.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_continuous_handles_mixed_zero_and_windowed_tasks(self) -> None:
        """A zero-window task enqueued before a windowed task: both must emit.

        Regression: with the bug present, the windowed task could complete
        and trigger emission of the zero-window task only by accident -- and
        any all-zero-window stream stalled outright.  Both task ids must
        surface within the deadline regardless of ordering.
        """
        stage = self._make_stage()
        input_q: asyncio.Queue[ContinuousTaskInput] = asyncio.Queue()
        output_q: asyncio.Queue[ContinuousTaskOutput] = asyncio.Queue()
        stop = asyncio.Event()

        await input_q.put(self._make_zero_window_task_input(task_id="zero-first"))
        await input_q.put(self._make_task_input(task_id="windowed", num_windows=1))

        results = await self._run_until_outputs(stage, input_q, output_q, stop, expected=2)

        assert {r.task_id for r in results} == {"zero-first", "windowed"}

    @pytest.mark.asyncio
    async def test_run_continuous_terminates_on_stop_event(self) -> None:
        """``stop_event`` exits the loop even when the input queue is empty."""
        stage = self._make_stage()
        input_q: asyncio.Queue[ContinuousTaskInput] = asyncio.Queue()
        output_q: asyncio.Queue[ContinuousTaskOutput] = asyncio.Queue()
        stop = asyncio.Event()

        async def _stop_soon() -> None:
            await asyncio.sleep(0.05)
            stop.set()

        await asyncio.gather(
            asyncio.wait_for(stage.run_continuous(input_q, output_q, stop), timeout=2.0),
            _stop_soon(),
        )

    @pytest.mark.asyncio
    async def test_render_failure_propagates_through_run_continuous(self) -> None:
        """Per the simplified contract, a render error bubbles out of ``run_continuous``."""
        stage = self._make_stage()
        stage._engine.renderer.render_cmpl = MagicMock(side_effect=RuntimeError("render boom"))

        input_q: asyncio.Queue[ContinuousTaskInput] = asyncio.Queue()
        output_q: asyncio.Queue[ContinuousTaskOutput] = asyncio.Queue()
        stop = asyncio.Event()

        await input_q.put(self._make_task_input(task_id="t99", num_windows=1))

        with pytest.raises(RuntimeError, match="render boom"):
            await asyncio.wait_for(stage.run_continuous(input_q, output_q, stop), timeout=5.0)
