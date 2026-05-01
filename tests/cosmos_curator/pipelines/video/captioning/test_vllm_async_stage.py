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
from unittest.mock import AsyncMock, MagicMock, call, patch
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
    VllmAsyncPromptRenderStage,
    _build_engine_args,
    _build_render_engine_args,
    _build_render_payload,
    _RenderedWindow,
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


def _mock_renderer(mock_engine: MagicMock) -> None:
    """Set up ``mock_engine.renderer.render_cmpl_async`` as a passthrough.

    ``_render_chunk`` calls ``engine.renderer.render_cmpl_async(prompts)``
    to convert raw TextPrompts into ProcessorInputs.  This helper
    configures the mock to return the inputs unchanged so tests exercising
    the generation path work without a real Renderer.
    """
    mock_engine.renderer.render_cmpl_async = AsyncMock(side_effect=lambda prompts: prompts)


def _rendered_prompt_state(frames: np.ndarray, *, prompt: str = "test") -> dict[str, object]:
    """Build cached prompt primitives stored after render-stage processing."""
    return {"raw_prompt": prompt, "decoded_rgb_frames": frames}


def _qwen_video_mm_processor_kwargs() -> dict[str, object]:
    """Return the nested Qwen video processor kwargs used in async config defaults."""
    return {
        "videos_kwargs": {
            "size": {
                "shortest_edge": VIDEO_MIN_PIXELS,
                "longest_edge": VIDEO_MAX_PIXELS,
            }
        }
    }


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


class TestRenderPayloadHelpers:
    """Tests for helper functions that build async renderer payloads."""

    def test_build_render_payload_uses_fresh_containers(self) -> None:
        """Each render payload should get fresh wrapper state around decoded frames."""
        frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        mm_kwargs = json.dumps(_qwen_video_mm_processor_kwargs())

        first = _build_render_payload("describe", frames, mm_kwargs)
        second = _build_render_payload("describe", frames, mm_kwargs)

        assert first["prompt"] == "describe"
        assert first["multi_modal_data"] is not second["multi_modal_data"]
        assert first["multi_modal_data"]["video"] is not second["multi_modal_data"]["video"]
        assert first["multi_modal_data"]["video"][0] is frames
        assert first["mm_processor_kwargs"] == _qwen_video_mm_processor_kwargs()
        assert first["mm_processor_kwargs"] is not second["mm_processor_kwargs"]

    def test_build_render_payload_omits_empty_mm_processor_kwargs(self) -> None:
        """Empty processor kwargs should not add a per-prompt override."""
        frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)

        payload = _build_render_payload("describe", frames, "")

        assert "mm_processor_kwargs" not in payload


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

    def test_destroy_closes_runner(self) -> None:
        """destroy() should close the asyncio.Runner."""
        config = self._make_config()
        stage = VllmAsyncCaptionStage(serve_config=config, model_name="qwen")
        mock_runner = MagicMock(spec=asyncio.Runner)
        stage._runner = mock_runner

        with patch.object(vllm_async_stage, "gpu_stage_cleanup"):
            stage.destroy()

        mock_runner.close.assert_called_once()

    def test_destroy_handles_runner_close_failure(self) -> None:
        """destroy() should log warning and continue when runner.close() raises."""
        config = self._make_config()
        stage = VllmAsyncCaptionStage(serve_config=config, model_name="qwen")
        mock_runner = MagicMock(spec=asyncio.Runner)
        mock_runner.close.side_effect = RuntimeError("event loop already closed")
        stage._runner = mock_runner

        mock_engine = MagicMock()
        stage._engine = mock_engine

        with patch.object(vllm_async_stage, "gpu_stage_cleanup") as mock_cleanup:
            stage.destroy()

        mock_runner.close.assert_called_once()
        mock_engine.shutdown.assert_called_once()
        assert stage._engine is None
        mock_cleanup.assert_called_once_with("VllmAsyncCaptionStage")

    def test_process_all_tasks_async_early_return_on_empty_windows(self) -> None:
        """_process_all_tasks_async should return immediately when no rendered windows exist."""
        config = self._make_config()
        stage = VllmAsyncCaptionStage(serve_config=config, model_name="qwen")
        stage._engine = MagicMock()
        stage._sampling_params = MagicMock()

        task = _make_task(b"\x00", num_windows=1)

        with patch.object(stage, "_extract_rendered_windows", return_value=[]) as mock_extract:
            asyncio.run(stage._process_all_tasks_async([task]))

        mock_extract.assert_called_once_with([task])

    def test_runner_preserves_background_tasks_across_calls(self) -> None:
        """asyncio.Runner should keep background tasks alive between run() calls."""
        config = self._make_config()
        stage = VllmAsyncCaptionStage(serve_config=config, model_name="qwen")
        survived: list[bool] = []

        async def bg_task() -> None:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                survived.append(False)
                raise

        async def first_call() -> None:
            asyncio.get_running_loop().create_task(bg_task())

        async def second_call() -> None:
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            survived.append(len(tasks) > 0)

        stage._runner.run(first_call())
        stage._runner.run(second_call())
        assert survived == [True]
        stage._runner.close()

    def test_pickle_roundtrip_recreates_runner(self) -> None:
        """Pickle roundtrip should exclude _runner and recreate a usable one."""
        config = self._make_config()
        stage = VllmAsyncCaptionStage(serve_config=config, model_name="qwen")

        data = pickle.dumps(stage)
        restored = pickle.loads(data)  # noqa: S301

        assert isinstance(restored._runner, asyncio.Runner)
        restored._runner.run(asyncio.sleep(0))
        restored._runner.close()

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

    def test_process_data_captions_windows(self) -> None:
        """process_data should extract pre-rendered ProcessorInputs and caption windows."""
        config = self._make_config()
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
            max_concurrent_requests=1,
        )

        mock_engine = MagicMock()
        mock_engine.generate = MagicMock(side_effect=_async_gen_side_effect(_mock_request_output("A cat video")))
        stage._engine = mock_engine
        stage._sampling_params = MagicMock()

        task = _make_task(b"\x00\x01")
        fake_frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        rendered_mock = MagicMock()
        for clip in task.video.clips:
            for window in clip.windows:
                window.model_input["vllm_async"] = {
                    "rendered_prompt": rendered_mock,
                    "frames_shape": fake_frames.shape,
                    **_rendered_prompt_state(fake_frames, prompt="describe"),
                }
        result = stage.process_data([task])

        window = result[0].video.clips[0].windows[0]
        assert window.caption["vllm_async"] == "A cat video"
        assert window.caption_status == "success"
        assert window.caption_failure_reason is None
        assert not result[0].video.clips[0].errors

    def test_process_data_multiple_tasks_gathers_all_windows(self) -> None:
        """process_data with multiple tasks should caption all windows."""
        config = self._make_config()
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
            max_concurrent_requests=4,
        )

        mock_engine = MagicMock()
        mock_engine.generate = MagicMock(side_effect=_async_gen_side_effect(_mock_request_output("Caption")))
        stage._engine = mock_engine
        stage._sampling_params = MagicMock()

        task1 = _make_task(b"\x00\x01", num_windows=2)
        task2 = _make_task(b"\x00\x02", num_windows=2)
        fake_frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        for task in [task1, task2]:
            for clip in task.video.clips:
                for window in clip.windows:
                    window.model_input["vllm_async"] = {
                        "rendered_prompt": MagicMock(),
                        "frames_shape": fake_frames.shape,
                        **_rendered_prompt_state(fake_frames, prompt="describe"),
                    }
        result = stage.process_data([task1, task2])

        assert len(result) == 2
        for task in result:
            for clip in task.video.clips:
                for window in clip.windows:
                    assert window.caption.get("vllm_async") == "Caption"
                    assert window.caption_status == "success"
                    assert window.caption_failure_reason is None
                assert not clip.errors

    def test_full_lifecycle_setup_caption_destroy(self) -> None:
        """End-to-end: stage_setup -> process_data -> destroy composes correctly."""
        config = self._make_config()
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
            max_concurrent_requests=1,
        )

        mock_engine = MagicMock()
        mock_engine.generate = MagicMock(side_effect=_async_gen_side_effect(_mock_request_output("Test caption")))
        _mock_renderer(mock_engine)

        mock_async_llm_cls = MagicMock()
        mock_async_llm_cls.from_engine_args.return_value = mock_engine

        with (
            patch.object(vllm_async_stage, "get_vllm_model_id", return_value="Qwen/Qwen2.5-VL-7B-Instruct"),
            patch.object(vllm_async_stage, "resolve_model_path", return_value="/config/models/Qwen"),
            patch.object(vllm_async_stage, "_build_engine_args", return_value=MagicMock()),
            patch.object(vllm_async_stage, "AsyncLLM", mock_async_llm_cls, create=True),
            patch.object(vllm_async_stage, "gpu_stage_startup"),
            patch.object(vllm_async_stage, "gpu_stage_cleanup"),
            patch.object(vllm_async_stage, "build_sampling_params", return_value=MagicMock(), create=True),
        ):
            # Phase 1: setup
            stage.stage_setup()
            mock_async_llm_cls.from_engine_args.assert_called_once()
            assert stage._engine is not None

            # Phase 2: caption (pre-populate model_input as render stage would)
            task = _make_task(b"\x00\x01")
            fake_frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
            for clip in task.video.clips:
                for window in clip.windows:
                    window.model_input["vllm_async"] = {
                        "rendered_prompt": MagicMock(),
                        "frames_shape": fake_frames.shape,
                        **_rendered_prompt_state(fake_frames),
                    }
            result = stage.process_data([task])
            assert result[0].video.clips[0].windows[0].caption["vllm_async"] == "Test caption"

            # Phase 3: destroy
            stage.destroy()
            mock_engine.shutdown.assert_called_once()
            assert stage._engine is None

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


class TestEngineDeath:
    """Tests for crash-on-death engine failure handling.

    When ``EngineDeadError`` occurs, the stage completes the current
    batch (other windows may still succeed), then raises ``RuntimeError``
    in ``process_data`` to crash the actor. Xenna automatically
    replaces the dead actor with a fresh one.
    """

    def _make_config(self, **overrides: object) -> VllmAsyncConfig:
        defaults: dict[str, object] = {
            "model_variant": "qwen",
            "num_gpus": 1,
        }
        defaults.update(overrides)
        return VllmAsyncConfig(**defaults)

    def _make_stage(self, **config_overrides: object) -> VllmAsyncCaptionStage:
        config = self._make_config(**config_overrides)
        return VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
            max_concurrent_requests=4,
        )

    def test_engine_dead_crashes_actor(self) -> None:
        """When EngineDeadError is raised, process_data should raise RuntimeError after batch."""
        stage = self._make_stage()

        engine_dead_error = vllm_async_stage.EngineDeadError("OOM killed")

        async def _always_fail(**_kwargs: object) -> AsyncGenerator[MagicMock, None]:
            raise engine_dead_error
            yield  # type: ignore[misc]  # unreachable, makes it an async generator

        mock_engine = MagicMock()
        mock_engine.generate = MagicMock(side_effect=_always_fail)
        stage._engine = mock_engine
        stage._sampling_params = MagicMock()

        fake_frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        task = _make_task(b"\x00", num_windows=1)
        for clip in task.video.clips:
            for window in clip.windows:
                window.model_input["vllm_async"] = {
                    "rendered_prompt": MagicMock(),
                    "frames_shape": fake_frames.shape,
                    **_rendered_prompt_state(fake_frames),
                }
        with pytest.raises(RuntimeError, match="EngineDeadError"):
            stage.process_data([task])

        assert stage._engine_dead is True

    def test_engine_dead_completes_batch_before_crash(self) -> None:
        """Other windows in the batch should still be processed before actor crashes."""
        stage = self._make_stage()

        engine_dead_error = vllm_async_stage.EngineDeadError("OOM killed")
        call_count = 0

        async def _fail_then_succeed(**_kwargs: object) -> AsyncGenerator[MagicMock, None]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise engine_dead_error
            yield _mock_request_output("Still OK")

        mock_engine = MagicMock()
        mock_engine.generate = MagicMock(side_effect=_fail_then_succeed)
        stage._engine = mock_engine
        stage._sampling_params = MagicMock()

        fake_frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        task = _make_task(b"\x00", num_windows=2)
        for clip in task.video.clips:
            for window in clip.windows:
                window.model_input["vllm_async"] = {
                    "rendered_prompt": MagicMock(),
                    "frames_shape": fake_frames.shape,
                    **_rendered_prompt_state(fake_frames),
                }
        with pytest.raises(RuntimeError, match="EngineDeadError"):
            stage.process_data([task])

        clip = task.video.clips[0]
        succeeded = sum(1 for w in clip.windows if "vllm_async" in w.caption)
        errored = sum(1 for k in clip.errors if "caption" in k)
        assert succeeded + errored == 2, "All windows should have been processed"


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
        assert json.loads(config.mm_processor_kwargs) == {"max_pixels": VIDEO_MAX_PIXELS}
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

    def test_direct_construction_returns_field_default_not_qwen_override(self) -> None:
        """Direct construction returns the field default instead of the builder's qwen override."""
        config = VllmAsyncConfig(model_variant="qwen")
        assert json.loads(config.mm_processor_kwargs) == {"max_pixels": VIDEO_MAX_PIXELS}

    def test_mm_processor_kwargs_wired_to_engine_args(self) -> None:
        """mm_processor_kwargs JSON should be parsed to a dict in AsyncEngineArgs."""
        with self._patch_engine() as (mock_engine_args_cls, _):
            config = VllmAsyncConfig(
                model_variant="test/model",
                mm_processor_kwargs=json.dumps(_qwen_video_mm_processor_kwargs()),
            )
            _build_engine_args(config, "/tmp/model")  # noqa: S108
        assert mock_engine_args_cls.call_args.kwargs["mm_processor_kwargs"] == _qwen_video_mm_processor_kwargs()

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
        """When CLI does not set a field, the builder-managed qwen defaults apply."""
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
        assert json.loads(config.mm_processor_kwargs) == _qwen_video_mm_processor_kwargs()
        assert config.skip_mm_profiling is True

    def test_non_qwen_mm_processor_kwargs_keep_legacy_field_default(self) -> None:
        """Non-Qwen variants should keep the legacy field default when no model override applies."""
        args = self._make_args(vllm_async_model_name="nemotron")
        config = build_vllm_async_config(args, sampling_config=self._default_sampling_config())

        assert config is not None
        assert json.loads(config.mm_processor_kwargs) == {"max_pixels": VIDEO_MAX_PIXELS}

    def test_empty_string_mm_processor_kwargs_disables_qwen_override_only(self) -> None:
        """Explicit empty CLI value should disable Qwen's builder override and fall back to the field default."""
        args = self._make_args(
            vllm_async_model_name="qwen",
            vllm_async_mm_processor_kwargs="",
        )
        config = build_vllm_async_config(args, sampling_config=self._default_sampling_config())

        assert config is not None
        assert json.loads(config.mm_processor_kwargs) == {"max_pixels": VIDEO_MAX_PIXELS}

    def test_cli_overrides_model_default(self) -> None:
        """Explicit CLI value should override the builder-managed default for that model."""
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
        assert stored["prompt_input"]["prompt"] == "<prompt>describe</prompt>"
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
        fake_frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="Prompt template not initialized"):
            stage._build_prompt(fake_frames)

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


class TestCaptionStageModelInputExtraction:
    """Tests for _extract_rendered_windows from pre-rendered model_input."""

    def _make_stage(self) -> VllmAsyncCaptionStage:
        config = VllmAsyncConfig(model_variant="qwen", num_gpus=1)
        return VllmAsyncCaptionStage(serve_config=config, model_name="qwen")

    def test_extracts_pre_rendered_windows(self) -> None:
        """When model_input[variant] has rendered_prompt, should build _RenderedWindow."""
        stage = self._make_stage()
        stage._sampling_params = MagicMock()

        frames_shape = (10, 384, 672, 3)
        rendered_mock = MagicMock()
        frames = np.zeros((10, 2, 2, 3), dtype=np.uint8)

        task = _make_task(b"\x00", num_windows=1)
        window = task.video.clips[0].windows[0]
        window.model_input["vllm_async"] = {
            "rendered_prompt": rendered_mock,
            "frames_shape": frames_shape,
            **_rendered_prompt_state(frames),
        }

        result = stage._extract_rendered_windows([task])
        assert len(result) == 1
        rw = result[0]
        assert isinstance(rw, _RenderedWindow)
        assert rw.rendered_prompt is rendered_mock
        assert rw.frames_shape == frames_shape
        assert rw.raw_prompt == "test"
        assert rw.decoded_rgb_frames is frames
        assert "vllm_async" not in window.model_input

    def test_raises_on_missing_rendered_prompt(self) -> None:
        """_extract_rendered_windows should record error when rendered_prompt is missing."""
        stage = self._make_stage()
        stage._sampling_params = MagicMock()

        task = _make_task(b"\x00", num_windows=1)
        clip = task.video.clips[0]
        clip.windows[0].model_input["vllm_async"] = {
            "frames_shape": (4, 224, 224, 3),
        }

        result = stage._extract_rendered_windows([task])
        assert len(result) == 0
        assert "vllm_async_caption_0" in clip.errors
        assert "VllmAsyncPromptRenderStage" in clip.errors["vllm_async_caption_0"]
        assert clip.windows[0].caption_status == "error"
        assert clip.windows[0].caption_failure_reason == "exception"

    def test_skips_missing_model_input(self) -> None:
        """Windows without model_input[variant] should be silently skipped."""
        stage = self._make_stage()
        stage._sampling_params = MagicMock()

        task = _make_task(b"\x00", num_windows=1)
        result = stage._extract_rendered_windows([task])
        assert len(result) == 0

    def test_cleanup_pops_model_input(self) -> None:
        """model_input[variant] should be popped after extraction."""
        stage = self._make_stage()
        stage._sampling_params = MagicMock()

        task = _make_task(b"\x00", num_windows=2)
        for window in task.video.clips[0].windows:
            window.model_input["vllm_async"] = {
                "rendered_prompt": MagicMock(),
                "frames_shape": (4, 224, 224, 3),
            }

        stage._extract_rendered_windows([task])
        for window in task.video.clips[0].windows:
            assert "vllm_async" not in window.model_input


class TestStage2CaptionRefinement:
    """Tests for stage-2 caption refinement in VllmAsyncCaptionStage."""

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
        """stage2_caption should default to False."""
        stage = self._make_stage()
        assert stage._stage2_caption is False
        assert stage._stage2_prompt_text is None
        assert stage._stage2_processor is None

    def test_stage2_enabled_stores_params(self) -> None:
        """When stage2_caption=True, init should store the parameters."""
        stage = self._make_stage(stage2_caption=True, stage2_prompt_text="Refine this.")
        assert stage._stage2_caption is True
        assert stage._stage2_prompt_text == "Refine this."

    def test_getstate_excludes_stage2_processor(self) -> None:
        """__getstate__ should exclude _stage2_processor from pickle state."""
        stage = self._make_stage(stage2_caption=True)
        stage._stage2_processor = MagicMock()
        state = stage.__getstate__()
        assert "_stage2_processor" not in state

    def test_setstate_restores_stage2_processor_as_none(self) -> None:
        """__setstate__ should set _stage2_processor to None."""
        stage = self._make_stage(stage2_caption=True)
        state = stage.__getstate__()
        stage.__setstate__(state)
        assert stage._stage2_processor is None

    def test_generate_and_assign_stage1_only(self) -> None:
        """Without stage2, _generate_and_assign assigns caption directly and cleans rendered_prompt."""
        stage = self._make_stage(stage2_caption=False)
        stage._engine = MagicMock()
        stage._sampling_params = MagicMock()

        window = Window(start_frame=0, end_frame=10)
        frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 1.0))
        clip.windows = [window]

        rw = _RenderedWindow(
            clip=clip,
            window_index=0,
            window=window,
            rendered_prompt=MagicMock(),
            sampling_params=MagicMock(),
            frames_shape=(4, 224, 224, 3),
            raw_prompt="test",
            decoded_rgb_frames=frames,
        )

        stage2_queue: collections.deque[tuple[_RenderedWindow, str]] = collections.deque()

        with patch.object(
            stage, "_generate_caption_async", return_value=("A test caption", TokenCounts(10, 5))
        ) as mock_gen:
            result = asyncio.run(stage._generate_and_assign(rw, asyncio.Semaphore(1), stage2_queue))

        assert result is None
        assert mock_gen.call_count == 1
        assert window.caption["vllm_async"] == "A test caption"
        assert window.caption_status == "success"
        assert window.caption_failure_reason is None
        assert rw.rendered_prompt is None
        assert rw.raw_prompt is None
        assert rw.decoded_rgb_frames is None  # cleaned: uint8 frames released when no stage-2
        assert len(stage2_queue) == 0

    def test_generate_and_assign_stage2_enqueues(self) -> None:
        """With stage2 enabled, _generate_and_assign enqueues to stage2_queue instead of assigning."""
        stage = self._make_stage(stage2_caption=True, stage2_prompt_text="Refine it.")
        stage._engine = MagicMock()
        stage._sampling_params = MagicMock()
        stage._stage2_processor = MagicMock()

        window = Window(start_frame=0, end_frame=10)
        frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 1.0))
        clip.windows = [window]

        rw = _RenderedWindow(
            clip=clip,
            window_index=0,
            window=window,
            rendered_prompt=MagicMock(),
            sampling_params=MagicMock(),
            frames_shape=(4, 224, 224, 3),
            raw_prompt="test",
            decoded_rgb_frames=frames,
        )

        stage2_queue: collections.deque[tuple[_RenderedWindow, str]] = collections.deque()

        with patch.object(
            stage, "_generate_caption_async", return_value=("stage1 caption", TokenCounts(10, 5))
        ) as mock_gen:
            asyncio.run(stage._generate_and_assign(rw, asyncio.Semaphore(1), stage2_queue))

        assert mock_gen.call_count == 1
        assert len(stage2_queue) == 1
        queued_rw, queued_caption = stage2_queue[0]
        assert queued_rw is rw
        assert queued_caption == "stage1 caption"
        assert "vllm_async" not in window.caption
        assert rw.rendered_prompt is None

    def test_generate_and_assign_skips_stage2_when_processor_none(self) -> None:
        """Stage-2 should not enqueue if _stage2_processor is None (setup not called)."""
        stage = self._make_stage(stage2_caption=True)
        stage._engine = MagicMock()
        stage._sampling_params = MagicMock()

        window = Window(start_frame=0, end_frame=10)
        frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 1.0))
        clip.windows = [window]

        rw = _RenderedWindow(
            clip=clip,
            window_index=0,
            window=window,
            rendered_prompt=MagicMock(),
            sampling_params=MagicMock(),
            frames_shape=(4, 224, 224, 3),
            raw_prompt="test",
            decoded_rgb_frames=frames,
        )

        stage2_queue: collections.deque[tuple[_RenderedWindow, str]] = collections.deque()

        with patch.object(
            stage, "_generate_caption_async", return_value=("Only caption", TokenCounts(10, 5))
        ) as mock_gen:
            asyncio.run(stage._generate_and_assign(rw, asyncio.Semaphore(1), stage2_queue))

        assert mock_gen.call_count == 1
        assert window.caption["vllm_async"] == "Only caption"
        assert window.caption_status == "success"
        assert window.caption_failure_reason is None
        assert len(stage2_queue) == 0

    def test_generate_and_assign_error_cleans_decoded_rgb_frames(self) -> None:
        """Error in _generate_caption_async must still clean decoded frames (no memory leak)."""
        stage = self._make_stage(stage2_caption=False)
        stage._engine = MagicMock()
        stage._sampling_params = MagicMock()

        window = Window(start_frame=0, end_frame=10)
        frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 1.0))
        clip.windows = [window]

        rw = _RenderedWindow(
            clip=clip,
            window_index=0,
            window=window,
            rendered_prompt=MagicMock(),
            sampling_params=MagicMock(),
            frames_shape=(4, 224, 224, 3),
            raw_prompt="test",
            decoded_rgb_frames=frames,
        )

        stage2_queue: collections.deque[tuple[_RenderedWindow, str]] = collections.deque()

        with patch.object(stage, "_generate_caption_async", side_effect=RuntimeError("engine boom")):
            asyncio.run(stage._generate_and_assign(rw, asyncio.Semaphore(1), stage2_queue))

        assert rw.rendered_prompt is None
        assert rw.raw_prompt is None
        assert rw.decoded_rgb_frames is None  # must be cleaned on error path too
        assert "vllm_async_caption_0" in clip.errors
        assert "vllm_async" not in window.caption
        assert window.caption_status == "error"
        assert window.caption_failure_reason == "exception"

    def test_rendered_window_cleanup_after_processing(self) -> None:
        """Full lifecycle: generate cleans rendered_prompt, stage2 cleans decoded frames."""
        stage = self._make_stage(stage2_caption=True, stage2_prompt_text="Refine it.")
        mock_engine = MagicMock()
        _mock_renderer(mock_engine)
        stage._engine = mock_engine
        stage._sampling_params = MagicMock()
        stage._stage2_processor = MagicMock()

        window = Window(start_frame=0, end_frame=10)
        frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 1.0))
        clip.windows = [window]

        rw = _RenderedWindow(
            clip=clip,
            window_index=0,
            window=window,
            rendered_prompt=MagicMock(),
            sampling_params=MagicMock(),
            frames_shape=(4, 224, 224, 3),
            raw_prompt="test",
            decoded_rgb_frames=frames,
        )

        stage2_queue: collections.deque[tuple[_RenderedWindow, str]] = collections.deque()

        with patch.object(stage, "_generate_caption_async", return_value=("Cap1", TokenCounts(10, 5))):
            asyncio.run(stage._generate_and_assign(rw, asyncio.Semaphore(1), stage2_queue))

        assert rw.rendered_prompt is None
        assert rw.decoded_rgb_frames is not None
        assert len(stage2_queue) == 1

        queued_rw, stage1_caption = stage2_queue[0]
        with (
            patch.object(stage, "_generate_caption_async", return_value=("Cap2", TokenCounts(10, 5))),
            patch(
                "cosmos_curator.pipelines.video.captioning.vllm_async_stage.build_refinement_prompt_text",
                return_value="<refined>",
            ),
        ):
            asyncio.run(stage._stage2_refine_and_assign(queued_rw, stage1_caption, mock_engine, asyncio.Semaphore(1)))

        assert rw.rendered_prompt is None
        assert rw.raw_prompt is None
        assert rw.decoded_rgb_frames is None
        assert window.caption_status == "success"
        assert window.caption_failure_reason is None

    def test_stage2_rebuilds_fresh_prompt_payload(self) -> None:
        """Stage-2 should rebuild a fresh render payload and reparse processor kwargs."""
        config = VllmAsyncConfig(
            model_variant="qwen",
            num_gpus=1,
            mm_processor_kwargs=json.dumps(_qwen_video_mm_processor_kwargs()),
        )
        stage = VllmAsyncCaptionStage(
            serve_config=config,
            model_name="qwen",
            stage2_caption=True,
            stage2_prompt_text="Refine it.",
        )
        mock_engine = MagicMock()
        _mock_renderer(mock_engine)
        stage._engine = mock_engine
        stage._sampling_params = MagicMock()
        stage._stage2_processor = MagicMock()

        window = Window(start_frame=0, end_frame=10)
        frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 1.0))
        clip.windows = [window]

        rw = _RenderedWindow(
            clip=clip,
            window_index=0,
            window=window,
            rendered_prompt=None,
            sampling_params=MagicMock(),
            frames_shape=(4, 224, 224, 3),
            raw_prompt="test",
            decoded_rgb_frames=frames,
        )

        with (
            patch.object(stage, "_generate_caption_async", return_value=("Refined caption", TokenCounts(10, 5))),
            patch(
                "cosmos_curator.pipelines.video.captioning.vllm_async_stage.build_refinement_prompt_text",
                return_value="<refined>",
            ),
        ):
            asyncio.run(stage._stage2_refine_and_assign(rw, "Stage-1 text", mock_engine, asyncio.Semaphore(1)))

        (render_prompt,) = mock_engine.renderer.render_cmpl_async.call_args.args[0]
        assert render_prompt["prompt"] == "<refined>"
        assert render_prompt["multi_modal_data"]["video"][0] is frames
        assert render_prompt["mm_processor_kwargs"] == _qwen_video_mm_processor_kwargs()

        render_prompt["mm_processor_kwargs"]["videos_kwargs"]["size"]["shortest_edge"] = 1
        second_payload = _build_render_payload("<refined-2>", frames, config.mm_processor_kwargs)
        assert second_payload["mm_processor_kwargs"]["videos_kwargs"]["size"]["shortest_edge"] == VIDEO_MIN_PIXELS
        assert window.caption["vllm_async"] == "Refined caption"

    def test_stage2_failure_falls_back_to_stage1_caption(self) -> None:
        """If stage-2 generate fails, _stage2_refine_and_assign falls back to stage-1 caption."""
        stage = self._make_stage(stage2_caption=True, stage2_prompt_text="Refine it.")
        mock_engine = MagicMock()
        _mock_renderer(mock_engine)
        stage._engine = mock_engine
        stage._sampling_params = MagicMock()
        stage._stage2_processor = MagicMock()

        window = Window(start_frame=0, end_frame=10)
        frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 1.0))
        clip.windows = [window]

        rw = _RenderedWindow(
            clip=clip,
            window_index=0,
            window=window,
            rendered_prompt=None,
            sampling_params=MagicMock(),
            frames_shape=(4, 224, 224, 3),
            raw_prompt="test",
            decoded_rgb_frames=frames,
        )

        with (
            patch.object(
                stage,
                "_generate_caption_async",
                side_effect=RuntimeError("stage-2 model error"),
            ),
            patch(
                "cosmos_curator.pipelines.video.captioning.vllm_async_stage.build_refinement_prompt_text",
                return_value="<refined>",
            ),
        ):
            asyncio.run(stage._stage2_refine_and_assign(rw, "Stage-1 caption", mock_engine, asyncio.Semaphore(1)))

        assert window.caption["vllm_async"] == "Stage-1 caption"
        assert window.caption_status == "success"
        assert window.caption_failure_reason is None
        assert rw.raw_prompt is None
        assert rw.decoded_rgb_frames is None
        err_key = f"{stage._model_variant}_caption_0"
        assert err_key in clip.errors
        assert "stage-2 refinement failed" in clip.errors[err_key]

    def test_custom_stage2_prompt_text_forwarded_to_builder(self) -> None:
        """Custom stage2_prompt_text must reach build_refinement_prompt_text, not the default."""
        custom_prompt = "Describe the scene with cinematic detail."
        stage = self._make_stage(stage2_caption=True, stage2_prompt_text=custom_prompt)
        mock_engine = MagicMock()
        _mock_renderer(mock_engine)
        stage._engine = mock_engine
        stage._sampling_params = MagicMock()
        stage._stage2_processor = MagicMock()

        window = Window(start_frame=0, end_frame=10)
        frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 1.0))
        clip.windows = [window]

        rw = _RenderedWindow(
            clip=clip,
            window_index=0,
            window=window,
            rendered_prompt=None,
            sampling_params=MagicMock(),
            frames_shape=(4, 224, 224, 3),
            raw_prompt="test",
            decoded_rgb_frames=frames,
        )

        with (
            patch.object(stage, "_generate_caption_async", return_value=("Refined caption", TokenCounts(10, 5))),
            patch(
                "cosmos_curator.pipelines.video.captioning.vllm_async_stage.build_refinement_prompt_text",
                return_value="<rendered>",
            ) as mock_build,
        ):
            asyncio.run(stage._stage2_refine_and_assign(rw, "Stage-1 text", mock_engine, asyncio.Semaphore(1)))

        mock_build.assert_called_once_with(stage._stage2_processor, "Stage-1 text", custom_prompt)
        assert window.caption["vllm_async"] == "Refined caption"
        assert window.caption_status == "success"
        assert window.caption_failure_reason is None

    def test_default_stage2_prompt_text_forwarded_as_none(self) -> None:
        """When no custom prompt is set, build_refinement_prompt_text receives None (uses default)."""
        stage = self._make_stage(stage2_caption=True, stage2_prompt_text=None)
        mock_engine = MagicMock()
        _mock_renderer(mock_engine)
        stage._engine = mock_engine
        stage._sampling_params = MagicMock()
        stage._stage2_processor = MagicMock()

        window = Window(start_frame=0, end_frame=10)
        frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 1.0))
        clip.windows = [window]

        rw = _RenderedWindow(
            clip=clip,
            window_index=0,
            window=window,
            rendered_prompt=None,
            sampling_params=MagicMock(),
            frames_shape=(4, 224, 224, 3),
            raw_prompt="test",
            decoded_rgb_frames=frames,
        )

        with (
            patch.object(stage, "_generate_caption_async", return_value=("Refined caption", TokenCounts(10, 5))),
            patch(
                "cosmos_curator.pipelines.video.captioning.vllm_async_stage.build_refinement_prompt_text",
                return_value="<rendered>",
            ) as mock_build,
        ):
            asyncio.run(stage._stage2_refine_and_assign(rw, "Stage-1 text", mock_engine, asyncio.Semaphore(1)))

        mock_build.assert_called_once_with(stage._stage2_processor, "Stage-1 text", None)
        assert window.caption["vllm_async"] == "Refined caption"
        assert window.caption_status == "success"
        assert window.caption_failure_reason is None

    def test_stage2_skips_when_prompt_primitives_are_missing(self) -> None:
        """When prompt primitives are missing, stage-2 should assign stage-1 caption and return early."""
        stage = self._make_stage(stage2_caption=True, stage2_prompt_text="Refine it.")
        mock_engine = MagicMock()
        _mock_renderer(mock_engine)
        stage._engine = mock_engine
        stage._sampling_params = MagicMock()
        stage._stage2_processor = MagicMock()

        window = Window(start_frame=0, end_frame=10)
        clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 1.0))
        clip.windows = [window]

        rw = _RenderedWindow(
            clip=clip,
            window_index=0,
            window=window,
            rendered_prompt=None,
            sampling_params=MagicMock(),
            frames_shape=(4, 224, 224, 3),
            raw_prompt=None,
            decoded_rgb_frames=None,
        )

        asyncio.run(stage._stage2_refine_and_assign(rw, "Stage-1 only", mock_engine, asyncio.Semaphore(1)))

        assert window.caption["vllm_async"] == "Stage-1 only"
        assert window.caption_status == "success"
        assert window.caption_failure_reason is None
        mock_engine.renderer.render_cmpl_async.assert_not_called()

    def test_stage2_engine_dead_error_sets_flag_and_falls_back(self) -> None:
        """EngineDeadError during stage-2 should set _engine_dead, record error, and use stage-1 caption."""
        stage = self._make_stage(stage2_caption=True, stage2_prompt_text="Refine it.")
        mock_engine = MagicMock()
        _mock_renderer(mock_engine)
        stage._engine = mock_engine
        stage._sampling_params = MagicMock()
        stage._stage2_processor = MagicMock()

        window = Window(start_frame=0, end_frame=10)
        frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 1.0))
        clip.windows = [window]

        rw = _RenderedWindow(
            clip=clip,
            window_index=0,
            window=window,
            rendered_prompt=None,
            sampling_params=MagicMock(),
            frames_shape=(4, 224, 224, 3),
            raw_prompt="test",
            decoded_rgb_frames=frames,
        )

        with (
            patch.object(
                stage,
                "_generate_caption_async",
                side_effect=vllm_async_stage.EngineDeadError("GPU OOM"),
            ),
            patch(
                "cosmos_curator.pipelines.video.captioning.vllm_async_stage.build_refinement_prompt_text",
                return_value="<refined>",
            ),
        ):
            asyncio.run(stage._stage2_refine_and_assign(rw, "Fallback caption", mock_engine, asyncio.Semaphore(1)))

        assert stage._engine_dead is True
        assert window.caption["vllm_async"] == "Fallback caption"
        assert window.caption_status == "success"
        assert window.caption_failure_reason is None
        err_key = f"{stage._model_variant}_caption_0"
        assert err_key in clip.errors
        assert "EngineDeadError during stage-2" in clip.errors[err_key]
        assert rw.raw_prompt is None
        assert rw.decoded_rgb_frames is None


class TestVllmAsyncPromptRenderStage:
    """Tests for VllmAsyncPromptRenderStage -- standalone Renderer on CPU actors."""

    def _make_stage(self, **overrides: object) -> VllmAsyncPromptRenderStage:
        defaults = {"model_variant": "qwen", "num_gpus": 1}
        defaults.update(overrides)
        config = VllmAsyncConfig(**defaults)
        return VllmAsyncPromptRenderStage(
            serve_config=config,
            model_name="qwen",
        )

    def test_resources_cpu_only(self) -> None:
        """Render stage should request 0.5 CPU and 0 GPUs."""
        stage = self._make_stage()
        assert stage.resources.cpus == 0.5
        assert stage.resources.gpus == 0

    def test_conda_env_name_is_unified(self) -> None:
        """Render stage must run in the unified pixi environment."""
        stage = self._make_stage()
        assert stage.conda_env_name == "unified"

    def test_model_returns_vllm_async_model(self) -> None:
        """Render stage should provide a model interface for weight download."""
        stage = self._make_stage()
        assert isinstance(stage.model, _VllmAsyncModel)

    def test_secondary_name_returns_variant(self) -> None:
        """secondary_name should return the model variant."""
        stage = self._make_stage()
        assert stage.secondary_name() == "vllm_async"

    def test_process_data_renders_all_windows(self) -> None:
        """process_data should render TextPrompt dicts to ProcessorInputs."""
        stage = self._make_stage()

        mock_renderer = MagicMock()
        rendered_mock = MagicMock()
        mock_renderer.render_cmpl_async = AsyncMock(return_value=[rendered_mock])
        stage._renderer = mock_renderer
        stage._runner = asyncio.Runner()

        task = _make_task(b"\x00\x01", num_windows=1)
        fake_frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        prompt_input = {"prompt": "describe", "multi_modal_data": {"video": [fake_frames]}}
        for clip in task.video.clips:
            for window in clip.windows:
                window.model_input["vllm_async"] = {
                    "prompt_input": prompt_input,
                    "frames_shape": fake_frames.shape,
                }

        result = stage.process_data([task])
        window = result[0].video.clips[0].windows[0]

        rendered_data = window.model_input.get("vllm_async")
        assert rendered_data is not None
        assert rendered_data["rendered_prompt"] is rendered_mock
        assert rendered_data["frames_shape"] == fake_frames.shape
        assert rendered_data["raw_prompt"] == "describe"
        assert rendered_data["decoded_rgb_frames"] is fake_frames
        stage._runner.close()

    def test_process_data_omits_mm_processor_kwargs_when_serve_config_is_empty(self) -> None:
        """Render stage should omit prompt-level processor kwargs when serve_config carries no override."""
        stage = self._make_stage(mm_processor_kwargs="")

        mock_renderer = MagicMock()
        mock_renderer.render_cmpl_async = AsyncMock(return_value=[MagicMock()])
        stage._renderer = mock_renderer
        stage._runner = asyncio.Runner()

        task = _make_task(b"\x00\x01", num_windows=1)
        fake_frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        prompt_input = {"prompt": "describe", "multi_modal_data": {"video": [fake_frames]}}
        for clip in task.video.clips:
            for window in clip.windows:
                window.model_input["vllm_async"] = {
                    "prompt_input": prompt_input,
                    "frames_shape": fake_frames.shape,
                }

        stage.process_data([task])

        (render_prompt,) = mock_renderer.render_cmpl_async.call_args.args[0]
        assert "mm_processor_kwargs" not in render_prompt
        stage._runner.close()

    def test_process_data_rebuilds_fresh_prompt_payloads_per_window(self) -> None:
        """Render stage should isolate prompt payloads across windows in the same render batch."""
        mm_processor_kwargs = json.dumps(_qwen_video_mm_processor_kwargs())
        stage = self._make_stage(mm_processor_kwargs=mm_processor_kwargs)

        mock_renderer = MagicMock()
        rendered_mocks = [MagicMock(), MagicMock()]

        async def _render(prompts: list[dict[str, object]]) -> list[MagicMock]:
            assert len(prompts) == 2
            assert prompts[0] is not prompts[1]
            assert prompts[0]["multi_modal_data"] is not prompts[1]["multi_modal_data"]
            assert prompts[0]["multi_modal_data"]["video"] is not prompts[1]["multi_modal_data"]["video"]
            assert prompts[0]["mm_processor_kwargs"] is not prompts[1]["mm_processor_kwargs"]

            prompts[0]["mm_processor_kwargs"]["videos_kwargs"]["size"]["shortest_edge"] = 1
            assert prompts[1]["mm_processor_kwargs"]["videos_kwargs"]["size"]["shortest_edge"] == VIDEO_MIN_PIXELS
            return rendered_mocks

        mock_renderer.render_cmpl_async = AsyncMock(side_effect=_render)
        stage._renderer = mock_renderer
        stage._runner = asyncio.Runner()

        task = _make_task(b"\x00\x01", num_windows=2)
        fake_frames_by_window: list[np.ndarray] = []
        for wi, window in enumerate(task.video.clips[0].windows):
            fake_frames = np.full((4, 224, 224, 3), wi, dtype=np.uint8)
            prompt_input = {"prompt": f"describe-{wi}", "multi_modal_data": {"video": [fake_frames]}}
            fake_frames_by_window.append(fake_frames)
            window.model_input["vllm_async"] = {
                "prompt_input": prompt_input,
                "frames_shape": fake_frames.shape,
            }

        result = stage.process_data([task])
        windows = result[0].video.clips[0].windows
        for wi, window in enumerate(windows):
            rendered_data = window.model_input.get("vllm_async")
            assert rendered_data is not None
            assert rendered_data["rendered_prompt"] is rendered_mocks[wi]
            assert rendered_data["raw_prompt"] == f"describe-{wi}"
            assert rendered_data["decoded_rgb_frames"] is fake_frames_by_window[wi]

        stage._runner.close()

    def test_process_data_chunk_render_failure_records_errors(self) -> None:
        """Chunk render failure should record errors on all windows in the chunk."""
        stage = self._make_stage()

        mock_renderer = MagicMock()
        mock_renderer.render_cmpl_async = AsyncMock(side_effect=RuntimeError("render boom"))
        stage._renderer = mock_renderer
        stage._runner = asyncio.Runner()

        task = _make_task(b"\x00", num_windows=1)
        fake_frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        for clip in task.video.clips:
            for window in clip.windows:
                window.model_input["vllm_async"] = {
                    "prompt_input": {"prompt": "test", "multi_modal_data": {"video": [fake_frames]}},
                    "frames_shape": fake_frames.shape,
                }

        result = stage.process_data([task])
        clip = result[0].video.clips[0]

        assert "vllm_async_render_0" in clip.errors
        assert "chunk render failed" in clip.errors["vllm_async_render_0"]
        assert "vllm_async" not in clip.windows[0].model_input
        stage._runner.close()

    def test_destroy_cleans_up(self) -> None:
        """Destroy should close the runner and release the renderer."""
        stage = self._make_stage()
        stage._runner = asyncio.Runner()
        stage._renderer = MagicMock()

        stage.destroy()
        assert stage._renderer is None
        assert stage._runner is None

    def test_pickle_roundtrip(self) -> None:
        """VllmAsyncPromptRenderStage should survive pickle (Xenna serialization)."""
        stage = self._make_stage()
        roundtripped = pickle.loads(pickle.dumps(stage))  # noqa: S301
        assert roundtripped.resources.cpus == 0.5
        assert roundtripped._renderer is None
        assert roundtripped._runner is None

    def test_stage_setup_creates_renderer_and_runner(self) -> None:
        """stage_setup should create a Renderer and asyncio.Runner."""
        stage = self._make_stage()
        mock_renderer = MagicMock()
        with (
            patch(
                "cosmos_curator.pipelines.video.captioning.vllm_async_stage.get_vllm_model_id",
                return_value="Qwen/Qwen2.5-VL-7B-Instruct",
            ),
            patch(
                "cosmos_curator.pipelines.video.captioning.vllm_async_stage.resolve_model_path",
                return_value="/models/qwen",
            ),
            patch(
                "cosmos_curator.pipelines.video.captioning.vllm_async_stage._build_render_engine_args",
            ) as mock_build_args,
            patch(
                "cosmos_curator.pipelines.video.captioning.vllm_async_stage.renderer_from_config",
                create=True,
            ) as mock_renderer_from_config,
        ):
            mock_engine_args = MagicMock()
            mock_build_args.return_value = mock_engine_args
            mock_renderer_from_config.return_value = mock_renderer
            # Patch the deferred import inside stage_setup
            with patch.dict(
                "sys.modules", {"vllm.renderers": MagicMock(renderer_from_config=mock_renderer_from_config)}
            ):
                stage.stage_setup()

            assert stage._renderer is not None
            assert stage._runner is not None
            assert isinstance(stage._runner, asyncio.Runner)
            stage._runner.close()

    def test_render_all_windows_raises_when_renderer_is_none(self) -> None:
        """_render_all_windows_async should raise RuntimeError when renderer is None."""
        stage = self._make_stage()
        stage._renderer = None

        task = _make_task(b"\x00", num_windows=1)
        fake_frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        for clip in task.video.clips:
            for window in clip.windows:
                window.model_input["vllm_async"] = {
                    "prompt_input": {"prompt": "test", "multi_modal_data": {"video": [fake_frames]}},
                    "frames_shape": fake_frames.shape,
                }

        with pytest.raises(RuntimeError, match="Renderer not initialized"):
            asyncio.run(stage._render_all_windows_async([task]))

    def test_process_data_empty_windows_skips_render(self) -> None:
        """process_data with no model_input should skip rendering entirely."""
        stage = self._make_stage()
        mock_renderer = MagicMock()
        mock_renderer.render_cmpl_async = AsyncMock(return_value=[])
        stage._renderer = mock_renderer
        stage._runner = asyncio.Runner()

        task = _make_task(b"\x00", num_windows=2)
        # windows have no model_input["vllm_async"] set

        result = stage.process_data([task])
        assert len(result) == 1
        mock_renderer.render_cmpl_async.assert_not_called()
        stage._runner.close()

    def test_process_data_multiple_tasks_and_clips(self) -> None:
        """process_data should render windows across multiple tasks and clips."""
        stage = self._make_stage()
        mock_renderer = MagicMock()
        rendered_mocks = [MagicMock() for _ in range(4)]
        mock_renderer.render_cmpl_async = AsyncMock(return_value=rendered_mocks)
        stage._renderer = mock_renderer
        stage._runner = asyncio.Runner()

        tasks = []
        for _ in range(2):
            task = _make_task(b"\x00", num_windows=2)
            fake_frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
            for clip in task.video.clips:
                for window in clip.windows:
                    window.model_input["vllm_async"] = {
                        "prompt_input": {"prompt": "describe", "multi_modal_data": {"video": [fake_frames]}},
                        "frames_shape": fake_frames.shape,
                    }
            tasks.append(task)

        result = stage.process_data(tasks)
        assert len(result) == 2
        # 4 windows < RENDER_CHUNK_SIZE (32), so exactly one chunk call
        assert mock_renderer.render_cmpl_async.call_count == 1
        call_args = mock_renderer.render_cmpl_async.call_args
        assert len(call_args[0][0]) == 4  # 2 tasks * 1 clip * 2 windows

        for task in result:
            for clip in task.video.clips:
                for window in clip.windows:
                    rendered_data = window.model_input.get("vllm_async")
                    assert rendered_data is not None
                    assert "rendered_prompt" in rendered_data
                    assert "frames_shape" in rendered_data
                    assert "raw_prompt" in rendered_data
                    assert "decoded_rgb_frames" in rendered_data
        stage._runner.close()

    def test_process_data_chunk_isolation_on_failure(self) -> None:
        """Chunk render failure should only affect windows in the failing chunk."""
        stage = self._make_stage()
        mock_renderer = MagicMock()
        # First chunk succeeds, second chunk fails
        rendered_ok = [MagicMock() for _ in range(VllmAsyncPromptRenderStage.RENDER_CHUNK_SIZE)]
        mock_renderer.render_cmpl_async = AsyncMock(side_effect=[rendered_ok, RuntimeError("chunk 2 boom")])
        stage._renderer = mock_renderer
        stage._runner = asyncio.Runner()

        total_windows = VllmAsyncPromptRenderStage.RENDER_CHUNK_SIZE + 2
        task = _make_task(b"\x00", num_windows=total_windows)
        fake_frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        for clip in task.video.clips:
            for window in clip.windows:
                window.model_input["vllm_async"] = {
                    "prompt_input": {"prompt": "describe", "multi_modal_data": {"video": [fake_frames]}},
                    "frames_shape": fake_frames.shape,
                }

        result = stage.process_data([task])
        clip = result[0].video.clips[0]

        # First RENDER_CHUNK_SIZE windows should have rendered_prompt
        for i in range(VllmAsyncPromptRenderStage.RENDER_CHUNK_SIZE):
            rendered_data = clip.windows[i].model_input.get("vllm_async")
            assert rendered_data is not None, f"window {i} should be rendered"
            assert "rendered_prompt" in rendered_data

        # Remaining windows should have errors and no model_input
        for i in range(VllmAsyncPromptRenderStage.RENDER_CHUNK_SIZE, total_windows):
            assert "vllm_async" not in clip.windows[i].model_input, f"window {i} should have model_input popped"
            assert f"vllm_async_render_{i}" in clip.errors, f"window {i} should have error recorded"

        stage._runner.close()

    def test_process_data_missing_prompt_input_skips_window(self) -> None:
        """Windows with model_input but missing prompt_input should be skipped with error."""
        stage = self._make_stage()
        mock_renderer = MagicMock()
        mock_renderer.render_cmpl_async = AsyncMock(return_value=[MagicMock()])
        stage._renderer = mock_renderer
        stage._runner = asyncio.Runner()

        task = _make_task(b"\x00", num_windows=2)
        clip = task.video.clips[0]
        fake_frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        # Window 0: missing prompt_input key
        clip.windows[0].model_input["vllm_async"] = {"frames_shape": fake_frames.shape}
        # Window 1: valid
        clip.windows[1].model_input["vllm_async"] = {
            "prompt_input": {"prompt": "test", "multi_modal_data": {"video": [fake_frames]}},
            "frames_shape": fake_frames.shape,
        }

        result = stage.process_data([task])
        clip = result[0].video.clips[0]

        assert "vllm_async_render_0" in clip.errors
        assert "missing prompt_input" in clip.errors["vllm_async_render_0"]
        assert "vllm_async" not in clip.windows[0].model_input

        rendered_data = clip.windows[1].model_input.get("vllm_async")
        assert rendered_data is not None
        assert "rendered_prompt" in rendered_data
        stage._runner.close()

    def test_process_data_raises_without_setup(self) -> None:
        """process_data should raise RuntimeError if stage_setup was not called."""
        stage = self._make_stage()
        task = _make_task(b"\x00", num_windows=1)
        with pytest.raises(RuntimeError, match="stage_setup"):
            stage.process_data([task])


class TestBuildRenderEngineArgs:
    """Tests for _build_render_engine_args helper."""

    def test_sets_tp_1_dp_1(self) -> None:
        """Render engine args should use TP=1, DP=1 (no GPU resources)."""
        mock_args_cls = MagicMock(return_value=MagicMock())
        with patch(
            "cosmos_curator.pipelines.video.captioning.vllm_async_stage.AsyncEngineArgs",
            mock_args_cls,
            create=True,
        ):
            config = VllmAsyncConfig(model_variant="qwen", num_gpus=4, data_parallel_size=2)
            _build_render_engine_args(config, "/models/qwen")

            call_kwargs = mock_args_cls.call_args.kwargs
            assert call_kwargs["tensor_parallel_size"] == 1
            assert call_kwargs["data_parallel_size"] == 1
            assert call_kwargs["model"] == "/models/qwen"

    def test_preserves_mm_config(self) -> None:
        """Render engine args should preserve multimodal config from serve_config."""
        mock_args_cls = MagicMock(return_value=MagicMock())
        with patch(
            "cosmos_curator.pipelines.video.captioning.vllm_async_stage.AsyncEngineArgs",
            mock_args_cls,
            create=True,
        ):
            config = VllmAsyncConfig(
                model_variant="qwen",
                num_gpus=1,
                mm_processor_cache_gb=8,
                mm_processor_cache_type="shm",
            )
            _build_render_engine_args(config, "/models/qwen")

            call_kwargs = mock_args_cls.call_args.kwargs
            assert call_kwargs["mm_processor_cache_gb"] == 8
            assert call_kwargs["mm_processor_cache_type"] == "shm"


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
