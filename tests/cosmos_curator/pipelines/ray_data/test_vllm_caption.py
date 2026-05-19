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

"""Tests for Ray Data vLLM captioning helpers."""

import asyncio
import json
import logging
import os
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import ray
from ray.llm._internal.batch.stages.vllm_engine_stage import vLLMEngineStage, vLLMEngineStageUDF

from cosmos_curator.pipelines.ray_data import _vllm_caption as _captioner
from cosmos_curator.pipelines.ray_data._vllm_caption import (
    _add_ray_llm_columns,
    _apply_ray_assigned_cuda_visible_devices,
    _arrow_data_column_to_rows,
    _assemble_ray_multimodal_data,
    _install_vllm_engine_stage_shim,
    _patch_vllm_inputs_data_namespace,
    make_normalize_caption_output_fn,
    sampling_params_dict,
    write_captioned_metadata_and_summary,
)
from cosmos_curator.pipelines.video.utils.data_model import VllmConfig, VllmSamplingConfig


def _base_window_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "video_path": "s3://bucket/raw/a.mp4",
        "video_size": 1000,
        "duration_s": 30.0,
        "clip_uuid": "clip-1",
        "clip_start_s": 0.0,
        "clip_end_s": 10.0,
        "clip_location": "out/clips/clip-1.mp4",
        "width_source": 1280,
        "height_source": 720,
        "framerate_source": 29.97,
        "width": 640,
        "height": 360,
        "framerate": 29.97,
        "num_frames": 300,
        "video_codec": "h264",
        "num_bytes": 1234,
        "window_index": 0,
        "start_frame": 0,
        "end_frame": 255,
        "caption_skip": False,
        "caption_status": "success",
        "caption_failure_reason": None,
        "caption_error": None,
        "qwen_caption": "caption",
        "qwen_prompt_tokens": 10,
        "qwen_output_tokens": 5,
    }
    row.update(overrides)
    return row


def test_sampling_params_dict_uses_vllm_sampling_config_defaults() -> None:
    """Sampling params are derived from the shared Xenna config object."""
    params = sampling_params_dict(VllmSamplingConfig(max_tokens=42, temperature=0.2))

    assert params["max_tokens"] == 42
    assert params["temperature"] == 0.2
    assert params["top_p"] == VllmSamplingConfig().top_p


def test_ray_llm_columns_are_arrow_native_and_reassemble_for_vllm_actor() -> None:
    """Frame payloads stay Arrow-native until the vLLM actor rebuilds multimodal_data."""
    pa = pytest.importorskip("pyarrow")
    frames = np.arange(2 * 3 * 4 * 4, dtype=np.float16).reshape((2, 3, 4, 4))
    metadata = {
        "fps": 2.0,
        "duration": 1.0,
        "width": 4,
        "height": 4,
        "total_num_frames": 2,
        "frames_indices": [0, 1],
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    sampling_params = {"temperature": 0.1}
    row: dict[str, object] = {}

    _add_ray_llm_columns(
        row,
        {
            "prompt_token_ids": [1, 2, 3],
            "multi_modal_data": {"video": [(frames, metadata)]},
            "mm_processor_kwargs": {"do_resize": False},
            "multi_modal_uuids": {"video": ["window-1"]},
        },
        sampling_params,
    )

    assert row["tokenized_prompt"] == [1, 2, 3]
    assert row["sampling_params"] == sampling_params
    assert row["video_frame_bytes"].type == pa.large_binary()  # type: ignore[attr-defined]
    assert row["video_frame_bytes"].as_py() == frames.tobytes()  # type: ignore[attr-defined]
    assert row["video_frame_shape"] == [2, 3, 4, 4]
    assert row["video_frame_dtype"] == "float16"
    assert row["video_metadata"] == metadata
    assert row["mm_processor_kwargs"] == {"do_resize": False}
    assert row["multimodal_uuids"] == {"video": ["window-1"]}
    assert "multimodal_data" not in row

    pa.array([{"__data": row}])

    _assemble_ray_multimodal_data(row)
    assert "video_frame_bytes" not in row
    actual_frames, actual_metadata = row["multimodal_data"]["video"][0]  # type: ignore[index]
    np.testing.assert_array_equal(actual_frames, frames)
    assert actual_frames.flags.writeable
    assert actual_metadata == metadata


def test_arrow_data_column_to_rows_preserves_chunked_large_binary_payloads() -> None:
    """PyArrow engine batches are unpacked without combining oversized payload chunks."""
    pa = pytest.importorskip("pyarrow")
    first = pa.Table.from_pylist(
        [{"__data": {"clip_uuid": "clip-1", "video_frame_bytes": pa.scalar(b"first", type=pa.large_binary())}}]
    )
    second = pa.Table.from_pylist(
        [{"__data": {"clip_uuid": "clip-2", "video_frame_bytes": pa.scalar(b"second", type=pa.large_binary())}}]
    )
    batch = pa.concat_tables([first, second])

    assert len(batch.column("__data").chunks) == 2
    rows = _arrow_data_column_to_rows(batch, "__data")

    assert [row["clip_uuid"] for row in rows] == ["clip-1", "clip-2"]
    assert [row["video_frame_bytes"] for row in rows] == [b"first", b"second"]


def test_vllm_engine_stage_shim_uses_pyarrow_batches() -> None:
    """The local vLLM shim must bypass Ray's default NumPy batch formatting."""
    stage = vLLMEngineStage(
        fn_constructor_kwargs={"engine_kwargs": {"tensor_parallel_size": 1}},
        map_batches_kwargs={},
    )

    class FakeProcessor:
        @staticmethod
        def get_stage_by_name(name: str) -> object:
            assert name == "vLLMEngineStage"
            return stage

    _install_vllm_engine_stage_shim(FakeProcessor())

    assert stage.map_batches_kwargs["batch_format"] == "pyarrow"
    assert stage.fn is not vLLMEngineStageUDF
    assert issubclass(stage.fn, vLLMEngineStageUDF)


def test_vllm_engine_stage_shim_collapses_clip_windows() -> None:
    """The local vLLM shim should keep one output row per input clip row."""
    stage = vLLMEngineStage(
        fn_constructor_kwargs={"engine_kwargs": {"tensor_parallel_size": 1}},
        map_batches_kwargs={},
    )

    class FakeProcessor:
        @staticmethod
        def get_stage_by_name(name: str) -> object:
            assert name == "vLLMEngineStage"
            return stage

    _install_vllm_engine_stage_shim(FakeProcessor())
    udf = object.__new__(stage.fn)
    udf.data_column = "__data"
    udf.expected_input_keys = set()
    udf.engine_kwargs = {"disable_log_stats": True}

    async def fake_generate(row: dict[str, object], _batch_uuid: object) -> dict[str, object]:
        return {
            udf.IDX_IN_BATCH_COLUMN: row[udf.IDX_IN_BATCH_COLUMN],
            "generated_text": f"caption-{row['window_index']}",
            "num_input_tokens": 10,
            "num_generated_tokens": 5,
            "__inference_error__": "",
        }

    udf._generate_with_error_handling = fake_generate
    clip_row = {
        "__data": [
            _base_window_row(
                caption_windows=[
                    _base_window_row(window_index=0, tokenized_prompt=[1], sampling_params={}),
                    _base_window_row(window_index=1, tokenized_prompt=[2], sampling_params={}),
                ]
            )
        ]
    }

    async def collect() -> list[dict[str, object]]:
        return [row async for row in udf(clip_row)]

    result = asyncio.run(collect())

    windows = result[0]["__data"][0]["caption_windows"]
    assert [window["generated_text"] for window in windows] == ["caption-0", "caption-1"]
    assert len(result[0]["__data"]) == 1


def test_add_ray_llm_columns_serializes_non_contiguous_frames() -> None:
    """Non-contiguous frame arrays should still round-trip through Arrow bytes."""
    frames = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)[:, :, ::2, :]
    assert not frames.flags.c_contiguous
    row: dict[str, object] = {}

    _add_ray_llm_columns(row, {"prompt_token_ids": [1], "multi_modal_data": {"video": [frames]}}, {})

    assert row["video_frame_bytes"].as_py() == np.ascontiguousarray(frames).tobytes()  # type: ignore[attr-defined]
    assert row["video_frame_shape"] == [2, 3, 2, 4]

    _assemble_ray_multimodal_data(row)
    np.testing.assert_array_equal(row["multimodal_data"]["video"][0], frames)  # type: ignore[index]


def test_build_processor_uses_qwen_ray_data_scheduler_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """The private processor keeps the established Qwen Ray Data scheduler defaults."""

    @dataclass
    class FakeProcessorConfig:
        model_source: str
        batch_size: int
        max_concurrent_batches: int
        concurrency: tuple[int, int]
        runtime_env: object
        experimental: dict[str, object]
        should_continue_on_error: bool
        chat_template_stage: bool
        tokenize_stage: bool
        detokenize_stage: bool
        prepare_image_stage: bool
        prepare_multimodal_stage: bool
        engine_kwargs: dict[str, object]

    captured_configs: list[FakeProcessorConfig] = []

    def fake_build_processor(config: FakeProcessorConfig) -> Callable[[ray.data.Dataset], ray.data.Dataset]:
        captured_configs.append(config)
        return lambda ds: ds

    fake_llm_module = types.ModuleType("ray.data.llm")
    fake_llm_module.build_processor = fake_build_processor
    fake_llm_module.vLLMEngineProcessorConfig = FakeProcessorConfig
    monkeypatch.setitem(sys.modules, "ray.data.llm", fake_llm_module)
    monkeypatch.setattr(_captioner, "_install_vllm_engine_stage_shim", lambda _processor: None)
    monkeypatch.setattr(_captioner, "_ray_vllm_max_tasks_in_flight_per_actor", lambda: 16)
    monkeypatch.setattr(_captioner, "PixiRuntimeEnv", lambda name: {"pixi": name})

    processor = _captioner._build_processor(
        model_source="/models/qwen",
        caption_workers=2,
        vllm_config=VllmConfig(model_variant="qwen", batch_size=4),
    )

    assert callable(processor)
    config = captured_configs[0]
    assert config.engine_kwargs["max_num_batched_tokens"] == 32768
    assert config.engine_kwargs["max_num_seqs"] == 64
    assert config.max_concurrent_batches == 16
    assert config.experimental == {"max_tasks_in_flight_per_actor": 16}
    assert config.concurrency == (2, 2)


def test_build_processor_derives_ray_batch_size_from_vllm_max_num_seqs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benchmark tuning knobs should express vLLM capacity and derive Ray batching."""
    captured_configs: list[dict[str, object]] = []

    def fake_build_processor(config: dict[str, object]) -> Callable[[ray.data.Dataset], ray.data.Dataset]:
        captured_configs.append(config)
        return lambda ds: ds

    fake_llm_module = types.ModuleType("ray.data.llm")
    fake_llm_module.build_processor = fake_build_processor
    fake_llm_module.vLLMEngineProcessorConfig = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "ray.data.llm", fake_llm_module)
    monkeypatch.setattr(_captioner, "_install_vllm_engine_stage_shim", lambda _processor: None)
    monkeypatch.setattr(_captioner, "_ray_vllm_max_tasks_in_flight_per_actor", lambda: 16)
    monkeypatch.setattr(_captioner, "PixiRuntimeEnv", lambda name: {"pixi": name})

    processor = _captioner._build_processor(
        model_source="/models/qwen",
        caption_workers=2,
        vllm_config=VllmConfig(model_variant="qwen", batch_size=4),
        max_num_seqs=384,
    )

    assert callable(processor)
    config = captured_configs[0]
    engine_kwargs = config["engine_kwargs"]
    assert isinstance(engine_kwargs, dict)
    assert engine_kwargs["max_num_batched_tokens"] == 32768
    assert engine_kwargs["max_num_seqs"] == 384
    assert config["batch_size"] == 24
    assert config["max_concurrent_batches"] == 16
    assert config["experimental"] == {"max_tasks_in_flight_per_actor": 16}


def test_assemble_ray_multimodal_data_rejects_bad_payload_size() -> None:
    """Shape/dtype metadata must match the transported frame bytes."""
    row: dict[str, object] = {
        "video_frame_bytes": b"\x00",
        "video_frame_shape": [2, 3],
        "video_frame_dtype": "float16",
    }

    with pytest.raises(ValueError, match="byte size does not match"):
        _assemble_ray_multimodal_data(row)


def test_vllm_inputs_namespace_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ray's vLLM stage can find prompt classes with the pinned vLLM namespace."""

    class TextPrompt(dict[str, object]):
        pass

    class TokensPrompt(dict[str, object]):
        pass

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_inputs = types.ModuleType("vllm.inputs")
    fake_inputs.TextPrompt = TextPrompt
    fake_inputs.TokensPrompt = TokensPrompt

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", fake_inputs)
    monkeypatch.delitem(sys.modules, "vllm.inputs.data", raising=False)

    _patch_vllm_inputs_data_namespace()

    assert fake_inputs.data.TextPrompt is TextPrompt
    assert fake_inputs.data.TokensPrompt is TokensPrompt
    assert sys.modules["vllm.inputs.data"] is fake_inputs.data


def test_caption_actor_sets_cuda_visible_devices_from_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    """The caption actor masks vLLM subprocesses to its assigned Ray GPU."""
    monkeypatch.setattr(_captioner.ray, "get_gpu_ids", lambda: [1.0])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    _apply_ray_assigned_cuda_visible_devices()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_caption_actor_logs_cuda_masking_impact_when_ray_gpu_lookup_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Ray GPU lookup failures should explain their effect on vLLM subprocess masking."""

    def raise_ray_gpu_lookup_error() -> list[object]:
        msg = "ray GPU lookup failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(_captioner.ray, "get_gpu_ids", raise_ray_gpu_lookup_error)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    caplog.set_level(logging.WARNING, logger=_captioner.logger.name)

    _apply_ray_assigned_cuda_visible_devices()

    assert "CUDA_VISIBLE_DEVICES will not be set for vLLM engine subprocesses" in caplog.text
    assert "ray GPU lookup failed" in caplog.text
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_normalize_caption_output_infers_statuses() -> None:
    """Ray processor outputs are normalized to success/truncated/error metadata."""
    normalize = make_normalize_caption_output_fn(VllmSamplingConfig(max_tokens=4))

    success = normalize(
        _base_window_row(
            generated_text="done",
            num_input_tokens=11,
            num_generated_tokens=3,
            __inference_error__="",
        )
    )
    truncated = normalize(
        _base_window_row(
            generated_text="truncated",
            num_input_tokens=12,
            num_generated_tokens=4,
            __inference_error__="",
        )
    )
    error = normalize(
        _base_window_row(
            generated_text="",
            num_input_tokens=12,
            num_generated_tokens=0,
            __inference_error__="RuntimeError: boom",
        )
    )

    assert success["caption_status"] == "success"
    assert success["qwen_prompt_tokens"] == 11
    assert success["qwen_output_tokens"] == 3
    assert truncated["caption_status"] == "truncated"
    assert truncated["qwen_caption"] == "truncated"
    assert error["caption_status"] == "error"
    assert error["caption_failure_reason"] == "exception"
    assert error["qwen_caption"] is None
    assert error["qwen_prompt_tokens"] == 0


def test_normalize_caption_output_preserves_skipped_rows() -> None:
    """Skipped rows bypass vLLM but still normalize to the caption metadata schema."""
    normalize = make_normalize_caption_output_fn(VllmSamplingConfig(max_tokens=4))

    skipped = normalize(
        _base_window_row(
            window_index=-1,
            start_frame=None,
            end_frame=None,
            caption_skip=True,
            caption_status="error",
            caption_failure_reason="exception",
            caption_error="no_caption_windows",
            __inference_error__="no_caption_windows",
            generated_text="",
            qwen_caption=None,
            qwen_prompt_tokens=0,
            qwen_output_tokens=0,
        )
    )

    assert skipped["caption_skip"] is True
    assert skipped["caption_status"] == "error"
    assert skipped["caption_error"] == "no_caption_windows"
    assert skipped["qwen_caption"] is None
    assert skipped["qwen_prompt_tokens"] == 0


def test_caption_window_rows_runs_window_generation_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Skipped rows flow through the same processor branch, avoiding duplicate window prep."""
    calls_path = tmp_path / "window_calls.txt"
    captured_batch_size: list[int] = []

    def fake_make_caption_window_rows_fn(
        *_args: object,
        **_kwargs: object,
    ) -> Callable[[dict[str, object]], list[dict[str, object]]]:
        def _make_rows(row: dict[str, object]) -> list[dict[str, object]]:
            with calls_path.open("a", encoding="utf-8") as calls:
                calls.write(f"{row['clip_uuid']}\n")

            request_row = _base_window_row(
                clip_uuid=row["clip_uuid"],
                caption_skip=False,
                caption_error=None,
                tokenized_prompt=[1, 2, 3],
                sampling_params={"temperature": 0.1},
            )
            skipped_row = _base_window_row(
                clip_uuid=row["clip_uuid"],
                window_index=-1,
                start_frame=None,
                end_frame=None,
                caption_skip=True,
                caption_status="error",
                caption_failure_reason="exception",
                caption_error="no_caption_windows",
                __inference_error__="no_caption_windows",
                qwen_caption=None,
                qwen_prompt_tokens=0,
                qwen_output_tokens=0,
            )
            return [request_row, skipped_row]

        return _make_rows

    def fake_build_processor(**kwargs: object) -> Callable[[ray.data.Dataset], ray.data.Dataset]:
        vllm_config = kwargs["vllm_config"]
        assert isinstance(vllm_config, VllmConfig)
        captured_batch_size.append(vllm_config.batch_size)

        def _processor(ds: ray.data.Dataset) -> ray.data.Dataset:
            def _fake_generate(row: dict[str, object]) -> dict[str, object]:
                windows = row["caption_windows"]
                assert isinstance(windows, list)
                for window in windows:
                    assert isinstance(window, dict)
                    if window.get("caption_skip", False):
                        continue
                    window.update(
                        {
                            "generated_text": "generated caption",
                            "num_input_tokens": 3,
                            "num_generated_tokens": 2,
                            "__inference_error__": "",
                        }
                    )
                return row

            return ds.map(_fake_generate)

        return _processor

    monkeypatch.setattr(_captioner, "make_caption_window_rows_fn", fake_make_caption_window_rows_fn)
    monkeypatch.setattr(_captioner, "_build_processor", fake_build_processor)
    monkeypatch.setattr(_captioner, "_ray_vllm_max_tasks_in_flight_per_actor", lambda: 16)
    monkeypatch.setattr(_captioner, "PixiRuntimeEnv", lambda _name: {})

    ds = ray.data.from_items([_base_window_row(clip_uuid="clip-1")])

    rows = _captioner.caption_window_rows(
        ds,
        model_source="unused",
        caption_workers=1,
        vllm_max_num_seqs=128,
    ).take_all()

    assert calls_path.read_text(encoding="utf-8").splitlines() == ["clip-1"]
    assert captured_batch_size == [8]
    assert len(rows) == 1
    assert sorted(window["caption_skip"] for window in rows[0]["caption_windows"]) == [False, True]


def test_write_captioned_metadata_and_summary(tmp_path: Path) -> None:
    """Captioned clip rows write per-clip metadata and summary totals without a shuffle."""
    rows = [
        _base_window_row(
            caption_windows=[
                _base_window_row(
                    window_index=1,
                    start_frame=256,
                    end_frame=511,
                    qwen_caption="second",
                    qwen_prompt_tokens=20,
                    qwen_output_tokens=8,
                ),
                _base_window_row(
                    window_index=0,
                    start_frame=0,
                    end_frame=255,
                    qwen_caption="first",
                    qwen_prompt_tokens=10,
                    qwen_output_tokens=4,
                ),
            ],
        ),
        _base_window_row(
            clip_uuid="clip-2",
            clip_start_s=10.0,
            clip_end_s=20.0,
            clip_location="out/clips/clip-2.mp4",
            caption_windows=[
                _base_window_row(
                    clip_uuid="clip-2",
                    clip_start_s=10.0,
                    clip_end_s=20.0,
                    clip_location="out/clips/clip-2.mp4",
                    window_index=-1,
                    start_frame=None,
                    end_frame=None,
                    caption_skip=True,
                    caption_status="error",
                    caption_failure_reason="exception",
                    caption_error="no_caption_windows",
                    qwen_caption=None,
                    qwen_prompt_tokens=0,
                    qwen_output_tokens=0,
                )
            ],
        ),
    ]

    ds = ray.data.from_items(rows)
    num_clips = write_captioned_metadata_and_summary(
        ds,
        input_video_path="s3://bucket/raw",
        output_path=str(tmp_path),
        num_input_videos=1,
    )

    assert num_clips == 2

    clip_1 = json.loads((tmp_path / "metas" / "v0" / "clip-1.json").read_text())
    assert [window["qwen_caption"] for window in clip_1["windows"]] == ["first", "second"]
    assert clip_1["has_caption"] is True
    assert clip_1["num_caption_windows"] == 2
    assert clip_1["total_prompt_tokens"] == 30
    assert clip_1["total_output_tokens"] == 12

    clip_2 = json.loads((tmp_path / "metas" / "v0" / "clip-2.json").read_text())
    assert clip_2["windows"] == []
    assert clip_2["valid"] is False
    assert clip_2["has_caption"] is False

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["total_num_clips_with_caption"] == 1
    assert summary["total_num_caption_windows"] == 2
    assert summary["total_prompt_tokens"] == 30
    assert summary["total_output_tokens"] == 12
