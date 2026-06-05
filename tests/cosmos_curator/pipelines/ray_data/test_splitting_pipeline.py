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
from typing import Any

import pytest
import ray
from ray.data import ActorPoolStrategy, TaskPoolStrategy

from cosmos_curator.pipelines.ray_data import splitting_pipeline as _pipeline


def _parse_args(*args: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    _pipeline._setup_parser(parser)
    return parser.parse_args(
        [
            "--input-video-path",
            "/input",
            "--output-clip-path",
            "/output",
            *args,
        ]
    )


class RecordingDataset:
    """Small stand-in for ``ray.data.Dataset`` that records map calls."""

    def __init__(self) -> None:
        """Initialize the call recorder."""
        self.map_calls: list[tuple[object, dict[str, Any]]] = []

    def map(self, fn: object, **kwargs: Any) -> "RecordingDataset":  # noqa: ANN401
        """Record a Ray Data map call."""
        self.map_calls.append((fn, kwargs))
        return self

    def flat_map(self, fn: object, **kwargs: Any) -> "RecordingDataset":  # noqa: ANN401
        """Record a Ray Data flat_map call."""
        self.map_calls.append((fn, kwargs))
        return self


def test_progress_cli_defaults_to_disabled_with_boolean_overrides() -> None:
    """The CLI exposes explicit --progress/--no-progress controls."""
    assert _parse_args().progress is False
    assert _parse_args("--progress").progress is True
    assert _parse_args("--no-progress").progress is False


def test_caption_batch_size_cli_defaults_to_model_config() -> None:
    """Caption batch size is optional so model defaults remain authoritative."""
    assert _parse_args().caption_batch_size is None
    assert _pipeline._caption_vllm_config(_parse_args()) is None


def test_caption_batch_size_cli_overrides_vllm_config() -> None:
    """Caption batch size maps to the Ray LLM clip-row batch size."""
    config = _pipeline._caption_vllm_config(_parse_args("--caption-batch-size", "9"))

    assert config is not None
    assert config.model_variant == "qwen"
    assert config.batch_size == 9


def test_caption_batch_size_cli_rejects_non_positive_values(capsys: pytest.CaptureFixture[str]) -> None:
    """Caption batch size must be positive when specified."""
    for invalid_value in ("0", "-1"):
        with pytest.raises(SystemExit):
            _parse_args("--caption-batch-size", invalid_value)

        captured = capsys.readouterr()
        assert captured.out == ""
        assert f"argument --caption-batch-size: '{invalid_value}' must be positive" in captured.err


def test_caption_workers_use_downloaded_gpu_count() -> None:
    """Caption worker max is derived from model-download cluster discovery."""
    assert _pipeline._caption_workers_from_downloaded_gpus(8, None) == 8
    assert (
        _pipeline._caption_workers_from_downloaded_gpus(8, _pipeline.VllmConfig(model_variant="qwen", num_gpus=2)) == 4
    )


def test_transnetv2_frame_decode_cpus_cli_accepts_positive_integer() -> None:
    """Decode CPU count maps directly to FFmpeg threads."""
    assert _parse_args("--transnetv2-frame-decode-cpus-per-worker", "2").transnetv2_frame_decode_cpus_per_worker == 2


@pytest.mark.parametrize(
    ("value", "expected_reason"),
    [
        ("1.5", "'1.5' is not an integer"),
        ("0", "'0' must be positive"),
    ],
)
def test_transnetv2_frame_decode_cpus_cli_rejects_non_integral_values(
    capsys: pytest.CaptureFixture[str],
    value: str,
    expected_reason: str,
) -> None:
    """Decode CPU count must be a positive integer; argparse rejects other values cleanly."""
    with pytest.raises(SystemExit):
        _parse_args("--transnetv2-frame-decode-cpus-per-worker", value)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"argument --transnetv2-frame-decode-cpus-per-worker: {expected_reason}" in captured.err


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
        _pipeline._configure_ray_data_progress(progress=False)
        assert ctx.enable_progress_bars is False
        assert ctx.enable_operator_progress_bars is False
        assert ctx.enable_rich_progress_bars is False
        assert ctx.print_on_execution_start is False
        assert ctx.use_ray_tqdm is False

        _pipeline._configure_ray_data_progress(progress=True)
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
    assert _pipeline._download_slots_for_video_count(num_videos=0, num_nodes=8) == 0
    assert _pipeline._download_slots_for_video_count(num_videos=1, num_nodes=8) == 1
    assert _pipeline._download_slots_for_video_count(num_videos=10, num_nodes=1) == 10
    assert _pipeline._download_slots_for_video_count(num_videos=100, num_nodes=2) == 32


def test_splitting_algorithm_defaults_to_transnetv2() -> None:
    """The Ray Data pipeline defaults to TransNetV2 like the Xenna split pipeline."""
    assert _parse_args().splitting_algorithm == "transnetv2"
    assert _parse_args("--splitting-algorithm", "fixed-stride").splitting_algorithm == "fixed-stride"


def test_transnetv2_cli_defaults_match_xenna_splitter() -> None:
    """Ray Data TransNetV2 CLI defaults should match the existing split pipeline."""
    args = _parse_args()

    assert args.transnetv2_threshold == 0.4
    assert args.transnetv2_min_length_s == 2.0
    assert args.transnetv2_min_length_frames == 48
    assert args.transnetv2_max_length_s == 60.0
    assert args.transnetv2_max_length_mode == "stride"
    assert args.transnetv2_crop_s == 0.5
    assert args.transnetv2_frame_decode_cpus_per_worker == 3
    assert args.transnetv2_gpus_per_worker == 0.25


@pytest.mark.parametrize(
    ("flag", "value", "expected_reason"),
    [
        ("--transnetv2-threshold", "-0.1", "'-0.1' must be between 0 and 1"),
        ("--transnetv2-threshold", "1.1", "'1.1' must be between 0 and 1"),
        ("--transnetv2-threshold", "nan", "'nan' must be finite"),
        ("--transnetv2-min-length-frames", "0", "'0' must be positive"),
        ("--transnetv2-min-length-frames", "-1", "'-1' must be positive"),
        ("--transnetv2-gpus-per-worker", "0", "'0' must be positive"),
        ("--transnetv2-gpus-per-worker", "-0.25", "'-0.25' must be positive"),
        ("--transnetv2-gpus-per-worker", "inf", "'inf' must be finite"),
    ],
)
def test_transnetv2_cli_rejects_invalid_values(
    capsys: pytest.CaptureFixture[str],
    flag: str,
    value: str,
    expected_reason: str,
) -> None:
    """TransNetV2 CLI values should fail before Ray startup when impossible."""
    with pytest.raises(SystemExit):
        _parse_args(flag, value)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"argument {flag}: {expected_reason}" in captured.err


def test_required_model_ids_are_gated_by_selected_features(monkeypatch: pytest.MonkeyPatch) -> None:
    """Download only the models needed by the selected split and captioning features."""
    monkeypatch.setattr(_pipeline, "transnetv2_model_ids", lambda: ["transnetv2"])
    monkeypatch.setattr(_pipeline, "qwen_model_id", lambda: "qwen")

    assert _pipeline._required_model_ids(_parse_args(), generate_captions=True) == ["transnetv2", "qwen"]
    assert _pipeline._required_model_ids(
        _parse_args("--splitting-algorithm", "fixed-stride"),
        generate_captions=True,
    ) == ["qwen"]
    assert (
        _pipeline._required_model_ids(
            _parse_args("--splitting-algorithm", "fixed-stride", "--no-generate-captions"),
            generate_captions=False,
        )
        == []
    )


def test_run_downloads_models_before_ray_startup_and_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model download side effects must run before Ray startup."""
    events: list[str] = []

    def fake_download_models(_model_ids: list[str], _model_weights_path: str) -> int:
        events.append("download")
        return 1

    def fake_required_model_ids(_args: argparse.Namespace, *, generate_captions: bool) -> list[str]:
        del generate_captions
        return ["model"]

    monkeypatch.setattr(_pipeline, "assert_ffmpeg_supports_h264", lambda: events.append("ffmpeg"))
    monkeypatch.setattr(_pipeline, "_required_model_ids", fake_required_model_ids)
    monkeypatch.setattr(_pipeline, "download_models", fake_download_models)
    monkeypatch.setattr(_pipeline, "qwen_model_source", lambda: events.append("qwen_source") or "/models/qwen")
    monkeypatch.setattr(_pipeline.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(_pipeline.ray, "init", lambda **_kwargs: events.append("ray_init"))

    def fake_configure_ray_data_progress(*, progress: bool) -> None:
        del progress
        events.append("progress")

    def fake_discover_videos(_input_path: str, *, limit: int = 0) -> list[str]:
        del limit
        events.append("discover")
        return []

    monkeypatch.setattr(_pipeline, "_configure_ray_data_progress", fake_configure_ray_data_progress)
    monkeypatch.setattr(_pipeline, "_discover_videos", fake_discover_videos)

    assert _pipeline.run(_parse_args()) == 0
    assert events == ["ffmpeg", "download", "qwen_source", "ray_init", "progress", "discover"]


def test_run_rejects_transnetv2_max_length_below_min_length_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Impossible TransNetV2 length bounds should fail before Ray or model setup side effects."""
    monkeypatch.setattr(_pipeline, "assert_ffmpeg_supports_h264", lambda: pytest.fail("unexpected ffmpeg check"))
    monkeypatch.setattr(_pipeline, "download_models", lambda *_args, **_kwargs: pytest.fail("unexpected download"))
    monkeypatch.setattr(_pipeline.ray, "init", lambda **_kwargs: pytest.fail("unexpected ray init"))

    with pytest.raises(ValueError, match="Max length is smaller than min length"):
        _pipeline.run(_parse_args("--transnetv2-min-length-s", "10", "--transnetv2-max-length-s", "5"))


def test_run_uses_downloaded_gpu_count_for_caption_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The caption pool ceiling comes from the model-download GPU count."""
    ds = RecordingDataset()
    captured_caption_kwargs: dict[str, object] = {}

    def fake_required_model_ids(_args: argparse.Namespace, *, generate_captions: bool) -> list[str]:
        del generate_captions
        return ["qwen"]

    def fake_configure_ray_data_progress(*, progress: bool) -> None:
        del progress

    def fake_discover_videos(_input_path: str, *, limit: int = 0) -> list[str]:
        del limit
        return ["/input/a.mp4"]

    def fake_caption_window_rows(dataset: RecordingDataset, **kwargs: object) -> RecordingDataset:
        captured_caption_kwargs.update(kwargs)
        return dataset

    monkeypatch.setattr(_pipeline, "assert_ffmpeg_supports_h264", lambda: None)
    monkeypatch.setattr(_pipeline, "_required_model_ids", fake_required_model_ids)
    monkeypatch.setattr(_pipeline, "download_models", lambda _model_ids, _model_weights_path: 8)
    monkeypatch.setattr(_pipeline, "qwen_model_source", lambda: "/models/qwen")
    monkeypatch.setattr(_pipeline.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(_pipeline.ray, "init", lambda **_kwargs: None)
    monkeypatch.setattr(_pipeline.ray, "nodes", lambda: [{"Alive": True, "Resources": {"CPU": 8, "GPU": 8}}])
    monkeypatch.setattr(_pipeline.ray, "cluster_resources", lambda: pytest.fail("unexpected GPU resource query"))
    monkeypatch.setattr(_pipeline.ray.data, "from_items", lambda _items: ds)
    monkeypatch.setattr(_pipeline, "_configure_ray_data_progress", fake_configure_ray_data_progress)
    monkeypatch.setattr(_pipeline, "_discover_videos", fake_discover_videos)
    monkeypatch.setattr(_pipeline, "caption_window_rows", fake_caption_window_rows)
    monkeypatch.setattr(_pipeline, "write_captioned_metadata_and_summary", lambda *_args, **_kwargs: 1)

    assert _pipeline.run(_parse_args()) == 1
    assert captured_caption_kwargs["caption_workers"] == 8


def test_apply_split_stage_wires_transnetv2_actor_resources() -> None:
    """TransNetV2 should run as a stateful Ray Data actor with declared CPU/GPU resources."""
    args = _parse_args("--limit-clips", "7")
    ds = RecordingDataset()

    result = _pipeline._apply_split_stage(ds, args, download_slots=3)  # type: ignore[arg-type]

    assert result is ds
    fn, kwargs = ds.map_calls[0]
    assert fn is _pipeline.TransNetV2Splitter
    assert kwargs["fn_constructor_kwargs"] == {
        "threshold": 0.4,
        "min_length_s": 2.0,
        "min_length_frames": 48,
        "max_length_s": 60.0,
        "max_length_mode": "stride",
        "crop_s": 0.5,
        "num_decode_cpus_per_worker": 3,
        "limit_clips": 7,
    }
    assert kwargs["num_cpus"] == 3
    assert kwargs["num_gpus"] == 0.25
    assert isinstance(kwargs["compute"], ActorPoolStrategy)
    assert kwargs["compute"].min_size == 1
    assert kwargs["compute"].max_size == 3
    assert kwargs["compute"].initial_size == 1
    assert kwargs["runtime_env"].get("py_executable") == "pixi run --as-is -e unified python"
    assert kwargs["runtime_env"].get("env_vars") == {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "0"}
    assert kwargs["scheduling_strategy"] == "DEFAULT"


def test_apply_split_stage_keeps_fixed_stride_as_task_pool() -> None:
    """Fixed-stride splitting remains a stateless task transform."""
    args = _parse_args("--splitting-algorithm", "fixed-stride")
    ds = RecordingDataset()

    result = _pipeline._apply_split_stage(ds, args, download_slots=2)  # type: ignore[arg-type]

    assert result is ds
    _, kwargs = ds.map_calls[0]
    assert kwargs["num_cpus"] == 0.25
    assert isinstance(kwargs["compute"], TaskPoolStrategy)
    assert kwargs["compute"].size == 2
    assert "runtime_env" not in kwargs


def test_validate_transnetv2_cluster_resources_fails_without_gpu() -> None:
    """Default TransNetV2 splitting should fail fast when model download found no GPUs."""
    with pytest.raises(ValueError, match="TransNetV2 splitting requires visible GPUs"):
        _pipeline._validate_transnetv2_cluster_resources(_parse_args(), total_visible_gpus=0)


def test_validate_transnetv2_cluster_resources_accepts_node_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """TransNetV2 only needs one live node that can fit one splitter worker."""
    monkeypatch.setattr(
        _pipeline.ray,
        "nodes",
        lambda: [
            {"Alive": True, "Resources": {"CPU": 2, "GPU": 8}},
            {"Alive": True, "Resources": {"CPU": 3, "GPU": 0.25}},
        ],
    )

    _pipeline._validate_transnetv2_cluster_resources(_parse_args(), total_visible_gpus=8)


@pytest.mark.parametrize(
    ("args", "nodes"),
    [
        (
            ["--transnetv2-gpus-per-worker", "2"],
            [
                {"Alive": True, "Resources": {"CPU": 16, "GPU": 1}},
                {"Alive": True, "Resources": {"CPU": 16, "GPU": 1}},
            ],
        ),
        (
            ["--transnetv2-frame-decode-cpus-per-worker", "16"],
            [{"Alive": True, "Resources": {"CPU": 8, "GPU": 1}}],
        ),
        (
            [],
            [
                {"Alive": False, "Resources": {"CPU": 3, "GPU": 0.25}},
                {"Alive": True, "Resources": {"CPU": 2, "GPU": 0.25}},
            ],
        ),
    ],
)
def test_validate_transnetv2_cluster_resources_rejects_unschedulable_nodes(
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    nodes: list[dict[str, object]],
) -> None:
    """Aggregate GPUs are not enough when no single node can fit a splitter worker."""
    monkeypatch.setattr(_pipeline.ray, "nodes", lambda: nodes)

    with pytest.raises(ValueError, match="at least one live Ray node"):
        _pipeline._validate_transnetv2_cluster_resources(_parse_args(*args), total_visible_gpus=8)


def test_validate_transnetv2_cluster_resources_skips_fixed_stride() -> None:
    """Fixed-stride splitting should remain usable on CPU-only Ray clusters."""
    _pipeline._validate_transnetv2_cluster_resources(
        _parse_args("--splitting-algorithm", "fixed-stride"),
        total_visible_gpus=0,
    )
