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
import json
from pathlib import Path

import pytest

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.core.utils.config.args_utils import fill_default_args
from cosmos_curator.pipelines.common.model_constraints import PreprocessMode
from cosmos_curator.pipelines.video.captioning.caption_quality_flags import (
    DEFAULT_CAPTION_QUALITY_THRESHOLDS,
    CaptionQualityThresholdConfig,
)
from cosmos_curator.pipelines.video.captioning.captioning_builders import CaptioningConfig, VllmAsyncCaptionConfig
from cosmos_curator.pipelines.video.clipping.clip_frame_extraction_stages import ClipFrameExtractionStage
from cosmos_curator.pipelines.video.embedding.embedding_builders import (
    CosmosEmbed1Config,
    EmbeddingBackendConfig,
    EmbeddingConfig,
    InternVideo2Config,
    OpenAIEmbeddingConfig,
)
from cosmos_curator.pipelines.video.read_write.metadata_writer_stage import ClipWriterStage
from cosmos_curator.pipelines.video.splitting_pipeline import _assemble_stages, _setup_parser
from cosmos_curator.pipelines.video.utils.data_model import (
    VllmAsyncConfig,
    VllmConfig,
    VllmSamplingConfig,
    WindowConfig,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SPLIT_INVOKE_TEMPLATES = (
    _REPO_ROOT / "examples/nvcf/function/invoke_video_split_full.json",
    _REPO_ROOT / "examples/workflow/template_invoke_video_split.json",
)
_CI_SPLIT_CONFIGS = sorted((_REPO_ROOT / "examples/ci").glob("*.json"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _setup_parser(parser)
    return parser


def _stage_object(stage: CuratorStage | CuratorStageSpec) -> CuratorStage:
    if isinstance(stage, CuratorStageSpec):
        return stage.stage
    return stage


class _EmbeddingMarkerStage(CuratorStage):
    """Lightweight marker returned by the patched embedding builder."""

    def __init__(self, target_fps: float) -> None:
        self.target_fps = target_fps


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


def _embedding_args(extra_args: list[str]) -> argparse.Namespace:
    input_path = Path.cwd() / "tmp-input"
    output_path = Path.cwd() / "tmp-output"
    return _parser().parse_args(
        [
            "--input-video-path",
            input_path.as_posix(),
            "--output-clip-path",
            output_path.as_posix(),
            "--no-generate-captions",
            *extra_args,
        ]
    )


def _capture_embedding_config(monkeypatch: pytest.MonkeyPatch) -> dict[str, EmbeddingConfig]:
    captured: dict[str, EmbeddingConfig] = {}

    def fake_build_embedding_stages(config: EmbeddingConfig) -> list[CuratorStage | CuratorStageSpec]:
        captured["config"] = config
        return [_EmbeddingMarkerStage(config.target_fps)]

    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.splitting_pipeline.get_embedding_model_version",
        lambda _config: "test-version",
    )
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.splitting_pipeline.build_embedding_stages",
        fake_build_embedding_stages,
    )
    return captured


def test_embedding_sampling_fps_defaults_to_two() -> None:
    """Omitting the option should preserve the existing 2-FPS behavior."""
    assert _parser().parse_args([]).embedding_sampling_fps == 2.0


def test_embedding_sampling_fps_help_names_default() -> None:
    """Direct CLI help should make the backward-compatible default explicit."""
    help_text = _parser().format_help()

    assert "--embedding-sampling-fps" in help_text
    assert "default: 2.0" in help_text


@pytest.mark.parametrize(("value", "expected"), [("1", 1.0), ("1.5", 1.5), ("0.001", 0.001)])
def test_embedding_sampling_fps_accepts_supported_values(value: str, expected: float) -> None:
    """Integer, fractional, and minimum supported values should parse as floats."""
    assert _parser().parse_args(["--embedding-sampling-fps", value]).embedding_sampling_fps == expected


@pytest.mark.parametrize("value", ["0", "-1", "0.0009", "nan", "inf", "-inf", "invalid"])
def test_embedding_sampling_fps_rejects_invalid_cli_values(
    value: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invalid direct CLI values should fail without leaking expected usage output."""
    with pytest.raises(SystemExit):
        _parser().parse_args([f"--embedding-sampling-fps={value}"])

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--embedding-sampling-fps" in captured.err
    assert "must be a finite number greater than or equal to 0.001" in captured.err
    assert "normalize_embedding_sampling_fps" not in captured.err


def test_fill_default_args_injects_embedding_sampling_fps_default() -> None:
    """JSON/API configs should receive the 2-FPS default when the field is omitted."""
    args = argparse.Namespace()

    fill_default_args(args, _setup_parser)

    assert args.embedding_sampling_fps == 2.0


@pytest.mark.parametrize("extra_args", [[], ["--embedding-sampling-fps", "2"]], ids=["omitted", "explicit"])
def test_embedding_sampling_fps_default_assembly_is_equivalent(
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
) -> None:
    """Omission and an explicit 2-FPS value should assemble the same wiring."""
    captured = _capture_embedding_config(monkeypatch)

    stages = [_stage_object(stage) for stage in _assemble_stages(_embedding_args(extra_args))]

    extraction_stage = next(stage for stage in stages if isinstance(stage, ClipFrameExtractionStage))
    marker_stage = next(stage for stage in stages if isinstance(stage, _EmbeddingMarkerStage))
    assert extraction_stage._target_fps == [2.0]
    assert marker_stage.target_fps == 2.0
    assert captured["config"].target_fps == 2.0


@pytest.mark.parametrize(
    ("algorithm", "backend_type"),
    [
        ("cosmos-embed1-336p", CosmosEmbed1Config),
        ("internvideo2", InternVideo2Config),
        ("openai", OpenAIEmbeddingConfig),
    ],
)
def test_embedding_sampling_fps_override_reaches_extraction_and_backend_config(
    monkeypatch: pytest.MonkeyPatch,
    algorithm: str,
    backend_type: type[EmbeddingBackendConfig],
) -> None:
    """Every backend should receive the same configured rate requested from extraction."""
    captured = _capture_embedding_config(monkeypatch)
    args = _embedding_args(
        [
            "--embedding-algorithm",
            algorithm,
            "--embedding-sampling-fps",
            "1.5",
        ]
    )

    stages = [_stage_object(stage) for stage in _assemble_stages(args)]

    extraction_stage = next(stage for stage in stages if isinstance(stage, ClipFrameExtractionStage))
    marker_stage = next(stage for stage in stages if isinstance(stage, _EmbeddingMarkerStage))
    config = captured["config"]
    assert isinstance(config.backend, backend_type)
    assert config.target_fps == 1.5
    assert extraction_stage._target_fps == [1.5]
    assert marker_stage.target_fps == 1.5
    assert stages.index(extraction_stage) < stages.index(marker_stage)


def test_embedding_sampling_fps_is_not_constructed_when_embeddings_are_disabled() -> None:
    """A config-only invalid value should not affect an aesthetics-only extraction requirement."""
    args = _embedding_args(["--no-generate-embeddings", "--aesthetic-threshold", "3.5"])
    args.embedding_sampling_fps = float("nan")

    stages = [_stage_object(stage) for stage in _assemble_stages(args)]

    extraction_stage = next(stage for stage in stages if isinstance(stage, ClipFrameExtractionStage))
    assert extraction_stage._target_fps == [1.0]


def test_embedding_sampling_fps_config_value_is_validated_when_embeddings_are_enabled() -> None:
    """Config-file values should be validated when stage assembly constructs embedding config."""
    args = _embedding_args([])
    args.embedding_sampling_fps = "invalid"

    with pytest.raises(ValueError, match=r"greater than or equal to 0\.001"):
        _assemble_stages(args)


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


def test_caption_quality_threshold_defaults() -> None:
    """Split CLI defaults should come from the shared default policy."""
    args = _parser().parse_args([])

    assert args.caption_quality_length_floor_words == DEFAULT_CAPTION_QUALITY_THRESHOLDS.length_floor_words
    assert args.caption_quality_length_ceiling_words == DEFAULT_CAPTION_QUALITY_THRESHOLDS.length_ceiling_words
    assert (
        args.caption_quality_repeated_trigram_min_count == DEFAULT_CAPTION_QUALITY_THRESHOLDS.repeated_trigram_min_count
    )
    assert (
        args.caption_quality_near_duplicate_jaccard_threshold
        == DEFAULT_CAPTION_QUALITY_THRESHOLDS.near_duplicate_jaccard_threshold
    )


def test_caption_quality_threshold_overrides_reach_captioning_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Split CLI threshold overrides should reach the captioning builder as one policy."""
    captured = _capture_captioning_config(monkeypatch)
    args = _caption_args(
        [
            "--caption-quality-length-floor-words",
            "6",
            "--caption-quality-length-ceiling-words",
            "900",
            "--caption-quality-repeated-trigram-min-count",
            "3",
            "--caption-quality-near-duplicate-jaccard-threshold",
            "0.75",
        ]
    )

    _assemble_stages(args)

    assert captured["config"].caption_quality_thresholds == CaptionQualityThresholdConfig(
        length_floor_words=6,
        length_ceiling_words=900,
        repeated_trigram_min_count=3,
        near_duplicate_jaccard_threshold=0.75,
    )


def test_caption_quality_threshold_invalid_combination_fails_stage_assembly() -> None:
    """Split stage assembly should validate the effective threshold policy."""
    args = _caption_args(
        [
            "--caption-quality-length-floor-words",
            "10",
            "--caption-quality-length-ceiling-words",
            "9",
        ]
    )

    with pytest.raises(ValueError, match="length_ceiling_words"):
        _assemble_stages(args)


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


def test_no_write_all_caption_json_flag_removed(capsys: pytest.CaptureFixture[str]) -> None:
    """The old negative flag should no longer be accepted."""
    with pytest.raises(SystemExit):
        _parser().parse_args(["--no-write-all-caption-json"])

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unrecognized arguments: --no-write-all-caption-json" in captured.err


def test_deprecated_vllm_preprocess_args_default_to_none() -> None:
    """Legacy preprocessing flags stay parseable with inert defaults."""
    args = _parser().parse_args([])

    assert args.qwen_preprocess_dtype is None
    assert args.qwen_model_does_preprocess is None


def test_deprecated_vllm_preprocess_args_are_documented() -> None:
    """Help text should point legacy preprocessing users at the new flag."""
    help_text = _parser().format_help()

    assert "--qwen-preprocess-dtype" in help_text
    assert "--qwen-model-does-preprocess" in help_text
    assert "--vllm-preprocess-mode" in help_text
    assert "Deprecated" in help_text


def test_debug_save_vllm_frames_help_text_names_png_preview_and_stats() -> None:
    """Debug frame help should describe both preview PNGs and tensor stats."""
    help_text = _parser().format_help()

    assert "--debug-save-vllm-frames" in help_text
    assert "PNG preview" in help_text
    assert "frame_stats.json" in help_text


@pytest.mark.parametrize(
    "legacy_args",
    [
        ["--qwen-preprocess-dtype", "float16"],
        ["--qwen-model-does-preprocess"],
    ],
)
def test_deprecated_vllm_preprocess_args_raise_migration_error(legacy_args: list[str]) -> None:
    """Using a legacy preprocessing flag should fail with the replacement flag named."""
    args = _caption_args(legacy_args)

    with pytest.raises(ValueError, match="--vllm-preprocess-mode"):
        _assemble_stages(args)


def test_ci_split_config_fixtures_exist() -> None:
    """The pre-canned CI split config library should be present and discoverable."""
    assert _CI_SPLIT_CONFIGS, "no examples/ci/*.json scenario configs found"


@pytest.mark.parametrize("config_path", _CI_SPLIT_CONFIGS, ids=lambda p: p.name)
def test_ci_split_configs_only_use_known_args(config_path: Path) -> None:
    """Pre-canned CI split configs must not carry typoed/unknown arg keys.

    Config mode (``run_pipeline <config.json>``) fills missing args with their
    defaults and silently ignores unrecognized keys, so a typo would otherwise
    sail through CI while quietly exercising the default instead of the intended
    value. Guard against that by checking every key against the split parser.
    """
    config = json.loads(config_path.read_text())
    assert config.get("pipeline") == "split", config_path.as_posix()

    valid_dests = set(vars(_parser().parse_args([])))
    unknown = sorted(set(config["args"]) - valid_dests)
    assert not unknown, f"{config_path.name}: unknown split args {unknown}"


def test_split_invoke_templates_use_supported_vllm_preprocess_mode() -> None:
    """Split invoke templates should not pass legacy qwen preprocessing args."""
    deprecated_args = {"qwen_preprocess_dtype", "qwen_model_does_preprocess"}

    for template_path in _SPLIT_INVOKE_TEMPLATES:
        invoke_args = json.loads(template_path.read_text())["args"]

        assert deprecated_args.isdisjoint(invoke_args), template_path.as_posix()
        assert invoke_args["vllm_preprocess_mode"] == PreprocessMode.CURATOR.value


def test_split_invoke_templates_expose_embedding_sampling_fps_default() -> None:
    """Full JSON invocation templates should expose the configurable embedding sampling default."""
    for template_path in _SPLIT_INVOKE_TEMPLATES:
        invoke_args = json.loads(template_path.read_text())["args"]

        assert invoke_args["embedding_sampling_fps"] == 2.0, template_path.as_posix()


def test_fill_default_args_does_not_inject_qwen_model_does_preprocess() -> None:
    """JSON config mode must not backfill a falsey default for the deprecated store_true flag."""
    input_path = Path.cwd() / "tmp-input"
    output_path = Path.cwd() / "tmp-output"
    args = argparse.Namespace(
        input_video_path=input_path.as_posix(),
        output_clip_path=output_path.as_posix(),
        vllm_preprocess_mode=PreprocessMode.CURATOR.value,
    )

    fill_default_args(args, _setup_parser, omit_dests=frozenset({"qwen_model_does_preprocess"}))
    assert not hasattr(args, "qwen_model_does_preprocess")


def test_fill_default_args_injects_caption_quality_threshold_defaults() -> None:
    """JSON/API split configs should receive the shared threshold defaults when omitted."""
    args = argparse.Namespace()

    fill_default_args(args, _setup_parser)

    assert args.caption_quality_length_floor_words == DEFAULT_CAPTION_QUALITY_THRESHOLDS.length_floor_words
    assert args.caption_quality_length_ceiling_words == DEFAULT_CAPTION_QUALITY_THRESHOLDS.length_ceiling_words
    assert (
        args.caption_quality_repeated_trigram_min_count == DEFAULT_CAPTION_QUALITY_THRESHOLDS.repeated_trigram_min_count
    )
    assert (
        args.caption_quality_near_duplicate_jaccard_threshold
        == DEFAULT_CAPTION_QUALITY_THRESHOLDS.near_duplicate_jaccard_threshold
    )


def test_legacy_qwen_model_does_preprocess_false_in_json_config_is_ignored() -> None:
    """Pre-migration invoke JSON used ``false`` for the deprecated store_true flag."""
    args = _caption_args([])
    args.qwen_model_does_preprocess = False
    args.vllm_preprocess_mode = PreprocessMode.CURATOR.value

    _assemble_stages(args)


def test_legacy_qwen_model_does_preprocess_true_raises_migration_error() -> None:
    """Explicit legacy model-preprocess opt-in must still fail."""
    args = _caption_args([])
    args.qwen_model_does_preprocess = True

    with pytest.raises(ValueError, match="--vllm-preprocess-mode"):
        _assemble_stages(args)


def test_qwen_captioning_keeps_legacy_sampling_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Qwen keeps the historical split-pipeline sampling defaults."""
    captured = _capture_captioning_config(monkeypatch)
    args = _caption_args(["--captioning-algorithm", "qwen"])

    _assemble_stages(args)

    config = captured["config"]
    assert isinstance(config.backend, VllmConfig)
    assert config.window_config.sampling_fps == WindowConfig().sampling_fps
    assert config.backend.sampling_config == VllmSamplingConfig()


@pytest.mark.parametrize("caption_algo", ["cosmos3_nano", "cosmos3_super"])
def test_cosmos3_sync_captioning_uses_model_generation_defaults(
    monkeypatch: pytest.MonkeyPatch,
    caption_algo: str,
) -> None:
    """Cosmos3 sync captioning resolves legacy generic defaults to model generation defaults."""
    captured = _capture_captioning_config(monkeypatch)
    args = _caption_args(["--captioning-algorithm", caption_algo])

    _assemble_stages(args)

    config = captured["config"]
    assert isinstance(config.backend, VllmConfig)
    assert config.window_config.sampling_fps == 4.0
    assert config.backend.sampling_config.temperature == 0.7
    assert config.backend.sampling_config.top_p == 0.8
    assert config.backend.sampling_config.top_k == 20
    assert config.backend.sampling_config.repetition_penalty == 1.0
    assert config.backend.sampling_config.min_tokens == 0
    assert config.backend.sampling_config.presence_penalty == 1.5


def test_cosmos3_sync_captioning_keeps_non_default_sampling_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-default CLI/config values still override Cosmos3 model defaults."""
    captured = _capture_captioning_config(monkeypatch)
    args = _caption_args(
        [
            "--captioning-algorithm",
            "cosmos3_nano",
            "--captioning-sampling-fps",
            "3.0",
            "--vllm-sampling-temperature",
            "0.2",
            "--vllm-sampling-top-p",
            "0.95",
            "--vllm-sampling-min-tokens",
            "7",
        ]
    )

    _assemble_stages(args)

    config = captured["config"]
    assert isinstance(config.backend, VllmConfig)
    assert config.window_config.sampling_fps == 3.0
    assert config.backend.sampling_config.temperature == 0.2
    assert config.backend.sampling_config.top_p == 0.95
    assert config.backend.sampling_config.top_k == 20
    assert config.backend.sampling_config.min_tokens == 7


def test_cosmos3_async_captioning_uses_model_generation_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """vllm_async uses the underlying Cosmos3 model name when resolving defaults."""
    captured = _capture_captioning_config(monkeypatch)
    args = _caption_args(
        [
            "--captioning-algorithm",
            "vllm_async",
            "--vllm-async-model-name",
            "cosmos3_nano",
        ]
    )

    _assemble_stages(args)

    config = captured["config"]
    assert config.window_config.sampling_fps == 4.0
    assert isinstance(config.backend, VllmAsyncCaptionConfig)
    assert config.backend.serve_config is not None
    assert config.backend.serve_config.sampling_config.temperature == 0.7
    assert config.backend.serve_config.sampling_config.top_p == 0.8
    assert config.backend.serve_config.sampling_config.top_k == 20
    assert config.backend.serve_config.sampling_config.repetition_penalty == 1.0
    assert config.backend.serve_config.sampling_config.min_tokens == 0
    assert config.backend.serve_config.sampling_config.presence_penalty == 1.5


def test_vllm_async_captioning_rejects_missing_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hand-built config namespaces should fail clearly when the async model name is missing."""
    _capture_captioning_config(monkeypatch)
    args = _caption_args(["--captioning-algorithm", "vllm_async"])
    args.vllm_async_model_name = None

    with pytest.raises(ValueError, match="--vllm-async-model-name"):
        _assemble_stages(args)


def _capture_event_vllm_async_config(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {}

    monkeypatch.setattr("cosmos_curator.pipelines.video.splitting_pipeline.build_sam3_tracking_stages", lambda _: [])

    def fake_build_event_caption_inner_stage(
        args: argparse.Namespace,
        *,
        vllm_async_config: VllmAsyncConfig | None = None,
        verbose: bool = False,  # noqa: ARG001
        log_stats: bool = False,  # noqa: ARG001
    ) -> object:
        captured["sampling_fps"] = args.event_caption_vllm_async_sampling_fps
        captured["config"] = vllm_async_config
        return object()

    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.splitting_pipeline.build_event_caption_inner_stage",
        fake_build_event_caption_inner_stage,
    )
    return captured


def test_cosmos3_event_vllm_async_uses_model_generation_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-event vllm_async resolves Cosmos3 sampling config and FPS defaults."""
    _capture_captioning_config(monkeypatch)
    captured = _capture_event_vllm_async_config(monkeypatch)
    args = _caption_args(
        [
            "--sam3",
            "--sam3-prompts",
            "person",
            "--event-captioning",
            "--event-caption-backend",
            "vllm_async",
            "--event-caption-vllm-async-model-name",
            "cosmos3_nano",
        ]
    )

    _assemble_stages(args)

    assert captured["sampling_fps"] == 4.0
    config = captured["config"]
    assert isinstance(config, VllmAsyncConfig)
    assert config.sampling_config.temperature == 0.7
    assert config.sampling_config.top_p == 0.8
    assert config.sampling_config.top_k == 20
    assert config.sampling_config.repetition_penalty == 1.0
    assert config.sampling_config.min_tokens == 0
    assert config.sampling_config.presence_penalty == 1.5


def test_cosmos3_event_vllm_async_keeps_sampling_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-event vllm_async keeps explicit FPS and global sampling overrides."""
    _capture_captioning_config(monkeypatch)
    captured = _capture_event_vllm_async_config(monkeypatch)
    args = _caption_args(
        [
            "--sam3",
            "--sam3-prompts",
            "person",
            "--event-captioning",
            "--event-caption-backend",
            "vllm_async",
            "--event-caption-vllm-async-model-name",
            "cosmos3_nano",
            "--event-caption-vllm-async-sampling-fps",
            "3.0",
            "--vllm-sampling-temperature",
            "0.2",
        ]
    )

    _assemble_stages(args)

    assert captured["sampling_fps"] == 3.0
    config = captured["config"]
    assert isinstance(config, VllmAsyncConfig)
    assert config.sampling_config.temperature == 0.2
    assert config.sampling_config.top_p == 0.8


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


def test_nemotron_forces_model_preprocess_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting nemotron must force vLLM/HF-owned preprocessing."""
    captured = _capture_captioning_config(monkeypatch)
    args = _caption_args(["--captioning-algorithm", "nemotron"])

    _assemble_stages(args)

    config = captured["config"]
    assert isinstance(config.backend, VllmConfig)
    assert config.backend.preprocess_mode == PreprocessMode.MODEL
    assert config.backend.model_preprocess_enabled is True


def test_vllm_video_max_pixels_help_text_names_scope_bounds_and_grid() -> None:
    """Help text should describe the upper-bound scope, bounds, and grid quantization."""
    help_text = " ".join(_parser().format_help().split())

    assert "--vllm-video-max-pixels-per-frame" in help_text
    assert "regular" in help_text
    assert "windowed sync vLLM" in help_text
    assert "[100352, 602112]" in help_text
    assert "28 for CPU prep" in help_text
    assert "32 for Qwen3" in help_text
