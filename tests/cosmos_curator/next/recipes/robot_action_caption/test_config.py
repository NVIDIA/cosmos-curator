# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify strict ownership and the public robot-action caption configuration.

Generation, video, and media settings define caption results; execution settings
only change how those results are produced. Tests resolve raw mappings through
the same strict Pydantic boundary used by YAML loading.
"""

import json
import pathlib
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from cosmos_curator.models import registry as model_registry
from cosmos_curator.next.recipes.robot_action_caption.config import (
    DEFAULT_MAX_CLIP_BYTES,
    KIND_NAME,
    MAX_SUPPORTED_CLIP_BYTES,
    ResolvedRobotActionCaptionConfig,
    RobotActionCaptionGenerationConfig,
    RobotActionCaptionModelConfig,
    RobotActionCaptionVideoConfig,
    config_template_payload,
    config_template_yaml,
    resolve_config_data,
)

CLIPS_URI = "s3://bucket-under-test/robot_clips/lance/clips.lance"
_QWEN2_TEMPORAL_PATCH_FRAMES = 2
_QWEN2_VISUAL_TOKEN_PIXELS = 28 * 28

type RawConfigFactory = Callable[..., dict[str, Any]]


@pytest.fixture
def raw_config() -> RawConfigFactory:
    """Return a builder for the smallest resolvable raw configuration."""

    def build(**sections: Any) -> dict[str, Any]:  # noqa: ANN401
        return {
            "schema_version": 1,
            "kind": KIND_NAME,
            "input": {"clips_lance_uri": CLIPS_URI},
            **sections,
        }

    return build


def test_a_minimal_config_resolves_with_every_section_defaulted(raw_config: RawConfigFactory) -> None:
    """Pin schema defaults independently of any runtime YAML."""
    resolved = resolve_config_data(raw_config())

    assert resolved.input.clips_lance_uri == CLIPS_URI
    assert resolved.input.selected_only is True
    assert resolved.model.variant == "qwen2.5_vl"
    assert resolved.generation == RobotActionCaptionGenerationConfig()
    assert resolved.video == RobotActionCaptionVideoConfig()
    assert (resolved.video.fps, resolved.video.min_frames, resolved.video.max_frames) == (None, None, None)
    assert (resolved.video.min_pixels, resolved.video.max_pixels) == (None, None)
    assert resolved.media.max_clip_bytes == DEFAULT_MAX_CLIP_BYTES
    assert resolved.execution.storage_profile == "default"
    assert resolved.execution.inference_concurrency == "auto"
    assert resolved.execution.inference_batch_size == 16
    assert resolved.execution.max_concurrent_batches == 8
    assert resolved.execution.tensor_parallel_size == 1
    assert resolved.execution.pipeline_parallel_size == 1
    assert resolved.execution.max_num_batched_tokens is None
    assert resolved.execution.media_attempts == 3
    assert resolved.execution.media_cpus_per_fetch == 0.25
    assert resolved.execution.lance_read_batch_size == 256
    assert resolved.execution.parquet_rows_per_file == 256
    assert resolved.execution.progress is False


def test_selected_only_can_be_disabled_by_a_dotted_override(raw_config: RawConfigFactory) -> None:
    """Allow an operator to caption every otherwise-pending row explicitly."""
    resolved = resolve_config_data(raw_config(), overrides=("input.selected_only=false",))

    assert resolved.input.selected_only is False


def test_default_video_config_emits_no_backend_overrides() -> None:
    """Leave temporal and pixel behavior to the selected backend by default."""
    video = RobotActionCaptionVideoConfig()

    assert video.media_io_kwargs() == {}
    assert video.mm_processor_kwargs() == {}


def test_explicit_qwen_runtime_policy_bounds_the_64_frame_visual_term() -> None:
    """Pin the documented run's visual bound without inventing a prompt allowance."""
    video = RobotActionCaptionVideoConfig(
        fps=3.0,
        min_frames=4,
        max_frames=64,
        min_pixels=3_136,
        max_pixels=602_112,
    )
    generation = RobotActionCaptionGenerationConfig()

    assert video.max_frames is not None
    assert video.max_pixels is not None
    temporal_positions = (video.max_frames + _QWEN2_TEMPORAL_PATCH_FRAMES - 1) // _QWEN2_TEMPORAL_PATCH_FRAMES
    visual_tokens = temporal_positions * (video.max_pixels // _QWEN2_VISUAL_TOKEN_PIXELS)
    residual_context_tokens = generation.max_model_len - visual_tokens

    assert visual_tokens == 24_576
    assert residual_context_tokens == 8_192
    assert generation.max_output_tokens < residual_context_tokens


def test_video_policy_emits_only_the_engine_keys_it_owns(raw_config: RawConfigFactory) -> None:
    """Translate configured temporal and spatial settings without inference."""
    video = resolve_config_data(
        raw_config(
            video={
                "fps": 5.0,
                "min_frames": 6,
                "max_frames": 80,
                "min_pixels": 100_352,
                "max_pixels": 1_204_224,
            }
        )
    ).video

    assert video.media_io_kwargs() == {"video": {"fps": 5.0, "min_frames": 6, "max_frames": 80}}
    assert video.mm_processor_kwargs() == {"min_pixels": 100_352, "max_pixels": 1_204_224}


@pytest.mark.parametrize(
    "video",
    [
        pytest.param({"fps": 5.0, "min_frames": 6}, id="missing-ceiling"),
        pytest.param({"fps": 5.0, "max_frames": 80}, id="missing-floor"),
        pytest.param({"min_frames": 6, "max_frames": 80}, id="missing-rate"),
    ],
)
def test_temporal_overrides_must_be_declared_as_a_group(
    raw_config: RawConfigFactory,
    video: dict[str, float | int],
) -> None:
    """Reject a partial temporal override group."""
    with pytest.raises(
        ValidationError,
        match=r"fps, video\.min_frames, and video\.max_frames must be set together",
    ):
        resolve_config_data(raw_config(video=video))


@pytest.mark.parametrize(
    "video",
    [
        pytest.param({"min_pixels": 100_352}, id="floor-only"),
        pytest.param({"max_pixels": 602_112}, id="ceiling-only"),
    ],
)
def test_pixel_overrides_must_be_declared_as_a_pair(
    raw_config: RawConfigFactory,
    video: dict[str, int | None],
) -> None:
    """Reject a partial processor-native pixel override."""
    with pytest.raises(ValidationError, match="must be set together"):
        resolve_config_data(raw_config(video=video))


@pytest.mark.parametrize(
    "video",
    [
        pytest.param({"min_pixels": 0, "max_pixels": 1}, id="zero-floor"),
        pytest.param({"min_pixels": 1, "max_pixels": 0}, id="zero-ceiling"),
        pytest.param({"min_pixels": -1, "max_pixels": 1}, id="negative-floor"),
        pytest.param({"min_pixels": 1, "max_pixels": -1}, id="negative-ceiling"),
    ],
)
def test_pixel_overrides_must_be_positive(raw_config: RawConfigFactory, video: dict[str, int]) -> None:
    """Reject non-positive processor-native pixel bounds."""
    with pytest.raises(ValidationError, match="greater than 0"):
        resolve_config_data(raw_config(video=video))


def test_pixel_floor_cannot_exceed_the_ceiling(raw_config: RawConfigFactory) -> None:
    """Reject an inverted processor-native pixel pair."""
    with pytest.raises(ValidationError, match=r"video\.max_pixels must be at least video\.min_pixels"):
        resolve_config_data(raw_config(video={"min_pixels": 602_112, "max_pixels": 100_352}))


def test_an_arbitrary_registered_alias_needs_no_recipe_allow_list(
    raw_config: RawConfigFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Accept any nonblank key that the global registry resolves."""
    alias = "future_video_model"
    registry_path = tmp_path / "all_models.json"
    registry_path.write_text(
        json.dumps(
            {
                alias: {
                    "model_id": "org/future-video-model",
                    "version": "a" * 40,
                    "precision": "bfloat16",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_registry, "_REGISTRY_PARTS", ())
    monkeypatch.setattr(model_registry, "files", lambda _package: registry_path)

    resolved = resolve_config_data(raw_config(model={"variant": alias}))

    assert resolved.model.variant == alias
    assert resolved.video.media_io_kwargs() == {}
    assert resolved.video.mm_processor_kwargs() == {}


def test_an_unregistered_model_alias_is_rejected(raw_config: RawConfigFactory) -> None:
    """Reject an alias absent from the global registry."""
    with pytest.raises(ValidationError, match="absent from"):
        resolve_config_data(raw_config(model={"variant": "not-a-registered-key"}))


def test_a_blank_model_alias_is_rejected(raw_config: RawConfigFactory) -> None:
    """Reject an alias that cannot identify a registry entry."""
    with pytest.raises(ValidationError, match="at least 1 character"):
        resolve_config_data(raw_config(model={"variant": ""}))


def test_semantic_models_expose_only_the_owned_fields() -> None:
    """Pin the public model, generation, and video field sets."""
    assert set(RobotActionCaptionModelConfig.model_fields) == {"variant"}
    assert set(RobotActionCaptionGenerationConfig.model_fields) == {
        "temperature",
        "top_p",
        "max_output_tokens",
        "max_model_len",
    }
    assert set(RobotActionCaptionVideoConfig.model_fields) == {
        "fps",
        "min_frames",
        "max_frames",
        "min_pixels",
        "max_pixels",
    }


def test_resolved_semantic_sections_use_default_factories() -> None:
    """Construct an independent model for each resolved semantic section."""
    fields = ResolvedRobotActionCaptionConfig.model_fields

    assert fields["model"].default_factory is RobotActionCaptionModelConfig
    assert fields["generation"].default_factory is RobotActionCaptionGenerationConfig
    assert fields["video"].default_factory is RobotActionCaptionVideoConfig


def test_the_rendered_template_documents_semantic_ownership() -> None:
    """Expose the defaults and recovery ownership in generated YAML."""
    rendered = config_template_yaml()

    assert "# `input.selected_only` is INPUT SCOPE" in rendered
    assert "# `model`, `generation`, `video`, and `media` are SEMANTIC" in rendered
    assert "# `execution` is NON-SEMANTIC" in rendered
    assert "# Temporal and pixel controls are independent optional groups." in rendered
    assert "# Omitting a complete group delegates that behavior to the selected backend." in rendered
    assert "  selected_only: true\n" in rendered
    assert "\nvideo:\n" in rendered


def test_template_payload_keeps_supported_optional_fields_explicitly_null() -> None:
    """Expose every optional video and scheduler setting in template metadata."""
    config = config_template_payload()["config"]

    assert config["video"]["fps"] is None
    assert config["video"]["min_frames"] is None
    assert config["video"]["max_frames"] is None
    assert config["video"]["min_pixels"] is None
    assert config["video"]["max_pixels"] is None
    assert config["execution"]["max_num_batched_tokens"] is None


def test_rendered_template_keeps_optional_fields_explicitly_null() -> None:
    """Keep null defaults visible through the public YAML renderer."""
    rendered = config_template_yaml()

    assert "  fps: null\n" in rendered
    assert "  max_num_batched_tokens: null\n" in rendered


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(MAX_SUPPORTED_CLIP_BYTES + 1, id="above-ceiling"),
    ],
)
def test_media_byte_cap_enforces_its_bounds(raw_config: RawConfigFactory, value: int) -> None:
    """Reject a byte cap outside the supported positive range."""
    with pytest.raises(ValidationError, match="max_clip_bytes"):
        resolve_config_data(raw_config(media={"max_clip_bytes": value}))


def test_media_byte_cap_accepts_its_supported_ceiling(raw_config: RawConfigFactory) -> None:
    """Accept the documented inclusive byte ceiling."""
    resolved = resolve_config_data(raw_config(media={"max_clip_bytes": MAX_SUPPORTED_CLIP_BYTES}))

    assert resolved.media.max_clip_bytes == MAX_SUPPORTED_CLIP_BYTES


@pytest.mark.parametrize(
    "removed",
    [
        pytest.param({"model": {"key": "a-model"}}, id="model-key-instead-of-variant"),
        pytest.param({"recovery": {"enabled": True}}, id="recovery-section"),
        pytest.param({"video": {"num_frames": 32}}, id="inert-frame-count"),
        pytest.param({"execution": {"rows_per_staged_file": 1000}}, id="rows-per-staged-file"),
        pytest.param({"media": {"num_frames": 16}}, id="frame-budget"),
        pytest.param({"media": {"max_output_tokens": 256}}, id="output-token-cap"),
        pytest.param({"execution": {"temperature": 0.2}}, id="semantic-under-execution"),
    ],
)
def test_an_unsupported_or_misplaced_knob_is_rejected(
    raw_config: RawConfigFactory,
    removed: dict[str, Any],
) -> None:
    """Reject an unowned knob instead of silently dropping it."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        resolve_config_data(raw_config(**removed))


@pytest.mark.parametrize(
    ("generation", "message"),
    [
        pytest.param({"temperature": -0.01}, "greater than or equal to 0", id="negative-temperature"),
        pytest.param({"temperature": float("inf")}, "finite number", id="infinite-temperature"),
        pytest.param({"top_p": 0.0}, "greater than 0", id="zero-top-p"),
        pytest.param({"top_p": 1.01}, "less than or equal to 1", id="top-p-above-one"),
        pytest.param({"max_output_tokens": 0}, "greater than or equal to 1", id="zero-output-tokens"),
        pytest.param({"max_model_len": 1}, "greater than or equal to 2", id="one-token-context"),
    ],
)
def test_generation_values_enforce_their_domains(
    raw_config: RawConfigFactory,
    generation: dict[str, Any],
    message: str,
) -> None:
    """Reject each generation value outside its declared domain."""
    with pytest.raises(ValidationError, match=message):
        resolve_config_data(raw_config(generation=generation))


def test_output_tokens_require_context_headroom(raw_config: RawConfigFactory) -> None:
    """Require at least one context token beyond the output cap."""
    with pytest.raises(ValidationError, match=r"generation\.max_output_tokens must be smaller"):
        resolve_config_data(raw_config(generation={"max_output_tokens": 8, "max_model_len": 8}))


@pytest.mark.parametrize("field", ["min_frames", "max_frames"])
def test_frame_bounds_must_be_positive(raw_config: RawConfigFactory, field: str) -> None:
    """Require at least one sampled frame."""
    with pytest.raises(ValidationError, match=rf"{field}[\s\S]*greater than or equal to 1"):
        resolve_config_data(raw_config(video={field: 0}))


def test_frame_floor_cannot_exceed_the_ceiling(raw_config: RawConfigFactory) -> None:
    """Reject inverted frame bounds."""
    with pytest.raises(ValidationError, match=r"video\.max_frames must be at least video\.min_frames"):
        resolve_config_data(raw_config(video={"fps": 3.0, "min_frames": 65, "max_frames": 64}))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        pytest.param(0.0, "greater than 0", id="zero"),
        pytest.param(-2.0, "greater than 0", id="negative"),
        pytest.param(float("inf"), "finite number", id="infinite"),
        pytest.param(float("nan"), "finite number", id="nan"),
    ],
)
def test_sampling_rate_must_be_usable(
    raw_config: RawConfigFactory,
    value: float,
    message: str,
) -> None:
    """Reject a target sampling rate the video loader cannot use."""
    with pytest.raises(ValidationError, match=rf"fps[\s\S]*{message}"):
        resolve_config_data(raw_config(video={"fps": value}))


@pytest.mark.parametrize(
    ("section", "message"),
    [
        pytest.param({"generation": {"temperature": "0.7"}}, "valid number", id="generation-string"),
        pytest.param({"video": {"max_frames": "32"}}, "valid integer", id="video-string"),
        pytest.param({"generation": {"unknown": 1}}, "Extra inputs", id="unknown-generation-field"),
        pytest.param({"video": {"unknown": 1}}, "Extra inputs", id="unknown-video-field"),
    ],
)
def test_semantic_sections_reject_coercion_and_unknown_fields(
    raw_config: RawConfigFactory,
    section: dict[str, Any],
    message: str,
) -> None:
    """Reject ambiguous input at the strict nested models."""
    with pytest.raises(ValidationError, match=message):
        resolve_config_data(raw_config(**section))


def test_execution_controls_enforce_progress_capable_values(raw_config: RawConfigFactory) -> None:
    """Reject execution settings that cannot perform work."""
    with pytest.raises(ValidationError, match="positive integer"):
        resolve_config_data(raw_config(execution={"inference_concurrency": 0}))
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        resolve_config_data(raw_config(execution={"media_attempts": 0}))


@pytest.mark.parametrize(
    "field",
    [
        pytest.param("inference_batch_size", id="inference-batch-size"),
        pytest.param("max_concurrent_batches", id="max-concurrent-batches"),
        pytest.param("tensor_parallel_size", id="tensor-parallel-size"),
        pytest.param("pipeline_parallel_size", id="pipeline-parallel-size"),
        pytest.param("lance_read_batch_size", id="lance-read-batch-size"),
        pytest.param("parquet_rows_per_file", id="parquet-rows-per-file"),
    ],
)
def test_execution_row_counts_require_positive_integers(raw_config: RawConfigFactory, field: str) -> None:
    """Reject zero for every configurable execution row count."""
    with pytest.raises(ValidationError, match=field):
        resolve_config_data(raw_config(execution={field: 0}))


@pytest.mark.parametrize("value", [0.0, -0.25, float("inf")])
def test_media_cpus_per_fetch_requires_a_positive_finite_value(
    raw_config: RawConfigFactory,
    value: float,
) -> None:
    """Reject a CPU reservation that cannot describe schedulable work."""
    with pytest.raises(ValidationError, match="media_cpus_per_fetch"):
        resolve_config_data(raw_config(execution={"media_cpus_per_fetch": value}))


def test_storage_profile_preserves_a_valid_unpadded_name(raw_config: RawConfigFactory) -> None:
    """Preserve an exact credential profile accepted at the config boundary."""
    resolved = resolve_config_data(raw_config(execution={"storage_profile": "production"}))

    assert resolved.execution.storage_profile == "production"


def test_storage_profile_rejects_surrounding_whitespace(raw_config: RawConfigFactory) -> None:
    """Reject a padded credential profile instead of normalizing it."""
    with pytest.raises(ValidationError, match="storage_profile cannot have surrounding whitespace"):
        resolve_config_data(raw_config(execution={"storage_profile": " default "}))


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(True, id="bool"),
        pytest.param(32_768.0, id="float"),
    ],
)
def test_scheduler_token_budget_requires_a_positive_integer(
    raw_config: RawConfigFactory,
    value: object,
) -> None:
    """Reject values vLLM cannot use as a scheduler token count."""
    with pytest.raises(ValidationError, match="max_num_batched_tokens"):
        resolve_config_data(raw_config(execution={"max_num_batched_tokens": value}))


def test_scheduler_token_budget_accepts_an_explicit_runtime_value(raw_config: RawConfigFactory) -> None:
    """Preserve a valid operator-supplied vLLM scheduler budget."""
    resolved = resolve_config_data(raw_config(execution={"max_num_batched_tokens": 32_768}))

    assert resolved.execution.max_num_batched_tokens == 32_768


def test_the_gpu_memory_fraction_defaults_to_deferring_to_vllm(raw_config: RawConfigFactory) -> None:
    """Absent from the YAML, the recipe must not invent a fraction of its own.

    ``None`` is the load-bearing default: it is what keeps vLLM's own value in
    force, and a recipe-side number would override the dependency silently on
    every run that never mentioned the knob.
    """
    assert resolve_config_data(raw_config()).execution.gpu_memory_utilization is None


def test_the_gpu_memory_fraction_preserves_an_operator_supplied_value(raw_config: RawConfigFactory) -> None:
    """Preserve the exact fraction an operator set.

    The value is neither vLLM's 0.9 default nor the 0.92 a later release moved it
    to, so a resolve that dropped the field and reported a dependency default
    cannot satisfy this.
    """
    resolved = resolve_config_data(raw_config(execution={"gpu_memory_utilization": 0.83}))

    assert resolved.execution.gpu_memory_utilization == 0.83


def test_the_gpu_memory_fraction_accepts_the_whole_device(raw_config: RawConfigFactory) -> None:
    """The inclusive ceiling must be reachable, or the bound is off by one device.

    Sits on the boundary the bound is written at: an exclusive ceiling would
    refuse the one value that asks for the whole GPU, which is a legitimate
    setting rather than an error.
    """
    assert (
        resolve_config_data(raw_config(execution={"gpu_memory_utilization": 1.0})).execution.gpu_memory_utilization
        == 1.0
    )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        pytest.param(0.0, "greater than 0", id="zero-leaves-no-memory"),
        pytest.param(-0.5, "greater than 0", id="negative"),
        pytest.param(1.01, "less than or equal to 1", id="above-the-whole-device"),
        pytest.param(float("inf"), "finite number", id="infinite"),
        pytest.param(float("nan"), "finite number", id="nan"),
    ],
)
def test_the_gpu_memory_fraction_rejects_an_unusable_share(
    raw_config: RawConfigFactory,
    value: float,
    message: str,
) -> None:
    """Refuse a share no GPU can honour, at parse time rather than at graph capture.

    vLLM resolves this into an absolute byte budget during startup profiling, so
    an out-of-range value surfaces as a CUDA failure minutes into a run that has
    already claimed its allocation.
    """
    with pytest.raises(ValidationError, match=rf"gpu_memory_utilization[\s\S]*{message}"):
        resolve_config_data(raw_config(execution={"gpu_memory_utilization": value}))


@pytest.mark.parametrize(
    "misspelling",
    [
        pytest.param("gpu_memory_utilisation", id="british-spelling"),
        pytest.param("gpu_mem_utilization", id="abbreviated"),
    ],
)
def test_a_near_miss_gpu_memory_key_is_refused_rather_than_ignored(
    raw_config: RawConfigFactory,
    misspelling: str,
) -> None:
    """A key that is nearly the real one must fail, not be silently dropped.

    The failure mode this closes is an operator raising residency in YAML,
    seeing no error, and getting the dependency default anyway.
    """
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        resolve_config_data(raw_config(execution={misspelling: 0.83}))


def test_prefix_caching_defaults_off(raw_config: RawConfigFactory) -> None:
    """The shipped default must be off, because vLLM's own default is on.

    Unlike its neighbours this field has no third "inherit the dependency" state,
    and that is the whole reason it exists: reuse is unreachable for this
    recipe's request shape, so inheriting vLLM's default would buy block hashing
    over every prompt and retained blocks inflating reported KV usage for a
    measured 0.0% hit rate. A default of ``True`` here, or a ``None`` that let
    the key be omitted, are the same defect.
    """
    assert resolve_config_data(raw_config()).execution.enable_prefix_caching is False


def test_prefix_caching_can_be_turned_on_from_the_yaml(raw_config: RawConfigFactory) -> None:
    """The other state must be reachable, or the field is a constant with a schema.

    Off is the right default rather than the only legal value: reshaping the
    request to lead with shared text makes reuse real, and an operator measuring
    that needs to be able to ask for it without editing the recipe.
    """
    assert resolve_config_data(raw_config(execution={"enable_prefix_caching": True})).execution.enable_prefix_caching


def test_a_file_uri_clip_table_normalizes_to_the_path_it_names(
    raw_config: RawConfigFactory,
    tmp_path: pathlib.Path,
) -> None:
    """A well-formed ``file://`` table URI resolves to its own local path."""
    table = tmp_path / "clips.lance"

    resolved = resolve_config_data(raw_config(input={"clips_lance_uri": table.as_uri()}))

    assert resolved.input.clips_lance_uri == str(table)


def test_a_file_uri_clip_table_with_an_unencoded_delimiter_is_refused(raw_config: RawConfigFactory) -> None:
    """A ``#`` or ``?`` in a ``file://`` table URI is refused rather than truncated.

    Both characters end a URI's path, so normalizing this location would hand the
    run ``/data/run`` - a shorter path that may well exist, and one no operator
    asked for. A percent-encoded delimiter still names the literal directory.
    """
    with pytest.raises(ValidationError, match=r"Unsupported local file URI for input\.clips_lance_uri"):
        resolve_config_data(raw_config(input={"clips_lance_uri": "file:///data/run#2/clips.lance"}))


def test_the_staging_root_defaults_beside_the_normalized_clip_table(raw_config: RawConfigFactory) -> None:
    """Derive one staging root for equivalent table URI spellings."""
    exact = resolve_config_data(raw_config())
    trailing_slash = resolve_config_data(raw_config(input={"clips_lance_uri": f"{CLIPS_URI}/"}))

    assert exact.staging_root == f"{CLIPS_URI}__caption_staging"
    assert trailing_slash.staging_root == exact.staging_root


def test_a_configured_staging_root_is_used_as_given(raw_config: RawConfigFactory) -> None:
    """Prefer an explicit staging root over the derived default."""
    configured = "s3://bucket-under-test/elsewhere/caption_staging"

    assert resolve_config_data(raw_config(output={"staging_root_uri": configured})).staging_root == configured
