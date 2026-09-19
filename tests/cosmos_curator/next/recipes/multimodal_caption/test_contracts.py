# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for compositional multimodal caption contracts."""

from copy import deepcopy
from typing import Any

import pyarrow as pa
import pytest

from cosmos_curator.core.sensors.types.types import TimestampSource
from cosmos_curator.next.recipes.multimodal_caption.contracts import (
    ABSOLUTE_POSES_SERIALIZATION,
    CONTEXT_DEFINITIONS,
    CONTRACT_DIGEST_METADATA,
    EGO_CONTEXT_CADENCE_PERIOD_NS,
    EGO_CONTEXT_MAX_NEAREST_POSE_DELTA_NS,
    EGO_PROMPT_SENTINEL,
    EGO_REQUIRED_TIMESTAMP_SOURCE,
    SUMMARY_SERIALIZATION,
    CaptionContextDefinition,
    CaptionInputProfile,
    EgoContextIdentity,
    build_terminal_result_row,
    caption_contract_digest,
    caption_inference_bindings,
    logical_result_fields,
    normalized_caption_contract,
    resolve_context_definition,
    resolve_ego_identity,
    resolve_model,
    resolve_prompt_template,
    result_field_names,
    result_fields,
    staged_result_schema,
    validate_terminal_row,
)
from cosmos_curator.next.recipes.video_caption.contracts import (
    DEFAULT_PROMPT,
    resolve_model_spec,
    resolved_max_output_tokens,
    resolved_sampling_params,
)
from cosmos_curator.next.recipes.video_caption.contracts import (
    normalized_caption_contract as normalized_clip_caption_contract,
)
from cosmos_curator.next.recipes.video_caption.inference import caption_llm_preprocess

_MODELS = ("qwen3_8_27b_fp8", "qwen3_8_27b", "cosmos3_nano")
_CONTEXTS = ("video_only", "ego_trajectory_summary", "ego_trajectory_absolute_poses")


def _resolved_context(variant: str) -> tuple[CaptionContextDefinition, EgoContextIdentity | None, str]:
    definition = resolve_context_definition(variant)
    ego = resolve_ego_identity(
        definition,
        sensor_key=None if variant == "video_only" else "ego",
        topic=None if variant == "video_only" else "/ego",
        frame=None if variant == "video_only" else "world",
    )
    return definition, ego, resolve_prompt_template(definition, mode="default", custom_template=None)


def _identity(variant: str = "video_only", model_key: str = "qwen3_8_27b_fp8") -> tuple[Any, ...]:
    definition, ego, prompt = _resolved_context(variant)
    model = resolve_model(model_key)
    digest = caption_contract_digest(model, "front_wide", definition, ego, prompt, "default")
    return model, definition, ego, prompt, digest


def test_exact_three_by_three_product_composes_without_pair_registry() -> None:
    """Each independent model and context choice yields one complete contract."""
    assert tuple(CONTEXT_DEFINITIONS) == _CONTEXTS
    identities = set()
    for model_key in _MODELS:
        for variant in _CONTEXTS:
            model, definition, ego, prompt, digest = _identity(variant, model_key)
            contract = normalized_caption_contract(model, "front_wide", definition, ego, prompt, "default")
            names = result_field_names(variant, "default", model_key)
            assert [field["name"] for field in contract["fields"]] == list(names)
            assert contract["context_variant"] == variant
            assert contract["model"]["id"] == model.model_id
            assert len(digest) == 64
            identities.add((names, digest))
    assert len(identities) == 9


def test_unknown_model_and_context_fail_independently() -> None:
    """Neither identity axis has fallback behavior."""
    with pytest.raises(ValueError, match="Unsupported caption context"):
        resolve_context_definition("future_context")
    with pytest.raises(KeyError, match="no model binding"):
        caption_inference_bindings("future_model")
    with pytest.raises(KeyError):
        resolve_model("future_model")


def test_context_capability_and_ego_source_invariants_are_exact() -> None:
    """Camera and Ego capability definitions cannot form invalid combinations."""
    video = resolve_context_definition("video_only")
    assert video.input_profile is CaptionInputProfile.CAMERA
    assert resolve_ego_identity(video, sensor_key=None, topic=None, frame=None) is None
    with pytest.raises(ValueError, match="forbids Ego"):
        resolve_ego_identity(video, sensor_key="ego", topic="/ego", frame="world")

    for variant in _CONTEXTS[1:]:
        definition = resolve_context_definition(variant)
        assert definition.input_profile is CaptionInputProfile.EGO_TRAJECTORY
        with pytest.raises(ValueError, match="requires non-empty"):
            resolve_ego_identity(definition, sensor_key=None, topic=None, frame=None)

    with pytest.raises(ValueError, match="configured together"):
        CaptionContextDefinition(
            variant="video_only",
            input_profile=CaptionInputProfile.EGO_TRAJECTORY,
            ego_builder=None,
            ego_serialization_contract=None,
            default_prompt_template=DEFAULT_PROMPT,
        )


def test_context_defaults_and_custom_prompt_sentinel_rules_are_exact() -> None:
    """Context owns defaults while caller strings retain literal braces and exact bytes."""
    video = resolve_context_definition("video_only")
    summary = resolve_context_definition("ego_trajectory_summary")
    absolute = resolve_context_definition("ego_trajectory_absolute_poses")
    assert resolve_prompt_template(video, mode="default", custom_template=None) == DEFAULT_PROMPT
    assert summary.default_prompt_template == absolute.default_prompt_template
    assert summary.default_prompt_template.count(EGO_PROMPT_SENTINEL) == 1

    custom_video = "literal {unrelated} braces\n"
    assert resolve_prompt_template(video, mode="custom", custom_template=custom_video) == custom_video
    custom_ego = f"before {{unrelated}} {EGO_PROMPT_SENTINEL} after"
    assert resolve_prompt_template(summary, mode="custom", custom_template=custom_ego) == custom_ego
    with pytest.raises(ValueError, match="exactly 0"):
        resolve_prompt_template(video, mode="custom", custom_template=EGO_PROMPT_SENTINEL)
    with pytest.raises(ValueError, match="exactly 1"):
        resolve_prompt_template(summary, mode="custom", custom_template="missing")
    with pytest.raises(ValueError, match="exactly 1"):
        resolve_prompt_template(summary, mode="custom", custom_template=EGO_PROMPT_SENTINEL * 2)


def test_field_formula_and_digest_tagged_registration_are_exact() -> None:
    """Logical schema precedes digest derivation and both physical fields claim that digest."""
    model, definition, ego, prompt, digest = _identity("ego_trajectory_summary")
    names = result_field_names(definition.variant, "driving_summary_v2", model.key)
    assert names == (
        "caption_ego_trajectory_summary_driving_summary_v2_qwen3_8_27b_fp8_v1",
        "caption_ego_trajectory_summary_driving_summary_v2_qwen3_8_27b_fp8_v1_metadata",
    )
    digest = caption_contract_digest(model, "front_wide", definition, ego, prompt, "driving_summary_v2")
    logical = logical_result_fields(definition.variant, "driving_summary_v2", model.key, ego=ego)
    physical = result_fields(definition.variant, "driving_summary_v2", model.key, ego=ego, digest=digest)
    assert all(CONTRACT_DIGEST_METADATA not in (field.metadata or {}) for field in logical)
    assert all(field.metadata[CONTRACT_DIGEST_METADATA] == digest.encode() for field in physical)
    contract = normalized_caption_contract(model, "front_wide", definition, ego, prompt, "driving_summary_v2")
    assert digest not in str(contract)
    assert contract["field_registration"] == {
        "metadata_key": "cosmos_curator.contract_digest",
        "ownership": "both_registered_top_level_fields_carry_digest_derived_from_this_contract",
    }
    with pytest.raises(ValueError, match="Invalid caption result key"):
        result_field_names("video_only", "not-valid", model.key)


def test_metadata_and_recovery_schema_bind_direct_identity() -> None:
    """Ego schemas append only their provenance and stage sensor source explicitly."""
    model, definition, ego, _prompt, digest = _identity("ego_trajectory_absolute_poses")
    assert ego is not None
    fields = result_fields(definition.variant, "default", model.key, ego=ego, digest=digest)
    metadata = fields.field(1).type
    assert [field.name for field in metadata][:14] == [
        "caption_schema_version",
        "result_key",
        "context_variant",
        "contract_digest",
        "episode_id",
        "materialization_digest",
        "camera_key",
        "media_sha256",
        "status",
        "model_key",
        "model_id",
        "model_revision",
        "model_precision",
        "prompt",
    ]
    assert [field.name for field in metadata][-9:] == [
        "representation_version",
        "frame",
        "cadence_period_ns",
        "max_delta_ns",
        "ego_sensor_key",
        "ego_sensor_type",
        "ego_media_sha256",
        "decoder_mapping_digest",
        "context_digest",
    ]
    staged = staged_result_schema(definition.variant, "default", model.key, ego=ego, digest=digest)
    assert staged.field("timestamp_source").nullable is False


def test_normalized_ego_contract_binds_sensor_alignment_and_representation() -> None:
    """Sensor source, timestamp roles, qualification, and serialization are identity."""
    model, summary, ego, prompt, summary_digest = _identity("ego_trajectory_summary")
    assert ego is not None
    contract = normalized_caption_contract(model, "front_wide", summary, ego, prompt, "default")
    context = contract["ego_context"]
    assert context["required_timestamp_source"] == EGO_REQUIRED_TIMESTAMP_SOURCE == TimestampSource.SENSOR.value
    assert context["alignment_domain"] == "episode_sensor"
    assert context["timestamp_roles"] == {
        "sensor_timestamp_ns": "embedded_PoseInFrame_timestamp_decoded_source_provenance_not_serialized",
        "align_timestamp_ns": "MCAP_log_time_ns_used_for_span_predecessor_nearest_and_summary",
    }
    assert context["pose_validity"] == "all_decoded_rows_including_predecessor_halo_must_be_true"
    assert context["prompt_rendering"] == {
        "sentinel": EGO_PROMPT_SENTINEL,
        "replacement": "literal_replace_once",
        "unrelated_braces": "preserved",
    }
    assert context["qualification"] == {
        "cadence_period_ns": EGO_CONTEXT_CADENCE_PERIOD_NS,
        "max_delta_ns": EGO_CONTEXT_MAX_NEAREST_POSE_DELTA_NS,
        "max_delta_boundary": "inclusive",
        "grid": "episode_start_plus_integer_cadence_half_open",
        "alignment": "nearest_without_interpolation_complete_split_artifact",
    }
    assert contract["failure_policy"] == {
        "item_media": "terminal_error",
        "multimodal_preparation": "fail_phase",
        "inference": "fail_phase",
        "shared_runtime_or_storage": "fail_phase",
        "ego_decode": "terminal_error",
        "ego_complete_artifact_pose_validity": "terminal_error",
        "ego_alignment": "terminal_error",
    }
    assert context["serialization"] == SUMMARY_SERIALIZATION
    absolute, absolute_ego, absolute_prompt = _resolved_context("ego_trajectory_absolute_poses")
    absolute_contract = normalized_caption_contract(
        model, "front_wide", absolute, absolute_ego, absolute_prompt, "default"
    )
    assert absolute_contract["ego_context"]["serialization"] == ABSOLUTE_POSES_SERIALIZATION
    assert summary_digest != caption_contract_digest(
        model, "front_wide", absolute, absolute_ego, absolute_prompt, "default"
    )


def test_model_owned_runtime_is_composed_once_and_clip_contract_is_unchanged() -> None:
    """All recipe models own the ceiling; contexts and clip-caption do not."""
    for model_key in _MODELS:
        runtime = caption_inference_bindings(model_key)
        assert runtime.engine_kwargs["max_model_len"] == 32_768
        model, definition, ego, prompt, _ = _identity("video_only", model_key)
        contract = normalized_caption_contract(model, "front_wide", definition, ego, prompt, "default")
        assert contract["model_runtime"]["engine_kwargs"]["max_model_len"] == 32_768
    for definition in CONTEXT_DEFINITIONS.values():
        assert not hasattr(definition, "max_model_len")

    clip = normalized_clip_caption_contract(resolve_model_spec("qwen3_8_27b_fp8"))
    assert "model_runtime" not in clip
    assert "cosmos_curator.contract_digest" not in str(clip["fields"])


def test_shared_sampling_matches_runtime_and_both_contracts() -> None:
    """The shared request and both result contracts use one sampling contract."""
    model, definition, ego, prompt, _ = _identity("video_only", "qwen3_8_27b_fp8")
    runtime = caption_inference_bindings(model.key)
    request = caption_llm_preprocess(
        {"video_data_url": "data:video/mp4;base64,AAAA", "download_error_type": None},
        prompt_template=prompt,
        request_mm_processor_kwargs=runtime.request_mm_processor_kwargs,
        sampling_params=resolved_sampling_params(runtime),
    )
    clip_contract = normalized_clip_caption_contract(resolve_model_spec(model.key))
    episode_contract = normalized_caption_contract(model, "front_wide", definition, ego, prompt, "default")

    assert clip_contract["sampling"] == episode_contract["sampling"] == request["sampling_params"]


def test_cosmos3_contract_binds_native_runtime_and_closed_thinking() -> None:
    """Cosmos3 identity records its native vLLM launch and assistant prefill."""
    model, definition, ego, prompt, _ = _identity("video_only", "cosmos3_nano")
    runtime = caption_inference_bindings(model.key)
    request = caption_llm_preprocess(
        {"video_data_url": "data:video/mp4;base64,AAAA", "download_error_type": None},
        assistant_prefill=runtime.assistant_prefill,
        prompt_template=prompt,
        request_mm_processor_kwargs=runtime.request_mm_processor_kwargs,
        sampling_params=resolved_sampling_params(runtime),
    )
    contract = normalized_caption_contract(model, "front_wide", definition, ego, prompt, "default")

    assert request["messages"][-1] == {"role": "assistant", "content": "<think></think>"}
    assert "mm_processor_kwargs" not in request
    assert contract["message"][-1] == {"role": "assistant", "content": "<think></think>"}
    assert contract["chat_template"] == {"source": "model", "kwargs": {}}
    assert runtime.sampling_params is not None
    assert contract["sampling"] == request["sampling_params"] == dict(runtime.sampling_params)
    assert contract["terminal_states"]["truncated"]["generated_token_count_if_present"] == {
        "greater_than_or_equal": resolved_max_output_tokens(runtime)
    }
    assert contract["model_runtime"] == {
        "prepare_model_config_kwargs": {
            "hf_overrides": {"architectures": ["Cosmos3ForConditionalGeneration"]},
        },
        "engine_kwargs": {
            "hf_overrides": {"architectures": ["Cosmos3ForConditionalGeneration"]},
            "mm_encoder_tp_mode": "data",
            "allowed_local_media_path": "/",
            "limit_mm_per_prompt": {"images": 0, "video": 1},
            "max_model_len": 32_768,
        },
    }
    assert contract["video"]["fps"] == 4
    assert contract["video"]["request_mm_processor_kwargs"] == {}
    assert contract["video"]["do_sample_frames"] is None


def test_contract_is_deterministic_and_sensitive_to_every_public_axis() -> None:
    """Changes in direct result identity produce a new normalized digest."""
    model, definition, ego, prompt, digest = _identity()
    assert digest == caption_contract_digest(model, "front_wide", definition, ego, prompt, "default")
    assert digest != caption_contract_digest(model, "front_left", definition, ego, prompt, "default")
    assert digest != caption_contract_digest(model, "front_wide", definition, ego, "custom", "custom")
    assert digest != caption_contract_digest(model, "front_wide", definition, ego, prompt, "default_front")
    other_model = resolve_model("qwen3_8_27b")
    assert digest != caption_contract_digest(other_model, "front_wide", definition, ego, prompt, "default")
    assert "prompt_mode" not in normalized_caption_contract(model, "front_wide", definition, ego, prompt, "default")


def test_contract_runtime_mappings_are_fresh_copies() -> None:
    """A caller cannot mutate static Cosmos3 bindings through a returned contract."""
    model, definition, ego, prompt, _ = _identity("video_only", "cosmos3_nano")
    contract = normalized_caption_contract(model, "front_wide", definition, ego, prompt, "default")
    changed = deepcopy(contract)
    changed["model_runtime"]["prepare_model_config_kwargs"]["hf_overrides"]["architectures"].append("Mutated")
    fresh = normalized_caption_contract(model, "front_wide", definition, ego, prompt, "default")
    assert fresh["model_runtime"]["prepare_model_config_kwargs"]["hf_overrides"] == {
        "architectures": ["Cosmos3ForConditionalGeneration"]
    }


def test_terminal_row_repeats_identity_and_sensor_binding() -> None:
    """Recovery validates contract metadata and the staged sensor-source evidence."""
    model, definition, ego, prompt, digest = _identity("ego_trajectory_summary")
    assert ego is not None
    source = {
        "fragment_id": 4,
        "row_offset": 2,
        "episode_id": "episode",
        "materialization_digest": "materialization",
        "timestamp_source": "sensor",
        "camera_key": "front_wide",
        "media_location": "/front-wide.mp4",
        "media_size_bytes": 123,
        "media_sha256": "a" * 64,
        "ego_sensor_key": "ego",
        "ego_sensor_type": "egotrajectory",
        "ego_media_location": "/ego.mcap",
        "ego_media_size_bytes": 456,
        "ego_media_sha256": "b" * 64,
        "context_digest": "c" * 64,
    }
    row = build_terminal_result_row(
        source,
        model=model,
        definition=definition,
        ego=ego,
        prompt_template=prompt,
        result_key="default",
        digest=digest,
        caption="caption",
        status="success",
        prompt_token_count=10,
        generated_token_count=20,
        error_type=None,
        error_message=None,
    )
    schema = staged_result_schema(definition.variant, "default", model.key, ego=ego, digest=digest)
    assert pa.Table.from_pylist([row], schema=schema).num_rows == 1
    validate_terminal_row(
        row,
        model=model,
        camera_key="front_wide",
        definition=definition,
        ego=ego,
        prompt_template=prompt,
        result_key="default",
        digest=digest,
    )
    _, metadata_name = result_field_names(definition.variant, "default", model.key)
    assert row[metadata_name]["prompt"] == prompt
    assert row[metadata_name]["model_precision"] == model.precision
    row["timestamp_source"] = "container_pts"
    with pytest.raises(ValueError, match="requires timestamp_source 'sensor'"):
        validate_terminal_row(
            row,
            model=model,
            camera_key="front_wide",
            definition=definition,
            ego=ego,
            prompt_template=prompt,
            result_key="default",
            digest=digest,
        )
    with pytest.raises(ValueError, match="requires timestamp_source 'sensor'"):
        build_terminal_result_row(
            source | {"timestamp_source": "container_pts"},
            model=model,
            definition=definition,
            ego=ego,
            prompt_template=prompt,
            result_key="default",
            digest=digest,
            caption="caption",
            status="success",
            prompt_token_count=10,
            generated_token_count=20,
            error_type=None,
            error_message=None,
        )


def test_ego_terminal_error_allows_null_context_digest_and_validates_non_null_digest() -> None:
    """Current Ego item errors omit context identity; any carried digest still fails closed."""
    model, definition, ego, prompt, digest = _identity("ego_trajectory_summary")
    assert ego is not None
    source = {
        "fragment_id": 4,
        "row_offset": 2,
        "episode_id": "episode",
        "materialization_digest": "materialization",
        "timestamp_source": "sensor",
        "camera_key": "front_wide",
        "media_location": "/front-wide.mp4",
        "media_size_bytes": 123,
        "media_sha256": "a" * 64,
        "ego_sensor_key": "ego",
        "ego_sensor_type": "egotrajectory",
        "ego_media_location": "/ego.mcap",
        "ego_media_size_bytes": 456,
        "ego_media_sha256": "b" * 64,
        "context_digest": None,
    }
    row = build_terminal_result_row(
        source,
        model=model,
        definition=definition,
        ego=ego,
        prompt_template=prompt,
        result_key="default",
        digest=digest,
        caption=None,
        status="error",
        prompt_token_count=None,
        generated_token_count=None,
        error_type="ValueError",
        error_message="Ego context construction failed",
    )
    validate_terminal_row(
        row,
        model=model,
        camera_key="front_wide",
        definition=definition,
        ego=ego,
        prompt_template=prompt,
        result_key="default",
        digest=digest,
    )
    _, metadata_name = result_field_names(definition.variant, "default", model.key)
    assert row[metadata_name]["context_digest"] is None

    row[metadata_name]["context_digest"] = "invalid"
    with pytest.raises(ValueError, match="context_digest"):
        validate_terminal_row(
            row,
            model=model,
            camera_key="front_wide",
            definition=definition,
            ego=ego,
            prompt_template=prompt,
            result_key="default",
            digest=digest,
        )
