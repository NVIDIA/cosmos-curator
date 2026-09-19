# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned multimodal caption fields, identity, and terminal semantics."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from re import fullmatch
from types import MappingProxyType
from typing import Any, Final, cast

import pyarrow as pa

from cosmos_curator.core.sensors.types.types import TimestampSource
from cosmos_curator.models.registry import RegisteredModel, resolve_registered_model
from cosmos_curator.next.recipes.multimodal_caption.config import ContextVariant, ModelVariant
from cosmos_curator.next.recipes.multimodal_caption.ego_context import (
    EgoContextBuilder,
    build_ego_trajectory_absolute_poses,
    build_ego_trajectory_summary,
)
from cosmos_curator.next.recipes.multimodal_split.identities import MEDIA_CONTRACT_VERSION
from cosmos_curator.next.recipes.multimodal_split.records import RECORD_SCHEMA_VERSION
from cosmos_curator.next.recipes.video_caption.bindings import CaptionInferenceBindings
from cosmos_curator.next.recipes.video_caption.contracts import (
    DEFAULT_PROMPT,
    VIDEO_NUM_FRAMES,
    resolved_max_output_tokens,
    resolved_sampling_params,
)
from cosmos_curator.next.utils.identity import canonical_digest

CAPTION_SCHEMA_VERSION: Final = 1
RESULT_VERSION: Final = 1
TERMINAL_STATUSES: Final = frozenset({"success", "truncated", "error"})
OWNER_METADATA: Final = b"cosmos_curator.owner"
FIELD_SET_METADATA: Final = b"cosmos_curator.field_set"
CONTRACT_DIGEST_METADATA: Final = b"cosmos_curator.contract_digest"
EGO_PROMPT_SENTINEL: Final = "{ego_trajectory_context}"
EGO_PROMPT_TEMPLATE: Final = (
    "The video depicts the view from a camera mounted on a car as it is driving. Pay special attention to the motion "
    "of the cars, including the primary car whose point-of-view we observe in the video. Also note important factors "
    "that would relate to driving safety like the relative positions of pedestrians, lane markers, road signs, traffic "
    "signals, and any aggressive driving behavior of other vehicles. Also pay attention to interesting landmarks and "
    "describe them in detail."
    "\n\nEgo trajectory context for this episode (duration is the full episode span; motion covers only the recorded "
    f"in-span poses):\n{EGO_PROMPT_SENTINEL}"
)
EGO_SCHEMA_NAME: Final = "foxglove.PoseInFrame"
EGO_DECODER_MAPPING_DIGEST: Final = "2d1b47c77fe8781997601044974049727d5882999461a3fa6b183cc3cc554cbc"
EGO_ALIGNMENT_DOMAIN: Final = "episode_sensor"
EGO_REQUIRED_TIMESTAMP_SOURCE: Final = TimestampSource.SENSOR.value
EGO_CONTEXT_REPRESENTATION_VERSION: Final = 1
EGO_CONTEXT_CADENCE_PERIOD_NS: Final = 500_000_000
EGO_CONTEXT_MAX_NEAREST_POSE_DELTA_NS: Final = 20_000_000
EGO_CONTEXT_MAX_EPISODE_DURATION_NS: Final = 10_000_000_000
_SHA256_HEX_LENGTH: Final = 64
_RESULT_KEY_PATTERN: Final = r"[a-z][a-z0-9_]{0,47}"


class CaptionInputProfile(StrEnum):
    """Physical input capability required by one caption context."""

    CAMERA = "camera"
    EGO_TRAJECTORY = "ego_trajectory"


SUMMARY_SERIALIZATION: Final = {
    "top_level_fields": ["frame", "representation_version", "summary"],
    "samples": "omitted_after_qualification",
}
ABSOLUTE_POSES_SERIALIZATION: Final = {
    "top_level_fields": ["frame", "representation_version", "samples", "summary"],
    "samples": {
        "membership": "one_per_qualification_request",
        "order": "qualification_request_order",
        "fields": [
            "orientation_xyzw",
            "position_m",
            "relative_time_ns",
            "requested_align_timestamp_ns",
        ],
    },
}


@dataclass(frozen=True)
class CaptionContextDefinition:
    """One public context's representation capability and default prompt."""

    variant: ContextVariant
    input_profile: CaptionInputProfile
    ego_builder: EgoContextBuilder | None
    ego_serialization_contract: Mapping[str, object] | None
    default_prompt_template: str

    def __post_init__(self) -> None:
        """Reject internally inconsistent context capability definitions."""
        requires_ego = self.input_profile is CaptionInputProfile.EGO_TRAJECTORY
        if requires_ego != (self.ego_builder is not None) or requires_ego != (
            self.ego_serialization_contract is not None
        ):
            msg = "Ego input profile, builder, and serialization contract must be configured together"
            raise ValueError(msg)


@dataclass(frozen=True)
class EgoContextIdentity:
    """Configured Ego source plus the fixed v1 decoder and qualification policy."""

    sensor_key: str
    topic: str
    frame: str
    schema_name: str = EGO_SCHEMA_NAME
    decoder_mapping_digest: str = EGO_DECODER_MAPPING_DIGEST
    alignment_domain: str = EGO_ALIGNMENT_DOMAIN
    required_timestamp_source: str = EGO_REQUIRED_TIMESTAMP_SOURCE
    max_episode_duration_ns: int = EGO_CONTEXT_MAX_EPISODE_DURATION_NS
    representation_version: int = EGO_CONTEXT_REPRESENTATION_VERSION
    cadence_period_ns: int = EGO_CONTEXT_CADENCE_PERIOD_NS
    max_delta_ns: int = EGO_CONTEXT_MAX_NEAREST_POSE_DELTA_NS


CONTEXT_DEFINITIONS: Final[Mapping[ContextVariant, CaptionContextDefinition]] = MappingProxyType(
    {
        "video_only": CaptionContextDefinition(
            variant="video_only",
            input_profile=CaptionInputProfile.CAMERA,
            ego_builder=None,
            ego_serialization_contract=None,
            default_prompt_template=DEFAULT_PROMPT,
        ),
        "ego_trajectory_summary": CaptionContextDefinition(
            variant="ego_trajectory_summary",
            input_profile=CaptionInputProfile.EGO_TRAJECTORY,
            ego_builder=build_ego_trajectory_summary,
            ego_serialization_contract=SUMMARY_SERIALIZATION,
            default_prompt_template=EGO_PROMPT_TEMPLATE,
        ),
        "ego_trajectory_absolute_poses": CaptionContextDefinition(
            variant="ego_trajectory_absolute_poses",
            input_profile=CaptionInputProfile.EGO_TRAJECTORY,
            ego_builder=build_ego_trajectory_absolute_poses,
            ego_serialization_contract=ABSOLUTE_POSES_SERIALIZATION,
            default_prompt_template=EGO_PROMPT_TEMPLATE,
        ),
    }
)

_COSMOS3_HF_OVERRIDES: Final[dict[str, Any]] = {"architectures": ["Cosmos3ForConditionalGeneration"]}
# These intentionally repeat the legacy Cosmos3 plugin's launch choices. This
# recipe owns a versioned result contract, so importing mutable plugin defaults
# here could silently change published caption behavior and identity.
_COSMOS3_PREPARE_MODEL_CONFIG_KWARGS: Final[dict[str, Any]] = {"hf_overrides": _COSMOS3_HF_OVERRIDES}
_COSMOS3_ENGINE_KWARGS: Final[dict[str, Any]] = {
    # Route older downloaded configs through vLLM's native Cosmos3 executor.
    "hf_overrides": _COSMOS3_HF_OVERRIDES,
    "mm_encoder_tp_mode": "data",
    "allowed_local_media_path": "/",
    "limit_mm_per_prompt": {"images": 0, "video": 1},
    "max_model_len": 32_768,
}
_COSMOS3_CLOSED_THINKING_PREFILL: Final = "<think></think>"
_COSMOS3_VIDEO_FPS: Final = 4
_COSMOS3_SAMPLING_PARAMS: Final[dict[str, float | int]] = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "max_tokens": 2_048,
}


@dataclass(frozen=True)
class _CaptionModelBinding:
    """One recipe-supported model's fixed inference runtime."""

    inference: CaptionInferenceBindings


_MODEL_BINDINGS: Final[dict[ModelVariant, _CaptionModelBinding]] = {
    "qwen3_8_27b_fp8": _CaptionModelBinding(
        inference=CaptionInferenceBindings(engine_kwargs={"max_model_len": 32_768}),
    ),
    "qwen3_8_27b": _CaptionModelBinding(
        inference=CaptionInferenceBindings(engine_kwargs={"max_model_len": 32_768}),
    ),
    "cosmos3_nano": _CaptionModelBinding(
        inference=CaptionInferenceBindings(
            assistant_prefill=_COSMOS3_CLOSED_THINKING_PREFILL,
            # Ray detects a final assistant turn and supplies
            # ``continue_final_message=True`` itself. Supplying it here would
            # pass the argument twice to Transformers.
            chat_template_kwargs={},
            video_fps=_COSMOS3_VIDEO_FPS,
            prepare_model_config_kwargs=_COSMOS3_PREPARE_MODEL_CONFIG_KWARGS,
            request_mm_processor_kwargs={},
            sampling_params=_COSMOS3_SAMPLING_PARAMS,
            engine_kwargs=_COSMOS3_ENGINE_KWARGS,
        ),
    ),
}


def _model_binding(model_key: str) -> _CaptionModelBinding:
    # Registry keys arrive as plain strings, while the table is keyed by the
    # config literal so a typo in either list is a static error.
    binding = _MODEL_BINDINGS.get(cast("ModelVariant", model_key))
    if binding is None:
        msg = f"multimodal-caption has no model binding for {model_key!r}. Supported: {sorted(_MODEL_BINDINGS)}"
        raise KeyError(msg)
    return binding


def result_field_names(context_variant: str, result_key: str, model_key: str) -> tuple[str, str]:
    """Return one context/result/model's deterministic versioned field names."""
    resolve_context_definition(context_variant)
    if fullmatch(_RESULT_KEY_PATTERN, result_key) is None:
        msg = f"Invalid caption result key: {result_key!r}"
        raise ValueError(msg)
    _model_binding(model_key)
    caption_name = f"caption_{context_variant}_{result_key}_{model_key}_v{RESULT_VERSION}"
    return caption_name, f"{caption_name}_metadata"


def caption_inference_bindings(model_key: str) -> CaptionInferenceBindings:
    """Return the fixed vLLM/chat behavior for one supported model variant."""
    return _model_binding(model_key).inference


def resolve_context_definition(variant: str) -> CaptionContextDefinition:
    """Resolve one of the recipe's exact three context definitions."""
    definition = CONTEXT_DEFINITIONS.get(cast("ContextVariant", variant))
    if definition is None:
        msg = f"Unsupported caption context variant: {variant!r}. Supported: {sorted(CONTEXT_DEFINITIONS)}"
        raise ValueError(msg)
    return definition


def resolve_ego_identity(
    definition: CaptionContextDefinition,
    *,
    sensor_key: str | None,
    topic: str | None,
    frame: str | None,
) -> EgoContextIdentity | None:
    """Resolve configured Ego identity exactly for Ego-capable contexts."""
    values = (sensor_key, topic, frame)
    if definition.input_profile is CaptionInputProfile.CAMERA:
        if any(value is not None for value in values):
            msg = f"{definition.variant} forbids Ego source values"
            raise ValueError(msg)
        return None
    if any(not isinstance(value, str) or not value for value in values):
        msg = f"{definition.variant} requires non-empty sensor key, topic, and frame"
        raise ValueError(msg)
    return EgoContextIdentity(sensor_key=str(sensor_key), topic=str(topic), frame=str(frame))


def resolve_prompt_template(
    definition: CaptionContextDefinition,
    *,
    mode: str,
    custom_template: str | None,
) -> str:
    """Resolve and validate the one exact unrendered run prompt."""
    if mode == "default":
        if custom_template is not None:
            msg = "default prompt mode forbids a custom template"
            raise ValueError(msg)
        prompt = definition.default_prompt_template
    elif mode == "custom":
        if not isinstance(custom_template, str) or not custom_template.strip():
            msg = "custom prompt mode requires a non-whitespace template"
            raise ValueError(msg)
        prompt = custom_template
    else:
        msg = f"Unsupported prompt mode: {mode!r}"
        raise ValueError(msg)
    sentinel_count = prompt.count(EGO_PROMPT_SENTINEL)
    expected_count = 0 if definition.input_profile is CaptionInputProfile.CAMERA else 1
    if sentinel_count != expected_count:
        msg = (
            f"{definition.variant} prompt requires exactly {expected_count} literal "
            f"{EGO_PROMPT_SENTINEL} sentinel(s), got {sentinel_count}"
        )
        raise ValueError(msg)
    return prompt


def _ownership_metadata(field_set: str) -> dict[bytes, bytes]:
    return {OWNER_METADATA: b"multimodal-caption", FIELD_SET_METADATA: field_set.encode("utf-8")}


def logical_result_fields(
    context_variant: str,
    result_key: str,
    model_key: str,
    *,
    ego: EgoContextIdentity | None,
) -> pa.Schema:
    """Return the two logical fields before the derived digest is attached."""
    caption_name, metadata_name = result_field_names(context_variant, result_key, model_key)
    metadata_fields = [
        pa.field("caption_schema_version", pa.int32(), nullable=False),
        pa.field("result_key", pa.string(), nullable=False),
        pa.field("context_variant", pa.string(), nullable=False),
        pa.field("contract_digest", pa.string(), nullable=False),
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("materialization_digest", pa.string(), nullable=False),
        pa.field("camera_key", pa.string(), nullable=False),
        pa.field("media_sha256", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("model_key", pa.string(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("model_revision", pa.string(), nullable=False),
        pa.field("model_precision", pa.string(), nullable=False),
        pa.field("prompt", pa.large_string(), nullable=False),
        pa.field("prompt_token_count", pa.int64()),
        pa.field("generated_token_count", pa.int64()),
        pa.field("error_type", pa.string()),
        pa.field("error_message", pa.large_string()),
    ]
    if ego is not None:
        metadata_fields.extend(
            [
                pa.field("representation_version", pa.int32(), nullable=False),
                pa.field("frame", pa.string(), nullable=False),
                pa.field("cadence_period_ns", pa.int64(), nullable=False),
                pa.field("max_delta_ns", pa.int64(), nullable=False),
                pa.field("ego_sensor_key", pa.string(), nullable=False),
                pa.field("ego_sensor_type", pa.string(), nullable=False),
                pa.field("ego_media_sha256", pa.string(), nullable=False),
                pa.field("decoder_mapping_digest", pa.string(), nullable=False),
                pa.field("context_digest", pa.string()),
            ]
        )
    return pa.schema(
        [
            pa.field(caption_name, pa.large_string(), nullable=True, metadata=_ownership_metadata(caption_name)),
            pa.field(
                metadata_name,
                pa.struct(metadata_fields),
                nullable=True,
                metadata=_ownership_metadata(caption_name),
            ),
        ]
    )


def result_fields(
    context_variant: str,
    result_key: str,
    model_key: str,
    *,
    ego: EgoContextIdentity | None,
    digest: str,
) -> pa.Schema:
    """Return the physical registration fields carrying their derived contract digest."""
    logical = logical_result_fields(context_variant, result_key, model_key, ego=ego)
    fields = []
    for field in logical:
        metadata = dict(field.metadata or {})
        metadata[CONTRACT_DIGEST_METADATA] = digest.encode("utf-8")
        fields.append(field.with_metadata(metadata))
    return pa.schema(fields)


def staged_result_schema(
    context_variant: str,
    result_key: str,
    model_key: str,
    *,
    ego: EgoContextIdentity | None,
    digest: str,
) -> pa.Schema:
    """Return the durable episode recovery schema."""
    binding_fields = [
        pa.field("fragment_id", pa.int64(), nullable=False),
        pa.field("row_offset", pa.int64(), nullable=False),
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("materialization_digest", pa.string(), nullable=False),
        pa.field("camera_key", pa.string(), nullable=False),
        pa.field("media_location", pa.large_string(), nullable=False),
        pa.field("media_size_bytes", pa.int64(), nullable=False),
        pa.field("media_sha256", pa.string(), nullable=False),
    ]
    if ego is not None:
        binding_fields.extend(
            [
                pa.field("timestamp_source", pa.string(), nullable=False),
                pa.field("ego_sensor_key", pa.string(), nullable=False),
                pa.field("ego_sensor_type", pa.string(), nullable=False),
                pa.field("ego_media_location", pa.large_string(), nullable=False),
                pa.field("ego_media_size_bytes", pa.int64(), nullable=False),
                pa.field("ego_media_sha256", pa.string(), nullable=False),
            ]
        )
    return pa.schema(
        [
            *binding_fields,
            *result_fields(context_variant, result_key, model_key, ego=ego, digest=digest),
        ]
    )


def normalized_caption_contract(  # noqa: PLR0913 -- each value is independent contract identity
    model: RegisteredModel,
    camera_key: str,
    definition: CaptionContextDefinition,
    ego: EgoContextIdentity | None,
    prompt_template: str,
    result_key: str,
) -> dict[str, Any]:
    """Return every result-defining value for one selected caption result."""
    fields = logical_result_fields(definition.variant, result_key, model.key, ego=ego)
    inference = caption_inference_bindings(model.key)
    sampling_params = resolved_sampling_params(inference)
    max_output_tokens = resolved_max_output_tokens(inference)
    request_mm_processor_kwargs = deepcopy(dict(inference.request_mm_processor_kwargs))
    do_sample_frames = request_mm_processor_kwargs.get("do_sample_frames")
    video_contract: dict[str, Any] = {
        "container": "exact_episode_camera_mp4",
        "num_frames": VIDEO_NUM_FRAMES,
        "fps": inference.video_fps,
        "do_sample_frames": do_sample_frames,
        "longest_edge_override": None,
        "audio": "ignored",
    }
    # ``do_sample_frames`` already captures a one-key mapping. Preserve Qwen's
    # existing identity, but retain the full mapping when it contains extra
    # settings or distinguishes an explicit ``None`` from omitting the request
    # processor arguments entirely.
    if request_mm_processor_kwargs != {"do_sample_frames": do_sample_frames} or do_sample_frames is None:
        video_contract["request_mm_processor_kwargs"] = request_mm_processor_kwargs
    message: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "<exact-episode-camera-mp4-data-url>"}},
                {"type": "text", "text": prompt_template},
            ],
        }
    ]
    if inference.assistant_prefill is not None:
        message.append({"role": "assistant", "content": inference.assistant_prefill})
    failure_policy = {
        "item_media": "terminal_error",
        "multimodal_preparation": "fail_phase",
        "inference": "fail_phase",
        "shared_runtime_or_storage": "fail_phase",
    }
    if ego is not None:
        failure_policy.update(
            {
                "ego_decode": "terminal_error",
                "ego_complete_artifact_pose_validity": "terminal_error",
                "ego_alignment": "terminal_error",
            }
        )
    contract: dict[str, Any] = {
        "caption_schema_version": CAPTION_SCHEMA_VERSION,
        "result_key": result_key,
        "context_variant": definition.variant,
        "episode_contract": {
            "record_schema_version": RECORD_SCHEMA_VERSION,
            "media_contract_version": MEDIA_CONTRACT_VERSION,
            "camera_key": camera_key,
        },
        "model": {"id": model.model_id, "revision": model.revision, "precision": model.precision},
        "message": message,
        "chat_template": {"source": "model", "kwargs": deepcopy(dict(inference.chat_template_kwargs))},
        "sampling": deepcopy(sampling_params),
        "video": video_contract,
        "fields": [_arrow_field_contract(field) for field in fields],
        "field_registration": {
            "metadata_key": CONTRACT_DIGEST_METADATA.decode("utf-8"),
            "ownership": "both_registered_top_level_fields_carry_digest_derived_from_this_contract",
        },
        "token_counts": {
            "prompt_token_count": {"type": "integer", "nullable": True, "minimum": 0},
            "generated_token_count": {"type": "integer", "nullable": True, "minimum": 0},
        },
        "failure_policy": failure_policy,
        "terminal_states": {
            "success": {
                "caption": "non_null_string",
                "error_type": "null",
                "error_message": "null",
                "generated_token_count_if_present": {"less_than": max_output_tokens},
            },
            "truncated": {
                "caption": "non_null_string",
                "error_type": "null",
                "error_message": "null",
                "generated_token_count_if_present": {"greater_than_or_equal": max_output_tokens},
            },
            "error": {
                "caption": "null",
                "error_type": "non_empty_string",
                "error_message": "non_empty_string",
            },
        },
    }
    model_runtime: dict[str, Any] = {}
    if inference.prepare_model_config_kwargs:
        model_runtime["prepare_model_config_kwargs"] = deepcopy(dict(inference.prepare_model_config_kwargs))
    if inference.engine_kwargs:
        model_runtime["engine_kwargs"] = deepcopy(dict(inference.engine_kwargs))
    if model_runtime:
        contract["model_runtime"] = model_runtime
    if ego is not None:
        serialization = definition.ego_serialization_contract
        if serialization is None:
            msg = "Ego context definition is missing its serialization contract"
            raise RuntimeError(msg)
        contract["ego_context"] = {
            "sensor_key": ego.sensor_key,
            "topic": ego.topic,
            "frame": ego.frame,
            "schema_name": ego.schema_name,
            "decoder_mapping_digest": ego.decoder_mapping_digest,
            "alignment_domain": ego.alignment_domain,
            "required_timestamp_source": ego.required_timestamp_source,
            "timestamp_roles": {
                "sensor_timestamp_ns": "embedded_PoseInFrame_timestamp_decoded_source_provenance_not_serialized",
                "align_timestamp_ns": "MCAP_log_time_ns_used_for_span_predecessor_nearest_and_summary",
            },
            "max_episode_duration_ns": ego.max_episode_duration_ns,
            "representation_version": ego.representation_version,
            "pose_validity": "all_decoded_rows_including_predecessor_halo_must_be_true",
            "prompt_rendering": {
                "sentinel": EGO_PROMPT_SENTINEL,
                "replacement": "literal_replace_once",
                "unrelated_braces": "preserved",
            },
            "summary": {
                "membership": "mapper_alignment_timestamp_in_episode_half_open_span",
                "fields": ["duration_ns", "recorded_pose_count", "path_length_m", "net_displacement_m"],
            },
            "qualification": {
                "cadence_period_ns": ego.cadence_period_ns,
                "max_delta_ns": ego.max_delta_ns,
                "max_delta_boundary": "inclusive",
                "grid": "episode_start_plus_integer_cadence_half_open",
                "alignment": "nearest_without_interpolation_complete_split_artifact",
            },
            "serialization": deepcopy(serialization),
        }
    return contract


def caption_contract_digest(  # noqa: PLR0913 -- mirrors the complete normalized contract identity
    model: RegisteredModel,
    camera_key: str,
    definition: CaptionContextDefinition,
    ego: EgoContextIdentity | None,
    prompt_template: str,
    result_key: str,
) -> str:
    """Hash the normalized result contract canonically."""
    return canonical_digest(
        normalized_caption_contract(model, camera_key, definition, ego, prompt_template, result_key)
    )


def build_terminal_result_row(  # noqa: PLR0913 -- each argument maps to a persisted terminal value
    row: dict[str, Any],
    *,
    model: RegisteredModel,
    definition: CaptionContextDefinition,
    ego: EgoContextIdentity | None,
    prompt_template: str,
    result_key: str,
    digest: str,
    caption: str | None,
    status: str,
    prompt_token_count: int | None,
    generated_token_count: int | None,
    error_type: str | None,
    error_message: str | None,
) -> dict[str, Any]:
    """Build one episode-owned staged terminal row."""
    caption_name, metadata_name = result_field_names(definition.variant, result_key, model.key)
    staged: dict[str, Any] = {
        "fragment_id": int(row["fragment_id"]),
        "row_offset": int(row["row_offset"]),
        "episode_id": str(row["episode_id"]),
        "materialization_digest": str(row["materialization_digest"]),
        "camera_key": str(row["camera_key"]),
        "media_location": str(row["media_location"]),
        "media_size_bytes": int(row["media_size_bytes"]),
        "media_sha256": str(row["media_sha256"]),
    }
    metadata: dict[str, Any] = {
        "caption_schema_version": CAPTION_SCHEMA_VERSION,
        "result_key": result_key,
        "context_variant": definition.variant,
        "contract_digest": digest,
        "episode_id": staged["episode_id"],
        "materialization_digest": staged["materialization_digest"],
        "camera_key": staged["camera_key"],
        "media_sha256": staged["media_sha256"],
        "status": status,
        "model_key": model.key,
        "model_id": model.model_id,
        "model_revision": model.revision,
        "model_precision": model.precision,
        "prompt": prompt_template,
        "prompt_token_count": prompt_token_count,
        "generated_token_count": generated_token_count,
        "error_type": error_type,
        "error_message": error_message,
    }
    if ego is not None:
        # Terminal construction validates the persisted source evidence independently
        # of the preflight and projection snapshots.
        timestamp_source = row.get("timestamp_source")
        if timestamp_source != EGO_REQUIRED_TIMESTAMP_SOURCE:
            msg = f"Ego caption source binding requires timestamp_source {EGO_REQUIRED_TIMESTAMP_SOURCE!r}"
            raise ValueError(msg)
        staged.update(
            {
                "timestamp_source": timestamp_source,
                "ego_sensor_key": str(row["ego_sensor_key"]),
                "ego_sensor_type": str(row["ego_sensor_type"]),
                "ego_media_location": str(row["ego_media_location"]),
                "ego_media_size_bytes": int(row["ego_media_size_bytes"]),
                "ego_media_sha256": str(row["ego_media_sha256"]),
            }
        )
        metadata.update(
            {
                "representation_version": ego.representation_version,
                "frame": ego.frame,
                "cadence_period_ns": ego.cadence_period_ns,
                "max_delta_ns": ego.max_delta_ns,
                "ego_sensor_key": staged["ego_sensor_key"],
                "ego_sensor_type": staged["ego_sensor_type"],
                "ego_media_sha256": staged["ego_media_sha256"],
                "decoder_mapping_digest": ego.decoder_mapping_digest,
                "context_digest": row.get("context_digest"),
            }
        )
    return staged | {caption_name: caption, metadata_name: metadata}


def validate_terminal_value(  # noqa: C901, PLR0912, PLR0913 -- one validator owns terminal semantics
    caption: str | None,
    metadata: dict[str, Any] | None,
    *,
    model: RegisteredModel,
    camera_key: str,
    definition: CaptionContextDefinition,
    ego: EgoContextIdentity | None,
    prompt_template: str,
    result_key: str,
    digest: str,
) -> None:
    """Validate one terminal value and its episode/visual contract metadata."""
    max_output_tokens = resolved_max_output_tokens(caption_inference_bindings(model.key))
    if metadata is None:
        msg = "terminal caption metadata must not be null"
        raise ValueError(msg)
    schema_version = metadata.get("caption_schema_version")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        msg = "terminal caption metadata caption_schema_version must be an integer"
        raise TypeError(msg)
    expected = {
        "caption_schema_version": CAPTION_SCHEMA_VERSION,
        "result_key": result_key,
        "context_variant": definition.variant,
        "contract_digest": digest,
        "camera_key": camera_key,
        "model_key": model.key,
        "model_id": model.model_id,
        "model_revision": model.revision,
        "model_precision": model.precision,
        "prompt": prompt_template,
    }
    if ego is not None:
        expected.update(
            {
                "representation_version": ego.representation_version,
                "frame": ego.frame,
                "cadence_period_ns": ego.cadence_period_ns,
                "max_delta_ns": ego.max_delta_ns,
                "ego_sensor_key": ego.sensor_key,
                "ego_sensor_type": "egotrajectory",
                "decoder_mapping_digest": ego.decoder_mapping_digest,
            }
        )
    mismatched = [name for name, value in expected.items() if metadata.get(name) != value]
    if mismatched:
        msg = f"terminal caption metadata has incompatible value(s): {', '.join(mismatched)}"
        raise ValueError(msg)
    for name in ("episode_id", "materialization_digest", "media_sha256"):
        _require_nonempty_string(metadata.get(name), f"terminal caption metadata {name}")
    if ego is not None:
        _require_sha256(metadata.get("ego_media_sha256"), "terminal caption metadata ego_media_sha256")
    status = metadata.get("status")
    if status not in TERMINAL_STATUSES:
        msg = f"terminal caption metadata has unsupported status {status!r}"
        raise ValueError(msg)
    for count_name in ("prompt_token_count", "generated_token_count"):
        count = metadata.get(count_name)
        if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count < 0):
            msg = f"terminal caption metadata {count_name} must be a non-negative integer or null"
            raise ValueError(msg)
    generated = metadata.get("generated_token_count")
    if status == "error":
        error_type = metadata.get("error_type")
        error_message = metadata.get("error_message")
        if (
            caption is not None
            or not isinstance(error_type, str)
            or not error_type
            or not isinstance(error_message, str)
            or not error_message
        ):
            msg = "terminal error requires null caption and non-empty error fields"
            raise ValueError(msg)
    elif (
        not isinstance(caption, str)
        or metadata.get("error_type") is not None
        or metadata.get("error_message") is not None
    ):
        msg = f"terminal {status} requires a string caption and null error fields"
        raise ValueError(msg)
    elif generated is not None and ((status == "truncated") != (generated >= max_output_tokens)):
        msg = "terminal caption status is inconsistent with generated_token_count"
        raise ValueError(msg)
    if ego is not None:
        context_digest = metadata.get("context_digest")
        if status in {"success", "truncated"} or context_digest is not None:
            _require_sha256(context_digest, "terminal caption metadata context_digest")


def validate_terminal_row(  # noqa: PLR0913 -- validates all direct contract identifiers
    row: dict[str, Any],
    *,
    model: RegisteredModel,
    camera_key: str,
    definition: CaptionContextDefinition,
    ego: EgoContextIdentity | None,
    prompt_template: str,
    result_key: str,
    digest: str,
) -> None:
    """Validate a terminal value and its repeated staged episode bindings."""
    caption_name, metadata_name = result_field_names(definition.variant, result_key, model.key)
    validate_terminal_value(
        row.get(caption_name),
        row.get(metadata_name),
        model=model,
        camera_key=camera_key,
        definition=definition,
        ego=ego,
        prompt_template=prompt_template,
        result_key=result_key,
        digest=digest,
    )
    metadata = row[metadata_name]
    if not isinstance(metadata, dict):
        msg = "terminal caption metadata must be a mapping"
        raise TypeError(msg)
    if ego is not None and row.get("timestamp_source") != EGO_REQUIRED_TIMESTAMP_SOURCE:
        msg = f"Ego caption source binding requires timestamp_source {EGO_REQUIRED_TIMESTAMP_SOURCE!r}"
        raise ValueError(msg)
    validate_terminal_binding(
        metadata,
        episode_id=row.get("episode_id"),
        materialization_digest=row.get("materialization_digest"),
        camera_key=row.get("camera_key"),
        media_sha256=row.get("media_sha256"),
        timestamp_source=row.get("timestamp_source") if ego is not None else None,
        ego_sensor_key=row.get("ego_sensor_key") if ego is not None else None,
        ego_sensor_type=row.get("ego_sensor_type") if ego is not None else None,
        ego_media_sha256=row.get("ego_media_sha256") if ego is not None else None,
    )


def validate_terminal_binding(  # noqa: PLR0913 -- validates the complete persisted source binding
    metadata: dict[str, Any],
    *,
    episode_id: object,
    materialization_digest: object,
    camera_key: object,
    media_sha256: object,
    timestamp_source: object | None = None,
    ego_sensor_key: object | None = None,
    ego_sensor_type: object | None = None,
    ego_media_sha256: object | None = None,
) -> None:
    """Validate terminal metadata against one current or staged episode binding."""
    expected = {
        "episode_id": episode_id,
        "materialization_digest": materialization_digest,
        "camera_key": camera_key,
        "media_sha256": media_sha256,
    }
    # Recovery re-derives the live binding and compares it with the staged terminal
    # evidence; this is intentionally independent of terminal construction.
    if timestamp_source is not None:
        if timestamp_source != EGO_REQUIRED_TIMESTAMP_SOURCE:
            msg = f"Ego caption source binding requires timestamp_source {EGO_REQUIRED_TIMESTAMP_SOURCE!r}"
            raise ValueError(msg)
        expected.update(
            {
                "ego_sensor_key": ego_sensor_key,
                "ego_sensor_type": ego_sensor_type,
                "ego_media_sha256": ego_media_sha256,
            }
        )
    for name, value in expected.items():
        if metadata.get(name) != value:
            msg = f"terminal caption metadata {name} does not match the episode binding"
            raise ValueError(msg)


def resolve_model(model_key: str) -> RegisteredModel:
    """Resolve the shared registered model authority."""
    _model_binding(model_key)
    return resolve_registered_model(model_key)


def _require_nonempty_string(value: object, label: str) -> None:
    if not isinstance(value, str) or not value:
        msg = f"{label} must be a non-empty string"
        raise ValueError(msg)


def _require_sha256(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        msg = f"{label} must be a lowercase SHA-256 digest"
        raise ValueError(msg)


def _arrow_field_contract(field: pa.Field) -> dict[str, Any]:
    field_type = (
        {"name": "struct", "fields": [_arrow_field_contract(child) for child in field.type]}
        if pa.types.is_struct(field.type)
        else {"name": str(field.type)}
    )
    metadata = {key.decode(): value.decode() for key, value in sorted((field.metadata or {}).items())}
    return {
        "name": field.name,
        "type": field_type,
        "nullable": field.nullable,
        "metadata": metadata,
    }
