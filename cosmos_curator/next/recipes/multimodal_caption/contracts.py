# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned ``video_only`` caption fields, identity, and terminal semantics."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Final, cast

import pyarrow as pa

from cosmos_curator.models.registry import RegisteredModel, resolve_registered_model
from cosmos_curator.next.recipes.multimodal_caption.config import ModelVariant
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
RESULT_VARIANT: Final = "video_only"
TERMINAL_STATUSES: Final = frozenset({"success", "truncated", "error"})
OWNER_METADATA: Final = b"cosmos_curator.owner"
FIELD_SET_METADATA: Final = b"cosmos_curator.field_set"

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
class _CaptionVariant:
    """One supported variant's owned field set and fixed inference runtime.

    Both halves are declared together so a new variant cannot publish a field
    set without also pinning the runtime its result identity records.
    """

    caption_field_name: str
    inference: CaptionInferenceBindings


_VARIANTS: Final[dict[ModelVariant, _CaptionVariant]] = {
    "qwen3_8_27b_fp8": _CaptionVariant(
        caption_field_name="caption_video_only_qwen3_8_27b_fp8_v1",
        inference=CaptionInferenceBindings(),
    ),
    "qwen3_8_27b": _CaptionVariant(
        caption_field_name="caption_video_only_qwen3_8_27b_v1",
        inference=CaptionInferenceBindings(),
    ),
    "cosmos3_nano": _CaptionVariant(
        caption_field_name="caption_video_only_cosmos3_nano_v1",
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


def _variant(model_key: str) -> _CaptionVariant:
    # Registry keys arrive as plain strings, while the table is keyed by the
    # config literal so a typo in either list is a static error.
    variant = _VARIANTS.get(cast("ModelVariant", model_key))
    if variant is None:
        msg = f"multimodal-caption has no variant binding for {model_key!r}. Supported: {sorted(_VARIANTS)}"
        raise KeyError(msg)
    return variant


def result_field_names(model_key: str) -> tuple[str, str]:
    """Return the versioned field set owned by one registered model variant."""
    caption_name = _variant(model_key).caption_field_name
    return caption_name, f"{caption_name}_metadata"


def caption_inference_bindings(model_key: str) -> CaptionInferenceBindings:
    """Return the fixed vLLM/chat behavior for one supported model variant."""
    return _variant(model_key).inference


def _ownership_metadata(field_set: str) -> dict[bytes, bytes]:
    return {OWNER_METADATA: b"multimodal-caption", FIELD_SET_METADATA: field_set.encode("utf-8")}


def result_fields(model_key: str) -> pa.Schema:
    """Return the two nullable public fields registered atomically."""
    caption_name, metadata_name = result_field_names(model_key)
    return pa.schema(
        [
            pa.field(caption_name, pa.large_string(), nullable=True, metadata=_ownership_metadata(caption_name)),
            pa.field(
                metadata_name,
                pa.struct(
                    [
                        pa.field("caption_schema_version", pa.int32(), nullable=False),
                        pa.field("result_variant", pa.string(), nullable=False),
                        pa.field("contract_digest", pa.string(), nullable=False),
                        pa.field("episode_id", pa.string(), nullable=False),
                        pa.field("materialization_digest", pa.string(), nullable=False),
                        pa.field("camera_key", pa.string(), nullable=False),
                        pa.field("media_sha256", pa.string(), nullable=False),
                        pa.field("status", pa.string(), nullable=False),
                        pa.field("model_id", pa.string(), nullable=False),
                        pa.field("model_revision", pa.string(), nullable=False),
                        pa.field("prompt", pa.large_string(), nullable=False),
                        pa.field("prompt_token_count", pa.int64()),
                        pa.field("generated_token_count", pa.int64()),
                        pa.field("error_type", pa.string()),
                        pa.field("error_message", pa.large_string()),
                    ]
                ),
                nullable=True,
                metadata=_ownership_metadata(caption_name),
            ),
        ]
    )


def staged_result_schema(model_key: str) -> pa.Schema:
    """Return the durable episode recovery schema."""
    return pa.schema(
        [
            pa.field("fragment_id", pa.int64(), nullable=False),
            pa.field("row_offset", pa.int64(), nullable=False),
            pa.field("episode_id", pa.string(), nullable=False),
            pa.field("materialization_digest", pa.string(), nullable=False),
            pa.field("camera_key", pa.string(), nullable=False),
            pa.field("media_location", pa.large_string(), nullable=False),
            pa.field("media_size_bytes", pa.int64(), nullable=False),
            pa.field("media_sha256", pa.string(), nullable=False),
            *result_fields(model_key),
        ]
    )


def normalized_caption_contract(model: RegisteredModel, camera_key: str) -> dict[str, Any]:
    """Return every result-defining ``video_only`` value."""
    fields = result_fields(model.key)
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
                {"type": "text", "text": DEFAULT_PROMPT},
            ],
        }
    ]
    if inference.assistant_prefill is not None:
        message.append({"role": "assistant", "content": inference.assistant_prefill})
    contract: dict[str, Any] = {
        "caption_schema_version": CAPTION_SCHEMA_VERSION,
        "result_variant": RESULT_VARIANT,
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
        "token_counts": {
            "prompt_token_count": {"type": "integer", "nullable": True, "minimum": 0},
            "generated_token_count": {"type": "integer", "nullable": True, "minimum": 0},
        },
        "failure_policy": {
            "item_media": "terminal_error",
            "multimodal_preparation": "fail_phase",
            "inference": "fail_phase",
            "shared_runtime_or_storage": "fail_phase",
        },
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
    return contract


def caption_contract_digest(model: RegisteredModel, camera_key: str) -> str:
    """Hash the normalized result contract canonically."""
    return canonical_digest(normalized_caption_contract(model, camera_key))


def build_terminal_result_row(  # noqa: PLR0913 -- each argument maps to a persisted terminal value
    row: dict[str, Any],
    *,
    model: RegisteredModel,
    digest: str,
    caption: str | None,
    status: str,
    prompt_token_count: int | None,
    generated_token_count: int | None,
    error_type: str | None,
    error_message: str | None,
) -> dict[str, Any]:
    """Build one episode-owned staged terminal row."""
    caption_name, metadata_name = result_field_names(model.key)
    return {
        "fragment_id": int(row["fragment_id"]),
        "row_offset": int(row["row_offset"]),
        "episode_id": str(row["episode_id"]),
        "materialization_digest": str(row["materialization_digest"]),
        "camera_key": str(row["camera_key"]),
        "media_location": str(row["media_location"]),
        "media_size_bytes": int(row["media_size_bytes"]),
        "media_sha256": str(row["media_sha256"]),
        caption_name: caption,
        metadata_name: {
            "caption_schema_version": CAPTION_SCHEMA_VERSION,
            "result_variant": RESULT_VARIANT,
            "contract_digest": digest,
            "episode_id": str(row["episode_id"]),
            "materialization_digest": str(row["materialization_digest"]),
            "camera_key": str(row["camera_key"]),
            "media_sha256": str(row["media_sha256"]),
            "status": status,
            "model_id": model.model_id,
            "model_revision": model.revision,
            "prompt": DEFAULT_PROMPT,
            "prompt_token_count": prompt_token_count,
            "generated_token_count": generated_token_count,
            "error_type": error_type,
            "error_message": error_message,
        },
    }


def validate_terminal_value(  # noqa: C901 -- terminal states are intentionally validated in one place
    caption: str | None,
    metadata: dict[str, Any] | None,
    *,
    model: RegisteredModel,
    camera_key: str,
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
        "result_variant": RESULT_VARIANT,
        "contract_digest": digest,
        "camera_key": camera_key,
        "model_id": model.model_id,
        "model_revision": model.revision,
        "prompt": DEFAULT_PROMPT,
    }
    mismatched = [name for name, value in expected.items() if metadata.get(name) != value]
    if mismatched:
        msg = f"terminal caption metadata has incompatible value(s): {', '.join(mismatched)}"
        raise ValueError(msg)
    for name in ("episode_id", "materialization_digest", "media_sha256"):
        value = metadata.get(name)
        if not isinstance(value, str) or not value:
            msg = f"terminal caption metadata {name} must be a non-empty string"
            raise ValueError(msg)
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


def validate_terminal_row(
    row: dict[str, Any],
    *,
    model: RegisteredModel,
    camera_key: str,
    digest: str,
) -> None:
    """Validate a terminal value and its repeated staged episode bindings."""
    caption_name, metadata_name = result_field_names(model.key)
    validate_terminal_value(
        row.get(caption_name),
        row.get(metadata_name),
        model=model,
        camera_key=camera_key,
        digest=digest,
    )
    metadata = row[metadata_name]
    if not isinstance(metadata, dict):
        msg = "terminal caption metadata must be a mapping"
        raise TypeError(msg)
    validate_terminal_binding(
        metadata,
        episode_id=row.get("episode_id"),
        materialization_digest=row.get("materialization_digest"),
        camera_key=row.get("camera_key"),
        media_sha256=row.get("media_sha256"),
    )


def validate_terminal_binding(
    metadata: dict[str, Any],
    *,
    episode_id: object,
    materialization_digest: object,
    camera_key: object,
    media_sha256: object,
) -> None:
    """Validate terminal metadata against one current or staged episode binding."""
    expected = {
        "episode_id": episode_id,
        "materialization_digest": materialization_digest,
        "camera_key": camera_key,
        "media_sha256": media_sha256,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            msg = f"terminal caption metadata {name} does not match the episode binding"
            raise ValueError(msg)


def resolve_model(model_key: str) -> RegisteredModel:
    """Resolve the shared registered model authority."""
    return resolve_registered_model(model_key)


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
