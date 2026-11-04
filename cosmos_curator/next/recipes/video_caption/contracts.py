# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned model identities, Arrow fields, and caption-result semantics."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Final

import pyarrow as pa

from cosmos_curator.core.utils import environment
from cosmos_curator.next.enrichment.contracts import owned_fields
from cosmos_curator.next.recipes.video_caption.config import ModelVariant
from cosmos_curator.next.utils.identity import canonical_digest

CAPTION_SCHEMA_VERSION: Final = 1
DEFAULT_PROMPT: Final = "Elaborate on the visual and narrative elements of the video in detail."
# Clip columns processing reads besides clip_id and row addresses.
READ_COLUMNS: Final = ("clip_uri", "clip_size_bytes")
MAX_OUTPUT_TOKENS: Final = 2_048
SAMPLING_PARAMS: Final[dict[str, float | int]] = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "max_tokens": MAX_OUTPUT_TOKENS,
}
VIDEO_NUM_FRAMES: Final = -1
VIDEO_FPS: Final = 2
VIDEO_DO_SAMPLE_FRAMES: Final = False
TERMINAL_STATUSES: Final = frozenset({"success", "truncated", "error"})
_MODEL_WEIGHT_SUFFIXES: Final = frozenset({".bin", ".pt", ".safetensors"})


@dataclass(frozen=True)
class CaptionInferenceBindings:
    """Model-specific settings consumed by the shared Ray vLLM processor.

    Public recipes select one of these static bindings. They are deliberately
    not user configuration: per-run execution sizing belongs in the recipe's
    execution config and model behavior belongs in the result contract.
    """

    assistant_prefill: str | None = None
    chat_template_kwargs: Mapping[str, Any] = field(default_factory=lambda: {"enable_thinking": False})
    video_fps: int | float = 2
    prepare_model_config_kwargs: Mapping[str, Any] = field(default_factory=dict)
    request_mm_processor_kwargs: Mapping[str, Any] = field(default_factory=lambda: {"do_sample_frames": False})
    sampling_params: Mapping[str, float | int] | None = None
    engine_kwargs: Mapping[str, Any] = field(default_factory=dict)


def resolved_sampling_params(inference: CaptionInferenceBindings) -> dict[str, float | int]:
    """Return one binding's sampling values, falling back to the legacy Qwen contract."""
    return dict(SAMPLING_PARAMS if inference.sampling_params is None else inference.sampling_params)


def resolved_max_output_tokens(inference: CaptionInferenceBindings) -> int:
    """Return the model-bound generation cutoff used for terminal-state classification."""
    max_tokens = resolved_sampling_params(inference).get("max_tokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        msg = f"Caption inference sampling max_tokens must be a positive integer, got {max_tokens!r}"
        raise ValueError(msg)
    return max_tokens


@dataclass(frozen=True)
class CaptionModelSpec:
    """One supported pinned model and its versioned output field set."""

    variant: ModelVariant
    model_id: str
    revision: str
    precision: str
    caption_field_name: str
    metadata_field_name: str

    @property
    def runtime_model_dir(self) -> Path:
        """Return the path mounted identically on every runtime worker."""
        return environment.CONTAINER_PATHS_MODEL_WEIGHT_CACHE_DIR / self.model_id

    @property
    def fields(self) -> pa.Schema:
        """Return the owned caption and terminal-metadata fields, registered in one schema transaction."""
        metadata = pa.struct(
            [
                pa.field("caption_schema_version", pa.int32(), nullable=False),
                pa.field("contract_digest", pa.string(), nullable=False),
                pa.field("status", pa.string(), nullable=False),
                pa.field("model_id", pa.string(), nullable=False),
                pa.field("model_revision", pa.string(), nullable=False),
                pa.field("prompt", pa.large_string(), nullable=False),
                pa.field("prompt_token_count", pa.int64()),
                pa.field("generated_token_count", pa.int64()),
                pa.field("error_type", pa.string()),
                pa.field("error_message", pa.large_string()),
            ]
        )
        return owned_fields(
            pa.schema([(self.caption_field_name, pa.large_string()), (self.metadata_field_name, metadata)]),
            owner="video-caption",
            field_set=self.caption_field_name,
        )


_VARIANT_FIELDS: Final[dict[ModelVariant, tuple[str, str]]] = {
    "qwen3_8_27b_fp8": ("caption_qwen3_8_27b_fp8_v1", "caption_qwen3_8_27b_fp8_v1_metadata"),
    "qwen3_8_27b": ("caption_qwen3_8_27b_v1", "caption_qwen3_8_27b_v1_metadata"),
}


def resolve_model_spec(variant: ModelVariant) -> CaptionModelSpec:
    """Resolve the model ID and immutable revision from ``all_models.json``."""
    config_path = files("cosmos_curator").joinpath("configs", "all_models.json")
    raw_models: Any = json.loads(config_path.read_text(encoding="utf-8"))
    raw_spec = raw_models.get(variant) if isinstance(raw_models, dict) else None
    if not isinstance(raw_spec, dict):
        msg = f"Model variant {variant!r} is absent from cosmos_curator/configs/all_models.json"
        raise TypeError(msg)
    model_id = raw_spec.get("model_id")
    revision = raw_spec.get("version")
    precision = raw_spec.get("precision")
    if (
        not isinstance(model_id, str)
        or not model_id
        or not isinstance(revision, str)
        or not revision
        or not isinstance(precision, str)
        or not precision
    ):
        msg = f"Model variant {variant!r} must have non-empty model_id, version, and precision values"
        raise TypeError(msg)
    caption_field, metadata_field = _VARIANT_FIELDS[variant]
    return CaptionModelSpec(
        variant=variant,
        model_id=model_id,
        revision=revision,
        precision=precision,
        caption_field_name=caption_field,
        metadata_field_name=metadata_field,
    )


def normalized_caption_contract(spec: CaptionModelSpec, *, max_model_len: int | None = None) -> dict[str, Any]:
    """Return every result-defining value in a JSON-compatible shape.

    A context limit ends a generation early once prompt plus output reaches it, and the result still
    validates as ``success`` below the output cap. A set limit is therefore result-defining. Null keeps
    vLLM's default and leaves the contract, and its digest, unchanged.
    """
    contract: dict[str, Any] = {
        "caption_schema_version": CAPTION_SCHEMA_VERSION,
        "model": {
            "id": spec.model_id,
            "revision": spec.revision,
            "precision": spec.precision,
        },
        "message": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": "<exact-clip-mp4-data-url>"}},
                    {"type": "text", "text": DEFAULT_PROMPT},
                ],
            }
        ],
        "chat_template": {"source": "model", "kwargs": {"enable_thinking": False}},
        "sampling": dict(SAMPLING_PARAMS),
        "video": {
            "container": "exact_clip_mp4",
            "num_frames": VIDEO_NUM_FRAMES,
            "fps": VIDEO_FPS,
            "do_sample_frames": VIDEO_DO_SAMPLE_FRAMES,
            "longest_edge_override": None,
            "audio": "ignored",
        },
        "fields": [_arrow_field_contract(field) for field in spec.fields],
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
                "generated_token_count_if_present": {"less_than": MAX_OUTPUT_TOKENS},
            },
            "truncated": {
                "caption": "non_null_string",
                "error_type": "null",
                "error_message": "null",
                "generated_token_count_if_present": {"greater_than_or_equal": MAX_OUTPUT_TOKENS},
            },
            "error": {
                "caption": "null",
                "error_type": "non_empty_string",
                "error_message": "non_empty_string",
            },
        },
    }
    if max_model_len is not None:
        contract["model_runtime"] = {"max_model_len": max_model_len}
    return contract


def caption_contract_digest(spec: CaptionModelSpec, *, max_model_len: int | None = None) -> str:
    """Hash the normalized caption result contract with canonical JSON rules."""
    return canonical_digest(normalized_caption_contract(spec, max_model_len=max_model_len))


def _arrow_field_contract(field: pa.Field) -> dict[str, Any]:
    field_type: dict[str, Any]
    if pa.types.is_struct(field.type):
        field_type = {"name": "struct", "fields": [_arrow_field_contract(child) for child in field.type]}
    else:
        field_type = {"name": str(field.type)}
    metadata = {key.decode("utf-8"): value.decode("utf-8") for key, value in sorted((field.metadata or {}).items())}
    return {
        "name": field.name,
        "type": field_type,
        "nullable": field.nullable,
        "metadata": metadata,
    }


def validate_model_directory(spec: CaptionModelSpec) -> Path:
    """Require pre-staged local files without attempting any download."""
    model_dir = spec.runtime_model_dir
    if not model_dir.is_dir():
        msg = (
            f"Model weights for {spec.variant!r} are not staged at {model_dir}. "
            f"Run: pixi run --as-is model-download --models {spec.variant}"
        )
        raise FileNotFoundError(msg)
    config_path = model_dir / "config.json"
    has_model_file = any(path.is_file() and path.suffix in _MODEL_WEIGHT_SUFFIXES for path in model_dir.rglob("*"))
    if not config_path.is_file() or not has_model_file:
        msg = f"Model directory {model_dir} is incomplete: expected config.json and staged model files"
        raise FileNotFoundError(msg)
    return model_dir


def terminal_metadata(  # noqa: PLR0913 -- each argument maps to a persisted field
    spec: CaptionModelSpec,
    digest: str,
    *,
    status: str,
    prompt_token_count: int | None,
    generated_token_count: int | None,
    error_type: str | None,
    error_message: str | None,
) -> dict[str, Any]:
    """Build one canonical terminal metadata value."""
    return {
        "caption_schema_version": CAPTION_SCHEMA_VERSION,
        "contract_digest": digest,
        "status": status,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "prompt": DEFAULT_PROMPT,
        "prompt_token_count": prompt_token_count,
        "generated_token_count": generated_token_count,
        "error_type": error_type,
        "error_message": error_message,
    }


def build_terminal_result_row(  # noqa: PLR0913 -- each argument maps to a persisted terminal value
    row: dict[str, Any],
    *,
    spec: CaptionModelSpec,
    digest: str,
    caption: str | None,
    status: str,
    prompt_token_count: int | None,
    generated_token_count: int | None,
    error_type: str | None,
    error_message: str | None,
) -> dict[str, Any]:
    """Build one clip-owned staged terminal row from shared inference values."""
    return {
        "fragment_id": int(row["fragment_id"]),
        "row_offset": int(row["row_offset"]),
        "clip_id": str(row["clip_id"]),
        spec.caption_field_name: caption,
        spec.metadata_field_name: terminal_metadata(
            spec,
            digest,
            status=status,
            prompt_token_count=prompt_token_count,
            generated_token_count=generated_token_count,
            error_type=error_type,
            error_message=error_message,
        ),
    }


def validate_terminal_value(
    caption: object,
    metadata: dict[str, Any] | None,
    *,
    spec: CaptionModelSpec,
    digest: str,
) -> None:
    """Validate one published/staged terminal value against the selected contract."""
    if metadata is None:
        msg = "terminal caption metadata must not be null"
        raise ValueError(msg)
    schema_version = metadata.get("caption_schema_version")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        msg = "terminal caption metadata caption_schema_version must be an integer"
        raise TypeError(msg)
    expected = {
        "caption_schema_version": CAPTION_SCHEMA_VERSION,
        "contract_digest": digest,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "prompt": DEFAULT_PROMPT,
    }
    mismatched = [name for name, value in expected.items() if metadata.get(name) != value]
    if mismatched:
        msg = f"terminal caption metadata has incompatible value(s): {', '.join(mismatched)}"
        raise ValueError(msg)
    status = metadata.get("status")
    if status not in TERMINAL_STATUSES:
        msg = f"terminal caption metadata has unsupported status {status!r}"
        raise ValueError(msg)
    _validate_token_counts(metadata)
    _validate_terminal_outcome(caption, metadata, status=str(status))


def validate_terminal_row(
    row: dict[str, Any],
    *,
    spec: CaptionModelSpec,
    digest: str,
) -> None:
    """Validate one clip-owned staged or published terminal row."""
    validate_terminal_value(
        row.get(spec.caption_field_name),
        row.get(spec.metadata_field_name),
        spec=spec,
        digest=digest,
    )


def validate_terminal_batch(batch: pa.RecordBatch, *, spec: CaptionModelSpec, digest: str) -> None:
    """Validate every caption in the publisher's bounded batches against the selected contract."""
    captions = batch.column(spec.caption_field_name).to_pylist()
    for caption, metadata in zip(captions, batch.column(spec.metadata_field_name).to_pylist(), strict=True):
        validate_terminal_value(caption, metadata, spec=spec, digest=digest)


def _validate_token_counts(metadata: dict[str, Any]) -> None:
    for count_name in ("prompt_token_count", "generated_token_count"):
        count = metadata.get(count_name)
        if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count < 0):
            msg = f"terminal caption metadata {count_name} must be a non-negative integer or null"
            raise ValueError(msg)


def _validate_terminal_outcome(caption: object, metadata: dict[str, Any], *, status: str) -> None:
    error_type = metadata.get("error_type")
    error_message = metadata.get("error_message")
    if status == "error":
        if (
            caption is not None
            or not isinstance(error_type, str)
            or not error_type
            or not isinstance(error_message, str)
            or not error_message
        ):
            msg = "error captions require a null caption and non-empty error_type/error_message"
            raise ValueError(msg)
        return
    if not isinstance(caption, str) or error_type is not None or error_message is not None:
        msg = f"{status} captions require a non-null caption and null error fields"
        raise ValueError(msg)
    generated = metadata.get("generated_token_count")
    if status == "truncated" and generated is not None and generated < MAX_OUTPUT_TOKENS:
        msg = "truncated captions with a token count must reach max output tokens"
        raise ValueError(msg)
    if status == "success" and generated is not None and generated >= MAX_OUTPUT_TOKENS:
        msg = "captions that reach max output tokens must use truncated status"
        raise ValueError(msg)
