# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict configuration for the ``multimodal-caption`` recipe."""

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import unquote, urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cosmos_curator.next.core.config import apply_dotted_overrides
from cosmos_curator.next.recipes.video_split.uris import join_s3_uri, normalize_s3_uri

_YAML_SUFFIXES = frozenset({".yaml", ".yml"})
_MODEL_CONFIG = ConfigDict(frozen=True, strict=True, extra="forbid")

SchemaVersion = Literal[1]
MultimodalCaptionKind = Literal["multimodal-caption"]
ContextVariant = Literal["video_only"]
ModelVariant = Literal["qwen3_8_27b_fp8", "qwen3_8_27b", "cosmos3_nano"]


def _normalize_location(location: str, *, label: str) -> str:
    if not location or location != location.strip():
        msg = f"{label} must be non-empty and cannot have surrounding whitespace"
        raise ValueError(msg)
    parsed = urlsplit(location)
    if parsed.scheme.lower() == "s3":
        return normalize_s3_uri(location, strip_trailing_slash=True)
    if parsed.scheme.lower() == "file":
        if parsed.netloc.lower() not in {"", "localhost"} or not parsed.path:
            msg = f"Unsupported local file URI for {label}: {location!r}"
            raise ValueError(msg)
        return str(Path(unquote(parsed.path)).expanduser().resolve())
    if parsed.scheme:
        msg = f"{label} must be a local path or s3:// URI, got {location!r}"
        raise ValueError(msg)
    return str(Path(location).expanduser().resolve())


def _default_staging_root(episodes_lance_uri: str) -> str:
    if episodes_lance_uri.lower().startswith("s3://"):
        parent = episodes_lance_uri.rsplit("/", 1)[0]
        return join_s3_uri(parent, "staging", "multimodal-caption")
    return str(Path(episodes_lance_uri).parent / "staging" / "multimodal-caption")


class MultimodalCaptionInputConfig(BaseModel):
    """Canonical episodes table produced by ``multimodal-split``."""

    model_config = _MODEL_CONFIG

    episodes_lance_uri: str = Field(min_length=1)

    @field_validator("episodes_lance_uri")
    @classmethod
    def _normalize_episodes_uri(cls, location: str) -> str:
        return _normalize_location(location, label="input.episodes_lance_uri")


class MultimodalCaptionContextConfig(BaseModel):
    """The explicit visual treatment selected for each episode."""

    model_config = _MODEL_CONFIG

    variant: ContextVariant
    camera_key: str = Field(min_length=1)

    @field_validator("camera_key")
    @classmethod
    def _validate_camera_key(cls, value: str) -> str:
        if value != value.strip():
            msg = "context.camera_key cannot have surrounding whitespace"
            raise ValueError(msg)
        return value


class MultimodalCaptionModelConfig(BaseModel):
    """Pinned caption model choice."""

    model_config = _MODEL_CONFIG

    variant: ModelVariant = "qwen3_8_27b_fp8"


class MultimodalCaptionOutputConfig(BaseModel):
    """Durable recovery workspace location."""

    model_config = _MODEL_CONFIG

    staging_root_uri: str = ""

    @field_validator("staging_root_uri")
    @classmethod
    def _normalize_staging_uri(cls, location: str) -> str:
        return _normalize_location(location, label="output.staging_root_uri")


class MultimodalCaptionExecutionConfig(BaseModel):
    """Execution-only tuning excluded from caption identity."""

    model_config = _MODEL_CONFIG

    storage_profile: str = Field(default="default", min_length=1)
    inference_concurrency: int | Literal["auto"] = "auto"
    inference_batch_size: int = Field(default=32, ge=1)
    max_concurrent_batches: int = Field(default=8, ge=1)
    tensor_parallel_size: int = Field(default=1, ge=1)
    pipeline_parallel_size: int = Field(default=1, ge=1)
    media_concurrency: int | Literal["auto"] = "auto"
    media_batch_size: int = Field(default=4, ge=1)
    media_cpus: float = Field(default=0.25, gt=0.0, allow_inf_nan=False)
    media_attempts: int = Field(default=3, ge=1)
    lance_read_batch_size: int = Field(default=256, ge=1)
    parquet_rows_per_file: int = Field(default=4_096, ge=1)
    commit_attempts: int = Field(default=5, ge=1)
    progress: bool = False

    @field_validator("inference_concurrency", "media_concurrency")
    @classmethod
    def _validate_concurrency(cls, value: int | Literal["auto"]) -> int | Literal["auto"]:
        if isinstance(value, int) and value < 1:
            msg = f"concurrency must be 'auto' or a positive integer, got {value}"
            raise ValueError(msg)
        return value


class ResolvedMultimodalCaptionConfig(BaseModel):
    """Canonical v1 ``multimodal-caption`` configuration."""

    model_config = _MODEL_CONFIG

    schema_version: SchemaVersion
    kind: MultimodalCaptionKind
    input: MultimodalCaptionInputConfig
    context: MultimodalCaptionContextConfig
    model: MultimodalCaptionModelConfig = Field(default_factory=MultimodalCaptionModelConfig)
    output: MultimodalCaptionOutputConfig = Field(default_factory=MultimodalCaptionOutputConfig)
    execution: MultimodalCaptionExecutionConfig = Field(default_factory=MultimodalCaptionExecutionConfig)

    @model_validator(mode="before")
    @classmethod
    def _derive_staging_root(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        resolved = copy.deepcopy(dict(value))
        raw_input = resolved.get("input")
        if not isinstance(raw_input, Mapping):
            return resolved
        episodes_uri = raw_input.get("episodes_lance_uri")
        if not isinstance(episodes_uri, str):
            return resolved
        normalized_uri = _normalize_location(episodes_uri, label="input.episodes_lance_uri")
        raw_output = resolved.get("output", {})
        if not isinstance(raw_output, Mapping):
            return resolved
        output = dict(raw_output)
        output.setdefault("staging_root_uri", _default_staging_root(normalized_uri))
        resolved["output"] = output
        return resolved


_TEMPLATE_BASE: dict[str, Any] = {
    "schema_version": 1,
    "kind": "multimodal-caption",
    "input": {"episodes_lance_uri": "s3://example-bucket/curated/multimodal-split/episodes.lance"},
    "context": {"variant": "video_only", "camera_key": "front_wide"},
    "model": {"variant": "qwen3_8_27b_fp8"},
    "output": {},
    "execution": {"storage_profile": "default"},
}
_TEMPLATE = ResolvedMultimodalCaptionConfig.model_validate(_TEMPLATE_BASE).model_dump(mode="json", exclude_none=True)
_TEMPLATE_PREAMBLE = """\
# All supported settings and defaults are shown. Unchanged settings may be removed.
# staging_root_uri defaults beside episodes.lance under staging/multimodal-caption.
# camera_key is required; front_wide is only an example value, not a default camera policy.
# Model weights must already exist under /config/models on every inference worker.
"""


def load_config_data(config_path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML config mapping."""
    path = Path(config_path)
    if not path.exists():
        msg = f"Pipeline config file not found: {path}"
        raise FileNotFoundError(msg)
    try:
        with path.open(encoding="utf-8") as config_file:
            loaded = yaml.safe_load(config_file) if path.suffix.lower() in _YAML_SUFFIXES else json.load(config_file)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        msg = f"Failed to parse config {path}: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(loaded, dict):
        msg = f"Config file must contain a mapping at the top level, got {type(loaded).__name__}: {path}"
        raise TypeError(msg)
    return cast("dict[str, Any]", loaded)


def resolve_config(config_path: str | Path, *, overrides: Sequence[str] = ()) -> ResolvedMultimodalCaptionConfig:
    """Load and resolve one configuration file."""
    return resolve_config_data(load_config_data(config_path), overrides=overrides)


def resolve_config_data(
    raw_data: Mapping[str, Any], *, overrides: Sequence[str] = ()
) -> ResolvedMultimodalCaptionConfig:
    """Resolve one in-memory configuration mapping."""
    data = copy.deepcopy(dict(raw_data))
    apply_dotted_overrides(data, overrides)
    return ResolvedMultimodalCaptionConfig.model_validate(data)


def resolved_config_to_json(config: ResolvedMultimodalCaptionConfig, *, indent: int = 2) -> str:
    """Render a canonical resolved configuration."""
    return json.dumps(config.model_dump(mode="json"), indent=indent, ensure_ascii=False) + "\n"


def config_schema_json(*, indent: int = 2) -> str:
    """Render the public JSON schema."""
    return json.dumps(ResolvedMultimodalCaptionConfig.model_json_schema(), indent=indent, ensure_ascii=False) + "\n"


def config_template() -> dict[str, Any]:
    """Return a complete editable configuration template."""
    return copy.deepcopy(_TEMPLATE)


def config_template_yaml() -> str:
    """Render the annotated YAML template."""
    return _TEMPLATE_PREAMBLE + yaml.safe_dump(config_template(), sort_keys=False)


def config_template_payload() -> dict[str, Any]:
    """Return structured template metadata."""
    return {
        "kind": "multimodal-caption",
        "description": "Caption one explicitly selected complete camera MP4 per canonical episode.",
        "required_fields": [
            {"path": "schema_version", "example": 1},
            {"path": "kind", "example": "multimodal-caption"},
            {
                "path": "input.episodes_lance_uri",
                "example": "s3://example-bucket/curated/multimodal-split/episodes.lance",
            },
            {"path": "context.variant", "example": "video_only"},
            {"path": "context.camera_key", "example": "front_wide"},
        ],
        "config": config_template(),
    }
