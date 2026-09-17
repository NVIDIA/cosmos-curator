# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict configuration and deterministic resolution for ``video-caption``."""

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import unquote, urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cosmos_curator.next.core.config import apply_dotted_overrides
from cosmos_curator.next.utils.uris import normalize_location

_YAML_SUFFIXES = frozenset({".yaml", ".yml"})
_MODEL_CONFIG = ConfigDict(frozen=True, strict=True, extra="forbid")

SchemaVersion = Literal[1]
VideoCaptionKind = Literal["video-caption"]
ModelVariant = Literal["qwen3_8_27b_fp8", "qwen3_8_27b"]


def _normalize_location(location: str, *, label: str) -> str:
    """Normalize one local-or-S3 directory without changing an S3 object key."""
    if not location or location != location.strip():
        msg = f"{label} must be non-empty and cannot have surrounding whitespace"
        raise ValueError(msg)
    parsed = urlsplit(location)
    if parsed.scheme.lower() == "s3":
        return normalize_location(location)
    if parsed.scheme.lower() == "file":
        if parsed.netloc.lower() not in {"", "localhost"} or not parsed.path:
            msg = f"Unsupported local file URI for {label}: {location!r}"
            raise ValueError(msg)
        return str(Path(unquote(parsed.path)).expanduser().resolve())
    if parsed.scheme:
        msg = f"{label} must be a local path or s3:// URI, got {location!r}"
        raise ValueError(msg)
    return str(Path(location).expanduser().resolve())


def _join_location(root: str, *parts: str) -> str:
    if root.lower().startswith("s3://"):
        return "/".join((root.rstrip("/"), *parts))
    return str(Path(root).joinpath(*parts))


class VideoCaptionInputConfig(BaseModel):
    """The canonical clip table produced by ``video-transcode``."""

    model_config = _MODEL_CONFIG

    media_root: str = Field(min_length=1, examples=["s3://example-bucket/curated/video-transcode"])
    clips_lance_uri: str = Field(default="", description="Defaults to <media_root>/lance.")

    @field_validator("media_root")
    @classmethod
    def _normalize_media_root(cls, location: str) -> str:
        return _normalize_location(location, label="input.media_root")

    @field_validator("clips_lance_uri")
    @classmethod
    def _normalize_lance_uri(cls, location: str) -> str:
        return _normalize_location(location, label="input.clips_lance_uri")


class VideoCaptionModelConfig(BaseModel):
    """Pinned model choice; arbitrary model IDs and weights paths are forbidden."""

    model_config = _MODEL_CONFIG

    variant: ModelVariant = "qwen3_8_27b_fp8"


class VideoCaptionOutputConfig(BaseModel):
    """Durable inference workspace, derived from the media root by default."""

    model_config = _MODEL_CONFIG

    staging_root_uri: str = Field(default="", description="Defaults to <media_root>/staging/video-caption.")

    @field_validator("staging_root_uri")
    @classmethod
    def _normalize_staging_uri(cls, location: str) -> str:
        return _normalize_location(location, label="output.staging_root_uri")


class VideoCaptionExecutionConfig(BaseModel):
    """Execution-only tuning excluded from caption identity."""

    model_config = _MODEL_CONFIG

    storage_profile: str = Field(default="default", min_length=1)
    inference_concurrency: int | Literal["auto"] = Field(
        default="auto",
        description="Maximum vLLM replicas, or auto to let Ray scale with pending work and live GPU resources.",
    )
    inference_batch_size: int = Field(
        default=32,
        ge=1,
        description="Clip rows per Ray Data vLLM batch; 32 matches the benchmarked legacy captioning baseline.",
    )
    max_concurrent_batches: int = Field(
        default=8,
        ge=1,
        description="Concurrent Ray batch calls per vLLM replica; 8 gives Ray's default in-flight queue depth of 16.",
    )
    tensor_parallel_size: int = Field(default=1, ge=1)
    pipeline_parallel_size: int = Field(default=1, ge=1)
    media_concurrency: int | Literal["auto"] = Field(
        default="auto",
        description=(
            "Maximum concurrent media-fetch tasks, or auto to let Ray scale within each node's curator_io capacity."
        ),
    )
    media_batch_size: int = Field(default=4, ge=1)
    media_cpus: float = Field(default=0.25, gt=0.0, allow_inf_nan=False)
    media_attempts: int = Field(default=3, ge=1)
    lance_read_batch_size: int = Field(
        default=256,
        ge=1,
        description="Rows per Lance scan batch and strict pre-fetch block, exposing media work across the cluster.",
    )
    parquet_rows_per_file: int = Field(
        default=4_096,
        ge=1,
        description="Rows per durable result/checkpoint group.",
    )
    commit_attempts: int = Field(default=5, ge=1)
    progress: bool = False

    @field_validator("inference_concurrency")
    @classmethod
    def _validate_inference_concurrency(cls, value: int | Literal["auto"]) -> int | Literal["auto"]:
        if isinstance(value, int) and value < 1:
            msg = f"execution.inference_concurrency must be 'auto' or a positive integer, got {value}"
            raise ValueError(msg)
        return value

    @field_validator("media_concurrency")
    @classmethod
    def _validate_media_concurrency(cls, value: int | Literal["auto"]) -> int | Literal["auto"]:
        if isinstance(value, int) and value < 1:
            msg = f"execution.media_concurrency must be 'auto' or a positive integer, got {value}"
            raise ValueError(msg)
        return value


class ResolvedVideoCaptionConfig(BaseModel):
    """Canonical v1 execution config for Curator Next ``video-caption``."""

    model_config = _MODEL_CONFIG

    schema_version: SchemaVersion
    kind: VideoCaptionKind
    input: VideoCaptionInputConfig
    model: VideoCaptionModelConfig
    output: VideoCaptionOutputConfig = Field(default_factory=VideoCaptionOutputConfig)
    execution: VideoCaptionExecutionConfig = Field(default_factory=VideoCaptionExecutionConfig)

    @model_validator(mode="before")
    @classmethod
    def _derive_locations(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        resolved = copy.deepcopy(dict(value))
        raw_input = resolved.get("input")
        if not isinstance(raw_input, Mapping):
            return resolved
        input_config = dict(raw_input)
        media_root = input_config.get("media_root")
        if not isinstance(media_root, str):
            return resolved
        input_config.setdefault("clips_lance_uri", _join_location(media_root, "lance"))
        resolved["input"] = input_config
        raw_output = resolved.get("output", {})
        if not isinstance(raw_output, Mapping):
            return resolved
        output_config = dict(raw_output)
        output_config.setdefault("staging_root_uri", _join_location(media_root, "staging", "video-caption"))
        resolved["output"] = output_config
        return resolved


_TEMPLATE_BASE: dict[str, Any] = {
    "schema_version": 1,
    "kind": "video-caption",
    "input": {"media_root": "s3://example-bucket/curated/video-transcode"},
    "model": {"variant": "qwen3_8_27b_fp8"},
    "output": {},
    "execution": {"storage_profile": "default"},
}
_TEMPLATE = ResolvedVideoCaptionConfig.model_validate(_TEMPLATE_BASE).model_dump(mode="json", exclude_none=True)
_TEMPLATE_PREAMBLE = """\
# All supported settings and their defaults are shown. Unchanged settings may be removed.
# clips_lance_uri and staging_root_uri are derived from input.media_root when omitted.
# Model weights must already exist under /config/models on every inference worker.
# inference_concurrency=auto lets Ray place one TP=1 replica on each live GPU.
"""


def load_config_data(config_path: str | Path) -> dict[str, Any]:
    """Load a JSON/YAML config as a top-level mapping."""
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


def resolve_config(
    config_path: str | Path,
    *,
    overrides: Sequence[str] = (),
) -> ResolvedVideoCaptionConfig:
    """Load, override, and validate one config file."""
    return resolve_config_data(load_config_data(config_path), overrides=overrides)


def resolve_config_data(
    raw_data: Mapping[str, Any],
    *,
    overrides: Sequence[str] = (),
) -> ResolvedVideoCaptionConfig:
    """Apply dotted YAML-valued overrides and return canonical typed config."""
    data = copy.deepcopy(dict(raw_data))
    apply_dotted_overrides(data, overrides)
    return ResolvedVideoCaptionConfig.model_validate(data)


def resolved_config_to_json(config: ResolvedVideoCaptionConfig, *, indent: int = 2) -> str:
    """Render canonical resolved JSON."""
    return json.dumps(config.model_dump(mode="json"), indent=indent, ensure_ascii=False) + "\n"


def config_schema_json(*, indent: int = 2) -> str:
    """Return JSON Schema for v1 configs."""
    return json.dumps(ResolvedVideoCaptionConfig.model_json_schema(), indent=indent, ensure_ascii=False) + "\n"


def config_template() -> dict[str, Any]:
    """Return a complete editable config template with every default."""
    return copy.deepcopy(_TEMPLATE)


def config_template_yaml() -> str:
    """Render the complete template as annotated YAML."""
    return _TEMPLATE_PREAMBLE + yaml.safe_dump(config_template(), sort_keys=False)


def config_template_payload() -> dict[str, Any]:
    """Return structured template metadata for agents."""
    return {
        "kind": "video-caption",
        "description": "Caption complete video-transcode clips and atomically enrich their canonical Lance rows.",
        "required_fields": [
            {"path": "schema_version", "example": 1},
            {"path": "kind", "example": "video-caption"},
            {"path": "input.media_root", "example": "s3://example-bucket/curated/video-transcode"},
            {"path": "model.variant", "example": "qwen3_8_27b_fp8"},
        ],
        "config": config_template(),
    }
