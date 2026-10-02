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

"""Strict configuration for the ``multimodal-embedding`` recipe."""

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from cosmos_curator.next.core.config import apply_dotted_overrides
from cosmos_curator.next.recipes.result_keys import RESULT_KEY_PATTERN
from cosmos_curator.next.utils.uris import normalize_location

_YAML_SUFFIXES = frozenset({".yaml", ".yml"})
_MODEL_CONFIG = ConfigDict(frozen=True, strict=True, extra="forbid")

SchemaVersion = Literal[1]
MultimodalEmbeddingKind = Literal["multimodal-embedding"]
ModelVariant = Literal["cosmos_embed1_224p", "cosmos_embed1_336p", "cosmos_embed1_448p"]


def _default_staging_root(episodes_lance_uri: str) -> str:
    if episodes_lance_uri.lower().startswith("s3://"):
        parent = episodes_lance_uri.rsplit("/", 1)[0]
        return f"{parent.rstrip('/')}/staging/multimodal-embedding"
    return str(Path(episodes_lance_uri).parent / "staging" / "multimodal-embedding")


class MultimodalEmbeddingInputConfig(BaseModel):
    """Canonical ``episodes.lance`` produced by ``multimodal-split``."""

    model_config = _MODEL_CONFIG

    episodes_lance_uri: str = Field(min_length=1)

    @field_validator("episodes_lance_uri")
    @classmethod
    def _normalize_episodes_uri(cls, location: str) -> str:
        return normalize_location(location)


class VideoEmbeddingConfig(BaseModel):
    """The one complete-camera MP4 represented by every result row."""

    model_config = _MODEL_CONFIG

    camera_key: str = Field(min_length=1)

    @field_validator("camera_key")
    @classmethod
    def _validate_camera_key(cls, value: str) -> str:
        if value != value.strip():
            msg = "video.camera_key cannot have surrounding whitespace"
            raise ValueError(msg)
        return value


class MultimodalEmbeddingModelConfig(BaseModel):
    """Pinned video model choice; arbitrary checkpoints are not supported."""

    model_config = _MODEL_CONFIG

    variant: ModelVariant = "cosmos_embed1_336p"


class MultimodalEmbeddingOutputConfig(BaseModel):
    """Result namespace and durable staging location."""

    model_config = _MODEL_CONFIG

    result_key: str = Field(default="default", pattern=RESULT_KEY_PATTERN)
    staging_root_uri: str = ""

    @field_validator("staging_root_uri")
    @classmethod
    def _normalize_staging_uri(cls, location: str) -> str:
        return normalize_location(location)


class EmbeddingSelectionConfig(BaseModel):
    """Eligibility policy, deliberately excluded from the vector contract."""

    model_config = _MODEL_CONFIG

    max_episode_duration_s: float = Field(default=60.0, gt=0.0, allow_inf_nan=False)
    retry_errors: bool = False


class MultimodalEmbeddingExecutionConfig(BaseModel):
    """Runtime tuning, deliberately excluded from the vector contract."""

    model_config = _MODEL_CONFIG

    storage_profile: str = Field(default="default", min_length=1)
    media_concurrency: int | Literal["auto"] = "auto"
    decode_concurrency: int | Literal["auto"] = "auto"
    inference_concurrency: int | Literal["auto"] = "auto"
    inference_batch_size: int = Field(default=4, ge=1)
    inference_cpus: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    inference_gpus: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    decode_cpus: int = Field(default=1, ge=1)
    decode_memory_gib: float = Field(default=2.0, gt=0.0, allow_inf_nan=False)
    media_attempts: int = Field(default=3, ge=1)
    lance_read_batch_size: int = Field(default=256, ge=1)
    parquet_rows_per_file: int = Field(default=4_096, ge=1)
    commit_attempts: int = Field(default=5, ge=1)
    progress: bool = False

    @field_validator("media_concurrency", "decode_concurrency", "inference_concurrency")
    @classmethod
    def _validate_concurrency(cls, value: int | Literal["auto"], info: ValidationInfo) -> int | Literal["auto"]:
        if isinstance(value, int) and value < 1:
            msg = f"execution.{info.field_name} must be 'auto' or a positive integer, got {value}"
            raise ValueError(msg)
        return value


class ResolvedMultimodalEmbeddingConfig(BaseModel):
    """Canonical v1 configuration for video-only AV embedding."""

    model_config = _MODEL_CONFIG

    schema_version: SchemaVersion
    kind: MultimodalEmbeddingKind
    input: MultimodalEmbeddingInputConfig
    video: VideoEmbeddingConfig
    model: MultimodalEmbeddingModelConfig = Field(default_factory=MultimodalEmbeddingModelConfig)
    output: MultimodalEmbeddingOutputConfig = Field(default_factory=MultimodalEmbeddingOutputConfig)
    selection: EmbeddingSelectionConfig = Field(default_factory=EmbeddingSelectionConfig)
    execution: MultimodalEmbeddingExecutionConfig = Field(default_factory=MultimodalEmbeddingExecutionConfig)

    @model_validator(mode="before")
    @classmethod
    def _derive_staging_root(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        resolved = copy.deepcopy(dict(value))
        input_mapping = resolved.get("input")
        if not isinstance(input_mapping, Mapping):
            return resolved
        episodes_uri = input_mapping.get("episodes_lance_uri")
        if not isinstance(episodes_uri, str):
            return resolved
        normalized_uri = normalize_location(episodes_uri)
        output_mapping = resolved.get("output", {})
        if not isinstance(output_mapping, Mapping):
            return resolved
        output = dict(output_mapping)
        output.setdefault("staging_root_uri", _default_staging_root(normalized_uri))
        resolved["output"] = output
        return resolved


_TEMPLATE_BASE: dict[str, Any] = {
    "schema_version": 1,
    "kind": "multimodal-embedding",
    "input": {"episodes_lance_uri": "s3://example-bucket/curated/multimodal-split/episodes.lance"},
    "video": {"camera_key": "front_wide"},
    "model": {"variant": "cosmos_embed1_336p"},
    "output": {"result_key": "default"},
    "selection": {"max_episode_duration_s": 60.0},
    "execution": {"storage_profile": "default"},
}
_TEMPLATE = ResolvedMultimodalEmbeddingConfig.model_validate(_TEMPLATE_BASE).model_dump(
    mode="json", exclude_none=True, exclude={"output": {"staging_root_uri"}}
)
_TEMPLATE_PREAMBLE = """\
# Embed one explicitly selected complete camera MP4 per canonical episode.
# staging_root_uri defaults beside episodes.lance under staging/multimodal-embedding.
# max_episode_duration_s decides which rows are eligible; it does not change eligible vector values.
# A new maximum scopes staging; completed rows are reconsidered in the planned rewrite phase.
# Stage model assets under /config/models on the driver and all nodes that may run recipe actors.
# Cosmos-Embed1 CPU preparation needs its checkpoint processor files too.
# Cosmos-Embed1 variants: cosmos_embed1_224p, cosmos_embed1_336p (default), cosmos_embed1_448p.
# Eight endpoint-inclusive frames, with the pinned processor defaults unchanged.
# decode_memory_gib is a scheduling reservation, not a hard memory limit.
# media_concurrency and decode_concurrency accept auto or a positive actor limit.
# inference_batch_size defaults to 4; increase it only when the target GPU has sufficient memory.
"""


def load_config_data(config_path: str | Path) -> dict[str, Any]:
    """Load one JSON or YAML config mapping."""
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


def resolve_config(config_path: str | Path, *, overrides: Sequence[str] = ()) -> ResolvedMultimodalEmbeddingConfig:
    """Load and resolve one configuration file."""
    return resolve_config_data(load_config_data(config_path), overrides=overrides)


def resolve_config_data(
    config_data: Mapping[str, Any], *, overrides: Sequence[str] = ()
) -> ResolvedMultimodalEmbeddingConfig:
    """Resolve one in-memory configuration mapping."""
    data = copy.deepcopy(dict(config_data))
    apply_dotted_overrides(data, overrides)
    return ResolvedMultimodalEmbeddingConfig.model_validate(data)


def resolved_config_to_json(config: ResolvedMultimodalEmbeddingConfig, *, indent: int = 2) -> str:
    """Render canonical resolved configuration JSON."""
    return json.dumps(config.model_dump(mode="json"), indent=indent, ensure_ascii=False) + "\n"


def config_schema_json(*, indent: int = 2) -> str:
    """Render the public JSON schema."""
    return json.dumps(ResolvedMultimodalEmbeddingConfig.model_json_schema(), indent=indent, ensure_ascii=False) + "\n"


def config_template() -> dict[str, Any]:
    """Return an editable complete configuration template."""
    return copy.deepcopy(_TEMPLATE)


def config_template_yaml() -> str:
    """Render the annotated YAML template."""
    return _TEMPLATE_PREAMBLE + yaml.safe_dump(config_template(), sort_keys=False)


def config_template_payload() -> dict[str, Any]:
    """Return structured template metadata for the generic CLI."""
    return {
        "kind": "multimodal-embedding",
        "description": "Embed one explicitly selected complete camera MP4 per canonical episode.",
        "required_fields": [
            {"path": "schema_version", "example": 1},
            {"path": "kind", "example": "multimodal-embedding"},
            {
                "path": "input.episodes_lance_uri",
                "example": "s3://example-bucket/curated/multimodal-split/episodes.lance",
            },
            {"path": "video.camera_key", "example": "front_wide"},
        ],
        "config": config_template(),
    }
