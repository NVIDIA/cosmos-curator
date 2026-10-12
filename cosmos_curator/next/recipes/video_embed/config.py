# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict video-embed configuration; model preprocessing is a pinned contract."""

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cosmos_curator.next.core.config import apply_dotted_overrides
from cosmos_curator.next.enrichment.config import EnrichmentExecutionConfig
from cosmos_curator.next.utils.uris import normalize_location

ModelVariant = Literal["224p", "336p", "448p"]
Concurrency = Annotated[int, Field(ge=1)] | Literal["auto"]


class _Config(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class InputConfig(_Config):
    """Completed video-transcode clips, from either cutting strategy."""

    media_root: str = Field(min_length=1)
    clips_lance_uri: str = Field(default="", description="Defaults to <media_root>/lance.")

    @field_validator("media_root", "clips_lance_uri")
    @classmethod
    def _location(cls, value: str) -> str:
        return normalize_location(value)


class OutputConfig(_Config):
    """Durable temporary results, readable by every worker and replacement driver."""

    staging_root_uri: str = Field(default="", description="Defaults to <media_root>/staging/video-embed.")

    @field_validator("staging_root_uri")
    @classmethod
    def _location(cls, value: str) -> str:
        return normalize_location(value)


class ModelConfig(_Config):
    """Only the three repository-pinned Cosmos-Embed1 checkpoints are supported."""

    variant: ModelVariant = "336p"


class ExecutionConfig(EnrichmentExecutionConfig):
    """Resource and durability tuning, excluded from embedding identity."""

    # Smaller scanner batches spread clip downloads across more Ray tasks.
    lance_read_batch_size: int = Field(default=128, ge=1, description="Clip rows per Lance scanner batch.")
    media_attempts: int = Field(default=3, ge=1)
    inference_concurrency: Concurrency = "auto"
    inference_batch_size: int = Field(default=16, ge=1)


class ResolvedVideoEmbedConfig(_Config):
    """One visual embedding per clip, atomically published across all selected fragments."""

    schema_version: Literal[1]
    kind: Literal["video-embed"]
    input: InputConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    @model_validator(mode="before")
    @classmethod
    def _locations(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        result = copy.deepcopy(dict(value))
        source = result.get("input")
        if not isinstance(source, Mapping) or not isinstance(source.get("media_root"), str):
            return result
        source = dict(source)
        root = source["media_root"].rstrip("/")
        source.setdefault("clips_lance_uri", f"{root}/lance")
        result["input"] = source
        output = result.get("output", {})
        if isinstance(output, Mapping):
            output = dict(output)
            output.setdefault("staging_root_uri", f"{root}/staging/video-embed")
            result["output"] = output
        return result

    @model_validator(mode="after")
    def _separate(self) -> Self:
        locations = [self.input.clips_lance_uri, self.output.staging_root_uri, f"{self.input.media_root}/clips"]
        for index, left in enumerate(locations):
            for right in locations[index + 1 :]:
                if left == right or left.startswith(right + "/") or right.startswith(left + "/"):
                    msg = "Lance, staging, and clip locations must be separate directories"
                    raise ValueError(msg)
        return self


def resolve_config_data(data: Mapping[str, object], *, overrides: Sequence[str] = ()) -> ResolvedVideoEmbedConfig:
    """Resolve defaults and typed overrides without importing the runtime."""
    values = copy.deepcopy(dict(data))
    apply_dotted_overrides(values, overrides)
    return ResolvedVideoEmbedConfig.model_validate(values)


def resolve_config(path: str | Path, *, overrides: Sequence[str] = ()) -> ResolvedVideoEmbedConfig:
    """Read YAML or JSON through the same strict resolver used by the CLI."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = "video-embed config must be a mapping"
        raise TypeError(msg)
    return resolve_config_data(data, overrides=overrides)


def resolved_config_to_json(config: ResolvedVideoEmbedConfig) -> str:
    """Render canonical configuration without runtime credentials."""
    return json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True)


def config_schema_json() -> str:
    """Render the strict configuration schema."""
    return json.dumps(ResolvedVideoEmbedConfig.model_json_schema(), indent=2)


def config_template() -> dict[str, object]:
    """Show every default, leaving derived locations tied to the media root."""
    return resolve_config_data(
        {"schema_version": 1, "kind": "video-embed", "input": {"media_root": "s3://example-bucket/curated/videos"}}
    ).model_dump(mode="json", exclude={"input": {"clips_lance_uri"}, "output": True})


def config_template_yaml() -> str:
    """Describe model and path defaults alongside a runnable template."""
    return (
        "# Cosmos-Embed1 variants: 224p, 336p, 448p. Weights must already be staged.\n"
        "# NVDEC samples eight evenly spaced frames; the pinned processor runs on the GPU.\n"
        "# Optional input.clips_lance_uri defaults to <media_root>/lance.\n"
        "# Optional output.staging_root_uri defaults to <media_root>/staging/video-embed.\n"
        "# Concurrency 'auto' follows live resources; each inference actor reserves one GPU.\n"
        + yaml.safe_dump(config_template(), sort_keys=False)
    )


def config_template_payload() -> dict[str, object]:
    """Describe the recipe for the generic pipeline CLI."""
    return {
        "kind": "video-embed",
        "description": "Embed transcoded clips with Cosmos-Embed1 and atomically enrich their Lance rows.",
        "required_fields": [
            {"path": "schema_version", "example": 1},
            {"path": "kind", "example": "video-embed"},
            {"path": "input.media_root", "example": "s3://example-bucket/curated/videos"},
        ],
        "config": config_template(),
    }
