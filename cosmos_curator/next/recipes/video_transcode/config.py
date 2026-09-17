# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict configuration for encoding pending fragments of an append-only video-cut table."""

import copy
import json
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cosmos_curator.next.core.config import apply_dotted_overrides
from cosmos_curator.next.utils.identity import canonical_digest
from cosmos_curator.next.utils.uris import normalize_location


class _Config(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class InputConfig(_Config):
    """Latest span table, enriched one full fragment at a time."""

    media_root: str = Field(min_length=1)
    spans_lance_uri: str = Field(default="", description="Defaults to <media_root>/lance.")

    @field_validator("media_root", "spans_lance_uri")
    @classmethod
    def _location(cls, value: str) -> str:
        return normalize_location(value)


class OutputConfig(_Config):
    """Temporary checkpoint storage, derived from the input media root by default."""

    staging_root_uri: str = Field(default="", description="Defaults to <input.media_root>/staging/video-transcode.")

    @field_validator("staging_root_uri")
    @classmethod
    def _location(cls, value: str) -> str:
        return normalize_location(value)


class TranscodeConfig(_Config):
    """Result-defining encoding choices, independent of cutting and scheduling."""

    video_encoder: Literal["libopenh264", "libx264"] = "libopenh264"
    video_bitrate: str = Field(default="4M", pattern=r"^[1-9][0-9]*(?:\.[0-9]+)?[KkMm]$")
    audio_mode: Literal["copy", "aac"] = "copy"

    @field_validator("video_bitrate", mode="before")
    @classmethod
    def _bitrate(cls, value: object) -> object:
        if not isinstance(value, str) or not value[:-1]:
            return value
        try:
            number = Decimal(value[:-1])
        except ArithmeticError:
            return value
        return f"{format(number.normalize(), 'f')}{value[-1].upper()}"

    @property
    def contract_id(self) -> str:
        """Identify compatible clip bytes and metadata."""
        return canonical_digest({"behavior_version": 1, "encoding": self.model_dump(mode="json")})


class ExecutionConfig(_Config):
    """Resource and checkpoint tuning excluded from clip identity."""

    storage_profile: str = Field(default="default", min_length=1)
    transcode_cpus: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    transcode_memory_gib: float = Field(
        default=4.0,
        gt=0,
        allow_inf_nan=False,
        description="Memory reserved per transcode task in GiB, including the Python worker and FFmpeg subprocess.",
    )
    ffmpeg_batch_size: int = Field(default=16, ge=1)
    encoder_threads: int = Field(default=1, ge=1)
    transcode_timeout_s: int = Field(default=600, ge=1)
    probe_timeout_s: int = Field(default=120, ge=1)
    storage_attempts: int = Field(default=3, ge=1)
    transcode_attempts: int = Field(default=3, ge=1)
    probe_attempts: int = Field(default=3, ge=1)
    spans_per_checkpoint: int = Field(default=1_000, ge=1)
    source_partitions: int = Field(
        default=256, ge=1, description="Target read block count for the source-grouping sort."
    )
    lance_read_batch_size: int = Field(default=1_000, ge=1)
    commit_attempts: int = Field(default=3, ge=1)
    progress: bool = False


class ResolvedVideoTranscodeConfig(_Config):
    """One encoding contract applied to pending fragments of a growing cut table."""

    schema_version: Literal[1]
    kind: Literal["video-transcode"]
    input: InputConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    transcode: TranscodeConfig = Field(default_factory=TranscodeConfig)
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
        source.setdefault("spans_lance_uri", f"{root}/lance")
        result["input"] = source
        output = result.get("output", {})
        if isinstance(output, Mapping):
            output = dict(output)
            output.setdefault("staging_root_uri", f"{root}/staging/video-transcode")
            result["output"] = output
        return result

    @model_validator(mode="after")
    def _separate(self) -> Self:
        locations = [self.input.spans_lance_uri, self.output.staging_root_uri, f"{self.input.media_root}/clips"]
        for index, left in enumerate(locations):
            for right in locations[index + 1 :]:
                if left == right or left.startswith(right + "/") or right.startswith(left + "/"):
                    msg = "Lance, staging, and clip locations must be separate directories"
                    raise ValueError(msg)
        return self


def resolve_config_data(data: Mapping[str, object], *, overrides: Sequence[str] = ()) -> ResolvedVideoTranscodeConfig:
    """Resolve defaults and typed dotted overrides without opening storage."""
    values = copy.deepcopy(dict(data))
    apply_dotted_overrides(values, overrides)
    return ResolvedVideoTranscodeConfig.model_validate(values)


def resolve_config(path: str | Path, *, overrides: Sequence[str] = ()) -> ResolvedVideoTranscodeConfig:
    """Load a YAML or JSON config."""
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        msg = "video-transcode config must be a mapping"
        raise TypeError(msg)
    return resolve_config_data(data, overrides=overrides)


def resolved_config_to_json(config: ResolvedVideoTranscodeConfig) -> str:
    """Render the resolved execution contract."""
    return json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True)


def config_schema_json() -> str:
    """Render the strict JSON schema."""
    return json.dumps(ResolvedVideoTranscodeConfig.model_json_schema(), indent=2)


def config_template() -> dict[str, object]:
    """Show defaults while leaving derived locations editable through the media root."""
    return resolve_config_data(
        {"schema_version": 1, "kind": "video-transcode", "input": {"media_root": "s3://example-bucket/curated/videos"}}
    ).model_dump(mode="json", exclude={"input": {"spans_lance_uri"}, "output": True})


def config_template_yaml() -> str:
    """Render defaults with optional path overrides and execution guidance."""
    preamble = (
        "# Defaults are shown. Unchanged settings may be removed.\n"
        "# Optional paths are derived from input.media_root when omitted:\n"
        "#   input.spans_lance_uri: <input.media_root>/lance\n"
        "#   output.staging_root_uri: <input.media_root>/staging/video-transcode\n"
        "# Execution settings:\n"
        "#   transcode_cpus: Ray CPU reservation per task, not an FFmpeg thread limit.\n"
        "#   ffmpeg_batch_size: Clips encoded per FFmpeg invocation.\n"
        "#   encoder_threads: Threads per output encoder.\n"
        "#   transcode_timeout_s: Timeout per FFmpeg invocation, including a batch of clips.\n"
        "#   source_partitions: Target metadata block count for source grouping and sorting.\n"
        "#   lance_read_batch_size: Span rows per Lance scanner batch.\n"
    )
    return preamble + yaml.safe_dump(config_template(), sort_keys=False)


def config_template_payload() -> dict[str, object]:
    """Describe the recipe for the generic CLI."""
    return {
        "kind": "video-transcode",
        "description": "Encode video-cut spans with Ray checkpoints and atomically enrich their Lance rows.",
        "required_fields": [
            {"path": "schema_version", "example": 1},
            {"path": "kind", "example": "video-transcode"},
            {"path": "input.media_root", "example": "s3://example-bucket/curated/videos"},
        ],
        "config": config_template(),
    }
