# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light config for fixed-stride source video cutting."""

import copy
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cosmos_curator.next.core.config import apply_dotted_overrides
from cosmos_curator.next.media.spans import seconds_to_nanoseconds
from cosmos_curator.next.utils.identity import canonical_digest

# Match S3Prefix's bucket policy, including underscores for S3-compatible stores.
_BUCKET = re.compile(r"^[a-z0-9][a-z0-9_.-]{1,61}[a-z0-9]$")


def normalize_s3_uri(value: str) -> str:
    """Preserve literal S3 keys, including percent escapes and repeated slashes."""
    parsed = urlsplit(value)
    if (
        value != value.strip()
        or any(character in value for character in "\r\n\t?#")
        or parsed.scheme.lower() != "s3"
        or not _BUCKET.fullmatch(parsed.netloc)
        or parsed.query
        or parsed.fragment
    ):
        msg = f"Expected an undecorated S3 URI, got {value!r}"
        raise ValueError(msg)
    return f"s3://{parsed.netloc}/{parsed.path.removeprefix('/')}"


class _Config(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class InputConfig(_Config):
    """Exactly one explicit immutable source set or recursive S3 root."""

    uris: tuple[str, ...] | None = Field(default=None, min_length=1, description="Explicit S3 MP4 object URIs.")
    root_uri: str | None = Field(
        default=None, min_length=1, description="S3 prefix whose MP4 descendants are discovered recursively."
    )

    @field_validator("uris", mode="before")
    @classmethod
    def _tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("uris")
    @classmethod
    def _sources(cls, values: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if values is None:
            return None
        normalized = {normalize_s3_uri(value) for value in values}
        if any(not urlsplit(value).path.lower().endswith(".mp4") for value in normalized):
            msg = "video-cut inputs must be S3 MP4 object URIs"
            raise ValueError(msg)
        return tuple(sorted(normalized))

    @field_validator("root_uri")
    @classmethod
    def _root(cls, value: str | None) -> str | None:
        return None if value is None else normalize_s3_uri(value).rstrip("/")

    @model_validator(mode="after")
    def _exactly_one_input(self) -> Self:
        if (self.uris is None) == (self.root_uri is None):
            msg = "input must set exactly one of 'uris' or 'root_uri'"
            raise ValueError(msg)
        return self


class FixedStrideConfig(_Config):
    """Result-defining cutting settings; timestamps use integer nanoseconds."""

    method: Literal["fixed-stride"] = "fixed-stride"
    duration_s: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    stride_s: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    min_duration_s: float = Field(default=2.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _intervals(self) -> Self:
        values = [seconds_to_nanoseconds(value) for value in (self.duration_s, self.stride_s, self.min_duration_s)]
        if min(values) < 1 or self.min_duration_s > self.duration_s:
            msg = "Cutting intervals must be at least 1 ns and min_duration_s must not exceed duration_s"
            raise ValueError(msg)
        return self

    @property
    def cutting_contract(self) -> dict[str, object]:
        """Describe output semantics independently of execution and destinations."""
        return {
            # Video-cut owns this version. Bump when implementation changes alter
            # source plans for identical inputs and settings, including changes to
            # shared rounding, interval boundaries, tail retention, or duration
            # interpretation. Execution tuning and behavior-preserving refactors
            # do not require a bump.
            "behavior_version": 1,
            "method": self.method,
            "stream_index": 0,
            "duration_ns": seconds_to_nanoseconds(self.duration_s),
            "stride_ns": seconds_to_nanoseconds(self.stride_s),
            "min_duration_ns": seconds_to_nanoseconds(self.min_duration_s),
        }

    @property
    def contract_id(self) -> str:
        """Identify compatible cutting work."""
        return canonical_digest(self.cutting_contract)


class OutputConfig(_Config):
    """Canonical table and temporary recovery workspace beneath a common root."""

    media_root: str = Field(min_length=1)
    spans_lance_uri: str = Field(default="", min_length=1, description="Defaults to <media_root>/lance.")
    staging_root_uri: str = Field(default="", min_length=1, description="Defaults to <media_root>/staging/video-cut.")

    @model_validator(mode="before")
    @classmethod
    def _derive_locations(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        resolved = dict(value)
        media_root = resolved.get("media_root")
        if isinstance(media_root, str):
            root = media_root.rstrip("/")
            resolved.setdefault("spans_lance_uri", f"{root}/lance")
            resolved.setdefault("staging_root_uri", f"{root}/staging/video-cut")
        return resolved

    @field_validator("media_root", "spans_lance_uri", "staging_root_uri")
    @classmethod
    def _location(cls, value: str) -> str:
        if value != value.strip():
            msg = "Output locations cannot contain surrounding whitespace"
            raise ValueError(msg)
        return normalize_s3_uri(value).rstrip("/") if "://" in value else str(Path(value).expanduser().resolve())

    @model_validator(mode="after")
    def _separate(self) -> Self:
        a, b = self.spans_lance_uri, self.staging_root_uri
        if a == b or a.startswith(b + "/") or b.startswith(a + "/"):
            msg = "Lance and staging locations must be separate directories"
            raise ValueError(msg)
        return self


class ExecutionConfig(_Config):
    """Tuning excluded from cutting identity and workspace compatibility."""

    storage_profile: str = Field(default="default", min_length=1)
    probe_mode: Literal["auto", "download"] = "auto"
    probe_timeout_s: int = Field(default=120, ge=1)
    probe_cpus: float = Field(default=0.25, gt=0, allow_inf_nan=False)
    sources_per_checkpoint: int = Field(default=1024, ge=1)
    spans_per_fragment: int = Field(default=100_000, ge=1)
    commit_attempts: int = Field(default=3, ge=1)
    progress: bool = False


class ResolvedVideoCutConfig(_Config):
    """Canonical v1 recipe configuration."""

    schema_version: Literal[1]
    kind: Literal["video-cut"]
    input: InputConfig
    split: FixedStrideConfig = Field(default_factory=FixedStrideConfig)
    output: OutputConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)


def resolve_config_data(raw: Mapping[str, Any], *, overrides: Sequence[str] = ()) -> ResolvedVideoCutConfig:
    """Apply overrides and validate config without importing runtime dependencies."""
    data = copy.deepcopy(dict(raw))
    apply_dotted_overrides(data, overrides)
    return ResolvedVideoCutConfig.model_validate(data)


def resolve_config(path: str | Path, *, overrides: Sequence[str] = ()) -> ResolvedVideoCutConfig:
    """Read a YAML or JSON recipe config."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = "video-cut config must be a mapping"
        raise TypeError(msg)
    return resolve_config_data(data, overrides=overrides)


def resolved_config_to_json(config: ResolvedVideoCutConfig) -> str:
    """Render canonical JSON."""
    return config.model_dump_json(indent=2) + "\n"


def config_schema_json() -> str:
    """Render the public config schema."""
    return json.dumps(ResolvedVideoCutConfig.model_json_schema(), indent=2) + "\n"


def config_template() -> dict[str, Any]:
    """Show defaults while leaving derived output locations editable through their root."""
    return resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-cut",
            "input": {"uris": ["s3://example-bucket/raw/example.mp4"]},
            "output": {"media_root": "s3://example-bucket/curated/video-cut"},
        }
    ).model_dump(mode="json", exclude_none=True, exclude={"output": {"spans_lance_uri", "staging_root_uri"}})


def config_template_yaml() -> str:
    """Render all supported settings and defaults with input/output guidance."""
    preamble = (
        "# All supported settings and their defaults are shown. Unchanged settings may be removed.\n"
        "# To discover a prefix recursively, replace `uris` with `root_uri: s3://example-bucket/raw/`.\n"
        "# Optional output overrides are derived from `media_root` when omitted:\n"
        "#   spans_lance_uri: <media_root>/lance\n"
        "#   staging_root_uri: <media_root>/staging/video-cut\n"
    )
    return preamble + yaml.safe_dump(config_template(), sort_keys=False)


def config_template_payload() -> dict[str, Any]:
    """Describe the editable template for CLI consumers."""
    return {
        "kind": "video-cut",
        "description": "Plan fixed-stride spans with source-level Ray checkpoints.",
        "required_fields": [
            {"path": "schema_version", "example": 1},
            {"path": "kind", "example": "video-cut"},
            {"path": "input.uris|input.root_uri", "example": ["s3://example-bucket/raw/example.mp4"]},
            {"path": "output.media_root", "example": "s3://example-bucket/curated/video-cut"},
        ],
        "config": config_template(),
    }
