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

"""Typed config and deterministic resolution for Curator Next ``robot-action-split``."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cosmos_curator.next.append.config import AppendExecutionConfig
from cosmos_curator.next.core.config import apply_dotted_overrides
from cosmos_curator.next.recipes.robot_action_split.contracts import ACTION_CONTRACT_VERSION, MEDIA_CONTRACT_VERSION
from cosmos_curator.next.utils.uris import normalize_s3_uri

_YAML_SUFFIXES = frozenset({".yaml", ".yml"})
_MODEL_CONFIG = ConfigDict(frozen=True, strict=True, extra="forbid")

RobotActionSplitKind = Literal["robot-action-split"]
SchemaVersion = Literal[1]
ActionFormat = Literal["bin", "pickle"]

# The value Ray Data reads as "do not watch for progress at all".
_NO_PROGRESS_TIMEOUT_DISABLED = -1.0


class RobotActionSplitInputConfig(BaseModel):
    """Source dataset roots and dataset identity."""

    model_config = _MODEL_CONFIG

    uris: tuple[str, ...] = Field(
        min_length=1,
        strict=False,
        description="One or more LeRobot/Mecka dataset roots (shard directories).",
        examples=[["s3://example-bucket/robot_data/lerobot_v30/my_dataset/"]],
    )
    source_dataset: str = Field(
        min_length=1,
        description="Registered dataset name; selects the ActionBinarySpec for .bin serialization.",
        examples=["my_dataset_name"],
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Cap on the number of data chunk directories (data/chunk-NNN/) discovery reads, counted across "
            "all input URIs and shards in discovery order. Every data file in a kept directory is read, and "
            "its spans may reference several chunk MP4s per view, so one directory can yield many work items "
            "and clips. Applied before dedup; keeping whole directories keeps each episode complete. Use for "
            "smoke runs and local iteration."
        ),
    )
    storage_profile: str = Field(
        default="default",
        min_length=1,
        description=(
            "Profile for reading the source datasets. Defaults to execution.storage_profile, which serves "
            "the outputs. Set it when the input lives in a different object store than the output: a profile "
            "carries one endpoint_url, so a single profile cannot address both (e.g. GCS-hosted source "
            "video, S3-hosted clips)."
        ),
    )


class SpanFilterConfig(BaseModel):
    """Subtask span filter and per-episode dedup settings."""

    model_config = _MODEL_CONFIG

    min_duration_s: float = Field(default=4.0, gt=0.0, description="Drop spans shorter than this.")
    max_duration_s: float | None = Field(default=20.0, description="Drop spans longer than this. null = no cap.")
    skip_labels: tuple[str, ...] = Field(
        default=("no action", "no actions"),
        strict=False,
        description="Exact subtask_name matches to drop (case-insensitive).",
    )
    skip_label_prefixes: tuple[str, ...] = Field(
        default=("hold", "adjust"),
        strict=False,
        description="Drop spans whose subtask_name starts with any of these prefixes (case-insensitive).",
    )
    skip_label_substrings: tuple[str, ...] = Field(
        default=("idle",),
        strict=False,
        description="Drop spans whose subtask_name contains any of these substrings (case-insensitive).",
    )

    @field_validator("max_duration_s")
    @classmethod
    def _max_duration_positive(cls, v: float | None) -> float | None:
        if v is not None and v <= 0.0:
            msg = f"max_duration_s must be positive, got {v}"
            raise ValueError(msg)
        return v

    max_keep_per_description: int = Field(
        default=3,
        ge=1,
        description="Per-episode dedup: keep at most this many spans per unique subtask_name.",
    )
    dedup_prefer_min_s: float = Field(
        default=5.0,
        gt=0.0,
        description="Preferred duration lower bound for per-episode dedup ranking.",
    )
    dedup_prefer_max_s: float = Field(
        default=10.0,
        gt=0.0,
        description="Preferred duration upper bound for per-episode dedup ranking.",
    )


class RobotActionSplitOutputConfig(BaseModel):
    """Durable output locations and format settings."""

    model_config = _MODEL_CONFIG

    media_root: str = Field(
        min_length=1,
        description="Root for clip MP4s, action payloads, and JSON sidecars.",
        examples=["s3://example-bucket/robot_clips/"],
    )
    lance_uri: str = Field(
        default="",
        min_length=1,
        description="Lance clip table URI. Defaults to <media_root>/lance/clips.lance.",
        examples=["s3://example-bucket/robot_clips/lance/clips.lance"],
    )
    staging_root_uri: str = Field(
        default="",
        min_length=1,
        description=(
            "Recovery workspace for checkpointed chunk outcomes awaiting publication. "
            "Defaults to <media_root>/staging/robot-action-split."
        ),
    )
    action_format: ActionFormat = Field(
        default="bin",
        description="Per-span action serialization format.",
    )
    video_bitrate: str = Field(
        default="2M",
        pattern=r"^[1-9][0-9]*(\.[0-9]+)?[kKmMgG]?$",
        description=(
            "Target bitrate for libopenh264 re-encoded frames. "
            "In smart-cut mode this only applies to the pre-keyframe head (~0.5 frames/clip on average "
            "for GOP=2 sources like Mecka); interior frames are stream-copied bit-exact and are unaffected "
            "by this setting. In the full re-encode fallback it applies to all frames. "
            "Unlike libx264 CRF, this is a fixed bitrate ceiling rather than a quality target, "
            "so the right value depends on source resolution and content complexity."
        ),
    )
    views: tuple[str, ...] = Field(
        default=(),
        strict=False,
        description="Camera views to extract. Empty = all available views.",
        examples=[["observation.images.main", "observation.images.wrist_image"]],
    )

    @model_validator(mode="before")
    @classmethod
    def _derive_locations(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        resolved = dict(value)
        media_root = resolved.get("media_root")
        if isinstance(media_root, str):
            root = media_root.rstrip("/")
            resolved.setdefault("lance_uri", f"{root}/lance/clips.lance")
            resolved.setdefault("staging_root_uri", f"{root}/staging/robot-action-split")
        return resolved

    @field_validator("media_root", "lance_uri", "staging_root_uri")
    @classmethod
    def _location(cls, value: str) -> str:
        if value != value.strip():
            msg = "Output locations cannot contain surrounding whitespace"
            raise ValueError(msg)
        return normalize_s3_uri(value).rstrip("/") if "://" in value else str(Path(value).expanduser().resolve())

    @model_validator(mode="after")
    def _separate(self) -> Self:
        a, b = self.lance_uri, self.staging_root_uri
        if a == b or a.startswith(b + "/") or b.startswith(a + "/"):
            msg = "Lance and staging locations must be separate directories"
            raise ValueError(msg)
        return self


class RobotActionSplitExecutionConfig(AppendExecutionConfig):
    """Execution-only settings excluded from clip identities and recovery compatibility.

    ``storage_profile`` serves the clip table, staging root, and media payloads;
    ``input.storage_profile`` serves the source datasets. A work item is one
    source chunk, so ``max_items`` and ``rows_per_checkpoint`` count chunks,
    while ``rows_per_fragment`` counts clip rows.
    """

    rows_per_checkpoint: int = Field(
        default=8,
        ge=1,
        description=(
            "Chunk outcomes per checkpoint group. A chunk takes minutes to cut, so small groups bound the work "
            "an interrupted run repeats."
        ),
    )
    max_items: int | None = Field(
        default=1_000,
        ge=1,
        description=(
            "Maximum chunks in a new publication selection; saved selections resume unchanged. Each chunk's "
            "spans are saved in the selection manifest, so this also bounds its size. Lower it to publish "
            "sooner; null selects every pending chunk. Rerun while remaining_items is nonzero or null."
        ),
    )
    rows_per_fragment: int = Field(
        default=8_000,
        ge=1,
        description="Clip rows per appended Lance fragment, sized for downstream Ray Data parallelism.",
    )
    cut_cpus: float = Field(
        default=4.0, gt=0.0, description="CPUs per chunk task; its cuts run that many ffmpeg processes at once."
    )
    discovery_workers: int = Field(default=4, ge=1, description="Parquet discovery thread pool size.")
    media_write_attempts: int = Field(
        default=3, ge=1, description="Attempts per clip, action, or sidecar write before the chunk task fails."
    )
    chunk_download_attempts: int = Field(
        default=3, ge=1, description="Attempts to stage a remote chunk MP4 before the chunk task fails."
    )
    no_progress_timeout_s: float = Field(
        default=2 * 60 * 60,
        description=(
            "Seconds the run may go without any chunk finishing before Ray Data calls it wedged; -1 disables "
            "the check. Well past Ray's own 30 minutes, because one task downloads and cuts a whole chunk MP4, "
            "which can hold hours of video and hundreds of spans, and every chunk starts at once."
        ),
    )
    tmp_dir: str | None = Field(
        default=None,
        description=(
            "Base directory for temporary chunk MP4 files during cutting. "
            "Defaults to the system temp dir (typically /tmp on Linux). "
            "Set to a path with more space (e.g. /config/tmp, backed by the "
            "workspace Lustre mount) when running many parallel Ray Data workers "
            "that would otherwise exhaust node-local /tmp."
        ),
    )

    @field_validator("no_progress_timeout_s")
    @classmethod
    def _timeout(cls, value: float) -> float:
        if value <= 0 and value != _NO_PROGRESS_TIMEOUT_DISABLED:
            msg = f"no_progress_timeout_s must be positive, or {_NO_PROGRESS_TIMEOUT_DISABLED} to disable it"
            raise ValueError(msg)
        return value


class ResolvedRobotActionSplitConfig(BaseModel):
    """Canonical v1 execution contract for Curator Next ``robot-action-split``."""

    model_config = _MODEL_CONFIG

    schema_version: SchemaVersion
    kind: RobotActionSplitKind
    input: RobotActionSplitInputConfig
    split: SpanFilterConfig = Field(default_factory=SpanFilterConfig)
    output: RobotActionSplitOutputConfig
    execution: RobotActionSplitExecutionConfig = Field(default_factory=RobotActionSplitExecutionConfig)

    @model_validator(mode="before")
    @classmethod
    def _default_input_profile_to_execution(cls, data: object) -> object:
        """Fill an omitted ``input.storage_profile`` from ``execution.storage_profile``.

        Resolved here so the canonical config always names both profiles, and a
        single-store config keeps working unchanged. Absence is the key being
        missing: an explicit empty or null profile is a broken override and fails
        validation instead of silently reading from the output store.
        """
        if not isinstance(data, Mapping):
            return data
        source = data.get("input")
        if not isinstance(source, Mapping) or "storage_profile" in source:
            return data
        execution = data.get("execution")
        profile = execution.get("storage_profile", "default") if isinstance(execution, Mapping) else "default"
        return {**data, "input": {**source, "storage_profile": profile}}

    @model_validator(mode="after")
    def _registered_action_spec(self) -> Self:
        """Refuse an unregistered dataset for ACT2 actions before any clip is cut.

        Otherwise every clip would fail to encode its action data, and those
        terminal errors would be published, permanently, for the whole table.
        """
        if self.output.action_format == "bin":
            # Deferred: only an ACT2 config needs the registry.
            from cosmos_curator.next.media.action_binary import get_action_binary_spec  # noqa: PLC0415

            try:
                get_action_binary_spec(self.input.source_dataset)
            except ValueError as exc:
                msg = (
                    f"{exc} Register it in cosmos_curator/next/media/action_binary.py, or set "
                    "output.action_format: pickle."
                )
                raise ValueError(msg) from exc
        return self

    @property
    def split_contract(self) -> dict[str, object]:
        """Describe every result-defining setting: which spans become clips and how their payloads are encoded.

        This is the append result contract, which binds the clip table and scopes
        recovery. ``media_root`` belongs to it because finished clips record payload
        URIs beneath it and validation checks them there. The table and staging
        locations, views, discovery limits, and execution tuning select or place
        work without changing a chunk's outcome, so they stay outside it.
        """
        return {
            # Robot-action-split owns this version. Bump when implementation changes
            # alter chunk outcomes for identical inputs and settings, such as span
            # boundaries, filtering, dedup, the terminal-error policy, or row values.
            "behavior_version": 1,
            "media_contract_version": MEDIA_CONTRACT_VERSION,
            "action_contract_version": ACTION_CONTRACT_VERSION,
            "source_dataset": self.input.source_dataset,
            "action_format": self.output.action_format,
            "video_bitrate": self.output.video_bitrate,
            "split": self.split.model_dump(mode="json"),
            "media_root": self.output.media_root,
        }


def load_config(config_path: str | Path) -> ResolvedRobotActionSplitConfig:
    """Load and validate a ``robot-action-split`` config file."""
    path = Path(config_path)
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) if path.suffix.lower() in _YAML_SUFFIXES else json.load(f)
    return ResolvedRobotActionSplitConfig.model_validate(raw)


def resolve_config(
    config_path: str | Path,
    overrides: tuple[str, ...] | list[str] = (),
) -> ResolvedRobotActionSplitConfig:
    """Load a config file and apply ``--set`` overrides before validation.

    Each override has the form ``"path.to.key=value"`` where *value* is
    parsed with ``yaml.safe_load`` so numeric and boolean literals are
    converted to native types automatically.

    Example::

        resolve_config("config.yaml", overrides=["split.min_duration_s=3.0"])
    """
    path = Path(config_path)
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8") as f:
        loaded: object = yaml.safe_load(f) if path.suffix.lower() in _YAML_SUFFIXES else json.load(f)
    if not isinstance(loaded, dict):
        msg = f"Config file must contain a mapping at the top level, got {type(loaded).__name__}: {path}"
        raise TypeError(msg)
    raw: dict[str, Any] = loaded
    apply_dotted_overrides(raw, overrides)
    return ResolvedRobotActionSplitConfig.model_validate(raw)
