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
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cosmos_curator.next.core.config import apply_dotted_overrides

_YAML_SUFFIXES = frozenset({".yaml", ".yml"})
_MODEL_CONFIG = ConfigDict(frozen=True, strict=True, extra="forbid")

RobotActionSplitKind = Literal["robot-action-split"]
SchemaVersion = Literal[1]
ActionFormat = Literal["bin", "pickle"]


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
            "Cap on the number of distinct source chunk MP4s processed during discovery. "
            "Matches video-split semantics where limit bounds the number of source videos. "
            "Each chunk may yield multiple spans. Use for smoke runs and local iteration."
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
        description="Root for clip MP4s, action .bin files, JSON sidecars, manifests, and receipts.",
        examples=["s3://example-bucket/robot_clips/"],
    )
    lance_uri: str = Field(
        min_length=1,
        description="Lance clip dataset URI.",
        examples=["s3://example-bucket/robot_clips/lance/clips.lance"],
    )
    action_format: ActionFormat = Field(
        default="bin",
        description="Per-span action serialization format.",
    )
    video_bitrate: str = Field(
        default="2M",
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


class RobotActionSplitExecutionConfig(BaseModel):
    """Execution-only settings excluded from stable span and clip IDs."""

    model_config = _MODEL_CONFIG

    storage_profile: str = Field(default="default", min_length=1)
    output_storage_profile: str = Field(
        default="default",
        min_length=1,
        description=(
            "Profile for writing clips, action payloads, and the Lance table. Defaults to "
            "storage_profile. Set it when output lives in a different object store than the "
            "input: a profile carries one endpoint_url, so a single profile cannot address "
            "both (e.g. GCS-hosted source video, S3-hosted clips)."
        ),
    )
    cut_cpus: float = Field(default=4.0, gt=0.0, description="CPUs per smart-cut worker.")
    discovery_workers: int = Field(default=4, ge=1, description="Parquet discovery thread pool size.")
    max_segments_per_batch: int = Field(
        default=50,
        ge=1,
        description="Split chunk batches larger than this to allow Ray to schedule multiple tasks per chunk.",
    )
    cut_attempts: int = Field(default=3, ge=1)
    media_write_attempts: int = Field(default=3, ge=1)
    storage_attempts: int = Field(default=3, ge=1, description="Retries for an idempotent Lance fragment append.")
    clips_per_publish_batch: int = Field(
        default=8_000,
        ge=1,
        description=(
            "Successful clip rows buffered before staging and committing one Lance fragment. "
            "Bounds recomputation on a crash and sizes fragments for downstream Ray Data parallelism."
        ),
    )
    progress: bool = False
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
    ray_data: bool = Field(
        default=True,
        description=(
            "Use Ray Data flat_map for parallel clip cutting (default). "
            "When true, ray.init connects to an existing cluster (address='auto' in managed "
            "Slurm-Ray jobs, local single-node otherwise). Workers process batches in parallel; "
            "each worker downloads its own chunk copy. Set false to fall back to the sequential "
            "single-threaded loop, e.g. for debugging."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _default_output_profile_to_input(cls, data: object) -> object:
        """Fill an omitted ``output_storage_profile`` from ``storage_profile``.

        Resolved here rather than at each call site so the canonical config always
        carries a concrete value, and a single-store config keeps working unchanged.

        Absence is the key being missing, not the value being falsy: an explicit
        ``output_storage_profile: ""`` is a broken override, and rewriting it to the
        input profile would silently send output to the wrong store instead of
        failing the ``min_length=1`` check.
        """
        if isinstance(data, dict) and "output_storage_profile" not in data:
            return {**data, "output_storage_profile": data.get("storage_profile", "default")}
        return data


class ResolvedRobotActionSplitConfig(BaseModel):
    """Canonical v1 execution contract for Curator Next ``robot-action-split``."""

    model_config = _MODEL_CONFIG

    schema_version: SchemaVersion
    kind: RobotActionSplitKind
    input: RobotActionSplitInputConfig
    split: SpanFilterConfig = Field(default_factory=SpanFilterConfig)
    output: RobotActionSplitOutputConfig
    execution: RobotActionSplitExecutionConfig = Field(default_factory=RobotActionSplitExecutionConfig)


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
