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
"""Config contract for the caption judge pipeline."""

import copy
import json
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, Field

from cosmos_curator.core.utils.config.pydantic_config import STRICT_CONFIG_MODEL_CONFIG
from cosmos_curator.pipelines.ray_data.constants import DEFAULT_IO_SLOTS_PER_NODE

_YAML_SUFFIXES = frozenset({".yaml", ".yml"})

DEFAULT_JUDGE_MODEL_NAME = "gcp/google/gemini-3.1-pro-preview"
DEFAULT_MAX_WORKERS_PER_NODE = DEFAULT_IO_SLOTS_PER_NODE
DEFAULT_METADATA_RELATIVE_PATH = "lance/v0"

CaptionJudgeKind = Literal["caption_judge"]
SchemaVersion = Literal[1]
CaptionSide = Literal["baseline", "candidate"]
ReportFormat = Literal["json"]
OpenAIEndpointKey = Literal["caption", "enhance", "embedding", "filter", "classifier"]


class CaptionJudgeConfigResolutionError(ValueError):
    """Raised when user caption judge config cannot resolve."""


class CaptionJudgeProviderConfig(BaseModel):
    """Hosted judge endpoint config."""

    model_config = STRICT_CONFIG_MODEL_CONFIG

    provider: Literal["openai"] = "openai"
    endpoint_key: OpenAIEndpointKey = "caption"
    model_name: str = Field(default=DEFAULT_JUDGE_MODEL_NAME, min_length=1)
    max_output_tokens: int = Field(default=8192, ge=1)
    max_retries: int = Field(default=3, ge=1)
    retry_delay_seconds: float = Field(default=1.0, ge=0.0)


class CaptionJudgeInputConfig(BaseModel):
    """Input artifact settings for caption judge configs."""

    model_config = STRICT_CONFIG_MODEL_CONFIG

    baseline: str = Field(min_length=1)
    candidate: str = Field(min_length=1)
    clip_limit: int | None = Field(default=None, ge=1)


class CaptionJudgeOutputConfig(BaseModel):
    """Output report settings for caption judge configs."""

    model_config = STRICT_CONFIG_MODEL_CONFIG

    report_path: str = Field(min_length=1)
    report_format: ReportFormat = "json"


class CaptionJudgeExecutionConfig(BaseModel):
    """Runtime execution settings for caption judge configs."""

    model_config = STRICT_CONFIG_MODEL_CONFIG

    progress: bool = True
    max_workers_per_node: int = Field(default=DEFAULT_MAX_WORKERS_PER_NODE, ge=1)
    allow_unsafe_clip_location: bool = False
    """Skip candidate-root containment checks for clip locations.

    Use only with trusted datasets that intentionally reference media outside the candidate root.
    """


class CaptionJudgePipelineConfig(BaseModel):
    """Fully resolved v1 ``caption_judge`` config used for execution."""

    model_config = STRICT_CONFIG_MODEL_CONFIG

    schema_version: SchemaVersion
    kind: CaptionJudgeKind
    input: CaptionJudgeInputConfig
    output: CaptionJudgeOutputConfig
    judge: CaptionJudgeProviderConfig = Field(default_factory=CaptionJudgeProviderConfig)
    execution: CaptionJudgeExecutionConfig = Field(default_factory=CaptionJudgeExecutionConfig)

    @property
    def baseline(self) -> str:
        """Baseline pipeline output root."""
        return self.input.baseline

    @property
    def candidate(self) -> str:
        """Candidate pipeline output root."""
        return self.input.candidate

    @property
    def baseline_metadata(self) -> str:
        """Derived Lance metadata URI for the baseline output."""
        return caption_judge_metadata_uri(self.input.baseline)

    @property
    def candidate_metadata(self) -> str:
        """Derived Lance metadata URI for the candidate output."""
        return caption_judge_metadata_uri(self.input.candidate)


class UserCaptionJudgeConfig(CaptionJudgePipelineConfig):
    """User-authored v1 ``caption_judge`` config before overrides resolve."""


_CAPTION_JUDGE_REQUIRED_FIELDS: tuple[dict[str, Any], ...] = (
    {
        "path": "schema_version",
        "description": "Config schema version. Use 1 for v1 caption_judge configs.",
        "example": 1,
    },
    {
        "path": "kind",
        "description": "Pipeline kind. Use caption_judge for the Ray Data caption judge pipeline.",
        "example": "caption_judge",
    },
    {
        "path": "input.baseline",
        "description": "Baseline pipeline output root. Lance metadata is read from <input.baseline>/lance/v0.",
        "example": "/config/output/baseline",
    },
    {
        "path": "input.candidate",
        "description": "Candidate pipeline output root. Lance metadata is read from <input.candidate>/lance/v0.",
        "example": "/config/output/candidate",
    },
    {
        "path": "output.report_path",
        "description": "Path where the caption judge report should be written.",
        "example": "/config/output/caption_judge_report.json",
    },
)

_CAPTION_JUDGE_TEMPLATE_DESCRIPTION = "Smallest runnable caption_judge config; relies on packaged defaults."

_CAPTION_JUDGE_TEMPLATE: dict[str, Any] = {
    "schema_version": 1,
    "kind": "caption_judge",
    "input": {
        "baseline": "/config/output/baseline",
        "candidate": "/config/output/candidate",
    },
    "output": {"report_path": "/config/output/caption_judge_report.json"},
}


def caption_judge_metadata_uri(output_uri: str) -> str:
    """Return the Lance metadata URI for a pipeline output root."""
    stripped = output_uri.rstrip("/")
    if not stripped:
        stripped = "/"
    separator = "" if stripped.endswith("/") else "/"
    return f"{stripped}{separator}{DEFAULT_METADATA_RELATIVE_PATH}"


def load_caption_judge_config_data(config_path: str | pathlib.Path) -> dict[str, Any]:
    """Load a user pipeline config file as a mapping."""
    path = pathlib.Path(config_path)
    if not path.exists():
        msg = f"Pipeline config file not found: {path}"
        raise FileNotFoundError(msg)

    try:
        with path.open(encoding="utf-8") as config_file:
            loaded = yaml.safe_load(config_file) if path.suffix.lower() in _YAML_SUFFIXES else json.load(config_file)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        msg = f"Failed to parse config {path}: {exc}"
        raise ValueError(msg) from exc

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        msg = f"Config file must contain a mapping at the top level, got {type(loaded).__name__}: {path}"
        raise TypeError(msg)

    return cast("dict[str, Any]", loaded)


def resolve_caption_judge_config(
    config_path: str | pathlib.Path,
    *,
    overrides: Sequence[str] = (),
) -> CaptionJudgePipelineConfig:
    """Load and resolve a user-authored ``caption_judge`` config file."""
    raw_data = load_caption_judge_config_data(config_path)
    return resolve_caption_judge_config_data(raw_data, overrides=overrides)


def resolve_caption_judge_config_data(
    raw_data: Mapping[str, Any],
    *,
    overrides: Sequence[str] = (),
) -> CaptionJudgePipelineConfig:
    """Resolve raw user config data into the canonical execution config."""
    user_config = UserCaptionJudgeConfig.model_validate(raw_data)
    resolved = user_config.model_dump(mode="python")
    _apply_cli_overrides(resolved, overrides)
    return CaptionJudgePipelineConfig.model_validate(resolved)


def caption_judge_config_to_json(config: CaptionJudgePipelineConfig, *, indent: int = 2) -> str:
    """Render a resolved caption judge config as canonical JSON."""
    return json.dumps(config.model_dump(mode="json"), indent=indent) + "\n"


def user_caption_judge_schema_json(*, indent: int = 2) -> str:
    """Return JSON Schema for user-authored ``caption_judge`` configs."""
    return json.dumps(UserCaptionJudgeConfig.model_json_schema(), indent=indent) + "\n"


def caption_judge_config_template() -> dict[str, Any]:
    """Return a packaged user-authored ``caption_judge`` config template."""
    return copy.deepcopy(_CAPTION_JUDGE_TEMPLATE)


def caption_judge_required_fields() -> list[dict[str, Any]]:
    """Return required author inputs for a runnable ``caption_judge`` config."""
    return [copy.deepcopy(field) for field in _CAPTION_JUDGE_REQUIRED_FIELDS]


def caption_judge_template_payload() -> dict[str, Any]:
    """Return machine-readable template metadata for agent and tool use."""
    return {
        "kind": "caption_judge",
        "description": _CAPTION_JUDGE_TEMPLATE_DESCRIPTION,
        "required_fields": caption_judge_required_fields(),
        "config": caption_judge_config_template(),
    }


def caption_judge_config_to_yaml(data: Mapping[str, Any]) -> str:
    """Render a user-authored caption judge config mapping as editable YAML."""
    return yaml.safe_dump(copy.deepcopy(data), sort_keys=False)


def _apply_cli_overrides(data: dict[str, Any], overrides: Sequence[str]) -> None:
    for raw_override in overrides:
        path, value = _parse_cli_override(raw_override)
        target = data
        for key in path[:-1]:
            next_value = target.get(key)
            if not isinstance(next_value, dict):
                msg = f"--set path {'.'.join(path)} cannot descend into non-object key {key!r}"
                raise CaptionJudgeConfigResolutionError(msg)
            target = cast("dict[str, Any]", next_value)
        target[path[-1]] = value


def _parse_cli_override(raw_override: str) -> tuple[list[str], Any]:
    if "=" not in raw_override:
        msg = f"--set override must be PATH=VALUE, got {raw_override!r}"
        raise CaptionJudgeConfigResolutionError(msg)
    raw_path, raw_value = raw_override.split("=", maxsplit=1)
    path = raw_path.split(".")
    if any(not part for part in path):
        msg = f"--set override path must contain non-empty keys, got {raw_path!r}"
        raise CaptionJudgeConfigResolutionError(msg)
    try:
        value = yaml.safe_load(raw_value) if raw_value else ""
    except yaml.YAMLError as exc:
        msg = f"Failed to parse --set value for {raw_path}: {exc}"
        raise CaptionJudgeConfigResolutionError(msg) from exc
    return path, value
