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

"""Shared config resolution helpers and execution settings for Curator Next recipes."""

from collections.abc import Sequence
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


def apply_dotted_overrides(data: dict[str, Any], overrides: Sequence[str]) -> None:
    """Apply YAML-valued ``PATH=VALUE`` assignments to a config mapping."""
    for raw_override in overrides:
        if "=" not in raw_override:
            msg = f"--set override must be PATH=VALUE, got {raw_override!r}"
            raise ValueError(msg)
        raw_path, raw_value = raw_override.split("=", maxsplit=1)
        path = raw_path.split(".")
        if any(not part for part in path):
            msg = f"--set override path must contain non-empty keys, got {raw_path!r}"
            raise ValueError(msg)
        try:
            # An empty assignment means the empty string. Null remains available
            # explicitly through YAML's ``null`` or ``~`` spellings.
            value = yaml.safe_load(raw_value) if raw_value else ""
        except yaml.YAMLError as exc:
            msg = f"Failed to parse --set value for {raw_path}: {exc}"
            raise ValueError(msg) from exc

        target = data
        for key in path[:-1]:
            child = target.setdefault(key, {})
            if not isinstance(child, dict):
                msg = f"--set path {raw_path!r} passes through non-object key {key!r}"
                raise TypeError(msg)
            target = child
        target[path[-1]] = value


class CheckpointedExecutionConfig(BaseModel):
    """Storage, checkpoint, and publication tuning shared by checkpointed recipes, excluded from result identity.

    ``EnrichmentExecutionConfig`` and ``AppendExecutionConfig`` add their
    selection limits and publication settings; recipes subclass those for their
    ``execution`` section. Override a default only for a recipe-specific reason.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    storage_profile: str = Field(default="default", min_length=1)
    parquet_rows_per_file: int = Field(
        default=1_000, ge=1, description="Staged results per Parquet file, which is also the checkpoint group size."
    )
    commit_attempts: int = Field(
        default=5, ge=1, description="Attempts for preparing the output table and for publication."
    )
    progress: bool = False
