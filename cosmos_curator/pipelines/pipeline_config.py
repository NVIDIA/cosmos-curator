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

"""Shared config loading for the deployment CLI and packaged pipeline runtime."""

import json
from pathlib import Path
from typing import Any, cast

import yaml

_YAML_SUFFIXES = frozenset({".yaml", ".yml"})


def load_pipeline_kind_name(config: Path) -> str:
    """Load the exact string ``kind`` discriminator from a JSON/YAML config."""
    data = _load_config_data(config)
    kind = data.get("kind")
    if not isinstance(kind, str):
        msg = f"Config must contain a string 'kind' key (got: {kind!r})"
        raise TypeError(msg)
    return kind


def _load_config_data(config: Path) -> dict[str, Any]:
    if not config.exists():
        msg = f"Pipeline config file not found: {config}"
        raise FileNotFoundError(msg)
    try:
        with config.open(encoding="utf-8") as config_file:
            loaded = yaml.safe_load(config_file) if config.suffix.lower() in _YAML_SUFFIXES else json.load(config_file)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        msg = f"Failed to parse config {config}: {exc}"
        raise ValueError(msg) from exc
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        msg = f"Config file must contain a mapping at the top level, got {type(loaded).__name__}: {config}"
        raise TypeError(msg)
    return cast("dict[str, Any]", loaded)
