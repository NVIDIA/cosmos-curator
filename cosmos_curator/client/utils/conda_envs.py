# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Stores information about the available conda envs and contains utilities for building them."""

import sys
from pathlib import Path

import attrs
import tomli
from loguru import logger

from cosmos_curator.client.environment import CONTAINER_PATHS_CODE_DIR


@attrs.define
class CommonTemplateParams:
    """Template params that are in common."""

    code_dir_str: str = CONTAINER_PATHS_CODE_DIR.as_posix()

    @classmethod
    def make(cls) -> "CommonTemplateParams":
        """Return a CommonTemplateParams instance with default template parameters."""
        return CommonTemplateParams()


def get_pixi_envs() -> list[str]:
    """Get list of environments from pixi.toml environments section."""
    toml_path = Path.cwd() / "pixi.toml"
    if not toml_path.is_file():
        logger.error(f"pixi.toml not found at {toml_path}")
        sys.exit(1)

    with toml_path.open("rb") as f:
        data = tomli.load(f)

    envs_section = data.get("environments")
    if not isinstance(envs_section, dict):
        logger.error("No 'environments' section found in pixi.toml")
        sys.exit(1)

    return list(envs_section.keys())
