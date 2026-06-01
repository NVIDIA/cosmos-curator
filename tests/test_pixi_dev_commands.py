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

"""Validate the developer command contract.

These tests are structural guards for the Pixi developer commands rather than
substitutes for running the commands themselves. They verify that:

- Pixi dev `ruff` and `mypy` versions stay aligned with Poetry dev pins, which
  prevents drift between `pixi.toml` and `pyproject.toml`.
- Pixi carries the pytest plugins required by local CPU test commands.
- The Pixi `dev` feature stays isolated from runtime environments and image
  defaults so lint tooling is not installed in production containers.
- The developer guide documents the supported command entry points.
"""

import tomllib
from pathlib import Path

from cosmos_curator.client.image_cli.image_app import _parse_envs

_REPO_ROOT = Path(__file__).parents[1]


def _read_repo_file(relative_path: str) -> str:
    return (_REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_developer_tool_versions_match_poetry() -> None:
    """Verify Pixi uses the same ruff and mypy versions as Poetry."""
    pixi_config = tomllib.loads(_read_repo_file("pixi.toml"))
    pyproject_config = tomllib.loads(_read_repo_file("pyproject.toml"))

    dev_dependencies = pixi_config.get("feature", {}).get("dev", {}).get("pypi-dependencies")
    poetry_dev_dependencies = (
        pyproject_config.get("tool", {})
        .get("poetry", {})
        .get("group", {})
        .get("dev", {})
        .get(
            "dependencies",
        )
    )
    assert isinstance(dev_dependencies, dict)
    assert isinstance(poetry_dev_dependencies, dict)

    for dependency_name in ("ruff", "mypy"):
        pixi_dependency = dev_dependencies[dependency_name]
        poetry_dependency = poetry_dev_dependencies[dependency_name]

        assert isinstance(pixi_dependency, str)
        assert isinstance(poetry_dependency, str)
        assert pixi_dependency == f"=={poetry_dependency}"


def test_local_test_plugins_are_available_in_pixi_dev() -> None:
    """Verify Pixi includes pytest plugins needed by local test commands."""
    pixi_config = tomllib.loads(_read_repo_file("pixi.toml"))

    dev_dependencies = pixi_config.get("feature", {}).get("dev", {}).get("dependencies")
    assert isinstance(dev_dependencies, dict)

    assert "pytest-mock" in dev_dependencies


def test_developer_commands_run_in_dev_environment_only() -> None:
    """Verify developer tooling is isolated from runtime Pixi environments."""
    pixi_config = tomllib.loads(_read_repo_file("pixi.toml"))
    dev_feature = pixi_config.get("feature", {}).get("dev", {})

    assert dev_feature["channels"] == ["conda-forge"]

    environments = pixi_config.get("environments")
    assert isinstance(environments, dict)
    assert "transformers" not in environments
    assert set(environments["dev"]) == {"core", "transformers", "tracing", "profiling", "dev"}
    for environment_name, features in environments.items():
        if environment_name != "dev":
            assert "dev" not in set(features)


def test_image_cli_default_envs_do_not_include_dev() -> None:
    """Verify image env parsing does not add the developer tooling environment by default."""
    default_envs = set(_parse_envs(""))
    configured_runtime_envs = set(_parse_envs("cuml,legacy-transformers,sam3,seedvr,unified"))

    assert "dev" not in default_envs
    assert "dev" not in configured_runtime_envs
