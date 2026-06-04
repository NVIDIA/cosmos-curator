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

- Pixi carries the tools needed for local development, CI checks, and package
  builds.
- The `gputest` task is exposed on `core` so every GPU runtime env can run it.
- The Pixi `dev` feature stays isolated from runtime environments and image
  defaults so lint tooling is not installed in production containers.
"""

import tomllib
from pathlib import Path

import yaml

from cosmos_curator.client.image_cli.image_app import _parse_envs

_REPO_ROOT = Path(__file__).parents[1]


def _read_repo_file(relative_path: str) -> str:
    return (_REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _read_ci_job(job_name: str) -> dict[str, object]:
    ci_config = yaml.safe_load(_read_repo_file(".gitlab-ci.yml"))
    assert isinstance(ci_config, dict)
    job = ci_config[job_name]
    assert isinstance(job, dict)
    return job


def _script_lines(script: object) -> list[str]:
    assert isinstance(script, list)
    script_lines = []
    for command in script:
        assert isinstance(command, str)
        script_lines.append(command)
    return script_lines


def test_developer_tools_are_declared_in_pixi_dev() -> None:
    """Verify Pixi is the source of truth for developer tooling."""
    pixi_config = tomllib.loads(_read_repo_file("pixi.toml"))

    conda_dependencies = pixi_config.get("feature", {}).get("dev", {}).get("dependencies")
    dev_dependencies = pixi_config.get("feature", {}).get("dev", {}).get("pypi-dependencies")
    dev_tasks = pixi_config.get("feature", {}).get("dev", {}).get("tasks")
    assert isinstance(conda_dependencies, dict)
    assert isinstance(dev_dependencies, dict)
    assert isinstance(dev_tasks, dict)

    for dependency_name in ("mypy", "ruff", "twine"):
        pixi_dependency = dev_dependencies[dependency_name]
        assert isinstance(pixi_dependency, str)
        assert pixi_dependency.startswith("==")

    assert conda_dependencies["pre-commit"] == "==4.2.0"
    assert "nvtop" in conda_dependencies
    assert "python-build" in conda_dependencies
    assert "cosmos-curator" in dev_dependencies
    assert "awscli" in dev_dependencies
    assert "awscli-plugin-endpoint" in dev_dependencies
    required_tasks = {"build", "cosmos-curator", "mypy", "nvtop", "pre-commit", "pytest", "ruff"}
    assert required_tasks.issubset(dev_tasks)
    for task_name in required_tasks:
        task_command = dev_tasks[task_name]
        assert isinstance(task_command, str)
        assert task_command

    assert dev_tasks["cosmos-curator"] == "python -m cosmos_curator.client.cli"


def test_local_test_plugins_are_available_in_pixi_dev() -> None:
    """Verify Pixi includes pytest plugins needed by local test commands."""
    pixi_config = tomllib.loads(_read_repo_file("pixi.toml"))

    dev_dependencies = pixi_config.get("feature", {}).get("dev", {}).get("dependencies")
    assert isinstance(dev_dependencies, dict)

    assert "pytest-mock" in dev_dependencies


def test_gputest_task_is_defined_on_core() -> None:
    """Verify the GPU/env test task lives on `core` so every runtime env has it."""
    pixi_config = tomllib.loads(_read_repo_file("pixi.toml"))
    core_tasks = pixi_config.get("feature", {}).get("core", {}).get("tasks")
    assert isinstance(core_tasks, dict)

    gputest = core_tasks["gputest"]
    assert isinstance(gputest, str)
    assert "pytest -m env" in gputest
    assert "gputest" in core_tasks


def test_user_facing_core_tasks_use_hyphenated_names() -> None:
    """Verify public Pixi tasks follow the CLI naming style."""
    pixi_config = tomllib.loads(_read_repo_file("pixi.toml"))
    core_tasks = pixi_config.get("feature", {}).get("core", {}).get("tasks")
    assert isinstance(core_tasks, dict)

    expected_tasks = {"hello-world", "model-download", "video-pipeline"}
    for task_name in expected_tasks:
        task_command = core_tasks[task_name]
        assert isinstance(task_command, str)
        assert task_command.startswith("python -m ")
        assert task_command.removeprefix("python -m ").strip()

    assert "hello_world" not in core_tasks
    assert "model_download" not in core_tasks
    assert "video_pipeline" not in core_tasks


def test_developer_commands_run_in_dev_environment_only() -> None:
    """Verify developer tooling is isolated from production runtime Pixi environments."""
    pixi_config = tomllib.loads(_read_repo_file("pixi.toml"))
    dev_feature = pixi_config.get("feature", {}).get("dev", {})

    assert dev_feature["channels"] == ["conda-forge"]

    environments = pixi_config.get("environments")
    assert isinstance(environments, dict)
    assert "transformers" not in environments
    assert set(environments["dev"]) == {"core", "transformers", "tracing", "profiling", "dev"}
    for environment_name, features in environments.items():
        if environment_name not in {"dev", "dev-hooks"}:
            assert "dev" not in set(features)


def test_dev_hooks_environment_supports_cross_platform_presubmit() -> None:
    """Verify the minimal pre-submit environment can solve on macOS."""
    pixi_config = tomllib.loads(_read_repo_file("pixi.toml"))

    dev_dependencies = pixi_config.get("feature", {}).get("dev", {}).get("pypi-dependencies")
    dev_hooks_feature = pixi_config.get("feature", {}).get("dev-hooks", {})
    dev_hooks_dependencies = dev_hooks_feature.get("dependencies")
    dev_hooks_pypi_dependencies = dev_hooks_feature.get("pypi-dependencies")
    environments = pixi_config.get("environments")
    assert isinstance(dev_dependencies, dict)
    assert isinstance(dev_hooks_dependencies, dict)
    assert isinstance(dev_hooks_pypi_dependencies, dict)
    assert isinstance(environments, dict)

    assert dev_hooks_feature["channels"] == ["conda-forge"]
    assert dev_hooks_feature["platforms"] == ["linux-64", "linux-aarch64", "osx-arm64"]
    assert dev_hooks_dependencies["python"] == ">=3.12.13,<3.13"
    assert dev_hooks_dependencies["pre-commit"].startswith("=="), (
        "pre-commit should be pinned to an exact version for reproducibility"
    )
    assert "python-build" in dev_hooks_dependencies
    assert dev_hooks_pypi_dependencies["ruff"] == dev_dependencies["ruff"]
    assert environments["dev-hooks"] == {"features": ["dev-hooks"], "no-default-feature": True}


def test_pre_commit_ruff_hooks_use_pixi_dev_hooks_environment() -> None:
    """Verify pre-commit avoids Linux-only runtime dependencies."""
    pre_commit_config = yaml.safe_load(_read_repo_file(".pre-commit-config.yaml"))
    assert isinstance(pre_commit_config, dict)

    repos = pre_commit_config["repos"]
    assert isinstance(repos, list)
    assert all(repo.get("repo") != "https://github.com/astral-sh/ruff-pre-commit" for repo in repos)

    local_repo = next(repo for repo in repos if repo.get("repo") == "local")
    hooks = local_repo["hooks"]
    assert isinstance(hooks, list)
    hooks_by_id = {hook["id"]: hook for hook in hooks}

    assert hooks_by_id["ruff"]["entry"] == (
        "pixi run -e dev-hooks ruff check --fix --force-exclude --config=pyproject.toml"
    )
    assert hooks_by_id["ruff-format"]["entry"] == (
        "pixi run -e dev-hooks ruff format --force-exclude --config=pyproject.toml"
    )


def test_slurm_end_to_end_uses_pixi_dev_for_submit_cli() -> None:
    """Verify the host-side Slurm submit CLI runs from Pixi's Python 3.12 dev environment."""
    slurm_job = _read_ci_job("slurm_end_to_end")
    before_script = _script_lines(slurm_job["before_script"])
    script = _script_lines(slurm_job["script"])
    commands = "\n".join([*before_script, *script])
    pixi_bootstrap_index = next(index for index, command in enumerate(before_script) if "pixi.sh/install.sh" in command)
    pixi_setup_index = next(
        index for index, command in enumerate(before_script) if "pixi install --frozen -e dev" in command
    )

    assert pixi_bootstrap_index < pixi_setup_index
    assert "pixi install --frozen -e dev" in commands
    assert ".gitlab/scripts/slurm_end_to_end.sh" in script
    assert "pip install -e ." not in commands
    assert "source venv/bin/activate" not in commands
    assert "uv venv" not in commands


def test_nvcf_split_benchmark_runs_as_package_module() -> None:
    """Verify the NVCF split benchmark preserves repo-root imports."""
    script = _read_repo_file(".gitlab/scripts/nvcf_split_benchmark.sh")

    assert "python -m benchmarks.split_pipeline.nvcf_split_benchmark" in script
    assert "python benchmarks/split_pipeline/nvcf_split_benchmark.py" not in script


def test_image_cli_default_envs_do_not_include_dev() -> None:
    """Verify image env parsing does not add the developer tooling environment by default."""
    default_envs = set(_parse_envs(""))
    configured_runtime_envs = set(_parse_envs("cuml,legacy-transformers,sam3,seedvr,unified"))

    assert "dev" not in default_envs
    assert "dev" not in configured_runtime_envs
