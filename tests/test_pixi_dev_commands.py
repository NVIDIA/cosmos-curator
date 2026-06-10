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

- Pixi separates cross-platform tools, Linux development, cluster operations,
  and runtime environments.
- The `gputest` task is exposed on `core` so every GPU runtime env can run it.
- Developer and operational environments stay isolated from image defaults so
  lint tooling is not installed in production containers.
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


def test_workspace_default_feature_is_cross_platform_minimal() -> None:
    """Verify the implicit Pixi default feature stays portable and boring."""
    pixi_config = tomllib.loads(_read_repo_file("pixi.toml"))

    assert pixi_config["workspace"]["channels"] == ["rapidsai", "conda-forge"]
    assert pixi_config["workspace"]["platforms"] == ["linux-64", "linux-aarch64", "osx-arm64"]
    assert pixi_config["dependencies"] == {"python": ">=3.12.13,<3.13", "pip": "*"}
    assert "pypi-dependencies" not in pixi_config
    assert "nvidia" not in pixi_config["workspace"]["channels"]


def test_tools_environment_declares_cross_platform_repo_tooling() -> None:
    """Verify tools is the cross-platform source of truth for basic repo tooling."""
    pixi_config = tomllib.loads(_read_repo_file("pixi.toml"))

    tools_feature = pixi_config["feature"]["tools"]
    conda_dependencies = tools_feature["dependencies"]
    tools_dependencies = tools_feature["pypi-dependencies"]
    tools_tasks = tools_feature["tasks"]

    assert tools_feature["channels"] == ["conda-forge"]
    assert tools_feature["platforms"] == ["linux-64", "linux-aarch64", "osx-arm64"]
    assert conda_dependencies["pre-commit"] == "==4.2.0"
    assert "python-build" in conda_dependencies
    assert "setuptools-scm" in conda_dependencies
    assert tools_dependencies["cosmos-curator"] == {"path": ".", "editable": True}
    assert tools_dependencies["ruff"].startswith("==")
    assert "mypy" not in tools_dependencies
    assert "torch" not in tools_dependencies

    required_tasks = {"build", "cosmos-curator", "pre-commit", "ruff"}
    assert required_tasks.issubset(tools_tasks)
    for task_name in required_tasks:
        task_command = tools_tasks[task_name]
        assert isinstance(task_command, str)
        assert task_command

    assert tools_tasks["cosmos-curator"] == "python -m cosmos_curator.client.cli"


def test_cluster_environment_declares_slurm_and_node_diagnostics() -> None:
    """Verify cluster is Linux-only and carries operational CLI dependencies."""
    pixi_config = tomllib.loads(_read_repo_file("pixi.toml"))

    cluster_feature = pixi_config["feature"]["cluster"]
    conda_dependencies = cluster_feature["dependencies"]
    cluster_dependencies = cluster_feature["pypi-dependencies"]
    cluster_tasks = cluster_feature["tasks"]

    assert cluster_feature["channels"] == ["conda-forge"]
    assert cluster_feature["platforms"] == ["linux-64", "linux-aarch64"]
    assert "nvitop" in conda_dependencies
    assert "nvtop" in conda_dependencies
    assert "awscli" in cluster_dependencies
    assert "awscli-plugin-endpoint" in cluster_dependencies
    assert cluster_tasks["nvitop"] == "nvitop"
    assert cluster_tasks["nvtop"] == "nvtop"
    assert pixi_config["environments"]["cluster"] == ["tools", "cluster"]


def test_dev_environment_declares_linux_cpu_test_tooling() -> None:
    """Verify dev is Linux-only, includes cluster tools, and omits main runtime."""
    pixi_config = tomllib.loads(_read_repo_file("pixi.toml"))

    dev_feature = pixi_config["feature"]["dev"]
    dev_conda_dependencies = dev_feature["dependencies"]
    dev_pypi_dependencies = dev_feature["pypi-dependencies"]

    assert dev_feature["channels"] == ["conda-forge"]
    assert dev_feature["platforms"] == ["linux-64", "linux-aarch64"]
    for dependency_name in ("gpustat", "pytest-mock", "safetensors"):
        assert dependency_name in dev_conda_dependencies
    assert dev_pypi_dependencies["mypy"].startswith("==")
    assert dev_pypi_dependencies["twine"].startswith("==")
    for dependency_name in ("torch", "torchvision"):
        assert dependency_name in dev_pypi_dependencies
    assert "ruff" not in dev_pypi_dependencies
    assert "cosmos-curator" not in dev_pypi_dependencies
    assert "awscli" not in dev_pypi_dependencies
    assert pixi_config["environments"]["dev"] == [
        "tools",
        "cluster",
        "core",
        "transformers",
        "tracing",
        "profiling",
        "dev",
    ]
    assert "runtime" not in pixi_config["environments"]["dev"]


def test_runtime_features_are_separated_from_core() -> None:
    """Verify core is slim and heavy default runtime dependencies live on runtime."""
    pixi_config = tomllib.loads(_read_repo_file("pixi.toml"))
    features = pixi_config["feature"]

    core_feature = features["core"]
    core_dependencies = core_feature["dependencies"]
    core_pypi_dependencies = core_feature["pypi-dependencies"]
    runtime_feature = features["runtime"]
    runtime_pypi_dependencies = runtime_feature["pypi-dependencies"]

    assert "channels" not in core_feature
    assert core_pypi_dependencies["cosmos-xenna"] == "==0.4.3"
    assert core_pypi_dependencies["ray"] == {"version": "==2.55.1", "extras": ["default", "data"]}
    for dependency_name in ("fastapi", "starlette", "uvicorn", "websockets"):
        assert dependency_name in core_dependencies
    for dependency_name in ("google-genai", "webdataset"):
        assert dependency_name in core_pypi_dependencies
    for dependency_name in ("torch", "torchvision", "vllm", "cvcuda-cu13", "PyNvVideoCodec"):
        assert dependency_name not in core_pypi_dependencies

    assert runtime_feature["channels"] == ["conda-forge", "nvidia"]
    assert runtime_feature["platforms"] == ["linux-64", "linux-aarch64"]
    assert runtime_feature["system-requirements"]["cuda"] == "13.0.2"
    for dependency_name in ("torch", "torchvision", "vllm", "cvcuda-cu13", "PyNvVideoCodec"):
        assert dependency_name in runtime_pypi_dependencies

    assert "unified" not in features
    assert pixi_config["environments"]["default"] == ["core", "runtime", "transformers", "tracing", "profiling"]


def test_legacy_transformers_environment_is_model_specific_runtime() -> None:
    """Verify legacy models avoid the main runtime's newer transformers stack."""
    pixi_config = tomllib.loads(_read_repo_file("pixi.toml"))

    legacy_feature = pixi_config["feature"]["legacy-transformers"]
    legacy_pypi_dependencies = legacy_feature["pypi-dependencies"]

    assert legacy_feature["dependencies"]["transformers"] == "==4.55.4"
    for dependency_name in ("av", "opencv-python-headless", "timm", "torch", "torchvision"):
        assert dependency_name in legacy_pypi_dependencies
    assert "transformers-cosmos3" not in legacy_pypi_dependencies
    assert "vllm" not in legacy_pypi_dependencies
    assert pixi_config["environments"]["legacy-transformers"] == [
        "core",
        "legacy-transformers",
        "tracing",
        "profiling",
    ]
    assert "runtime" not in pixi_config["environments"]["legacy-transformers"]


def test_cuml_owns_rapids_channel() -> None:
    """Verify direct RAPIDS dependencies are scoped to cuML."""
    pixi_config = tomllib.loads(_read_repo_file("pixi.toml"))

    features = pixi_config["feature"]
    assert pixi_config["workspace"]["channels"][0] == "rapidsai"
    assert features["cuml"]["channels"] == ["nvidia"]
    assert pixi_config["environments"]["cuml"] == ["core", "cuml", "tracing"]
    assert features["cuml"]["dependencies"]["cuml"] == "==26.02"
    assert "raft-dask" in features["cuml"]["dependencies"]


def test_model_download_environment_reuses_core() -> None:
    """Verify model download workers get core dependencies, including Ray."""
    pixi_config = tomllib.loads(_read_repo_file("pixi.toml"))

    assert "model-download" not in pixi_config["feature"]
    assert pixi_config["environments"]["model-download"] == ["core", "tracing"]


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

    environments = pixi_config.get("environments")
    assert isinstance(environments, dict)
    assert "transformers" not in environments
    assert "dev-hooks" not in environments
    assert "tools" in environments
    assert "cluster" in environments
    for environment_name, features in environments.items():
        if environment_name != "dev":
            assert "dev" not in set(features)
        if environment_name not in {"tools", "cluster", "dev"}:
            assert "tools" not in set(features)
        if environment_name not in {"cluster", "dev"}:
            assert "cluster" not in set(features)


def test_no_dev_hooks_environment_is_declared() -> None:
    """Verify tools replaces the old hooks-only environment."""
    pixi_config = tomllib.loads(_read_repo_file("pixi.toml"))

    assert "dev-hooks" not in pixi_config["feature"]
    assert "dev-hooks" not in pixi_config["environments"]


def test_pre_commit_ruff_hooks_use_pixi_tools_environment() -> None:
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
        "pixi run --frozen -e tools ruff check --fix --force-exclude --config=pyproject.toml"
    )
    assert hooks_by_id["ruff-format"]["entry"] == (
        "pixi run --frozen -e tools ruff format --force-exclude --config=pyproject.toml"
    )

    devset_script = _read_repo_file("devset.sh")
    assert "pixi run -e tools pre-commit install" in devset_script
    assert "pixi run -e tools python -m build" in devset_script
    assert "dev-hooks" not in devset_script


def test_slurm_end_to_end_uses_pixi_cluster_for_submit_cli() -> None:
    """Verify the host-side Slurm submit CLI runs from Pixi's cluster environment."""
    slurm_job = _read_ci_job("slurm_end_to_end")
    before_script = _script_lines(slurm_job["before_script"])
    script = _script_lines(slurm_job["script"])
    after_script = _script_lines(slurm_job["after_script"])
    commands = "\n".join([*before_script, *script])
    pixi_bootstrap_index = next(index for index, command in enumerate(before_script) if "pixi.sh/install.sh" in command)
    pixi_cache_index = next(
        index
        for index, command in enumerate(before_script)
        if 'PIXI_CACHE_DIR="${SLURM_E2E_PIXI_CACHE_DIR}"' in command
    )
    pixi_setup_index = next(
        index for index, command in enumerate(before_script) if "pixi install --frozen -e cluster" in command
    )

    assert pixi_bootstrap_index < pixi_cache_index < pixi_setup_index
    assert slurm_job["variables"]["SLURM_E2E_PIXI_CACHE_DIR"] == (
        "/lustre/fsw/coreai_dlalgo_ci/nemo_video_curator/pixi/cache"
    )
    assert "pixi install --frozen -e cluster" in commands
    assert ".gitlab/scripts/slurm_end_to_end.sh" in script
    assert 'rm -rf "${CI_PROJECT_DIR}/.pixi"' in after_script
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
    configured_runtime_envs = set(_parse_envs("cuml,legacy-transformers,sam3,seedvr"))

    for env_name in ("tools", "cluster", "dev"):
        assert env_name not in default_envs
        assert env_name not in configured_runtime_envs
