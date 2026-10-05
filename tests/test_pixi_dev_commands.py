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
- The `gpu-test` task is exposed on `core` so every GPU runtime env can run it.
- Developer and operational environments stay isolated from image defaults so
  lint tooling is not installed in production containers.
"""

import inspect
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import yaml

from cosmos_curator.client.image_cli.image_app import _parse_envs
from cosmos_curator.client.image_cli.image_app import build as build_image

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
    assert pixi_config["workspace"]["conda-pypi-map"] == {"conda-forge": "conda-pypi-map.json"}
    assert pixi_config["workspace"]["platforms"] == [
        "linux-64",
        "linux-aarch64",
        "osx-arm64",
        {"name": "linux-64-cuda", "platform": "linux-64", "cuda": "13.0.3", "glibc": "2.35"},
        {"name": "linux-aarch64-cuda", "platform": "linux-aarch64", "cuda": "13.0.3", "glibc": "2.35"},
    ]
    assert pixi_config["dependencies"] == {"python": "==3.13.14", "pip": "*"}
    assert "pypi-dependencies" not in pixi_config
    assert "nvidia" not in pixi_config["workspace"]["channels"]
    assert pixi_config["feature"]["linux"]["platforms"] == ["linux-64", "linux-aarch64"]

    conda_pypi_map = json.loads(_read_repo_file("conda-pypi-map.json"))
    assert conda_pypi_map == {
        "libopencv": "opencv-python-headless",
        "opencv": "opencv-contrib-python",
        "py-opencv": "opencv-python",
    }


def test_package_metadata_uses_python_313_baseline() -> None:
    """Keep standalone wheel support aligned with the canonical Pixi environment."""
    project_config = tomllib.loads(_read_repo_file("pyproject.toml"))["project"]

    assert project_config["requires-python"] == ">=3.13,<3.14"
    assert "Programming Language :: Python :: 3.13" in project_config["classifiers"]
    assert "Programming Language :: Python :: 3.12" not in project_config["classifiers"]


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
    """Verify dev provides CPU test tooling without shadowing runtime pytest."""
    pixi_config = tomllib.loads(_read_repo_file("pixi.toml"))

    dev_feature = pixi_config["feature"]["dev"]
    dev_conda_dependencies = dev_feature["dependencies"]
    dev_pypi_dependencies = dev_feature["pypi-dependencies"]

    assert {"unit-test", "unit-test-fast"}.issubset(dev_feature["tasks"])
    assert "pytest" not in pixi_config.get("tasks", {})
    for feature_name, feature in pixi_config["feature"].items():
        assert "pytest" not in feature.get("tasks", {}), (
            f"The {feature_name} feature defines a pytest task that shadows environment executables."
        )

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
        "media",
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
    media_feature = features["media"]
    media_dependencies = media_feature["dependencies"]

    assert "channels" not in core_feature
    assert core_pypi_dependencies["cosmos-xenna"] == "==0.5.11"
    assert core_pypi_dependencies["ray"] == {"version": "==2.58.0", "extras": ["default", "data"]}
    for dependency_name in ("fastapi", "starlette", "uvicorn", "websockets"):
        assert dependency_name in core_dependencies
    for dependency_name in ("google-genai", "webdataset"):
        assert dependency_name in core_pypi_dependencies
    for dependency_name in ("torch", "torchvision", "vllm", "cvcuda-cu13", "PyNvVideoCodec"):
        assert dependency_name not in core_pypi_dependencies

    assert runtime_feature["channels"] == ["conda-forge", "nvidia"]
    assert runtime_feature["platforms"] == ["linux-64-cuda", "linux-aarch64-cuda"]
    assert "system-requirements" not in runtime_feature
    for dependency_name in ("torch", "torchvision", "vllm", "cvcuda-cu13", "PyNvVideoCodec"):
        assert dependency_name in runtime_pypi_dependencies
    for dependency_name in ("transformers-cosmos3", "vllm-cosmos3"):
        assert dependency_name not in runtime_pypi_dependencies
    for dependency_name in ("av", "opencv-python-headless", "opencv-python", "opencv-contrib-python"):
        assert dependency_name not in runtime_pypi_dependencies

    assert media_feature["channels"] == ["conda-forge"]
    assert "platforms" not in media_feature
    assert media_dependencies["av"] == "==17.0.0"
    assert media_dependencies["ffmpeg"] == {"version": "==8.1.2", "build": "lgpl_*"}, (
        "FFmpeg must stay exactly pinned to the LGPL build in pixi.toml. "
        "When changing this version, keep pixi.toml [feature.media.dependencies.ffmpeg], "
        "package/cosmos_curator/default.dockerfile.jinja2 ENV FFMPEG_VERSION, and pixi.lock in sync."
    )
    for dependency_name in ("libopencv", "opencv", "py-opencv"):
        assert media_dependencies[dependency_name]["build"] == "headless_*"

    assert "unified" not in features
    assert pixi_config["environments"]["default"] == [
        "core",
        "runtime",
        "media",
        "transformers",
        "tracing",
        "profiling",
    ]


def test_vllm_omni_is_isolated_to_style_transfer_environment() -> None:
    """Verify transfer-only dependencies do not leak into shared runtime environments."""
    pixi_config = tomllib.loads(_read_repo_file("pixi.toml"))
    features = pixi_config["feature"]

    assert "vllm-omni" not in features["runtime"]["pypi-dependencies"]
    assert features["vllm-omni"]["pypi-dependencies"] == {"vllm-omni": "==0.30.0"}
    assert pixi_config["environments"]["style-transfer"] == [
        "core",
        "runtime",
        "media",
        "transformers",
        "vllm-omni",
        "tracing",
        "profiling",
    ]
    for environment_name in ("default", "seedvr", "sam3"):
        assert "vllm-omni" not in pixi_config["environments"][environment_name]


def test_legacy_transformers_environment_is_model_specific_runtime() -> None:
    """Verify legacy models avoid the main runtime's newer transformers stack."""
    pixi_config = tomllib.loads(_read_repo_file("pixi.toml"))

    legacy_feature = pixi_config["feature"]["legacy-transformers"]
    legacy_pypi_dependencies = legacy_feature["pypi-dependencies"]

    assert legacy_feature["dependencies"]["transformers"] == "==4.55.4"
    for dependency_name in ("timm", "torch", "torchvision"):
        assert dependency_name in legacy_pypi_dependencies
    for dependency_name in ("av", "opencv-python-headless"):
        assert dependency_name not in legacy_pypi_dependencies
    assert "transformers-cosmos3" not in legacy_pypi_dependencies
    assert "vllm" not in legacy_pypi_dependencies
    assert pixi_config["environments"]["legacy-transformers"] == [
        "core",
        "media",
        "legacy-transformers",
        "tracing",
        "profiling",
    ]
    assert "runtime" not in pixi_config["environments"]["legacy-transformers"]


def test_distributable_pixi_manifest_is_generated_runtime_subset() -> None:
    """Verify the redistributable Pixi manifest keeps custom-media placeholders isolated."""
    result = subprocess.run(
        [sys.executable, "tools/update_distributable_pixi.py", "--check"],
        cwd=_REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout

    pixi_config = tomllib.loads(_read_repo_file("distributable/pixi.toml"))
    media_feature = pixi_config["feature"]["media"]

    assert pixi_config["workspace"]["name"] == "cosmos-curator-distributable"
    assert "conda-pypi-map" not in pixi_config["workspace"]
    assert set(pixi_config["environments"]) == {
        "cuml",
        "default",
        "legacy-transformers",
        "model-download",
        "paddle-ocr",
        "seedvr",
        "sam3",
        "style-transfer",
    }
    for feature_names in pixi_config["environments"].values():
        assert not {"tools", "cluster", "dev"} & set(feature_names)

    assert "dependencies" not in media_feature
    assert media_feature["pypi-dependencies"] == {
        "av": "==17.0.0",
        "opencv-python-headless": "*",
        "cython": "<3.3",
    }


def test_media_policy_matches_root_and_distributable_locks() -> None:
    """Verify normal and redistributable locks keep distinct media policies."""
    pixi_config = tomllib.loads(_read_repo_file("pixi.toml"))
    dockerfile = _read_repo_file("package/cosmos_curator/default.dockerfile.jinja2")
    root_lock = _read_repo_file("pixi.lock")
    distributable_lock = _read_repo_file("distributable/pixi.lock")
    ffmpeg_version = pixi_config["feature"]["media"]["dependencies"]["ffmpeg"]["version"].removeprefix("==")
    dockerfile_match = re.search(r"^ENV FFMPEG_VERSION=([^\s]+)$", dockerfile, flags=re.MULTILINE)

    assert dockerfile_match is not None, (
        "Missing package/cosmos_curator/default.dockerfile.jinja2 ENV FFMPEG_VERSION; "
        "keep it in sync with pixi.toml [feature.media.dependencies.ffmpeg]."
    )
    assert dockerfile_match.group(1) == ffmpeg_version, (
        "FFmpeg versions are out of sync: update package/cosmos_curator/default.dockerfile.jinja2 "
        "ENV FFMPEG_VERSION to match pixi.toml [feature.media.dependencies.ffmpeg]."
    )
    assert f"ffmpeg-{ffmpeg_version}-lgpl_" in root_lock, (
        "pixi.lock does not contain the LGPL FFmpeg build matching pixi.toml. "
        "Refresh pixi.lock after updating pixi.toml [feature.media.dependencies.ffmpeg]."
    )
    assert f"ffmpeg-{ffmpeg_version}-gpl_" not in root_lock, (
        "pixi.lock contains the GPL FFmpeg build; keep media FFmpeg on the LGPL build in pixi.toml "
        "and refresh pixi.lock."
    )
    assert "opencv_python-" not in root_lock
    assert "av-17.0.0-cp" not in root_lock

    assert "conda-forge/linux-64/ffmpeg-" not in distributable_lock
    assert "conda-forge/linux-aarch64/ffmpeg-" not in distributable_lock
    assert "openh264" not in distributable_lock
    assert "opencv_python_headless-" in distributable_lock
    assert "av-17.0.0-cp" in distributable_lock


def test_cuml_owns_rapids_channel() -> None:
    """Verify direct RAPIDS dependencies are scoped to cuML."""
    pixi_config = tomllib.loads(_read_repo_file("pixi.toml"))

    features = pixi_config["feature"]
    assert pixi_config["workspace"]["channels"][0] == "rapidsai"
    assert features["cuml"]["channels"] == ["nvidia"]
    assert features["cuml"]["platforms"] == ["linux-64-cuda", "linux-aarch64-cuda"]
    assert "system-requirements" not in features["cuml"]
    assert pixi_config["environments"]["cuml"] == ["core", "cuml", "tracing"]
    assert features["cuml"]["dependencies"]["cuml"] == "==26.04"
    assert "raft-dask" in features["cuml"]["dependencies"]


def test_model_download_environment_reuses_core() -> None:
    """Verify model download workers get core dependencies, including Ray."""
    pixi_config = tomllib.loads(_read_repo_file("pixi.toml"))

    assert "model-download" not in pixi_config["feature"]
    assert pixi_config["environments"]["model-download"] == ["linux", "core", "tracing"]


def test_gpu_test_task_is_defined_on_core() -> None:
    """Verify the GPU/env test task lives on `core` so every runtime env has it."""
    pixi_config = tomllib.loads(_read_repo_file("pixi.toml"))
    core_tasks = pixi_config.get("feature", {}).get("core", {}).get("tasks")
    assert isinstance(core_tasks, dict)

    gpu_test = core_tasks["gpu-test"]
    assert isinstance(gpu_test, str)
    assert "pytest -m env" in gpu_test
    assert "gpu-test" in core_tasks


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
    assert (
        hooks_by_id["distributable-pixi-check"]["entry"]
        == "pixi run --frozen -e tools python tools/update_distributable_pixi.py --check"
    )

    devset_script = _read_repo_file("devset.sh")
    assert "pixi run -e tools pre-commit install" in devset_script
    assert "pixi run -e tools python -m build" in devset_script
    assert "dev-hooks" not in devset_script


def test_distributable_pixi_ci_check_uses_generator() -> None:
    """Verify CI uses the generator as the manifest and lockfile check owner."""
    distributable_pixi_job = _read_ci_job("distributable_pixi_check")
    script = _script_lines(distributable_pixi_job["script"])
    script_text = "\n".join(script)

    assert "python tools/update_distributable_pixi.py --check" in script_text
    assert "pixi lock --manifest-path distributable/pixi.toml --check" not in script_text


def test_slurm_end_to_end_uses_pixi_cluster_for_submit_cli() -> None:
    """Verify the host-side Slurm submit CLI runs from Pixi's cluster environment."""
    slurm_job = _read_ci_job("slurm_end_to_end")
    before_script = _script_lines(slurm_job["before_script"])
    script = _script_lines(slurm_job["script"])
    after_script = _script_lines(slurm_job["after_script"])
    submit_script = _read_repo_file(".gitlab/scripts/slurm_end_to_end.sh")
    commands = "\n".join([*before_script, *script])
    pixi_cache_index = next(
        index
        for index, command in enumerate(before_script)
        if 'PIXI_CACHE_DIR="${SLURM_E2E_PIXI_CACHE_DIR}"' in command
    )
    pixi_setup_index = next(
        index for index, command in enumerate(before_script) if "pixi install --frozen -e cluster" in command
    )

    assert pixi_cache_index < pixi_setup_index
    assert slurm_job["variables"]["CI_PIXI_BIN"] == "/lustre/fsw/coreai_dlalgo_ci/nemo_video_curator/pixi/bin/pixi"
    assert slurm_job["variables"]["SLURM_E2E_PIXI_CACHE_DIR"] == (
        "/lustre/fsw/coreai_dlalgo_ci/nemo_video_curator/pixi/cache"
    )
    assert "CI_PIXI_BIN does not point to an executable Pixi binary" in commands
    assert "pixi.sh/install.sh" not in commands
    assert "pixi install --frozen -e cluster" in commands
    assert ".gitlab/scripts/slurm_end_to_end.sh" in script
    assert 'rm -rf "${CI_PROJECT_DIR}/.pixi"' in after_script
    assert "pip install -e ." not in commands
    assert "source venv/bin/activate" not in commands
    assert "uv venv" not in commands
    assert '--pixi-envs "default,cuml,legacy-transformers,model-download"' in submit_script

    smoke_job = _read_ci_job("slurm_distributable_media_smoke")
    smoke_commands = "\n".join(_script_lines(smoke_job["before_script"]))
    assert smoke_job["variables"]["CI_PIXI_BIN"] == "/lustre/fsw/coreai_dlalgo_ci/nemo_video_curator/pixi/bin/pixi"
    assert "CI_PIXI_BIN does not point to an executable Pixi binary" in smoke_commands
    assert "pixi.sh/install.sh" not in smoke_commands
    assert "pixi install --frozen -e cluster" in smoke_commands


def test_nvcf_split_benchmark_runs_as_package_module() -> None:
    """Verify the NVCF split benchmark preserves repo-root imports."""
    script = _read_repo_file(".gitlab/scripts/nvcf_split_benchmark.sh")
    ci_config = yaml.safe_load(_read_repo_file(".gitlab-ci.yml"))

    assert "python -m benchmarks.split_pipeline.nvcf_split_benchmark" in script
    assert "python benchmarks/split_pipeline/nvcf_split_benchmark.py" not in script
    assert 'MAX_ATTEMPTS="${NVCF_SPLIT_BENCHMARK_MAX_ATTEMPTS:-4}"' in script
    assert '--max-attempts "${MAX_ATTEMPTS}"' in script
    assert ci_config["variables"]["NVCF_SPLIT_BENCHMARK_MAX_ATTEMPTS"]["value"] == "4"
    assert "NVCF_SPLIT_BENCHMARK_OTLP_ENABLED" not in ci_config["variables"]
    for signal in ("METRICS", "TRACES", "LOGS"):
        variable = f"NVCF_SPLIT_BENCHMARK_OTLP_{signal}_ENABLED"
        assert ci_config["variables"][variable]["value"] == "False"
        assert variable in script
    assert ci_config["variables"]["NVCF_SPLIT_BENCHMARK_OTLP_NVCF_MTLS_ENABLED"]["value"] == "True"


def test_nvcf_helm_deploy_invokes_without_status_logs() -> None:
    """Verify helm CI invokes do not download status log payloads while polling."""
    script = _read_repo_file(".gitlab/scripts/nvcf_helm_deploy.sh")

    invoke_lines = [
        line for line in script.splitlines() if line.startswith("cosmos-curator nvcf function invoke-function")
    ]

    assert len(invoke_lines) == 4
    assert all("--no-include-logs" in line for line in invoke_lines)


def test_image_cli_default_envs_do_not_include_dev() -> None:
    """Verify image env parsing does not add the developer tooling environment by default."""
    default_envs = set(_parse_envs(""))
    configured_envs = inspect.signature(build_image).parameters["envs"].default
    assert isinstance(configured_envs, str)
    configured_runtime_envs = set(_parse_envs(configured_envs))

    for env_name in ("tools", "cluster", "dev"):
        assert env_name not in default_envs
        assert env_name not in configured_runtime_envs
    assert "style-transfer" in configured_runtime_envs
