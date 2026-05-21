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
"""Shared helpers for Slurm container commands backed by srun/Pyxis."""

import logging
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

from typer import BadParameter

from cosmos_curator.client.environment import (
    CONTAINER_PATHS_CODE_DIR,
    CONTAINER_PATHS_COSMOS_CURATOR_CONFIG_FILE,
    CONTAINER_PATHS_DEFAULT_WORKSPACE_DIR,
    LOCAL_AWS_CREDENTIALS_FILE,
    LOCAL_AZURE_CREDENTIALS_FILE,
    LOCAL_COSMOS_CURATOR_CONFIG_FILE,
    SLURM_RAY_ENV_VAR_NAME,
)
from cosmos_curator.client.utils.container_launch import SLIM_IMAGE_WARMUP_COMMAND, command_contains

logger = logging.getLogger(__name__)

_CACHE_MOUNT_PATH = Path("/cache")
_CONTAINER_AZURE_CREDS_PATH = Path("/creds/azure_creds")
_CONTAINER_SOURCE_DIR = Path("/src/cosmos-curator")
_CONTAINER_S3_CREDS_PATH = Path("/creds/s3_creds")
_DEFAULT_CACHE_PATH = Path("~/.cache").expanduser()
_DEFAULT_CONTAINER_IMAGE = "~/container_images/cosmos-curator+1.0.0.sqsh"
_DEFAULT_CONDA_OVERRIDE_CUDA = "13.0.2"
_SOURCE_DIRNAMES = ("cosmos_curator", "tools")
_SOURCE_FILENAMES = ("pixi.toml", "pixi.lock", "pyproject.toml", "pytest.ini", ".coveragerc")
_SLURM_ENV_VARS_TO_FORWARD = (
    "SLURM_JOB_ID",
    "SLURM_JOBID",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_NUM_NODES",
    "SLURM_NNODES",
    "SLURM_NTASKS_PER_NODE",
    "SLURMD_NODENAME",
)


@dataclass
class SlurmContainerRuntime:
    """Shared container runtime configuration for Slurm commands."""

    container_image: str
    curator_path: Path | None
    command: list[str]
    workspace_path: Path
    cache_path: Path
    mount_s3_creds: bool
    mount_azure_creds: bool
    extra_mounts: list[str]
    environment: list[str]
    conda_override_cuda: str | None
    pixi_envs: list[str] | None


@dataclass
class SrunCommand:
    """Concrete srun command plus the environment it should inherit."""

    command: list[str]
    environment: dict[str, str]
    container_env_keys: list[str]


def _parse_environment(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def _parse_pixi_envs(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    envs = [entry.strip() for entry in raw.split(",") if entry.strip()]
    if not envs:
        msg = "--pixi-envs must include at least one Pixi environment"
        raise BadParameter(msg)
    return envs


def _mount_string(source: Path | str, dest: Path | str, mode: str = "rw") -> str:
    mount = f"{source}:{dest}"
    if mode != "rw":
        mount += f":{mode}"
    return mount


def _resolve_existing_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _get_code_mounts(curator_path: Path | None) -> list[str]:
    if curator_path is None:
        return []

    root = _resolve_existing_path(curator_path)
    package_path = root / "cosmos_curator"
    if not package_path.is_dir():
        logger.error("Curator package directory does not exist at %s", package_path)
        sys.exit(1)

    return [_mount_string(root, _CONTAINER_SOURCE_DIR)]


def _get_workspace_mount(workspace_path: Path) -> str:
    workspace = workspace_path.expanduser()
    workspace.mkdir(parents=True, exist_ok=True)
    return _mount_string(workspace.resolve(), CONTAINER_PATHS_DEFAULT_WORKSPACE_DIR)


def _get_cache_mount(cache_path: Path) -> str:
    cache = cache_path.expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    for subdir in ("rattler/cache/uv-cache", "pip", "torch", "triton", "nv/ComputeCache"):
        (cache / subdir).mkdir(parents=True, exist_ok=True)
    return _mount_string(cache.resolve(), _CACHE_MOUNT_PATH)


def _get_credential_mounts(opts: SlurmContainerRuntime) -> list[str]:
    mounts: list[str] = []
    if opts.mount_s3_creds:
        if LOCAL_AWS_CREDENTIALS_FILE.exists():
            mounts.append(_mount_string(LOCAL_AWS_CREDENTIALS_FILE, _CONTAINER_S3_CREDS_PATH, mode="ro"))
        else:
            logger.warning(
                "No AWS credentials file found at %s; S3 operations will not work",
                LOCAL_AWS_CREDENTIALS_FILE,
            )

    if opts.mount_azure_creds:
        if LOCAL_AZURE_CREDENTIALS_FILE.exists():
            mounts.append(_mount_string(LOCAL_AZURE_CREDENTIALS_FILE, _CONTAINER_AZURE_CREDS_PATH, mode="ro"))
        else:
            logger.warning(
                "No Azure credentials file found at %s; Azure operations will not work", LOCAL_AZURE_CREDENTIALS_FILE
            )

    return mounts


def _get_config_mounts(*, is_model_cli: bool) -> list[str]:
    if LOCAL_COSMOS_CURATOR_CONFIG_FILE.exists():
        return [
            _mount_string(
                LOCAL_COSMOS_CURATOR_CONFIG_FILE,
                CONTAINER_PATHS_COSMOS_CURATOR_CONFIG_FILE,
                mode="ro",
            )
        ]

    logger.warning("No config file found at %s", LOCAL_COSMOS_CURATOR_CONFIG_FILE)
    logger.warning("Model download and database operation will not work")
    if is_model_cli:
        sys.exit(1)
    return []


def _get_srun_mounts(opts: SlurmContainerRuntime) -> list[str]:
    return [
        _get_workspace_mount(opts.workspace_path),
        _get_cache_mount(opts.cache_path),
        *_get_code_mounts(opts.curator_path),
        *_get_credential_mounts(opts),
        *_get_config_mounts(is_model_cli=command_contains(opts.command, "model_cli")),
        *opts.extra_mounts,
    ]


def _get_cache_environment() -> dict[str, str]:
    pixi_cache_dir = _CACHE_MOUNT_PATH / "rattler" / "cache"
    return {
        "PIXI_CACHE_DIR": str(pixi_cache_dir),
        "RATTLER_CACHE_DIR": str(pixi_cache_dir),
        "XDG_CACHE_HOME": str(_CACHE_MOUNT_PATH),
        "UV_CACHE_DIR": str(pixi_cache_dir / "uv-cache"),
        "PIP_CACHE_DIR": str(_CACHE_MOUNT_PATH / "pip"),
        "TORCH_HOME": str(_CACHE_MOUNT_PATH / "torch"),
        "TRITON_HOME": str(_CACHE_MOUNT_PATH / "triton"),
        "CUDA_CACHE_PATH": str(_CACHE_MOUNT_PATH / "nv" / "ComputeCache"),
    }


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _link_source_entry_command(source: Path, dest: Path, *, is_dir: bool) -> str:
    test_flag = "-d" if is_dir else "-f"
    return (
        f"if [ {test_flag} {shlex.quote(str(source))} ]; then "
        f"mkdir -p {shlex.quote(str(dest.parent))} && "
        f"rm -rf {shlex.quote(str(dest))} && "
        f"ln -s {shlex.quote(str(source))} {shlex.quote(str(dest))}; "
        "fi"
    )


def _get_source_link_command() -> str:
    dest_dir = shlex.quote(str(CONTAINER_PATHS_CODE_DIR))
    commands = [f"mkdir -p {dest_dir}"]
    commands.extend(
        _link_source_entry_command(
            _CONTAINER_SOURCE_DIR / dirname,
            CONTAINER_PATHS_CODE_DIR / dirname,
            is_dir=True,
        )
        for dirname in _SOURCE_DIRNAMES
    )
    commands.append(
        _link_source_entry_command(
            _CONTAINER_SOURCE_DIR / "tests" / "cosmos_curator",
            CONTAINER_PATHS_CODE_DIR / "tests" / "cosmos_curator",
            is_dir=True,
        )
    )
    commands.extend(
        _link_source_entry_command(
            _CONTAINER_SOURCE_DIR / filename,
            CONTAINER_PATHS_CODE_DIR / filename,
            is_dir=False,
        )
        for filename in _SOURCE_FILENAMES
    )
    return " && ".join(commands)


def _get_srun_environment(
    opts: SlurmContainerRuntime, *, include_slurm_env: bool = True
) -> tuple[dict[str, str], list[str]]:
    env = os.environ.copy()
    container_env = {
        SLURM_RAY_ENV_VAR_NAME: "True",
        "COSMOS_S3_PROFILE_PATH": str(_CONTAINER_S3_CREDS_PATH),
        "COSMOS_AZURE_PROFILE_PATH": str(_CONTAINER_AZURE_CREDS_PATH),
        "NVCF_REQUEST_STATUS": "false",
        "TQDM_MININTERVAL": "9000",
        **_get_cache_environment(),
    }
    if opts.conda_override_cuda is not None:
        container_env["CONDA_OVERRIDE_CUDA"] = opts.conda_override_cuda

    env.update(container_env)
    container_env_keys = list(container_env)

    for entry in opts.environment:
        if "=" in entry:
            key, value = entry.split("=", 1)
            env[key] = value
            container_env_keys.append(key)
        elif entry in env:
            container_env_keys.append(entry)
        else:
            logger.warning("Environment variable %s is not set; not forwarding it to the container", entry)

    if opts.pixi_envs is not None:
        env["COSMOS_CURATOR_SLIM_ENVS"] = ",".join(opts.pixi_envs)
        container_env_keys.append("COSMOS_CURATOR_SLIM_ENVS")

    if include_slurm_env:
        container_env_keys.extend(_SLURM_ENV_VARS_TO_FORWARD)
    return env, _dedupe(container_env_keys)


def _resolve_container_image(container_image: str) -> str:
    path = Path(container_image).expanduser()
    if container_image.startswith("~") or path.exists():
        return str(path)
    return container_image


def _get_container_entrypoint_command() -> str:
    code_dir = shlex.quote(str(CONTAINER_PATHS_CODE_DIR))
    return f'{_get_source_link_command()} && cd {code_dir} && {SLIM_IMAGE_WARMUP_COMMAND} && exec "$@"'


def _build_srun_command(
    opts: SlurmContainerRuntime,
    *,
    slurm_args: list[str] | None = None,
    container_mounts: list[str] | None = None,
    pty: bool = False,
) -> SrunCommand:
    if not opts.command:
        msg = "A command must be provided"
        raise ValueError(msg)

    subprocess_env, container_env_keys = _get_srun_environment(opts)
    srun_command = ["srun"]
    if pty:
        srun_command.append("--pty")
    if slurm_args is not None:
        srun_command.extend(slurm_args)

    srun_command.extend(
        [
            "--container-writable",
            "--no-container-mount-home",
            "--no-container-remap-root",
            "--container-image",
            _resolve_container_image(opts.container_image),
            "--container-mounts",
            ",".join(container_mounts if container_mounts is not None else _get_srun_mounts(opts)),
            "--container-env",
            ",".join(container_env_keys),
            "bash",
            "-c",
            _get_container_entrypoint_command(),
            "_",
            *opts.command,
        ]
    )
    return SrunCommand(command=srun_command, environment=subprocess_env, container_env_keys=container_env_keys)
