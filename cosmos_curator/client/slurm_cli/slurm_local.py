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
"""Interactive Slurm launcher backed by srun/Pyxis."""

import logging
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from typer import Argument, BadParameter, Option

from cosmos_curator.client.environment import (
    CONTAINER_PATHS_CODE_DIR,
    CONTAINER_PATHS_COSMOS_CURATOR_CONFIG_FILE,
    CONTAINER_PATHS_DEFAULT_WORKSPACE_DIR,
    LOCAL_AWS_CREDENTIALS_FILE,
    LOCAL_AZURE_CREDENTIALS_FILE,
    LOCAL_COSMOS_CURATOR_CONFIG_FILE,
    LOCAL_WORKSPACE_PATH,
    SLURM_RAY_ENV_VAR_NAME,
)
from cosmos_curator.client.utils.container_launch import SLIM_IMAGE_WARMUP_COMMAND, command_contains, parse_extra_mounts

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
_SLURM_ALLOCATION_ENV_VARS = ("SLURM_JOB_ID", "SLURM_JOBID")
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
class LaunchSlurmLocal:
    """Configuration for launching a command inside an interactive Slurm allocation."""

    container_image: str
    curator_path: Path | None
    command: list[str]
    workspace_path: Path
    cache_path: Path
    mount_s3_creds: bool
    mount_azure_creds: bool
    extra_mounts: list[str]
    environment: list[str]
    require_slurm_allocation: bool
    conda_override_cuda: str | None
    pixi_envs: list[str] | None
    overlap: bool
    interactive: bool


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


def _get_credential_mounts(opts: LaunchSlurmLocal) -> list[str]:
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


def _get_srun_mounts(opts: LaunchSlurmLocal) -> list[str]:
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


def _get_srun_environment(opts: LaunchSlurmLocal) -> tuple[dict[str, str], list[str]]:
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

    container_env_keys.extend(env_var for env_var in _SLURM_ENV_VARS_TO_FORWARD if env_var in env)
    return env, _dedupe(container_env_keys)


def _resolve_container_image(container_image: str) -> str:
    path = Path(container_image).expanduser()
    if container_image.startswith("~") or path.exists():
        return str(path)
    return container_image


def _verify_slurm_allocation(opts: LaunchSlurmLocal) -> None:
    if not opts.require_slurm_allocation:
        return
    if any(env_var in os.environ for env_var in _SLURM_ALLOCATION_ENV_VARS):
        return
    logger.error(
        "This command is intended to run inside an interactive Slurm allocation. "
        "Use --no-require-slurm-allocation to override this guard."
    )
    sys.exit(1)


def _launch_with_srun(opts: LaunchSlurmLocal) -> None:
    if not opts.command:
        msg = "A command must be provided"
        raise ValueError(msg)

    _verify_slurm_allocation(opts)

    subprocess_env, container_env_keys = _get_srun_environment(opts)
    container_command = (
        f"cd {shlex.quote(str(CONTAINER_PATHS_CODE_DIR))} && "
        f"{_get_source_link_command()} && "
        f'{SLIM_IMAGE_WARMUP_COMMAND} && exec "$@"'
    )
    srun_command = [
        "srun",
        "--mpi=none",
    ]
    if opts.overlap:
        srun_command.append("--overlap")
    if opts.interactive:
        srun_command.append("--pty")

    srun_command.extend(
        [
            "--nodes=1",
            "--ntasks=1",
            "--container-writable",
            "--no-container-mount-home",
            "--no-container-remap-root",
            "--container-image",
            _resolve_container_image(opts.container_image),
            "--container-mounts",
            ",".join(_get_srun_mounts(opts)),
            "--container-env",
            ",".join(container_env_keys),
            "bash",
            "-c",
            container_command,
            "_",
            *opts.command,
        ]
    )
    logger.info("Slurm command:\n%s", shlex.join(srun_command))

    result = subprocess.call(srun_command, shell=False, env=subprocess_env)  # noqa: S603
    if result != 0:
        logger.error("Failed to run command via srun")
        sys.exit(1)


def launch_cli(  # noqa: PLR0913
    *,
    command: Annotated[list[str], Argument(help="The command to run", rich_help_panel="common")],
    container_image: Annotated[
        str,
        Option(
            "--container-image",
            help="Path to the .sqsh image for srun/Pyxis.",
            rich_help_panel="container",
        ),
    ] = _DEFAULT_CONTAINER_IMAGE,
    curator_path: Annotated[
        Path | None,
        Option(
            help="Path to the cosmos-curator repo directory; set to mount live curator code into the container.",
            rich_help_panel="interactive-slurm",
        ),
    ] = None,
    workspace_path: Annotated[
        Path,
        Option(
            help="Host workspace directory to mount as /config inside the container.",
            rich_help_panel="interactive-slurm",
        ),
    ] = LOCAL_WORKSPACE_PATH,
    cache_path: Annotated[
        Path,
        Option(
            help="Host cache directory to mount as /cache for Pixi/rattler, uv, Torch, Triton, pip, and CUDA caches.",
            rich_help_panel="interactive-slurm",
        ),
    ] = _DEFAULT_CACHE_PATH,
    mount_s3_creds: Annotated[
        bool,
        Option(
            "--mount-s3-creds/--no-mount-s3-creds",
            help="Mount the host AWS credentials file into the container when present.",
            rich_help_panel="interactive-slurm",
        ),
    ] = True,
    mount_azure_creds: Annotated[
        bool,
        Option(
            "--mount-azure-creds/--no-mount-azure-creds",
            help="Mount the host Azure credentials file into the container when present.",
            rich_help_panel="interactive-slurm",
        ),
    ] = False,
    extra_mounts: Annotated[
        str,
        Option(
            "--extra-mounts",
            "--extra-volumes",
            help=(
                "Comma-separated extra container mounts in HOST_PATH:CONTAINER_PATH format, "
                "for example /data/models:/config/models,/data/videos:/workspace/input"
            ),
            rich_help_panel="interactive-slurm",
        ),
    ] = "",
    environment: Annotated[
        str | None,
        Option(
            help="Comma-separated list of additional environment variables to set in the container.",
            rich_help_panel="interactive-slurm",
        ),
    ] = None,
    require_slurm_allocation: Annotated[
        bool,
        Option(
            "--require-slurm-allocation/--no-require-slurm-allocation",
            help="Require SLURM_JOB_ID or SLURM_JOBID to be present before starting srun.",
            rich_help_panel="interactive-slurm",
        ),
    ] = True,
    conda_override_cuda: Annotated[
        str | None,
        Option(
            help="Set CONDA_OVERRIDE_CUDA during Pixi warmup. Use an empty value to omit it.",
            rich_help_panel="interactive-slurm",
        ),
    ] = _DEFAULT_CONDA_OVERRIDE_CUDA,
    pixi_envs: Annotated[
        str | None,
        Option(
            "--pixi-envs",
            help=(
                "Comma-separated Pixi environments to install during slim-image warmup, overriding "
                "COSMOS_CURATOR_SLIM_ENVS from the image."
            ),
            rich_help_panel="interactive-slurm",
        ),
    ] = None,
    overlap: Annotated[
        bool,
        Option(
            "--overlap/--no-overlap",
            help="Pass --overlap to srun so nested launches from an srun --pty shell can reuse the allocation.",
            rich_help_panel="interactive-slurm",
        ),
    ] = True,
    interactive: Annotated[
        bool,
        Option(
            "--interactive/--no-interactive",
            help="Pass --pty to srun for interactive commands such as bash.",
            rich_help_panel="interactive-slurm",
        ),
    ] = False,
) -> None:
    """Launch a command with srun/Pyxis inside an existing interactive Slurm allocation."""
    cuda_override = conda_override_cuda if conda_override_cuda else None
    opts = LaunchSlurmLocal(
        container_image=container_image,
        curator_path=curator_path,
        command=command,
        workspace_path=workspace_path,
        cache_path=cache_path,
        mount_s3_creds=mount_s3_creds,
        mount_azure_creds=mount_azure_creds,
        extra_mounts=parse_extra_mounts(extra_mounts, description="extra mount"),
        environment=_parse_environment(environment),
        require_slurm_allocation=require_slurm_allocation,
        conda_override_cuda=cuda_override,
        pixi_envs=_parse_pixi_envs(pixi_envs),
        overlap=overlap,
        interactive=interactive,
    )
    _launch_with_srun(opts)
