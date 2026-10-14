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
import pwd
import re
import shlex
import socket
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import attrs
from attrs import field, validators
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
from cosmos_curator.client.utils.container_launch import SLIM_IMAGE_WARMUP_COMMAND, command_contains, parse_extra_mounts

logger = logging.getLogger(__name__)

_CACHE_MOUNT_PATH = Path("/cache")
_CONTAINER_AZURE_CREDS_PATH = Path("/creds/azure_creds")
_CONTAINER_SOURCE_DIR = Path("/src/cosmos-curator")
_CONTAINER_S3_CREDS_PATH = Path("/creds/s3_creds")
_DEFAULT_CACHE_PATH = Path("~/.cache").expanduser()
_DEFAULT_CONTAINER_IMAGE = "~/container_images/cosmos-curator+1.0.0.sqsh"
_DEFAULT_CONDA_OVERRIDE_CUDA = "13.0.3"
_DEFAULT_LOGIN_NODE = "localhost"
_SLURM_ACCOUNT_ENV_VAR = "SBATCH_ACCOUNT"
_SLURM_MEMORY_PATTERN = re.compile(r"^[1-9][0-9]*[KMGT]?$")
_SOURCE_DIRNAMES = ("cosmos_curator", "tools")
_SOURCE_FILENAMES = ("pixi.toml", "pixi.lock", "pyproject.toml", "pytest.ini", ".coveragerc")
_PIXI_ACTIVATION_ENV_VARS = (
    "PIXI_ENVIRONMENT_NAME",
    "PIXI_ENVIRONMENT_PLATFORMS",
    "PIXI_EXE",
    "PIXI_IN_SHELL",
    "PIXI_PROJECT_MANIFEST",
    "PIXI_PROJECT_NAME",
    "PIXI_PROJECT_ROOT",
    "PIXI_PROJECT_VERSION",
    "PIXI_PROMPT",
)
_CONDA_ACTIVATION_ENV_VARS = (
    "CONDA_DEFAULT_ENV",
    "CONDA_EXE",
    "CONDA_PREFIX",
    "CONDA_PROMPT_MODIFIER",
    "CONDA_PYTHON_EXE",
    "CONDA_SHLVL",
    "_CE_CONDA",
    "_CE_M",
)
_LAUNCHER_ENV_PREFIXES = ("CONDA_ENV_SHLVL_", "CONDA_PREFIX_")
_SLURM_ENV_VARS_TO_FORWARD = (
    # Head and worker jobs pass this to `ray start --num-cpus` so a sliced job advertises its allocation rather than
    # whatever Ray autodetects from the host: three lanes sharing a 96-core node would otherwise each claim 96.
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_ID",
    "SLURM_JOBID",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_NUM_NODES",
    "SLURM_NNODES",
    "SLURM_NTASKS_PER_NODE",
    "SLURMD_NODENAME",
    "SLURM_RESTART_COUNT",
)
# Structured-logging and tracing toggles forwarded from the launching environment into
# the container (when set) so the Ray head and workers log identically.
_LOG_ENV_VARS_TO_FORWARD = (
    "PYTHON_LOG",
    "PYTHON_LOG_FORMAT",
    "PYTHON_LOG_TO_DRIVER",
    "PYTHON_LOG_RAY_LEVEL",
    "RAY_BACKEND_LOG_JSON",
    "CURATOR_RUN_ID",
    "COSMOS_CURATOR_PROFILE_TRACING",
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


@attrs.define
class MountSpec:
    """Represents a mount and its mount point."""

    source: str
    dest: str
    mode: str = field(default="rw", validator=validators.in_(["rw", "ro"]))

    @classmethod
    def from_str(cls, mount_str: str) -> Self:
        """Create a MountSpec instance from a string."""
        min_parts = 2
        max_parts = 3

        parts = mount_str.split(":")
        if len(parts) < min_parts or len(parts) > max_parts:
            error_message = f"`{mount_str}` must have between {min_parts} and {max_parts} colon-separated parts"
            raise ValueError(error_message)

        source = parts[0]
        dest = parts[1]
        mode = "rw" if len(parts) == min_parts else parts[2]

        return cls(source=source, dest=dest, mode=mode)

    def __str__(self) -> str:
        """Return a string representation of the mount suitable for use with Docker or Enroot."""
        return f"{self.source}:{self.dest}:{self.mode}"


def _get_username() -> str:
    """Retrieve the username of the current user."""
    uid = os.getuid()
    return pwd.getpwuid(uid).pw_name


def _infer_curator_path(curator_path: Path | None) -> Path | None:
    """Use the current checkout by default when a Slurm command is run from a repo root."""
    if curator_path is not None:
        return curator_path

    cwd = Path.cwd()
    if (cwd / "cosmos_curator").is_dir() and (cwd / "pixi.toml").is_file():
        return cwd
    return None


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


def _parse_mount_specs(raw: str | None) -> list[MountSpec]:
    if raw is None:
        return []
    return [MountSpec.from_str(entry.strip()) for entry in raw.split(",") if entry.strip()]


def _mount_specs_from_strings(mounts: list[str]) -> list[MountSpec]:
    return [MountSpec.from_str(mount) for mount in mounts]


def _merge_mount_specs_by_destination(mounts: list[MountSpec]) -> list[MountSpec]:
    merged: dict[str, MountSpec] = {}
    for mount in mounts:
        if mount.dest in merged:
            logger.warning(
                "Replacing duplicate container mount destination %s: %s -> %s",
                mount.dest,
                merged[mount.dest],
                mount,
            )
        merged[mount.dest] = mount
    return list(merged.values())


def _build_slurm_container_runtime(  # noqa: PLR0913
    *,
    command: list[str],
    container_image: str,
    curator_path: Path | None,
    workspace_path: Path,
    cache_path: Path,
    mount_s3_creds: bool,
    mount_azure_creds: bool,
    extra_mounts: str,
    environment: str | None,
    conda_override_cuda: str | None,
    pixi_envs: str | None,
) -> SlurmContainerRuntime:
    cuda_override = conda_override_cuda or None
    return SlurmContainerRuntime(
        container_image=container_image,
        curator_path=curator_path,
        command=command,
        workspace_path=workspace_path,
        cache_path=cache_path,
        mount_s3_creds=mount_s3_creds,
        mount_azure_creds=mount_azure_creds,
        extra_mounts=parse_extra_mounts(extra_mounts, description="extra mount"),
        environment=_parse_environment(environment),
        conda_override_cuda=cuda_override,
        pixi_envs=_parse_pixi_envs(pixi_envs),
    )


def _container_mount_specs_from_runtime(
    runtime: SlurmContainerRuntime, container_mounts: str | None = None
) -> list[MountSpec]:
    return _merge_mount_specs_by_destination(
        [
            *_mount_specs_from_strings(_get_srun_mounts(runtime)),
            *_parse_mount_specs(container_mounts),
        ]
    )


def _get_remote_srun_mounts(opts: SlurmContainerRuntime) -> list[str]:
    """Like _get_srun_mounts but without local filesystem side effects (no mkdir, no path validation).

    Used when the mount paths live on a remote login node rather than the local machine.
    """
    mounts = [
        _mount_string(opts.workspace_path.expanduser(), CONTAINER_PATHS_DEFAULT_WORKSPACE_DIR),
        _mount_string(opts.cache_path.expanduser(), _CACHE_MOUNT_PATH),
    ]
    if opts.curator_path is not None:
        mounts.append(_mount_string(opts.curator_path.expanduser(), _CONTAINER_SOURCE_DIR))
    if opts.mount_s3_creds:
        mounts.append(_mount_string(LOCAL_AWS_CREDENTIALS_FILE, _CONTAINER_S3_CREDS_PATH, mode="ro"))
    if opts.mount_azure_creds:
        mounts.append(_mount_string(LOCAL_AZURE_CREDENTIALS_FILE, _CONTAINER_AZURE_CREDS_PATH, mode="ro"))
    if command_contains(opts.command, "model_cli"):
        mounts.append(
            _mount_string(
                LOCAL_COSMOS_CURATOR_CONFIG_FILE,
                CONTAINER_PATHS_COSMOS_CURATOR_CONFIG_FILE,
                mode="ro",
            )
        )
    return [
        *mounts,
        *opts.extra_mounts,
    ]


def _remote_container_mount_specs_from_runtime(
    runtime: SlurmContainerRuntime, container_mounts: str | None = None
) -> list[MountSpec]:
    return _merge_mount_specs_by_destination(
        [
            *_mount_specs_from_strings(_get_remote_srun_mounts(runtime)),
            *_parse_mount_specs(container_mounts),
        ]
    )


def _normalize_optional_slurm_directive(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _is_valid_slurm_memory(value: str) -> bool:
    """Return whether a value is a positive Slurm memory size."""
    return _SLURM_MEMORY_PATTERN.fullmatch(value) is not None


def _resolve_slurm_account(account: str | None) -> str | None:
    """Resolve the account directive without making it mandatory for all clusters."""
    account = _normalize_optional_slurm_directive(account)
    if account is not None:
        return account

    env_account = os.getenv(_SLURM_ACCOUNT_ENV_VAR)
    return _normalize_optional_slurm_directive(env_account)


def _validate_gpu_options(
    *, gres: str | None, gpus: str | None, gpus_per_node: str | None
) -> tuple[str | None, str | None, str | None]:
    gres = _normalize_optional_slurm_directive(gres)
    gpus = _normalize_optional_slurm_directive(gpus)
    gpus_per_node = _normalize_optional_slurm_directive(gpus_per_node)
    requested_options = [
        flag
        for flag, value in (("--gres", gres), ("--gpus", gpus), ("--gpus-per-node", gpus_per_node))
        if value is not None
    ]
    if len(requested_options) > 1:
        msg = f"{' and '.join(requested_options)} cannot be used together"
        raise BadParameter(msg)
    return gres, gpus, gpus_per_node


def _is_local_host(remote_host: str) -> bool:
    """Determine whether the provided host refers to the current machine."""
    normalized = remote_host.lower()
    if normalized in {"localhost", "127.0.0.1"}:
        return True

    local_hostnames = {
        socket.gethostname().lower(),
        socket.getfqdn().lower(),
        os.uname().nodename.lower(),
    }
    if normalized in local_hostnames:
        return True

    try:
        remote_ip = socket.gethostbyname(remote_host)
        local_ip = socket.gethostbyname(socket.gethostname())
        if remote_ip == local_ip:
            return True
    except OSError:
        pass

    return False


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
    for subdir in ("rattler/cache/uv-cache", "pip", "torch", "triton", "nv/ComputeCache", "huggingface", "laion"):
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
        "HF_HOME": str(_CACHE_MOUNT_PATH / "huggingface"),
        "LAION_CACHE_HOME": str(_CACHE_MOUNT_PATH / "laion"),
    }


def _base_container_environment(
    *,
    conda_override_cuda: str | None,
    pixi_envs: Sequence[str] | None,
) -> dict[str, str]:
    """Return container values shared by Slurm submit, shell, and managed Ray jobs."""
    values = {
        SLURM_RAY_ENV_VAR_NAME: "True",
        # Bootstrap scripts are executed before cosmos_curator is installed in
        # every Pixi environment. Keep the source package importable.
        "PYTHONPATH": str(CONTAINER_PATHS_CODE_DIR),
        "COSMOS_S3_PROFILE_PATH": str(_CONTAINER_S3_CREDS_PATH),
        "COSMOS_AZURE_PROFILE_PATH": str(_CONTAINER_AZURE_CREDS_PATH),
        "NVCF_REQUEST_STATUS": "false",
        "TQDM_MININTERVAL": "9000",
        **_get_cache_environment(),
    }
    if conda_override_cuda is not None:
        values["CONDA_OVERRIDE_CUDA"] = conda_override_cuda
    if pixi_envs is not None:
        values["COSMOS_CURATOR_SLIM_ENVS"] = ",".join(pixi_envs)
    return values


def _resolve_forwarded_environment(
    entries: Sequence[str],
    *,
    source: Mapping[str, str],
) -> dict[str, str]:
    """Resolve ``NAME`` or ``NAME=value`` entries against a launching environment."""
    values: dict[str, str] = {}
    for entry in entries:
        if "=" in entry:
            key, value = entry.split("=", 1)
            values[key] = value
        elif entry in source:
            values[entry] = source[entry]
        else:
            logger.warning("Environment variable %s is not set; not forwarding it to the container", entry)
    return values


def _build_container_srun_argv(  # noqa: PLR0913
    *,
    container_image: str,
    container_mounts: Sequence[str],
    container_env_keys: Sequence[str],
    command: Sequence[str],
    slurm_args: Sequence[str] = (),
    pty: bool = False,
) -> list[str]:
    """Build the common Pyxis ``srun`` argument vector used by all Slurm launchers."""
    if not command:
        msg = "A command must be provided"
        raise ValueError(msg)

    srun_command = ["srun"]
    if pty:
        srun_command.append("--pty")
    srun_command.extend(slurm_args)
    srun_command.extend(
        [
            # Login-node shell limits can be smaller than the requested Slurm allocation.
            "--propagate=NONE",
            "--container-writable",
            "--no-container-mount-home",
            "--no-container-remap-root",
            "--container-image",
            _resolve_container_image(container_image),
            "--container-mounts",
            ",".join(container_mounts),
            "--container-env",
            ",".join(dict.fromkeys(container_env_keys)),
            "bash",
            "-c",
            _get_container_entrypoint_command(),
            "_",
            *command,
        ]
    )
    return srun_command


def _get_clean_subprocess_environment() -> dict[str, str]:
    """Return host environment without Pixi/Conda activation state from the launcher shell."""
    env = os.environ.copy()
    for key in (*_PIXI_ACTIVATION_ENV_VARS, *_CONDA_ACTIVATION_ENV_VARS):
        env.pop(key, None)
    for key in list(env):
        if key.startswith(_LAUNCHER_ENV_PREFIXES):
            env.pop(key)
    return env


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
    env = _get_clean_subprocess_environment()
    container_env = _base_container_environment(
        conda_override_cuda=opts.conda_override_cuda,
        pixi_envs=opts.pixi_envs,
    )

    env.update(container_env)
    container_env_keys = list(container_env)
    forwarded = _resolve_forwarded_environment(opts.environment, source=env)
    env.update(forwarded)
    container_env_keys.extend(forwarded)
    if opts.pixi_envs is not None:
        env["COSMOS_CURATOR_SLIM_ENVS"] = ",".join(opts.pixi_envs)

    # Forward structured-logging and tracing toggles present in the launching environment.
    container_env_keys.extend(name for name in _LOG_ENV_VARS_TO_FORWARD if name in env)

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
    subprocess_env, container_env_keys = _get_srun_environment(opts)
    srun_command = _build_container_srun_argv(
        container_image=opts.container_image,
        container_mounts=container_mounts if container_mounts is not None else _get_srun_mounts(opts),
        container_env_keys=container_env_keys,
        command=opts.command,
        slurm_args=slurm_args or (),
        pty=pty,
    )
    return SrunCommand(command=srun_command, environment=subprocess_env, container_env_keys=container_env_keys)
