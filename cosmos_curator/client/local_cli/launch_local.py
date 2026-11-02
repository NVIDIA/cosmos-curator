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

"""CLI to launch commands."""

import grp
import os
import pwd
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import psutil
import tomli
import typer
from loguru import logger
from typer import Argument, Option

from cosmos_curator.client.environment import (
    AZURE_PROFILE_PATH,
    CONTAINER_PATHS_CODE_DIR,
    CONTAINER_PATHS_COSMOS_CURATOR_CONFIG_FILE,
    CONTAINER_PATHS_DEFAULT_WORKSPACE_DIR,
    CURATOR_IO_SLOTS_PER_NODE_ENV_VAR,
    DEFAULT_CURATOR_IO_SLOTS_PER_NODE,
    LOCAL_AWS_CREDENTIALS_FILE,
    LOCAL_AZURE_CREDENTIALS_FILE,
    LOCAL_COSMOS_CURATOR_CONFIG_FILE,
    LOCAL_DOCKER_ENV_VAR_NAME,
    LOCAL_WORKSPACE_PATH,
    S3_PROFILE_PATH,
)
from cosmos_curator.client.image_cli.image_app import get_image_label
from cosmos_curator.client.utils.container_launch import SLIM_IMAGE_WARMUP_COMMAND, command_contains, parse_extra_mounts

cc_client_local = typer.Typer(
    help="Commands for building container image.",
    no_args_is_help=True,
)


@dataclass
class LaunchDocker:
    """Configuration class for launching Docker containers with specified parameters.

    This class holds the configuration needed to launch a Docker container, including
    image details, paths, and credential mounting options.
    """

    image_label: str
    curator_path: str | None
    pixi_path: str | None
    mount_xenna: bool
    command: str
    gpus: str | None
    docker_network: str | None
    mount_s3_creds: bool
    mount_azure_creds: bool
    extra_volumes: list[str]  # each entry is HOST_PATH:CONTAINER_PATH[:MODE]
    ray_io_slots_per_node: int


# Stable while we use nvidia/cuda:*-devel-ubuntu*: those base images ship a
# default `ubuntu:x:1000:1000` entry, so a host UID of 1000 already resolves
# inside the container and synthesis would only override `ubuntu` with the
# host username for no benefit. Revisit if the base image changes.
_IMAGE_BAKED_UIDS = frozenset({1000})


def _parse_extra_volumes(raw: str) -> list[str]:
    """Parse and validate comma-separated volume mount specifications.

    Each entry must be in ``HOST_PATH:CONTAINER_PATH`` or
    ``HOST_PATH:CONTAINER_PATH:MODE`` format.

    Raises:
        typer.BadParameter: If any entry is malformed.

    """
    return parse_extra_mounts(raw, description="volume mount")


@cc_client_local.command(no_args_is_help=True)
def launch(  # noqa: PLR0913
    *,
    command: Annotated[list[str], Argument(help="The command to run", rich_help_panel="common")],
    image_name: Annotated[
        str,
        Option(
            help=("The docker image name string to use."),
            rich_help_panel="container-image",
        ),
    ] = "cosmos-curator",
    image_tag: Annotated[
        str,
        Option(
            help=("The docker image tag to use."),
            rich_help_panel="container-image",
        ),
    ] = "1.0.0",
    curator_path: Annotated[
        str | None,
        Option(
            help=("Path to the cosmos-curator repo directory; set to mount local curator code into the container."),
            rich_help_panel="local-docker",
        ),
    ] = None,
    pixi_path: Annotated[
        str | None,
        Option(
            help=(
                "Path to a directory containing a .pixi subdirectory to mount into the container. "
                "For a pixi workspace (e.g. --pixi-path .), the command runs in that workspace at its host path, "
                "using the environments host pixi installed there and the workspace's code unless --curator-path "
                "names other code. Otherwise the .pixi directory replaces the image's, and slim images install "
                "their environments into it."
            ),
            rich_help_panel="local-docker",
        ),
    ] = None,
    mount_xenna: Annotated[
        bool,
        Option(
            help=(
                "Mount the local cosmos_xenna into the container; python code & default env only. "
                "WARNING: very hacky, for local development only."
            ),
            rich_help_panel="local-docker",
            is_flag=True,
        ),
    ] = False,
    gpus: Annotated[
        str | None,
        Option(
            help=("The GPUs to use for local-docker mode, e.g. `1 or 0,1`. If not specified, defaults to all."),
            rich_help_panel="local-docker",
        ),
    ] = None,
    docker_network: Annotated[
        str | None,
        Option(
            help=("Docker network to connect the container to, e.g. host, bridge, or a custom network name."),
            rich_help_panel="local-docker",
        ),
    ] = None,
    mount_s3_creds: Annotated[
        bool,
        Option(
            help=("Skip mounting the AWS credentials file into the container."),
            rich_help_panel="local-docker",
            is_flag=True,
        ),
    ] = True,
    mount_azure_creds: Annotated[
        bool,
        Option(
            help=("Mount the Azure credentials file into the container."),
            rich_help_panel="local-docker",
            is_flag=True,
        ),
    ] = False,
    extra_volumes: Annotated[
        str,
        Option(
            help=(
                "Comma-separated extra volume mounts in HOST_PATH:CONTAINER_PATH[:MODE] format. "
                "e.g. --extra-volumes /data/models:/config/models,/data/videos:/workspace/input:ro"
            ),
            rich_help_panel="local-docker",
        ),
    ] = "",
    ray_io_slots_per_node: Annotated[
        int,
        Option(
            "--ray-io-slots-per-node",
            help="Logical source-IO capacity advertised when the command starts a local Ray node.",
            rich_help_panel="local-docker",
            min=1,
        ),
    ] = DEFAULT_CURATOR_IO_SLOTS_PER_NODE,
) -> None:
    """Launch video-curation pipeline in local docker container.

    The function supports mounting AWS S3 and Azure credentials into the container,
    which can be controlled independently with the mount_s3_creds and mount_azure_creds
    flags. This allows using either S3 storage, Azure storage, or both together.
    """
    # Use ``shlex.join`` so argv tokens containing spaces survive the
    # round-trip into the container's ``bash -c`` string.
    command_str = shlex.join(command)

    opts = LaunchDocker(
        image_label=get_image_label(image_name, image_tag),
        curator_path=curator_path,
        pixi_path=pixi_path,
        mount_xenna=mount_xenna,
        command=command_str,
        gpus=gpus,
        docker_network=docker_network,
        mount_s3_creds=mount_s3_creds,
        mount_azure_creds=mount_azure_creds,
        extra_volumes=_parse_extra_volumes(extra_volumes),
        ray_io_slots_per_node=ray_io_slots_per_node,
    )
    return _launch_in_docker_container(opts)


def _verify_local_path_exists(local_paths: list[Path]) -> None:
    for local_path in local_paths:
        if not local_path.exists():
            logger.error(f"Local path {local_path} does not exist")
            sys.exit(1)


def _pause_for_warnings(timeout: int = 5) -> None:
    logger.info(f"Pausing for {timeout} seconds to show above warnings")
    time.sleep(timeout)


def _get_s3_creds_mount_strings(opts: LaunchDocker) -> list[str]:
    """Handle S3 credentials mounting."""
    s3_creds_strings = []
    if LOCAL_AWS_CREDENTIALS_FILE.exists():
        s3_creds_strings += [
            "-v",
            f"{LOCAL_AWS_CREDENTIALS_FILE}:{S3_PROFILE_PATH}",
        ]
    elif opts.mount_s3_creds:
        logger.warning(f"No AWS creds file found at {LOCAL_AWS_CREDENTIALS_FILE}; S3 operations will not work")
        _pause_for_warnings()
    return s3_creds_strings


def _get_azure_creds_mount_strings(opts: LaunchDocker) -> list[str]:
    """Handle Azure credentials mounting."""
    azure_creds_strings = []
    if LOCAL_AZURE_CREDENTIALS_FILE.exists():
        azure_creds_strings += [
            "-v",
            f"{LOCAL_AZURE_CREDENTIALS_FILE}:{AZURE_PROFILE_PATH}",
        ]
    elif opts.mount_azure_creds:
        logger.warning("No Azure creds file found at {LOCAL_AZURE_CREDENTIALS_FILE}; Azure operations will not work")
        _pause_for_warnings()
    return azure_creds_strings


def _get_config_file_mount_strings(*, is_model_cli: bool) -> list[str]:
    """Handle cosmos-curator config file mounting."""
    config_file_strings = []
    if LOCAL_COSMOS_CURATOR_CONFIG_FILE.exists():
        config_file_strings += [
            "-v",
            f"{LOCAL_COSMOS_CURATOR_CONFIG_FILE}:{CONTAINER_PATHS_COSMOS_CURATOR_CONFIG_FILE}",
        ]
    else:
        logger.warning(f"No config file found at {LOCAL_COSMOS_CURATOR_CONFIG_FILE}")
        logger.warning("Model download and database operation will not work")
        if is_model_cli:
            _verify_local_path_exists([LOCAL_COSMOS_CURATOR_CONFIG_FILE])
        else:
            _pause_for_warnings()
    return config_file_strings


def _get_python_version_from_pixi_toml(curator_path: Path) -> str | None:
    pixi_toml_path = curator_path / Path("pixi.toml")
    if not pixi_toml_path.exists():
        return None
    try:
        with pixi_toml_path.open("rb") as fp:
            pixi_toml = tomli.load(fp)
        python_version = pixi_toml["dependencies"]["python"]
    except (tomli.TOMLDecodeError, KeyError, AttributeError, ValueError) as e:
        logger.warning(f"Failed to parse python version from pixi.toml: {e}")
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Unexpected error reading pixi.toml: {e}")
        return None
    else:
        # validate the version format
        if not python_version or not isinstance(python_version, str):
            logger.warning("Python version not found or invalid in pixi.toml")
            return None
        # take the first major.minor version: "==3.13.14", ">=3.13,<3.14", and "3.13.*" all give 3.13
        version_match = re.search(r"\d+\.\d+", python_version)
        if version_match is None:
            logger.warning(f"Invalid python version format in pixi.toml: {python_version}")
            return None
        return version_match.group()


def _host_pixi_workspace(opts: LaunchDocker) -> Path | None:
    """Return the host pixi workspace that ``--pixi-path`` names, or None for a bare ``.pixi`` directory.

    Pixi records where it installed each environment and refuses to run in a
    workspace whose environments appear anywhere else, so the container must
    see a host workspace at its host path.
    """
    if opts.pixi_path is None:
        return None
    root = Path(opts.pixi_path).resolve()
    return root if (root / "pixi.toml").is_file() else None


def _prepare_workspace_mount_point(workspace: Path, *, home_dir: Path, scratch_home: Path) -> None:
    """Create a host workspace's mount point inside the scratch home mounted over ``$HOME``.

    Docker would otherwise create it as root on the host, or follow a link that
    earlier launches left on the path and mount the workspace at its target.
    """
    try:
        parts = workspace.relative_to(home_dir).parts
    except ValueError:
        return
    mount_point = scratch_home
    for part in parts:
        mount_point /= part
        if mount_point.is_symlink():
            mount_point.unlink()
    mount_point.mkdir(parents=True, exist_ok=True)


def _get_pixi_mount_strings(opts: LaunchDocker, workspace: Path | None) -> list[str]:
    if opts.pixi_path is None:
        return []
    pixi_root = Path(opts.pixi_path)
    pixi_dir = pixi_root / ".pixi"
    if not pixi_dir.is_dir():
        logger.error(f"Pixi directory not found at {pixi_dir}")
        sys.exit(1)
    if workspace is not None:
        return ["-v", f"{workspace}:{workspace}"]
    mount_strings = ["-v", f"{pixi_dir.absolute()}:{CONTAINER_PATHS_CODE_DIR / '.pixi'}"]
    # Also mount pixi.toml and pixi.lock so they stay in sync with the environments
    for name in ("pixi.toml", "pixi.lock"):
        host_file = pixi_root / name
        if host_file.is_file():
            mount_strings.extend(["-v", f"{host_file.absolute()}:{CONTAINER_PATHS_CODE_DIR / name}"])
    return mount_strings


def _get_code_mount_strings(opts: LaunchDocker, workspace: Path | None) -> list[str]:
    """Mount curator code where the image keeps it and, over a host workspace, where the command runs.

    A host workspace supplies its own code unless ``--curator-path`` names other code.
    """
    curator_path = Path(opts.curator_path) if opts.curator_path is not None else workspace
    if curator_path is None:
        return []
    curator_code_path = curator_path / Path("cosmos_curator")
    pipeline_code_path = curator_code_path / Path("pipelines")
    if not pipeline_code_path.exists():
        logger.error(f"Curator pipelines code does not exist at {pipeline_code_path}")
        sys.exit(1)
    code_dirs = [CONTAINER_PATHS_CODE_DIR]
    if workspace is not None and curator_path.resolve() != workspace:
        code_dirs.append(workspace)

    code_path_strings = []
    tests_path = curator_path / Path("tests") / Path("cosmos_curator")
    tools_path = curator_path / Path("tools")
    for code_dir in code_dirs:
        code_path_strings += ["-v", f"{curator_code_path.absolute()}:{code_dir}/cosmos_curator"]
        code_path_strings += ["-v", f"{tests_path.absolute()}:{code_dir}/tests/cosmos_curator"]
        if tools_path.is_dir():
            code_path_strings += ["-v", f"{tools_path.absolute()}:{code_dir}/tools"]

    if opts.mount_xenna:
        xenna_path = (curator_path / Path("cosmos-xenna") / Path("cosmos_xenna")).absolute()
        xenna_lib_path = _get_xenna_lib_path(opts, workspace, curator_path)
        if xenna_path.exists() and xenna_lib_path is not None:
            for python_module in ["pipelines", "ray_utils", "utils"]:
                code_path_strings += ["-v", f"{xenna_path / python_module}:{xenna_lib_path / python_module}"]
    return code_path_strings


def _get_xenna_lib_path(opts: LaunchDocker, workspace: Path | None, curator_path: Path) -> Path | None:
    """Return where the container's default environment installed ``cosmos_xenna``, or None if it can't be found.

    Docker creates a missing mount point as root, inside the host environment when
    ``--pixi-path`` supplies one, so a host environment is searched for the package.
    The image's environment can only be inferred from ``pixi.toml``.
    """
    if opts.pixi_path is None:
        python_version = _get_python_version_from_pixi_toml(curator_path)
        if python_version is None:
            return None
        return (
            CONTAINER_PATHS_CODE_DIR
            / Path(".pixi/envs/default/lib")
            / f"python{python_version}"
            / Path("site-packages/cosmos_xenna")
        )
    host_pixi_dir = Path(opts.pixi_path).absolute() / ".pixi"
    host_lib_paths = list(host_pixi_dir.glob("envs/default/lib/python*/site-packages/cosmos_xenna"))
    if len(host_lib_paths) != 1:
        logger.warning(
            f"Not mounting cosmos_xenna: expected one installed package under {host_pixi_dir / 'envs/default'}, "
            f"found {len(host_lib_paths)}"
        )
        return None
    # A workspace runs at its host path; a bare .pixi directory replaces the image's.
    container_pixi_dir = workspace / ".pixi" if workspace is not None else CONTAINER_PATHS_CODE_DIR / ".pixi"
    return container_pixi_dir / host_lib_paths[0].relative_to(host_pixi_dir)


# Structured-logging and tracing toggles forwarded from the host into the container (when
# set) so curator/xenna/Ray inside the container honor the same PYTHON_LOG_FORMAT.
_LOG_ENV_VARS_TO_FORWARD = (
    "PYTHON_LOG",
    "PYTHON_LOG_FORMAT",
    "PYTHON_LOG_TO_DRIVER",
    "PYTHON_LOG_RAY_LEVEL",
    "RAY_BACKEND_LOG_JSON",
    "CURATOR_RUN_ID",
    "COSMOS_CURATOR_PROFILE_TRACING",
)


def _get_log_env_forward_strings() -> list[str]:
    """Return docker ``-e VAR=value`` args for logging/tracing toggles present in the host env."""
    forward: list[str] = []
    for name in _LOG_ENV_VARS_TO_FORWARD:
        value = os.environ.get(name)
        if value is not None:
            forward.extend(["-e", f"{name}={value}"])
    return forward


def _get_system_memory_gb() -> float:
    mem = psutil.virtual_memory()
    return mem.total / (1024**3)


def _get_shm_size_str() -> str:
    default_proportion = 0.4
    mem_proportion_str = os.environ.get("RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION", str(default_proportion))
    try:
        fraction = float(mem_proportion_str)
    except ValueError:
        logger.warning(
            f"Found RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION in env, but value must be a float. "
            f"Got: {mem_proportion_str}. Using default 0.4."
        )
        fraction = default_proportion
    return f"{_get_system_memory_gb() * fraction:.2f}gb"


def _get_identity_mounts(scratch_home: Path) -> list[str]:
    """Synthesize minimal /etc/passwd and /etc/group so pwd.getpwuid succeeds.

    Docker applies numeric UID/GID but does not synthesize NSS entries; libraries
    like Torch Inductor call pwd.getpwuid() and crash without one. pwd/grp here
    go through NSS, so this also resolves AD/SSSD/NIS users that don't appear in
    the host's /etc/passwd at all (e.g. corp laptops, HPC clusters).
    """
    if os.getuid() in _IMAGE_BAKED_UIDS:
        return []
    user = pwd.getpwuid(os.getuid())
    group = grp.getgrgid(os.getgid())
    etc = scratch_home / "etc"
    etc.mkdir(parents=True, exist_ok=True)
    (etc / "passwd").write_text(
        "root:x:0:0:root:/root:/bin/bash\n"
        f"{user.pw_name}:x:{user.pw_uid}:{user.pw_gid}:{user.pw_gecos}:{Path.home()}:{user.pw_shell}\n"
    )
    (etc / "group").write_text(f"root:x:0:\n{group.gr_name}:x:{group.gr_gid}:{user.pw_name}\n")
    return [
        "-v",
        f"{etc / 'passwd'}:/etc/passwd:ro",
        "-v",
        f"{etc / 'group'}:/etc/group:ro",
        "-e",
        f"USER={user.pw_name}",
        "-e",
        f"LOGNAME={user.pw_name}",
    ]


def _launch_in_docker_container(opts: LaunchDocker) -> None:
    """Launch the command inside a local Docker container."""
    if not LOCAL_WORKSPACE_PATH.exists():
        Path(LOCAL_WORKSPACE_PATH).mkdir()
    is_model_cli = command_contains(opts.command, "model_cli")

    gpus_string = f'"device={opts.gpus}"' if opts.gpus else "all"

    user_strings = ["-u", f"{os.getuid()}:{os.getgid()}"]
    # $HOME doesn't exist inside the image for an arbitrary UID; bind-mount a
    # dedicated host scratch dir so libs can write there (caches persist across
    # runs). The scratch path can be overridden via COSMOS_CURATOR_LOCAL_HOME_DIR
    # (useful for remote homedirs or tight quotas).
    home_dir = Path.home()
    scratch_home_override = os.environ.get("COSMOS_CURATOR_LOCAL_HOME_DIR")
    scratch_home = (
        Path(scratch_home_override).expanduser()
        if scratch_home_override
        else home_dir / ".cache" / "cosmos-curator-home"
    )
    scratch_home.mkdir(parents=True, exist_ok=True)
    home_strings = ["-v", f"{scratch_home}:{home_dir}", "-e", f"HOME={home_dir}"]
    # A host pixi workspace runs at its host path, where its environments were installed.
    workspace = _host_pixi_workspace(opts)
    if workspace is not None:
        _prepare_workspace_mount_point(workspace, home_dir=home_dir, scratch_home=scratch_home)

    docker_command = [
        "docker",
        "run",
        "--rm",
        f"--gpus={gpus_string}",
        "--device=/dev/dri:/dev/dri",
        f"--shm-size={_get_shm_size_str()}",
        "-e",
        f"{LOCAL_DOCKER_ENV_VAR_NAME}=1",
        "-e",
        "NVCF_REQUEST_STATUS=false",
        "-e",
        f"{CURATOR_IO_SLOTS_PER_NODE_ENV_VAR}={opts.ray_io_slots_per_node}",
        # Keep redirected output responsive too.
        "-e",
        "PYTHONUNBUFFERED=1",
    ]
    if opts.docker_network:
        docker_command.append(f"--network={opts.docker_network}")
    docker_command.extend(user_strings)
    docker_command.extend(home_strings)
    docker_command.extend(_get_log_env_forward_strings())
    docker_command.extend(
        [
            "-v",
            f"{LOCAL_WORKSPACE_PATH}:{CONTAINER_PATHS_DEFAULT_WORKSPACE_DIR}",
        ]
    )
    docker_command.extend(_get_code_mount_strings(opts, workspace))
    docker_command.extend(_get_pixi_mount_strings(opts, workspace))
    if workspace is not None:
        docker_command.extend(["-w", str(workspace)])
    docker_command.extend(_get_s3_creds_mount_strings(opts))
    docker_command.extend(_get_azure_creds_mount_strings(opts))
    docker_command.extend(_get_config_file_mount_strings(is_model_cli=is_model_cli))
    for vol in opts.extra_volumes:
        docker_command.extend(["-v", vol])
    docker_command.extend(_get_identity_mounts(scratch_home))
    # Ray's live progress display requires a TTY. Keep stdout/stderr separate
    # when either is redirected; Docker combines them when a TTY is allocated.
    if sys.stdout.isatty() and sys.stderr.isatty():
        docker_command.append("-t")
    docker_command.extend(
        [
            f"{opts.image_label}",
            "bash",
            "-c",
        ]
    )

    # Ray workers run in background process groups. Disconnect stdin before
    # starting them so inherited terminal reads cannot stop children with SIGTTIN.
    preamble_parts: list[str] = ["exec < /dev/null"]

    # Prepend slim-image environment warmup. When COSMOS_CURATOR_SLIM_ENVS is set
    # (slim images only), install the declared environments before running the user command.
    # A host workspace's environments are installed and updated by host pixi. A .pixi
    # directory without a workspace, such as shared storage for container environments,
    # keeps the warmup.
    if workspace is None:
        preamble_parts.append(SLIM_IMAGE_WARMUP_COMMAND)
    container_command = " && ".join(preamble_parts) + f" && {opts.command}"

    docker_command_to_print = " ".join([*docker_command, f'"{container_command}"'])
    logger.info(f"Docker command:\n{docker_command_to_print}")

    docker_command.append(container_command)

    result = subprocess.call(  # noqa: S603
        docker_command,
        shell=False,
    )
    if result != 0:
        logger.error("Failed to run command via docker")
        sys.exit(1)
