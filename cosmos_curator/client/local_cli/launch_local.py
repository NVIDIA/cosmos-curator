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
    mount_s3_creds: bool
    mount_azure_creds: bool
    extra_volumes: list[str]  # each entry is HOST_PATH:CONTAINER_PATH


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
                "Use with --curator-path to avoid reinstalling environments at runtime (e.g. --pixi-path .)."
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
                "Comma-separated extra volume mounts in HOST_PATH:CONTAINER_PATH format. "
                "e.g. --extra-volumes /data/models:/config/models,/data/videos:/workspace/input"
            ),
            rich_help_panel="local-docker",
        ),
    ] = "",
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
        mount_s3_creds=mount_s3_creds,
        mount_azure_creds=mount_azure_creds,
        extra_volumes=_parse_extra_volumes(extra_volumes),
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
        # strip the possible ">=" or "<=" prefix
        if python_version.startswith((">=", "<=")):
            python_version = python_version[2:]
        # strip the minor version
        version_parts = python_version.split(".")
        _TARGET_PYTHON_VERSION_PARTS = 2
        if len(version_parts) < _TARGET_PYTHON_VERSION_PARTS:
            logger.warning(f"Invalid python version format in pixi.toml: {python_version}")
            return None
        return ".".join(version_parts[:_TARGET_PYTHON_VERSION_PARTS])


def _get_pixi_mount_strings(opts: LaunchDocker) -> list[str]:
    if opts.pixi_path is None:
        return []
    pixi_root = Path(opts.pixi_path)
    pixi_dir = pixi_root / ".pixi"
    if not pixi_dir.is_dir():
        logger.error(f"Pixi directory not found at {pixi_dir}")
        sys.exit(1)
    mount_strings = ["-v", f"{pixi_dir.absolute()}:{CONTAINER_PATHS_CODE_DIR / '.pixi'}"]
    # Also mount pixi.toml and pixi.lock so they stay in sync with the environments
    for name in ("pixi.toml", "pixi.lock"):
        host_file = pixi_root / name
        if host_file.is_file():
            mount_strings.extend(["-v", f"{host_file.absolute()}:{CONTAINER_PATHS_CODE_DIR / name}"])
    return mount_strings


def _get_code_mount_strings(opts: LaunchDocker) -> list[str]:
    code_path_strings = []
    if opts.curator_path is not None:
        curator_code_path = Path(opts.curator_path) / Path("cosmos_curator")
        pipeline_code_path = curator_code_path / Path("pipelines")
        if not pipeline_code_path.exists():
            logger.error(f"Curator pipelines code does not exist at {pipeline_code_path}")
            sys.exit(1)
        code_path_strings += ["-v", f"{curator_code_path.absolute()}:{CONTAINER_PATHS_CODE_DIR}/cosmos_curator"]

        if opts.mount_xenna:
            xenna_path = (Path(opts.curator_path) / Path("cosmos-xenna") / Path("cosmos_xenna")).absolute()
            _python_version = _get_python_version_from_pixi_toml(Path(opts.curator_path))
            if xenna_path.exists() and _python_version is not None:
                xenna_lib_path = (
                    CONTAINER_PATHS_CODE_DIR
                    / Path(".pixi/envs/default/lib")
                    / f"python{_python_version}"
                    / Path("site-packages/cosmos_xenna")
                )
                for python_module in ["pipelines", "ray_utils", "utils"]:
                    code_path_strings += ["-v", f"{xenna_path / python_module}:{xenna_lib_path / python_module}"]

        tests_path = Path(opts.curator_path) / Path("tests") / Path("cosmos_curator")
        code_path_strings += ["-v", f"{tests_path.absolute()}:{CONTAINER_PATHS_CODE_DIR}/tests/cosmos_curator"]

        tools_path = Path(opts.curator_path) / Path("tools")
        if tools_path.is_dir():
            code_path_strings += ["-v", f"{tools_path.absolute()}:{CONTAINER_PATHS_CODE_DIR}/tools"]
    return code_path_strings


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
    is_postgres_cli = command_contains(opts.command, "postgres_cli")

    gpus_string = f'"device={opts.gpus}"' if opts.gpus else "all"

    user_strings = ["-u", f"{os.getuid()}:{os.getgid()}"]
    # $HOME doesn't exist inside the image for an arbitrary UID; bind-mount a
    # dedicated host scratch dir so libs can write there (caches persist across
    # runs) and the --pixi-path symlink preamble works. The scratch path can be
    # overridden via COSMOS_CURATOR_LOCAL_HOME_DIR (useful for remote homedirs or
    # tight quotas).
    home_dir = Path.home()
    scratch_home_override = os.environ.get("COSMOS_CURATOR_LOCAL_HOME_DIR")
    scratch_home = (
        Path(scratch_home_override).expanduser()
        if scratch_home_override
        else home_dir / ".cache" / "cosmos-curator-home"
    )
    scratch_home.mkdir(parents=True, exist_ok=True)
    home_strings = ["-v", f"{scratch_home}:{home_dir}", "-e", f"HOME={home_dir}"]
    interactive_strings = ["-i"] if is_postgres_cli else []

    docker_command = [
        "docker",
        "run",
        "--rm",
        f"--gpus={gpus_string}",
        "--device=/dev/dri:/dev/dri",
        f"--shm-size={_get_shm_size_str()}",
        "--network=host",
        "--cap-add=SYS_ADMIN",
        "-e",
        f"{LOCAL_DOCKER_ENV_VAR_NAME}=1",
        "-e",
        "NVCF_REQUEST_STATUS=false",
    ]
    docker_command.extend(user_strings)
    docker_command.extend(home_strings)
    docker_command.extend(
        [
            "-v",
            f"{LOCAL_WORKSPACE_PATH}:{CONTAINER_PATHS_DEFAULT_WORKSPACE_DIR}",
        ]
    )
    docker_command.extend(_get_code_mount_strings(opts))
    docker_command.extend(_get_pixi_mount_strings(opts))
    docker_command.extend(_get_s3_creds_mount_strings(opts))
    docker_command.extend(_get_azure_creds_mount_strings(opts))
    docker_command.extend(_get_config_file_mount_strings(is_model_cli=is_model_cli))
    for vol in opts.extra_volumes:
        docker_command.extend(["-v", vol])
    docker_command.extend(_get_identity_mounts(scratch_home))
    docker_command.extend(interactive_strings)
    docker_command.extend(
        [
            "-t",
            f"{opts.image_label}",
            "bash",
            "-c",
        ]
    )

    # When pixi environments are bind-mounted from the host (--pixi-path), scripts
    # in .pixi/envs/*/bin/ carry shebangs with the host's absolute path (e.g.
    # #!/home/user/project/.pixi/envs/unified/bin/python3.12) which don't exist
    # inside the container.  Create a symlink so the kernel can resolve them.
    preamble_parts: list[str] = []
    if opts.pixi_path is not None:
        host_code_dir = Path(opts.pixi_path).resolve()
        if host_code_dir != CONTAINER_PATHS_CODE_DIR:
            host_parent = shlex.quote(str(host_code_dir.parent))
            container = shlex.quote(str(CONTAINER_PATHS_CODE_DIR))
            host = shlex.quote(str(host_code_dir))
            preamble_parts.append(f"mkdir -p {host_parent} && ln -sfn {container} {host}")

    # Prepend slim-image environment warmup. When COSMOS_CURATOR_SLIM_ENVS is set
    # (slim images only), install the declared environments before running the user command.
    # With --pixi-path this is a fast no-op since environments are already present.
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
