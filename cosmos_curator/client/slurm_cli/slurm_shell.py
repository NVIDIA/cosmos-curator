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
"""Interactive Slurm shell and image import commands."""

import logging
import os
import shlex
import subprocess
import time as time_module
from pathlib import Path
from typing import Annotated

import attrs
import typer
from typer import Argument, Option

from cosmos_curator.client.slurm_cli.slurm_common import (
    _CACHE_MOUNT_PATH,
    _CONTAINER_AZURE_CREDS_PATH,
    _CONTAINER_S3_CREDS_PATH,
    _CONTAINER_SOURCE_DIR,
    _DEFAULT_CACHE_PATH,
    _DEFAULT_CONDA_OVERRIDE_CUDA,
    _DEFAULT_CONTAINER_IMAGE,
    _DEFAULT_LOGIN_NODE,
    _SLURM_ACCOUNT_ENV_VAR,
    MountSpec,
    SlurmContainerRuntime,
    SrunCommand,
    _build_slurm_container_runtime,
    _build_srun_command,
    _container_mount_specs_from_runtime,
    _get_username,
    _infer_curator_path,
    _is_local_host,
    _is_valid_slurm_memory,
    _merge_mount_specs_by_destination,
    _mount_specs_from_strings,
    _normalize_optional_slurm_directive,
    _parse_mount_specs,
    _resolve_slurm_account,
    _validate_gpu_options,
)
from cosmos_curator.client.utils.container_launch import command_contains
from cosmos_curator.core.utils import environment

logger = logging.getLogger(__name__)

_DEFAULT_IMPORT_IMAGE_DIR = Path("~/container_images")
_DEFAULT_IMPORT_IMAGE_OUTPUT_FILENAME = Path(_DEFAULT_CONTAINER_IMAGE).name
_DEFAULT_IMPORT_IMAGE_PARTITION = "cpu"
_DEFAULT_IMPORT_IMAGE_CPUS_PER_TASK = 16
_DEFAULT_IMPORT_IMAGE_MEMORY = "64G"
_DEFAULT_IMPORT_IMAGE_RETRIES = 5
_DEFAULT_IMPORT_IMAGE_RETRY_DELAY_SECONDS = 5
_DEFAULT_ENROOT_TEMP_PATH = Path("/") / "tmp"


@attrs.define
class ImportImageCommand:
    """Concrete Slurm image import command and user-facing metadata."""

    srun_command: SrunCommand
    output_path: Path
    image_uri: str


def _remote_shell_container_mount_specs(
    runtime: SlurmContainerRuntime, container_mounts: str | None = None
) -> list[MountSpec]:
    mounts = [
        MountSpec(source=str(runtime.workspace_path), dest=str(environment.CONTAINER_PATHS_DEFAULT_WORKSPACE_DIR)),
        MountSpec(source=str(runtime.cache_path), dest=str(_CACHE_MOUNT_PATH)),
    ]
    if runtime.curator_path is not None:
        mounts.append(MountSpec(source=str(runtime.curator_path), dest=str(_CONTAINER_SOURCE_DIR)))
    if runtime.mount_s3_creds:
        mounts.append(
            MountSpec(source=str(environment.LOCAL_AWS_CREDENTIALS_FILE), dest=str(_CONTAINER_S3_CREDS_PATH), mode="ro")
        )
    if runtime.mount_azure_creds:
        mounts.append(
            MountSpec(
                source=str(environment.LOCAL_AZURE_CREDENTIALS_FILE),
                dest=str(_CONTAINER_AZURE_CREDS_PATH),
                mode="ro",
            )
        )
    if command_contains(runtime.command, "model_cli"):
        mounts.append(
            MountSpec(
                source=str(environment.LOCAL_COSMOS_CURATOR_CONFIG_FILE),
                dest=str(environment.CONTAINER_PATHS_COSMOS_CURATOR_CONFIG_FILE),
                mode="ro",
            )
        )
    return _merge_mount_specs_by_destination(
        [
            *mounts,
            *_mount_specs_from_strings(runtime.extra_mounts),
            *_parse_mount_specs(container_mounts),
        ]
    )


def _resolve_shell_paths(
    *,
    curator_path: Path | None,
    workspace_path: Path | None,
    cache_path: Path | None,
    is_remote_login_node: bool,
) -> tuple[Path | None, Path, Path]:
    if not is_remote_login_node:
        return (
            _infer_curator_path(curator_path),
            workspace_path or environment.LOCAL_WORKSPACE_PATH,
            cache_path or _DEFAULT_CACHE_PATH,
        )

    if curator_path is None:
        msg = "--curator-path must point to a path on the remote login node when --login-node is remote"
        raise typer.BadParameter(msg)
    return curator_path, workspace_path or environment.LOCAL_WORKSPACE_PATH, cache_path or _DEFAULT_CACHE_PATH


def _should_show_shell_help(ctx: typer.Context, command: list[str] | None) -> bool:
    if command:
        return False
    return all(
        getattr(ctx.get_parameter_source(parameter_name), "name", None) == "DEFAULT"
        for parameter_name in ctx.params
        if parameter_name != "command"
    )


def _srun_allocation_args(  # noqa: PLR0913
    *,
    account: str | None,
    partition: str | None,
    qos: str | None,
    gres: str | None,
    gpus: str | None,
    gpus_per_node: str | None,
    job_name: str,
    num_nodes: int | None,
    exclusive: bool,
    time_limit: str | None,
    cpus_per_task: int | None = None,
    memory: str | None = None,
) -> list[str]:
    if num_nodes is not None and num_nodes < 1:
        msg = "--nodes must be at least 1"
        raise typer.BadParameter(msg)
    if cpus_per_task is not None and cpus_per_task < 1:
        msg = "--cpus-per-task must be at least 1"
        raise typer.BadParameter(msg)
    if memory is not None and not _is_valid_slurm_memory(memory):
        msg = f"--mem must be a positive Slurm size such as 64G or 65536M, got {memory!r}"
        raise typer.BadParameter(msg)

    gres, gpus, gpus_per_node = _validate_gpu_options(gres=gres, gpus=gpus, gpus_per_node=gpus_per_node)
    args = [f"--nodes={num_nodes}"] if num_nodes is not None else []
    optional_args = [
        ("--account", _resolve_slurm_account(account)),
        ("--partition", partition),
        ("--qos", qos),
        ("--gpus", gpus),
        ("--gpus-per-node", gpus_per_node),
        ("--gres", gres),
        ("--time", time_limit),
        ("--job-name", job_name),
        ("--cpus-per-task", str(cpus_per_task) if cpus_per_task is not None else None),
        ("--mem", memory),
    ]
    args.extend(
        f"{flag}={normalized}"
        for flag, value in optional_args
        if (normalized := _normalize_optional_slurm_directive(value))
    )
    if exclusive:
        args.append("--exclusive")
    return args


def _normalize_enroot_image_uri(image: str) -> str:
    image = image.strip()
    if not image:
        msg = "image cannot be empty"
        raise typer.BadParameter(msg)
    if "://" in image:
        return image
    return f"docker://{image}"


def _validate_output_filename(output_filename: str) -> str:
    filename = output_filename.strip()
    if not filename:
        msg = "--output-filename cannot be empty"
        raise typer.BadParameter(msg)
    if Path(filename).name != filename:
        msg = "--output-filename must be a filename, not a path; use --output-dir for the directory"
        raise typer.BadParameter(msg)
    return filename


def _expand_slurm_user_path(path: Path, username: str, *, is_remote_login_node: bool) -> Path:
    raw_path = str(path)
    if not is_remote_login_node or not raw_path.startswith("~"):
        return path.expanduser()

    remote_home = Path(os.getenv("REMOTE_HOME_DIR", f"/home/{username or _get_username()}"))
    if raw_path == "~":
        return remote_home
    if raw_path.startswith("~/"):
        return remote_home / raw_path[2:]
    return path


def _build_import_image_srun_command(  # noqa: PLR0913
    *,
    image: str,
    account: str | None,
    partition: str | None,
    output_dir: Path,
    output_filename: str,
    overwrite: bool,
    retries: int,
    retry_delay_seconds: int,
    enroot_temp_path: Path,
    slurm_home_path: Path,
    job_name: str,
    time_limit: str | None,
    cpus_per_task: int,
    memory: str,
) -> ImportImageCommand:
    if retries < 1:
        msg = "--retries must be at least 1"
        raise typer.BadParameter(msg)
    if retry_delay_seconds < 0:
        msg = "--retry-delay must be at least 0"
        raise typer.BadParameter(msg)

    image_uri = _normalize_enroot_image_uri(image)
    filename = _validate_output_filename(output_filename)
    output_path = output_dir / filename
    enroot_cache_path = output_dir / ".enroot-cache"

    subprocess_env = os.environ.copy()
    subprocess_env.update(
        {
            "HOME": str(slurm_home_path),
            "ENROOT_CACHE_PATH": str(enroot_cache_path),
            "ENROOT_DATA_PATH": str(output_dir),
            "ENROOT_TEMP_PATH": str(enroot_temp_path),
        }
    )
    srun_export_keys = ["HOME", "PATH", "ENROOT_CACHE_PATH", "ENROOT_DATA_PATH", "ENROOT_TEMP_PATH"]
    remote_env_keys = ["HOME", "ENROOT_CACHE_PATH", "ENROOT_DATA_PATH", "ENROOT_TEMP_PATH"]

    import_script_parts = [
        'mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH"',
    ]
    if overwrite:
        import_script_parts.append('rm -f -- "$1"')
    import_script_parts.append('exec enroot import --output "$1" "$2"')
    import_script = " && ".join(import_script_parts)

    command = [
        "srun",
        *_srun_allocation_args(
            account=account,
            partition=partition,
            qos=None,
            gres=None,
            gpus=None,
            gpus_per_node=None,
            job_name=job_name,
            num_nodes=1,
            exclusive=False,
            time_limit=time_limit,
            cpus_per_task=cpus_per_task,
            memory=memory,
        ),
        "--ntasks=1",
        # Login-node shell limits can be smaller than the requested Slurm allocation.
        "--propagate=NONE",
        f"--export={','.join(srun_export_keys)}",
        "bash",
        "-c",
        import_script,
        "_",
        str(output_path),
        image_uri,
    ]
    return ImportImageCommand(
        srun_command=SrunCommand(command=command, environment=subprocess_env, container_env_keys=remote_env_keys),
        output_path=output_path,
        image_uri=image_uri,
    )


def _remote_srun_command(shell_command: SrunCommand) -> str:
    env_assignments = [
        f"{key}={shell_command.environment[key]}"
        for key in shell_command.container_env_keys
        if key in shell_command.environment
    ]
    if not env_assignments:
        return shlex.join(shell_command.command)
    return shlex.join(["env", *env_assignments, *shell_command.command])


def _ssh_srun_command(login_node: str, username: str, shell_command: SrunCommand, *, pty: bool) -> list[str]:
    ssh_target = f"{username}@{login_node}" if username else login_node
    command = ["ssh"]
    if pty:
        command.append("-t")
    command.extend([ssh_target, _remote_srun_command(shell_command)])
    return command


def _print_srun_command(login_node: str, username: str, shell_command: SrunCommand, *, pty: bool) -> None:
    if _is_local_host(login_node):
        typer.echo(shlex.join(shell_command.command))
        return
    typer.echo(shlex.join(_ssh_srun_command(login_node, username, shell_command, pty=pty)))


def _run_srun_command(login_node: str, username: str, shell_command: SrunCommand, *, pty: bool = True) -> None:
    logger.info("Slurm command:\n%s", shlex.join(shell_command.command))

    if _is_local_host(login_node):
        try:
            result = subprocess.call(shell_command.command, shell=False, env=shell_command.environment)  # noqa: S603
        except OSError as exc:
            logger.exception("Failed to start local srun")
            raise typer.Exit(1) from exc
        if result != 0:
            logger.error("Failed to run command via srun")
            raise typer.Exit(1)
        return

    ssh_command = _ssh_srun_command(login_node, username, shell_command, pty=pty)
    logger.info("SSH command:\n%s", shlex.join(ssh_command))
    result = subprocess.call(ssh_command, shell=False)  # noqa: S603
    if result != 0:
        logger.error("Failed to run command via ssh on %s", login_node)
        raise typer.Exit(1)


def _run_srun_command_with_retries(
    login_node: str,
    username: str,
    shell_command: SrunCommand,
    *,
    retries: int,
    retry_delay_seconds: int,
) -> None:
    for attempt in range(1, retries + 1):
        try:
            _run_srun_command(login_node, username, shell_command)
        except typer.Exit:
            if attempt == retries:
                raise
            logger.warning(
                "Import attempt %s/%s failed; retrying in %ss",
                attempt,
                retries,
                retry_delay_seconds,
            )
            time_module.sleep(retry_delay_seconds)
        else:
            return


def _run_srun_shell(login_node: str, username: str, shell_command: SrunCommand, *, pty: bool) -> None:
    _run_srun_command(login_node, username, shell_command, pty=pty)


def import_image_cli(  # noqa: PLR0913
    image: Annotated[
        str,
        Argument(
            help=("Docker image reference or Enroot URI to import. Unprefixed values are treated as docker:// images."),
            rich_help_panel="common",
        ),
    ],
    *,
    login_node: Annotated[
        str,
        Option(
            help="Hostname of SLURM login node to run import on. Defaults to local srun execution.",
            rich_help_panel="cluster",
        ),
    ] = _DEFAULT_LOGIN_NODE,
    account: Annotated[
        str | None,
        Option(
            "-A",
            "--account",
            help=(
                f"Name of account for billing. Defaults to ${_SLURM_ACCOUNT_ENV_VAR} when set; "
                "otherwise omit to use the cluster default."
            ),
            rich_help_panel="cluster",
        ),
    ] = None,
    partition: Annotated[
        str | None,
        Option(
            "-p",
            "--partition",
            help="The Slurm partition to use for the import job.",
            rich_help_panel="cluster",
        ),
    ] = _DEFAULT_IMPORT_IMAGE_PARTITION,
    job_name: Annotated[
        str,
        Option("-J", "--job-name", help="Name of the Slurm import job.", rich_help_panel="cluster"),
    ] = "cosmos_curator_import_image",
    time: Annotated[
        str | None,
        Option(
            "-t",
            "--time",
            help="Time limit for the import, e.g. 01:00:00 for 1 hour. See srun --time for more details.",
            rich_help_panel="cluster",
        ),
    ] = None,
    cpus_per_task: Annotated[
        int,
        Option(
            "-c",
            "--cpus-per-task",
            help="Number of CPUs allocated to the Enroot import task.",
            rich_help_panel="cluster",
        ),
    ] = _DEFAULT_IMPORT_IMAGE_CPUS_PER_TASK,
    memory: Annotated[
        str,
        Option(
            "--mem",
            help="Memory allocated to the import node in Slurm --mem format, for example 64G.",
            rich_help_panel="cluster",
        ),
    ] = _DEFAULT_IMPORT_IMAGE_MEMORY,
    output_dir: Annotated[
        Path,
        Option(
            help="Cluster-visible directory for the imported .sqsh image and Enroot cache.",
            rich_help_panel="output",
        ),
    ] = _DEFAULT_IMPORT_IMAGE_DIR,
    output_filename: Annotated[
        str,
        Option(
            help="Filename for the imported .sqsh image within --output-dir.",
            rich_help_panel="output",
        ),
    ] = _DEFAULT_IMPORT_IMAGE_OUTPUT_FILENAME,
    overwrite: Annotated[
        bool,
        Option(
            "--overwrite/--no-overwrite",
            help="Remove an existing output file before importing.",
            rich_help_panel="output",
        ),
    ] = True,
    retries: Annotated[
        int,
        Option(
            help="Number of times to retry the import command.",
            rich_help_panel="runtime",
        ),
    ] = _DEFAULT_IMPORT_IMAGE_RETRIES,
    retry_delay_seconds: Annotated[
        int,
        Option(
            "--retry-delay",
            help="Seconds to wait between failed import attempts.",
            rich_help_panel="runtime",
        ),
    ] = _DEFAULT_IMPORT_IMAGE_RETRY_DELAY_SECONDS,
    enroot_temp_path: Annotated[
        Path,
        Option(
            help="Path to use for ENROOT_TEMP_PATH during import.",
            rich_help_panel="runtime",
        ),
    ] = _DEFAULT_ENROOT_TEMP_PATH,
    username: Annotated[
        str,
        Option(help="Optional cluster username.", rich_help_panel="misc"),
    ] = f"{_get_username()}",
) -> None:
    """Import a Docker image into an Enroot squashfs image through Slurm."""
    is_remote_login_node = not _is_local_host(login_node)
    import_command = _build_import_image_srun_command(
        image=image,
        account=account,
        partition=partition,
        output_dir=_expand_slurm_user_path(output_dir, username, is_remote_login_node=is_remote_login_node),
        output_filename=output_filename,
        overwrite=overwrite,
        retries=retries,
        retry_delay_seconds=retry_delay_seconds,
        enroot_temp_path=_expand_slurm_user_path(enroot_temp_path, username, is_remote_login_node=is_remote_login_node),
        slurm_home_path=_expand_slurm_user_path(Path("~"), username, is_remote_login_node=is_remote_login_node),
        job_name=job_name,
        time_limit=time,
        cpus_per_task=cpus_per_task,
        memory=memory,
    )

    typer.echo(f"Importing {import_command.image_uri} to {import_command.output_path}")
    _run_srun_command_with_retries(
        login_node,
        username,
        import_command.srun_command,
        retries=retries,
        retry_delay_seconds=retry_delay_seconds,
    )
    typer.echo(f"Imported {import_command.image_uri} to {import_command.output_path}")


def shell_cli(  # noqa: PLR0913
    ctx: typer.Context,
    command: Annotated[
        list[str] | None,
        Argument(
            help="The command to run inside the interactive container. Defaults to bash.",
            rich_help_panel="common",
        ),
    ] = None,
    *,
    login_node: Annotated[
        str,
        Option(
            help="Hostname of SLURM login node to run command on. Defaults to local srun execution.",
            rich_help_panel="cluster",
        ),
    ] = _DEFAULT_LOGIN_NODE,
    account: Annotated[
        str | None,
        Option(
            "-A",
            "--account",
            help=(
                f"Name of account for billing. Defaults to ${_SLURM_ACCOUNT_ENV_VAR} when set; "
                "otherwise omit to use the cluster default."
            ),
            rich_help_panel="cluster",
        ),
    ] = None,
    partition: Annotated[
        str | None,
        Option(
            "-p",
            "--partition",
            help="The slurm partition to use. Omit to use the cluster default.",
            rich_help_panel="cluster",
        ),
    ] = None,
    qos: Annotated[
        str | None,
        Option("-q", "--qos", help="Optional Slurm quality of service to request.", rich_help_panel="cluster"),
    ] = None,
    gres: Annotated[
        str | None,
        Option(
            help=("GPUs per node in GRES form, e.g. 'gpu:8'. Cannot be used with --gpus or --gpus-per-node."),
            rich_help_panel="cluster",
        ),
    ] = None,
    gpus: Annotated[
        str | None,
        Option(
            "-G",
            "--gpus",
            help=("Total GPUs for the job, e.g. '8' or 'h100:8'. Cannot be used with --gres or --gpus-per-node."),
            rich_help_panel="cluster",
        ),
    ] = None,
    gpus_per_node: Annotated[
        str | None,
        Option(
            help=(
                "GPUs per node, e.g. '8' or 'h100:8'. Use on clusters that require per-node GPU requests. "
                "Cannot be used with --gpus or --gres."
            ),
            rich_help_panel="cluster",
        ),
    ] = None,
    job_name: Annotated[
        str,
        Option("-J", "--job-name", help="Name of the interactive Slurm job.", rich_help_panel="cluster"),
    ] = "cosmos_curator_shell",
    num_nodes: Annotated[
        int | None,
        Option(
            "-N",
            "--nodes",
            "--num-nodes",
            help="Optional number of nodes to allocate for the interactive shell.",
            rich_help_panel="cluster",
        ),
    ] = None,
    exclusive: Annotated[
        bool,
        Option(help="Whether to use nodes exclusively.", rich_help_panel="cluster"),
    ] = True,
    time: Annotated[
        str | None,
        Option(
            "-t",
            "--time",
            help="Time limit for the shell, e.g. 01:00:00 for 1 hour. See srun --time for more details.",
            rich_help_panel="cluster",
        ),
    ] = None,
    container_image: Annotated[
        str,
        Option("--container-image", help="Path to the .sqsh image for srun/Pyxis.", rich_help_panel="container"),
    ] = _DEFAULT_CONTAINER_IMAGE,
    curator_path: Annotated[
        Path | None,
        Option(
            help=(
                "Path to the cosmos-curator repo directory; defaults to the current directory when it looks like "
                "a checkout."
            ),
            rich_help_panel="container",
        ),
    ] = None,
    workspace_path: Annotated[
        Path | None,
        Option(
            help=(
                "Host workspace directory to mount as /config inside the container. Defaults locally to the "
                "Cosmos Curator workspace."
            ),
            rich_help_panel="container",
        ),
    ] = None,
    cache_path: Annotated[
        Path | None,
        Option(
            help=(
                "Host cache directory to mount as /cache for Pixi/rattler, uv, Torch, Triton, pip, and CUDA caches. "
                "Defaults to ~/.cache."
            ),
            rich_help_panel="container",
        ),
    ] = None,
    mount_s3_creds: Annotated[
        bool,
        Option(
            "--mount-s3-creds/--no-mount-s3-creds",
            help="Mount the host AWS credentials file into the container when present.",
            rich_help_panel="container",
        ),
    ] = True,
    mount_azure_creds: Annotated[
        bool,
        Option(
            "--mount-azure-creds/--no-mount-azure-creds",
            help="Mount the host Azure credentials file into the container when present.",
            rich_help_panel="container",
        ),
    ] = False,
    container_mounts: Annotated[
        str | None,
        Option(
            help=(
                "Comma-separated container mounts in HOST_PATH:CONTAINER_PATH[:ro|rw] format. Mounts are merged "
                "by CONTAINER_PATH with later entries overriding earlier ones."
            ),
            rich_help_panel="container",
        ),
    ] = None,
    extra_mounts: Annotated[
        str,
        Option(
            "--extra-mounts",
            "--extra-volumes",
            help=(
                "Comma-separated extra container mounts in HOST_PATH:CONTAINER_PATH[:ro|rw] format. Mounts are merged "
                "by CONTAINER_PATH with later entries overriding earlier defaults."
            ),
            rich_help_panel="container",
        ),
    ] = "",
    environment_vars: Annotated[
        str | None,
        Option(
            "--environment",
            help="Comma separated list of environment variables to set in the container.",
            rich_help_panel="container",
        ),
    ] = None,
    conda_override_cuda: Annotated[
        str | None,
        Option(
            help="Set CONDA_OVERRIDE_CUDA during Pixi warmup. Use an empty value to omit it.",
            rich_help_panel="container",
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
            rich_help_panel="container",
        ),
    ] = None,
    username: Annotated[
        str,
        Option(help="Optional cluster username.", rich_help_panel="misc"),
    ] = f"{_get_username()}",
    pty: Annotated[
        bool,
        Option(
            "--pty/--no-pty",
            help="Request a pseudo-terminal from srun and remote ssh. Use --no-pty for non-interactive callers.",
            rich_help_panel="runtime",
        ),
    ] = True,
    dry_run: Annotated[
        bool,
        Option(
            "--dry-run",
            help="Print the local srun or remote ssh command without executing it.",
            rich_help_panel="runtime",
        ),
    ] = False,
) -> None:
    """Start an interactive shell or command inside a Slurm/Pyxis allocation."""
    if _should_show_shell_help(ctx, command):
        typer.echo(ctx.get_help())
        raise typer.Exit

    is_remote_login_node = not _is_local_host(login_node)
    resolved_curator_path, resolved_workspace_path, resolved_cache_path = _resolve_shell_paths(
        curator_path=curator_path,
        workspace_path=workspace_path,
        cache_path=cache_path,
        is_remote_login_node=is_remote_login_node,
    )

    user_command = command or ["bash"]
    runtime = _build_slurm_container_runtime(
        container_image=container_image,
        curator_path=resolved_curator_path,
        command=user_command,
        workspace_path=resolved_workspace_path,
        cache_path=resolved_cache_path,
        mount_s3_creds=mount_s3_creds,
        mount_azure_creds=mount_azure_creds,
        extra_mounts=extra_mounts,
        environment=environment_vars,
        conda_override_cuda=conda_override_cuda,
        pixi_envs=pixi_envs,
    )
    shell_command = _build_srun_command(
        runtime,
        slurm_args=_srun_allocation_args(
            account=account,
            partition=partition,
            qos=qos,
            gres=gres,
            gpus=gpus,
            gpus_per_node=gpus_per_node,
            job_name=job_name,
            num_nodes=num_nodes,
            exclusive=exclusive,
            time_limit=time,
        ),
        container_mounts=[
            str(mount)
            for mount in (
                _remote_shell_container_mount_specs(runtime, container_mounts)
                if is_remote_login_node
                else _container_mount_specs_from_runtime(runtime, container_mounts)
            )
        ],
        pty=pty,
    )
    if dry_run:
        _print_srun_command(login_node, username, shell_command, pty=pty)
        return

    _run_srun_shell(login_node, username, shell_command, pty=pty)
