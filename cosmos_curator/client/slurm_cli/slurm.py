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
"""Remote SLURM job submission and management utilities."""

import logging
import os
import pwd
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
import time as time_module
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Protocol, Self, cast

import attrs
import fabric  # type: ignore[import-untyped]
import invoke
import jinja2
import typer
from attrs import field, validators
from invoke.context import Context
from invoke.runners import Result as InvokeResult
from typer import Argument, Option

from cosmos_curator.client.slurm_cli.slurm_local import (
    _CACHE_MOUNT_PATH,
    _CONTAINER_AZURE_CREDS_PATH,
    _CONTAINER_S3_CREDS_PATH,
    _CONTAINER_SOURCE_DIR,
    _DEFAULT_CACHE_PATH,
    _DEFAULT_CONDA_OVERRIDE_CUDA,
    _DEFAULT_CONTAINER_IMAGE,
    SlurmContainerRuntime,
    SrunCommand,
    _build_srun_command,
    _get_container_entrypoint_command,
    _get_srun_environment,
    _get_srun_mounts,
    _parse_environment,
    _parse_pixi_envs,
    _resolve_container_image,
)
from cosmos_curator.client.utils.container_launch import command_contains, parse_extra_mounts
from cosmos_curator.core.utils import environment

logger = logging.getLogger(__name__)

_SBATCH_TEMPLATE_PATH = Path("sbatch.sh.j2")
_PROM_SVC_DISC_SCRIPT_PATH = Path("prometheus_service_discovery.py")
_START_RAY = environment.CONTAINER_PATHS_CODE_DIR / "cosmos_curator" / "scripts" / "onto_slurm.py"
_MAX_FILE_MODE = 0o7777
_HOME_DIR = Path(os.getenv("REMOTE_HOME_DIR", Path.home()))
_DEFAULT_LOGIN_NODE = "localhost"
_DEFAULT_IMPORT_IMAGE_DIR = Path("~/container_images")
_DEFAULT_IMPORT_IMAGE_OUTPUT_FILENAME = Path(_DEFAULT_CONTAINER_IMAGE).name
_DEFAULT_IMPORT_IMAGE_PARTITION = "cpu"
_DEFAULT_IMPORT_IMAGE_RETRIES = 5
_DEFAULT_IMPORT_IMAGE_RETRY_DELAY_SECONDS = 5
_DEFAULT_ENROOT_TEMP_PATH = Path("/") / "tmp"
_SLURM_ACCOUNT_ENV_VAR = "SBATCH_ACCOUNT"
_SBATCH_DYNAMIC_CONTAINER_ENV_KEYS = (
    "HEAD_NODE_ADDR",
    "HEAD_NODE_PORT",
    "PRIMARY_NODE_HOSTNAME",
    "PRIMARY_NODE_PORT",
    "RAY_STOP_RETRIES_AFTER",
    "SLURM_JOB_ID",
    "SLURM_JOBID",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_NUM_NODES",
    "SLURM_LOCALID",
    "SLURM_NNODES",
    "SLURM_NODEID",
    "SLURM_NTASKS_PER_NODE",
    "SLURM_PROCID",
    "SLURMD_NODENAME",
    "WORLD_SIZE",
)


def _quote_remote_path(path: Path) -> str:
    return shlex.quote(str(path))


class ConnectionProtocol(Protocol):
    """Protocol capturing the subset of fabric.Connection used by this module."""

    host: str

    def run(self, command: str, **kwargs: Any) -> InvokeResult:  # noqa: ANN401
        """Run a command on the target host."""
        ...

    def put(self, local: str, remote: str) -> None:
        """Upload a local file to the target host."""
        ...

    def close(self) -> None:
        """Close the connection."""
        ...


class LocalConnection:
    """Connection implementation that executes commands locally without SSH."""

    def __init__(self, host: str, user: str) -> None:
        """Create a LocalConnection."""
        self.host = host
        self.user = user
        self._context = Context()

    def run(self, command: str, **kwargs: Any) -> InvokeResult:  # noqa: ANN401
        """Execute a shell command locally."""
        return self._context.run(command, **kwargs)

    def put(self, local: str, remote: str) -> None:
        """Copy a local file path to the destination on the same host."""
        local_path = Path(local).expanduser()
        if not local_path.exists():
            error_message = f"Source file does not exist: {local_path}"
            raise FileNotFoundError(error_message)

        remote_path = Path(remote).expanduser()
        try:
            remote_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            error_message = f"Failed to create parent directory for {remote_path}: {exc}"
            raise OSError(error_message) from exc

        try:
            shutil.copy2(local_path, remote_path)
        except OSError as exc:
            error_message = f"Failed to copy {local_path} to {remote_path}: {exc}"
            raise OSError(error_message) from exc

    def close(self) -> None:
        """Close the connection."""


def _get_username() -> str:
    """Retrieve the username of the current user.

    Returns:
        str: The username of the current user.

    Raises:
        KeyError: If the user ID is not found in the password database.
        OSError: If an operating system error occurs while retrieving the user ID or username.

    """
    uid = os.getuid()
    return pwd.getpwuid(uid).pw_name


def _get_user_dir(user_dir: Path | None = None) -> Path:
    """Get the user's directory."""
    if user_dir is not None:
        return user_dir

    return _HOME_DIR


def _get_log_dir(log_dir: Path | None = None, user_dir: Path | None = None) -> Path:
    """Get the default log directory.

    If the SLURM_LOG_DIR environment variable is set, it will be used. Otherwise, it will be placed into
    the user's directory.

    Args:
        log_dir: The path to the log directory.
        user_dir: The path to the user's directory.

    Returns:
        The path to the log directory.

    """
    if log_dir is not None:
        return log_dir

    log_dir_str = os.environ.get("SLURM_LOG_DIR")
    if log_dir_str is not None:
        return Path(log_dir_str)

    return _get_user_dir(user_dir) / "job_logs"


def _get_remote_job_path(remote_files_path: Path, job_name: str) -> Path:
    """Get the remote job path for the job on the cluster.

    Args:
        remote_files_path (Path): The path to the remote files directory for all jobs
        job_name (str): The name of the job.

    Returns:
        The remote files path for the job

    """
    return remote_files_path / f"{job_name}.{datetime.now().strftime('%Y%m%dT%H%M%S.%f')}"  # noqa: DTZ005


def _infer_curator_path(curator_path: Path | None) -> Path | None:
    """Use the current checkout by default when a Slurm command is run from a repo root."""
    if curator_path is not None:
        return curator_path

    cwd = Path.cwd()
    if (cwd / "cosmos_curator").is_dir() and (cwd / "pixi.toml").is_file():
        return cwd
    return None


def _parse_mount_specs(raw: str | None) -> list["MountSpec"]:
    if raw is None:
        return []
    return [MountSpec.from_str(entry.strip()) for entry in raw.split(",") if entry.strip()]


def _mount_specs_from_strings(mounts: list[str]) -> list["MountSpec"]:
    return [MountSpec.from_str(mount) for mount in mounts]


def _merge_mount_specs_by_destination(mounts: list["MountSpec"]) -> list["MountSpec"]:
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


def _environment_entries_from_srun_defaults(opts: SlurmContainerRuntime) -> list[str]:
    env, container_env_keys = _get_srun_environment(opts, include_slurm_env=False)
    return [f"{key}={env[key]}" for key in container_env_keys if key in env]


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
) -> list["MountSpec"]:
    return _merge_mount_specs_by_destination(
        [
            *_mount_specs_from_strings(_get_srun_mounts(runtime)),
            *_parse_mount_specs(container_mounts),
        ]
    )


def _remote_shell_container_mount_specs(
    runtime: SlurmContainerRuntime, container_mounts: str | None = None
) -> list["MountSpec"]:
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


def _normalize_optional_slurm_directive(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _resolve_slurm_account(account: str | None) -> str | None:
    """Resolve the account directive without making it mandatory for all clusters."""
    account = _normalize_optional_slurm_directive(account)
    if account is not None:
        return account

    env_account = os.getenv(_SLURM_ACCOUNT_ENV_VAR)
    return _normalize_optional_slurm_directive(env_account)


def _validate_gpu_options(*, gres: str | None, gpus: str | None) -> tuple[str | None, str | None]:
    gres = _normalize_optional_slurm_directive(gres)
    gpus = _normalize_optional_slurm_directive(gpus)
    if gres is not None and gpus is not None:
        msg = "--gres and --gpus cannot be used together"
        raise typer.BadParameter(msg)
    return gres, gpus


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


@attrs.define
class MountSpec:
    """Represents a mount and its mount point."""

    source: str
    dest: str
    mode: str = field(default="rw", validator=validators.in_(["rw", "ro"]))

    @classmethod
    def from_str(cls, mount_str: str) -> Self:
        """Create a MountSpec instance from a string.

        The string must have the format:
        source:dest:mode or source:dest

        Args:
            mount_str: The mount string formatted as one of:
                source:dest
                source:dest:mode

                mode is either "ro" or "rw"

        Returns:
            A MountSpec instance

        """
        _MIN_PARTS = 2
        _MAX_PARTS = 3

        parts = mount_str.split(":")
        if len(parts) < _MIN_PARTS or len(parts) > _MAX_PARTS:
            error_message = f"`{mount_str}` must have at least {_MIN_PARTS} or colon separated parts"
            raise ValueError(error_message)

        source = parts[0]
        dest = parts[1]
        mode = "rw" if len(parts) == _MIN_PARTS else parts[2]

        return cls(source=source, dest=dest, mode=mode)

    def __str__(self) -> str:
        """Return a string representation of the mount suitable for use with Docker or Enroot."""
        return f"{self.source}:{self.dest}:{self.mode}"


@attrs.define
class ContainerSpec:
    """Configuration for a container to run in the SLURM job."""

    command: list[str]
    squashfs_path: str
    mounts: list[MountSpec]
    environment: list[str]


@attrs.define
class SlurmJobSpec:
    """Configuration for a SLURM job."""

    login_node: str
    username: str
    account: str | None
    partition: str | None
    job_name: str
    remote_job_path: Path
    log_dir: Path
    num_nodes: int
    exclusive: bool
    container: ContainerSpec
    gres: str | None = None
    gpus: str | None = None
    qos: str | None = None
    time_limit: str | None = None
    stop_retries_after: int = 600
    exclude_nodes: list[str] | None = None
    comment: str | None = None
    prometheus_service_discovery_path: Path | None = None
    mail_type: str | None = None
    mail_user: str | None = None


@attrs.define
class ImportImageCommand:
    """Concrete Slurm image import command and user-facing metadata."""

    srun_command: SrunCommand
    output_path: Path
    image_uri: str


def _render_sbatch_script(spec: SlurmJobSpec) -> str:
    """Render a Slurm batch script using the provided job specification and cluster configuration.

    Args:
        spec (SlurmJobSpec): Job specification, including name, account,
            partition, and command.

    Returns:
        str: Rendered Slurm batch script as a string.

    Notes:
        This function assumes the _SBATCH_TEMPLATE_PATH template file
        exists and contains necessary template variables.

    """
    container_mounts = ",".join(str(x) for x in spec.container.mounts) if spec.container.mounts is not None else None
    command = shlex.join(spec.container.command)
    container_command = shlex.quote(_get_container_entrypoint_command())
    template_dir = Path(__file__).parent
    sbatch_template = template_dir / _SBATCH_TEMPLATE_PATH

    env_vars: dict[str, str] = {}
    if spec.container.environment is not None:
        for env_entry in spec.container.environment:
            key: str
            value: str | None
            if "=" in env_entry:
                key, value = env_entry.split("=", 1)
            else:
                key = env_entry
                value = os.environ.get(key)
            if value is None:
                continue
            env_vars[key] = value
    container_env_keys = list(dict.fromkeys([*env_vars, *_SBATCH_DYNAMIC_CONTAINER_ENV_KEYS]))

    return jinja2.Template(sbatch_template.read_text()).render(
        job_name=spec.job_name,
        account=spec.account,
        partition=spec.partition,
        num_nodes=spec.num_nodes,
        command=command,
        gres=spec.gres,
        gpus=spec.gpus,
        qos=spec.qos,
        exclusive=spec.exclusive,
        container_image=spec.container.squashfs_path,
        container_mounts=container_mounts,
        container_command=container_command,
        container_env_keys=container_env_keys,
        env_vars=env_vars,
        time_limit_string=spec.time_limit,
        stop_retries_after=spec.stop_retries_after,
        exclude_nodes=spec.exclude_nodes,
        log_dir=str(spec.log_dir),
        comment=spec.comment,
        enable_metrics_scraping="yes" if spec.prometheus_service_discovery_path is not None else "no",
        job_artifact_path=str(spec.remote_job_path),
        prometheus_service_discovery_path=str(spec.prometheus_service_discovery_path),
        mail_type=spec.mail_type,
        mail_user=spec.mail_user,
    )


def _parse_job_id(output: str) -> str:
    """Parse the job ID from a string that contains the submission confirmation of a batch job.

    Args:
        output: The output from a job submission command, expected to contain
            "Submitted batch job <job_id>".

    Returns:
        The job ID parsed from the output string.

    Raises:
        ValueError: If the job ID cannot be found or is not valid

    """
    # job_ids are not always an integer, they can contain dots and underscores
    pattern = r"Submitted batch job (.*)"
    match = re.search(pattern, output)

    if match:
        return match.group(1)

    error_message = f"Output '{output}' does not contain 'Submitted batch job' followed by a job ID."
    raise ValueError(error_message)


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


def connect(remote_host: str, user: str) -> ConnectionProtocol:
    """Connect to a SLURM cluster host.

    Args:
        remote_host (str): the hostname to connect to
        user (str): the username to connect with

    Returns:
        A fabric.Connection object if successful

    Raises:
        typer.Abort: if connection could not be established

    """
    logger.info("Connecting to %s as %s...", remote_host, user)

    if _is_local_host(remote_host):
        logger.info("Detected local SLURM login host; executing commands without SSH")
        conn = LocalConnection(remote_host, user)
        conn.run("ls", hide=True)
        return conn

    conn = fabric.Connection(remote_host, user=user)
    conn.run("ls", hide=True)
    return cast("ConnectionProtocol", conn)


def upload_text(connection: ConnectionProtocol, files: list[tuple[str, Path, int]]) -> None:
    """Upload multiple text strings the provided connection.

    Args:
        connection (fabric.Connection): An established connection
        files (list[tuple[str, Path, int]]): A list of tuples containing
            the text to upload, the remote path, and the octal file mode.

    Returns:
        None

    Raises:
        ValueError: If the number of files is not greater than zero.
        ValueError: If the octal file mode is not a valid integer.

    """
    if len(files) == 0:
        error_message = "Must upload at least one file"
        raise ValueError(error_message)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for text, remote_path, file_mode in files:
            if not isinstance(file_mode, int) or file_mode < 0 or file_mode > _MAX_FILE_MODE:
                error_message = f"Invalid octal file mode: {oct(file_mode)}"
                raise ValueError(error_message)

            tmp_file = Path(tmp_dir) / remote_path.name
            tmp_file.write_text(text)
            logger.debug("Uploading %s to %s", tmp_file, remote_path)

            connection.put(str(tmp_file), remote=str(remote_path))
            connection.run(f"chmod {file_mode:o} {_quote_remote_path(remote_path)}")


def remote_path_exists(connection: ConnectionProtocol, path: Path) -> bool:
    """Check if a path exists on the remote host.

    Args:
        connection: Connection to the login node
        path (Path): The path to check

    Returns:
        bool: True if the directory exists, False otherwise

    """
    dir_exists = False
    try:
        dir_exists = connection.run(f"test -e {_quote_remote_path(path)}", hide=True).ok
    except invoke.exceptions.UnexpectedExit as e:
        if e.result.exited != 1:  # test returns 1 when file doesn't exist
            # If exit code is something other than 1, there's another issue
            raise

    return dir_exists


def create_remote_path(connection: ConnectionProtocol, path: Path, mode: int = 0o700) -> None:
    """Create a remote path on the cluster.

    Args:
        connection: Connection to the login node
        path (Path): The path to create
        mode (int): The mode to set for the path

    """
    quoted_path = _quote_remote_path(path)
    connection.run(f"mkdir -p {quoted_path}")
    connection.run(f"chmod {mode:o} {quoted_path}")


def create_remote_job_path(connection: ConnectionProtocol, job_spec: SlurmJobSpec) -> None:
    """Create a remote job files path for the particular job on the cluster.

    Args:
        connection: Connection to the login node
        job_spec (SlurmJobSpec): The job specification, including job name,
            account, partition, and command.

    Raises:
        ValueError: If the remote files path already exists

    """
    # If the directory exists, raise an error because there might be a race condition
    if remote_path_exists(connection, job_spec.remote_job_path):
        error_message = f"Remote files path already exists: {job_spec.remote_job_path}"
        raise ValueError(error_message)

    create_remote_path(connection, job_spec.remote_job_path)


def _unexpected_exit_output(error: invoke.exceptions.UnexpectedExit) -> str:
    parts: list[str] = []
    for stream in ("stderr", "stdout"):
        value = getattr(error.result, stream, "")
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n".join(parts)


def _looks_like_missing_account_error(output: str) -> bool:
    normalized = output.lower()
    return "account" in normalized and any(
        phrase in normalized
        for phrase in (
            "invalid account",
            "missing account",
            "no account",
            "requires an account",
            "specify an account",
            "account is required",
        )
    )


def _raise_helpful_sbatch_error(error: invoke.exceptions.UnexpectedExit, job_spec: SlurmJobSpec) -> None:
    output = _unexpected_exit_output(error)
    if job_spec.account is not None or not _looks_like_missing_account_error(output):
        return

    message = (
        "Slurm rejected the submission because this cluster appears to require an account. "
        f"Rerun with --account <slurm_account> or set {_SLURM_ACCOUNT_ENV_VAR}."
    )
    if output:
        message = f"{message}\n\nsbatch output:\n{output}"
    raise ValueError(message) from error


def curator_submit(slurm_job_spec: SlurmJobSpec) -> str:
    """Submit a curator pipeline batch job to the cluster.

    Args:
        slurm_job_spec: The job specification

    Returns:
        The slurm job id

    Raises:
        ValueError: If the job ID cannot be parsed from the submission
            output, or if required mount source paths do not exist on the cluster.

    """
    connection = connect(slurm_job_spec.login_node, slurm_job_spec.username)
    create_remote_job_path(connection, slurm_job_spec)

    # Validate that all mount source paths exist on the remote cluster
    missing_mounts = [
        mount.source
        for mount in slurm_job_spec.container.mounts
        if not remote_path_exists(connection, Path(mount.source))
    ]

    if missing_mounts:
        error_message = (
            f"The following mount source paths do not exist on the cluster:\n"
            f"{chr(10).join(f'  - {path}' for path in missing_mounts)}\n"
            f"Cannot submit job due to missing mount paths."
        )

        raise ValueError(error_message)

    slurm_job_spec.container.mounts = _merge_mount_specs_by_destination(
        [*slurm_job_spec.container.mounts, MountSpec(source=str(slurm_job_spec.remote_job_path), dest="/remote_files")]
    )
    logger.debug("Container mounts: %s", slurm_job_spec.container.mounts)
    remote_sbatch_path = slurm_job_spec.remote_job_path / "sbatch.sh"

    remote_files = [
        (
            _render_sbatch_script(slurm_job_spec),
            remote_sbatch_path,
            0o700,
        ),
        (
            (Path(__file__).parent / _PROM_SVC_DISC_SCRIPT_PATH).read_text(encoding="utf-8"),
            slurm_job_spec.remote_job_path / _PROM_SVC_DISC_SCRIPT_PATH,
            0o700,
        ),
    ]

    upload_text(connection, remote_files)
    try:
        out = connection.run(f"sbatch {_quote_remote_path(remote_sbatch_path)}")
    except invoke.exceptions.UnexpectedExit as e:
        _raise_helpful_sbatch_error(e, slurm_job_spec)
        raise
    return _parse_job_id(out.stdout)


def remote_find_job_log_file(connection: ConnectionProtocol, log_dir: Path, job_id: str) -> Path:
    """Find a log file for a given job ID in the log directory.

    Args:
        connection: SSH connection to the remote host
        log_dir: Directory to search for log files
        job_id: The job ID to search for

    Returns:
        Path to the log file

    Raises:
        FileNotFoundError: If the log directory doesn't exist or no log file is found

    """
    if not remote_path_exists(connection, log_dir):
        error_message = f"Directory `{log_dir}` does not exist on {connection.host}"
        raise FileNotFoundError(error_message)

    find_pattern = shlex.quote(f"*_{job_id}.log")
    find_result = connection.run(
        f"find {_quote_remote_path(log_dir)} -name {find_pattern} -type f | head -n 1", hide=True
    )
    if not find_result.stdout.strip():
        error_message = f"No log file found for job ID {job_id} in {log_dir}"
        raise FileNotFoundError(error_message)

    return Path(find_result.stdout.strip())


def remote_tail(connection: ConnectionProtocol, file_path: Path) -> None:
    """Tail a file on a remote host using the provided connection.

    Args:
        connection: SSH connection to the remote host
        file_path: Path to the file to tail

    """
    cmd: list[str] = ["tail", "-f", _quote_remote_path(file_path)]
    cmd_str = " ".join(cmd)
    logger.info("Running `%s`, press Ctrl+C to stop", cmd_str)
    connection.run(cmd_str)


def job_log(hostname: str, username: str, job_id: str, log_dir: Path | None = None) -> None:
    """Connect to a login node and tails the log for a specific job ID.

    Args:
        hostname: The hostname of the node that has access to the logs
        username: The username to use for the connection
        job_id: The Slurm job ID to find logs for
        log_dir: The path to the log directory

    Raises:
        ValueError: If no log file is found for the given job ID

    """
    connection = connect(hostname, username)
    _log_dir = _get_log_dir(log_dir)
    log_file = remote_find_job_log_file(connection, _log_dir, job_id)
    remote_tail(connection, log_file)


def job_log_cli(
    *,
    job_id: Annotated[str, Option(help="The Slurm job ID to find logs for", rich_help_panel="common")],
    login_node: Annotated[str, Option(help="Hostname of SLURM login node to use", rich_help_panel="common")],
    username: Annotated[
        str, Option(help=("Optional cluster username"), rich_help_panel="common")
    ] = f"{_get_username()}",
    log_dir: Annotated[Path | None, Option(help="Path to the log directory", rich_help_panel="common")] = None,
) -> None:
    """View the logs for a specific job running on the cluster.

    Either slurm_config or login_node must be provided, but not both.

    """
    job_log(login_node, username, job_id, log_dir)


def _srun_allocation_args(  # noqa: PLR0913
    *,
    account: str | None,
    partition: str | None,
    qos: str | None,
    gres: str | None,
    gpus: str | None,
    job_name: str,
    num_nodes: int | None,
    exclusive: bool,
    time_limit: str | None,
) -> list[str]:
    if num_nodes is not None and num_nodes < 1:
        msg = "--nodes must be at least 1"
        raise typer.BadParameter(msg)

    gres, gpus = _validate_gpu_options(gres=gres, gpus=gpus)
    args = [f"--nodes={num_nodes}"] if num_nodes is not None else []
    optional_args = [
        ("--account", _resolve_slurm_account(account)),
        ("--partition", partition),
        ("--qos", qos),
        ("--gpus", gpus),
        ("--gres", gres),
        ("--time", time_limit),
        ("--job-name", job_name),
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
            job_name=job_name,
            num_nodes=1,
            exclusive=False,
            time_limit=time_limit,
        ),
        "--ntasks=1",
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


def _run_srun_command(login_node: str, username: str, shell_command: SrunCommand) -> None:
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

    ssh_target = f"{username}@{login_node}" if username else login_node
    ssh_command = ["ssh", "-t", ssh_target, _remote_srun_command(shell_command)]
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


def _run_srun_shell(login_node: str, username: str, shell_command: SrunCommand) -> None:
    _run_srun_command(login_node, username, shell_command)


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
            help="Alternative Slurm GPU request in GRES form, e.g. 'gpu:8'. Cannot be used with --gpus.",
            rich_help_panel="cluster",
        ),
    ] = None,
    gpus: Annotated[
        str | None,
        Option(
            "-G",
            "--gpus",
            help="Common Slurm GPU request form, e.g. '8' or 'h100:8'. Cannot be used with --gres.",
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
        pty=True,
    )
    _run_srun_shell(login_node, username, shell_command)


def submit_cli(  # noqa: PLR0913
    command: Annotated[list[str], Argument(help="The command to run", rich_help_panel="common")],
    *,
    login_node: Annotated[
        str,
        Option(
            help="Hostname of SLURM login node to run command on. Defaults to local sbatch submission.",
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
            help=("The slurm partition to use. Omit to use the cluster default."),
            rich_help_panel="cluster",
        ),
    ] = None,
    remote_files_path: Annotated[
        Path,
        Option(
            help="Path to remote files directory, where the sbatch script will be placed",
            rich_help_panel="cluster",
        ),
    ] = _HOME_DIR / "curator_launch_files",
    container_image: Annotated[
        str,
        Option(help=("Canonical path to the .sqsh container image"), rich_help_panel="container"),
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
        Path,
        Option(
            help="Host workspace directory to mount as /config inside the container.",
            rich_help_panel="container",
        ),
    ] = environment.LOCAL_WORKSPACE_PATH,
    cache_path: Annotated[
        Path,
        Option(
            help="Host cache directory to mount as /cache for Pixi/rattler, uv, Torch, Triton, pip, and CUDA caches.",
            rich_help_panel="container",
        ),
    ] = _DEFAULT_CACHE_PATH,
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
                "by CONTAINER_PATH with later entries overriding earlier ones; for example, "
                "/tmp/work:/config:ro replaces the default /config mount."
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
                "Comma-separated container mounts in HOST_PATH:CONTAINER_PATH[:ro|rw] format. Mounts are merged "
                "by CONTAINER_PATH with later entries overriding earlier defaults; for example, "
                "/data/config:/config:ro replaces the default /config mount."
            ),
            rich_help_panel="container",
        ),
    ] = "",
    environment: Annotated[
        str | None,
        Option(
            help="Comma separated list of environment variables to set in the container", rich_help_panel="container"
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
        Option(
            help=("Optional cluster username"),
            rich_help_panel="misc",
        ),
    ] = f"{_get_username()}",
    job_name: Annotated[
        str,
        Option(
            "-J",
            "--job-name",
            help="Name of the job",
            rich_help_panel="cluster",
        ),
    ] = "cosmos_curator",
    num_nodes: Annotated[
        int,
        Option(
            "-N",
            "--nodes",
            "--num-nodes",
            help="Number of nodes to use on the cluster",
            rich_help_panel="cluster",
        ),
    ] = 1,
    gres: Annotated[
        str | None,
        Option(
            help="Alternative Slurm GPU request in GRES form, e.g. 'gpu:8'. Cannot be used with --gpus.",
            rich_help_panel="cluster",
        ),
    ] = None,
    gpus: Annotated[
        str | None,
        Option(
            "-G",
            "--gpus",
            help="Common Slurm GPU request form, e.g. '8' or 'h100:8'. Cannot be used with --gres.",
            rich_help_panel="cluster",
        ),
    ] = None,
    qos: Annotated[
        str | None,
        Option(
            "-q",
            "--qos",
            help="Optional Slurm quality of service to request.",
            rich_help_panel="cluster",
        ),
    ] = None,
    exclusive: Annotated[
        bool,
        Option(
            help="Whether to use nodes exclusively",
            rich_help_panel="cluster",
        ),
    ] = True,
    time: Annotated[
        str | None,
        Option(
            "-t",
            "--time",
            help="Time limit for the job, e.g. 01:00:00 for 1 hour. See sbatch --time for more details.",
            rich_help_panel="cluster",
        ),
    ] = None,
    stop_retries_after: Annotated[
        int,
        Option(
            help="Maximum time in seconds to wait before `ray start` retries end",
            rich_help_panel="cluster",
        ),
    ] = 600,
    exclude_nodes: Annotated[
        str | None,
        Option(help="Comma separated list of nodes to exclude", rich_help_panel="cluster"),
    ] = None,
    log_dir: Annotated[
        Path | None,
        Option(help="Path to the log directory", rich_help_panel="cluster"),
    ] = None,
    comment: Annotated[
        str | None,
        Option(help="Comment to add to the job", rich_help_panel="cluster"),
    ] = None,
    prometheus_service_discovery_path: Annotated[
        Path | None,
        Option(
            help=(
                "Path on the cluster under which to create the Prometheus service discovery file. "
                "If not provided, it will not be created"
            ),
            rich_help_panel="cluster",
        ),
    ] = None,
    mail_type: Annotated[
        str | None,
        Option(
            help=(
                "Comma separated mail notification type: BEGIN, END, FAIL, REQUEUE, ALL, "
                "STAGE_OUT, TIME_LIMIT, TIME_LIMIT_90, TIME_LIMIT_80. If not provided, "
                "and --mail-user is provided, then slurm will likely default to END,FAIL"
            ),
            rich_help_panel="cluster",
        ),
    ] = None,
    mail_user: Annotated[
        str | None,
        Option(
            help="Email address for job notifications",
            rich_help_panel="cluster",
        ),
    ] = None,
) -> None:
    """Submit a job to a SLURM cluster."""
    if not command:
        error_message = "A command must be provided"
        raise ValueError(error_message)

    if mail_type is not None and mail_user is None:
        error_message = "If --mail-type is provided, --mail-user must also be provided"
        raise ValueError(error_message)

    if num_nodes < 1:
        msg = "--nodes must be at least 1"
        raise typer.BadParameter(msg)
    gres, gpus = _validate_gpu_options(gres=gres, gpus=gpus)

    submit_runtime = _build_slurm_container_runtime(
        container_image=container_image,
        curator_path=_infer_curator_path(curator_path),
        command=command,
        workspace_path=workspace_path,
        cache_path=cache_path,
        mount_s3_creds=mount_s3_creds,
        mount_azure_creds=mount_azure_creds,
        extra_mounts=extra_mounts,
        environment=environment,
        conda_override_cuda=conda_override_cuda,
        pixi_envs=pixi_envs,
    )
    container_mount_specs = _container_mount_specs_from_runtime(submit_runtime, container_mounts)
    env_list = _environment_entries_from_srun_defaults(submit_runtime)
    exclude_nodes_list = exclude_nodes.split(",") if exclude_nodes is not None else None

    container_spec = ContainerSpec(
        command=["pixi", "run", "--as-is", str(_START_RAY), *command],
        squashfs_path=_resolve_container_image(container_image),
        environment=env_list,
        mounts=container_mount_specs,
    )

    slurm_job_spec = SlurmJobSpec(
        login_node=login_node,
        username=username,
        log_dir=_get_log_dir(log_dir),
        job_name=job_name,
        remote_job_path=_get_remote_job_path(remote_files_path, job_name),
        account=_resolve_slurm_account(account),
        partition=_normalize_optional_slurm_directive(partition),
        num_nodes=num_nodes,
        container=container_spec,
        gres=gres,
        gpus=gpus,
        qos=_normalize_optional_slurm_directive(qos),
        exclusive=exclusive,
        time_limit=time,
        stop_retries_after=stop_retries_after,
        exclude_nodes=exclude_nodes_list,
        comment=comment,
        prometheus_service_discovery_path=prometheus_service_discovery_path,
        mail_type=mail_type,
        mail_user=mail_user,
    )

    job_id = curator_submit(slurm_job_spec)
    logger.info("Job submitted with ID: %s", job_id)
    typer.echo(f"Job submitted with ID: {job_id}")


slurm_cli = typer.Typer(
    context_settings={
        "max_content_width": 120,
    },
    pretty_exceptions_enable=False,
    no_args_is_help=True,
)

slurm_cli.command("submit", no_args_is_help=True)(submit_cli)
slurm_cli.command("shell")(shell_cli)
slurm_cli.command("import-image", no_args_is_help=True)(import_image_cli)
slurm_cli.command("job-log", no_args_is_help=True)(job_log_cli)


if __name__ == "__main__":
    slurm_cli()
