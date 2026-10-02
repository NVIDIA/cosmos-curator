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
import re
import shlex
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Protocol, cast

import attrs
import fabric  # type: ignore[import-untyped]
import invoke
import jinja2
import typer
from invoke.context import Context
from invoke.runners import Result as InvokeResult
from typer import Argument, Option

from cosmos_curator.client.slurm_cli.slurm_common import (
    _CONDA_ACTIVATION_ENV_VARS,
    _DEFAULT_CACHE_PATH,
    _DEFAULT_CONDA_OVERRIDE_CUDA,
    _DEFAULT_CONTAINER_IMAGE,
    _DEFAULT_LOGIN_NODE,
    _LAUNCHER_ENV_PREFIXES,
    _PIXI_ACTIVATION_ENV_VARS,
    _SLURM_ACCOUNT_ENV_VAR,
    MountSpec,
    SlurmContainerRuntime,
    _build_slurm_container_runtime,
    _container_mount_specs_from_runtime,
    _get_container_entrypoint_command,
    _get_srun_environment,
    _get_username,
    _infer_curator_path,
    _is_local_host,
    _merge_mount_specs_by_destination,
    _normalize_optional_slurm_directive,
    _parse_mount_specs,
    _remote_container_mount_specs_from_runtime,
    _resolve_container_image,
    _resolve_slurm_account,
    _validate_gpu_options,
)
from cosmos_curator.core.utils import environment as core_environment

logger = logging.getLogger(__name__)

_SBATCH_TEMPLATE_PATH = Path("sbatch.sh.j2")
_PROM_SVC_DISC_SCRIPT_PATH = Path("prometheus_service_discovery.py")
_START_RAY = core_environment.CONTAINER_PATHS_CODE_DIR / "cosmos_curator" / "scripts" / "onto_slurm.py"
_MAX_FILE_MODE = 0o7777
_HOME_DIR = Path(os.getenv("REMOTE_HOME_DIR", Path.home()))
_SBATCH_DYNAMIC_CONTAINER_ENV_KEYS = (
    core_environment.CURATOR_IO_SLOTS_PER_NODE_ENV_VAR,
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
        result = self._context.run(command, **kwargs)
        if result is None:
            msg = "LocalConnection.run does not support disown=True"
            raise ValueError(msg)
        return result

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


def _environment_entries_from_srun_defaults(opts: SlurmContainerRuntime) -> list[str]:
    env, container_env_keys = _get_srun_environment(opts, include_slurm_env=False)
    return [f"{key}={env[key]}" for key in container_env_keys if key in env]


def _environment_entries_to_map(entries: list[str]) -> dict[str, str]:
    env_vars: dict[str, str] = {}
    for env_entry in entries:
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
    return env_vars


@attrs.define
class ContainerSpec:
    """Configuration for a container to run in the SLURM job."""

    command: list[str]
    squashfs_path: str
    mounts: list[MountSpec]
    environment: list[str]


@attrs.define
class SlurmSubmitOptions:
    """User-facing Slurm submit options before normalization into a job spec."""

    login_node: str = _DEFAULT_LOGIN_NODE
    username: str | None = None
    account: str | None = None
    partition: str | None = None
    remote_files_path: Path = _HOME_DIR / "curator_launch_files"
    remote_job_path: Path | None = None
    container_image: str = _DEFAULT_CONTAINER_IMAGE
    curator_path: Path | None = None
    workspace_path: Path = core_environment.LOCAL_WORKSPACE_PATH
    cache_path: Path = _DEFAULT_CACHE_PATH
    mount_s3_creds: bool = True
    mount_azure_creds: bool = False
    container_mounts: str | None = None
    extra_mounts: str = ""
    node_local_mounts: str = ""
    prepare_node_local_mounts: bool = False
    environment: str | None = None
    extra_environment: list[str] = attrs.field(factory=list)
    conda_override_cuda: str | None = _DEFAULT_CONDA_OVERRIDE_CUDA
    pixi_envs: str | None = None
    job_name: str = "cosmos_curator"
    num_nodes: int = 1
    gres: str | None = None
    gpus: str | None = None
    gpus_per_node: str | None = None
    qos: str | None = None
    exclusive: bool = True
    time: str | None = None
    stop_retries_after: int = 600
    ray_io_slots_per_node: int = core_environment.DEFAULT_CURATOR_IO_SLOTS_PER_NODE
    exclude_nodes: str | None = None
    log_dir: Path | None = None
    comment: str | None = None
    prometheus_service_discovery_path: Path | None = None
    mail_type: str | None = None
    mail_user: str | None = None
    extra_remote_files: list[tuple[str, Path, int]] = attrs.field(factory=list)
    source_environment_file: Path | None = None
    extra_container_env_keys: list[str] = attrs.field(factory=list)


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
    gpus_per_node: str | None = None
    qos: str | None = None
    time_limit: str | None = None
    stop_retries_after: int = 600
    ray_io_slots_per_node: int = core_environment.DEFAULT_CURATOR_IO_SLOTS_PER_NODE
    exclude_nodes: list[str] | None = None
    comment: str | None = None
    prometheus_service_discovery_path: Path | None = None
    mail_type: str | None = None
    mail_user: str | None = None
    extra_remote_files: list[tuple[str, Path, int]] = attrs.field(factory=list)
    source_environment_file: Path | None = None
    extra_container_env_keys: list[str] = attrs.field(factory=list)
    node_local_mount_sources: list[str] = attrs.field(factory=list)
    prepare_node_local_mounts: bool = False


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

    env_vars = _environment_entries_to_map(spec.container.environment or [])
    container_env_keys = list(
        dict.fromkeys([*env_vars, *spec.extra_container_env_keys, *_SBATCH_DYNAMIC_CONTAINER_ENV_KEYS])
    )

    return jinja2.Template(sbatch_template.read_text()).render(
        job_name=spec.job_name,
        account=spec.account,
        partition=spec.partition,
        num_nodes=spec.num_nodes,
        command=command,
        gres=spec.gres,
        gpus=spec.gpus,
        gpus_per_node=spec.gpus_per_node,
        qos=spec.qos,
        exclusive=spec.exclusive,
        container_image=spec.container.squashfs_path,
        container_mounts=container_mounts,
        container_command=container_command,
        container_env_keys=container_env_keys,
        env_vars=env_vars,
        source_environment_file=str(spec.source_environment_file) if spec.source_environment_file else "",
        launcher_env_prefixes_to_unset=_LAUNCHER_ENV_PREFIXES,
        launcher_env_vars_to_unset=(*_PIXI_ACTIVATION_ENV_VARS, *_CONDA_ACTIVATION_ENV_VARS),
        time_limit_string=spec.time_limit,
        stop_retries_after=spec.stop_retries_after,
        ray_io_slots_per_node=spec.ray_io_slots_per_node,
        exclude_nodes=spec.exclude_nodes,
        log_dir=str(spec.log_dir),
        comment=spec.comment,
        enable_metrics_scraping="yes" if spec.prometheus_service_discovery_path is not None else "no",
        job_artifact_path=str(spec.remote_job_path),
        prometheus_service_discovery_path=str(spec.prometheus_service_discovery_path),
        mail_type=spec.mail_type,
        mail_user=spec.mail_user,
        node_local_mount_sources=list(dict.fromkeys(spec.node_local_mount_sources)),
        prepare_node_local_mounts=spec.prepare_node_local_mounts,
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


def _with_remote_files_mount(slurm_job_spec: SlurmJobSpec) -> SlurmJobSpec:
    remote_files_mount = MountSpec(source=str(slurm_job_spec.remote_job_path), dest="/remote_files")
    if remote_files_mount in slurm_job_spec.container.mounts:
        return slurm_job_spec

    container = attrs.evolve(
        slurm_job_spec.container,
        mounts=_merge_mount_specs_by_destination(
            [
                *slurm_job_spec.container.mounts,
                remote_files_mount,
            ]
        ),
    )
    return attrs.evolve(slurm_job_spec, container=container)


def _create_extra_remote_file_dirs(connection: ConnectionProtocol, job_spec: SlurmJobSpec) -> None:
    if not job_spec.extra_remote_files:
        return

    extra_remote_dirs = {
        remote_path.parent
        for _, remote_path, _ in job_spec.extra_remote_files
        if remote_path.parent != job_spec.remote_job_path
    }
    for remote_dir in sorted(extra_remote_dirs, key=str):
        create_remote_path(connection, remote_dir)


def _upload_extra_remote_files(connection: ConnectionProtocol, job_spec: SlurmJobSpec) -> None:
    if not job_spec.extra_remote_files:
        return

    upload_text(connection, job_spec.extra_remote_files)


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
    _create_extra_remote_file_dirs(connection, slurm_job_spec)

    # Validate that all mount source paths exist on the remote cluster
    node_local_mount_sources = set(slurm_job_spec.node_local_mount_sources)
    missing_mounts = [
        mount.source
        for mount in slurm_job_spec.container.mounts
        if mount.source not in node_local_mount_sources and not remote_path_exists(connection, Path(mount.source))
    ]

    if missing_mounts:
        error_message = (
            f"The following mount source paths do not exist on the cluster:\n"
            f"{chr(10).join(f'  - {path}' for path in missing_mounts)}\n"
            f"Cannot submit job due to missing mount paths."
        )

        raise ValueError(error_message)

    slurm_job_spec = _with_remote_files_mount(slurm_job_spec)
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
    _upload_extra_remote_files(connection, slurm_job_spec)
    try:
        out = connection.run(f"sbatch {_quote_remote_path(remote_sbatch_path)}")
    except invoke.exceptions.UnexpectedExit as e:
        _raise_helpful_sbatch_error(e, slurm_job_spec)
        raise
    return _parse_job_id(out.stdout)


def render_slurm_submit_script(slurm_job_spec: SlurmJobSpec) -> str:
    """Render the sbatch script that would be uploaded by :func:`curator_submit`."""
    return _render_sbatch_script(_with_remote_files_mount(slurm_job_spec))


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


def build_slurm_submit_job_spec(command: list[str], options: SlurmSubmitOptions | None = None) -> SlurmJobSpec:
    """Build a Slurm job spec for batch submission without submitting it."""
    opts = options or SlurmSubmitOptions()
    if not command:
        error_message = "A command must be provided"
        raise ValueError(error_message)

    if opts.mail_type is not None and opts.mail_user is None:
        error_message = "If --mail-type is provided, --mail-user must also be provided"
        raise ValueError(error_message)

    if opts.num_nodes < 1:
        msg = "--nodes must be at least 1"
        raise typer.BadParameter(msg)
    if opts.ray_io_slots_per_node < 1:
        msg = "--ray-io-slots-per-node must be at least 1"
        raise typer.BadParameter(msg)

    gres, gpus, gpus_per_node = _validate_gpu_options(gres=opts.gres, gpus=opts.gpus, gpus_per_node=opts.gpus_per_node)
    submit_runtime = _build_slurm_container_runtime(
        container_image=opts.container_image,
        curator_path=opts.curator_path,
        command=command,
        workspace_path=opts.workspace_path,
        cache_path=opts.cache_path,
        mount_s3_creds=opts.mount_s3_creds,
        mount_azure_creds=opts.mount_azure_creds,
        extra_mounts=opts.extra_mounts,
        environment=opts.environment,
        conda_override_cuda=opts.conda_override_cuda,
        pixi_envs=opts.pixi_envs,
    )
    mount_builder = (
        _container_mount_specs_from_runtime
        if _is_local_host(opts.login_node)
        else _remote_container_mount_specs_from_runtime
    )
    container_mount_specs = mount_builder(submit_runtime, opts.container_mounts)
    node_local_mount_specs = _parse_mount_specs(opts.node_local_mounts)
    container_mount_specs = _merge_mount_specs_by_destination([*container_mount_specs, *node_local_mount_specs])
    env_list = _environment_entries_from_srun_defaults(submit_runtime)
    env_list.extend(opts.extra_environment)
    exclude_nodes_list = opts.exclude_nodes.split(",") if opts.exclude_nodes is not None else None

    container_spec = ContainerSpec(
        command=["pixi", "run", "--as-is", str(_START_RAY), *command],
        squashfs_path=_resolve_container_image(opts.container_image),
        environment=env_list,
        mounts=container_mount_specs,
    )

    return SlurmJobSpec(
        login_node=opts.login_node,
        username=opts.username or _get_username(),
        log_dir=_get_log_dir(opts.log_dir),
        job_name=opts.job_name,
        remote_job_path=opts.remote_job_path or _get_remote_job_path(opts.remote_files_path, opts.job_name),
        account=_resolve_slurm_account(opts.account),
        partition=_normalize_optional_slurm_directive(opts.partition),
        num_nodes=opts.num_nodes,
        container=container_spec,
        gres=gres,
        gpus=gpus,
        gpus_per_node=gpus_per_node,
        qos=_normalize_optional_slurm_directive(opts.qos),
        exclusive=opts.exclusive,
        time_limit=opts.time,
        stop_retries_after=opts.stop_retries_after,
        ray_io_slots_per_node=opts.ray_io_slots_per_node,
        exclude_nodes=exclude_nodes_list,
        comment=opts.comment,
        prometheus_service_discovery_path=opts.prometheus_service_discovery_path,
        mail_type=opts.mail_type,
        mail_user=opts.mail_user,
        extra_remote_files=list(opts.extra_remote_files),
        source_environment_file=opts.source_environment_file,
        extra_container_env_keys=list(opts.extra_container_env_keys),
        node_local_mount_sources=[mount.source for mount in node_local_mount_specs],
        prepare_node_local_mounts=opts.prepare_node_local_mounts,
    )


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
    ] = core_environment.LOCAL_WORKSPACE_PATH,
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
    node_local_mounts: Annotated[
        str,
        Option(
            "--node-local-mounts",
            help=(
                "Comma-separated container mounts whose source paths exist only on allocated compute nodes. "
                "These mounts are included in srun but skipped during login-node source validation."
            ),
            rich_help_panel="container",
        ),
    ] = "",
    prepare_node_local_mounts: Annotated[
        bool,
        Option(
            "--prepare-node-local-mounts",
            help="Create node-local mount source directories on allocated compute nodes before container launch.",
            rich_help_panel="container",
        ),
    ] = False,
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
    ray_io_slots_per_node: Annotated[
        int,
        Option(
            "--ray-io-slots-per-node",
            help="Logical source-IO capacity advertised by each Ray node.",
            rich_help_panel="cluster",
            min=1,
        ),
    ] = core_environment.DEFAULT_CURATOR_IO_SLOTS_PER_NODE,
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
    dry_run: Annotated[
        bool,
        Option(
            "--dry-run",
            help="Render the sbatch script and exit without uploading files or submitting the job.",
            rich_help_panel="misc",
        ),
    ] = False,
) -> None:
    """Submit a job to a SLURM cluster."""
    slurm_job_spec = build_slurm_submit_job_spec(
        command,
        SlurmSubmitOptions(
            login_node=login_node,
            username=username,
            account=account,
            partition=partition,
            remote_files_path=remote_files_path,
            container_image=container_image,
            curator_path=_infer_curator_path(curator_path),
            workspace_path=workspace_path,
            cache_path=cache_path,
            mount_s3_creds=mount_s3_creds,
            mount_azure_creds=mount_azure_creds,
            container_mounts=container_mounts,
            extra_mounts=extra_mounts,
            node_local_mounts=node_local_mounts,
            prepare_node_local_mounts=prepare_node_local_mounts,
            environment=environment,
            conda_override_cuda=conda_override_cuda,
            pixi_envs=pixi_envs,
            job_name=job_name,
            num_nodes=num_nodes,
            gres=gres,
            gpus=gpus,
            gpus_per_node=gpus_per_node,
            qos=qos,
            exclusive=exclusive,
            time=time,
            stop_retries_after=stop_retries_after,
            ray_io_slots_per_node=ray_io_slots_per_node,
            exclude_nodes=exclude_nodes,
            log_dir=log_dir,
            comment=comment,
            prometheus_service_discovery_path=prometheus_service_discovery_path,
            mail_type=mail_type,
            mail_user=mail_user,
        ),
    )

    if dry_run:
        typer.echo(render_slurm_submit_script(slurm_job_spec))
        return

    job_id = curator_submit(slurm_job_spec)
    logger.info("Job submitted with ID: %s", job_id)
    typer.echo(f"Job submitted with ID: {job_id}")
