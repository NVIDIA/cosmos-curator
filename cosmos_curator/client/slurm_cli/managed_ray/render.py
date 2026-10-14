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
"""Runtime path resolution and batch script rendering for managed Slurm-Ray runs."""

import os
import shlex
from collections.abc import Callable
from pathlib import Path

from cosmos_curator.client.environment import (
    CONTAINER_PATHS_COSMOS_CURATOR_CONFIG_FILE,
    CONTAINER_PATHS_DEFAULT_WORKSPACE_DIR,
)
from cosmos_curator.client.slurm_cli.managed_ray.config import (
    SlurmRayAllocationConfig,
    SlurmRayConfig,
    SlurmRayConfigError,
    SlurmRayMount,
    expand_host_path,
    resolve_container_image,
    startup_timeout_seconds,
    validate_state_dir,
)
from cosmos_curator.client.slurm_cli.managed_ray.onnode.slurm_ray_runtime import (
    RUNTIME_MODULE_FILENAME,
    WORKER_HEAD_JOB_ID_ENV,
    WORKER_LANE_ENV,
)
from cosmos_curator.client.slurm_cli.managed_ray.onnode.slurm_ray_state import (
    BOOTSTRAP_FILENAME,
    ENVIRONMENT_FILENAME,
    LOG_DIR_NAME,
    MANIFEST_FILENAME,
    JsonObject,
    SlurmRayRuntimePaths,
)
from cosmos_curator.client.slurm_cli.slurm_common import (
    _CACHE_MOUNT_PATH,
    _CONDA_ACTIVATION_ENV_VARS,
    _CONTAINER_AZURE_CREDS_PATH,
    _CONTAINER_S3_CREDS_PATH,
    _CONTAINER_SOURCE_DIR,
    _DEFAULT_CONDA_OVERRIDE_CUDA,
    _LAUNCHER_ENV_PREFIXES,
    _LOG_ENV_VARS_TO_FORWARD,
    _PIXI_ACTIVATION_ENV_VARS,
    _SLURM_ENV_VARS_TO_FORWARD,
    MountSpec,
    _base_container_environment,
    _build_container_srun_argv,
    _merge_mount_specs_by_destination,
    _resolve_forwarded_environment,
)

_CONTAINER_RUN_STATE_DIR = "/run/cosmos-curator/slurm-ray"
_CONTAINER_MANIFEST = f"{_CONTAINER_RUN_STATE_DIR}/{MANIFEST_FILENAME}"
_CONTAINER_BOOTSTRAP = f"{_CONTAINER_RUN_STATE_DIR}/{BOOTSTRAP_FILENAME}"
_CONTAINER_RUNTIME_MODULE = f"{_CONTAINER_RUN_STATE_DIR}/{RUNTIME_MODULE_FILENAME}"


def _configured_mount_spec(mount: SlurmRayMount, home: Path) -> MountSpec:
    return MountSpec(
        source=str(expand_host_path(mount.source, home)),
        dest=mount.destination,
        mode=mount.mode,
    )


def resolve_runtime_paths(  # noqa: PLR0913
    config: SlurmRayConfig,
    *,
    home: Path,
    run_id: str,
    state_dir: str,
    host_path_exists: Callable[[Path], bool],
    forwarded_environment_keys: list[str],
) -> SlurmRayRuntimePaths:
    """Resolve the container image, mounts, and run directory recorded in the manifest.

    ``host_path_exists`` decides whether an optional credential mount has a source to bind, which is the only
    thing here that has to ask the cluster rather than compute an answer.
    """
    directory = expand_host_path(validate_state_dir(state_dir), home) / run_id
    runtime = config.runtime

    mounts: list[MountSpec] = [
        MountSpec(
            source=str(expand_host_path(runtime.workspace_path, home)),
            dest=str(CONTAINER_PATHS_DEFAULT_WORKSPACE_DIR),
        ),
        MountSpec(
            source=str(expand_host_path(runtime.cache_path, home)),
            dest=str(_CACHE_MOUNT_PATH),
        ),
    ]
    if runtime.curator_path is not None:
        mounts.append(
            MountSpec(
                source=str(expand_host_path(runtime.curator_path, home)),
                dest=str(_CONTAINER_SOURCE_DIR),
            )
        )

    credential_mounts = [
        (
            runtime.mount_s3_creds,
            home / ".aws" / "credentials",
            str(_CONTAINER_S3_CREDS_PATH),
        ),
        (
            runtime.mount_azure_creds,
            home / ".azure" / "credentials",
            str(_CONTAINER_AZURE_CREDS_PATH),
        ),
        (
            True,
            home / ".config" / "cosmos_curator" / "config.yaml",
            str(CONTAINER_PATHS_COSMOS_CURATOR_CONFIG_FILE),
        ),
    ]
    for enabled, source, destination in credential_mounts:
        if enabled and host_path_exists(source):
            mounts.append(MountSpec(source=str(source), dest=destination, mode="ro"))

    mounts.extend(_configured_mount_spec(mount, home) for mount in (*runtime.mounts, *runtime.node_local_mounts))

    # Node-local mount sources exist only on an allocated node, so they have to be created before the container
    # starts. The launcher's own sources are always created; user-declared ones are opt-in.
    prepare_directories: list[str] = []
    if runtime.prepare_node_local_mounts:
        prepare_directories.extend(str(expand_host_path(mount.source, home)) for mount in runtime.node_local_mounts)

    ray_temp_dir: str | None = None
    if config.ray.temp_dir is not None:
        ray_temp_dir = str(expand_host_path(config.ray.temp_dir, home))
        mounts.append(MountSpec(source=ray_temp_dir, dest=ray_temp_dir))
        prepare_directories.append(ray_temp_dir)

    mounts.append(MountSpec(source=str(directory), dest=_CONTAINER_RUN_STATE_DIR))
    mount_payloads: list[JsonObject] = [
        {"source": mount.source, "destination": mount.dest, "mode": mount.mode}
        for mount in _merge_mount_specs_by_destination(mounts)
    ]

    return {
        "run_dir": str(directory),
        "container_image": resolve_container_image(runtime.container_image, home),
        "mounts": mount_payloads,
        "prepare_directories": list(dict.fromkeys(prepare_directories)),
        "forwarded_environment_keys": forwarded_environment_keys,
        "ray_temp_dir": ray_temp_dir,
    }


def capture_forwarded_environment(config: SlurmRayConfig) -> dict[str, str]:
    """Capture the launching environment values this run forwards into its containers."""
    values = {
        name: os.environ[name] for name in _LOG_ENV_VARS_TO_FORWARD if name != "CURATOR_RUN_ID" and name in os.environ
    }
    values.update(_resolve_forwarded_environment(config.runtime.environment, source=os.environ))
    return values


def render_environment_file(values: dict[str, str]) -> str:
    """Render the private per-run environment file sourced by both batch scripts."""
    lines = ["# Private environment captured by cosmos-curator slurm ray submit."]
    lines.extend(f"export {name}={shlex.quote(value)}" for name, value in values.items())
    return "\n".join(lines) + "\n"


def run_dir(runtime_paths: SlurmRayRuntimePaths) -> Path:
    """Return the private run directory that every other run path derives from."""
    return Path(runtime_paths["run_dir"])


def _container_environment_values(config: SlurmRayConfig) -> dict[str, str]:
    return _base_container_environment(
        conda_override_cuda=_DEFAULT_CONDA_OVERRIDE_CUDA,
        pixi_envs=config.runtime.pixi_envs,
    )


def _container_environment_keys(
    config: SlurmRayConfig,
    runtime_paths: SlurmRayRuntimePaths,
    *,
    extra: tuple[str, ...] = (),
) -> list[str]:
    return list(
        dict.fromkeys(
            [
                *_container_environment_values(config),
                *runtime_paths["forwarded_environment_keys"],
                "CURATOR_RUN_ID",
                *_SLURM_ENV_VARS_TO_FORWARD,
                *extra,
            ]
        )
    )


def _render_environment_setup(
    config: SlurmRayConfig,
    runtime_paths: SlurmRayRuntimePaths,
) -> list[str]:
    values = _container_environment_values(config)
    lines = [
        f"unset {' '.join((*_PIXI_ACTIVATION_ENV_VARS, *_CONDA_ACTIVATION_ENV_VARS))}",
        (
            "for launcher_env_var in "
            + " ".join(f"${{!{prefix}@}}" for prefix in _LAUNCHER_ENV_PREFIXES)
            + '; do unset "$launcher_env_var"; done'
        ),
        f"source {shlex.quote(str(run_dir(runtime_paths) / ENVIRONMENT_FILENAME))} || exit 1",
    ]
    lines.extend(f"export {name}={shlex.quote(value)}" for name, value in values.items())
    return lines


def _container_launch_command(
    config: SlurmRayConfig,
    runtime_paths: SlurmRayRuntimePaths,
    command: list[str],
    *,
    extra_environment_keys: tuple[str, ...] = (),
    extra_slurm_args: tuple[str, ...] = (),
) -> str:
    srun_command = _build_container_srun_argv(
        container_image=runtime_paths["container_image"],
        container_mounts=[
            f"{mount['source']}:{mount['destination']}:{mount['mode']}" for mount in runtime_paths["mounts"]
        ],
        container_env_keys=_container_environment_keys(
            config,
            runtime_paths,
            extra=extra_environment_keys,
        ),
        command=command,
        slurm_args=["--mpi=none", "--nodes=1", "--ntasks=1", "--ntasks-per-node=1", *extra_slurm_args],
    )
    return shlex.join(srun_command)


def _slurm_directive(name: str, value: str | None) -> list[str]:
    if value is None:
        return []
    return [f"#SBATCH --{name}={value}"]


def _slurm_output_directive(runtime_paths: SlurmRayRuntimePaths, *, array: bool = False) -> str:
    log_dir = str(run_dir(runtime_paths) / LOG_DIR_NAME)
    if any(character.isspace() for character in log_dir) or any(character in log_dir for character in "#%"):
        # `validate_state_dir` already cleared the configured prefix; this catches what home expansion added.
        msg = f"Resolved Slurm log directory contains unsupported directive characters: {log_dir!r}"
        raise SlurmRayConfigError(msg)
    # An array task's %j is its own unrelated allocation number, so name renewing lanes by array job and task
    # instead. A plain job keeps %j, because %a on a non-array job expands to Slurm's no-task sentinel.
    suffix = "%A_%a" if array else "%j"
    return f"#SBATCH --output={log_dir}/%x-{suffix}.out"


def job_name(base_name: str, run_id: str, role: str) -> str:
    """Return the deterministic per-run Slurm job name that makes a lost sbatch response recoverable."""
    return f"{base_name}-{run_id}-{role}"


def _batch_prelude(
    config: SlurmRayConfig,
    *,
    run_id: str,
    role: str,
    allocation: SlurmRayAllocationConfig,
) -> list[str]:
    """Render each role's independently configured Slurm allocation."""
    return [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name(config.job_name, run_id, role)}",
        *_slurm_directive("account", config.slurm.account),
        *_slurm_directive("partition", allocation.partition),
        *_slurm_directive("qos", allocation.qos),
        *_slurm_directive("time", allocation.time),
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --propagate=NONE",
        *_slurm_directive("gpus", str(allocation.gpus) if allocation.gpus is not None else None),
        *_slurm_directive(
            "gpus-per-node", str(allocation.gpus_per_node) if allocation.gpus_per_node is not None else None
        ),
        *(
            ["#SBATCH --exclusive", "#SBATCH --mem=0"]
            if allocation.exclusive
            else [f"#SBATCH --cpus-per-task={allocation.cpus}", f"#SBATCH --mem={allocation.memory}"]
        ),
    ]


def _runtime_setup(
    config: SlurmRayConfig,
    *,
    run_id: str,
    runtime_paths: SlurmRayRuntimePaths,
) -> list[str]:
    """Render environment and node-local setup shared by head and workers."""
    return [
        *_render_environment_setup(config, runtime_paths),
        f"export CURATOR_RUN_ID={shlex.quote(run_id)}",
        *(f"mkdir -p -- {shlex.quote(source)}" for source in runtime_paths["prepare_directories"]),
    ]


def render_head_script(
    config: SlurmRayConfig,
    *,
    run_id: str,
    runtime_paths: SlurmRayRuntimePaths,
    command: list[str],
) -> str:
    """Render the non-requeueable head batch job."""
    timeout = startup_timeout_seconds(config.ray.startup_timeout)
    head_command = [
        "pixi",
        "run",
        "--as-is",
        "python",
        _CONTAINER_RUNTIME_MODULE,
        "head",
        "--run-id",
        run_id,
        "--manifest",
        _CONTAINER_MANIFEST,
        "--startup-timeout-seconds",
        str(timeout),
        "--io-slots-per-node",
        str(config.ray.io_slots_per_node),
    ]
    if runtime_paths["ray_temp_dir"] is not None:
        head_command.extend(["--temp-dir", runtime_paths["ray_temp_dir"]])
    head_command.extend(["--", *command])

    head = config.slurm.head
    lines = [
        *_batch_prelude(
            config,
            run_id=run_id,
            role="head",
            allocation=head,
        ),
        "#SBATCH --no-requeue",
        _slurm_output_directive(runtime_paths),
        "",
        "set -uo pipefail",
        'if [ "${SLURM_RESTART_COUNT:-0}" -ne 0 ]; then',
        '  echo "The managed Ray head must not be requeued" >&2',
        "  exit 1",
        "fi",
        "",
        # Cleanup runs outside the container so that it still happens when the supervisor itself was killed. It
        # cancels every recorded lane, waits for Slurm to release them, and publishes the terminal run state.
        "finalize_run() {",
        "  trap - EXIT",
        f"  {_cleanup_command(run_id, runtime_paths)} >&2",
        '  exit "$1"',
        "}",
        "trap 'finalize_run \"$?\"' EXIT",
        "trap 'exit 143' TERM",
        "trap 'exit 130' INT",
        "",
        *_runtime_setup(config, run_id=run_id, runtime_paths=runtime_paths),
        "",
        _container_launch_command(
            config,
            runtime_paths,
            head_command,
            # Whole-node allocations use all cores; sliced allocations must restate the CPU request at the step.
            extra_slurm_args=() if head.exclusive else (f"--cpus-per-task={head.cpus}",),
        ),
        "",
    ]
    return "\n".join(lines)


def _host_runtime_command(runtime_paths: SlurmRayRuntimePaths, *arguments: str) -> str:
    """Build a host-side invocation of the runtime module this run shipped with.

    These run outside the container on an allocated node, using whatever ``python3`` it provides, so the
    scheduler logic they need exists once in Python instead of once more in bash.
    """
    return shlex.join(["python3", str(run_dir(runtime_paths) / RUNTIME_MODULE_FILENAME), *arguments])


def _cleanup_command(run_id: str, runtime_paths: SlurmRayRuntimePaths) -> str:
    """Build the host-side cleanup invocation used by the head's exit trap."""
    return _host_runtime_command(
        runtime_paths,
        "cleanup",
        "--run-id",
        run_id,
        "--manifest",
        str(run_dir(runtime_paths) / MANIFEST_FILENAME),
    )


def render_worker_script(
    config: SlurmRayConfig,
    *,
    run_id: str,
    runtime_paths: SlurmRayRuntimePaths,
    allocations: int,
) -> str:
    """Render the reusable requeueable worker batch job for one run.

    ``allocations`` is the lane budget the manifest recorded, passed in rather than derived again here so the
    script's log naming and the submitter's ``--array`` cannot disagree about whether a lane renews.
    """
    timeout = startup_timeout_seconds(config.ray.startup_timeout)
    worker_command = [
        "pixi",
        "run",
        "--as-is",
        "python",
        _CONTAINER_RUNTIME_MODULE,
        "worker",
        "--run-id",
        run_id,
        "--bootstrap",
        _CONTAINER_BOOTSTRAP,
        "--startup-timeout-seconds",
        str(timeout),
        "--io-slots-per-node",
        str(config.ray.io_slots_per_node),
    ]
    if runtime_paths["ray_temp_dir"] is not None:
        worker_command.extend(["--temp-dir", runtime_paths["ray_temp_dir"]])

    worker = config.slurm.worker
    lines = [
        *_batch_prelude(
            config,
            run_id=run_id,
            role="worker",
            allocation=worker,
        ),
        # Preemption requeues the same allocation rather than consuming one of the lane's, so preemption survival
        # and walltime succession compose instead of competing for the same budget.
        "#SBATCH --requeue",
        "#SBATCH --open-mode=append",
        _slurm_output_directive(runtime_paths, array=allocations > 1),
        "",
        "set -uo pipefail",
        'if [ "$#" -ne 3 ]; then',
        '  echo "Worker wrapper requires RUN_ID HEAD_JOB_ID LANE" >&2',
        "  exit 2",
        "fi",
        "submitted_run_id=$1",
        "head_job_id=$2",
        "lane=$3",
        f'if [ "$submitted_run_id" != {shlex.quote(run_id)} ]; then',
        '  echo "Worker run ID does not match its submitted script" >&2',
        "  exit 1",
        "fi",
        'case "$lane" in (*[!0-9]*|"") echo "Worker lane must be a nonnegative integer" >&2; exit 2;; esac',
        # Refuse to join a run whose head is already gone, before this allocation starts anything.
        f'{_host_runtime_command(runtime_paths, "probe-head", "--job-id")} "$head_job_id" || exit 1',
        "",
        *_runtime_setup(config, run_id=run_id, runtime_paths=runtime_paths),
        f'export {WORKER_HEAD_JOB_ID_ENV}="$head_job_id"',
        f'export {WORKER_LANE_ENV}="$lane"',
        "",
        _container_launch_command(
            config,
            runtime_paths,
            worker_command,
            extra_environment_keys=(WORKER_HEAD_JOB_ID_ENV, WORKER_LANE_ENV),
            # An exclusive allocation already owns every core, but a sliced one must restate its request:
            # since Slurm 22.05 a step does not inherit the batch job's --cpus-per-task and would land on one core.
            extra_slurm_args=() if worker.exclusive else (f"--cpus-per-task={worker.cpus}",),
        ),
        "",
    ]
    return "\n".join(lines)
