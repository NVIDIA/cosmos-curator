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
"""Manifest-backed lifecycle orchestration for managed Ray clusters on Slurm.

This module owns the commands that change a run; the read-only view lives in :mod:`.status`.
"""

import uuid
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import NoReturn

from cosmos_curator.client.slurm_cli.managed_ray.config import (
    DEFAULT_STATE_DIR,
    SlurmRayConfig,
    lane_allocations,
)
from cosmos_curator.client.slurm_cli.managed_ray.onnode.slurm_ray_state import (
    ACTIVE_HEAD_SLURM_STATES,
    ENVIRONMENT_FILENAME,
    HEAD_SCRIPT_FILENAME,
    LOG_DIR_NAME,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    TERMINAL_RUN_STATES,
    WORKER_SCRIPT_FILENAME,
    JsonObject,
    SlurmRayLane,
    SlurmRayManifest,
    SlurmRayRuntimePaths,
    run_state,
    utc_now,
    validate_manifest,
)
from cosmos_curator.client.slurm_cli.managed_ray.remote import (
    RemoteRun,
    SlurmRayOperationError,
    atomic_upload_json,
    list_run_manifest_paths,
    mutate_remote_manifest,
    open_remote_run,
    read_remote_json_files,
    remote_home,
    slurm_cluster_name,
    upload_runtime_modules,
    verify_remote_python,
)
from cosmos_curator.client.slurm_cli.managed_ray.render import (
    HOST_CLEANUP_FILENAME,
    capture_forwarded_environment,
    job_name,
    render_environment_file,
    render_head_script,
    render_host_cleanup_script,
    render_worker_script,
    resolve_runtime_paths,
    run_dir,
)
from cosmos_curator.client.slurm_cli.managed_ray.scheduler import (
    SlurmJobState,
    cancel_jobs,
    is_nonterminal,
    query_job_states,
    submit_script,
)
from cosmos_curator.client.slurm_cli.slurm_common import _get_username
from cosmos_curator.client.slurm_cli.slurm_submit import (
    ConnectionProtocol,
    connect,
    create_remote_path,
    remote_path_exists,
    upload_text,
)

# A head that is up, or not yet up. Both can still adopt a lane, because a lane waits on the head job before it
# starts; a head on its way out cannot.
_SCALABLE_HEAD_STATES = ACTIVE_HEAD_SLURM_STATES | {"PENDING"}


class SlurmRayPartialOperationError(SlurmRayOperationError):
    """Raised after a lifecycle mutation has already changed Slurm state."""

    def __init__(self, message: str, *, result: JsonObject) -> None:
        """Preserve the observable result alongside the operation error."""
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class SlurmRaySubmission:
    """Identifiers returned after a successful initial submission."""

    run_id: str
    slurm_cluster_name: str
    state_dir: str
    manifest_path: str
    log_dir: str
    head_job_id: str
    lane_job_ids: list[str]

    def json_payload(self) -> JsonObject:
        """Return the stable machine-readable submission result."""
        return {
            "run_id": self.run_id,
            "slurm_cluster_name": self.slurm_cluster_name,
            "state_dir": self.state_dir,
            "manifest_path": self.manifest_path,
            "log_dir": self.log_dir,
            "head_job_id": self.head_job_id,
            "lane_job_ids": self.lane_job_ids,
        }


@dataclass(frozen=True)
class SlurmRayScale:
    """The lane reconciliation one ``scale`` command performed."""

    run_id: str
    target_worker_lanes: int
    previous_worker_lanes: int
    submitted_job_ids: list[str]
    canceled_job_ids: list[str]

    @property
    def worker_lanes(self) -> int:
        """Return the lane count this command left behind, which need not be the target it was given."""
        return self.previous_worker_lanes + len(self.submitted_job_ids) - len(self.canceled_job_ids)

    def json_payload(self) -> JsonObject:
        """Return the stable machine-readable scale result."""
        return {
            "run_id": self.run_id,
            "target_worker_lanes": self.target_worker_lanes,
            "previous_worker_lanes": self.previous_worker_lanes,
            "worker_lanes": self.worker_lanes,
            "submitted_job_ids": list(self.submitted_job_ids),
            "canceled_job_ids": list(self.canceled_job_ids),
        }


@dataclass(frozen=True)
class SlurmRayStop:
    """The jobs one ``stop`` command canceled, and the state it left the run in."""

    run_id: str
    state: str
    canceled_lane_job_ids: list[str]
    canceled_head_job_ids: list[str]

    def json_payload(self) -> JsonObject:
        """Return the stable machine-readable stop result."""
        return {
            "run_id": self.run_id,
            "state": self.state,
            "canceled_lane_job_ids": list(self.canceled_lane_job_ids),
            "canceled_head_job_ids": list(self.canceled_head_job_ids),
        }


def _new_run_id() -> str:
    return f"cc-ray-{uuid.uuid4().hex[:12]}"


def _manifest_path(runtime_paths: SlurmRayRuntimePaths) -> Path:
    return run_dir(runtime_paths) / MANIFEST_FILENAME


def _new_manifest(
    config: SlurmRayConfig,
    *,
    run_id: str,
    command: list[str],
    runtime_paths: SlurmRayRuntimePaths,
    cluster_name: str,
) -> SlurmRayManifest:
    timestamp = utc_now()
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "job_name": config.job_name,
        "slurm_cluster_name": cluster_name,
        "state": "SUBMITTING",
        "revision": 0,
        "created_at": timestamp,
        "updated_at": timestamp,
        "started_at": None,
        "stop_requested": False,
        "pipeline_exit_status": None,
        "error": None,
        "command": list(command),
        "config": config.model_dump(mode="json"),
        "runtime_paths": runtime_paths,
        "head_job_id": None,
        "lane_allocations": lane_allocations(config),
        "lanes": [],
    }


def _lane_options(head_job_id: str, allocations: int) -> tuple[str, ...]:
    """Build the sbatch options that give one lane its capacity budget.

    ``--array=0-N%1`` makes the lane a chain of allocations that run one at a time, which is what keeps a lane alive
    on a partition whose walltime is shorter than the run: Slurm requeues a preempted job but not one that hit its
    time limit, so the next array task succeeds it instead.
    """
    options = [f"--dependency=after:{head_job_id}"]
    if allocations > 1:
        options.append(f"--array=0-{allocations - 1}%1")
    return tuple(options)


def _submit_lane(  # noqa: PLR0913
    connection: ConnectionProtocol,
    *,
    base_job_name: str,
    run_id: str,
    head_job_id: str,
    lane: int,
    allocations: int,
    directory: Path,
) -> SlurmRayLane:
    """Submit one lane through the run's reusable worker wrapper."""
    job_id = submit_script(
        connection,
        directory / WORKER_SCRIPT_FILENAME,
        job_name=job_name(base_job_name, run_id, f"lane-{lane}"),
        options=_lane_options(head_job_id, allocations),
        arguments=(run_id, head_job_id, str(lane)),
    )
    return SlurmRayLane(lane=lane, job_id=job_id, submitted_at=utc_now())


def submit_slurm_ray_run(
    config: SlurmRayConfig,
    command: list[str],
    *,
    login_node: str,
    username: str | None = None,
    state_dir: str = DEFAULT_STATE_DIR,
) -> SlurmRaySubmission:
    """Create private state and submit one head plus all initial lanes."""
    if not command:
        msg = "A pipeline command must be provided after '--'"
        raise SlurmRayOperationError(msg)

    run_id = _new_run_id()
    connection = connect(login_node, username or _get_username())
    try:
        verify_remote_python(connection)
        cluster_name = slurm_cluster_name(connection)
        runtime_paths = _create_run_directory(
            connection,
            config,
            command,
            run_id=run_id,
            state_dir=state_dir,
            cluster_name=cluster_name,
        )
        return _submit_jobs(
            connection,
            config,
            run_id=run_id,
            runtime_paths=runtime_paths,
            cluster_name=cluster_name,
        )
    finally:
        connection.close()


def _create_run_directory(  # noqa: PLR0913
    connection: ConnectionProtocol,
    config: SlurmRayConfig,
    command: list[str],
    *,
    run_id: str,
    state_dir: str,
    cluster_name: str,
) -> SlurmRayRuntimePaths:
    """Create the private run directory and its initial manifest."""
    home = remote_home(connection)
    forwarded_environment = capture_forwarded_environment(config)
    runtime_paths = resolve_runtime_paths(
        config,
        home=home,
        run_id=run_id,
        state_dir=state_dir,
        host_path_exists=partial(remote_path_exists, connection),
        forwarded_environment_keys=list(forwarded_environment),
    )
    directory = run_dir(runtime_paths)
    manifest_path = _manifest_path(runtime_paths)
    if remote_path_exists(connection, directory):
        msg = f"Run directory already exists: {directory}"
        raise SlurmRayOperationError(msg)

    create_remote_path(connection, directory, mode=0o700)
    create_remote_path(connection, directory / LOG_DIR_NAME, mode=0o700)
    upload_runtime_modules(connection, directory)
    atomic_upload_json(
        connection,
        manifest_path,
        _new_manifest(
            config,
            run_id=run_id,
            command=command,
            runtime_paths=runtime_paths,
            cluster_name=cluster_name,
        ),
    )
    upload_text(
        connection,
        [
            (render_environment_file(forwarded_environment), directory / ENVIRONMENT_FILENAME, 0o600),
            *(
                [(render_host_cleanup_script(runtime_paths), directory / HOST_CLEANUP_FILENAME, 0o700)]
                if runtime_paths.get("host_cleanup_script")
                else []
            ),
            (
                render_head_script(config, run_id=run_id, runtime_paths=runtime_paths, command=command),
                directory / HEAD_SCRIPT_FILENAME,
                0o700,
            ),
            (
                render_worker_script(
                    config,
                    run_id=run_id,
                    runtime_paths=runtime_paths,
                    allocations=lane_allocations(config),
                ),
                directory / WORKER_SCRIPT_FILENAME,
                0o700,
            ),
        ],
    )
    return runtime_paths


def _submit_jobs(
    connection: ConnectionProtocol,
    config: SlurmRayConfig,
    *,
    run_id: str,
    runtime_paths: SlurmRayRuntimePaths,
    cluster_name: str,
) -> SlurmRaySubmission:
    """Submit the head and every initial lane, recording each job ID before completing submission."""
    directory = run_dir(runtime_paths)
    manifest_path = _manifest_path(runtime_paths)
    submitted_job_ids: list[str] = []

    def record(mutation: JsonObject) -> SlurmRayManifest:
        return mutate_remote_manifest(connection, manifest_path, mutation, run_id=run_id)

    try:
        head_script_path = directory / HEAD_SCRIPT_FILENAME
        head_job_id = submit_script(
            connection,
            head_script_path,
            job_name=job_name(config.job_name, run_id, "head"),
        )
        submitted_job_ids.append(head_job_id)
        manifest = record({"operation": "record-head", "job_id": head_job_id})

        lane_job_ids: list[str] = []
        # Read back rather than recomputed, so submission and a later scale size their lanes from one place.
        allocations = manifest["lane_allocations"]
        for lane in range(config.worker_lanes):
            lane_record = _submit_lane(
                connection,
                base_job_name=config.job_name,
                run_id=run_id,
                head_job_id=head_job_id,
                lane=lane,
                allocations=allocations,
                directory=directory,
            )
            submitted_job_ids.append(lane_record["job_id"])
            lane_job_ids.append(lane_record["job_id"])
            record({"operation": "append-lane", "lane": lane_record})

        record({"operation": "finish-submission"})
        return SlurmRaySubmission(
            run_id=run_id,
            slurm_cluster_name=cluster_name,
            state_dir=str(directory.parent),
            manifest_path=str(manifest_path),
            log_dir=str(directory / LOG_DIR_NAME),
            head_job_id=head_job_id,
            lane_job_ids=lane_job_ids,
        )
    except (Exception, KeyboardInterrupt) as exc:
        with suppress(Exception):
            record({"operation": "record-failure", "error": f"Submission failed: {exc}"})
        with suppress(Exception):
            cancel_jobs(connection, list(reversed(submitted_job_ids)))
        if not submitted_job_ids:
            with suppress(Exception):
                record({"operation": "finalize"})
        raise


def _required_head_job_id(manifest: SlurmRayManifest) -> str:
    head_job_id = manifest["head_job_id"]
    if head_job_id is None:
        msg = "Run manifest does not contain a head job ID"
        raise SlurmRayOperationError(msg)
    return head_job_id


def _lane_number(lane: SlurmRayLane) -> int:
    return lane["lane"]


def scale_slurm_ray_run(
    run_id: str,
    *,
    worker_lanes: int,
    login_node: str,
    username: str | None = None,
    state_dir: str = DEFAULT_STATE_DIR,
) -> SlurmRayScale:
    """Reconcile nonterminal recorded lanes to a requested count."""
    if worker_lanes < 0:
        msg = "--workers must be at least 0"
        raise SlurmRayOperationError(msg)

    submitted: list[str] = []
    canceled: list[str] = []
    with open_remote_run(
        run_id,
        login_node=login_node,
        username=username,
        state_dir=state_dir,
    ) as remote_run:
        manifest = remote_run.manifest
        lanes = manifest["lanes"]
        head_job_id = _required_head_job_id(manifest)
        states = _verified_scale_job_states(remote_run, head_job_id)
        active_lanes = [lane for lane in lanes if is_nonterminal(states[lane["job_id"]])]
        previous_worker_lanes = len(active_lanes)

        # Built from the accumulating lists, so it describes what actually happened even on a partial failure.
        def result() -> SlurmRayScale:
            return SlurmRayScale(
                run_id=run_id,
                target_worker_lanes=worker_lanes,
                previous_worker_lanes=previous_worker_lanes,
                submitted_job_ids=submitted,
                canceled_job_ids=canceled,
            )

        try:
            if previous_worker_lanes < worker_lanes:
                next_lane = max((lane["lane"] for lane in lanes), default=-1) + 1
                for lane_number in range(next_lane, next_lane + worker_lanes - previous_worker_lanes):
                    _add_lane(
                        remote_run,
                        head_job_id=head_job_id,
                        lane_number=lane_number,
                        submitted=submitted,
                        canceled=canceled,
                    )
            elif previous_worker_lanes > worker_lanes:
                for lane in sorted(active_lanes, key=_lane_number, reverse=True)[
                    : previous_worker_lanes - worker_lanes
                ]:
                    _cancel_lane(remote_run, lane, canceled=canceled)
        except Exception as exc:
            if submitted or canceled:
                msg = f"Scale operation only partially completed: {exc}"
                raise SlurmRayPartialOperationError(msg, result=result().json_payload()) from exc
            raise

        return result()


def _verified_scale_job_states(remote_run: RemoteRun, head_job_id: str) -> dict[str, SlurmJobState]:
    """Reject a run that cannot be scaled and return fresh Slurm state for its recorded jobs.

    This is a precondition, not an exclusion: a stop or teardown landing after the check still wins, because the
    manifest refuses to record a lane once the run is stopping.
    """
    manifest = remote_run.manifest
    run_id = manifest["run_id"]
    if run_state(manifest) not in {"STARTING", "ACTIVE"}:
        msg = f"Cannot scale run {run_id} in state {manifest['state']!r}"
        raise SlurmRayOperationError(msg)

    job_ids = [head_job_id, *(lane["job_id"] for lane in manifest["lanes"])]
    states = query_job_states(remote_run.connection, job_ids)
    unknown = [job_id for job_id, status in states.items() if status["state"] == "UNKNOWN"]
    if unknown:
        msg = f"Cannot scale because Slurm state is unknown for: {', '.join(unknown)}"
        raise SlurmRayOperationError(msg)
    if states[head_job_id]["state"] not in _SCALABLE_HEAD_STATES:
        msg = f"Cannot scale because head job {head_job_id} is {states[head_job_id]['state']}"
        raise SlurmRayOperationError(msg)
    return states


def _add_lane(
    remote_run: RemoteRun,
    *,
    head_job_id: str,
    lane_number: int,
    submitted: list[str],
    canceled: list[str],
) -> None:
    """Submit and durably record one new lane, rolling it back on write failure.

    Every job ID that reached Slurm is appended to ``submitted`` before the manifest write, so a partial scale
    still reports exactly what changed.
    """
    manifest = remote_run.manifest
    lane_record = _submit_lane(
        remote_run.connection,
        base_job_name=manifest["job_name"],
        run_id=manifest["run_id"],
        head_job_id=head_job_id,
        lane=lane_number,
        allocations=manifest["lane_allocations"],
        directory=remote_run.manifest_path.parent,
    )
    job_id = lane_record["job_id"]
    submitted.append(job_id)
    try:
        remote_run.mutate({"operation": "append-lane", "lane": lane_record})
    except Exception:
        if cancel_jobs(remote_run.connection, [job_id]) == [job_id]:
            canceled.append(job_id)
        raise


def _cancel_lane(remote_run: RemoteRun, lane: SlurmRayLane, *, canceled: list[str]) -> None:
    """Cancel one existing lane.

    Nothing is written back: the lane keeps its manifest record, and whether it is still a lane is Slurm's answer
    about its job ID.
    """
    job_id = lane["job_id"]
    if cancel_jobs(remote_run.connection, [job_id]) != [job_id]:
        msg = f"Failed to cancel lane {lane['lane']} job {job_id}"
        raise SlurmRayOperationError(msg)
    canceled.append(job_id)


def stop_slurm_ray_run(
    run_id: str,
    *,
    login_node: str,
    username: str | None = None,
    state_dir: str = DEFAULT_STATE_DIR,
) -> SlurmRayStop:
    """Stop exactly the lane and head jobs recorded for one run."""
    with open_remote_run(
        run_id,
        login_node=login_node,
        username=username,
        state_dir=state_dir,
    ) as remote_run:
        connection = remote_run.connection
        manifest = remote_run.manifest
        original_state = run_state(manifest)
        lane_job_ids = [lane["job_id"] for lane in manifest["lanes"]]
        head_job_id = manifest["head_job_id"]
        states = query_job_states(connection, lane_job_ids + ([head_job_id] if head_job_id is not None else []))
        unknown = [job_id for job_id, status in states.items() if status["state"] == "UNKNOWN"]
        if unknown:
            msg = f"Cannot stop because Slurm state is unknown for: {', '.join(unknown)}"
            raise SlurmRayOperationError(msg)

        def is_active(job_id: str) -> bool:
            return is_nonterminal(states[job_id])

        active_lanes = [job_id for job_id in lane_job_ids if is_active(job_id)]
        active_head = [head_job_id] if head_job_id is not None and is_active(head_job_id) else []

        # A terminal run keeps its published outcome; stopping it again only cleans up stragglers.
        if original_state in TERMINAL_RUN_STATES:
            _cancel_lanes_then_head(remote_run, active_lanes, active_head, state=original_state)
            return SlurmRayStop(run_id, original_state, active_lanes, active_head)

        remote_run.mutate({"operation": "request-stop"})
        _cancel_lanes_then_head(remote_run, active_lanes, active_head, state="STOPPING")
        # Allocations that were still live are torn down by the head wrapper, which publishes the terminal state
        # once Slurm has released them. With nothing left running, this command finalizes the run itself.
        state = "STOPPING" if active_lanes or active_head else run_state(remote_run.mutate({"operation": "finalize"}))
        return SlurmRayStop(run_id, state, active_lanes, active_head)


def _cancel_lanes_then_head(
    remote_run: RemoteRun,
    lane_job_ids: list[str],
    head_job_ids: list[str],
    *,
    state: str,
) -> None:
    """Cancel every lane before the head, so a failure partway through leaves no lane outliving its cluster."""
    run_id = remote_run.manifest["run_id"]
    canceled_lanes = cancel_jobs(remote_run.connection, lane_job_ids)
    if canceled_lanes != lane_job_ids:
        _report_uncanceled("worker", lane_job_ids, canceled_lanes, SlurmRayStop(run_id, state, canceled_lanes, []))

    canceled_head = cancel_jobs(remote_run.connection, head_job_ids)
    if canceled_head != head_job_ids:
        _report_uncanceled(
            "head", head_job_ids, canceled_head, SlurmRayStop(run_id, state, canceled_lanes, canceled_head)
        )


def _report_uncanceled(label: str, requested: list[str], canceled: list[str], result: SlurmRayStop) -> NoReturn:
    missing = sorted(set(requested) - set(canceled))
    msg = f"Failed to cancel {label} jobs: {', '.join(missing)}"
    raise SlurmRayPartialOperationError(msg, result=result.json_payload())


def list_slurm_ray_runs(
    *,
    login_node: str,
    username: str | None = None,
    state_dir: str = DEFAULT_STATE_DIR,
) -> list[JsonObject]:
    """Summarize every run directory under one state directory, newest first.

    This reads only recorded state; use ``status`` for a run's live Slurm and Ray view.
    """
    connection = connect(login_node, username or _get_username())
    try:
        manifest_paths = list_run_manifest_paths(connection, state_dir)
        manifests = read_remote_json_files(connection, sorted(manifest_paths.values()))
        runs = [
            _run_summary(run_id, manifest_path, manifests.get(str(manifest_path)))
            for run_id, manifest_path in manifest_paths.items()
        ]
    finally:
        connection.close()
    return sorted(runs, key=lambda run: str(run["created_at"] or ""), reverse=True)


def _run_summary(run_id: str, manifest_path: Path, manifest: JsonObject | None) -> JsonObject:
    """Summarize one run directory, reporting damaged state rather than hiding it."""
    summary: JsonObject = {
        "run_id": run_id,
        "slurm_cluster_name": None,
        "state": "UNREADABLE",
        "job_name": None,
        "created_at": None,
        "recorded_worker_lanes": None,
        "manifest_path": str(manifest_path),
    }
    if manifest is None:
        return summary
    with suppress(TypeError):
        validated = validate_manifest(manifest)
        summary["state"] = validated["state"]
        summary["slurm_cluster_name"] = validated["slurm_cluster_name"]
        summary["job_name"] = validated["job_name"]
        summary["created_at"] = validated.get("created_at")
        summary["recorded_worker_lanes"] = len(validated["lanes"])
    return summary
