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
"""Private run state for managed Slurm-Ray clusters.

This module is the single authority for the run manifest: its schema, its state machine, its lock, and every
mutation that may be applied to it. It uses only the Python standard library so that one implementation can serve
all three hosts that write run state:

- the client and the in-container head supervisor import it directly, and
- submission uploads this file verbatim into the private run directory, where a login or compute host without a
  Cosmos Curator installation executes it as ``python3 slurm_ray_state.py``.

One ``flock`` on ``manifest.lock`` protects every write. No caller holds it across round trips: each mutation takes
the lock, re-reads state, and is accepted or refused against the state machine below, which is what orders one
lifecycle operation against another.

See :mod:`.` for the Python floor this module is held to and why it differs from the rest of the codebase.
"""

# Deferred annotations keep PEP 585 and PEP 604 syntax out of the runtime; see the package docstring.
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, TypedDict, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

STATE_MODULE_FILENAME = "slurm_ray_state.py"
MANIFEST_SCHEMA_VERSION = 1
# Oldest interpreter this module and its peers are written against; see the package docstring for why 3.8. It
# lives here so submission can check a cluster against the same number the contract tests hold this file to.
MINIMUM_PYTHON_VERSION = (3, 8)

# Layout of one private run directory. Every host derives these from the run directory rather than reading them
# back out of the manifest, so the manifest records only what cannot be recomputed.
MANIFEST_FILENAME = "manifest.json"
MANIFEST_LOCK_FILENAME = "manifest.lock"
BOOTSTRAP_FILENAME = "bootstrap.json"
RAY_STATUS_FILENAME = "ray-status.json"
ENVIRONMENT_FILENAME = "environment.sh"
HEAD_SCRIPT_FILENAME = "head.sbatch"
WORKER_SCRIPT_FILENAME = "worker.sbatch"
LOG_DIR_NAME = "logs"

MANIFEST_REVISION_CONFLICT_EXIT_CODE = 74

# ``Dict`` rather than ``dict``: a module-level alias is a runtime expression, so deferred annotations do not
# cover it and PEP 585 subscripting would fail on the 3.8 floor.
JsonObject = Dict[str, Any]
RunState = Literal["SUBMITTING", "STARTING", "ACTIVE", "STOPPING", "SUCCEEDED", "FAILED", "STOPPED"]
TERMINAL_RUN_STATES = frozenset({"SUCCEEDED", "FAILED", "STOPPED"})

# Slurm job states that still hold or may yet hold an allocation. Every host that has to decide whether a
# recorded job is gone reads this one set: the client before it mutates, and head cleanup as it drains lanes.
NONTERMINAL_SLURM_STATES = frozenset(
    {
        "PENDING",
        "RUNNING",
        "REQUEUED",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "RESIZING",
        "REVOKED",
        "SIGNALING",
        "STAGE_OUT",
        "SUSPENDED",
        "CONFIGURING",
        "COMPLETING",
    }
)
# The states in which a head job is up and can adopt a worker, which is what a starting lane checks before it
# starts anything. Deliberately narrower than "nonterminal": a queued head has no cluster to join yet.
ACTIVE_HEAD_SLURM_STATES = frozenset({"RUNNING", "CONFIGURING"})

# ``SUBMITTING`` is the launcher's own recording and ends when every initial job ID is durable; ``STARTING`` is
# waiting on the cluster; ``ACTIVE`` is the driver running. The design note covers why the first two are split.
_ALLOWED_RUN_STATE_TRANSITIONS: dict[str, frozenset[str]] = {
    "SUBMITTING": frozenset({"STARTING", "STOPPING", "FAILED"}),
    "STARTING": frozenset({"ACTIVE", "STOPPING", "FAILED"}),
    "ACTIVE": frozenset({"STOPPING", "FAILED"}),
    "STOPPING": frozenset({"SUCCEEDED", "FAILED", "STOPPED"}),
    "SUCCEEDED": frozenset(),
    "FAILED": frozenset(),
    "STOPPED": frozenset(),
}


class SlurmRayLane(TypedDict):
    """One persisted worker-lane submission.

    A lane is recorded when it is submitted and never unrecorded: whether it is still alive is Slurm's answer
    about its job ID, not a flag kept here.
    """

    lane: int
    job_id: str
    submitted_at: str


class SlurmRayRuntimePaths(TypedDict):
    """Resolved container runtime settings for one run.

    Only values that cannot be recomputed from the run directory are recorded; see the layout constants above.
    """

    run_dir: str
    container_image: str
    mounts: list[JsonObject]
    prepare_directories: list[str]
    forwarded_environment_keys: list[str]
    ray_temp_dir: str | None
    host_setup_script: str | None
    host_cleanup_script: str | None


class SlurmRayManifest(TypedDict):
    """Version 1 managed-run manifest.

    :func:`_finalize` also writes a ``stopped_at`` timestamp when a run reaches ``STOPPED``. Nothing reads it back,
    so it is omitted here rather than declared ``NotRequired``, which the Python floor does not offer.
    """

    schema_version: int
    run_id: str
    job_name: str
    slurm_cluster_name: str
    state: RunState
    revision: int
    created_at: str
    updated_at: str
    started_at: str | None
    stop_requested: bool
    pipeline_exit_status: int | None
    error: str | None
    command: list[str]
    config: JsonObject
    runtime_paths: SlurmRayRuntimePaths
    head_job_id: str | None
    lane_allocations: int
    lanes: list[SlurmRayLane]


class ManifestRevisionConflictError(RuntimeError):
    """Raised when a conditional mutation no longer matches the manifest snapshot it observed."""

    def __init__(self, expected: int, actual: int) -> None:
        """Record the revisions needed to diagnose the lost race."""
        super().__init__(f"Manifest revision changed from {expected} to {actual}")
        self.expected = expected
        self.actual = actual


def utc_now() -> str:
    """Return a stable UTC timestamp for persisted state."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def base_job_id(job_id: str) -> str:
    """Return the job ID that owns an allocation, collapsing an array task onto its array job.

    A lane submitted with ``--array`` is one Slurm job whose tasks run one at a time, and the manifest records only
    that array job ID. Slurm reports its tasks as ``<id>_<task>`` and its pending tail as ``<id>_[a-b%1]``, so every
    reader that compares reported IDs against recorded ones has to collapse them the same way.
    """
    return job_id.split("_", maxsplit=1)[0]


def array_task_id(job_id: str) -> int | None:
    """Return the array task ID when a Slurm job ID names one concrete task.

    A collapsed pending range names several unordered tasks, so it deliberately has no single task ID. Slurm does
    not guarantee execution order for array indices; callers must never interpret this value as lane progress.
    """
    _, separator, task = job_id.partition("_")
    return int(task) if separator and task.isdigit() else None


def normalize_slurm_state(value: str) -> str:
    """Reduce a reported Slurm state to the bare state name.

    Slurm decorates some states: a cancelled job reads ``CANCELLED by <uid>``, naming whoever issued the scancel,
    and a truncated state can carry a trailing ``+``. Every reader below tells states apart by exact name, so a
    decorated state has to be reduced before it is compared or displayed.
    """
    name, _, _ = value.strip().upper().partition("+")
    bare, _, _ = name.partition(" ")
    return bare


def slurm_state_rank(state: str) -> int:
    """Rank one Slurm state by liveliness, so a renewing lane is judged by its liveliest task.

    A lane submitted as an array is several scheduler records at once: the running task, its pending tail, and
    every task it already finished. The lane is as alive as its liveliest task, so ranking picks the record to
    keep and, critically, stops a lane from looking terminal while any unused task is still queued.
    """
    if state == "RUNNING":
        return 3
    if state in NONTERMINAL_SLURM_STATES:
        return 2
    if state == "UNKNOWN":
        return 1
    return 0


def read_json(path: Path) -> JsonObject:
    """Read one JSON object from disk."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        msg = f"Expected a JSON object in {path}"
        raise TypeError(msg)
    return value


def read_manifest(path: Path) -> SlurmRayManifest:
    """Read and validate a managed-run manifest."""
    return validate_manifest(read_json(path))


def validate_manifest(value: JsonObject) -> SlurmRayManifest:
    """Validate an already-decoded manifest so that callers may index it directly.

    Every field a caller reads without further checking is validated here, including the shape of each lane. Use
    :func:`manifest_lane_job_ids` instead when best-effort cleanup must tolerate damaged state.
    """
    schema_version = value.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != MANIFEST_SCHEMA_VERSION
    ):
        msg = f"Unsupported manifest schema version: {schema_version!r}"
        raise TypeError(msg)

    required_types: tuple[tuple[str, type[object] | tuple[type[object], ...]], ...] = (
        ("run_id", str),
        ("job_name", str),
        ("slurm_cluster_name", str),
        ("state", str),
        ("revision", int),
        ("lane_allocations", int),
        ("runtime_paths", dict),
        ("lanes", list),
        ("head_job_id", (str, type(None))),
        ("pipeline_exit_status", (int, type(None))),
        ("error", (str, type(None))),
    )
    for field, expected_type in required_types:
        if not isinstance(value.get(field), expected_type):
            msg = f"Manifest contains an invalid {field}: {value.get(field)!r}"
            raise TypeError(msg)
    if not value["slurm_cluster_name"]:
        msg = "Manifest contains an empty slurm_cluster_name"
        raise TypeError(msg)
    run_state(value)  # Rejects a state outside the state machine.
    if not isinstance(value["runtime_paths"].get("run_dir"), str) or not value["runtime_paths"]["run_dir"]:
        msg = f"Manifest contains invalid runtime paths: {value['runtime_paths']!r}"
        raise TypeError(msg)
    for lane in value["lanes"]:
        if not isinstance(lane, dict) or not isinstance(lane.get("lane"), int):
            msg = f"Manifest contains an invalid lane: {lane!r}"
            raise TypeError(msg)
        if not isinstance(lane.get("job_id"), str) or not lane["job_id"]:
            msg = f"Manifest contains an invalid lane job ID: {lane.get('job_id')!r}"
            raise TypeError(msg)
    return cast("SlurmRayManifest", value)


def run_state(manifest: Mapping[str, object]) -> RunState:
    """Return a validated persisted run state."""
    value = manifest.get("state")
    if value not in _ALLOWED_RUN_STATE_TRANSITIONS:
        msg = f"Manifest contains an invalid run state: {value!r}"
        raise TypeError(msg)
    return cast("RunState", value)


def transition_run_state(manifest: JsonObject, target: RunState) -> bool:
    """Apply one legal run-state transition without regressing terminal state."""
    current = run_state(manifest)
    if current == target or current in TERMINAL_RUN_STATES:
        return False
    if target not in _ALLOWED_RUN_STATE_TRANSITIONS[current]:
        msg = f"Run cannot transition from {current!r} to {target!r}"
        raise RuntimeError(msg)
    manifest["state"] = target
    return True


def expected_terminal_state(manifest: Mapping[str, object]) -> RunState | None:
    """Derive a completed run's terminal state from its persisted outcome."""
    current = run_state(manifest)
    if current in TERMINAL_RUN_STATES:
        return current
    if current != "STOPPING":
        return None
    if manifest.get("stop_requested") is True:
        return "STOPPED"
    if isinstance(manifest.get("error"), str):
        return "FAILED"
    pipeline_exit_status = manifest.get("pipeline_exit_status")
    if not isinstance(pipeline_exit_status, int) or isinstance(pipeline_exit_status, bool):
        return None
    return "SUCCEEDED" if pipeline_exit_status == 0 else "FAILED"


def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    """Atomically replace a private JSON state file in an existing directory."""
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary_path.chmod(0o600)
        temporary_path.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def manifest_lane_job_ids(path: Path) -> list[str]:
    """Return valid recorded lane job IDs, or no IDs when state is unreadable.

    Head cleanup calls this on a possibly damaged or partially written manifest, so it never raises.
    """
    try:
        lanes = read_json(path).get("lanes", [])
        if not isinstance(lanes, list):
            return []
        return [job_id for lane in lanes if isinstance(lane, dict) and isinstance((job_id := lane.get("job_id")), str)]
    except (OSError, TypeError, ValueError):
        return []


@contextmanager
def _run_lock(run_dir: Path) -> Iterator[None]:
    """Hold the one advisory lock that protects every write to a run directory."""
    lock_path = run_dir / MANIFEST_LOCK_FILENAME
    lock_path.touch(mode=0o600, exist_ok=True)
    lock_path.chmod(0o600)
    with lock_path.open("r+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield


def mutate_manifest(
    path: Path,
    mutation: Mapping[str, Any],
    *,
    run_id: str,
    expected_revision: int | None = None,
) -> SlurmRayManifest:
    """Apply one named mutation to freshly read state under the run lock.

    ``expected_revision`` additionally makes the write conditional on the snapshot the caller derived it from, for
    a mutation decided by an observation made before the lock was taken.
    """
    run_dir = path.parent
    with _run_lock(run_dir):
        manifest = read_json(path)
        validated = validate_manifest(manifest)
        if validated["run_id"] != run_id:
            msg = f"Manifest belongs to another run: {manifest['run_id']!r}"
            raise RuntimeError(msg)
        if expected_revision is not None and validated["revision"] != expected_revision:
            raise ManifestRevisionConflictError(expected_revision, validated["revision"])

        before = json.dumps(manifest, sort_keys=True)
        _apply_manifest_mutation(manifest, mutation)
        if json.dumps(manifest, sort_keys=True) == before:
            return validate_manifest(manifest)
        manifest["revision"] += 1
        manifest["updated_at"] = utc_now()
        # Validate before publishing so a mutator can never persist state a reader would reject.
        validated = validate_manifest(manifest)
        atomic_write_json(path, manifest)
        return validated


def _apply_manifest_mutation(manifest: JsonObject, mutation: Mapping[str, Any]) -> None:
    operation = mutation.get("operation")
    mutator = _MANIFEST_MUTATORS.get(operation) if isinstance(operation, str) else None
    if mutator is None:
        msg = f"Unsupported manifest mutation: {operation!r}"
        raise ValueError(msg)
    mutator(manifest, mutation)


def _record_head(manifest: JsonObject, mutation: Mapping[str, Any]) -> None:
    job_id = _required_string(mutation, "job_id")
    current = manifest.get("head_job_id")
    if current is not None and current != job_id:
        msg = f"Manifest already records another head job: {current!r}"
        raise RuntimeError(msg)
    manifest["head_job_id"] = job_id


def _append_lane(manifest: JsonObject, mutation: Mapping[str, Any]) -> None:
    """Record one submitted lane, refusing once the run has begun stopping.

    This refusal is what orders scale against teardown: a scale already in flight can reach here after teardown
    fenced the run into ``STOPPING``. The caller cancels the job it just submitted when this raises.
    """
    state = run_state(manifest)
    if state == "STOPPING" or state in TERMINAL_RUN_STATES:
        msg = f"Cannot add a worker lane to a run that is {state}"
        raise RuntimeError(msg)
    lane = mutation.get("lane")
    if not isinstance(lane, dict):
        msg = "append-lane requires a lane object"
        raise TypeError(msg)
    lanes = manifest["lanes"]
    for current in lanes:
        if current.get("lane") == lane.get("lane") or current.get("job_id") == lane.get("job_id"):
            if current == lane:
                return
            msg = f"Manifest already records lane {lane.get('lane')!r} or job {lane.get('job_id')!r}"
            raise RuntimeError(msg)
    lanes.append(dict(lane))


def _finish_submission(manifest: JsonObject, _mutation: Mapping[str, Any]) -> None:
    """Record that submission finished, handing the run over to the cluster."""
    if run_state(manifest) == "SUBMITTING":
        transition_run_state(manifest, "STARTING")


def _request_stop(manifest: JsonObject, _mutation: Mapping[str, Any]) -> None:
    if run_state(manifest) in TERMINAL_RUN_STATES:
        return
    transition_run_state(manifest, "STOPPING")
    manifest["stop_requested"] = True


def _activate(manifest: JsonObject, _mutation: Mapping[str, Any]) -> None:
    """Record that a worker joined and the driver is starting."""
    if run_state(manifest) != "STARTING":
        return
    transition_run_state(manifest, "ACTIVE")
    manifest["started_at"] = utc_now()


def _record_driver_exit(manifest: JsonObject, mutation: Mapping[str, Any]) -> None:
    if run_state(manifest) in TERMINAL_RUN_STATES:
        return
    manifest["pipeline_exit_status"] = _required_int(mutation, "exit_status")
    transition_run_state(manifest, "STOPPING")


def _begin_teardown(manifest: JsonObject, mutation: Mapping[str, Any]) -> None:
    """Fence a run into ``STOPPING`` so that nothing new can be recorded behind head cleanup.

    Head cleanup applies this before it snapshots the lanes it is about to cancel. A run that already has an
    outcome keeps it; only a head that vanished without recording anything takes the error named here.
    """
    if run_state(manifest) in TERMINAL_RUN_STATES or run_state(manifest) == "STOPPING":
        return
    manifest["error"] = _required_string(mutation, "error")
    transition_run_state(manifest, "STOPPING")


def _record_failure(manifest: JsonObject, mutation: Mapping[str, Any]) -> None:
    """Persist a failure outcome while cleanup remains in progress."""
    if run_state(manifest) in TERMINAL_RUN_STATES:
        return
    if "exit_status" in mutation:
        manifest["pipeline_exit_status"] = _required_int(mutation, "exit_status")
    manifest["error"] = _required_string(mutation, "error")
    transition_run_state(manifest, "STOPPING")


def _finalize(manifest: JsonObject, mutation: Mapping[str, Any]) -> None:
    """Move a stopping run to the terminal state implied by its persisted outcome."""
    if run_state(manifest) in TERMINAL_RUN_STATES:
        return
    target = expected_terminal_state(manifest)
    if target is None:
        fallback_error = mutation.get("fallback_error")
        if not isinstance(fallback_error, str) or not fallback_error:
            msg = "Run has no persisted outcome to finalize"
            raise RuntimeError(msg)
        target = "FAILED"
        manifest["error"] = fallback_error
    if transition_run_state(manifest, target) and target == "STOPPED":
        manifest["stopped_at"] = utc_now()


_MANIFEST_MUTATORS: dict[str, Callable[[JsonObject, Mapping[str, Any]], None]] = {
    "record-head": _record_head,
    "append-lane": _append_lane,
    "finish-submission": _finish_submission,
    "request-stop": _request_stop,
    "activate": _activate,
    "record-driver-exit": _record_driver_exit,
    "begin-teardown": _begin_teardown,
    "record-failure": _record_failure,
    "finalize": _finalize,
}


def _required_string(value: Mapping[str, Any], field: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result:
        msg = f"Mutation contains an invalid {field}: {result!r}"
        raise TypeError(msg)
    return result


def _required_int(value: Mapping[str, Any], field: str) -> int:
    result = value.get(field)
    if not isinstance(result, int) or isinstance(result, bool):
        msg = f"Mutation contains an invalid {field}: {result!r}"
        raise TypeError(msg)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    mutate = actions.add_parser("mutate")
    mutate.add_argument("path", type=Path)
    mutate.add_argument("run_id")
    mutate.add_argument("mutation", type=json.loads)
    mutate.add_argument("--expected-revision", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one state operation and return a stable process exit status."""
    args = _parser().parse_args(argv)
    try:
        result = mutate_manifest(
            args.path,
            args.mutation,
            run_id=args.run_id,
            expected_revision=args.expected_revision,
        )
        sys.stdout.write(json.dumps(result))
    except ManifestRevisionConflictError as exc:
        sys.stdout.write(f"{exc.actual}\n")
        return MANIFEST_REVISION_CONFLICT_EXIT_CODE
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
