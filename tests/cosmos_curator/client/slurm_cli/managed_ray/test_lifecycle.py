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
"""Tests for the managed Ray commands that submit, scale, stop, and list a run."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from cosmos_curator.client.slurm_cli.managed_ray.lifecycle import (
    SlurmRayPartialOperationError,
    list_slurm_ray_runs,
    scale_slurm_ray_run,
    stop_slurm_ray_run,
    submit_slurm_ray_run,
)
from cosmos_curator.client.slurm_cli.managed_ray.onnode.slurm_ray_runtime import (
    RUNTIME_MODULE_FILENAME,
)
from cosmos_curator.client.slurm_cli.managed_ray.onnode.slurm_ray_state import (
    STATE_MODULE_FILENAME,
    atomic_write_json,
    mutate_manifest,
    read_json,
    transition_run_state,
)
from cosmos_curator.client.slurm_cli.managed_ray.remote import (
    SlurmRayOperationError,
    open_remote_run,
    upload_runtime_modules,
)
from cosmos_curator.client.slurm_cli.managed_ray.render import (
    capture_forwarded_environment,
)
from cosmos_curator.client.slurm_cli.slurm_submit import LocalConnection
from tests.cosmos_curator.client.slurm_cli.managed_ray.launcher_stubs import (
    FakeConnection,
    FakeResult,
    apply_test_manifest_mutation,
    make_active_manifest,
    make_config,
    mock_remote_run,
    mock_submission_manifest,
    record_test_mutations,
)

MODULE = "cosmos_curator.client.slurm_cli.managed_ray.lifecycle"
REMOTE_MODULE = "cosmos_curator.client.slurm_cli.managed_ray.remote"


@pytest.mark.parametrize("worker_lanes", [0, 2])
def test_submit_records_each_returned_job_id_immediately(monkeypatch: pytest.MonkeyPatch, worker_lanes: int) -> None:
    """Initial submission persists head and lane IDs before advancing to STARTING."""
    config = make_config(worker_lanes=worker_lanes, environment=["HOST_ONLY"])
    connection = FakeConnection()
    snapshots: list[dict[str, object]] = []
    submitted_paths: list[Path] = []
    submitted_options: list[dict[str, object]] = []
    uploaded_files: list[tuple[str, Path, int]] = []
    job_ids = iter(["100", "101", "102"])
    monkeypatch.setenv("HOST_ONLY", "secret value")
    assert capture_forwarded_environment(config)["HOST_ONLY"] == "secret value"

    monkeypatch.setattr(f"{MODULE}.connect", lambda *_args, **_kwargs: connection)
    monkeypatch.setattr(f"{MODULE}.remote_path_exists", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(f"{MODULE}.create_remote_path", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        f"{MODULE}.upload_text",
        lambda _connection, files: uploaded_files.extend(files),
    )
    runtime_module_uploads: list[Path] = []
    monkeypatch.setattr(
        f"{MODULE}.upload_runtime_modules",
        lambda _connection, path: runtime_module_uploads.append(path),
    )
    mock_submission_manifest(monkeypatch, snapshots)

    def submit_script(_connection: object, path: Path, **kwargs: object) -> str:
        submitted_paths.append(path)
        submitted_options.append(kwargs)
        return next(job_ids)

    monkeypatch.setattr(f"{MODULE}.submit_script", submit_script)

    result = submit_slurm_ray_run(
        config,
        ["python", "-m", "pipeline"],
        login_node="login",
        username="user",
    )

    assert result.head_job_id == "100"
    lane_job_ids = [str(101 + lane) for lane in range(worker_lanes)]
    assert result.lane_job_ids == lane_job_ids
    assert result.slurm_cluster_name == "test-cluster"
    assert result.log_dir == f"{result.state_dir}/{result.run_id}/logs"
    assert result.json_payload()["log_dir"] == result.log_dir
    assert runtime_module_uploads == [Path(result.manifest_path).parent]
    assert [path.name for path in submitted_paths] == ["head.sbatch", *["worker.sbatch"] * worker_lanes]
    for lane in range(worker_lanes):
        assert submitted_options[lane + 1]["options"] == ("--dependency=after:100",)
        assert submitted_options[lane + 1]["arguments"] == (result.run_id, "100", str(lane))
        assert [record["job_id"] for record in snapshots[lane + 1]["lanes"]] == lane_job_ids[: lane + 1]  # type: ignore[index]
    assert snapshots[0]["head_job_id"] == "100"
    assert snapshots[0]["slurm_cluster_name"] == "test-cluster"
    assert snapshots[-1]["state"] == "STARTING"
    assert [lane["job_id"] for lane in snapshots[-1]["lanes"]] == lane_job_ids  # type: ignore[index]
    environment_upload = next(upload for upload in uploaded_files if upload[1].name == "environment.sh")
    assert "export HOST_ONLY='secret value'" in environment_upload[0]
    assert environment_upload[2] == 0o600
    runtime_paths = snapshots[-1]["runtime_paths"]
    assert isinstance(runtime_paths, dict)
    assert "HOST_ONLY" in runtime_paths["forwarded_environment_keys"]
    assert "secret value" not in json.dumps(snapshots[-1])
    assert connection.closed is True


def test_submission_failure_cancels_only_recorded_jobs_and_stays_stopping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial sbatch failure cancels exact recorded IDs and leaves cleanup visibly in progress."""
    config = make_config()
    connection = FakeConnection()
    snapshots: list[dict[str, object]] = []
    canceled: list[list[str]] = []
    calls = 0

    monkeypatch.setattr(f"{MODULE}.connect", lambda *_args, **_kwargs: connection)
    monkeypatch.setattr(f"{MODULE}.remote_path_exists", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(f"{MODULE}.create_remote_path", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(f"{MODULE}.upload_text", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(f"{MODULE}.upload_runtime_modules", lambda *_args, **_kwargs: None)
    mock_submission_manifest(monkeypatch, snapshots)

    def submit_script(_connection: object, _path: Path, **_kwargs: object) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return "100"
        msg = "injected sbatch failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(f"{MODULE}.submit_script", submit_script)
    monkeypatch.setattr(
        f"{MODULE}.cancel_jobs",
        lambda _connection, job_ids, **_kwargs: canceled.append(job_ids) or job_ids,
    )

    with pytest.raises(RuntimeError, match="injected sbatch failure"):
        submit_slurm_ray_run(
            config,
            ["python", "-m", "pipeline"],
            login_node="login",
            username="user",
        )

    assert canceled == [["100"]]
    assert snapshots[-1]["state"] == "STOPPING"
    assert "injected sbatch failure" in str(snapshots[-1]["error"])
    assert connection.closed is True


def test_submission_failure_before_any_job_is_terminal_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no Slurm jobs to drain, failed submission can safely publish its terminal state."""
    config = make_config()
    connection = FakeConnection()
    snapshots: list[dict[str, object]] = []
    canceled: list[list[str]] = []

    monkeypatch.setattr(f"{MODULE}.connect", lambda *_args, **_kwargs: connection)
    monkeypatch.setattr(f"{MODULE}.remote_path_exists", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(f"{MODULE}.create_remote_path", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(f"{MODULE}.upload_text", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(f"{MODULE}.upload_runtime_modules", lambda *_args, **_kwargs: None)
    mock_submission_manifest(monkeypatch, snapshots)
    monkeypatch.setattr(
        f"{MODULE}.submit_script",
        Mock(side_effect=RuntimeError("injected head sbatch failure")),
    )
    monkeypatch.setattr(
        f"{MODULE}.cancel_jobs",
        lambda _connection, job_ids, **_kwargs: canceled.append(job_ids) or job_ids,
    )

    with pytest.raises(RuntimeError, match="injected head sbatch failure"):
        submit_slurm_ray_run(
            config,
            ["python", "-m", "pipeline"],
            login_node="login",
            username="user",
        )

    assert canceled == [[]]
    assert [snapshot["state"] for snapshot in snapshots] == ["STOPPING", "FAILED"]
    assert connection.closed is True


def test_submission_rejects_a_cluster_whose_python_cannot_run_the_uploaded_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe names both versions instead of letting an uploaded module fail with an ImportError."""

    class _OldPythonConnection(FakeConnection):
        def run(self, command: str, **kwargs: object) -> FakeResult:
            if "sys.version_info" in command:
                return FakeResult(stdout="3.6\n")
            return super().run(command, **kwargs)

    connection = _OldPythonConnection()
    monkeypatch.setattr(f"{MODULE}.connect", lambda *_args, **_kwargs: connection)

    with pytest.raises(SlurmRayOperationError, match=r"is 3\.6.*needs at least 3\.8"):
        submit_slurm_ray_run(make_config(), ["python", "-m", "pipeline"], login_node="login", username="user")

    # Nothing was created, so a rejected cluster leaves no run directory behind.
    assert not any(command.startswith(("sbatch", "mkdir")) for command in connection.commands)
    assert connection.closed


def test_submission_rejects_a_cluster_with_no_usable_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """A login node without python3 at all fails the same way rather than part way through submission."""

    class _NoPythonConnection(FakeConnection):
        def run(self, command: str, **kwargs: object) -> FakeResult:
            if "sys.version_info" in command:
                return FakeResult(ok=False)
            return super().run(command, **kwargs)

    monkeypatch.setattr(f"{MODULE}.connect", lambda *_args, **_kwargs: _NoPythonConnection())

    with pytest.raises(SlurmRayOperationError, match="No usable python3"):
        submit_slurm_ray_run(make_config(), ["python", "-m", "pipeline"], login_node="login", username="user")


def test_submission_uploads_the_exact_container_runtime(tmp_path: Path) -> None:
    """The run mount carries the same state and runtime implementation used by the submitting CLI."""
    connection = LocalConnection("localhost", "user")
    connection._context.config.run.in_stream = False

    upload_runtime_modules(connection, tmp_path)

    for filename in (STATE_MODULE_FILENAME, RUNTIME_MODULE_FILENAME):
        path = tmp_path / filename
        assert path.is_file()
        assert path.stat().st_mode & 0o777 == 0o700
    subprocess.run(  # noqa: S603
        [sys.executable, str(tmp_path / RUNTIME_MODULE_FILENAME), "--help"],
        check=True,
        capture_output=True,
    )


def test_custom_state_directory_resolves_from_the_run_id_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run directory is always <state_dir>/<run_id>, so no index is consulted to find it."""
    connection = FakeConnection()
    manifest = make_active_manifest(make_config())

    monkeypatch.setattr(f"{REMOTE_MODULE}.connect", lambda *_args, **_kwargs: connection)
    monkeypatch.setattr(f"{REMOTE_MODULE}._read_remote_manifest", lambda *_args, **_kwargs: manifest)

    with open_remote_run(
        "cc-ray-deadbeef",
        login_node="login",
        username="user",
        state_dir="/custom/shared/state",
    ) as remote_run:
        assert remote_run.manifest_path == Path("/custom/shared/state/cc-ray-deadbeef/manifest.json")
        assert remote_run.manifest["state"] == "ACTIVE"

    # Resolution is arithmetic: the only command issued is the manifest read itself.
    assert not [command for command in connection.commands if "runs/" in command]
    assert connection.closed is True


def test_list_summarizes_runs_and_surfaces_damaged_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run directory whose manifest cannot be read is reported rather than silently dropped."""
    connection = FakeConnection()
    manifest = make_active_manifest(make_config())
    manifest["created_at"] = "2026-01-02T00:00:00Z"

    monkeypatch.setattr(f"{MODULE}.connect", lambda *_args, **_kwargs: connection)
    monkeypatch.setattr(
        f"{MODULE}.list_run_manifest_paths",
        lambda *_args: {
            "cc-ray-deadbeef": Path("/state/cc-ray-deadbeef/manifest.json"),
            "cc-ray-00000000": Path("/state/cc-ray-00000000/manifest.json"),
        },
    )
    monkeypatch.setattr(
        f"{MODULE}.read_remote_json_files",
        lambda *_args: {"/state/cc-ray-deadbeef/manifest.json": manifest},
    )

    runs = list_slurm_ray_runs(login_node="login", username="user")

    assert [(run["run_id"], run["state"], run["recorded_worker_lanes"]) for run in runs] == [
        ("cc-ray-deadbeef", "ACTIVE", 2),
        ("cc-ray-00000000", "UNREADABLE", None),
    ]
    assert connection.closed is True


def test_scale_replaces_terminal_lanes_with_fresh_lane_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scale-up counts only nonterminal jobs and never reuses an old lane number."""
    config = make_config()
    manifest = make_active_manifest(config)
    connection = FakeConnection()
    snapshots: list[dict[str, object]] = []
    submitted_paths: list[Path] = []

    mock_remote_run(monkeypatch, connection, manifest)
    monkeypatch.setattr(
        f"{MODULE}.query_job_states",
        lambda *_args, **_kwargs: {
            "100": {"state": "RUNNING"},
            "101": {"state": "RUNNING"},
            "102": {"state": "FAILED"},
        },
    )
    monkeypatch.setattr(f"{MODULE}.upload_text", lambda *_args, **_kwargs: None)
    writer = record_test_mutations(manifest, snapshots)
    for module in (MODULE, REMOTE_MODULE):
        monkeypatch.setattr(f"{module}.mutate_remote_manifest", writer)

    def submit_script(_connection: object, path: Path, **_kwargs: object) -> str:
        submitted_paths.append(path)
        return "103"

    monkeypatch.setattr(f"{MODULE}.submit_script", submit_script)

    result = scale_slurm_ray_run(
        "cc-ray-deadbeef",
        worker_lanes=2,
        login_node="login",
        username="user",
    )

    assert result.submitted_job_ids == ["103"]
    assert result.worker_lanes == 2
    assert submitted_paths[0].name == "worker.sbatch"
    assert snapshots[-1]["lanes"][-1]["lane"] == 2  # type: ignore[index]
    assert connection.closed is True


def test_scale_down_cancels_highest_numbered_nonterminal_lanes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scale-down is deterministic for pending and running lanes."""
    config = make_config()
    manifest = make_active_manifest(config)
    connection = FakeConnection()
    manifest["lanes"].append(  # type: ignore[union-attr]
        {
            "lane": 2,
            "job_id": "103",
            "submitted_at": "2026-01-01T00:00:00Z",
        }
    )

    mock_remote_run(monkeypatch, connection, manifest)
    monkeypatch.setattr(
        f"{MODULE}.query_job_states",
        lambda *_args, **_kwargs: {
            "100": {"state": "RUNNING"},
            "101": {"state": "RUNNING"},
            "102": {"state": "FAILED"},
            "103": {"state": "PENDING"},
        },
    )
    result = scale_slurm_ray_run(
        "cc-ray-deadbeef",
        worker_lanes=0,
        login_node="login",
        username="user",
    )

    assert result.canceled_job_ids == ["103", "101"]
    assert result.worker_lanes == 0
    scancel_commands = [command for command in connection.commands if command.startswith("scancel")]
    assert scancel_commands == ["scancel --quiet 103", "scancel --quiet 101"]


def test_scale_rejects_a_head_that_is_already_completing(monkeypatch: pytest.MonkeyPatch) -> None:
    """No new worker may be submitted once the head has begun leaving Slurm."""
    manifest = make_active_manifest(make_config())
    connection = FakeConnection()

    mock_remote_run(monkeypatch, connection, manifest)
    monkeypatch.setattr(
        f"{MODULE}.query_job_states",
        lambda *_args, **_kwargs: {
            "100": {"state": "COMPLETING"},
            "101": {"state": "RUNNING"},
            "102": {"state": "RUNNING"},
        },
    )

    with pytest.raises(SlurmRayOperationError, match="head job 100 is COMPLETING"):
        scale_slurm_ray_run(
            "cc-ray-deadbeef",
            worker_lanes=3,
            login_node="login",
            username="user",
        )

    assert not any(command.startswith("sbatch") for command in connection.commands)


def test_stop_cancels_recorded_lanes_before_the_head(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop acts on exact manifest IDs and remains stopping until Slurm cleanup completes."""
    config = make_config()
    manifest = make_active_manifest(config)
    connection = FakeConnection()
    cancellation_batches: list[list[str]] = []
    snapshots: list[dict[str, object]] = []

    mock_remote_run(monkeypatch, connection, manifest)
    monkeypatch.setattr(
        f"{MODULE}.query_job_states",
        lambda *_args, **_kwargs: {
            "100": {"state": "RUNNING"},
            "101": {"state": "RUNNING"},
            "102": {"state": "PENDING"},
        },
    )
    monkeypatch.setattr(
        f"{MODULE}.cancel_jobs",
        lambda _connection, job_ids, **_kwargs: cancellation_batches.append(job_ids) or job_ids,
    )
    writer = record_test_mutations(manifest, snapshots)
    for module in (MODULE, REMOTE_MODULE):
        monkeypatch.setattr(f"{module}.mutate_remote_manifest", writer)

    result = stop_slurm_ray_run("cc-ray-deadbeef", login_node="login", username="user")

    assert cancellation_batches == [["101", "102"], ["100"]]
    assert result.state == "STOPPING"
    assert snapshots[0]["state"] == "STOPPING"
    assert snapshots[0]["stop_requested"] is True
    assert snapshots[-1]["state"] == "STOPPING"


def test_scale_partial_failure_reports_every_changed_job_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed scale-up reports both durable and rolled-back Slurm mutations."""
    manifest = make_active_manifest(make_config())
    connection = FakeConnection()
    write_attempts = 0
    submitted_job_ids = iter(["103", "104"])
    canceled: list[list[str]] = []

    mock_remote_run(monkeypatch, connection, manifest)
    monkeypatch.setattr(
        f"{MODULE}.query_job_states",
        lambda *_args, **_kwargs: {
            "100": {"state": "RUNNING"},
            "101": {"state": "RUNNING"},
            "102": {"state": "RUNNING"},
        },
    )
    monkeypatch.setattr(f"{MODULE}.upload_text", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(f"{MODULE}.submit_script", lambda *_args, **_kwargs: next(submitted_job_ids))
    monkeypatch.setattr(
        f"{MODULE}.cancel_jobs",
        lambda _connection, job_ids, **_kwargs: canceled.append(job_ids) or job_ids,
    )

    def fake_mutate(
        _connection: object,
        _path: Path,
        mutation: dict[str, object],
        **_kwargs: object,
    ) -> dict[str, object]:
        nonlocal write_attempts
        write_attempts += 1
        if write_attempts == 2:
            msg = "injected manifest failure"
            raise RuntimeError(msg)
        return apply_test_manifest_mutation(manifest, mutation)

    for module in (MODULE, REMOTE_MODULE):
        monkeypatch.setattr(f"{module}.mutate_remote_manifest", fake_mutate)

    with pytest.raises(SlurmRayPartialOperationError, match="injected manifest failure") as exc_info:
        scale_slurm_ray_run(
            "cc-ray-deadbeef",
            worker_lanes=4,
            login_node="login",
            username="user",
        )

    assert exc_info.value.result == {
        "run_id": "cc-ray-deadbeef",
        "target_worker_lanes": 4,
        "previous_worker_lanes": 2,
        "worker_lanes": 3,
        "submitted_job_ids": ["103", "104"],
        "canceled_job_ids": ["104"],
    }
    assert canceled == [["104"]]


def test_stop_cleans_up_live_jobs_even_after_run_is_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A terminal manifest never prevents cleanup after an outer-wrapper failure."""
    manifest = make_active_manifest(make_config())
    manifest["state"] = "FAILED"
    connection = FakeConnection()
    cancellation_batches: list[list[str]] = []

    mock_remote_run(monkeypatch, connection, manifest)
    monkeypatch.setattr(
        f"{MODULE}.query_job_states",
        lambda *_args, **_kwargs: {
            "100": {"state": "COMPLETED"},
            "101": {"state": "RUNNING"},
            "102": {"state": "CANCELLED"},
        },
    )
    monkeypatch.setattr(
        f"{MODULE}.cancel_jobs",
        lambda _connection, job_ids, **_kwargs: cancellation_batches.append(job_ids) or job_ids,
    )
    result = stop_slurm_ray_run("cc-ray-deadbeef", login_node="login", username="user")

    assert cancellation_batches == [["101"], []]
    assert result.state == "FAILED"
    assert result.canceled_lane_job_ids == ["101"]


def test_stop_is_idempotent_after_terminal_jobs_leave_slurm_accounting(monkeypatch: pytest.MonkeyPatch) -> None:
    """A completed run remains stoppable after successful Slurm queries find no jobs."""
    manifest = make_active_manifest(make_config())
    manifest["state"] = "SUCCEEDED"
    connection = FakeConnection()

    mock_remote_run(monkeypatch, connection, manifest)

    result = stop_slurm_ray_run("cc-ray-deadbeef", login_node="login", username="user")

    assert result.json_payload() == {
        "run_id": "cc-ray-deadbeef",
        "state": "SUCCEEDED",
        "canceled_lane_job_ids": [],
        "canceled_head_job_ids": [],
    }
    assert not any(command.startswith("scancel") for command in connection.commands)


def test_private_state_writes_are_atomic_and_mode_restricted(tmp_path: Path) -> None:
    """Runtime state writes preserve private permissions across updates."""
    manifest_path = tmp_path / "manifest.json"
    manifest = make_active_manifest(make_config())
    manifest["state"] = "STARTING"
    atomic_write_json(manifest_path, manifest)

    updated = mutate_manifest(manifest_path, {"operation": "activate"}, run_id="cc-ray-deadbeef")

    assert updated["state"] == "ACTIVE"
    assert read_json(manifest_path)["state"] == "ACTIVE"
    assert manifest_path.stat().st_mode & 0o777 == 0o600
    assert (manifest_path.parent / "manifest.lock").stat().st_mode & 0o777 == 0o600


def test_state_transitions_require_order_and_never_regress_terminal_state() -> None:
    """All manifest writers share one explicit run-state machine."""
    manifest = make_active_manifest(make_config())

    with pytest.raises(RuntimeError, match=r"ACTIVE.*SUCCEEDED"):
        transition_run_state(manifest, "SUCCEEDED")

    manifest["state"] = "FAILED"
    assert transition_run_state(manifest, "ACTIVE") is False
    assert manifest["state"] == "FAILED"
