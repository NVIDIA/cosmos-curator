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
"""Tests for the container and compute-node entrypoints a managed Ray run executes."""

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from cosmos_curator.client.slurm_cli.managed_ray.onnode.slurm_ray_runtime import (
    _HEAD_PORT_NAMES,
    _HEAD_PROBE_ATTEMPTS,
    _WORKER_PORT_NAMES,
    _WORKER_START_ATTEMPTS,
    _allocated_cpus,
    _parser,
    _prepare_ray_temp_dir,
    _queued_job_states,
    _ray_head_command,
    _ray_worker_command,
    _reserve_head_ports,
    _reserve_worker_ports,
    _run_cleanup,
    _run_driver,
    _run_head,
    _run_probe_head,
    _start_worker_ray,
    _stop_process,
    _terminate_process_group,
    _wait_for_compute,
)
from cosmos_curator.client.slurm_cli.managed_ray.onnode.slurm_ray_state import (
    atomic_write_json,
    read_json,
)
from tests.cosmos_curator.client.slurm_cli.managed_ray.launcher_stubs import (
    apply_test_manifest_mutation,
    make_active_manifest,
    make_config,
)

RUNTIME_MODULE = "cosmos_curator.client.slurm_cli.managed_ray.onnode.slurm_ray_runtime"


@pytest.mark.parametrize("value", ["0", "-1"])
def test_worker_parser_rejects_nonpositive_io_slots(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    """A worker cannot advertise a zero or negative logical IO capacity to Ray."""
    with pytest.raises(SystemExit, match="2"):
        _parser().parse_args(
            [
                "worker",
                "--run-id",
                "cc-ray-deadbeef",
                "--bootstrap",
                "bootstrap.json",
                "--startup-timeout-seconds",
                "60",
                "--io-slots-per-node",
                value,
            ]
        )

    assert "--io-slots-per-node must be at least 1" in capsys.readouterr().err


def test_ray_temp_dir_is_scoped_by_slurm_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The allocation isolates cleanup without adding the longer cluster ID to Ray's socket paths."""
    monkeypatch.setenv("SLURM_JOB_ID", "1899271")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "3")

    temp_dir = _prepare_ray_temp_dir(str(tmp_path), "lane-2")

    assert temp_dir is not None
    assert temp_dir == tmp_path / "lane-2-1899271-3"
    assert temp_dir.is_dir()
    socket_path = (
        Path("/raid/scratch")
        / temp_dir.name
        / "session_2026-08-19_15-39-00_322429_2032562"
        / "sockets"
        / "plasma_store"
    )
    assert len(str(socket_path).encode()) <= 107


def test_head_cleanup_cancels_lanes_then_publishes_terminal_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The head's exit trap cancels every recorded lane and finalizes once Slurm releases them."""
    manifest_path = tmp_path / "manifest.json"
    manifest = make_active_manifest(make_config())
    manifest["state"] = "STOPPING"
    manifest["pipeline_exit_status"] = 0
    atomic_write_json(manifest_path, manifest)
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> Mock:
        calls.append(command)
        return Mock(returncode=0, stdout="")

    monkeypatch.setattr(f"{RUNTIME_MODULE}.subprocess.run", fake_run)

    status = _run_cleanup(Namespace(run_id="cc-ray-deadbeef", manifest=str(manifest_path)))

    assert status == 0
    assert calls[0] == ["scancel", "--quiet", "101", "102"]
    assert read_json(manifest_path)["state"] == "SUCCEEDED"


def test_head_cleanup_leaves_run_stopping_when_lanes_outlive_the_drain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Undrained lanes leave the terminal state for a later status to reconcile."""
    manifest_path = tmp_path / "manifest.json"
    manifest = make_active_manifest(make_config())
    manifest["state"] = "STOPPING"
    manifest["pipeline_exit_status"] = 0
    atomic_write_json(manifest_path, manifest)

    monkeypatch.setattr(f"{RUNTIME_MODULE}.subprocess.run", lambda *_a, **_k: Mock(returncode=0, stdout=""))
    monkeypatch.setattr(f"{RUNTIME_MODULE}._wait_for_lanes_to_drain", lambda *_a, **_k: False)

    status = _run_cleanup(Namespace(run_id="cc-ray-deadbeef", manifest=str(manifest_path)))

    assert status == 1
    assert read_json(manifest_path)["state"] == "STOPPING"


def test_abrupt_head_cleanup_fences_scaling_before_canceling_lanes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exit trap serializes cleanup and enters STOPPING before acting on its lane snapshot."""
    manifest_path = tmp_path / "manifest.json"
    atomic_write_json(manifest_path, make_active_manifest(make_config()))
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> Mock:
        calls.append(command)
        if command[0] == "scancel":
            assert read_json(manifest_path)["state"] == "STOPPING"
        return Mock(returncode=0, stdout="")

    monkeypatch.setattr(f"{RUNTIME_MODULE}.subprocess.run", fake_run)

    status = _run_cleanup(Namespace(run_id="cc-ray-deadbeef", manifest=str(manifest_path)))

    assert status == 0
    assert calls[0] == ["scancel", "--quiet", "101", "102"]
    final_manifest = read_json(manifest_path)
    assert final_manifest["state"] == "FAILED"
    assert final_manifest["error"] == "Head job exited before recording a terminal outcome"


def test_lane_drain_retries_a_transient_squeue_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """One failed query cannot be mistaken for lanes that have already been released."""
    responses = iter([Mock(returncode=1, stdout=""), Mock(returncode=0, stdout="101|RUNNING\n")])
    monkeypatch.setattr(f"{RUNTIME_MODULE}.subprocess.run", lambda *_a, **_k: next(responses))

    assert _queued_job_states() is None
    assert _queued_job_states() == {"101": "RUNNING"}


def test_a_renewing_lane_reports_its_liveliest_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """An array lane's running incarnation must not be hidden behind its own pending tail."""
    queue = "101_3|RUNNING\n101_[4-41%1]|PENDING\n102|PENDING\n"
    monkeypatch.setattr(f"{RUNTIME_MODULE}.subprocess.run", lambda *_a, **_k: Mock(returncode=0, stdout=queue))

    # Both tasks collapse onto the array job ID the manifest recorded, so a lane still matches while it renews.
    assert _queued_job_states() == {"101": "RUNNING", "102": "PENDING"}


def _probe_head(monkeypatch: pytest.MonkeyPatch, queues: list[dict[str, str] | None]) -> int:
    """Run the head probe against a scripted sequence of scheduler answers."""
    responses = iter(queues)
    monkeypatch.setattr(f"{RUNTIME_MODULE}._queued_job_states", lambda: next(responses))
    monkeypatch.setattr(f"{RUNTIME_MODULE}.time.sleep", lambda _seconds: None)
    return _run_probe_head(Namespace(job_id="4812733"))


def test_lane_joins_only_when_the_scheduler_says_its_head_is_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lane must not attach itself to a run whose head has already gone."""
    assert _probe_head(monkeypatch, [{"4812733": "RUNNING", "4812734": "PENDING"}]) == 0
    assert _probe_head(monkeypatch, [{"4812733": "CONFIGURING"}]) == 0
    assert _probe_head(monkeypatch, [{"4812734": "RUNNING"}]) == 1
    assert _probe_head(monkeypatch, [{"4812733": "COMPLETING"}]) == 1


def test_lane_survives_a_scheduler_that_answers_late_and_refuses_one_that_never_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed query and an absent job look alike to squeue, so only an answer it gave may condemn a run.

    Without the retry one hiccup costs the lane an allocation, and nothing replaces a terminal lane automatically.
    """
    assert _probe_head(monkeypatch, [None, None, {"4812733": "RUNNING"}]) == 0
    assert _probe_head(monkeypatch, [None] * _HEAD_PROBE_ATTEMPTS) == 1


# Flag to RayParams field, as `ray start` itself maps them. --port is head-only and lands on gcs_server_port.
_RAY_PORT_FIELDS = {
    "--port": "gcs_server_port",
    "--object-manager-port": "object_manager_port",
    "--node-manager-port": "node_manager_port",
    "--dashboard-port": "dashboard_port",
    "--dashboard-agent-grpc-port": "metrics_agent_port",
    "--dashboard-agent-listen-port": "dashboard_agent_listen_port",
    "--runtime-env-agent-port": "runtime_env_agent_port",
    "--metrics-export-port": "metrics_export_port",
    "--ray-client-server-port": "ray_client_server_port",
    "--min-worker-port": "min_worker_port",
    "--max-worker-port": "max_worker_port",
}


def _assert_no_pre_selected_port_collision(command: list[str]) -> None:
    """Replay a rendered ``ray start`` command through Ray's own pre-selected-port check.

    ``ray start`` defaults every port flag a command omits and then refuses to start when two pre-selected ports
    collide, so the defaults are the whole point: reading them off the real installed Ray is what makes this fail
    here rather than on a cluster when one of them moves.
    """
    from ray._private.parameter import RayParams  # noqa: PLC0415
    from ray.scripts.scripts import start  # noqa: PLC0415

    # Bare flags such as --head and --block carry no value, so pair a flag only with a non-flag token.
    given = {
        item: command[index + 1]
        for index, item in enumerate(command[:-1])
        if item.startswith("--") and not command[index + 1].startswith("--")
    }
    # Click reports an unset port as a sentinel rather than None, which RayParams reads as "choose one".
    defaults = {option.opts[0]: option.default if isinstance(option.default, int) else None for option in start.params}
    parameters: dict[str, int | None] = {}
    for flag, field in _RAY_PORT_FIELDS.items():
        if flag == "--port" and "--head" not in command:
            continue
        value = given.get(flag, defaults[flag])
        parameters[field] = None if value is None else int(value)
    # `ray start` resolves an unset client server port to 10001 whenever ray[client] can be imported.
    if parameters.get("ray_client_server_port") is None:
        parameters["ray_client_server_port"] = 10001
    RayParams(**parameters).update_pre_selected_port()


def test_head_ports_never_collide_with_a_ray_default() -> None:
    """Ray rejects a head whose pre-selected ports overlap, and three of its defaults are fixed values.

    A node whose ``ip_local_port_range`` starts below 20000 reserves ports inside Ray's default 10002-19999
    worker range, and 52365 is reachable wherever the usual range applies, so the head must name all of them.
    """
    _assert_no_pre_selected_port_collision(_ray_head_command("head-node", _reserve_head_ports(), None))

    # Every port this node could hand out that Ray would otherwise have claimed for itself.
    hostile = (14897, 18137, 10001, 52365, 10002, 19999, 40001, 40002, 40003)
    reserved = dict(zip(_HEAD_PORT_NAMES, hostile, strict=True))
    _assert_no_pre_selected_port_collision(_ray_head_command("head-node", reserved, None))


def test_worker_reserved_ports_clear_ray_defaults() -> None:
    """A worker reserves free ports, which still must not overlap the ranges Ray reserves."""
    _assert_no_pre_selected_port_collision(_ray_worker_command("head-node:6379", _reserve_worker_ports(), None))

    # Ports inside Ray's default worker range, which a node with a low ip_local_port_range can hand out.
    hostile = (10002, 19999, 14897, 18137, 40001, 52365)
    reserved = dict(zip(_WORKER_PORT_NAMES, hostile, strict=True))
    _assert_no_pre_selected_port_collision(_ray_worker_command("head-node:6379", reserved, None))


def test_one_worker_does_not_reuse_a_port_across_its_own_services() -> None:
    """Every socket stays open until the last is assigned, so one lane's ports are distinct from each other."""
    reserved = _reserve_worker_ports()

    assert set(reserved) == set(_WORKER_PORT_NAMES)
    assert len(set(reserved.values())) == len(_WORKER_PORT_NAMES), "one worker reused a port across services"


def test_a_worker_avoids_ports_another_lane_still_holds() -> None:
    """Lanes that ask for a slice can land on the same node, and a shared port kills the second raylet.

    Asserting two successive reservations are disjoint would be wrong: ``_reserve_worker_ports`` closes its
    sockets before returning, so the kernel may legitimately hand the same port to the next caller. What must
    hold is that a lane never selects a port another lane is still bound to -- so the occupied ports are held
    open here for the duration of the second reservation.
    """
    held = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in _WORKER_PORT_NAMES]
    try:
        for sock in held:
            sock.bind(("", 0))
        occupied = {sock.getsockname()[1] for sock in held}

        reserved = _reserve_worker_ports()

        assert not occupied & set(reserved.values()), "a lane selected a port another lane still holds"
    finally:
        for sock in held:
            sock.close()


def test_runtime_ray_commands_advertise_resources_on_both_roles() -> None:
    """The head and workers expose CPUs, GPUs, and the configured download capacity."""
    ports = _reserve_head_ports()
    head = _ray_head_command("head-node", ports, None, io_slots_per_node=9)
    worker = _ray_worker_command("head-node:6379", _reserve_worker_ports(), None, io_slots_per_node=9)

    assert head[head.index("--dashboard-host") + 1] == "127.0.0.1"
    assert "--block" in head
    assert worker[worker.index("--address") + 1] == "head-node:6379"
    assert "--block" in worker
    for command in (head, worker):
        assert json.loads(command[command.index("--resources") + 1]) == {"curator_io": 9}
        # Whole-node allocations let Ray detect CPUs and the GPUs exposed by Slurm.
        assert "--num-cpus" not in command
        assert "--num-gpus" not in command


def test_head_reserves_distinct_free_ports_so_two_heads_can_share_a_node() -> None:
    """The head can share its node, so it cannot assume the well-known Ray ports are free."""
    first = _reserve_head_ports()
    second = _reserve_head_ports()

    assert set(first) == set(_HEAD_PORT_NAMES)
    assert len(set(first.values())) == len(_HEAD_PORT_NAMES)
    assert not set(first.values()) & set(second.values())
    # Every reserved port reaches the command, so nothing silently falls back to a Ray default.
    command = _ray_head_command("head-node", first, None)
    assert set(first.values()) <= {int(argument) for argument in command if argument.isdigit()}


def test_compute_wait_accepts_resources_after_registration() -> None:
    """The driver waits until Ray reports an alive node with compute resources."""
    ray_module = Mock()
    ray_module.nodes.side_effect = [
        [{"Alive": True, "Resources": {"CPU": 0.0, "GPU": 0.0}}],
        [{"Alive": True, "Resources": {"CPU": 0.0, "GPU": 0.0}}],
        [{"Alive": True, "Resources": {"CPU": 1.0, "GPU": 0.0}}],
    ]
    head_process = Mock()
    head_process.poll.return_value = None
    stop_signal = Mock()
    stop_signal.is_set.return_value = False
    stop_signal.wait.return_value = False

    _wait_for_compute(
        ray_module,
        head_process=head_process,
        stop_signal=stop_signal,
    )

    assert ray_module.nodes.call_count == 3
    assert stop_signal.wait.call_count == 2


def _head_args(tmp_path: Path) -> Namespace:
    return Namespace(
        manifest=str(tmp_path / "manifest.json"),
        run_id="cc-ray-deadbeef",
        startup_timeout_seconds=60,
        io_slots_per_node=9,
        temp_dir=None,
        command=["python", "-m", "pipeline"],
    )


def _mock_head_startup(
    monkeypatch: pytest.MonkeyPatch,
    manifest: dict[str, object],
    *,
    wait_for_compute: object | None = None,
) -> tuple[Mock, list[dict[str, object]]]:
    """Stub out Ray startup and drive the supervisor's manifest writes through the real state machine."""
    status_writer = Mock()
    mutations: list[dict[str, object]] = []

    def mutate(_path: Path, _run_id: str, mutation: dict[str, object]) -> dict[str, object]:
        mutations.append(mutation)
        return apply_test_manifest_mutation(manifest, mutation)

    monkeypatch.setattr(f"{RUNTIME_MODULE}.signal.signal", Mock())
    monkeypatch.setattr(f"{RUNTIME_MODULE}._wait_for_submitted_manifest", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(f"{RUNTIME_MODULE}._connect_to_ray", lambda *_args, **_kwargs: Mock())
    monkeypatch.setattr(f"{RUNTIME_MODULE}._state.atomic_write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(f"{RUNTIME_MODULE}._StatusWriter", lambda **_kwargs: status_writer)
    monkeypatch.setattr(
        f"{RUNTIME_MODULE}._wait_for_compute",
        wait_for_compute if wait_for_compute is not None else (lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(f"{RUNTIME_MODULE}._mutate", mutate)
    monkeypatch.setattr(f"{RUNTIME_MODULE}._stop_ray", Mock())
    return status_writer, mutations


@pytest.mark.parametrize("driver_status", [0, 7])
def test_head_supervisor_runs_driver_with_only_head_and_records_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    driver_status: int,
) -> None:
    """Head resources suffice to run the driver and record its outcome while lanes are pending."""
    manifest = make_active_manifest(make_config())
    manifest["state"] = "STARTING"
    head_process = Mock()
    head_process.poll.return_value = None
    driver_process = Mock()
    driver_process.poll.return_value = driver_status
    driver_process.wait.return_value = driver_status
    status_writer, mutations = _mock_head_startup(monkeypatch, manifest)
    ray_module = Mock()
    ray_module.nodes.return_value = [{"Alive": True, "NodeID": "head", "Resources": {"CPU": 32.0, "curator_io": 9.0}}]
    monkeypatch.setattr(f"{RUNTIME_MODULE}._connect_to_ray", lambda *_args, **_kwargs: ray_module)
    monkeypatch.setattr(f"{RUNTIME_MODULE}._wait_for_compute", _wait_for_compute)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "32")
    cleanup_states: list[tuple[str, object]] = []
    status_writer.stop.side_effect = lambda: cleanup_states.append(("status", manifest["state"]))

    def stop_ray(*_args: object) -> None:
        cleanup_states.append(("ray", manifest["state"]))

    monkeypatch.setattr(f"{RUNTIME_MODULE}._stop_ray", stop_ray)
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    popen = Mock(side_effect=[head_process, driver_process])
    monkeypatch.setattr(f"{RUNTIME_MODULE}.subprocess.Popen", popen)

    result = _run_head(_head_args(tmp_path))

    assert result == driver_status
    ray_module.nodes.assert_called_once_with()
    head_command = popen.call_args_list[0].args[0]
    assert head_command[head_command.index("--num-cpus") + 1] == "32"
    assert json.loads(head_command[head_command.index("--resources") + 1]) == {"curator_io": 9}
    assert mutations == [
        {"operation": "activate"},
        {"operation": "record-driver-exit", "exit_status": driver_status},
    ]
    assert manifest["state"] == "STOPPING"
    assert manifest["pipeline_exit_status"] == driver_status
    assert cleanup_states == [("status", "STOPPING"), ("ray", "STOPPING")]
    status_writer.start.assert_called_once_with()
    status_writer.stop.assert_called_once_with()


def test_head_supervisor_records_cleanup_failure_after_driver_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A teardown failure is caught before an orderly driver outcome becomes terminal."""
    manifest = make_active_manifest(make_config())
    manifest["state"] = "STARTING"
    head_process = Mock()
    head_process.poll.return_value = None
    driver_process = Mock()
    driver_process.poll.return_value = 0
    driver_process.wait.return_value = 0
    _, mutations = _mock_head_startup(monkeypatch, manifest)

    def fail_cleanup(*_args: object) -> None:
        msg = "injected Ray cleanup failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(f"{RUNTIME_MODULE}._stop_ray", fail_cleanup)
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    monkeypatch.setattr(f"{RUNTIME_MODULE}.subprocess.Popen", Mock(side_effect=[head_process, driver_process]))

    result = _run_head(_head_args(tmp_path))

    assert result == 1
    assert [mutation["operation"] for mutation in mutations] == [
        "activate",
        "record-driver-exit",
        "record-failure",
    ]
    assert manifest["state"] == "STOPPING"
    assert manifest["pipeline_exit_status"] == 0
    assert manifest["error"] == "injected Ray cleanup failure"


def test_driver_is_stopped_when_monitoring_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pipeline process cannot escape if driver monitoring itself fails."""
    driver_process = Mock()
    driver_process.poll.return_value = None
    head_process = Mock()
    head_process.poll.side_effect = RuntimeError("injected poll failure")
    stop_process = Mock()
    monkeypatch.setattr(f"{RUNTIME_MODULE}.subprocess.Popen", Mock(return_value=driver_process))
    monkeypatch.setattr(f"{RUNTIME_MODULE}._stop_process", stop_process)

    with pytest.raises(RuntimeError, match="injected poll failure"):
        _run_driver(["python", "-m", "pipeline"], "head:6379", head_process, Mock())

    stop_process.assert_called_once_with(driver_process)


def test_head_supervisor_does_not_publish_terminal_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The head leaves a durable outcome for wrapper or status finalization."""
    manifest = make_active_manifest(make_config())
    manifest["state"] = "STARTING"
    head_process = Mock()
    head_process.poll.return_value = None
    driver_process = Mock()
    driver_process.poll.return_value = 0
    driver_process.wait.return_value = 0
    _, mutations = _mock_head_startup(monkeypatch, manifest)
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    monkeypatch.setattr(f"{RUNTIME_MODULE}.subprocess.Popen", Mock(side_effect=[head_process, driver_process]))

    result = _run_head(_head_args(tmp_path))

    assert result == 0
    assert [mutation["operation"] for mutation in mutations] == [
        "activate",
        "record-driver-exit",
    ]
    assert manifest["state"] == "STOPPING"
    assert manifest["pipeline_exit_status"] == 0


def test_head_supervisor_fails_run_when_ray_head_exits_unexpectedly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead Ray head terminates the driver and records a useful failure."""
    manifest = make_active_manifest(make_config())
    manifest["state"] = "STARTING"
    head_process = Mock(returncode=1)
    head_process.poll.return_value = 1
    driver_process = Mock()
    driver_process.poll.side_effect = [None, None, 143]
    driver_process.wait.return_value = 143
    _, mutations = _mock_head_startup(monkeypatch, manifest)
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    monkeypatch.setattr(f"{RUNTIME_MODULE}.subprocess.Popen", Mock(side_effect=[head_process, driver_process]))
    monkeypatch.setattr(f"{RUNTIME_MODULE}._stop_process", Mock())

    result = _run_head(_head_args(tmp_path))

    assert result == 1
    assert mutations[-1] == {
        "operation": "record-failure",
        "exit_status": 143,
        "error": "Ray head exited unexpectedly with status 1",
    }
    assert manifest["state"] == "STOPPING"
    assert manifest["pipeline_exit_status"] == 143


def test_head_supervisor_stops_when_the_run_left_submitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run stopped during Ray startup never starts the driver."""
    manifest = make_active_manifest(make_config())
    manifest["state"] = "STOPPING"
    head_process = Mock()
    head_process.poll.return_value = None
    _, mutations = _mock_head_startup(monkeypatch, manifest)
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    popen = Mock(return_value=head_process)
    monkeypatch.setattr(f"{RUNTIME_MODULE}.subprocess.Popen", popen)

    result = _run_head(_head_args(tmp_path))

    assert result == 143
    assert mutations == [{"operation": "activate"}]
    assert manifest["state"] == "STOPPING"
    assert popen.call_count == 1


def test_head_supervisor_honors_shutdown_signal_before_driver_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A signal during startup prevents the pipeline driver from launching."""
    manifest = make_active_manifest(make_config())
    manifest["state"] = "STARTING"
    head_process = Mock()
    head_process.poll.return_value = None
    handlers: dict[int, object] = {}

    def wait_for_compute(*_args: object, **_kwargs: object) -> None:
        handler = handlers[signal.SIGTERM]
        assert callable(handler)
        handler(signal.SIGTERM, None)

    status_writer, mutations = _mock_head_startup(monkeypatch, manifest, wait_for_compute=wait_for_compute)
    monkeypatch.setattr(f"{RUNTIME_MODULE}.signal.signal", lambda number, handler: handlers.update({number: handler}))
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    popen = Mock(return_value=head_process)
    monkeypatch.setattr(f"{RUNTIME_MODULE}.subprocess.Popen", popen)

    result = _run_head(_head_args(tmp_path))

    assert result == 143
    assert popen.call_count == 1
    assert mutations == []
    status_writer.stop.assert_called_once_with()


def test_process_shutdown_escalates_after_terminate_timeout() -> None:
    """A process that ignores SIGTERM is killed after a bounded wait."""
    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired(cmd="driver", timeout=15), 137]

    _stop_process(process)

    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.wait.call_args_list == [call(timeout=15), call(timeout=5)]


@pytest.mark.parametrize("role", ["head", "worker"])
def test_sliced_allocation_advertises_its_cpus(monkeypatch: pytest.MonkeyPatch, role: str) -> None:
    """An allocation that shares a node must tell Ray its slice, not the machine's core count.

    Ray autodetects CPUs from the host, so three 32-core lanes on a 96-core node would each advertise 96 and
    let the scheduler place three times the work the allocation can actually run.
    """
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "32")

    command = (
        _ray_head_command("head-node", _reserve_head_ports(), None, 8, _allocated_cpus())
        if role == "head"
        else _ray_worker_command("head-node:6379", _reserve_worker_ports(), None, 8, _allocated_cpus())
    )

    assert command[command.index("--num-cpus") + 1] == "32"


@pytest.mark.parametrize("role", ["head", "worker"])
@pytest.mark.parametrize("value", [None, "", "   ", "not-a-number", "0", "-4"])
def test_allocation_lets_ray_autodetect_when_slurm_gives_no_usable_cpu_count(
    monkeypatch: pytest.MonkeyPatch, role: str, value: str | None
) -> None:
    """Whole-node allocations and non-Slurm runs keep Ray's autodetection."""
    if value is None:
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    else:
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", value)

    command = (
        _ray_head_command("head-node", _reserve_head_ports(), None, 8, _allocated_cpus())
        if role == "head"
        else _ray_worker_command("head-node:6379", _reserve_worker_ports(), None, 8, _allocated_cpus())
    )

    assert "--num-cpus" not in command


def test_stopping_one_run_leaves_a_co_located_run_alive() -> None:
    """Teardown must not reach another allocation's Ray processes on the same node.

    `ray stop --force` swept every local Ray process the user owned, so a head finishing killed a co-located
    run's raylet. Lanes that ask for a slice share nodes routinely, so this is now the normal case rather than
    an edge one. Two real process groups stand in for the two runs here: each leader spawns a child, and
    stopping one group must leave the other group's leader and child running.
    """
    script = "import time; time.sleep(60)"
    spawn = f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {script!r}]); time.sleep(60)"

    mine = subprocess.Popen([sys.executable, "-c", spawn], start_new_session=True)  # noqa: S603
    theirs = subprocess.Popen([sys.executable, "-c", spawn], start_new_session=True)  # noqa: S603
    try:
        _wait_until(lambda: _process_group_size(theirs) >= 2, "the co-located run to spawn its child")
        _wait_until(lambda: _process_group_size(mine) >= 2, "this run to spawn its child")
        survivors_before = _process_group_size(theirs)

        _terminate_process_group(mine)

        assert mine.poll() is not None, "this run's Ray process group outlived teardown"
        assert theirs.poll() is None, "teardown killed the co-located run's leader"
        assert _process_group_size(theirs) == survivors_before, "teardown killed the co-located run's children"
    finally:
        for process in (mine, theirs):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)


def _process_group_size(process: subprocess.Popen[bytes]) -> int:
    """Count live processes in the group *process* leads."""
    try:
        group_id = os.getpgid(process.pid)
    except ProcessLookupError:
        return 0
    listing = subprocess.run(
        ["ps", "-e", "-o", "pgid="],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    return sum(1 for line in listing.stdout.split() if line.strip() == str(group_id))


def _wait_until(condition: Callable[[], bool], description: str, timeout_seconds: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.05)
    msg = f"Timed out waiting for {description}"
    raise AssertionError(msg)


def _fake_ray_start(script: str) -> list[str]:
    """Build a stand-in for ``ray start`` that prints *script*'s output and exits with its status."""
    return [sys.executable, "-c", script]


def test_worker_retries_with_fresh_ports_when_a_lane_loses_the_port_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reserving free ports does not close the race; the loser must re-draw rather than die.

    The sockets have to close before Ray can bind them, so two lanes starting together on one node can be
    handed the same port. The first attempt here fails the way Ray does, and the lane must start again on a
    different port rather than surfacing "Address already in use" as a dead lane.
    """
    attempts: list[list[str]] = []
    real_popen = subprocess.Popen

    def fake_popen(command: list[str], **kwargs: object) -> subprocess.Popen[str]:
        attempts.append(command)
        failing = "import sys; print('Address already in use', file=sys.stdout); sys.exit(1)"
        script = failing if len(attempts) == 1 else "print('started')"
        return real_popen(_fake_ray_start(script), **kwargs)  # type: ignore[arg-type,return-value]

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    returncode = _start_worker_ray("head-node:6379", None, 8)

    assert returncode == 0, "the lane should have started on its second attempt"
    assert len(attempts) == 2, f"expected one retry, got {len(attempts)} attempt(s)"

    def ports(command: list[str]) -> set[str]:
        return {command[i + 1] for i, flag in enumerate(command) if flag == "--node-manager-port"}

    assert ports(attempts[0]) != ports(attempts[1]), "the retry reused the port that was already taken"


def test_worker_does_not_retry_a_failure_that_is_not_a_port_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real misconfiguration must fail fast instead of being retried three times."""
    attempts: list[list[str]] = []
    real_popen = subprocess.Popen

    def fake_popen(command: list[str], **kwargs: object) -> subprocess.Popen[str]:
        attempts.append(command)
        script = "import sys; print('Unknown flag --nonsense'); sys.exit(2)"
        return real_popen(_fake_ray_start(script), **kwargs)  # type: ignore[arg-type,return-value]

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    returncode = _start_worker_ray("head-node:6379", None, 8)

    assert returncode == 2
    assert len(attempts) == 1, "a non-port failure must not be retried"


def test_worker_gives_up_after_bounded_port_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lane that loses every race reports the failure rather than looping forever."""
    attempts: list[list[str]] = []
    real_popen = subprocess.Popen

    def fake_popen(command: list[str], **kwargs: object) -> subprocess.Popen[str]:
        attempts.append(command)
        script = "import sys; print('Address already in use'); sys.exit(1)"
        return real_popen(_fake_ray_start(script), **kwargs)  # type: ignore[arg-type,return-value]

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    returncode = _start_worker_ray("head-node:6379", None, 8)

    assert returncode == 1
    assert len(attempts) == _WORKER_START_ATTEMPTS


def _write_ray_session_log(temp_dir: Path, name: str, text: str) -> Path:
    """Write *text* to a log under the Ray session directory Ray would create beneath *temp_dir*."""
    logs = temp_dir / "session_latest" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / name
    path.write_text(text)
    return path


def test_worker_retries_when_only_raylet_err_names_the_bind_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`ray start` hides a lost port behind a generic startup timeout.

    Holding a pre-selected object-manager port open makes the raylet fail to bind, but the command itself
    only prints a startup timeout -- "Address already in use" reaches `raylet.err` alone. Scanning the
    command's output was therefore blind to the one failure the retry exists for, and the lane exited after
    a single attempt.
    """
    attempts: list[list[str]] = []
    real_popen = subprocess.Popen

    def fake_popen(command: list[str], **kwargs: object) -> subprocess.Popen[str]:
        attempts.append(command)
        if len(attempts) == 1:
            written = _write_ray_session_log(tmp_path, "raylet.err", "Failed to bind: Address already in use\n")
            # Backdate it to what a filesystem with one-second timestamps would stamp on a file written
            # moments ago: comparing that mtime against a clock read before the attempt calls it stale, which
            # is exactly how this detection failed in CI while passing on a nanosecond-timestamp filesystem.
            stale = time.time() - 1.5
            os.utime(written, (stale, stale))
            script = "import sys; print('Timed out while waiting for Ray to start'); sys.exit(1)"
        else:
            script = "print('started')"
        return real_popen(_fake_ray_start(script), **kwargs)  # type: ignore[arg-type,return-value]

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    returncode = _start_worker_ray("head-node:6379", tmp_path, 8)

    assert returncode == 0, "the lane should have retried and started"
    assert len(attempts) == 2, f"expected one retry, got {len(attempts)} attempt(s)"


def test_worker_ignores_a_bind_error_left_by_an_earlier_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stale log must not make an unrelated failure look like a port collision."""
    stale = _write_ray_session_log(tmp_path, "raylet.err", "Failed to bind: Address already in use\n")
    os.utime(stale, (time.time() - 3600, time.time() - 3600))
    attempts: list[list[str]] = []
    real_popen = subprocess.Popen

    def fake_popen(command: list[str], **kwargs: object) -> subprocess.Popen[str]:
        attempts.append(command)
        return real_popen(  # type: ignore[arg-type,return-value]
            _fake_ray_start("import sys; print('Unknown flag'); sys.exit(2)"), **kwargs
        )

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    returncode = _start_worker_ray("head-node:6379", tmp_path, 8)

    assert returncode == 2
    assert len(attempts) == 1, "a stale bind error must not trigger a retry"
