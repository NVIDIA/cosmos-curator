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
"""Tests for the batch scripts and container paths a managed Ray run is rendered into."""

import subprocess
from pathlib import Path

import pytest

from cosmos_curator.client.slurm_cli.managed_ray.config import (
    SlurmRayConfig,
    lane_allocations,
    resolve_slurm_ray_config_data,
    slurm_time_seconds,
)
from cosmos_curator.client.slurm_cli.managed_ray.lifecycle import (
    _lane_options,
)
from cosmos_curator.client.slurm_cli.managed_ray.onnode.slurm_ray_runtime import (
    RUNTIME_MODULE_FILENAME,
)
from cosmos_curator.client.slurm_cli.managed_ray.render import (
    capture_forwarded_environment,
    render_head_script,
    render_worker_script,
    resolve_runtime_paths,
)
from cosmos_curator.client.slurm_cli.slurm_common import _LOG_ENV_VARS_TO_FORWARD
from tests.cosmos_curator.client.slurm_cli.managed_ray.launcher_stubs import (
    make_config,
)


def test_rendered_jobs_ask_for_what_each_role_needs(tmp_path: Path) -> None:
    """The head and every worker lane have the intended Slurm lifecycle."""
    config = make_config()
    run_id = "cc-ray-deadbeef"
    runtime_paths = resolve_runtime_paths(
        config,
        home=Path("/remote/home"),
        run_id=run_id,
        state_dir="~/slurm-ray",
        host_path_exists=lambda _path: True,
        forwarded_environment_keys=["HOST_ONLY"],
    )
    head_script = render_head_script(
        config,
        run_id=run_id,
        runtime_paths=runtime_paths,
        command=["python", "-m", "pipeline", "--flag", "value with spaces"],
    )
    worker_script = render_worker_script(config, run_id=run_id, runtime_paths=runtime_paths, allocations=1)

    assert runtime_paths["run_dir"] == "/remote/home/slurm-ray/cc-ray-deadbeef"
    assert {
        (mount["source"], mount["destination"])
        for mount in runtime_paths["mounts"]
        if mount["destination"] in {"/config", "/run/cosmos-curator/slurm-ray"}
    } == {
        ("/remote/home/cosmos_curator_local_workspace", "/config"),
        ("/remote/home/slurm-ray/cc-ray-deadbeef", "/run/cosmos-curator/slurm-ray"),
    }
    assert "#SBATCH --partition=cpu" in head_script
    assert "#SBATCH --exclusive" in head_script
    assert "#SBATCH --mem=0" in head_script
    assert "--cpus-per-task" not in head_script
    assert "#SBATCH --gpus" not in head_script
    assert "#SBATCH --no-requeue" in head_script
    assert "source /remote/home/slurm-ray/cc-ray-deadbeef/environment.sh" in head_script
    assert "NVIDIA_VISIBLE_DEVICES" not in head_script
    assert "HOST_ONLY" in head_script.split("--container-env", maxsplit=1)[1]
    assert "CURATOR_RUN_ID" in head_script.split("--container-env", maxsplit=1)[1]
    assert "--num-cpus 0" not in head_script
    assert "'value with spaces'" in head_script
    assert f"/{RUNTIME_MODULE_FILENAME} head" in head_script
    assert f"/{RUNTIME_MODULE_FILENAME} cleanup" in head_script
    assert "trap 'finalize_run \"$?\"' EXIT" in head_script
    assert "squeue" not in head_script
    assert "scancel" not in head_script
    assert "#SBATCH --partition=gpu" in worker_script
    assert "#SBATCH --gpus=8" in worker_script
    # Workers use the same allocation defaults but keep their own GPU request and lifecycle.
    assert "#SBATCH --exclusive" in worker_script
    assert "#SBATCH --mem=0" in worker_script
    assert "#SBATCH --requeue" in worker_script
    assert "#SBATCH --dependency" not in worker_script
    assert "#SBATCH --open-mode=append" in worker_script
    assert "NVIDIA_VISIBLE_DEVICES" not in worker_script
    assert "submitted_run_id=$1" in worker_script
    assert "head_job_id=$2" in worker_script
    assert "lane=$3" in worker_script
    assert f"/{RUNTIME_MODULE_FILENAME} worker" in worker_script
    assert f'/{RUNTIME_MODULE_FILENAME} probe-head --job-id "$head_job_id" || exit 1' in worker_script
    assert "squeue" not in worker_script

    for name, script in [
        ("head.sbatch", head_script),
        ("worker.sbatch", worker_script),
    ]:
        assert "#SBATCH --propagate=NONE" in script
        assert "--propagate=NONE" in script.split("srun", maxsplit=1)[1]
        script_path = tmp_path / name
        script_path.write_text(script, encoding="utf-8")
        subprocess.run(["bash", "-n", str(script_path)], check=True)  # noqa: S603,S607


def test_lane_renewal_is_derived_from_the_two_walltimes() -> None:
    """A worker walltime shorter than the head's is the whole expression of renewal."""

    def _config_with(head: str, worker: str) -> SlurmRayConfig:
        return resolve_slurm_ray_config_data(
            {"schema_version": 1, "slurm": {"head": {"time": head}, "worker": {"time": worker}}}
        )

    assert lane_allocations(_config_with("7-00:00:00", "7-00:00:00")) == 1
    assert lane_allocations(_config_with("7-00:00:00", "04:00:00")) == 42
    assert lane_allocations(_config_with("7-00:00:00", "08:00:00")) == 21
    # A worker allowed to outlive its head still needs exactly one allocation.
    assert lane_allocations(_config_with("04:00:00", "7-00:00:00")) == 1
    # Uneven division rounds up rather than leaving the tail of the run uncovered.
    assert lane_allocations(_config_with("05:00:00", "04:00:00")) == 2


def test_renewing_lanes_are_submitted_as_independent_throttled_arrays() -> None:
    """A lane that renews gets its own throttled array; a lane that does not gets a plain job."""
    assert _lane_options("100", 1) == ("--dependency=after:100",)
    assert _lane_options("100", 42) == ("--dependency=after:100", "--array=0-41%1")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("30", 1800),
        ("30:00", 1800),
        ("4:00:00", 14400),
        ("04:00:00", 14400),
        ("7-0", 604800),
        ("7-00:00", 604800),
        ("7-00:00:00", 604800),
    ],
)
def test_every_sbatch_walltime_spelling_is_understood(value: str, expected: int) -> None:
    """All six formats sbatch accepts resolve, so the derived depth cannot be silently wrong."""
    assert slurm_time_seconds(value) == expected


@pytest.mark.parametrize("value", ["UNLIMITED", "INFINITE", "0", "0:00", "", "4h", "-1", "1-2-3"])
def test_walltimes_without_a_finite_bound_are_rejected(value: str) -> None:
    """The head's walltime is the run's lifetime, so it has to be a stated finite number."""
    with pytest.raises(ValueError, match="Walltime"):
        slurm_time_seconds(value)


def test_launcher_owned_node_local_sources_are_always_created() -> None:
    """A configured Ray temp dir is mounted and created without needing an unrelated opt-in flag."""
    config = resolve_slurm_ray_config_data(
        {
            "schema_version": 1,
            "ray": {"temp_dir": "/raid/ray", "io_slots_per_node": 9},
            "runtime": {"node_local_mounts": [{"source": "/raid/scratch", "destination": "/scratch"}]},
        }
    )
    runtime_paths = resolve_runtime_paths(
        config,
        home=Path("/remote/home"),
        run_id="cc-ray-deadbeef",
        state_dir="~/slurm-ray",
        host_path_exists=lambda _path: True,
        forwarded_environment_keys=[],
    )
    script = render_worker_script(config, run_id="cc-ray-deadbeef", runtime_paths=runtime_paths, allocations=1)
    head_script = render_head_script(
        config, run_id="cc-ray-deadbeef", runtime_paths=runtime_paths, command=["python", "-m", "pipeline"]
    )

    assert "/raid/ray" in {mount["source"] for mount in runtime_paths["mounts"]}
    # prepare_node_local_mounts is off by default, so only the launcher's own source is created.
    assert runtime_paths["prepare_directories"] == ["/raid/ray"]
    assert "mkdir -p -- /raid/ray" in script
    assert "mkdir -p -- /raid/scratch" not in script
    assert "--io-slots-per-node 9" in script
    assert "--io-slots-per-node 9" in head_script


def test_configured_node_local_mounts_are_created_on_request() -> None:
    """Opting in adds the user's node-local sources alongside the launcher's own."""
    config = resolve_slurm_ray_config_data(
        {
            "schema_version": 1,
            "ray": {"temp_dir": "/raid/ray"},
            "runtime": {
                "prepare_node_local_mounts": True,
                "node_local_mounts": [{"source": "/raid/scratch", "destination": "/scratch"}],
            },
        }
    )
    runtime_paths = resolve_runtime_paths(
        config,
        home=Path("/remote/home"),
        run_id="cc-ray-deadbeef",
        state_dir="~/slurm-ray",
        host_path_exists=lambda _path: True,
        forwarded_environment_keys=[],
    )

    assert runtime_paths["prepare_directories"] == ["/raid/scratch", "/raid/ray"]


def test_capture_forwarded_environment_silently_skips_unset_logging_variables(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Only explicitly requested missing variables produce warnings."""
    for name in _LOG_ENV_VARS_TO_FORWARD:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYTHON_LOG", "debug")
    monkeypatch.setenv("PYTHON_LOG_FORMAT", "json")

    values = capture_forwarded_environment(make_config(environment=["PYTHON_LOG_FORMAT", "EXPLICIT_MISSING"]))

    assert values == {"PYTHON_LOG": "debug", "PYTHON_LOG_FORMAT": "json"}
    assert [record.getMessage() for record in caplog.records] == [
        "Environment variable EXPLICIT_MISSING is not set; not forwarding it to the container"
    ]


@pytest.mark.parametrize("role", ["head", "worker"])
def test_allocation_requests_a_slice_when_not_exclusive(role: str) -> None:
    """A non-exclusive allocation states cpus/memory instead of claiming the whole node.

    A whole-node request only fits a wholly free node; a slice also packs onto partly used ones,
    which is what a CPU-only recipe on a shared partition needs.
    """
    config = resolve_slurm_ray_config_data(
        {
            "schema_version": 1,
            "job_name": "curator",
            "worker_lanes": 2,
            "slurm": {
                "account": "acct",
                role: {
                    "partition": "cpu",
                    "qos": "cpu-free",
                    "time": "04:00:00",
                    "exclusive": False,
                    "cpus": 24,
                    "memory": "48G",
                },
            },
            "runtime": {"mount_s3_creds": False},
            "ray": {"startup_timeout": "5m"},
        }
    )
    run_id = "cc-ray-deadbeef"
    runtime_paths = resolve_runtime_paths(
        config,
        home=Path("/remote/home"),
        run_id=run_id,
        state_dir="~/slurm-ray",
        host_path_exists=lambda _path: True,
        forwarded_environment_keys=[],
    )

    script = (
        render_head_script(config, run_id=run_id, runtime_paths=runtime_paths, command=["python", "-m", "pipeline"])
        if role == "head"
        else render_worker_script(config, run_id=run_id, runtime_paths=runtime_paths, allocations=1)
    )

    assert "#SBATCH --cpus-per-task=24" in script
    assert "#SBATCH --mem=48G" in script
    assert "#SBATCH --exclusive" not in script
    assert "#SBATCH --mem=0" not in script
    # Since Slurm 22.05 the step does not inherit --cpus-per-task, so it must be restated or the
    # step lands on a single core while the rest of the allocation idles.
    assert "--cpus-per-task=24" in script.split("srun")[-1]


@pytest.mark.parametrize("role", ["head", "worker"])
def test_allocation_takes_whole_node_by_default(role: str) -> None:
    """Both roles use the whole node without limiting the step to the sliced CPU default."""
    config = make_config()
    run_id = "cc-ray-deadbeef"
    runtime_paths = resolve_runtime_paths(
        config,
        home=Path("/remote/home"),
        run_id=run_id,
        state_dir="~/slurm-ray",
        host_path_exists=lambda _path: True,
        forwarded_environment_keys=[],
    )

    script = (
        render_head_script(config, run_id=run_id, runtime_paths=runtime_paths, command=["python", "-m", "pipeline"])
        if role == "head"
        else render_worker_script(config, run_id=run_id, runtime_paths=runtime_paths, allocations=1)
    )

    assert "#SBATCH --exclusive" in script
    assert "#SBATCH --mem=0" in script
    assert "--cpus-per-task" not in script


@pytest.mark.parametrize("gpu_field", ["gpus", "gpus_per_node"])
def test_head_and_worker_gpu_requests_are_independent(gpu_field: str) -> None:
    """A GPU head keeps its requested GPUs visible and uses its own partition and GPU count."""
    config = resolve_slurm_ray_config_data(
        {
            "schema_version": 1,
            "slurm": {
                "head": {"partition": "gpu-long", gpu_field: 4},
                "worker": {"partition": "gpu-batch", gpu_field: 8},
            },
        }
    )
    run_id = "cc-ray-deadbeef"
    runtime_paths = resolve_runtime_paths(
        config,
        home=Path("/remote/home"),
        run_id=run_id,
        state_dir="~/slurm-ray",
        host_path_exists=lambda _path: True,
        forwarded_environment_keys=[],
    )
    head_script = render_head_script(
        config, run_id=run_id, runtime_paths=runtime_paths, command=["python", "-m", "pipeline"]
    )
    worker_script = render_worker_script(config, run_id=run_id, runtime_paths=runtime_paths, allocations=1)

    flag = gpu_field.replace("_", "-")
    assert f"#SBATCH --{flag}=4" in head_script
    assert "#SBATCH --partition=gpu-long" in head_script
    assert "NVIDIA_VISIBLE_DEVICES" not in head_script
    assert f"#SBATCH --{flag}=8" in worker_script
    assert "#SBATCH --partition=gpu-batch" in worker_script
