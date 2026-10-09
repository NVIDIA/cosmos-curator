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
"""Tests for managed Slurm-Ray submission config resolution."""

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from cosmos_curator.client.cli import cosmos_curator
from cosmos_curator.client.slurm_cli.managed_ray.config import (
    lane_allocations,
    resolve_container_image,
    resolve_slurm_ray_config,
    resolve_slurm_ray_config_data,
    startup_timeout_seconds,
    validate_state_dir,
)

runner = CliRunner()


def _write_config(path: Path, payload: dict[str, object]) -> Path:
    if path.suffix == ".json":
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_yaml_and_json_resolve_to_the_same_canonical_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equivalent JSON and YAML inputs resolve identically."""
    monkeypatch.delenv("SBATCH_ACCOUNT", raising=False)
    payload: dict[str, object] = {
        "schema_version": 1,
        "worker_lanes": 4,
        "slurm": {
            "head": {"partition": "cpu", "time": "1-00:00:00"},
            "worker": {"partition": "gpu", "gpus": 8},
        },
    }

    yaml_config = resolve_slurm_ray_config(_write_config(tmp_path / "cluster.yaml", payload))
    json_config = resolve_slurm_ray_config(_write_config(tmp_path / "cluster.json", payload))

    assert yaml_config == json_config
    assert yaml_config.worker_lanes == 4
    assert yaml_config.slurm.worker.gpus == 8
    assert yaml_config.runtime.workspace_path == "~/cosmos_curator_local_workspace"
    assert "state_dir" not in yaml_config.model_dump()


def test_container_image_accepts_registry_references(tmp_path: Path) -> None:
    """Registry references (tag or digest) pass through for the node runtime to pull."""
    refs = [
        "registry.example.com#org/repository:2.2.0",
        "registry.example.com#org/repository:sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ]
    for ref in refs:
        payload: dict[str, object] = {"schema_version": 1, "runtime": {"container_image": ref}}
        config = resolve_slurm_ray_config(_write_config(tmp_path / "cluster.yaml", payload))
        assert config.runtime.container_image == ref
        assert resolve_container_image(ref, Path("/home/user")) == ref
    assert resolve_container_image("~/images/run.sqsh", Path("/home/user")) == "/home/user/images/run.sqsh"


@pytest.mark.parametrize(
    "bad",
    ["images/run.sqsh", "registry.example.com:5005/org/repository:2.2.0"],
)
def test_container_image_rejects_non_reference_values(tmp_path: Path, bad: str) -> None:
    """Values that are neither host paths nor '#' registry references fail host-path validation."""
    payload: dict[str, object] = {"schema_version": 1, "runtime": {"container_image": bad}}
    with pytest.raises(ValidationError):
        resolve_slurm_ray_config(_write_config(tmp_path / "cluster.yaml", payload))


def test_max_lane_allocations_caps_derived_lane_array(tmp_path: Path) -> None:
    """max_lane_allocations caps the head/worker-derived lane array; a loose cap is a no-op."""

    def resolve_with(max_cap: int | None) -> int:
        worker: dict[str, object] = {"partition": "gpu", "gpus": 1, "time": "00:30:00"}
        if max_cap is not None:
            worker["max_lane_allocations"] = max_cap
        payload: dict[str, object] = {
            "schema_version": 1,
            "worker_lanes": 1,
            "slurm": {"head": {"partition": "cpu", "time": "12:00:00"}, "worker": worker},
        }
        config = resolve_slurm_ray_config(_write_config(tmp_path / f"cluster-{max_cap}.yaml", payload))
        return lane_allocations(config)

    assert resolve_with(None) == 24  # 12h head over 30m slots
    assert resolve_with(48) == 24  # loose cap is a no-op
    assert resolve_with(20) == 20  # tight cap binds
    with pytest.raises(ValidationError):
        resolve_with(0)


def test_environment_account_default_and_set_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment-derived defaults precede config and typed --set overrides."""
    monkeypatch.setenv("SBATCH_ACCOUNT", "account-from-environment")

    config = resolve_slurm_ray_config_data(
        {"schema_version": 1},
        overrides=[
            "worker_lanes=8",
            "slurm.worker.gpus=8",
            "runtime.mount_s3_creds=false",
            "ray.startup_timeout=2h",
            "ray.io_slots_per_node=9",
        ],
    )

    assert config.slurm.account == "account-from-environment"
    assert config.worker_lanes == 8
    assert config.slurm.worker.gpus == 8
    assert config.runtime.mount_s3_creds is False
    assert startup_timeout_seconds(config.ray.startup_timeout) == 7200
    assert config.ray.io_slots_per_node == 9


def test_explicit_null_account_uses_slurm_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit null account overrides the environment-derived default."""
    monkeypatch.setenv("SBATCH_ACCOUNT", "account-from-environment")

    config = resolve_slurm_ray_config_data(
        {
            "schema_version": 1,
            "slurm": {"account": None},
        }
    )

    assert config.slurm.account is None


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 1, "unknown": True},
        {
            "schema_version": 1,
            "slurm": {"worker": {"gpus": "8"}},
        },
        {
            "schema_version": 1,
            "slurm": {"worker": {"gpus": 0}},
        },
        {
            "schema_version": 1,
            "slurm": {"worker": {"gres": "gpu:8"}},
        },
        {
            "schema_version": 1,
            "slurm": {"worker": {"constraint": "h100"}},
        },
        # A worker may state a slice, but the values still have to be sane.
        {
            "schema_version": 1,
            "slurm": {"worker": {"cpus": 0}},
        },
        {
            "schema_version": 1,
            "slurm": {"worker": {"memory": "64 GB"}},
        },
        # "0" is how sbatch spells "all of the node"; exclusive=true expresses that instead.
        {
            "schema_version": 1,
            "slurm": {"worker": {"memory": "0"}},
        },
        {
            "schema_version": 1,
            "slurm": {"head": {"cpus": 0}},
        },
        # "0" is how sbatch spells "all of the node", which is what the head must not ask for.
        {
            "schema_version": 1,
            "slurm": {"head": {"memory": "0"}},
        },
        {
            "schema_version": 1,
            "slurm": {"head": {"memory": "64 GB"}},
        },
        {
            "schema_version": 1,
            "ray": {"io_slots_per_node": 0},
        },
        {
            "schema_version": 1,
            "runtime": {"environment": ["TOKEN=value"]},
        },
        {
            "schema_version": 1,
            "runtime": {
                "mounts": [
                    {
                        "source": "relative/path",
                        "destination": "/data",
                        "mode": "rw",
                    }
                ]
            },
        },
    ],
)
def test_config_rejects_unknown_or_unsafe_values(payload: dict[str, object]) -> None:
    """Strict models reject unknown fields, invalid values, and unsafe paths."""
    with pytest.raises(ValidationError):
        resolve_slurm_ray_config_data(payload)


@pytest.mark.parametrize("state_dir", ["/shared/ray state", "/shared/ray#state", "/shared/ray%state"])
def test_launcher_rejects_state_directories_unsafe_for_sbatch_directives(state_dir: str) -> None:
    """The launcher-wide state path must be safe to embed in an SBATCH directive."""
    with pytest.raises(ValueError, match="whitespace"):
        validate_state_dir(state_dir)


@pytest.mark.parametrize("mount_field", ["mounts", "node_local_mounts"])
@pytest.mark.parametrize(
    "destination",
    ["/run/cosmos-curator/slurm-ray", "/run/cosmos-curator/slurm-ray/cache"],
)
def test_config_reserves_the_managed_state_subtree(mount_field: str, destination: str) -> None:
    """User mounts cannot hide all or part of the managed run state."""
    with pytest.raises(ValidationError, match="descendants are reserved"):
        resolve_slurm_ray_config_data(
            {
                "schema_version": 1,
                "runtime": {
                    mount_field: [
                        {
                            "source": "/shared/data",
                            "destination": destination,
                            "mode": "rw",
                        }
                    ]
                },
            }
        )


def test_template_schema_validate_and_render_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The public config CLI exposes an editable template and canonical JSON."""
    monkeypatch.delenv("SBATCH_ACCOUNT", raising=False)
    template_result = runner.invoke(cosmos_curator, ["slurm", "ray", "template"])
    schema_result = runner.invoke(cosmos_curator, ["slurm", "ray", "schema"])
    config_path = tmp_path / "cluster.yaml"
    config_path.write_text(template_result.stdout, encoding="utf-8")

    validate_result = runner.invoke(
        cosmos_curator,
        ["slurm", "ray", "validate", str(config_path), "--set", "worker_lanes=3", "--json"],
    )
    render_result = runner.invoke(
        cosmos_curator,
        ["slurm", "ray", "render", str(config_path), "--set", "worker_lanes=3"],
    )

    assert template_result.exit_code == 0
    template = yaml.safe_load(template_result.stdout)
    assert template["schema_version"] == 1
    assert "state_dir" not in template
    assert template["slurm"]["worker"]["gpus"] is None
    assert schema_result.exit_code == 0
    schema = json.loads(schema_result.stdout)
    assert schema["title"] == "SlurmRayConfig"
    assert "gpus" in schema["$defs"]["SlurmRayWorkerConfig"]["properties"]
    assert validate_result.exit_code == 0
    assert json.loads(validate_result.stdout)["config"]["worker_lanes"] == 3
    assert render_result.exit_code == 0
    assert json.loads(render_result.stdout)["worker_lanes"] == 3


def test_validate_json_reports_field_located_errors(tmp_path: Path) -> None:
    """Machine-readable validation errors retain Pydantic field locations."""
    config_path = _write_config(
        tmp_path / "cluster.yaml",
        {
            "schema_version": 1,
            "slurm": {"worker": {"gpus": "8"}},
        },
    )

    result = runner.invoke(cosmos_curator, ["slurm", "ray", "validate", str(config_path), "--json"])

    assert result.exit_code == 2
    payload = json.loads(result.stderr)
    assert payload["ok"] is False
    assert payload["error"] == "invalid_config"
    assert payload["details"][0]["loc"] == ["slurm", "worker", "gpus"]
