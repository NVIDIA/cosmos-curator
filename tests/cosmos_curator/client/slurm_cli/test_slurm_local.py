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
"""Test the interactive Slurm launcher."""

from pathlib import Path
from unittest.mock import patch

from _pytest.monkeypatch import MonkeyPatch
from typer.testing import CliRunner

from cosmos_curator.client.cli import cosmos_curator

MODULE_NAME = "cosmos_curator.client.slurm_cli.slurm_local"
runner = CliRunner()


def _create_repo(root: Path) -> Path:
    repo = root / "repo"
    (repo / "cosmos_curator" / "pipelines").mkdir(parents=True)
    (repo / "tests" / "cosmos_curator").mkdir(parents=True)
    (repo / "tools").mkdir()
    for filename in ("pixi.toml", "pixi.lock", "pyproject.toml", "pytest.ini", ".coveragerc"):
        (repo / filename).write_text("test")
    return repo


def _container_mounts(command: list[str]) -> list[str]:
    return command[command.index("--container-mounts") + 1].split(",")


def _container_env_keys(command: list[str]) -> list[str]:
    return command[command.index("--container-env") + 1].split(",")


def test_slurm_launch_command_uses_srun_and_live_source_mounts(  # noqa: PLR0915
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Verify slurm launch forms an srun/Pyxis command without mounting host .pixi."""
    repo = _create_repo(tmp_path)
    workspace = tmp_path / "workspace"
    cache = tmp_path / "cache"
    config = tmp_path / "config.yaml"
    aws_creds = tmp_path / "aws_credentials"
    read_only_data = tmp_path / "read_only_data"
    config.write_text("config")
    aws_creds.write_text("creds")

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("HOST_ONLY", "host-value")
    image = tmp_path / "images" / "cosmos-curator+1.0.0-slim.sqsh"

    with (
        patch(f"{MODULE_NAME}.LOCAL_COSMOS_CURATOR_CONFIG_FILE", config),
        patch(f"{MODULE_NAME}.LOCAL_AWS_CREDENTIALS_FILE", aws_creds),
        patch(f"{MODULE_NAME}.subprocess.call", return_value=0) as mock_call,
    ):
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "launch",
                "--container-image",
                str(image),
                "--curator-path",
                str(repo),
                "--workspace-path",
                str(workspace),
                "--cache-path",
                str(cache),
                "--environment",
                "EXTRA=value,HOST_ONLY",
                "--extra-mounts",
                f"{tmp_path / 'data'}:/data,{read_only_data}:/readonly:ro",
                "--",
                "pixi",
                "run",
                "--as-is",
                "python",
                "-m",
                "cosmos_curator.pipelines.examples.hello_world_pipeline",
            ],
        )

    assert result.exit_code == 0
    srun_cmd = mock_call.call_args[0][0]
    subprocess_env = mock_call.call_args.kwargs["env"]
    assert srun_cmd[:3] == ["srun", "--mpi=none", "--overlap"]
    assert "--pty" not in srun_cmd
    assert "--container-image" in srun_cmd
    assert srun_cmd[srun_cmd.index("--container-image") + 1] == str(image)
    assert "enroot" not in srun_cmd

    mount_values = _container_mounts(srun_cmd)
    assert f"{workspace.resolve()}:/config" in mount_values
    assert f"{cache.resolve()}:/cache" in mount_values
    assert f"{repo}:/src/cosmos-curator" in mount_values
    assert not any("/opt/cosmos-curator/cosmos_curator" in mount for mount in mount_values)
    assert not any("/opt/cosmos-curator/pixi.toml" in mount for mount in mount_values)
    assert not any("/opt/cosmos-curator/pixi.lock" in mount for mount in mount_values)
    assert f"{config}:/cosmos_curator/config/cosmos_curator.yaml:ro" in mount_values
    assert f"{aws_creds}:/creds/s3_creds:ro" in mount_values
    assert f"{tmp_path / 'data'}:/data" in mount_values
    assert f"{read_only_data}:/readonly:ro" in mount_values
    assert not any(".pixi" in mount for mount in mount_values)

    env_keys = _container_env_keys(srun_cmd)
    assert "COSMOS_CURATOR_RAY_SLURM_JOB" in env_keys
    assert "PIXI_CACHE_DIR" in env_keys
    assert "RATTLER_CACHE_DIR" in env_keys
    assert "UV_CACHE_DIR" in env_keys
    assert "TORCH_HOME" in env_keys
    assert "TRITON_HOME" in env_keys
    assert "CONDA_OVERRIDE_CUDA" in env_keys
    assert "SLURM_JOB_ID" in env_keys
    assert "EXTRA" in env_keys
    assert "HOST_ONLY" in env_keys
    assert subprocess_env["COSMOS_CURATOR_RAY_SLURM_JOB"] == "True"
    assert subprocess_env["PIXI_CACHE_DIR"] == "/cache/rattler/cache"
    assert subprocess_env["RATTLER_CACHE_DIR"] == "/cache/rattler/cache"
    assert subprocess_env["UV_CACHE_DIR"] == "/cache/rattler/cache/uv-cache"
    assert subprocess_env["TORCH_HOME"] == "/cache/torch"
    assert subprocess_env["TRITON_HOME"] == "/cache/triton"
    assert subprocess_env["CONDA_OVERRIDE_CUDA"] == "13.0.2"
    assert subprocess_env["EXTRA"] == "value"
    assert subprocess_env["HOST_ONLY"] == "host-value"

    container_command = srun_cmd[srun_cmd.index("bash") + 2]
    assert "cd /opt/cosmos-curator" in container_command
    assert container_command.index("/src/cosmos-curator") < container_command.index("pixi install --frozen")
    assert "ln -s /src/cosmos-curator/pixi.toml /opt/cosmos-curator/pixi.toml" in container_command
    assert "ln -s /src/cosmos-curator/pixi.lock /opt/cosmos-curator/pixi.lock" in container_command
    assert "pixi install --frozen" in container_command
    assert 'exec "$@"' in container_command
    assert srun_cmd[-6:] == [
        "pixi",
        "run",
        "--as-is",
        "python",
        "-m",
        "cosmos_curator.pipelines.examples.hello_world_pipeline",
    ]


def test_slurm_launch_interactive_uses_pty(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Interactive shell launches should request a pseudo-terminal from srun."""
    repo = _create_repo(tmp_path)
    workspace = tmp_path / "workspace"
    cache = tmp_path / "cache"

    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    image = tmp_path / "container_images" / "cosmos-curator+1.0.0.sqsh"

    with patch(f"{MODULE_NAME}.subprocess.call", return_value=0) as mock_call:
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "launch",
                "--interactive",
                "--container-image",
                str(image),
                "--curator-path",
                str(repo),
                "--workspace-path",
                str(workspace),
                "--cache-path",
                str(cache),
                "--no-mount-s3-creds",
                "--",
                "bash",
            ],
        )

    assert result.exit_code == 0
    srun_cmd = mock_call.call_args[0][0]
    assert srun_cmd[:4] == ["srun", "--mpi=none", "--overlap", "--pty"]
    container_command = srun_cmd[srun_cmd.index("bash") + 2]
    assert "pixi install --frozen" in container_command
    assert 'exec "$@"' in container_command
    assert srun_cmd[-2:] == ["_", "bash"]


def test_slurm_launch_pixi_envs_overrides_slim_warmup_envs(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Users can limit slim-image Pixi warmup to the environments needed for an interactive session."""
    repo = _create_repo(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    with patch(f"{MODULE_NAME}.subprocess.call", return_value=0) as mock_call:
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "launch",
                "--container-image",
                "cosmos-curator+1.0.0-slim",
                "--curator-path",
                str(repo),
                "--workspace-path",
                str(tmp_path / "workspace"),
                "--cache-path",
                str(tmp_path / "cache"),
                "--no-mount-s3-creds",
                "--pixi-envs",
                "model-download,default,unified",
                "--",
                "bash",
            ],
        )

    assert result.exit_code == 0
    srun_cmd = mock_call.call_args[0][0]
    subprocess_env = mock_call.call_args.kwargs["env"]
    env_keys = _container_env_keys(srun_cmd)
    assert "COSMOS_CURATOR_SLIM_ENVS" in env_keys
    assert subprocess_env["COSMOS_CURATOR_SLIM_ENVS"] == "model-download,default,unified"

    container_command = srun_cmd[srun_cmd.index("bash") + 2]
    assert "pixi install --frozen -e ${COSMOS_CURATOR_SLIM_ENVS//,/ -e }" in container_command


def test_slurm_launch_pixi_envs_rejects_empty_value(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """An empty --pixi-envs value should not become an accidental no-warmup mode."""
    repo = _create_repo(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    with patch(f"{MODULE_NAME}.subprocess.call") as mock_call:
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "launch",
                "--container-image",
                "cosmos-curator+1.0.0-slim",
                "--curator-path",
                str(repo),
                "--workspace-path",
                str(tmp_path / "workspace"),
                "--cache-path",
                str(tmp_path / "cache"),
                "--no-mount-s3-creds",
                "--pixi-envs",
                "",
                "--",
                "bash",
            ],
        )

    assert result.exit_code == 2
    assert "--pixi-envs must include at least one Pixi environment" in result.output
    mock_call.assert_not_called()


def test_slurm_launch_requires_slurm_allocation_by_default(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """The command should not accidentally run on a login node."""
    repo = _create_repo(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)

    with patch(f"{MODULE_NAME}.subprocess.call") as mock_call:
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "launch",
                "--container-image",
                "cosmos-curator+1.0.0-slim",
                "--curator-path",
                str(repo),
                "--workspace-path",
                str(tmp_path / "workspace"),
                "--cache-path",
                str(tmp_path / "cache"),
                "--",
                "echo",
                "hello",
            ],
        )

    assert result.exit_code == 1
    mock_call.assert_not_called()


def test_slurm_launch_model_cli_requires_config_for_module_path(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """model_cli config validation should work when invoked by module path."""
    repo = _create_repo(tmp_path)
    missing_config = tmp_path / "missing_config.yaml"
    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    with (
        patch(f"{MODULE_NAME}.LOCAL_COSMOS_CURATOR_CONFIG_FILE", missing_config),
        patch(f"{MODULE_NAME}.subprocess.call") as mock_call,
    ):
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "launch",
                "--container-image",
                "cosmos-curator+1.0.0-slim",
                "--curator-path",
                str(repo),
                "--workspace-path",
                str(tmp_path / "workspace"),
                "--cache-path",
                str(tmp_path / "cache"),
                "--no-mount-s3-creds",
                "--",
                "pixi",
                "run",
                "--as-is",
                "python",
                "-m",
                "cosmos_curator.core.managers.model_cli",
                "download",
            ],
        )

    assert result.exit_code == 1
    mock_call.assert_not_called()


def test_slurm_launch_allows_explicit_non_slurm_override(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Developers can opt out of the allocation guard for local command construction tests."""
    repo = _create_repo(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)

    with patch(f"{MODULE_NAME}.subprocess.call", return_value=0) as mock_call:
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "launch",
                "--no-require-slurm-allocation",
                "--container-image",
                "cosmos-curator+1.0.0-slim",
                "--curator-path",
                str(repo),
                "--workspace-path",
                str(tmp_path / "workspace"),
                "--cache-path",
                str(tmp_path / "cache"),
                "--no-mount-s3-creds",
                "--",
                "echo",
                "hello",
            ],
        )

    assert result.exit_code == 0
    mock_call.assert_called_once()
    srun_cmd = mock_call.call_args[0][0]
    assert "cosmos-curator+1.0.0-slim" in srun_cmd


def test_slurm_launch_defaults_container_image(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """The interactive shortcut should have a useful conventional image path."""
    repo = _create_repo(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    with patch(f"{MODULE_NAME}.subprocess.call", return_value=0) as mock_call:
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "launch",
                "--curator-path",
                str(repo),
                "--workspace-path",
                str(tmp_path / "workspace"),
                "--cache-path",
                str(tmp_path / "cache"),
                "--no-mount-s3-creds",
                "--",
                "echo",
                "hello",
            ],
        )

    assert result.exit_code == 0
    srun_cmd = mock_call.call_args[0][0]
    assert str(Path("~/container_images/cosmos-curator+1.0.0.sqsh").expanduser()) in srun_cmd


def test_slurm_launch_nonzero_srun_exits(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """A non-zero srun exit should surface as a CLI failure."""
    repo = _create_repo(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    with patch(f"{MODULE_NAME}.subprocess.call", return_value=2) as mock_call:
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "launch",
                "--container-image",
                "cosmos-curator+1.0.0-slim",
                "--curator-path",
                str(repo),
                "--workspace-path",
                str(tmp_path / "workspace"),
                "--cache-path",
                str(tmp_path / "cache"),
                "--no-mount-s3-creds",
                "--",
                "echo",
                "hello",
            ],
        )

    assert result.exit_code == 1
    mock_call.assert_called_once()
