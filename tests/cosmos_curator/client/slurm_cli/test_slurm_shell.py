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

import shlex
from pathlib import Path
from unittest.mock import patch

import pytest
from _pytest.monkeypatch import MonkeyPatch
from typer.testing import CliRunner

from cosmos_curator.client.cli import cosmos_curator
from cosmos_curator.client.slurm_cli.slurm_common import _SLURM_ACCOUNT_ENV_VAR
from cosmos_curator.core.utils import environment

SLURM_MODULE_NAME = "cosmos_curator.client.slurm_cli.slurm_shell"
SLURM_COMMON_MODULE_NAME = "cosmos_curator.client.slurm_cli.slurm_common"
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


def test_slurm_shell_bare_command_prints_help() -> None:
    """The bare shell command should describe required options instead of trying to run srun."""
    with patch(f"{SLURM_MODULE_NAME}.subprocess.call") as mock_call:
        result = runner.invoke(cosmos_curator, ["slurm", "shell"])

    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Start an interactive shell or command inside a Slurm/Pyxis allocation." in result.output
    mock_call.assert_not_called()


def test_slurm_shell_command_uses_srun_and_live_source_mounts(  # noqa: PLR0915
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Verify slurm shell forms an allocating srun/Pyxis command without mounting host .pixi."""
    repo = _create_repo(tmp_path)
    workspace = tmp_path / "workspace"
    cache = tmp_path / "cache"
    config = tmp_path / "config.yaml"
    aws_creds = tmp_path / "aws_credentials"
    read_only_data = tmp_path / "read_only_data"
    config.write_text("config")
    aws_creds.write_text("creds")

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    monkeypatch.setenv("HOST_ONLY", "host-value")
    monkeypatch.setenv("PIXI_PROJECT_MANIFEST", str(repo / "pixi.toml"))
    monkeypatch.setenv("PIXI_PROJECT_ROOT", str(repo))
    monkeypatch.setenv("PIXI_ENVIRONMENT_NAME", "cluster")
    monkeypatch.setenv("PIXI_IN_SHELL", "1")
    monkeypatch.setenv("CONDA_PREFIX", str(repo / ".pixi" / "envs" / "cluster"))
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "cosmos-curator:cluster")
    monkeypatch.setenv("CONDA_ENV_SHLVL_2_PIXI_PROJECT_MANIFEST", str(repo / "pixi.toml"))
    image = tmp_path / "images" / "cosmos-curator+1.0.0-slim.sqsh"

    with (
        patch(f"{SLURM_COMMON_MODULE_NAME}.LOCAL_COSMOS_CURATOR_CONFIG_FILE", config),
        patch(f"{SLURM_COMMON_MODULE_NAME}.LOCAL_AWS_CREDENTIALS_FILE", aws_creds),
        patch(f"{SLURM_MODULE_NAME}.subprocess.call", return_value=0) as mock_call,
    ):
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "shell",
                "--account",
                "test_account",
                "--partition",
                "test_partition",
                "--qos",
                "normal",
                "--gres",
                "gpu:1",
                "--time",
                "01:00:00",
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
    assert srun_cmd[:2] == ["srun", "--pty"]
    assert "--propagate=NONE" in srun_cmd
    assert "--mpi=none" not in srun_cmd
    assert "--overlap" not in srun_cmd
    assert not any(arg.startswith("--nodes=") for arg in srun_cmd)
    assert not any(arg.startswith("--ntasks=") for arg in srun_cmd)
    assert "--account=test_account" in srun_cmd
    assert "--partition=test_partition" in srun_cmd
    assert "--qos=normal" in srun_cmd
    assert "--gres=gpu:1" in srun_cmd
    assert "--time=01:00:00" in srun_cmd
    assert "--job-name=cosmos_curator_shell" in srun_cmd
    assert "--exclusive" in srun_cmd
    assert "--container-image" in srun_cmd
    assert srun_cmd[srun_cmd.index("--container-image") + 1] == str(image)
    assert "enroot" not in srun_cmd

    mount_values = _container_mounts(srun_cmd)
    assert f"{workspace.resolve()}:/config:rw" in mount_values
    assert f"{cache.resolve()}:/cache:rw" in mount_values
    assert f"{repo.resolve()}:/src/cosmos-curator:rw" in mount_values
    assert not any("/opt/cosmos-curator/cosmos_curator" in mount for mount in mount_values)
    assert not any("/opt/cosmos-curator/pixi.toml" in mount for mount in mount_values)
    assert not any("/opt/cosmos-curator/pixi.lock" in mount for mount in mount_values)
    assert f"{config}:/cosmos_curator/config/cosmos_curator.yaml:ro" in mount_values
    assert f"{aws_creds}:/creds/s3_creds:ro" in mount_values
    assert f"{tmp_path / 'data'}:/data:rw" in mount_values
    assert f"{read_only_data}:/readonly:ro" in mount_values
    assert not any(".pixi" in mount for mount in mount_values)

    env_keys = _container_env_keys(srun_cmd)
    assert "COSMOS_CURATOR_RAY_SLURM_JOB" in env_keys
    assert "PYTHONPATH" in env_keys
    assert "PIXI_CACHE_DIR" in env_keys
    assert "RATTLER_CACHE_DIR" in env_keys
    assert "UV_CACHE_DIR" in env_keys
    assert "TORCH_HOME" in env_keys
    assert "TRITON_HOME" in env_keys
    assert "CONDA_OVERRIDE_CUDA" in env_keys
    assert "SLURM_JOB_ID" in env_keys
    assert "SLURM_PROCID" not in env_keys
    assert "PIXI_PROJECT_MANIFEST" not in env_keys
    assert "PIXI_ENVIRONMENT_NAME" not in env_keys
    assert "CONDA_PREFIX" not in env_keys
    assert "EXTRA" in env_keys
    assert "HOST_ONLY" in env_keys
    assert subprocess_env["COSMOS_CURATOR_RAY_SLURM_JOB"] == "True"
    assert subprocess_env["PYTHONPATH"] == "/opt/cosmos-curator"
    assert subprocess_env["PIXI_CACHE_DIR"] == "/cache/rattler/cache"
    assert subprocess_env["RATTLER_CACHE_DIR"] == "/cache/rattler/cache"
    assert subprocess_env["UV_CACHE_DIR"] == "/cache/rattler/cache/uv-cache"
    assert subprocess_env["TORCH_HOME"] == "/cache/torch"
    assert subprocess_env["TRITON_HOME"] == "/cache/triton"
    assert subprocess_env["CONDA_OVERRIDE_CUDA"] == "13.0.3"
    assert subprocess_env["EXTRA"] == "value"
    assert subprocess_env["HOST_ONLY"] == "host-value"
    assert "PIXI_PROJECT_MANIFEST" not in subprocess_env
    assert "PIXI_PROJECT_ROOT" not in subprocess_env
    assert "PIXI_ENVIRONMENT_NAME" not in subprocess_env
    assert "PIXI_IN_SHELL" not in subprocess_env
    assert "CONDA_PREFIX" not in subprocess_env
    assert "CONDA_DEFAULT_ENV" not in subprocess_env
    assert "CONDA_ENV_SHLVL_2_PIXI_PROJECT_MANIFEST" not in subprocess_env

    container_command = srun_cmd[srun_cmd.index("bash") + 2]
    assert "cd /opt/cosmos-curator" in container_command
    assert container_command.index("mkdir -p /opt/cosmos-curator") < container_command.index("cd /opt/cosmos-curator")
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


def test_slurm_shell_defaults_to_bash_and_uses_pty(tmp_path: Path) -> None:
    """Interactive shell launches should default to bash and request a pseudo-terminal from srun."""
    repo = _create_repo(tmp_path)
    workspace = tmp_path / "workspace"
    cache = tmp_path / "cache"

    image = tmp_path / "container_images" / "cosmos-curator+1.0.0.sqsh"

    with patch(f"{SLURM_MODULE_NAME}.subprocess.call", return_value=0) as mock_call:
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "shell",
                "--container-image",
                str(image),
                "--curator-path",
                str(repo),
                "--workspace-path",
                str(workspace),
                "--cache-path",
                str(cache),
                "--no-mount-s3-creds",
            ],
        )

    assert result.exit_code == 0
    srun_cmd = mock_call.call_args[0][0]
    assert srun_cmd[:2] == ["srun", "--pty"]
    assert "--mpi=none" not in srun_cmd
    assert not any(arg.startswith("--nodes=") for arg in srun_cmd)
    assert not any(arg.startswith("--ntasks=") for arg in srun_cmd)
    container_command = srun_cmd[srun_cmd.index("bash") + 2]
    assert "pixi install --frozen" in container_command
    assert 'exec "$@"' in container_command
    assert srun_cmd[-1:] == ["bash"]


@pytest.mark.parametrize(
    ("gpu_option", "gpu_value", "expected_gpu_args"),
    [
        ("--gpus", "8", ["--gpus=8"]),
        ("--gpus-per-node", "4", ["--gpus-per-node=4"]),
        ("--gpus-per-node", " h100:4 ", ["--gpus-per-node=h100:4"]),
        ("--gpus-per-node", "   ", []),
    ],
)
def test_slurm_shell_accepts_gpu_requests(
    tmp_path: Path, gpu_option: str, gpu_value: str, expected_gpu_args: list[str]
) -> None:
    """GPU requests retain their job or per-node scope when allocating multiple nodes."""
    repo = _create_repo(tmp_path)

    with patch(f"{SLURM_MODULE_NAME}.subprocess.call", return_value=0) as mock_call:
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "shell",
                "--curator-path",
                str(repo),
                "--workspace-path",
                str(tmp_path / "workspace"),
                "--cache-path",
                str(tmp_path / "cache"),
                "--no-mount-s3-creds",
                "-N",
                "2",
                gpu_option,
                gpu_value,
            ],
        )

    assert result.exit_code == 0
    srun_cmd = mock_call.call_args[0][0]
    assert "--nodes=2" in srun_cmd
    gpu_args = [arg for arg in srun_cmd if arg.startswith(("--gpus=", "--gpus-per-node=", "--gres="))]
    assert gpu_args == expected_gpu_args


def test_slurm_shell_accepts_srun_style_short_options(tmp_path: Path) -> None:
    """Interactive shell users can use familiar srun-style allocation aliases."""
    repo = _create_repo(tmp_path)

    with patch(f"{SLURM_MODULE_NAME}.subprocess.call", return_value=0) as mock_call:
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "shell",
                "-A",
                "test_account",
                "-p",
                "interactive",
                "-q",
                "normal",
                "-G",
                "8",
                "-J",
                "interactive_job",
                "-N",
                "2",
                "-t",
                "01:00:00",
                "--curator-path",
                str(repo),
                "--workspace-path",
                str(tmp_path / "workspace"),
                "--cache-path",
                str(tmp_path / "cache"),
                "--no-mount-s3-creds",
            ],
        )

    assert result.exit_code == 0
    srun_cmd = mock_call.call_args[0][0]
    assert "--account=test_account" in srun_cmd
    assert "--partition=interactive" in srun_cmd
    assert "--qos=normal" in srun_cmd
    assert "--gpus=8" in srun_cmd
    assert not any(arg.startswith("--gpus-per-node=") for arg in srun_cmd)
    assert "--job-name=interactive_job" in srun_cmd
    assert "--nodes=2" in srun_cmd
    assert "--time=01:00:00" in srun_cmd


@pytest.mark.parametrize(
    ("gpu_args", "expected_message"),
    [
        (["--gres", "gpu:8", "--gpus", "8"], "--gres and --gpus cannot be used together"),
        (["--gres", "gpu:8", "--gpus-per-node", "8"], "--gres and --gpus-per-node cannot be used together"),
        (["-G", "8", "--gpus-per-node", "8"], "--gpus and --gpus-per-node cannot be used together"),
    ],
)
def test_slurm_shell_rejects_conflicting_gpu_options(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], gpu_args: list[str], expected_message: str
) -> None:
    """Conflicting GPU request forms fail before invoking srun."""
    repo = _create_repo(tmp_path)

    with patch(f"{SLURM_MODULE_NAME}.subprocess.call", return_value=0) as mock_call:
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "shell",
                "--curator-path",
                str(repo),
                "--workspace-path",
                str(tmp_path / "workspace"),
                "--cache-path",
                str(tmp_path / "cache"),
                "--no-mount-s3-creds",
                *gpu_args,
            ],
        )

    assert result.exit_code == 2
    assert expected_message in result.output
    mock_call.assert_not_called()
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_slurm_shell_pixi_envs_overrides_slim_warmup_envs(tmp_path: Path) -> None:
    """Users can limit slim-image Pixi warmup to the environments needed for an interactive session."""
    repo = _create_repo(tmp_path)

    with patch(f"{SLURM_MODULE_NAME}.subprocess.call", return_value=0) as mock_call:
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "shell",
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
                "model-download,default,seedvr",
                "--",
                "bash",
            ],
        )

    assert result.exit_code == 0
    srun_cmd = mock_call.call_args[0][0]
    subprocess_env = mock_call.call_args.kwargs["env"]
    env_keys = _container_env_keys(srun_cmd)
    assert "COSMOS_CURATOR_SLIM_ENVS" in env_keys
    assert subprocess_env["COSMOS_CURATOR_SLIM_ENVS"] == "model-download,default,seedvr"

    container_command = srun_cmd[srun_cmd.index("bash") + 2]
    assert "pixi install --frozen -e ${COSMOS_CURATOR_SLIM_ENVS//,/ -e }" in container_command


def test_slurm_shell_pixi_envs_rejects_empty_value(tmp_path: Path) -> None:
    """An empty --pixi-envs value should not become an accidental no-warmup mode."""
    repo = _create_repo(tmp_path)

    with patch(f"{SLURM_MODULE_NAME}.subprocess.call") as mock_call:
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "shell",
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


def test_slurm_shell_runs_without_existing_slurm_allocation(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """The shell command should allocate directly instead of requiring an existing Slurm job."""
    repo = _create_repo(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)

    with patch(f"{SLURM_MODULE_NAME}.subprocess.call", return_value=0) as mock_call:
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "shell",
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

    assert result.exit_code == 0
    mock_call.assert_called_once()
    assert "--pty" in mock_call.call_args.args[0]


def test_slurm_shell_model_cli_requires_config_for_module_path(tmp_path: Path) -> None:
    """model_cli config validation should work when invoked by module path."""
    repo = _create_repo(tmp_path)
    missing_config = tmp_path / "missing_config.yaml"

    with (
        patch(f"{SLURM_COMMON_MODULE_NAME}.LOCAL_COSMOS_CURATOR_CONFIG_FILE", missing_config),
        patch(f"{SLURM_MODULE_NAME}.subprocess.call") as mock_call,
    ):
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "shell",
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


def test_slurm_shell_remote_login_node_runs_srun_through_ssh() -> None:
    """A remote login node should use ssh directly instead of Python PTY streaming."""
    repo = Path("/cluster/repo")
    workspace = Path("/cluster/workspace")
    cache = Path("/cluster/cache")

    with (
        patch(f"{SLURM_MODULE_NAME}._is_local_host", return_value=False),
        patch(f"{SLURM_MODULE_NAME}.subprocess.call", return_value=0) as mock_call,
    ):
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "shell",
                "--login-node",
                "remote-login",
                "--username",
                "cluster-user",
                "--container-image",
                "cosmos-curator+1.0.0-slim",
                "--curator-path",
                str(repo),
                "--workspace-path",
                str(workspace),
                "--cache-path",
                str(cache),
                "--no-mount-s3-creds",
                "--",
                "echo",
                "hello",
            ],
        )

    assert result.exit_code == 0
    mock_call.assert_called_once()
    ssh_command = mock_call.call_args.args[0]
    remote_command = ssh_command[3]
    assert ssh_command[:3] == ["ssh", "-t", "cluster-user@remote-login"]
    assert "env COSMOS_CURATOR_RAY_SLURM_JOB=True" in remote_command
    assert "srun --pty" in remote_command
    assert "--propagate=NONE" in shlex.split(remote_command)
    assert "--mpi=none" not in remote_command
    assert "--container-image cosmos-curator+1.0.0-slim" in remote_command
    assert f"{repo}:/src/cosmos-curator:rw" in remote_command
    assert f"{workspace}:/config:rw" in remote_command
    assert f"{cache}:/cache:rw" in remote_command
    assert "echo hello" in remote_command


def test_slurm_shell_remote_login_node_no_pty_omits_ssh_and_srun_pty() -> None:
    """--no-pty should make remote shell execution usable from non-interactive SSH contexts."""
    repo = Path("/cluster/repo")
    workspace = Path("/cluster/workspace")
    cache = Path("/cluster/cache")

    with (
        patch(f"{SLURM_MODULE_NAME}._is_local_host", return_value=False),
        patch(f"{SLURM_MODULE_NAME}.subprocess.call", return_value=0) as mock_call,
    ):
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "shell",
                "--login-node",
                "remote-login",
                "--username",
                "cluster-user",
                "--no-pty",
                "--container-image",
                "cosmos-curator+1.0.0-slim",
                "--curator-path",
                str(repo),
                "--workspace-path",
                str(workspace),
                "--cache-path",
                str(cache),
                "--no-mount-s3-creds",
                "--",
                "echo",
                "hello",
            ],
        )

    assert result.exit_code == 0
    mock_call.assert_called_once()
    ssh_command = mock_call.call_args.args[0]
    remote_command = ssh_command[2]
    assert ssh_command[:2] == ["ssh", "cluster-user@remote-login"]
    assert "-t" not in ssh_command
    remote_tokens = shlex.split(remote_command)
    assert "srun" in remote_tokens
    assert "--pty" not in remote_tokens
    assert "--container-writable" in remote_tokens


def test_slurm_shell_dry_run_prints_remote_command_without_executing() -> None:
    """Dry-run should print the same remote ssh/srun command the shell would execute."""
    repo = Path("/cluster/repo")
    workspace = Path("/cluster/workspace")
    cache = Path("/cluster/cache")

    with (
        patch(f"{SLURM_MODULE_NAME}._is_local_host", return_value=False),
        patch(f"{SLURM_MODULE_NAME}.subprocess.call") as mock_call,
    ):
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "shell",
                "--login-node",
                "remote-login",
                "--username",
                "cluster-user",
                "--dry-run",
                "--container-image",
                "cosmos-curator+1.0.0-slim",
                "--curator-path",
                str(repo),
                "--workspace-path",
                str(workspace),
                "--cache-path",
                str(cache),
                "--no-mount-s3-creds",
                "--",
                "echo",
                "hello",
            ],
        )

    assert result.exit_code == 0
    mock_call.assert_not_called()
    assert result.output.startswith("ssh -t cluster-user@remote-login ")
    assert "env COSMOS_CURATOR_RAY_SLURM_JOB=True" in result.output
    assert "srun --pty" in result.output
    assert "--container-image cosmos-curator+1.0.0-slim" in result.output
    assert "echo hello" in result.output


def test_slurm_shell_remote_login_node_allows_explicit_default_like_paths() -> None:
    """Remote shell launches should reject omitted paths, not explicit paths that look like local defaults."""
    repo = Path("/cluster/repo")
    workspace = environment.LOCAL_WORKSPACE_PATH
    cache = Path("~/.cache").expanduser()

    with (
        patch(f"{SLURM_MODULE_NAME}._is_local_host", return_value=False),
        patch(f"{SLURM_MODULE_NAME}.subprocess.call", return_value=0) as mock_call,
    ):
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "shell",
                "--login-node",
                "remote-login",
                "--username",
                "cluster-user",
                "--container-image",
                "cosmos-curator+1.0.0-slim",
                "--curator-path",
                str(repo),
                "--workspace-path",
                str(workspace),
                "--cache-path",
                str(cache),
                "--no-mount-s3-creds",
            ],
        )

    assert result.exit_code == 0
    remote_command = mock_call.call_args.args[0][3]
    assert f"{workspace}:/config:rw" in remote_command
    assert f"{cache}:/cache:rw" in remote_command


def test_slurm_shell_remote_login_node_mounts_credentials_and_model_config() -> None:
    """Remote shell launches should preserve credential and model config mount behavior."""
    repo = Path("/cluster/repo")
    workspace = Path("/cluster/workspace")
    cache = Path("/cluster/cache")

    with (
        patch(f"{SLURM_MODULE_NAME}._is_local_host", return_value=False),
        patch(f"{SLURM_MODULE_NAME}.subprocess.call", return_value=0) as mock_call,
    ):
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "shell",
                "--login-node",
                "remote-login",
                "--username",
                "cluster-user",
                "--container-image",
                "cosmos-curator+1.0.0-slim",
                "--curator-path",
                str(repo),
                "--workspace-path",
                str(workspace),
                "--cache-path",
                str(cache),
                "--mount-azure-creds",
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

    assert result.exit_code == 0
    remote_command = mock_call.call_args.args[0][3]
    assert f"{environment.LOCAL_AWS_CREDENTIALS_FILE}:/creds/s3_creds:ro" in remote_command
    assert f"{environment.LOCAL_AZURE_CREDENTIALS_FILE}:/creds/azure_creds:ro" in remote_command
    assert (
        f"{environment.LOCAL_COSMOS_CURATOR_CONFIG_FILE}:/cosmos_curator/config/cosmos_curator.yaml:ro"
        in remote_command
    )


def test_slurm_shell_remote_login_node_defaults_workspace_and_cache() -> None:
    """Remote shell launches should keep conventional workspace and cache defaults when omitted."""
    repo = Path("/cluster/repo")

    with (
        patch(f"{SLURM_MODULE_NAME}._is_local_host", return_value=False),
        patch(f"{SLURM_MODULE_NAME}.subprocess.call", return_value=0) as mock_call,
    ):
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "shell",
                "--login-node",
                "remote-login",
                "--username",
                "cluster-user",
                "--curator-path",
                str(repo),
                "--no-mount-s3-creds",
            ],
        )

    assert result.exit_code == 0
    remote_command = mock_call.call_args.args[0][3]
    assert f"{repo}:/src/cosmos-curator:rw" in remote_command
    assert f"{environment.LOCAL_WORKSPACE_PATH}:/config:rw" in remote_command
    assert f"{Path('~/.cache').expanduser()}:/cache:rw" in remote_command


def test_slurm_shell_remote_login_node_requires_explicit_curator_path() -> None:
    """Remote shell launches should not infer the local checkout path for the remote login node."""
    with (
        patch(f"{SLURM_MODULE_NAME}._is_local_host", return_value=False),
        patch(f"{SLURM_MODULE_NAME}.subprocess.call", return_value=0) as mock_call,
    ):
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "shell",
                "--login-node",
                "remote-login",
                "--username",
                "cluster-user",
                "--no-mount-s3-creds",
            ],
        )

    assert result.exit_code != 0
    assert "--curator-path must point to a path on the remote login node" in result.output
    mock_call.assert_not_called()


def test_slurm_shell_defaults_container_image(tmp_path: Path) -> None:
    """The interactive shortcut should have a useful conventional image path."""
    repo = _create_repo(tmp_path)

    with patch(f"{SLURM_MODULE_NAME}.subprocess.call", return_value=0) as mock_call:
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "shell",
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


def test_slurm_shell_nonzero_srun_exits(tmp_path: Path) -> None:
    """A non-zero srun exit should surface as a CLI failure."""
    repo = _create_repo(tmp_path)

    with patch(f"{SLURM_MODULE_NAME}.subprocess.call", return_value=2) as mock_call:
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "shell",
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


def test_slurm_shell_missing_local_srun_exits_cleanly(tmp_path: Path) -> None:
    """A missing local srun executable should surface as a CLI failure instead of a traceback."""
    repo = _create_repo(tmp_path)

    with patch(f"{SLURM_MODULE_NAME}.subprocess.call", side_effect=FileNotFoundError("srun")) as mock_call:
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "shell",
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


def test_import_image_uses_enroot_defaults(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Import an image through srun with Enroot paths placed under the image directory."""
    output_dir = tmp_path / "container_images"
    monkeypatch.setenv(_SLURM_ACCOUNT_ENV_VAR, "env_account")

    with patch(f"{SLURM_MODULE_NAME}.subprocess.call", return_value=0) as mock_call:
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "import-image",
                "nvcr.io/nvidia/cosmos-curator:1.0.0",
                "--output-dir",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0, result.output
    assert "Imported docker://nvcr.io/nvidia/cosmos-curator:1.0.0" in result.output
    srun_cmd = mock_call.call_args.args[0]
    subprocess_env = mock_call.call_args.kwargs["env"]

    assert srun_cmd[0] == "srun"
    assert "--nodes=1" in srun_cmd
    assert "--ntasks=1" in srun_cmd
    assert "--propagate=NONE" in srun_cmd
    assert "--export=HOME,PATH,ENROOT_CACHE_PATH,ENROOT_DATA_PATH,ENROOT_TEMP_PATH" in srun_cmd
    assert "--export=ALL" not in srun_cmd
    assert "--account=env_account" in srun_cmd
    assert "--partition=cpu" in srun_cmd
    assert "--job-name=cosmos_curator_import_image" in srun_cmd
    assert "--cpus-per-task=16" in srun_cmd
    assert "--mem=64G" in srun_cmd
    assert str(output_dir / "cosmos-curator+1.0.0.sqsh") in srun_cmd
    assert "docker://nvcr.io/nvidia/cosmos-curator:1.0.0" in srun_cmd
    assert any('mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH"' in arg for arg in srun_cmd)
    assert any('rm -f -- "$1"' in arg for arg in srun_cmd)
    assert any('exec enroot import --output "$1" "$2"' in arg for arg in srun_cmd)

    assert subprocess_env["ENROOT_CACHE_PATH"] == str(output_dir / ".enroot-cache")
    assert subprocess_env["ENROOT_DATA_PATH"] == str(output_dir)
    assert subprocess_env["ENROOT_TEMP_PATH"] == str(Path("/") / "tmp")
    assert "HOME" in subprocess_env


def test_import_image_accepts_custom_output_and_no_overwrite(tmp_path: Path) -> None:
    """Allow callers to preserve existing outputs and pass explicit Enroot URI schemes."""
    output_dir = tmp_path / "images"

    with patch(f"{SLURM_MODULE_NAME}.subprocess.call", return_value=0) as mock_call:
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "import-image",
                "dockerd://cosmos-curator:hello-world",
                "--output-dir",
                str(output_dir),
                "--output-filename",
                "cosmos-curator_hello-world.sqsh",
                "--no-overwrite",
                "-c",
                "8",
                "--mem",
                "32G",
            ],
        )

    assert result.exit_code == 0, result.output
    srun_cmd = mock_call.call_args.args[0]
    assert "Imported dockerd://cosmos-curator:hello-world" in result.output
    assert str(output_dir / "cosmos-curator_hello-world.sqsh") in result.output
    assert "--cpus-per-task=8" in srun_cmd
    assert "--mem=32G" in srun_cmd
    assert not any('rm -f -- "$1"' in arg for arg in srun_cmd)


@pytest.mark.parametrize(
    ("options", "error_message"),
    [
        (["--cpus-per-task", "0"], "--cpus-per-task must be at least 1"),
        (["--mem", "0"], "--mem must be a positive Slurm size"),
        (["--mem", "64 GB"], "--mem must be a positive Slurm size"),
    ],
)
def test_import_image_rejects_invalid_resources(options: list[str], error_message: str) -> None:
    """Reject resource requests that Slurm cannot interpret safely."""
    with patch(f"{SLURM_MODULE_NAME}.subprocess.call") as mock_call:
        result = runner.invoke(
            cosmos_curator,
            ["slurm", "import-image", "cosmos-curator:1.0.0", *options],
        )

    assert result.exit_code != 0
    assert error_message in result.output
    mock_call.assert_not_called()


def test_import_image_remote_login_expands_default_output_dir_with_cluster_username() -> None:
    """Remote imports should not expand the default output directory against the local user's home."""
    with (
        patch(f"{SLURM_MODULE_NAME}._is_local_host", return_value=False),
        patch(f"{SLURM_MODULE_NAME}.subprocess.call", return_value=0) as mock_call,
    ):
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "import-image",
                "cosmos-curator:1.0.0",
                "--login-node",
                "login.example.com",
                "--username",
                "cluster_user",
            ],
        )

    assert result.exit_code == 0, result.output
    ssh_cmd = mock_call.call_args.args[0]
    assert ssh_cmd[:3] == ["ssh", "-t", "cluster_user@login.example.com"]
    remote_command = ssh_cmd[-1]
    assert "--propagate=NONE" in shlex.split(remote_command)
    assert "HOME=/home/cluster_user" in remote_command
    assert "ENROOT_CACHE_PATH=/home/cluster_user/container_images/.enroot-cache" in remote_command
    assert "ENROOT_DATA_PATH=/home/cluster_user/container_images" in remote_command
    assert "/home/cluster_user/container_images/cosmos-curator+1.0.0.sqsh" in remote_command


def test_import_image_rejects_output_filename_path() -> None:
    """Keep output directory and filename options unambiguous."""
    with patch(f"{SLURM_MODULE_NAME}.subprocess.call") as mock_call:
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "import-image",
                "cosmos-curator:1.0.0",
                "--output-filename",
                "nested/cosmos-curator.sqsh",
            ],
        )

    assert result.exit_code != 0
    assert "--output-filename must be a filename" in result.output
    mock_call.assert_not_called()


def test_import_image_retries_failed_srun(tmp_path: Path) -> None:
    """Transient Enroot or Slurm failures should retry like the previous helper script."""
    with (
        patch(f"{SLURM_MODULE_NAME}.subprocess.call", side_effect=[1, 0]) as mock_call,
        patch(f"{SLURM_MODULE_NAME}.time_module.sleep") as mock_sleep,
    ):
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "import-image",
                "cosmos-curator:1.0.0",
                "--output-dir",
                str(tmp_path / "images"),
                "--retries",
                "2",
                "--retry-delay",
                "0",
            ],
        )

    assert result.exit_code == 0, result.output
    assert mock_call.call_count == 2
    mock_sleep.assert_called_once_with(0)
