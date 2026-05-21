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
from cosmos_curator.core.utils import environment

SLURM_MODULE_NAME = "cosmos_curator.client.slurm_cli.slurm"
SLURM_LOCAL_MODULE_NAME = "cosmos_curator.client.slurm_cli.slurm_local"
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
    image = tmp_path / "images" / "cosmos-curator+1.0.0-slim.sqsh"

    with (
        patch(f"{SLURM_LOCAL_MODULE_NAME}.LOCAL_COSMOS_CURATOR_CONFIG_FILE", config),
        patch(f"{SLURM_LOCAL_MODULE_NAME}.LOCAL_AWS_CREDENTIALS_FILE", aws_creds),
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
    assert "PIXI_CACHE_DIR" in env_keys
    assert "RATTLER_CACHE_DIR" in env_keys
    assert "UV_CACHE_DIR" in env_keys
    assert "TORCH_HOME" in env_keys
    assert "TRITON_HOME" in env_keys
    assert "CONDA_OVERRIDE_CUDA" in env_keys
    assert "SLURM_JOB_ID" in env_keys
    assert "SLURM_PROCID" not in env_keys
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


def test_slurm_shell_accepts_gpus_request(tmp_path: Path) -> None:
    """The shell can request GPUs with Slurm's --gpus option."""
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
                "--gpus",
                "8",
            ],
        )

    assert result.exit_code == 0
    srun_cmd = mock_call.call_args[0][0]
    assert "--gpus=8" in srun_cmd
    assert not any(arg.startswith("--gres=") for arg in srun_cmd)


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
    assert "--job-name=interactive_job" in srun_cmd
    assert "--nodes=2" in srun_cmd
    assert "--time=01:00:00" in srun_cmd


def test_slurm_shell_rejects_gres_with_gpus(tmp_path: Path) -> None:
    """The shell should not ask Slurm for GPUs through two incompatible option styles."""
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
                "--gres",
                "gpu:8",
                "--gpus",
                "8",
            ],
        )

    assert result.exit_code != 0
    mock_call.assert_not_called()


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
        patch(f"{SLURM_LOCAL_MODULE_NAME}.LOCAL_COSMOS_CURATOR_CONFIG_FILE", missing_config),
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
    assert "--mpi=none" not in remote_command
    assert "--container-image cosmos-curator+1.0.0-slim" in remote_command
    assert f"{repo}:/src/cosmos-curator:rw" in remote_command
    assert f"{workspace}:/config:rw" in remote_command
    assert f"{cache}:/cache:rw" in remote_command
    assert "echo hello" in remote_command


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
