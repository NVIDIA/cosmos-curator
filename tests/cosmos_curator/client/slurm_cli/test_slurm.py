# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Test the slurm module."""

import pathlib
import unittest
from contextlib import AbstractContextManager, nullcontext
from typing import Any
from unittest.mock import Mock, patch

import invoke
import pytest
from typer.testing import CliRunner

from cosmos_curator.client.cli import cosmos_curator
from cosmos_curator.client.slurm_cli.slurm import (
    _SLURM_ACCOUNT_ENV_VAR,
    _START_RAY,
    ContainerSpec,
    MountSpec,
    SlurmJobSpec,
    _get_username,
    _parse_job_id,
    _render_sbatch_script,
    connect,
    curator_submit,
    submit_cli,
    upload_text,
)
from cosmos_curator.scripts.onto_slurm import SlurmEnv

MODULE_NAME = "cosmos_curator.client.slurm_cli.slurm"
GRES = "gpu:8"
runner = CliRunner()


def _create_repo(root: pathlib.Path) -> pathlib.Path:
    repo = root / "repo"
    (repo / "cosmos_curator" / "pipelines").mkdir(parents=True)
    (repo / "tests" / "cosmos_curator").mkdir(parents=True)
    (repo / "tools").mkdir()
    for filename in ("pixi.toml", "pixi.lock", "pyproject.toml", "pytest.ini", ".coveragerc"):
        (repo / filename).write_text("test")
    return repo


@pytest.mark.parametrize(
    ("command", "raises"),
    [
        (["echo", "test"], nullcontext()),
        ([], pytest.raises(ValueError, match="A command must be provided")),
    ],
)
@patch(f"{MODULE_NAME}.curator_submit")
def test_submit_cmd(mock_curator_submit: Mock, command: list[str], raises: AbstractContextManager[Any]) -> None:
    """Test that the submit command executes without errors."""
    with raises:
        submit_cli(
            command=command,
            login_node="login_node",
            account="test_account",
            partition="test_partition",
            container_image="test_image",
            num_nodes=1,
            container_mounts=None,  # default
            environment=None,  # default
            remote_files_path=pathlib.Path("/remote/files"),
        )

    if isinstance(raises, nullcontext):
        mock_curator_submit.assert_called_once()
    else:
        mock_curator_submit.assert_not_called()


@pytest.mark.parametrize(
    ("exclude_nodes"),
    [
        (None),
        (["node1", "node2"]),
    ],
)
def test_render_sbatch_script(exclude_nodes: list[str] | None) -> None:
    """Test that the render sbatch script function returns the correct sbatch script."""
    job_spec = SlurmJobSpec(
        login_node="login_node",
        container=ContainerSpec(
            squashfs_path="test_path", command=[str(_START_RAY), "arg1", "arg2"], mounts=[], environment=[]
        ),
        job_name="test_job",
        account="test_account",
        partition="test_partition",
        username="test_user",
        num_nodes=1,
        gres=GRES,
        exclusive=True,
        remote_job_path=pathlib.Path("/remote/files") / "test_job.20250611",
        time_limit="01:00:00",
        log_dir=pathlib.Path("/logs"),
        stop_retries_after=100,
        exclude_nodes=exclude_nodes,
        comment="test_comment",
    )
    sbatch_script = _render_sbatch_script(job_spec)
    expected_exclude_nodes = ",".join(job_spec.exclude_nodes) if job_spec.exclude_nodes else None
    assert "test_job" in sbatch_script
    assert "test_account" in sbatch_script
    assert "test_partition" in sbatch_script
    assert str(_START_RAY) in sbatch_script
    assert "arg1" in sbatch_script
    assert "arg2" in sbatch_script
    assert f"--gres={GRES}" in sbatch_script
    assert f"--time={job_spec.time_limit}" in sbatch_script
    assert f"STOP_RETRIES_AFTER={job_spec.stop_retries_after}" in sbatch_script
    if exclude_nodes:
        assert f"--exclude={expected_exclude_nodes}" in sbatch_script
    else:
        assert "--exclude=" not in sbatch_script
    assert f"--output={job_spec.log_dir!s}" in sbatch_script
    assert f'--comment="{job_spec.comment}"' in sbatch_script
    assert "COSMOS_S3_PROFILE_PATH" in sbatch_script
    assert "COSMOS_AZURE_PROFILE_PATH" in sbatch_script


def test_render_sbatch_script_with_qos() -> None:
    """Test that QoS is rendered into the sbatch script when provided."""
    job_spec = SlurmJobSpec(
        login_node="login_node",
        container=ContainerSpec(
            squashfs_path="test_path", command=[str(_START_RAY), "arg1", "arg2"], mounts=[], environment=[]
        ),
        job_name="test_job",
        account="test_account",
        partition="test_partition",
        username="test_user",
        num_nodes=1,
        gres=GRES,
        qos="normal",
        exclusive=True,
        remote_job_path=pathlib.Path("/remote/files") / "test_job.20260611",
        time_limit="01:00:00",
        log_dir=pathlib.Path("/logs"),
        stop_retries_after=100,
    )
    sbatch_script = _render_sbatch_script(job_spec)
    assert "#SBATCH --qos=normal" in sbatch_script


def test_render_sbatch_script_with_gpu_options() -> None:
    """Test that GPU-centered Slurm options are rendered into the sbatch script when provided."""
    job_spec = SlurmJobSpec(
        login_node="login_node",
        container=ContainerSpec(
            squashfs_path="test_path", command=[str(_START_RAY), "arg1", "arg2"], mounts=[], environment=[]
        ),
        job_name="test_job",
        account="test_account",
        partition="test_partition",
        username="test_user",
        num_nodes=1,
        gpus="8",
        exclusive=True,
        remote_job_path=pathlib.Path("/remote/files") / "test_job.20260611",
        log_dir=pathlib.Path("/logs"),
    )

    sbatch_script = _render_sbatch_script(job_spec)
    assert "#SBATCH --gpus=8" in sbatch_script
    assert "#SBATCH --gres" not in sbatch_script


def test_submit_uses_shared_defaults_for_container_runtime(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The batch submit path should inherit the shared Slurm container defaults."""
    repo = _create_repo(tmp_path)
    workspace = tmp_path / "workspace"
    cache = tmp_path / "cache"
    config = tmp_path / "config.yaml"
    aws_creds = tmp_path / "aws_credentials"
    config.write_text("config")
    aws_creds.write_text("creds")

    monkeypatch.chdir(repo)
    monkeypatch.delenv(_SLURM_ACCOUNT_ENV_VAR, raising=False)
    monkeypatch.setenv("HOST_ONLY", "host-value")
    monkeypatch.setenv("SLURM_JOB_ID", "outer-allocation")

    with (
        patch("cosmos_curator.client.slurm_cli.slurm_local.LOCAL_COSMOS_CURATOR_CONFIG_FILE", config),
        patch("cosmos_curator.client.slurm_cli.slurm_local.LOCAL_AWS_CREDENTIALS_FILE", aws_creds),
        patch(f"{MODULE_NAME}.curator_submit", return_value="12345") as mock_curator_submit,
    ):
        submit_cli(
            command=[
                "pixi",
                "run",
                "--as-is",
                "python",
                "-m",
                "cosmos_curator.pipelines.examples.hello_world_pipeline",
            ],
            workspace_path=workspace,
            cache_path=cache,
            remote_files_path=tmp_path / "job_files",
            environment="EXTRA=value,HOST_ONLY",
        )

    mock_curator_submit.assert_called_once()
    job_spec: SlurmJobSpec = mock_curator_submit.call_args.args[0]
    assert job_spec.login_node == "localhost"
    assert job_spec.account is None
    assert job_spec.partition is None
    assert job_spec.container.squashfs_path == str(
        pathlib.Path("~/container_images/cosmos-curator+1.0.0.sqsh").expanduser()
    )
    assert job_spec.container.command == [
        "pixi",
        "run",
        "--as-is",
        str(_START_RAY),
        "pixi",
        "run",
        "--as-is",
        "python",
        "-m",
        "cosmos_curator.pipelines.examples.hello_world_pipeline",
    ]

    mount_values = [str(mount) for mount in job_spec.container.mounts]
    assert f"{workspace.resolve()}:/config:rw" in mount_values
    assert f"{cache.resolve()}:/cache:rw" in mount_values
    assert f"{repo.resolve()}:/src/cosmos-curator:rw" in mount_values
    assert f"{config}:/cosmos_curator/config/cosmos_curator.yaml:ro" in mount_values
    assert f"{aws_creds}:/creds/s3_creds:ro" in mount_values

    env_vars = dict(entry.split("=", 1) for entry in job_spec.container.environment)
    assert env_vars["COSMOS_CURATOR_RAY_SLURM_JOB"] == "True"
    assert env_vars["PIXI_CACHE_DIR"] == "/cache/rattler/cache"
    assert env_vars["UV_CACHE_DIR"] == "/cache/rattler/cache/uv-cache"
    assert env_vars["TORCH_HOME"] == "/cache/torch"
    assert env_vars["TRITON_HOME"] == "/cache/triton"
    assert env_vars["CONDA_OVERRIDE_CUDA"] == "13.0.2"
    assert env_vars["EXTRA"] == "value"
    assert env_vars["HOST_ONLY"] == "host-value"
    assert "SLURM_JOB_ID" not in env_vars

    sbatch_script = _render_sbatch_script(job_spec)
    assert "#SBATCH -A" not in sbatch_script
    assert "#SBATCH -p" not in sbatch_script
    assert "bash -c" in sbatch_script
    assert 'exec "$@"' in sbatch_script
    assert "SLURM_PROCID" in sbatch_script
    assert "SLURM_JOB_ID" in sbatch_script


def test_submit_container_mounts_override_default_targets(tmp_path: pathlib.Path) -> None:
    """User-specified mount targets should not duplicate auto-detected defaults."""
    workspace = tmp_path / "workspace"
    cache = tmp_path / "cache"
    explicit_workspace = tmp_path / "explicit_workspace"
    explicit_cache = tmp_path / "explicit_cache"

    with patch(f"{MODULE_NAME}.curator_submit", return_value="12345") as mock_curator_submit:
        submit_cli(
            command=["echo", "test"],
            container_image="test_image",
            workspace_path=workspace,
            cache_path=cache,
            mount_s3_creds=False,
            remote_files_path=tmp_path / "job_files",
            container_mounts=f"{explicit_workspace}:/config:ro,{explicit_cache}:/cache:rw",
        )

    mock_curator_submit.assert_called_once()
    job_spec: SlurmJobSpec = mock_curator_submit.call_args.args[0]
    mounts_by_destination = {mount.dest: mount for mount in job_spec.container.mounts}
    assert [mount.dest for mount in job_spec.container.mounts].count("/config") == 1
    assert [mount.dest for mount in job_spec.container.mounts].count("/cache") == 1
    assert mounts_by_destination["/config"] == MountSpec(source=str(explicit_workspace), dest="/config", mode="ro")
    assert mounts_by_destination["/cache"] == MountSpec(source=str(explicit_cache), dest="/cache", mode="rw")


def test_submit_uses_sbatch_account_environment_default(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use the standard sbatch account environment variable without making account required."""
    monkeypatch.setenv(_SLURM_ACCOUNT_ENV_VAR, "env_account")

    with patch(f"{MODULE_NAME}.curator_submit", return_value="12345") as mock_curator_submit:
        submit_cli(
            command=["echo", "test"],
            container_image="test_image",
            workspace_path=tmp_path / "workspace",
            cache_path=tmp_path / "cache",
            mount_s3_creds=False,
            remote_files_path=tmp_path / "job_files",
        )

    mock_curator_submit.assert_called_once()
    job_spec: SlurmJobSpec = mock_curator_submit.call_args.args[0]
    assert job_spec.account == "env_account"
    assert "#SBATCH -A env_account" in _render_sbatch_script(job_spec)


def test_submit_account_option_overrides_environment_default(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit account should win over the environment default."""
    monkeypatch.setenv(_SLURM_ACCOUNT_ENV_VAR, "env_account")

    with patch(f"{MODULE_NAME}.curator_submit", return_value="12345") as mock_curator_submit:
        submit_cli(
            command=["echo", "test"],
            account="cli_account",
            container_image="test_image",
            workspace_path=tmp_path / "workspace",
            cache_path=tmp_path / "cache",
            mount_s3_creds=False,
            remote_files_path=tmp_path / "job_files",
        )

    mock_curator_submit.assert_called_once()
    job_spec: SlurmJobSpec = mock_curator_submit.call_args.args[0]
    assert job_spec.account == "cli_account"
    assert "#SBATCH -A cli_account" in _render_sbatch_script(job_spec)


def test_submit_account_option_trims_whitespace(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Trim an explicit account before using it in the sbatch script."""
    monkeypatch.setenv(_SLURM_ACCOUNT_ENV_VAR, "env_account")

    with patch(f"{MODULE_NAME}.curator_submit", return_value="12345") as mock_curator_submit:
        submit_cli(
            command=["echo", "test"],
            account=" cli_account ",
            container_image="test_image",
            workspace_path=tmp_path / "workspace",
            cache_path=tmp_path / "cache",
            mount_s3_creds=False,
            remote_files_path=tmp_path / "job_files",
        )

    mock_curator_submit.assert_called_once()
    job_spec: SlurmJobSpec = mock_curator_submit.call_args.args[0]
    assert job_spec.account == "cli_account"
    assert "#SBATCH -A cli_account" in _render_sbatch_script(job_spec)


def test_submit_blank_account_option_uses_environment_default(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Treat a blank account option the same as an omitted account option."""
    monkeypatch.setenv(_SLURM_ACCOUNT_ENV_VAR, "env_account")

    with patch(f"{MODULE_NAME}.curator_submit", return_value="12345") as mock_curator_submit:
        submit_cli(
            command=["echo", "test"],
            account="   ",
            container_image="test_image",
            workspace_path=tmp_path / "workspace",
            cache_path=tmp_path / "cache",
            mount_s3_creds=False,
            remote_files_path=tmp_path / "job_files",
        )

    mock_curator_submit.assert_called_once()
    job_spec: SlurmJobSpec = mock_curator_submit.call_args.args[0]
    assert job_spec.account == "env_account"
    assert "#SBATCH -A env_account" in _render_sbatch_script(job_spec)


def test_submit_trims_optional_slurm_directives(tmp_path: pathlib.Path) -> None:
    """Trim optional Slurm directives before rendering the sbatch script."""
    with patch(f"{MODULE_NAME}.curator_submit", return_value="12345") as mock_curator_submit:
        submit_cli(
            command=["echo", "test"],
            partition=" test_partition ",
            qos=" high ",
            container_image="test_image",
            workspace_path=tmp_path / "workspace",
            cache_path=tmp_path / "cache",
            mount_s3_creds=False,
            remote_files_path=tmp_path / "job_files",
        )

    mock_curator_submit.assert_called_once()
    job_spec: SlurmJobSpec = mock_curator_submit.call_args.args[0]
    sbatch_script = _render_sbatch_script(job_spec)
    assert job_spec.partition == "test_partition"
    assert job_spec.qos == "high"
    assert "#SBATCH -p test_partition" in sbatch_script
    assert "#SBATCH --qos=high" in sbatch_script


def test_submit_blank_optional_slurm_directives_are_omitted(tmp_path: pathlib.Path) -> None:
    """Omit optional Slurm directives when only whitespace is provided."""
    with patch(f"{MODULE_NAME}.curator_submit", return_value="12345") as mock_curator_submit:
        submit_cli(
            command=["echo", "test"],
            partition="   ",
            qos="   ",
            container_image="test_image",
            workspace_path=tmp_path / "workspace",
            cache_path=tmp_path / "cache",
            mount_s3_creds=False,
            remote_files_path=tmp_path / "job_files",
        )

    mock_curator_submit.assert_called_once()
    job_spec: SlurmJobSpec = mock_curator_submit.call_args.args[0]
    sbatch_script = _render_sbatch_script(job_spec)
    assert job_spec.partition is None
    assert job_spec.qos is None
    assert "#SBATCH -p" not in sbatch_script
    assert "#SBATCH --qos" not in sbatch_script


def test_submit_accepts_slurm_style_short_options(tmp_path: pathlib.Path) -> None:
    """Submit users can use familiar Slurm allocation aliases."""
    with patch(f"{MODULE_NAME}.curator_submit", return_value="12345") as mock_curator_submit:
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "submit",
                "-A",
                "test_account",
                "-p",
                "batch",
                "-q",
                "normal",
                "-G",
                "8",
                "-J",
                "batch_job",
                "-N",
                "2",
                "-t",
                "01:00:00",
                "--container-image",
                "test_image",
                "--workspace-path",
                str(tmp_path / "workspace"),
                "--cache-path",
                str(tmp_path / "cache"),
                "--remote-files-path",
                str(tmp_path / "job_files"),
                "--no-mount-s3-creds",
                "--",
                "echo",
                "hello",
            ],
        )

    assert result.exit_code == 0
    assert "Job submitted with ID: 12345" in result.output
    mock_curator_submit.assert_called_once()
    job_spec: SlurmJobSpec = mock_curator_submit.call_args.args[0]
    assert job_spec.account == "test_account"
    assert job_spec.partition == "batch"
    assert job_spec.qos == "normal"
    assert job_spec.gpus == "8"
    assert job_spec.job_name == "batch_job"
    assert job_spec.num_nodes == 2
    assert job_spec.time_limit == "01:00:00"

    sbatch_script = _render_sbatch_script(job_spec)
    assert "#SBATCH --gpus=8" in sbatch_script


def test_submit_rejects_gres_with_gpus(tmp_path: pathlib.Path) -> None:
    """Submit should not ask Slurm for GPUs through two incompatible option styles."""
    with patch(f"{MODULE_NAME}.curator_submit") as mock_curator_submit:
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "submit",
                "--gres",
                "gpu:8",
                "-G",
                "8",
                "--container-image",
                "test_image",
                "--workspace-path",
                str(tmp_path / "workspace"),
                "--cache-path",
                str(tmp_path / "cache"),
                "--remote-files-path",
                str(tmp_path / "job_files"),
                "--no-mount-s3-creds",
                "--",
                "echo",
                "hello",
            ],
        )

    assert result.exit_code != 0
    assert "--gres and --gpus cannot be used together" in result.output
    mock_curator_submit.assert_not_called()


@pytest.mark.parametrize(
    ("mail_type", "mail_user", "should_include_mail_type", "should_include_mail_user"),
    [
        (None, None, False, False),  # No mail options - should not include mail directives
        ("END,FAIL", "user@example.com", True, True),  # Both provided - should include both directives
        ("BEGIN", "user@example.com", True, True),  # Both provided with different type
        (None, "user@example.com", False, True),  # Only mail_user - should include only mail_user
    ],
)
def test_render_sbatch_script_with_mail_options(
    mail_type: str | None, mail_user: str | None, *, should_include_mail_type: bool, should_include_mail_user: bool
) -> None:
    """Test that mail options are correctly rendered in the sbatch script."""
    job_spec = SlurmJobSpec(
        login_node="login_node",
        container=ContainerSpec(
            squashfs_path="test_path", command=[str(_START_RAY), "arg1", "arg2"], mounts=[], environment=[]
        ),
        job_name="test_job",
        account="test_account",
        partition="test_partition",
        username="test_user",
        num_nodes=1,
        gres=GRES,
        exclusive=True,
        remote_job_path=pathlib.Path("/remote/files") / "test_job.20250611",
        time_limit="01:00:00",
        log_dir=pathlib.Path("/logs"),
        stop_retries_after=100,
        mail_type=mail_type,
        mail_user=mail_user,
    )
    sbatch_script = _render_sbatch_script(job_spec)

    if should_include_mail_type:
        assert "--mail-type=" in sbatch_script
        if mail_type:
            assert f"--mail-type={mail_type}" in sbatch_script
    else:
        assert "--mail-type=" not in sbatch_script

    if should_include_mail_user:
        assert "--mail-user=" in sbatch_script
        if mail_user:
            assert f"--mail-user={mail_user}" in sbatch_script
    else:
        assert "--mail-user=" not in sbatch_script


class TestSubmitCmd(unittest.TestCase):
    """Test the submit command."""

    def test_get_username(self) -> None:
        """Test that the get_username function returns the correct username."""
        with patch("os.getuid") as mock_getuid:
            mock_getuid.return_value = 123
            with patch("pwd.getpwuid") as mock_getpwuid:
                mock_getpwuid.return_value = Mock(pw_name="test_user")
                username = _get_username()
                assert username == "test_user"

    def test_mount_spec_from_str(self) -> None:
        """Test that the mount spec from string function returns the correct mount spec."""
        mount_str = "/src:/dst:rw"
        mount_spec = MountSpec.from_str(mount_str)
        assert mount_spec.source == "/src"
        assert mount_spec.dest == "/dst"
        assert mount_spec.mode == "rw"

    def test_slurm_job_spec(self) -> None:
        """Test that the slurm job spec function returns the correct slurm job spec."""
        job_spec = SlurmJobSpec(
            login_node="login_node",
            container=ContainerSpec(squashfs_path="test_path", command=["cmd"], mounts=[], environment=[]),
            job_name="test_job",
            account="test_account",
            partition="test_partition",
            username="test_user",
            num_nodes=1,
            gres=GRES,
            exclusive=True,
            remote_job_path=pathlib.Path("/remote/files") / "test_job.20250611",
            log_dir=pathlib.Path("/logs"),
        )
        assert job_spec.job_name == "test_job"
        assert job_spec.account == "test_account"
        assert job_spec.partition == "test_partition"
        assert job_spec.username == "test_user"
        assert job_spec.num_nodes == 1
        assert job_spec.gres == GRES
        assert job_spec.exclusive
        assert job_spec.remote_job_path == pathlib.Path("/remote/files") / "test_job.20250611"
        assert job_spec.log_dir == pathlib.Path("/logs")

    def test_parse_job_id(self) -> None:
        """Test that the parse job id function returns the correct job id."""
        output = "Submitted batch job 12345"
        job_id = _parse_job_id(output)
        assert job_id == "12345"

    def test_parse_job_id_with_dots_and_underscores(self) -> None:
        """Test that the parse job id function returns the correct job id with dots and underscores."""
        output = "Submitted batch job job_123.45"
        job_id = _parse_job_id(output)
        assert job_id == "job_123.45"

    def test_parse_job_id_missing_job_id(self) -> None:
        """Test that the parse job id function raises an error if the job id is missing."""
        output = "Submitted batch job"
        with pytest.raises(
            ValueError,
            match=r"Output 'Submitted batch job' does not contain 'Submitted batch job' followed by a job ID\.",
        ):
            _parse_job_id(output)

    def test_parse_job_id_invalid_output(self) -> None:
        """Test that the parse job id function raises an error if the output is invalid."""
        output = "Invalid output"
        with pytest.raises(
            ValueError, match=r"Output 'Invalid output' does not contain 'Submitted batch job' followed by a job ID\."
        ):
            _parse_job_id(output)

    def test_parse_job_id_empty_string(self) -> None:
        """Test that the parse job id function raises an error if the output is empty."""
        output = ""
        with pytest.raises(
            ValueError, match=r"Output '' does not contain 'Submitted batch job' followed by a job ID\."
        ):
            _parse_job_id(output)

    @patch("fabric.Connection")
    def test_connect_login_creates_connection(self, mock_connection: Mock) -> None:
        """Test that the connect function creates a connection with correct params."""
        conn = connect(remote_host="test_host", user="test_user")
        mock_connection.assert_called_once_with("test_host", user="test_user")
        assert conn == mock_connection.return_value

    @patch("fabric.Connection")
    def test_connect_verifies_connection_works(self, mock_connection: Mock) -> None:
        """Test that the connect function verifies the connection by running 'ls'."""
        mock_conn = mock_connection.return_value
        connect(remote_host="test_host", user="test_user")
        mock_conn.run.assert_called_once_with("ls", hide=True)

    def test_upload_text(self) -> None:
        """Test that the upload text function uploads the correct files."""
        connection = Mock()
        files = [("text1", pathlib.Path("/remote/path1"), 0o644), ("text2", pathlib.Path("/remote/path2"), 0o755)]
        upload_text(connection, files)
        EXPECTED_CALL_COUNT = 2
        assert connection.put.call_count == EXPECTED_CALL_COUNT
        assert connection.run.call_count == EXPECTED_CALL_COUNT

    def test_upload_text_empty_list(self) -> None:
        """Test that the upload text function raises an error if the list of files is empty."""
        connection = Mock()
        files: list[tuple[str, pathlib.Path, int]] = []
        with pytest.raises(ValueError, match="Must upload at least one file"):
            upload_text(connection, files)

    def test_upload_text_file_mode_too_low(self) -> None:
        """Test that the upload text function raises an error if the file mode is too low."""
        connection = Mock()
        files = [("text", pathlib.Path("/remote/path"), -1)]
        with pytest.raises(ValueError, match="Invalid octal file mode: -0o1"):
            upload_text(connection, files)

    def test_upload_text_file_mode_too_high(self) -> None:
        """Test that the upload text function raises an error if the file mode is too high."""
        connection = Mock()
        files = [("text", pathlib.Path("/remote/path"), 0o7777777)]
        with pytest.raises(ValueError, match="Invalid octal file mode: 0o7777777"):
            upload_text(connection, files)


class TestSubmitCurationJob:
    """Test the submit curation job function."""

    @pytest.fixture
    def job_spec(self) -> SlurmJobSpec:
        """Test that the submit curation job function returns the correct job spec."""
        return SlurmJobSpec(
            login_node="login_node",
            container=ContainerSpec(squashfs_path="test_path", command=["cmd"], mounts=[], environment=[]),
            job_name="test_job",
            account="test_account",
            partition="test_partition",
            username="test_user",
            num_nodes=1,
            gres=GRES,
            exclusive=True,
            remote_job_path=pathlib.Path("/remote/files") / "test_job.20250611",
            time_limit="01:00:00",
            log_dir=pathlib.Path("/logs"),
        )

    def test_curator_submit(self, mock_connection: Mock, job_spec: SlurmJobSpec) -> None:
        """Test that the submit curation job function submits the correct job."""
        conn = mock_connection.return_value

        failed_result = Mock()
        failed_result.exited = 1

        # Create an exception that will be raised on first call
        unexpected_exit = invoke.exceptions.UnexpectedExit(result=failed_result)

        # Create a mock for successful run with job ID for sbatch command
        success_result = Mock()
        success_result.stdout = "Submitted batch job 12345"

        # Configure the run method to raise exception when checking that the remote dir exists
        conn.run.side_effect = [
            Mock(),  # ls call succeeds
            unexpected_exit,  # directory check should fail as expected (test -e)
            Mock(),  # mkdir call succeeds
            Mock(),  # chmod job dir
            Mock(),  # chmod sbatch script
            Mock(),  # chmod prometheus service discovery script
            success_result,  # sbatch command returns job ID
        ]

        job_id = curator_submit(job_spec)

        assert job_id == "12345"
        sbatch_calls = [
            call[0][0]
            for call in conn.run.call_args_list
            if isinstance(call[0][0], str) and call[0][0].startswith("sbatch")
        ]
        EXPECTED_SBATCH_CALL_COUNT = 1
        assert len(sbatch_calls) == EXPECTED_SBATCH_CALL_COUNT

    def test_curator_submit_quotes_remote_paths(self, mock_connection: Mock, job_spec: SlurmJobSpec) -> None:
        """Quote remote paths passed through shell commands."""
        job_spec.remote_job_path = pathlib.Path("/remote/files/test job;touch bad")
        conn = mock_connection.return_value

        failed_result = Mock()
        failed_result.exited = 1
        unexpected_exit = invoke.exceptions.UnexpectedExit(result=failed_result)
        success_result = Mock()
        success_result.stdout = "Submitted batch job 12345"
        conn.run.side_effect = [
            Mock(),
            unexpected_exit,
            Mock(),
            Mock(),
            Mock(),
            Mock(),
            success_result,
        ]

        curator_submit(job_spec)

        commands = [call_args.args[0] for call_args in conn.run.call_args_list]
        assert "mkdir -p '/remote/files/test job;touch bad'" in commands
        assert "sbatch '/remote/files/test job;touch bad/sbatch.sh'" in commands

    def test_curator_submit_suggests_account_when_sbatch_requires_one(
        self, mock_connection: Mock, job_spec: SlurmJobSpec
    ) -> None:
        """Make missing account failures actionable without requiring accounts everywhere."""
        job_spec.account = None
        conn = mock_connection.return_value

        missing_dir_result = Mock()
        missing_dir_result.exited = 1
        missing_dir = invoke.exceptions.UnexpectedExit(result=missing_dir_result)

        sbatch_result = Mock()
        sbatch_result.exited = 1
        sbatch_result.stderr = (
            "sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified"
        )
        sbatch_failure = invoke.exceptions.UnexpectedExit(result=sbatch_result)

        conn.run.side_effect = [
            Mock(),
            missing_dir,
            Mock(),
            Mock(),
            Mock(),
            Mock(),
            sbatch_failure,
        ]

        with pytest.raises(ValueError, match=f"Rerun with --account <slurm_account> or set {_SLURM_ACCOUNT_ENV_VAR}"):
            curator_submit(job_spec)


class TestMountSpec:
    """Test the mount spec class."""

    def test_mount_spec_can_be_created_with_source_and_dest(self) -> None:
        """Test that the mount spec can be created with source and dest."""
        mount_spec = MountSpec(source="/src", dest="/dst")
        assert mount_spec.source == "/src"
        assert mount_spec.dest == "/dst"
        assert mount_spec.mode == "rw"

    def test_mount_spec_can_be_created_with_source_dest_and_mode(self) -> None:
        """Test that the mount spec can be created with source, dest, and mode."""
        mount_spec = MountSpec(source="/src", dest="/dst", mode="ro")
        assert mount_spec.source == "/src"
        assert mount_spec.dest == "/dst"
        assert mount_spec.mode == "ro"

    def test_mount_spec_from_str(self) -> None:
        """Test that the mount spec can be created from a string."""
        mount_spec = MountSpec.from_str("/src:/dst")
        assert mount_spec.source == "/src"
        assert mount_spec.dest == "/dst"
        assert mount_spec.mode == "rw"

    def test_mount_spec_from_str_with_mode(self) -> None:
        """Test that the mount spec can be created from a string with mode."""
        mount_spec = MountSpec.from_str("/src:/dst:ro")
        assert mount_spec.source == "/src"
        assert mount_spec.dest == "/dst"
        assert mount_spec.mode == "ro"

    def test_mount_spec_str(self) -> None:
        """Test that the mount spec can be converted to a string."""
        mount_spec = MountSpec(source="/src", dest="/dst", mode="ro")
        assert str(mount_spec) == "/src:/dst:ro"

    def test_mount_spec_from_str_with_invalid_format(self) -> None:
        """Test that the mount spec raises an error if the format is invalid."""
        with pytest.raises(ValueError, match="`/src` must have at least 2 or colon separated parts"):
            MountSpec.from_str("/src")

    def test_mount_spec_from_str_with_too_many_parts(self) -> None:
        """Test that the mount spec raises an error if the format has too many parts."""
        with pytest.raises(ValueError, match="`/src:/dst:ro:extra` must have at least 2 or colon separated parts"):
            MountSpec.from_str("/src:/dst:ro:extra")

    def test_mount_spec_valid_mode(self) -> None:
        """Test that the mount spec can be created with valid modes."""
        MountSpec(source="/src", dest="/dst", mode="rw")
        MountSpec(source="/src", dest="/dst", mode="ro")

    def test_mount_spec_invalid_mode(self) -> None:
        """Test that the mount spec raises an error if the mode is invalid."""
        with pytest.raises(ValueError):  # noqa: PT011
            MountSpec(source="/src", dest="/dst", mode="rx")


class TestContainerSpec:
    """Test the container spec class."""

    @pytest.mark.parametrize("missing_fields", [[], ["command"], ["mounts"], ["environment"], ["squashfs_path"]])
    def test_container_spec_creation(self, missing_fields: list[str]) -> None:
        """Test that the container spec can be created with the correct fields."""
        args: dict[str, Any] = {}
        mounts = [MountSpec(source="/src", dest="/dst")]
        command = ["python", "script.py"]
        squashfs_path = "/path/to/image.sqsh"
        environment = ["a", "b"]

        if "mounts" not in missing_fields:
            args["mounts"] = mounts

        if "command" not in missing_fields:
            args["command"] = command

        if "squashfs_path" not in missing_fields:
            args["squashfs_path"] = squashfs_path

        if "environment" not in missing_fields:
            args["environment"] = environment

        ctx = nullcontext() if len(missing_fields) == 0 else pytest.raises(TypeError)
        with ctx:
            container_spec = ContainerSpec(**args)

        if len(missing_fields) == 0:
            assert container_spec.mounts == mounts
            assert container_spec.command == command
            assert container_spec.squashfs_path == squashfs_path
            assert container_spec.environment == environment


class TestSubmit:
    """Test the submit function."""

    @pytest.fixture
    def mock_curator_submit(self, mocker: Mock) -> Any:  # noqa: ANN401
        """Mock Slurm job submission."""
        return mocker.patch(f"{MODULE_NAME}.curator_submit")

    def test_submit(self, mock_curator_submit: Mock) -> None:
        """Test that the submit function launches the correct job."""
        submit_cli(
            command=[str(_START_RAY), "arg1", "arg2"],
            login_node="login_node",
            account="test_account",
            partition="test_partition",
            container_image="test_image",
            num_nodes=1,
            remote_files_path=pathlib.Path("/remote/files"),
            gres=GRES,
            exclusive=True,
        )
        mock_curator_submit.assert_called_once()

    def test_submit_container_mounts(self, mock_curator_submit: Mock) -> None:
        """Test that the submit function launches the correct job with container mounts."""
        submit_cli(
            command=[str(_START_RAY), "arg1", "arg2"],
            login_node="login_node",
            account="test_account",
            partition="test_partition",
            container_image="test_image",
            container_mounts="src0:dst0,src1:dst1",
            num_nodes=1,
            remote_files_path=pathlib.Path("/remote/files"),
            gres=GRES,
            exclusive=True,
        )
        mock_curator_submit.assert_called_once()

    def test_submit_environment(self, mock_curator_submit: Mock) -> None:
        """Test that the submit function launches the correct job with environment variables."""
        submit_cli(
            command=[str(_START_RAY), "arg1", "arg2"],
            login_node="login_node",
            account="test_account",
            partition="test_partition",
            container_image="test_image",
            environment="VA1,VA2",
            num_nodes=1,
            remote_files_path=pathlib.Path("/remote/files"),
            gres=GRES,
            exclusive=True,
        )
        mock_curator_submit.assert_called_once()

    def test_submit_invalid_mounts(self, mock_curator_submit: Mock) -> None:
        """Test that the submit function raises an error if the container mounts are invalid."""
        with pytest.raises(ValueError, match=r"(?i).*must have at least 2 or colon separated parts.*"):
            submit_cli(
                command=[str(_START_RAY), "arg1", "arg2"],
                login_node="login_node",
                account="test_account",
                partition="test_partition",
                container_image="test_image",
                num_nodes=1,
                remote_files_path=pathlib.Path("/remote/files"),
                gres=GRES,
                exclusive=True,
                container_mounts="invalid_mounts",
            )
        mock_curator_submit.assert_not_called()

    @pytest.mark.parametrize(
        ("mail_type", "mail_user", "expected_mail_type", "should_raise"),
        [
            (None, None, None, False),  # No mail options - valid
            ("BEGIN", "user@example.com", "BEGIN", False),  # Both provided - valid
            (None, "user@example.com", None, False),  # Only mail_user - valid, SLURM will use default
            ("END", None, None, True),  # Only mail_type without user - should raise error
        ],
    )
    def test_submit_with_mail_options(
        self,
        mock_curator_submit: Mock,
        mail_type: str | None,
        mail_user: str | None,
        expected_mail_type: str | None,
        *,
        should_raise: bool,
    ) -> None:
        """Test that mail options are correctly handled in the submit function."""
        ctx = (
            pytest.raises(ValueError, match="If --mail-type is provided, --mail-user must also be provided")
            if should_raise
            else nullcontext()
        )

        with ctx:
            submit_cli(
                command=[str(_START_RAY), "arg1", "arg2"],
                login_node="login_node",
                account="test_account",
                partition="test_partition",
                container_image="test_image",
                num_nodes=1,
                remote_files_path=pathlib.Path("/remote/files"),
                gres=GRES,
                exclusive=True,
                mail_type=mail_type,
                mail_user=mail_user,
            )

        if should_raise:
            mock_curator_submit.assert_not_called()
        else:
            mock_curator_submit.assert_called_once()

            # Get the SlurmJobSpec that was passed to curator_submit
            call_args = mock_curator_submit.call_args
            job_spec: SlurmJobSpec = call_args[0][0]

            assert job_spec.mail_user == mail_user
            assert job_spec.mail_type == expected_mail_type

    def test_submit_with_qos(self, mock_curator_submit: Mock) -> None:
        """Test that the submit command forwards QoS into the job spec."""
        submit_cli(
            command=[str(_START_RAY), "arg1", "arg2"],
            login_node="login_node",
            account="test_account",
            partition="test_partition",
            container_image="test_image",
            num_nodes=1,
            remote_files_path=pathlib.Path("/remote/files"),
            gres=GRES,
            qos="high",
            exclusive=True,
        )

        mock_curator_submit.assert_called_once()
        job_spec: SlurmJobSpec = mock_curator_submit.call_args.args[0]
        assert job_spec.qos == "high"


@pytest.mark.parametrize(
    ("num_nodes", "head_node", "nodename", "procid", "stop_retries_after", "is_head_node"),
    [
        (1, "head_node", "head_node", 0, 100, True),
        (1, "head_node", "worker_node", 1, 100, False),
    ],
)
def test_head_node_is_head_node(  # noqa: PLR0913
    num_nodes: int, head_node: str, nodename: str, procid: int, stop_retries_after: int, *, is_head_node: bool
) -> None:
    """Test that the head node is the head node."""
    slurm_env = SlurmEnv(
        num_nodes=num_nodes,
        head_node=head_node,
        nodename=nodename,
        procid=procid,
        stop_retries_after=stop_retries_after,
    )
    assert slurm_env.is_head_node() == is_head_node
