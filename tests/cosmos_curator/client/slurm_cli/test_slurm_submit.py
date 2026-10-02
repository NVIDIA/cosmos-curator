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
"""Test the slurm module."""

import pathlib
import shutil
import subprocess
import unittest
from contextlib import AbstractContextManager, nullcontext
from typing import Any
from unittest.mock import Mock, patch

import invoke
import pytest
from typer.testing import CliRunner

from cosmos_curator.client.cli import cosmos_curator
from cosmos_curator.client.slurm_cli.slurm_common import _SLURM_ACCOUNT_ENV_VAR, MountSpec, _get_username
from cosmos_curator.client.slurm_cli.slurm_submit import (
    _START_RAY,
    ContainerSpec,
    SlurmJobSpec,
    SlurmSubmitOptions,
    _parse_job_id,
    _render_sbatch_script,
    build_slurm_submit_job_spec,
    connect,
    curator_submit,
    render_slurm_submit_script,
    submit_cli,
    upload_text,
)
from cosmos_curator.scripts.onto_slurm import SlurmEnv

MODULE_NAME = "cosmos_curator.client.slurm_cli.slurm_submit"
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


def _assert_launcher_env_scrubbed_from_submit(env_vars: dict[str, str], sbatch_script: str) -> None:
    launcher_env_vars = (
        "PIXI_PROJECT_MANIFEST",
        "PIXI_PROJECT_ROOT",
        "PIXI_ENVIRONMENT_NAME",
        "PIXI_IN_SHELL",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_ENV_SHLVL_2_PIXI_PROJECT_MANIFEST",
    )
    for env_var in launcher_env_vars:
        assert env_var not in env_vars

    unset_line = next(
        (line for line in sbatch_script.splitlines() if line.startswith("unset PIXI_ENVIRONMENT_NAME")),
        None,
    )
    assert unset_line is not None, "Expected 'unset PIXI_ENVIRONMENT_NAME' line in sbatch script"
    assert "PIXI_PROJECT_MANIFEST" in unset_line
    assert "PIXI_ENVIRONMENT_NAME" in unset_line
    assert "CONDA_PREFIX" in unset_line
    assert "CONDA_DEFAULT_ENV" in unset_line
    assert "for launcher_env_var in ${!CONDA_ENV_SHLVL_@} ${!CONDA_PREFIX_@}; do" in sbatch_script
    assert sbatch_script.index("unset PIXI_ENVIRONMENT_NAME") < sbatch_script.index(
        "# Export container environment variables"
    )
    assert sbatch_script.index("unset PIXI_ENVIRONMENT_NAME") < sbatch_script.index("srun \\")


def _seed_launcher_activation_env(monkeypatch: pytest.MonkeyPatch, repo: pathlib.Path) -> None:
    monkeypatch.setenv("PIXI_PROJECT_MANIFEST", str(repo / "pixi.toml"))
    monkeypatch.setenv("PIXI_PROJECT_ROOT", str(repo))
    monkeypatch.setenv("PIXI_ENVIRONMENT_NAME", "cluster")
    monkeypatch.setenv("PIXI_IN_SHELL", "1")
    monkeypatch.setenv("CONDA_PREFIX", str(repo / ".pixi" / "envs" / "cluster"))
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "cosmos-curator:cluster")
    monkeypatch.setenv("CONDA_ENV_SHLVL_2_PIXI_PROJECT_MANIFEST", str(repo / "pixi.toml"))


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
            squashfs_path="test_path",
            command=[str(_START_RAY), "arg1", "arg2"],
            mounts=[MountSpec("/remote/files/test_job.20260611", "/remote_files")],
            environment=[],
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
        ray_io_slots_per_node=9,
        exclude_nodes=exclude_nodes,
        comment="test_comment",
    )
    sbatch_script = _render_sbatch_script(job_spec)
    expected_exclude_nodes = ",".join(job_spec.exclude_nodes) if job_spec.exclude_nodes else None
    assert "#SBATCH --propagate=NONE" in sbatch_script
    assert "test_job" in sbatch_script
    assert "test_account" in sbatch_script
    assert "test_partition" in sbatch_script
    assert str(_START_RAY) in sbatch_script
    assert "arg1" in sbatch_script
    assert "arg2" in sbatch_script
    assert f"--gres={GRES}" in sbatch_script
    assert f"--time={job_spec.time_limit}" in sbatch_script
    assert f"STOP_RETRIES_AFTER={job_spec.stop_retries_after}" in sbatch_script
    assert f"COSMOS_CURATOR_RAY_IO_SLOTS_PER_NODE={job_spec.ray_io_slots_per_node}" in sbatch_script
    if exclude_nodes:
        assert f"--exclude={expected_exclude_nodes}" in sbatch_script
    else:
        assert "--exclude=" not in sbatch_script
    assert f"--output={job_spec.log_dir!s}" in sbatch_script
    assert f'--comment="{job_spec.comment}"' in sbatch_script
    assert "COSMOS_S3_PROFILE_PATH" in sbatch_script
    assert "COSMOS_AZURE_PROFILE_PATH" in sbatch_script


def test_render_sbatch_script_omits_exclusive_when_disabled() -> None:
    """The rendered sbatch script should honor non-exclusive job specs."""
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
        exclusive=False,
        remote_job_path=pathlib.Path("/remote/files") / "test_job.20260611",
        time_limit="01:00:00",
        log_dir=pathlib.Path("/logs"),
        stop_retries_after=100,
    )

    sbatch_script = _render_sbatch_script(job_spec)

    assert "#SBATCH --exclusive" not in sbatch_script


def test_render_sbatch_script_passes_extra_container_environment_keys_without_values() -> None:
    """Caller-selected environment keys should be passed by name, not rendered with secret values."""
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
        exclusive=False,
        remote_job_path=pathlib.Path("/remote/files") / "test_job.20260611",
        time_limit="01:00:00",
        log_dir=pathlib.Path("/logs"),
        extra_container_env_keys=["GENERIC_SECRET_TOKEN", "GENERIC_ORG"],
    )

    sbatch_script = _render_sbatch_script(job_spec)

    assert "\n  . " not in sbatch_script
    assert "NGC_NVCF_API_KEY" not in sbatch_script
    assert "PERF_NGC_NVCF_API_KEY" not in sbatch_script
    assert "GENERIC_SECRET_TOKEN=secret" not in sbatch_script
    assert "GENERIC_ORG=secret" not in sbatch_script
    container_env_line = next(line for line in sbatch_script.splitlines() if line.strip().startswith("--container-env"))
    for key in ("GENERIC_SECRET_TOKEN", "GENERIC_ORG"):
        assert key in container_env_line


def test_render_sbatch_script_sources_environment_file_before_srun() -> None:
    """Sourced exported values should override Curator defaults before srun builds the container env."""
    job_spec = SlurmJobSpec(
        login_node="login_node",
        container=ContainerSpec(
            squashfs_path="test_path",
            command=[str(_START_RAY), "arg1", "arg2"],
            mounts=[],
            environment=["COSMOS_S3_PROFILE_PATH=/creds/s3_creds"],
        ),
        job_name="test_job",
        account="test_account",
        partition="test_partition",
        username="test_user",
        num_nodes=1,
        gres=GRES,
        exclusive=False,
        remote_job_path=pathlib.Path("/remote/files") / "test_job.20260611",
        time_limit="01:00:00",
        log_dir=pathlib.Path("/logs"),
        source_environment_file=pathlib.Path("/remote/files/test_job.20260611/secrets.env"),
        extra_container_env_keys=["NGC_NVCF_API_KEY"],
    )

    sbatch_script = _render_sbatch_script(job_spec)

    assert 'source "/remote/files/test_job.20260611/secrets.env"' in sbatch_script
    assert sbatch_script.index('export COSMOS_S3_PROFILE_PATH="/creds/s3_creds"') < sbatch_script.index(
        'source "/remote/files/test_job.20260611/secrets.env"'
    )
    assert sbatch_script.index('source "/remote/files/test_job.20260611/secrets.env"') < sbatch_script.index("srun \\")
    container_env_line = next(line for line in sbatch_script.splitlines() if line.strip().startswith("--container-env"))
    assert "NGC_NVCF_API_KEY" in container_env_line


def test_render_slurm_submit_script_uses_submit_mount_shape() -> None:
    """The public dry-run renderer should match the sbatch shape used by submit."""
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
        exclusive=False,
        remote_job_path=pathlib.Path("/remote/files") / "test_job.20260611",
        time_limit="01:00:00",
        log_dir=pathlib.Path("/logs"),
    )

    sbatch_script = render_slurm_submit_script(job_spec)

    assert "#SBATCH --job-name=test_job" in sbatch_script
    assert "/remote/files/test_job.20260611:/remote_files:rw" in sbatch_script
    assert sbatch_script.count(":/remote_files:rw") == 1


def test_render_sbatch_script_does_not_source_arbitrary_files() -> None:
    """The sbatch script should not source arbitrary files."""
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
        exclusive=False,
        remote_job_path=pathlib.Path("/remote/files") / "test_job.20260611",
        log_dir=pathlib.Path("/logs"),
    )

    sbatch_script = _render_sbatch_script(job_spec)

    assert "set -a" not in sbatch_script


def test_sbatch_template_uses_single_environment_file_source_hook() -> None:
    """The sbatch template should keep the environment import small and fixed."""
    template = (
        pathlib.Path(__file__).parents[4] / "cosmos_curator" / "client" / "slurm_cli" / "sbatch.sh.j2"
    ).read_text(encoding="utf-8")

    assert template.count('source "{{source_environment_file}}"') == 1
    assert template.count("set -a") == 1
    assert template.count("set +a") == 1


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
    _seed_launcher_activation_env(monkeypatch, repo)

    with (
        patch("cosmos_curator.client.slurm_cli.slurm_common.LOCAL_COSMOS_CURATOR_CONFIG_FILE", config),
        patch("cosmos_curator.client.slurm_cli.slurm_common.LOCAL_AWS_CREDENTIALS_FILE", aws_creds),
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
    assert env_vars["PYTHONPATH"] == "/opt/cosmos-curator"
    assert env_vars["PIXI_CACHE_DIR"] == "/cache/rattler/cache"
    assert env_vars["UV_CACHE_DIR"] == "/cache/rattler/cache/uv-cache"
    assert env_vars["TORCH_HOME"] == "/cache/torch"
    assert env_vars["TRITON_HOME"] == "/cache/triton"
    assert env_vars["HF_HOME"] == "/cache/huggingface"
    assert env_vars["LAION_CACHE_HOME"] == "/cache/laion"
    assert env_vars["CONDA_OVERRIDE_CUDA"] == "13.0.3"
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
    _assert_launcher_env_scrubbed_from_submit(env_vars, sbatch_script)


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


def test_submit_node_local_mounts_are_marked_to_skip_login_node_validation(tmp_path: pathlib.Path) -> None:
    """Node-local mount sources may exist only on allocated compute nodes."""
    node_local_source = "/raid/scratch/$USER/$SLURM_JOB_ID"
    job_spec = build_slurm_submit_job_spec(
        ["echo", "test"],
        SlurmSubmitOptions(
            login_node="remote-login",
            username="cluster-user",
            container_image="test_image",
            workspace_path=tmp_path / "workspace",
            cache_path=tmp_path / "cache",
            mount_s3_creds=False,
            node_local_mounts=f"{node_local_source}:/config/models:rw",
        ),
    )

    assert MountSpec(source=node_local_source, dest="/config/models", mode="rw") in job_spec.container.mounts
    assert job_spec.node_local_mount_sources == [node_local_source]
    sbatch_script = render_slurm_submit_script(job_spec)
    assert f'mkdir -p "{node_local_source}"' not in sbatch_script


def test_submit_can_prepare_node_local_mount_sources(tmp_path: pathlib.Path) -> None:
    """Node-local mount preparation is explicit so cluster-specific scratch policy stays opt-in."""
    node_local_source = "/raid/scratch/$USER/$SLURM_JOB_ID"
    other_node_local_source = "/local/scratch/$USER/$SLURM_JOB_ID"
    job_spec = build_slurm_submit_job_spec(
        ["echo", "test"],
        SlurmSubmitOptions(
            login_node="remote-login",
            username="cluster-user",
            container_image="test_image",
            workspace_path=tmp_path / "workspace",
            cache_path=tmp_path / "cache",
            mount_s3_creds=False,
            node_local_mounts=f"{node_local_source}:/config/models:rw,{other_node_local_source}:/scratch:rw",
            prepare_node_local_mounts=True,
        ),
    )

    sbatch_script = render_slurm_submit_script(job_spec)
    assert sbatch_script.count("srun --mpi=none --nodes=") == 1
    assert f'mkdir -p "{node_local_source}"' in sbatch_script
    assert f'mkdir -p "{other_node_local_source}"' in sbatch_script
    assert sbatch_script.index(f'mkdir -p "{node_local_source}"') < sbatch_script.index(
        f"{node_local_source}:/config/models:rw"
    )
    assert sbatch_script.index(f'mkdir -p "{other_node_local_source}"') < sbatch_script.index(
        f"{other_node_local_source}:/scratch:rw"
    )


def test_build_submit_job_spec_accepts_options_object(tmp_path: pathlib.Path) -> None:
    """Programmatic callers pass one options object instead of mirroring the CLI signature."""
    options = SlurmSubmitOptions(
        login_node="remote-login",
        username="cluster-user",
        account="acct",
        partition="batch",
        remote_files_path=pathlib.Path("/remote/files"),
        container_image="test_image",
        workspace_path=tmp_path / "workspace",
        cache_path=tmp_path / "cache",
        mount_s3_creds=False,
        mount_azure_creds=False,
        job_name="remote_job",
        gres="gpu:8",
        log_dir=pathlib.Path("/remote/logs"),
    )

    job_spec = build_slurm_submit_job_spec(["echo", "hello"], options)

    assert job_spec.login_node == "remote-login"
    assert job_spec.username == "cluster-user"
    assert job_spec.account == "acct"
    assert job_spec.gres == "gpu:8"
    assert job_spec.log_dir == pathlib.Path("/remote/logs")


def test_build_remote_submit_job_spec_does_not_prepare_local_workspace_or_cache(tmp_path: pathlib.Path) -> None:
    """Remote submit paths belong to the login node and should not be created on the launcher."""
    workspace = tmp_path / "remote-only" / "workspace"
    cache = tmp_path / "remote-only" / "cache"

    job_spec = build_slurm_submit_job_spec(
        ["echo", "hello"],
        SlurmSubmitOptions(
            login_node="remote-login",
            username="cluster-user",
            account="acct",
            partition="batch",
            remote_files_path=pathlib.Path("/remote/files"),
            container_image="test_image",
            workspace_path=workspace,
            cache_path=cache,
            mount_s3_creds=False,
            mount_azure_creds=False,
            job_name="remote_job",
            gres="gpu:8",
            log_dir=pathlib.Path("/remote/logs"),
        ),
    )

    assert not workspace.exists()
    assert not cache.exists()
    mount_values = [str(mount) for mount in job_spec.container.mounts]
    assert f"{workspace}:/config:rw" in mount_values
    assert f"{cache}:/cache:rw" in mount_values


def test_build_remote_submit_job_spec_does_not_infer_curator_path(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Programmatic builder callers should only mount source when curator_path is explicit."""
    repo = _create_repo(tmp_path)
    monkeypatch.chdir(repo)

    job_spec = build_slurm_submit_job_spec(
        ["echo", "hello"],
        SlurmSubmitOptions(
            login_node="remote-login",
            username="cluster-user",
            remote_files_path=pathlib.Path("/remote/files"),
            container_image="test_image",
            workspace_path=pathlib.Path("/remote/workspace"),
            cache_path=pathlib.Path("/remote/cache"),
            mount_s3_creds=False,
            mount_azure_creds=False,
        ),
    )

    mount_values = [str(mount) for mount in job_spec.container.mounts]
    assert f"{repo}:/src/cosmos-curator:rw" not in mount_values


def test_submit_cli_keeps_curator_path_inference(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI keeps the repo-root convenience behavior for existing users."""
    repo = _create_repo(tmp_path)
    monkeypatch.chdir(repo)

    with patch(f"{MODULE_NAME}.curator_submit", return_value="12345") as mock_curator_submit:
        submit_cli(
            command=["echo", "hello"],
            container_image="test_image",
            workspace_path=tmp_path / "workspace",
            cache_path=tmp_path / "cache",
            mount_s3_creds=False,
            remote_files_path=tmp_path / "job_files",
        )

    job_spec: SlurmJobSpec = mock_curator_submit.call_args.args[0]
    mount_values = [str(mount) for mount in job_spec.container.mounts]
    assert f"{repo.resolve()}:/src/cosmos-curator:rw" in mount_values


def test_build_remote_submit_job_spec_preserves_remote_credential_mounts(tmp_path: pathlib.Path) -> None:
    """Remote credential paths are validated on the login node, not on the launcher."""
    aws_creds = tmp_path / "missing" / "aws_credentials"
    azure_creds = tmp_path / "missing" / "azure_credentials"
    config = tmp_path / "missing" / "config.yaml"

    with (
        patch("cosmos_curator.client.slurm_cli.slurm_common.LOCAL_AWS_CREDENTIALS_FILE", aws_creds),
        patch("cosmos_curator.client.slurm_cli.slurm_common.LOCAL_AZURE_CREDENTIALS_FILE", azure_creds),
        patch("cosmos_curator.client.slurm_cli.slurm_common.LOCAL_COSMOS_CURATOR_CONFIG_FILE", config),
    ):
        job_spec = build_slurm_submit_job_spec(
            ["python", "-m", "cosmos_curator.client.model_cli"],
            SlurmSubmitOptions(
                login_node="remote-login",
                username="cluster-user",
                remote_files_path=pathlib.Path("/remote/files"),
                container_image="test_image",
                workspace_path=tmp_path / "workspace",
                cache_path=tmp_path / "cache",
                mount_s3_creds=True,
                mount_azure_creds=True,
                job_name="remote_job",
            ),
        )

    mount_values = [str(mount) for mount in job_spec.container.mounts]
    assert f"{aws_creds}:/creds/s3_creds:ro" in mount_values
    assert f"{azure_creds}:/creds/azure_creds:ro" in mount_values
    assert f"{config}:/cosmos_curator/config/cosmos_curator.yaml:ro" in mount_values


def test_build_local_submit_job_spec_keeps_workspace_and_cache_preparation(tmp_path: pathlib.Path) -> None:
    """Local submit keeps existing convenience behavior for workspace/cache directories."""
    workspace = tmp_path / "workspace"
    cache = tmp_path / "cache"

    job_spec = build_slurm_submit_job_spec(
        ["echo", "hello"],
        SlurmSubmitOptions(
            login_node="localhost",
            username="local-user",
            remote_files_path=tmp_path / "job_files",
            container_image="test_image",
            workspace_path=workspace,
            cache_path=cache,
            mount_s3_creds=False,
            mount_azure_creds=False,
            job_name="local_job",
        ),
    )

    assert workspace.is_dir()
    assert (cache / "rattler" / "cache" / "uv-cache").is_dir()
    assert (cache / "huggingface").is_dir()
    assert (cache / "laion").is_dir()
    mount_values = [str(mount) for mount in job_spec.container.mounts]
    assert f"{workspace.resolve()}:/config:rw" in mount_values
    assert f"{cache.resolve()}:/cache:rw" in mount_values


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


@pytest.mark.parametrize(
    ("gpu_option", "gpu_value", "expected_gpu_directives"),
    [
        ("-G", "8", ["#SBATCH --gpus=8"]),
        ("--gpus-per-node", "4", ["#SBATCH --gpus-per-node=4"]),
        ("--gpus-per-node", " h100:4 ", ["#SBATCH --gpus-per-node=h100:4"]),
        ("--gpus-per-node", "   ", []),
    ],
)
def test_submit_dry_run_prints_sbatch_without_submitting(
    tmp_path: pathlib.Path, gpu_option: str, gpu_value: str, expected_gpu_directives: list[str]
) -> None:
    """Dry-run preserves GPU request scope and one Ray task per node without submitting."""
    with patch(f"{MODULE_NAME}.curator_submit") as mock_curator_submit:
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "submit",
                "--dry-run",
                "-A",
                "test_account",
                "-p",
                "batch",
                gpu_option,
                gpu_value,
                "-J",
                "batch_job",
                "-N",
                "2",
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
    mock_curator_submit.assert_not_called()
    assert "#!/bin/bash" in result.output
    assert ":/remote_files:rw" in result.output
    assert "echo hello" in result.output
    assert "Job submitted with ID" not in result.output
    gpu_directives = [
        line
        for line in result.output.splitlines()
        if line.startswith(("#SBATCH --gpus=", "#SBATCH --gpus-per-node=", "#SBATCH --gres="))
    ]
    assert gpu_directives == expected_gpu_directives
    assert "#SBATCH --nodes=2" in result.output
    assert "#SBATCH --ntasks=2" in result.output
    assert "#SBATCH --ntasks-per-node=1" in result.output


def test_submit_dry_run_script_is_syntactically_valid_bash(tmp_path: pathlib.Path) -> None:
    """The rendered sbatch script should pass `bash -n` so template regressions fail here, not on Slurm."""
    if shutil.which("bash") is None:
        pytest.skip("bash is not available on this host")

    with patch(f"{MODULE_NAME}.curator_submit") as mock_curator_submit:
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "submit",
                "--dry-run",
                "-A",
                "test_account",
                "-p",
                "batch",
                "-G",
                "8",
                "-J",
                "batch_job",
                "-N",
                "2",
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
    mock_curator_submit.assert_not_called()

    # Slice from the shebang so any log preamble on stdout doesn't reach bash -n.
    script_start = result.output.find("#!/bin/bash")
    assert script_start != -1, "expected #!/bin/bash in dry-run output"
    script = result.output[script_start:]

    bash = shutil.which("bash")
    assert bash is not None

    completed = subprocess.run(  # noqa: S603 - validates generated sbatch syntax with bash -n; script is parsed, not executed.
        [bash, "-n"],
        input=script,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"bash -n rejected the rendered sbatch script:\n{completed.stderr}\n--- script ---\n{script}"
    )


@pytest.mark.parametrize(
    ("gpu_args", "expected_message"),
    [
        (["--gres", "gpu:8", "-G", "8"], "--gres and --gpus cannot be used together"),
        (["--gres", "gpu:8", "--gpus-per-node", "8"], "--gres and --gpus-per-node cannot be used together"),
        (["--gpus", "8", "--gpus-per-node", "8"], "--gpus and --gpus-per-node cannot be used together"),
    ],
)
def test_submit_rejects_conflicting_gpu_options(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], gpu_args: list[str], expected_message: str
) -> None:
    """Conflicting GPU request forms fail before uploading or submitting a job."""
    with patch(f"{MODULE_NAME}.curator_submit") as mock_curator_submit:
        result = runner.invoke(
            cosmos_curator,
            [
                "slurm",
                "submit",
                *gpu_args,
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

    assert result.exit_code == 2
    assert expected_message in result.output
    mock_curator_submit.assert_not_called()
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


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

    def test_curator_submit_uploads_extra_remote_files(self, mock_connection: Mock, job_spec: SlurmJobSpec) -> None:
        """Upload caller-provided files through the same SSH submission path."""
        conn = mock_connection.return_value
        job_spec.extra_remote_files = [
            ("SECRET=value\n", job_spec.remote_job_path / "secrets.env", 0o600),
            (
                "machine nvcr.io login $oauthtoken password token\n",
                job_spec.remote_job_path / "enroot/.credentials",
                0o600,
            ),
        ]

        failed_result = Mock()
        failed_result.exited = 1
        unexpected_exit = invoke.exceptions.UnexpectedExit(result=failed_result)
        success_result = Mock()
        success_result.stdout = "Submitted batch job 12345"
        conn.run.side_effect = [
            Mock(),  # ls call succeeds
            unexpected_exit,  # directory check should fail as expected (test -e)
            Mock(),  # mkdir job dir
            Mock(),  # chmod job dir
            Mock(),  # mkdir enroot dir
            Mock(),  # chmod enroot dir
            Mock(),  # chmod sbatch script
            Mock(),  # chmod prometheus service discovery script
            Mock(),  # chmod secrets.env
            Mock(),  # chmod enroot/.credentials
            success_result,  # sbatch command returns job ID
        ]

        job_id = curator_submit(job_spec)

        assert job_id == "12345"
        uploaded_paths = [call_args.kwargs["remote"] for call_args in conn.put.call_args_list]
        assert str(job_spec.remote_job_path / "secrets.env") in uploaded_paths
        assert str(job_spec.remote_job_path / "enroot/.credentials") in uploaded_paths
        commands = [call_args.args[0] for call_args in conn.run.call_args_list]
        assert f"mkdir -p {job_spec.remote_job_path / 'enroot'}" in commands

    def test_curator_submit_creates_extra_remote_file_dirs_before_mount_validation(
        self, monkeypatch: pytest.MonkeyPatch, job_spec: SlurmJobSpec
    ) -> None:
        """Generated extra remote file directories may be used as mount sources."""
        events: list[str] = []
        panoptes_dir = job_spec.remote_job_path / "panoptes_certs"
        job_spec.container.mounts.append(MountSpec(source=str(panoptes_dir), dest="/certs", mode="ro"))
        job_spec.extra_remote_files = [("", panoptes_dir / ".keep", 0o600)]

        connection = Mock()
        connection.run.return_value.stdout = "Submitted batch job 12345"

        def fake_remote_path_exists(_connection: Mock, path: pathlib.Path) -> bool:
            events.append(f"exists:{path}")
            if path == panoptes_dir:
                return "create_extra_dirs" in events
            return True

        monkeypatch.setattr(f"{MODULE_NAME}.connect", lambda _login_node, _username: connection)
        monkeypatch.setattr(
            f"{MODULE_NAME}.create_remote_job_path",
            lambda _connection, _job_spec: events.append("job_dir"),
        )
        monkeypatch.setattr(
            f"{MODULE_NAME}._create_extra_remote_file_dirs",
            lambda _connection, _job_spec: events.append("create_extra_dirs"),
        )
        monkeypatch.setattr(
            f"{MODULE_NAME}._upload_extra_remote_files",
            lambda _connection, _job_spec: events.append("upload_extra"),
        )
        monkeypatch.setattr(f"{MODULE_NAME}.remote_path_exists", fake_remote_path_exists)
        monkeypatch.setattr(f"{MODULE_NAME}.upload_text", lambda _connection, _files: events.append("upload_sbatch"))

        job_id = curator_submit(job_spec)

        assert job_id == "12345"
        assert events.index("create_extra_dirs") < events.index(f"exists:{panoptes_dir}")
        assert events.index("upload_extra") > events.index(f"exists:{panoptes_dir}")

    def test_curator_submit_skips_login_node_validation_for_node_local_mounts(
        self, monkeypatch: pytest.MonkeyPatch, job_spec: SlurmJobSpec
    ) -> None:
        """Node-local mounts are passed to sbatch without probing them on the login node."""
        node_local_source = "/raid/scratch/$USER/$SLURM_JOB_ID"
        job_spec.container.mounts.append(MountSpec(source=node_local_source, dest="/config/models", mode="rw"))
        job_spec.node_local_mount_sources = [node_local_source]
        checked_paths: list[pathlib.Path] = []

        connection = Mock()
        connection.run.return_value.stdout = "Submitted batch job 12345"

        def fake_remote_path_exists(_connection: Mock, path: pathlib.Path) -> bool:
            checked_paths.append(path)
            return True

        monkeypatch.setattr(f"{MODULE_NAME}.connect", lambda _login_node, _username: connection)
        monkeypatch.setattr(f"{MODULE_NAME}.create_remote_job_path", lambda _connection, _job_spec: None)
        monkeypatch.setattr(f"{MODULE_NAME}._create_extra_remote_file_dirs", lambda _connection, _job_spec: None)
        monkeypatch.setattr(f"{MODULE_NAME}._upload_extra_remote_files", lambda _connection, _job_spec: None)
        monkeypatch.setattr(f"{MODULE_NAME}.remote_path_exists", fake_remote_path_exists)
        monkeypatch.setattr(f"{MODULE_NAME}.upload_text", lambda _connection, _files: None)

        job_id = curator_submit(job_spec)

        assert job_id == "12345"
        assert pathlib.Path(node_local_source) not in checked_paths
        assert MountSpec(source=node_local_source, dest="/config/models", mode="rw") in job_spec.container.mounts

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
        with pytest.raises(ValueError, match="`/src` must have between 2 and 3 colon-separated parts"):
            MountSpec.from_str("/src")

    def test_mount_spec_from_str_with_too_many_parts(self) -> None:
        """Test that the mount spec raises an error if the format has too many parts."""
        with pytest.raises(ValueError, match="`/src:/dst:ro:extra` must have between 2 and 3 colon-separated parts"):
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
        with pytest.raises(ValueError, match=r"(?i).*must have between 2 and 3 colon-separated parts.*"):
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
def test_head_node_is_head_node(
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
