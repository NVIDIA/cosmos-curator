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
"""Test cosmos-curator local commands."""

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import typer
from _pytest.monkeypatch import MonkeyPatch
from loguru import logger
from typer.testing import CliRunner

from cosmos_curator.client.cli import cosmos_curator  # type: ignore[import-untyped]
from cosmos_curator.client.environment import CONTAINER_PATHS_CODE_DIR  # type: ignore[import-untyped]
from cosmos_curator.client.local_cli.launch_local import (  # type: ignore[import-untyped]
    _get_config_file_mount_strings,
    _get_identity_mounts,
    _get_python_version_from_pixi_toml,
    _parse_extra_volumes,
    _verify_local_path_exists,
)
from cosmos_curator.client.utils.container_launch import SLIM_IMAGE_WARMUP_COMMAND  # type: ignore[import-untyped]

runner = CliRunner()


@pytest.mark.parametrize(("stdout_tty", "stderr_tty"), [(True, True), (True, False), (False, True), (False, False)])
def test_launch_command(*, stdout_tty: bool, stderr_tty: bool) -> None:
    """Preserve terminal progress and redirected streams without exposing stdin."""
    args = [
        "local",
        "launch",
        "--image-name",
        "cosmos-curator",
        "--image-tag",
        "hello-world",
        "--curator-path",
        ".",
        "--",
        "python3",
        "-m",
        "cosmos_curator.pipelines.examples.hello_world_pipeline",
    ]
    with (
        patch("cosmos_curator.client.local_cli.launch_local.subprocess.call") as mock_call,
        patch("cosmos_curator.client.local_cli.launch_local.sys") as mock_sys,
    ):
        mock_call.return_value = 0
        mock_sys.stdout.isatty.return_value = stdout_tty
        mock_sys.stderr.isatty.return_value = stderr_tty

        result = runner.invoke(cosmos_curator, args)

        assert result.exit_code == 0

        # Extract the actual command that would be run
        called_args = mock_call.call_args[0][0]

    # check docker command is formed
    assert called_args[:3] == ["docker", "run", "--rm"]
    assert "--network=host" not in called_args
    assert "--cap-add=SYS_ADMIN" not in called_args
    assert "COSMOS_CURATOR_RAY_IO_SLOTS_PER_NODE=16" in called_args
    assert ("-t" in called_args) is (stdout_tty and stderr_tty)
    assert "-i" not in called_args
    assert "-it" not in called_args
    assert "PYTHONUNBUFFERED=1" in called_args
    assert called_args[-1].startswith("exec < /dev/null && ")


def test_launch_command_forwards_local_ray_io_capacity() -> None:
    """The containerized recipe can size the local Ray node's logical IO resource."""
    args = [
        "local",
        "launch",
        "--ray-io-slots-per-node",
        "7",
        "--",
        "echo",
        "hello",
    ]
    with patch("cosmos_curator.client.local_cli.launch_local.subprocess.call") as mock_call:
        mock_call.return_value = 0

        result = runner.invoke(cosmos_curator, args)

    assert result.exit_code == 0
    assert "COSMOS_CURATOR_RAY_IO_SLOTS_PER_NODE=7" in mock_call.call_args.args[0]


def test_launch_command_with_docker_network() -> None:
    """Test that --docker-network forwards the selected Docker network."""
    args = [
        "local",
        "launch",
        "--docker-network",
        "host",
        "--",
        "echo",
        "hello",
    ]
    with patch("cosmos_curator.client.local_cli.launch_local.subprocess.call") as mock_call:
        mock_call.return_value = 0

        result = runner.invoke(cosmos_curator, args)

        assert result.exit_code == 0
        called_args = mock_call.call_args[0][0]

    assert "--network=host" in called_args


def test_launch_command_with_empty_docker_network_omits_network_arg() -> None:
    """Test that a blank Docker network is treated as unset."""
    args = [
        "local",
        "launch",
        "--docker-network",
        "",
        "--",
        "echo",
        "hello",
    ]
    with patch("cosmos_curator.client.local_cli.launch_local.subprocess.call") as mock_call:
        mock_call.return_value = 0

        result = runner.invoke(cosmos_curator, args)

        assert result.exit_code == 0
        called_args = mock_call.call_args[0][0]

    assert all(not arg.startswith("--network=") for arg in called_args)


def _make_workspace(root: Path) -> Path:
    """Create a minimal curator checkout with installed pixi environments."""
    (root / ".pixi").mkdir(parents=True)
    (root / "cosmos_curator" / "pipelines").mkdir(parents=True)
    (root / "pixi.toml").write_text('[dependencies]\npython = "==3.13.14"\n')
    return root.resolve()


def _launched_docker_args(args: list[str]) -> list[str]:
    """Return the docker command a local launch runs for ``echo hello``."""
    with patch("cosmos_curator.client.local_cli.launch_local.subprocess.call") as mock_call:
        mock_call.return_value = 0
        result = runner.invoke(cosmos_curator, ["local", "launch", *args, "--", "echo", "hello"])
    assert result.exit_code == 0, result.output
    return list(mock_call.call_args.args[0])


def _volumes(docker_args: list[str]) -> list[str]:
    return [docker_args[index + 1] for index, arg in enumerate(docker_args) if arg == "-v"]


def test_host_workspace_runs_at_its_host_path(tmp_path: Path) -> None:
    """Pixi rejects environments found anywhere but where it installed them, so a workspace keeps its path."""
    workspace = _make_workspace(tmp_path / "ws")

    docker_args = _launched_docker_args(["--curator-path", str(workspace), "--pixi-path", str(workspace)])

    volumes = _volumes(docker_args)
    assert f"{workspace}:{workspace}" in volumes
    assert docker_args[docker_args.index("-w") + 1] == str(workspace)
    assert f"{workspace / 'cosmos_curator'}:{CONTAINER_PATHS_CODE_DIR}/cosmos_curator" in volumes
    assert not any(volume.endswith(":/opt/cosmos-curator/.pixi") for volume in volumes)
    assert not any(volume.startswith(f"{workspace / 'cosmos_curator'}:{workspace}") for volume in volumes)
    # Host pixi installs and updates the workspace's environments.
    assert SLIM_IMAGE_WARMUP_COMMAND not in docker_args[-1]
    assert docker_args[-1] == "exec < /dev/null && echo hello"


def test_curator_path_code_covers_the_host_workspace(tmp_path: Path) -> None:
    """Code from another checkout replaces the workspace's where commands run and where the image keeps code."""
    workspace = _make_workspace(tmp_path / "ws")
    code = _make_workspace(tmp_path / "code")

    volumes = _volumes(_launched_docker_args(["--curator-path", str(code), "--pixi-path", str(workspace)]))

    for code_dir in (CONTAINER_PATHS_CODE_DIR, workspace):
        assert f"{code / 'cosmos_curator'}:{code_dir}/cosmos_curator" in volumes
        assert f"{code / 'tests' / 'cosmos_curator'}:{code_dir}/tests/cosmos_curator" in volumes


_INSTALLED_XENNA = Path(".pixi/envs/default/lib/python3.13/site-packages/cosmos_xenna")


def test_xenna_overlay_uses_the_workspace_environment(tmp_path: Path) -> None:
    """The workspace's environment, not the code checkout's pixi.toml, decides where cosmos_xenna is installed."""
    workspace = _make_workspace(tmp_path / "ws")
    (workspace / _INSTALLED_XENNA).mkdir(parents=True)
    code = _make_workspace(tmp_path / "code")
    (code / "pixi.toml").write_text('[dependencies]\npython = "==3.14.0"\n')
    (code / "cosmos-xenna" / "cosmos_xenna").mkdir(parents=True)

    volumes = _volumes(
        _launched_docker_args(["--curator-path", str(code), "--pixi-path", str(workspace), "--mount-xenna"])
    )

    xenna = workspace / _INSTALLED_XENNA / "pipelines"
    assert f"{code / 'cosmos-xenna' / 'cosmos_xenna' / 'pipelines'}:{xenna}" in volumes


def test_xenna_overlay_maps_a_bare_pixi_directory_to_the_image_path(tmp_path: Path) -> None:
    """A bare .pixi directory is searched on the host and mounted where the image keeps its environments."""
    code = _make_workspace(tmp_path / "code")
    (code / "cosmos-xenna" / "cosmos_xenna").mkdir(parents=True)
    envs = tmp_path / "envs"
    (envs / _INSTALLED_XENNA).mkdir(parents=True)

    volumes = _volumes(_launched_docker_args(["--curator-path", str(code), "--pixi-path", str(envs), "--mount-xenna"]))

    xenna = CONTAINER_PATHS_CODE_DIR / _INSTALLED_XENNA / "pipelines"
    assert f"{code / 'cosmos-xenna' / 'cosmos_xenna' / 'pipelines'}:{xenna}" in volumes


def test_xenna_overlay_skips_a_host_environment_without_cosmos_xenna(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """Docker would create the missing mount points in the host environment as root."""
    workspace = _make_workspace(tmp_path / "ws")
    (workspace / "cosmos-xenna" / "cosmos_xenna").mkdir(parents=True)

    # Other tests can remove Loguru's default sinks; bind the captured stderr explicitly.
    sink_id = logger.add(sys.stderr, format="{message}", level="WARNING")
    try:
        volumes = _volumes(_launched_docker_args(["--pixi-path", str(workspace), "--mount-xenna"]))
    finally:
        logger.remove(sink_id)

    assert not any("site-packages" in volume for volume in volumes)
    assert "Not mounting cosmos_xenna" in capfd.readouterr().err


@pytest.mark.parametrize("python_spec", ["==3.13.14", ">=3.13.1,<3.14", "3.13.*"])
def test_python_version_from_pixi_toml(tmp_path: Path, python_spec: str) -> None:
    """Pins, ranges, and wildcards all name the image environment's python3.13 directory."""
    (tmp_path / "pixi.toml").write_text(f'[dependencies]\npython = "{python_spec}"\n')

    assert _get_python_version_from_pixi_toml(tmp_path) == "3.13"


def test_bare_pixi_directory_replaces_the_images(tmp_path: Path) -> None:
    """Shared storage for container environments keeps the image's paths and the slim install."""
    (tmp_path / ".pixi").mkdir()

    docker_args = _launched_docker_args(["--pixi-path", str(tmp_path)])

    assert f"{tmp_path / '.pixi'}:{CONTAINER_PATHS_CODE_DIR / '.pixi'}" in _volumes(docker_args)
    assert "-w" not in docker_args
    assert SLIM_IMAGE_WARMUP_COMMAND in docker_args[-1]


def test_host_workspace_under_home_gets_a_real_mount_point(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Docker follows links on mount paths, so links that earlier launches left there are replaced."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("COSMOS_CURATOR_LOCAL_HOME_DIR", raising=False)
    workspace = _make_workspace(home / "src" / "ws")
    mount_point = home / ".cache" / "cosmos-curator-home" / "src" / "ws"
    mount_point.parent.mkdir(parents=True)
    mount_point.symlink_to("/opt/cosmos-curator")

    docker_args = _launched_docker_args(["--pixi-path", str(workspace)])

    assert f"{workspace}:{workspace}" in _volumes(docker_args)
    assert mount_point.is_dir()
    assert not mount_point.is_symlink()


def test_verify_local_path_exists(monkeypatch: MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Test verify_local_path_exists function.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The tmp_path object.
        caplog: LogCaptureFixture object.

    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    mock_path_one = tmp_path / "test1"
    mock_path_one.touch()
    mock_path_two = tmp_path / "test2"
    mock_path_two.touch()
    mock_path_fail = tmp_path / "failure"

    with caplog.at_level(logging.DEBUG):
        mock_paths = [Path(str(mock_path_one)), Path(str(mock_path_two))]
        mock_paths_fail = [Path(str(mock_path_fail))]
        _verify_local_path_exists(mock_paths)

        assert "ERROR" not in caplog.text

        with pytest.raises(SystemExit):
            _verify_local_path_exists(mock_paths_fail)


def test_get_identity_mounts_skips_baked_uid(tmp_path: Path) -> None:
    """UIDs already in the image's /etc/passwd skip synthesis."""
    with patch("cosmos_curator.client.local_cli.launch_local.os.getuid", return_value=1000):
        assert _get_identity_mounts(tmp_path) == []
    assert not (tmp_path / "etc").exists()


def test_get_identity_mounts_synthesizes_for_ad_uid(tmp_path: Path) -> None:
    """High AD/SSSD-style UIDs get a synthesized minimal passwd/group."""
    with (
        patch("cosmos_curator.client.local_cli.launch_local.os.getuid", return_value=1776709216),
        patch("cosmos_curator.client.local_cli.launch_local.os.getgid", return_value=1776709216),
        patch(
            "cosmos_curator.client.local_cli.launch_local.pwd.getpwuid",
            return_value=SimpleNamespace(
                pw_name="ad_user",
                pw_uid=1776709216,
                pw_gid=1776709216,
                pw_gecos="AD User",
                pw_shell="/bin/bash",
            ),
        ),
        patch(
            "cosmos_curator.client.local_cli.launch_local.grp.getgrgid",
            return_value=SimpleNamespace(gr_name="ad_group", gr_gid=1776709216),
        ),
        patch("cosmos_curator.client.local_cli.launch_local.Path.home", return_value=Path("/home/ad_user")),
    ):
        result = _get_identity_mounts(tmp_path)

    assert f"{tmp_path / 'etc' / 'passwd'}:/etc/passwd:ro" in result
    assert f"{tmp_path / 'etc' / 'group'}:/etc/group:ro" in result
    assert "USER=ad_user" in result
    assert "LOGNAME=ad_user" in result

    passwd_text = (tmp_path / "etc" / "passwd").read_text()
    group_text = (tmp_path / "etc" / "group").read_text()
    assert "ad_user:x:1776709216:1776709216:AD User:/home/ad_user:/bin/bash\n" in passwd_text
    assert "root:x:0:0:root:/root:/bin/bash\n" in passwd_text
    assert "ad_group:x:1776709216:ad_user\n" in group_text
    assert "root:x:0:\n" in group_text


def test_get_config_file_mount_string(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test get_config_file_mount_string function.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The tmp_path object.

    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    mock_file = tmp_path / "mock_config.yaml"
    mock_file.write_text("anything")

    # check correctness when file exists
    with patch("cosmos_curator.client.local_cli.launch_local.LOCAL_COSMOS_CURATOR_CONFIG_FILE", mock_file):
        result = _get_config_file_mount_strings(is_model_cli=True)

    assert str(mock_file) in result[1]

    # check correctness when file does not exist
    mock_file_fail = tmp_path / "mock_fail.yaml"

    with patch("cosmos_curator.client.local_cli.launch_local.LOCAL_COSMOS_CURATOR_CONFIG_FILE", mock_file_fail):  # noqa: SIM117
        with pytest.raises(SystemExit):
            _get_config_file_mount_strings(is_model_cli=True)


# ---------------------------------------------------------------------------
# _parse_extra_volumes
# ---------------------------------------------------------------------------
class TestParseExtraVolumes:
    """Tests for the --extra-volumes parsing and validation."""

    def test_empty_string_returns_empty_list(self) -> None:
        """Empty string yields an empty volume list."""
        assert _parse_extra_volumes("") == []

    def test_single_volume(self) -> None:
        """Single host:container pair is parsed as one entry."""
        assert _parse_extra_volumes("/host:/container") == ["/host:/container"]

    def test_multiple_volumes(self) -> None:
        """Comma-separated pairs split into multiple mount strings."""
        result = _parse_extra_volumes("/a:/b,/c:/d")
        assert result == ["/a:/b", "/c:/d"]

    def test_volume_with_mode(self) -> None:
        """Read-only mode suffix on a mount is preserved."""
        assert _parse_extra_volumes("/host:/container:ro") == ["/host:/container:ro"]

    def test_whitespace_stripped(self) -> None:
        """Leading and trailing whitespace around entries is stripped."""
        result = _parse_extra_volumes(" /a:/b , /c:/d ")
        assert result == ["/a:/b", "/c:/d"]

    def test_trailing_comma_ignored(self) -> None:
        """Trailing comma after the last mount does not add an empty entry."""
        result = _parse_extra_volumes("/a:/b,")
        assert result == ["/a:/b"]

    def test_missing_colon_raises(self) -> None:
        """Value without a colon raises BadParameter."""
        with pytest.raises(typer.BadParameter, match="Invalid volume mount"):
            _parse_extra_volumes("no-colon-here")

    def test_too_many_colons_raises(self) -> None:
        """More than two colons in a mount raises BadParameter."""
        with pytest.raises(typer.BadParameter, match="Invalid volume mount"):
            _parse_extra_volumes("/a:/b:ro:extra")

    def test_launch_command_with_extra_volumes(self) -> None:
        """Verify --extra-volumes produces -v flags in the docker command."""
        args = [
            "local",
            "launch",
            "--image-name",
            "cosmos-curator",
            "--image-tag",
            "test",
            "--curator-path",
            ".",
            "--extra-volumes",
            "/models:/config/models,/data:/workspace/input",
            "--",
            "echo",
            "hello",
        ]
        with patch("cosmos_curator.client.local_cli.launch_local.subprocess.call") as mock_call:
            mock_call.return_value = 0
            result = runner.invoke(cosmos_curator, args)
            assert result.exit_code == 0

            docker_cmd = mock_call.call_args[0][0]

        volume_pairs = [(docker_cmd[i], docker_cmd[i + 1]) for i in range(len(docker_cmd) - 1) if docker_cmd[i] == "-v"]
        mounted_volumes = [pair[1] for pair in volume_pairs]
        assert "/models:/config/models" in mounted_volumes
        assert "/data:/workspace/input" in mounted_volumes
