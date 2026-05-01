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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import typer
from _pytest.monkeypatch import MonkeyPatch
from typer.testing import CliRunner

from cosmos_curator.client.cli import cosmos_curator  # type: ignore[import-untyped]
from cosmos_curator.client.local_cli.launch_local import (  # type: ignore[import-untyped]
    _get_config_file_mount_strings,
    _get_identity_mounts,
    _parse_extra_volumes,
    _verify_local_path_exists,
)

runner = CliRunner()


def test_launch_command() -> None:
    """Test that docker run command forms."""
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
    with patch("cosmos_curator.client.local_cli.launch_local.subprocess.call") as mock_call:
        mock_call.return_value = 0

        result = runner.invoke(cosmos_curator, args)

        assert result.exit_code == 0

        # Extract the actual command that would be run
        called_args = mock_call.call_args[0][0]

    # check docker command is formed
    assert called_args[:3] == ["docker", "run", "--rm"]


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
