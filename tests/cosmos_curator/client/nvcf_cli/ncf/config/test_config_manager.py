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
"""Test nvcf config commands."""

import json
import logging
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch
from typer.testing import CliRunner

from cosmos_curator.client.cli import cosmos_curator  # type: ignore[import-untyped]
from cosmos_curator.client.nvcf_cli.ncf.common import NvcfBase  # type: ignore[import-untyped]

runner = CliRunner()
_TIMEOUT = 30


def test_empty_get(monkeypatch: MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Test that correct message is displayed after attempting get without set.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The tmp_path object.
        caplog: The caplog object.

    """
    with caplog.at_level(logging.INFO):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        # Define temp directory for config file
        config_dir = tmp_path / ".cosmos_curator"
        config_dir.mkdir(parents=True, exist_ok=True)
        fname = config_dir / NvcfBase.CLIENT_NAME
        Path.open(fname, "w")
        # Check that all logger messages show appropriate warnings and errors
        args = [
            "nvcf",
            "config",
            "get",
        ]

        runner.invoke(cosmos_curator, args)
        assert "ERROR" in caplog.text
        assert "nvcf config set' to create configuration" in caplog.text


def test_nvcf_config_set(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test nvcf config set command works with backend information.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The tmp_path object.

    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Define temp directory for config file
    config_dir = tmp_path / ".config/cosmos_curator"
    config_dir.mkdir(parents=True, exist_ok=True)
    fname = config_dir / NvcfBase.CLIENT_NAME
    Path.open(fname, "w")

    # Set all args for nvcf config
    args = [
        "nvcf",
        "config",
        "set",
        "--key",
        "TESTKEY",
        "--org",
        "TESTORG",
        "--url",
        "testurl.com",
        "--nvcf-url",
        "testnvcfurl.com",
        "--timeout",
        "30",
        "--backend",
        "TESTBACKEND",
        "--gpu",
        "TESTGPU",
        "--instance",
        "TESTINSTANCE",
    ]
    runner.invoke(cosmos_curator, args)
    runner.invoke(cosmos_curator, ["nvcf", "config", "get"])
    content = fname.read_text()
    data = json.loads(content)
    # Assert that all have been set correctly
    assert data["key"] == "TESTKEY"
    assert data["org"] == "TESTORG"
    assert data["url"] == "testurl.com"
    assert data["nvcf_url"] == "testnvcfurl.com"
    assert data["timeout"] == _TIMEOUT
    assert data["backend"] == "TESTBACKEND"
    assert data["instance"] == "TESTINSTANCE"
    assert data["gpu"] == "TESTGPU"


def test_nvcf_config_set_in_parts(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test nvcf config set command works in multiple stages.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The tmp_path object.

    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Define temp directory for config file
    config_dir = tmp_path / ".config/cosmos_curator"
    config_dir.mkdir(parents=True, exist_ok=True)
    fname = config_dir / NvcfBase.CLIENT_NAME
    Path.open(fname, "w")

    # Set all args for nvcf config
    args = [
        "nvcf",
        "config",
        "set",
        "--key",
        "TESTKEY",
        "--org",
        "TESTORG",
    ]
    runner.invoke(cosmos_curator, args)
    runner.invoke(cosmos_curator, ["nvcf", "config", "get"])
    content = fname.read_text()
    data = json.loads(content)

    # Assert defaults, org, and key have been set correctly
    assert data["key"] == "TESTKEY"
    assert data["org"] == "TESTORG"
    # Change args to reset defaults
    args = [
        "nvcf",
        "config",
        "set",
        "--key",
        "TESTKEY",
        "--org",
        "TESTORG",
        "--url",
        "testurl.com",
        "--nvcf-url",
        "testnvcfurl.com",
        "--timeout",
        "30",
        "--backend",
        "TESTBACKEND",
        "--gpu",
        "TESTGPU",
        "--instance",
        "TESTINSTANCE",
    ]
    runner.invoke(cosmos_curator, args)
    runner.invoke(cosmos_curator, ["nvcf", "config", "get"])
    content = fname.read_text()
    data = json.loads(content)
    # Assert that defaults have been changed correctly
    assert data["key"] == "TESTKEY"
    assert data["url"] == "testurl.com"
    assert data["nvcf_url"] == "testnvcfurl.com"
    assert data["timeout"] == _TIMEOUT
    assert data["backend"] == "TESTBACKEND"
    assert data["instance"] == "TESTINSTANCE"
    assert data["gpu"] == "TESTGPU"


def test_nvcf_config_fail_set(monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """Test nvcf config set command throws error if missing --key or --org.

    Args:
        caplog: The caplog object.
        monkeypatch: The MonkeyPatch object.
        tmp_path: A temporary Path for tests.

    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Remove env vars for testing
    monkeypatch.delenv("NGC_NVCF_API_KEY", raising=False)
    monkeypatch.delenv("NGC_NVCF_ORG", raising=False)
    config_dir = tmp_path / ".config/cosmos_curator"
    config_dir.mkdir(parents=True, exist_ok=True)
    fname = config_dir / NvcfBase.CLIENT_NAME
    Path.open(fname, "w")
    with caplog.at_level(logging.INFO):
        # Check that all logger messages show appropriate warnings and errors
        args = [
            "nvcf",
            "config",
            "set",
        ]
        runner.invoke(cosmos_curator, args)
        assert "Missing '--key' or '--org'" in caplog.text
        caplog.clear()
        args = ["nvcf", "config", "set", "--org", "TESTORG"]
        runner.invoke(cosmos_curator, args)
        assert "Missing '--key' or '--org'" in caplog.text
