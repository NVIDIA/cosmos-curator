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
"""Test nvcf base functionality."""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import typer
from _pytest.monkeypatch import MonkeyPatch
from typer.testing import CliRunner

from cosmos_curator.client.nvcf_cli.ncf.common import base_callback  # type: ignore[import-untyped]
from cosmos_curator.client.nvcf_cli.ncf.common.nvcf_base import (  # type: ignore[import-untyped]
    NvcfBase,
    cc_client_instances,
    register_instance,
)

runner = CliRunner()

_TIMEOUT = 15


def test_base_callback(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that base_callback function sets up config defaults.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The tmp_path object.

    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Define a temporary directory for the config file
    config_dir = tmp_path / ".cosmos_curator"
    config_dir.mkdir(parents=True, exist_ok=True)
    fname = config_dir / "config.json"
    Path.open(fname, "w")

    fake_ctx = MagicMock()
    fake_ctx.command.name = "config"
    fake_ctx.obj = {
        "url": None,
        "nvcf_url": None,
        "key": "BASE",
        "org": "NVIDIA",
        "timeout": None,
        "config": None,
        "nvcfHdl": None,
        "test": True,
    }

    base_callback(ctx=fake_ctx)
    # assert that values have been changed to defaults
    assert fake_ctx.obj["url"] == "https://api.ngc.nvidia.com"
    assert fake_ctx.obj["nvcf_url"] == "https://api.nvcf.nvidia.com"
    assert fake_ctx.obj["key"] is None
    assert fake_ctx.obj["org"] is None
    assert fake_ctx.obj["timeout"] == _TIMEOUT


def test_base_callback_errors(caplog: pytest.LogCaptureFixture) -> None:
    """Test that base_callback function errors.

    Args:
        caplog: The caplog object.

    """
    fake_ctx = MagicMock()
    fake_ctx.command.name = None
    fake_ctx.obj = {
        "url": None,
        "nvcf_url": None,
        "key": "BASE",
        "org": "NVIDIA",
        "timeout": None,
        "config": None,
        "nvcfHdl": None,
    }

    with caplog.at_level(logging.ERROR), pytest.raises(typer.Exit) as e:  # noqa: PT012
        base_callback(ctx=fake_ctx)
        assert "FATAL: Instance None not registered" in caplog.text
    assert e.value.exit_code == 1


def test_base_callback_no_config(caplog: pytest.LogCaptureFixture) -> None:
    """Test base_callback when config is None and instance is not 'config'.

    Args:
        caplog: The caplog object.

    """
    # Create a mock class that will be returned when ins_type is called
    mock_instance = MagicMock()
    mock_instance.config = None
    mock_instance.exe = "test_exe"
    mock_instance.logger = MagicMock()

    mock_ins_type = MagicMock(return_value=mock_instance)

    register_instance("test_instance", "Test instance", mock_ins_type, typer.Typer())

    fake_ctx = MagicMock()
    fake_ctx.command.name = "test_instance"
    fake_ctx.obj = {
        "url": "https://api.ngc.nvidia.com",
        "nvcf_url": "https://api.nvcf.nvidia.com",
        "key": None,
        "org": None,
        "timeout": 15,
        "config": None,
        "nvcfHdl": None,
    }

    with caplog.at_level(logging.ERROR), pytest.raises(typer.Exit) as e:  # noqa: PT012
        base_callback(ctx=fake_ctx)
        assert "No Configurations found, Please run 'test_exe nvcf config set' to create configuration" in caplog.text
    assert e.value.exit_code == 1


def test_get_hf_token_from_config(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that get_hf_token_from_config function gets the token from the config file.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The tmp_path object.

    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_base = NvcfBase(url="", nvcf_url="", key="", org="", team="", timeout=15)

    # Test that None is returned if the file is not found
    token = nvcf_base.get_hf_token_from_config()
    assert token is None

    # Define fake config.yaml file
    config_dir = tmp_path / ".config/cosmos_curator"
    config_dir.mkdir(parents=True, exist_ok=True)
    fname = config_dir / "config.yaml"
    with Path.open(fname, "w") as fc:
        json.dump({"huggingface": {"api_key": "fake_token"}}, fc)

    nvcf_base = NvcfBase(url="", nvcf_url="", key="", org="", team="", timeout=15)
    token = nvcf_base.get_hf_token_from_config()

    assert token == "fake_token"  # noqa: S105


def test_load_config(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that load_config function loads the config file.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The tmp_path object.

    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_base = NvcfBase(url="", nvcf_url="", key="", org="", team="", timeout=15)

    # Test that None is returned if the file is not found
    config = nvcf_base.load_config()
    assert config is None

    config_dir = tmp_path / ".config/cosmos_curator"
    config_dir.mkdir(parents=True, exist_ok=True)
    fname = config_dir / "client.json"
    with Path.open(fname, "w") as fc:
        json.dump(
            {
                "url": "https://api.ngc.nvidia.com",
                "key": "FAKEKEY",
                "org": "FAKEORG",
                "nvcf_url": "https://api.nvcf.nvidia.com",
                "backend": None,
                "gpu": None,
                "instance": None,
                "timeout": 15,
            },
            fc,
        )

    config = nvcf_base.load_config()

    assert config == {
        "url": "https://api.ngc.nvidia.com",
        "key": "FAKEKEY",
        "org": "FAKEORG",
        "nvcf_url": "https://api.nvcf.nvidia.com",
        "backend": None,
        "gpu": None,
        "instance": None,
        "timeout": 15,
    }


def test_save_config() -> None:
    """Test that save_config function saves the config file."""
    nvcf_base = NvcfBase(url="", nvcf_url="", key="", org="", team="", timeout=15)

    config = nvcf_base.save_config(
        url="https://api.ngc.nvidia.com",
        nvcf_url="https://api.nvcf.nvidia.com",
        key="FAKEKEY",
        org="FAKEORG",
        team="FAKETEAM",
        backend="FAKEBACKEND",
        instance="FAKEINSTANCE",
        gpu="FAKEGPU",
        timeout=15,
    )

    assert config == {
        "url": "https://api.ngc.nvidia.com",
        "key": "FAKEKEY",
        "org": "FAKEORG",
        "team": "FAKETEAM",
        "nvcf_url": "https://api.nvcf.nvidia.com",
        "backend": "FAKEBACKEND",
        "instance": "FAKEINSTANCE",
        "gpu": "FAKEGPU",
        "timeout": 15,
    }


def test_get_cluster() -> None:
    """Test that get_cluster function gets the cluster configuration from the context or config."""
    nvcf_base = NvcfBase(url="", nvcf_url="", key="", org="", team="", timeout=15)

    # Test none context first
    ctx_none = MagicMock()
    ctx_none.obj.get.return_value = None

    success, backend, gpu, instance = nvcf_base.get_cluster(ctx_none, None, None, None)
    assert not success
    assert backend is None
    assert gpu is None
    assert instance is None

    # Test context with config
    ctx = MagicMock()
    ctx.obj = {"config": {"backend": "FAKEBACKEND", "gpu": "FAKEGPU", "instance": "FAKEINSTANCE"}}

    # Test with context and no backend, gpu, instance
    success, backend, gpu, instance = nvcf_base.get_cluster(ctx, None, None, None)
    assert success
    assert backend == "FAKEBACKEND"
    assert gpu == "FAKEGPU"
    assert instance == "FAKEINSTANCE"

    # Test with context and backend, gpu, instance
    success, backend, gpu, instance = nvcf_base.get_cluster(ctx, "FAKEBACKEND", "FAKEGPU", "FAKEINSTANCE")
    assert success
    assert backend == "FAKEBACKEND"
    assert gpu == "FAKEGPU"
    assert instance == "FAKEINSTANCE"


def test_register_instance() -> None:
    """Test that register_instance function registers a new instance."""
    ins_name = "FAKEINSTANCE"
    ins_help = "FAKEHELP"
    ins_type = NvcfBase
    ins_app = MagicMock()

    register_instance(ins_name, ins_help, ins_type, ins_app)

    instances = cc_client_instances()
    assert instances[ins_name] == {"help": ins_help, "type": ins_type, "app": ins_app}

    # Test that the instance is not registered again
    register_instance(ins_name, ins_help, ins_type, ins_app)
    instances = cc_client_instances()
    assert instances[ins_name] == {"help": ins_help, "type": ins_type, "app": ins_app}
