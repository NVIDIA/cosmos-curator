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
"""Test Asset Manager functionality."""

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from _pytest.monkeypatch import MonkeyPatch
from rich.table import Table
from typer.testing import CliRunner

from cosmos_curator.client.cli import cosmos_curator  # type: ignore[import-untyped]
from cosmos_curator.client.nvcf_cli.ncf.asset import AssetManager
from cosmos_curator.client.nvcf_cli.ncf.common import NotFoundError, NvcfBase, NVCFResponse

runner = CliRunner()

_USAGE_ERROR = 2
_BAD_EXIT_CODE = 1


@pytest.mark.parametrize(
    ("response", "exception"),
    [
        (None, RuntimeError),
        (NVCFResponse({"status": 500, "detail": "test-detail"}), RuntimeError),
        (NVCFResponse({"status": 200}), RuntimeError),
        ("Exception", RuntimeError),
    ],
)
def test_upload_asset_failure(response: NVCFResponse, exception: type[Exception]) -> None:
    """Test upload asset failure scenarios."""
    asset_manager = AssetManager(url="", nvcf_url="", key="", org="", team="", timeout=15)

    asset_manager.nvcf_api_hdl.post = MagicMock(return_value=response)

    if response == "Exception":
        asset_manager.nvcf_api_hdl.post = MagicMock(side_effect=OSError("test-exception"))

    with pytest.raises(exception):
        asset_manager.upload_asset(src_path=Path("nonexistent.txt"), desc="test", retries=1)


def test_upload_asset_success(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test upload asset success."""
    asset_manager = AssetManager(url="", nvcf_url="", key="", org="", team="", timeout=15)
    asset_manager.nvcf_api_hdl.post = MagicMock(
        return_value=NVCFResponse(
            {"status": 200, "assetId": "test-asset-id", "uploadUrl": "https://example.com/upload"}
        )
    )
    asset_manager.do_upload = MagicMock()

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    tmp_file = tmp_path / "test.txt"
    tmp_file.write_text("test")

    result = asset_manager.upload_asset(src_path=tmp_file, desc="test", retries=3)
    assert result["AssetId"] == "test-asset-id"


@pytest.mark.parametrize(
    ("response", "exception"),
    [
        (None, RuntimeError),
        (NVCFResponse({"status": 500, "detail": "test-detail"}), RuntimeError),
        (NVCFResponse({"status": 404}), NotFoundError),
        (NVCFResponse({"status": 200}), RuntimeError),
        ("Exception", RuntimeError),
    ],
)
def test_delete_asset(response: NVCFResponse, exception: type[Exception]) -> None:
    """Test delete asset."""
    asset_manager = AssetManager(url="", nvcf_url="", key="", org="", team="", timeout=15)
    asset_manager.nvcf_api_hdl.delete = MagicMock(return_value=response)

    if response == "Exception":
        asset_manager.nvcf_api_hdl.delete = MagicMock(side_effect=Exception("test-exception"))

    with pytest.raises(exception):
        asset_manager.delete_asset(asset_id="test-asset-id")


@pytest.mark.parametrize(
    ("response", "exception"),
    [
        (None, RuntimeError),
        (NVCFResponse({"status": 500, "detail": "test-detail"}), RuntimeError),
        (NVCFResponse({"status": 200}), NotFoundError),
        ("Exception", RuntimeError),
    ],
)
def test_list_all(response: NVCFResponse, exception: type[Exception]) -> None:
    """Test list assets."""
    asset_manager = AssetManager(url="", nvcf_url="", key="", org="", team="", timeout=15)
    asset_manager.nvcf_api_hdl.get = MagicMock(return_value=response)

    if response == "Exception":
        asset_manager.nvcf_api_hdl.get = MagicMock(side_effect=Exception("test-exception"))

    with pytest.raises(exception):
        asset_manager.list_all()


def test_list_all_success() -> None:
    """Test list assets success."""
    asset_manager = AssetManager(url="", nvcf_url="", key="", org="", team="", timeout=15)
    asset_manager.nvcf_api_hdl.get = MagicMock(
        return_value=NVCFResponse(
            {"status": 200, "assets": [{"assetId": "test-asset-id", "desc": "test", "size": 100}]}
        )
    )
    result = asset_manager.list_all()

    # Verify it's a Rich Table object
    assert isinstance(result, Table)
    assert result.title == "Assets"

    # Verify the table has the expected column
    assert len(result.columns) == _BAD_EXIT_CODE
    assert result.columns[0].header == "AssetId"


@pytest.mark.parametrize(
    ("response", "exception"),
    [
        (None, RuntimeError),
        (NVCFResponse({"status": 500, "detail": "test-detail"}), RuntimeError),
        (NVCFResponse({"status": 404}), NotFoundError),
        (NVCFResponse({"status": 200}), RuntimeError),
        ("Exception", RuntimeError),
    ],
)
def test_list_detail_failure(response: NVCFResponse, exception: type[Exception]) -> None:
    """Test list asset detail failure."""
    asset_manager = AssetManager(url="", nvcf_url="", key="", org="", team="", timeout=15)
    asset_manager.nvcf_api_hdl.get = MagicMock(return_value=response)

    if response == "Exception":
        asset_manager.nvcf_api_hdl.get = MagicMock(side_effect=Exception("test-exception"))

    with pytest.raises(exception):
        asset_manager.list_detail(asset_id="test-asset-id")


def test_nvcf_asset_list_assets_command(
    monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Test nvcf asset list-assets command with runner.invoke.

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
        # Test the list-assets command
        args = [
            "nvcf",
            "asset",
            "list-assets",
        ]
        result = runner.invoke(cosmos_curator, args)
        # The command should fail because no assets are found (no config set up)
        assert result.exit_code == _BAD_EXIT_CODE
        assert "Could not list assets" in caplog.text


def test_nvcf_asset_list_asset_detail_command(
    monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Test nvcf asset list-asset-detail command with runner.invoke.

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
        # Test the list-asset-detail command
        args = [
            "nvcf",
            "asset",
            "list-asset-detail",
            "--assetid",
            "test-asset-id",
        ]
        result = runner.invoke(cosmos_curator, args)
        # The command should fail because no assets are found (no config set up)
        assert result.exit_code == _USAGE_ERROR


def test_nvcf_asset_upload_asset_command(
    monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Test nvcf asset upload-asset command with runner.invoke.

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
        # Test the upload-asset command
        args = [
            "nvcf",
            "asset",
            "upload-asset",
            "--src-path",
            "test.txt",
            "--description",
            "test",
        ]
        result = runner.invoke(cosmos_curator, args)
        # The command should fail because no assets are found (no config set up)
        assert result.exit_code == _USAGE_ERROR


def test_nvcf_asset_delete_asset_command(
    monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Test nvcf asset delete-asset command with runner.invoke.

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
        # Test the delete-asset command
        args = [
            "nvcf",
            "asset",
            "delete-asset",
            "--assetid",
            "test-asset-id",
        ]
        result = runner.invoke(cosmos_curator, args)
        # The command should fail because no assets are found (no config set up)
        assert result.exit_code == _USAGE_ERROR
