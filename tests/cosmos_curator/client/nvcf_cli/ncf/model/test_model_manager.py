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
"""Test nvcf model commands."""

import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from _pytest.monkeypatch import MonkeyPatch
from typer.testing import CliRunner

from cosmos_curator.client.cli import cosmos_curator  # type: ignore[import-untyped]

runner = CliRunner()


def test_success_list_every_model() -> None:
    """Test that all models are displayed correctly."""
    # Define some fake models
    mock_model = Mock()
    fake_items = [
        [Mock(toDict=lambda: {"name": "image1", "latestVersionIdStr": "v1", "orgName": "1"})],
        [Mock(toDict=lambda: {"name": "image2", "latestVersionIdStr": "v2", "orgName": "2"})],
    ]
    mock_registry = Mock(model=mock_model)

    # Mock the client instance
    mock_client_instance = Mock(registry=mock_registry)
    mock_model.list.return_value = fake_items

    # show all models
    args = ["nvcf", "model", "list-models", "--all-models"]
    with patch("cosmos_curator.client.nvcf_cli.ncf.model.model_manager.Client", return_value=mock_client_instance):
        result = runner.invoke(cosmos_curator, args)

    output = result.stdout
    assert "All Models that My Org Has Access to" in output
    assert "v1" in output
    assert "v2" in output


def test_success_list_my_models() -> None:
    """Test that org models are displayed correctly."""
    # Define some fake models
    mock_model = Mock()
    fake_items = [
        [Mock(toDict=lambda: {"name": "image1", "latestVersionIdStr": "v1", "orgName": "1"})],
        [Mock(toDict=lambda: {"name": "image2", "latestVersionIdStr": "v2", "orgName": "2"})],
    ]
    mock_registry = Mock(model=mock_model)

    # Mock the client instance
    mock_client_instance = Mock(registry=mock_registry)
    mock_model.list.return_value = fake_items

    # show org models
    args = ["nvcf", "model", "--org", "1", "list-models"]
    with patch("cosmos_curator.client.nvcf_cli.ncf.model.model_manager.Client", return_value=mock_client_instance):
        result = runner.invoke(cosmos_curator, args)

    output = result.stdout
    assert "All Models from My Org" in output
    assert "v1" in output
    assert "v2" not in output


def test_upload_model(monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """Test upload model command.

    Args:
        monkeypatch: the MonkeyPatch object.
        caplog: A LogCaptureFixture.
        tmp_path: A Path object.

    """
    # Define some fake models
    mock_model = Mock()
    fake_items = [
        [Mock(toDict=lambda: {"name": "image1", "latestVersionIdStr": "v1", "orgName": "1"})],
        [Mock(toDict=lambda: {"name": "image2", "latestVersionIdStr": "v2", "orgName": "2"})],
    ]
    mock_registry = Mock(model=mock_model)
    # Mock the client instance
    mock_client_instance = Mock(registry=mock_registry)
    mock_model.list.return_value = fake_items
    mock_model.create.return_value = 1
    mock_model.upload_version.return_value = 1

    # fake model to upload
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    fname = tmp_path / "fakemodel.json"
    srcname = tmp_path / "fakemodeldest"
    srcname.mkdir()
    with Path.open(fname, "w") as f:
        json.dump({"version": "v1", "target": "test_target"}, f)

    # uoload model command
    args = [
        "nvcf",
        "model",
        "--org",
        "1",
        "upload-model",
        "--data-file",
        str(fname),
        "--src-path",
        str(srcname),
    ]
    with caplog.at_level(logging.DEBUG):  # noqa: SIM117
        with patch("cosmos_curator.client.nvcf_cli.ncf.model.model_manager.Client", return_value=mock_client_instance):
            runner.invoke(cosmos_curator, args)

    assert "uploaded successfully" in caplog.text
    assert "v1" in caplog.text
    assert "test_target" in caplog.text


def test_delete_model() -> None:
    """Test delete models."""
    # Define some fake models
    mock_model = Mock()
    fake_items = [
        [Mock(toDict=lambda: {"name": "image1", "latestVersionIdStr": "v1", "orgName": "1"})],
        [Mock(toDict=lambda: {"name": "image2", "latestVersionIdStr": "v2", "orgName": "2"})],
        [Mock(toDict=lambda: {"name": "1/todelete", "latestVersionIdStr": "v3", "orgName": "1"})],
    ]

    def remove_mock_by_name(target: str) -> None:
        """Fake remove function.

        Args:
            target: identify models to remove.

        """
        nonlocal fake_items
        fake_items[:] = [model for model in fake_items if model[0].toDict()["name"] != target]

    mock_registry = Mock(model=mock_model)
    # Mock the client instance
    mock_client_instance = Mock(registry=mock_registry)
    mock_model.remove = remove_mock_by_name
    mock_model.list.return_value = fake_items

    # show all models
    args = [
        "nvcf",
        "model",
        "--org",
        "1",
        "list-models",
    ]

    # delete model command
    args2 = [
        "nvcf",
        "model",
        "--org",
        "1",
        "delete-model",
        "--mname",
        "todelete",
    ]
    with patch("cosmos_curator.client.nvcf_cli.ncf.model.model_manager.Client", return_value=mock_client_instance):
        result = runner.invoke(cosmos_curator, args)
        runner.invoke(cosmos_curator, args2)
        result_del = runner.invoke(cosmos_curator, args)

    assert "v1" in result.output
    assert "v3" in result.output

    assert "v1" in result_del.output
    assert "v3" not in result_del.output
