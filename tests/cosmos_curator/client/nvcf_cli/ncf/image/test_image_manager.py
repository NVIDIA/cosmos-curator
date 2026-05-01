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
"""Test nvcf image commands."""

import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from _pytest.monkeypatch import MonkeyPatch
from ngcbase.errors import AccessDeniedException, ResourceAlreadyExistsException, ResourceNotFoundException
from typer.testing import CliRunner

from cosmos_curator.client.cli import cosmos_curator  # type: ignore[import-untyped]
from cosmos_curator.client.nvcf_cli.ncf.common import NotFoundError
from cosmos_curator.client.nvcf_cli.ncf.image.image_manager import ImageManager

runner = CliRunner()


def test_success_list_every_image() -> None:
    """Test that all images are displayed correctly.

    Args:
        mock_client:  Mock the client for tests.


    """
    # Define some fake images
    mock_image = Mock()
    fake_items = [
        [Mock(toDict=lambda: {"name": "image1", "latestTag": "v1", "sharedWithOrgs": "1"})],
        [Mock(toDict=lambda: {"name": "image2", "latestTag": "v2", "sharedWithOrgs": "2"})],
    ]
    mock_registry = Mock(image=mock_image)

    # Mock the client instance
    mock_client_instance = Mock(registry=mock_registry)
    mock_image.list.return_value = fake_items

    # show all images
    args = ["nvcf", "image", "list-images", "--all-images"]
    with patch("cosmos_curator.client.nvcf_cli.ncf.image.image_manager.Client", return_value=mock_client_instance):
        result = runner.invoke(cosmos_curator, args)

    output = result.stdout

    # check that table appears
    assert "All Images that My Org Has Access to:" in output
    assert "Name" in output
    assert "Tag" in output
    assert "Size" in output

    assert "Org" in output
    assert "Teams" in output

    # check that image1 appears
    assert "image1" in output
    assert "v1" in output

    # check that image2 appears
    assert "image2" in output
    assert "v2" in output


def test_success_list_own_image() -> None:
    """Test that your org images are displayed correctly."""
    # Define some fake images
    mock_image = Mock()
    fake_items = [
        [Mock(toDict=lambda: {"name": "image1", "latestTag": "v1", "sharedWithOrgs": "1"})],
        [Mock(toDict=lambda: {"name": "image2", "latestTag": "v2", "sharedWithOrgs": "2"})],
    ]
    mock_registry = Mock(image=mock_image)
    # Mock the client instance
    mock_client_instance = Mock(registry=mock_registry)
    mock_image.list.return_value = fake_items

    # show all images
    args = [
        "nvcf",
        "image",
        "--org",
        "1",
        "list-images",
    ]
    with patch("cosmos_curator.client.nvcf_cli.ncf.image.image_manager.Client", return_value=mock_client_instance):
        result = runner.invoke(cosmos_curator, args)
    output = result.stdout
    # check that table appears
    assert "All Images from My Org:" in output
    assert "Name" in output
    assert "Tag" in output
    assert "Size" in output
    assert "Org" in output
    assert "Teams" in output
    # check that image1 appears
    # assert "image1" in output
    assert "v1" in output
    # check that image2 does not appear
    assert "image2" not in output
    assert "v2" not in output


def test_image_details_two() -> None:
    """Test that your org images are displayed correctly."""
    # Define some fake images
    mock_image = Mock()
    fake_items = [
        [Mock(toDict=lambda: {"name": "image1", "latestTag": "v1", "sharedWithOrgs": "1"})],
        [Mock(toDict=lambda: {"name": "image2", "latestTag": "v2", "sharedWithOrgs": "2"})],
    ]
    mock_registry = Mock(image=mock_image)
    # Mock the client instance
    mock_client_instance = Mock(registry=mock_registry)
    mock_image.list.return_value = fake_items
    mock_image.info.return_value = (
        Mock(toDict=lambda: {"name": "image1", "latestTag": "v1", "sharedWithOrgs": "1"}),
        3,
    )

    # show all images
    args = [
        "nvcf",
        "image",
        "--org",
        "1",
        "list-image-detail",
        "--iname",
        "image1",
    ]
    with patch("cosmos_curator.client.nvcf_cli.ncf.image.image_manager.Client", return_value=mock_client_instance):
        result = runner.invoke(cosmos_curator, args)
    output = result.stdout
    assert "image1" in output
    assert "v1" in output
    assert "1" in output


def test_upload_image(monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """Test that your org images are displayed correctly."""
    # Define some fake images
    mock_image = Mock()
    fake_items = [
        [Mock(toDict=lambda: {"name": "image1", "latestTag": "v1", "sharedWithOrgs": "1"})],
        [Mock(toDict=lambda: {"name": "image2", "latestTag": "v2", "sharedWithOrgs": "2"})],
    ]
    mock_registry = Mock(image=mock_image)
    # Mock the client instance
    mock_client_instance = Mock(registry=mock_registry)
    mock_image.list.return_value = fake_items
    mock_image.create.return_value = 1
    mock_image.push.return_value = 1

    # fake image to upload
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    fname = tmp_path / "fakeimage.json"
    with Path.open(fname, "w") as f:
        json.dump({"image": "testimage", "tag": "testtag"}, f)

    # show all images
    args = [
        "nvcf",
        "image",
        "--org",
        "1",
        "upload-image",
        "--data-file",
        str(fname),
    ]
    with caplog.at_level(logging.DEBUG):  # noqa: SIM117
        with patch("cosmos_curator.client.nvcf_cli.ncf.image.image_manager.Client", return_value=mock_client_instance):
            runner.invoke(cosmos_curator, args)

    # assert upload message appears
    assert "testimage" in caplog.text
    assert "testtag" in caplog.text


def test_delete_image(caplog: pytest.LogCaptureFixture) -> None:
    """Test that your org images are displayed correctly."""
    # Define some fake images
    mock_image = Mock()
    fake_items = [
        [Mock(toDict=lambda: {"name": "image1", "latestTag": "v1", "sharedWithOrgs": "1"})],
        [Mock(toDict=lambda: {"name": "image2", "latestTag": "v2", "sharedWithOrgs": "2"})],
        [Mock(toDict=lambda: {"name": "1/todelete", "latestTag": "v3", "sharedWithOrgs": "1"})],
    ]

    def remove_mock_by_name(pattern: str, *, default_yes: bool = True) -> None:
        """Fake remove function.

        Args:
            pattern: identify images to remove.
            default_yes: remove the image.

        """
        if default_yes:
            nonlocal fake_items
            fake_items[:] = [image for image in fake_items if image[0].toDict()["name"] != pattern]

    mock_registry = Mock(image=mock_image)
    # Mock the client instance
    mock_client_instance = Mock(registry=mock_registry)
    mock_image.remove = remove_mock_by_name
    mock_image.list.return_value = fake_items

    # show all images
    args = [
        "nvcf",
        "image",
        "--org",
        "1",
        "list-images",
    ]

    args2 = [
        "nvcf",
        "image",
        "--org",
        "1",
        "delete-image",
        "--iname",
        "todelete",
    ]
    with caplog.at_level(logging.DEBUG):  # noqa: SIM117
        with patch("cosmos_curator.client.nvcf_cli.ncf.image.image_manager.Client", return_value=mock_client_instance):
            result = runner.invoke(cosmos_curator, args)
            runner.invoke(cosmos_curator, args2)
            result_del = runner.invoke(cosmos_curator, args)

    assert "image1" in result.output
    assert "todel" in result.output

    # after deletion
    assert "image1" in result_del.output
    assert "todel" not in result_del.output


class TestImageManager:
    """Test cases for ImageManager class methods."""

    # Mock the image API
    @pytest.fixture
    def image_manager_setup(self) -> tuple[ImageManager, Mock]:
        """Create ImageManager with all mocked dependencies."""
        mock_image_api = Mock()
        mock_registry = Mock(image=mock_image_api)
        mock_client = Mock(registry=mock_registry)

        with patch("cosmos_curator.client.nvcf_cli.ncf.image.image_manager.Client", return_value=mock_client):
            image_manager = ImageManager(url="test", nvcf_url="test", key="test", org="test", team="", timeout=30)
            return image_manager, mock_image_api

    def test_upload_image_success(self, image_manager_setup: Mock, tmp_path: Path) -> None:
        """Test successful image upload.

        Args:
            image_manager_setup: ImageManager with all mocked dependencies.
            tmp_path: Path to temporary directory.

        """
        image_manager, mock_image_api = image_manager_setup

        # Create test JSON file
        fname = tmp_path / "test_image.json"
        with Path.open(fname, "w") as f:
            json.dump({"image": "testimage", "tag": "testtag"}, f)

        # Mock successful create and push
        mock_image_api.create.return_value = Mock(toDict=lambda: {"id": "123", "name": "testimage"})
        mock_image_api.push.return_value = 1

        result = image_manager.upload_image(str(fname))

        assert result is not None
        assert result["id"] == "123"
        assert result["name"] == "testimage"

    def test_upload_image_resource_already_exists(self, image_manager_setup: Mock, tmp_path: Path) -> None:
        """Test upload when image already exists.

        Args:
            image_manager_setup: ImageManager with all mocked dependencies.
            tmp_path: Path to temporary directory.

        """
        image_manager, mock_image_api = image_manager_setup

        # Create test JSON file
        fname = tmp_path / "test_image.json"
        with Path.open(fname, "w") as f:
            json.dump({"image": "testimage", "tag": "testtag"}, f)

        # Mock create to raise ResourceAlreadyExistsException, then successful push
        mock_image_api.create.side_effect = ResourceAlreadyExistsException("Image already exists")
        mock_image_api.push.return_value = 1

        result = image_manager.upload_image(str(fname))

        # Should return None when push succeeds but create failed
        assert result is None

    def test_download_image_success(self, image_manager_setup: Mock) -> None:
        """Test successful image download.

        Args:
            image_manager_setup: ImageManager with all mocked dependencies.

        """
        image_manager, mock_image_api = image_manager_setup

        # Mock successful pull operation
        mock_image_api.pull.return_value = None

        image_manager.download_image("testimage")

        # Verify pull was called with correct image name
        mock_image_api.pull.assert_called_once_with(image="test/testimage")

    def test_download_image_not_found(self, image_manager_setup: Mock) -> None:
        """Test download when image is not found.

        Args:
            image_manager_setup: ImageManager with all mocked dependencies.

        """
        image_manager, mock_image_api = image_manager_setup

        # Mock pull to raise ResourceNotFoundException
        mock_image_api.pull.side_effect = ResourceNotFoundException("Image not found")

        with pytest.raises(NotFoundError):
            image_manager.download_image("testimage")

        mock_image_api.pull.side_effect = AccessDeniedException("Access denied")

        with pytest.raises(RuntimeError):
            image_manager.download_image("testimage")

    def test_delete_image_fails(self, image_manager_setup: Mock) -> None:
        """Test delete when image is not found.

        Args:
            image_manager_setup: ImageManager with all mocked dependencies.

        """
        image_manager, mock_image_api = image_manager_setup

        # Mock delete to raise ResourceNotFoundException
        mock_image_api.remove.side_effect = ResourceNotFoundException("Image not found")

        with pytest.raises(NotFoundError):
            image_manager.delete_image("testimage")

        mock_image_api.remove.side_effect = AccessDeniedException("Access denied")

        with pytest.raises(RuntimeError):
            image_manager.delete_image("testimage")

    def test_list_all_fails(self, image_manager_setup: Mock) -> None:
        """Test list all when image is not found.

        Args:
            image_manager_setup: ImageManager with all mocked dependencies.

        """
        image_manager, mock_image_api = image_manager_setup

        mock_image_api.list.side_effect = Exception("Image not found")

        with pytest.raises(RuntimeError):
            image_manager.list_all(all_accessible_orgs=True)

    def test_list_details_fails(self, image_manager_setup: Mock) -> None:
        """Test list details when image is not found.

        Args:
            image_manager_setup: ImageManager with all mocked dependencies.

        """
        image_manager, mock_image_api = image_manager_setup

        # Mock list to raise ResourceNotFoundException
        mock_image_api.info.side_effect = ResourceNotFoundException("Image not found")

        with pytest.raises(NotFoundError):
            image_manager.list_detail(iname="testimage")

        mock_image_api.info.side_effect = AccessDeniedException("Access denied")

        with pytest.raises(RuntimeError):
            image_manager.list_detail(iname="testimage")

        mock_image_api.info.return_value = (1, 2, 3, 4)
        with pytest.raises(RuntimeError):
            image_manager.list_detail(iname="testimage")
