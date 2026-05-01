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
"""Test nvcf driver commands."""

import logging
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from cosmos_curator.client.cli import cosmos_curator  # type: ignore[import-not-found]
from cosmos_curator.client.nvcf_cli.ncf.common import NotFoundError

runner = CliRunner()

_FAKE_UUID_ONE = "12345678-1234-1234-1234-123456789abc"
_FAKE_UUID_TWO = "87654321-4321-4321-4321-987654321def"


def test_deploy_function_no_id_version(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Test that  deploy-function fails with no id.

    Args:
        caplog: To capture the logs.
        tmp_path: A temporary path for tests.

    """
    mock_func = MagicMock()

    # A mock data file to pass to the command
    mock_func.id_version.return_value = (False, 1234, 5678)
    fname = tmp_path / "fake.json"
    fname.touch()

    args = [
        "nvcf",
        "function",
        "deploy-function",
        "--data-file",
        str(fname),
    ]

    with caplog.at_level(logging.DEBUG):
        runner.invoke(cosmos_curator, args)

    assert "ERROR" in caplog.text
    assert "Could not deploy function" in caplog.text


@patch("cosmos_curator.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_deploy_function_no_backend_gpu_instance(mock_cc: MagicMock, tmp_path: Path) -> None:
    """Test that deploy-function fails with no backened, gpu, or instance.

    Args:
        mock_cc: A mock to return a fake helper.
        tmp_path: A temporary path.

    """
    # A mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    # A mock data file to pass to the command
    fname = tmp_path / "fake.json"
    fname.touch()

    # Mock return values to test backend = None
    mock_instance.id_version.return_value = (True, 1234, 5678)
    mock_instance.get_cluster.return_value = (True, None, None, None)

    args = [
        "nvcf",
        "function",
        "deploy-function",
        "--data-file",
        str(fname),
    ]

    result = runner.invoke(cosmos_curator, args)
    mock_instance.logger.error.assert_called_with("Could not deploy function: backend is required")
    assert result.exit_code != 0

    # Mock return values to test gpu = None
    mock_instance.get_cluster.return_value = (True, True, None, None)
    result = runner.invoke(cosmos_curator, args)
    mock_instance.logger.error.assert_called_with("Could not deploy function: gpu is required")
    assert result.exit_code != 0

    # Mock return values to test instance = None
    mock_instance.get_cluster.return_value = (True, True, True, None)
    result = runner.invoke(cosmos_curator, args)
    mock_instance.logger.error.assert_called_with("Could not deploy function: instance is required")
    assert result.exit_code != 0


@patch("cosmos_curator.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_deploy_function_success(mock_cc: MagicMock, tmp_path: Path) -> None:
    """Test that deploy-function can succeeed.

    Args:
        mock_cc: A mock to return a fake helper.
        tmp_path: A temporary path.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    # Mock file to pass as arguement
    fname = tmp_path / "fake.json"
    fname.touch()

    mock_instance.id_version.return_value = (True, 1234, 5678)
    mock_instance.get_cluster.return_value = (True, True, True, True)

    args = [
        "nvcf",
        "function",
        "deploy-function",
        "--data-file",
        str(fname),
    ]

    result = runner.invoke(cosmos_curator, args)
    mock_instance.console.print.assert_called_with(
        "Function with id '1234' and version '5678' is being deployed; "
        "to check status: cosmos-curator nvcf function get-deployment-detail"
    )
    assert result.exit_code == 0


@patch("cosmos_curator.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_deploy_function_exception(mock_cc: MagicMock, tmp_path: Path) -> None:
    """Test that deploy-function fails with an exception.

    Args:
        mock_cc: A mock to return a fake helper.
        tmp_path: A temporary Path.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    fname = tmp_path / "fake.json"
    fname.touch()

    mock_instance.id_version.return_value = (True, 1234, 5678)
    mock_instance.get_cluster.return_value = (True, True, True, True)
    mock_instance.nvcf_helper_deploy_function.side_effect = RuntimeError("mock exception")

    args = [
        "nvcf",
        "function",
        "deploy-function",
        "--data-file",
        str(fname),
    ]

    result = runner.invoke(cosmos_curator, args)
    mock_instance.logger.error.assert_called_with("Could not deploy function: mock exception")
    assert result.exit_code == 1


@patch("cosmos_curator.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_deploy_function_validation_failures(mock_cc: MagicMock, tmp_path: Path) -> None:
    """Test that deploy-function fails with invalid arguments.

    Args:
        mock_cc: A mock to return a fake helper.
        tmp_path: A temporary path.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    fname = tmp_path / "fake.json"

    # file does not exist
    args = [
        "nvcf",
        "function",
        "deploy-function",
        "--data-file",
        str(fname),
    ]

    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code != 0

    fname.touch()

    mock_instance.id_version.return_value = (True, 1234, 5678)
    args = ["nvcf", "function", "deploy-function", "--data-file", str(fname), "-version", "badversion"]

    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code != 0

    # negatitive number
    args = [
        "nvcf",
        "function",
        "deploy-function",
        "--data-file",
        str(fname),
        "--min-instances",
        "-5",
    ]
    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code != 0


@patch("cosmos_curator.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_invoke_function_no_id_version(mock_cc: MagicMock, tmp_path: Path) -> None:
    """Test that invoke-function fails with no id.

    Args:
        mock_cc: A mock to return a fake helper.
        tmp_path: A temporary path.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    fname = tmp_path / "fake.json"
    fname.touch()

    mock_instance.id_version.return_value = (False, 1234, 5678)

    args = [
        "nvcf",
        "function",
        "invoke-function",
        "--data-file",
        str(fname),
    ]

    result = runner.invoke(cosmos_curator, args)
    mock_instance.logger.error.assert_called_with("Could not invoke function: id and version are required")
    assert result.exit_code == 1


@patch("cosmos_curator.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_invoke_function_bad_assetid(mock_cc: MagicMock, tmp_path: Path) -> None:
    """Test that invoke-function fails with a bad assetid.

    Args:
        mock_cc: A mock to return a fake helper.
        tmp_path: A temporary path.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    fname = tmp_path / "fake.json"
    fname.touch()

    mock_instance.id_version.return_value = (True, 1234, 5678)
    # A mock uuid
    my_uuid = uuid.uuid4()

    args = [
        "nvcf",
        "function",
        "invoke-function",
        "--data-file",
        str(fname),
        "--assetid",
        str(my_uuid),
        "--assetfile",
        str(fname),
    ]

    result = runner.invoke(cosmos_curator, args)
    mock_instance.logger.error.assert_called_with(
        "Could not invoke function: assetid and assetfile are mutually exclusive"
    )
    assert result.exit_code == 1


@patch("cosmos_curator.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_invoke_function_test_wait(mock_cc: MagicMock, tmp_path: Path) -> None:
    """Test that invoke-function call correct function with wait.

    Args:
        mock_cc: A mock to return a fake helper.
        tmp_path: A temporary path.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    fname = tmp_path / "fake.json"
    fname.touch()

    mock_instance.id_version.return_value = (True, 1234, 5678)

    args = [
        "nvcf",
        "function",
        "invoke-function",
        "--data-file",
        str(fname),
        "--wait",
    ]

    result = runner.invoke(cosmos_curator, args)
    mock_instance.nvcf_helper_invoke_wait_retry_function.assert_called()
    assert result.exit_code == 0

    args = ["nvcf", "function", "invoke-function", "--data-file", str(fname), "--no-wait"]

    result = runner.invoke(cosmos_curator, args)
    mock_instance.nvcf_helper_invoke_function.assert_called()
    mock_instance.console.print.assert_called()
    assert result.exit_code == 0


@patch("cosmos_curator.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_invoke_function_validation_failures(mock_cc: MagicMock, tmp_path: Path) -> None:
    """Test that invoke-function fails validation errors.

    Args:
        mock_cc: A mock to return a fake helper.
        tmp_path: A temporary path.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    fname = tmp_path / "fake.json"

    # file does not exist
    args = [
        "nvcf",
        "function",
        "invoke-function",
        "--data-file",
        str(fname),
    ]

    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code != 0

    fname.touch()

    mock_instance.id_version.return_value = (True, 1234, 5678)
    args = ["nvcf", "function", "invoke-function", "--data-file", str(fname), "--assetid", "badassetid"]

    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code != 0

    # negatitive number
    args = [
        "nvcf",
        "function",
        "invoke-function",
        "--data-file",
        str(fname),
        "--retry_cnt",
        "-5",
    ]
    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code != 0


@patch("cosmos_curator.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_list_clusters(mock_cc: MagicMock) -> None:
    """Test that list-clusters can succeed.

    Args:
        mock_cc: A mock to return a fake helper.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    # Mock the required method
    mock_instance.nvcf_helper_list_clusters.return_value = "mock clusters response"

    args = ["nvcf", "function", "list-clusters"]

    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 0

    # Test with exception in helper method
    mock_instance.nvcf_helper_list_clusters.side_effect = RuntimeError("mock exception")
    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 1


@patch("cosmos_curator.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_list_functions(mock_cc: MagicMock) -> None:
    """Test that list-functions can succeed.

    Args:
        mock_cc: A mock to return a fake helper.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    # Mock the required method
    mock_instance.nvcf_helper_list_functions.return_value = "mock functions response"

    args = ["nvcf", "function", "list-functions"]

    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 0

    # Test with exception in helper method
    mock_instance.nvcf_helper_list_functions.side_effect = RuntimeError("mock exception")
    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 1


@patch("cosmos_curator.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_list_function_detail(mock_cc: MagicMock) -> None:
    """Test that list-versions can succeed.

    Args:
        mock_cc: A mock to return a fake helper.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    # Mock the required method
    mock_instance.nvcf_helper_list_function_detail.return_value = "mock function detail response"

    args = ["nvcf", "function", "list-function-detail", "--name", "test-function"]

    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 0

    # Test with NotFoundError in helper method
    mock_instance.nvcf_helper_list_function_detail.side_effect = NotFoundError("mock exception")
    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 1

    # Test with RuntimeError in helper method
    mock_instance.nvcf_helper_list_function_detail.side_effect = RuntimeError("mock exception")
    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 1


@patch("cosmos_curator.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_create_function(mock_cc: MagicMock, tmp_path: Path) -> None:
    """Test that create-function can succeed.

    Args:
        tmp_path: A temporary path.
        mock_cc: A mock to return a fake helper.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    # Create a temporary data file
    fname = tmp_path / "fake.json"
    fname.touch()

    # Mock the required method
    mock_instance.nvcf_helper_create_function.return_value = {"name": "test-function", "id": "123", "version": "456"}
    mock_instance.store_ids.return_value = None

    args = ["nvcf", "function", "create-function", "--name", "test-function", "--data-file", str(fname)]

    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 0

    # Test with exception in helper method
    mock_instance.nvcf_helper_create_function.side_effect = RuntimeError("mock exception")
    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 1


@patch("cosmos_curator.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_nvcf_invoke_batch_function(mock_cc: MagicMock, tmp_path: Path) -> None:
    """Test that invoke-batch can succeed.

    Args:
        tmp_path: A temporary path.
        mock_cc: A mock to return a fake helper.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    # Create temporary files for all required parameters
    data_file = tmp_path / "data.json"
    id_file = tmp_path / "id.json"
    job_variant_file = tmp_path / "job_variant.json"
    s3_config_file = tmp_path / "s3_config.json"

    data_file.touch()
    id_file.touch()
    job_variant_file.touch()
    s3_config_file.touch()

    args = [
        "nvcf",
        "function",
        "invoke-batch",
        "--data-file",
        str(data_file),
        "--id-file",
        str(id_file),
        "--job-variant-file",
        str(job_variant_file),
        "--s3-config-file",
        str(s3_config_file),
    ]

    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 0


@patch("cosmos_curator.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_get_request_status(mock_cc: MagicMock, tmp_path: Path) -> None:
    """Test that get-request-status can succeed.

    Args:
        tmp_path: A temporary path.
        mock_cc: A mock to return a fake helper.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    # Mock the required methods
    mock_instance.id_version.return_value = (True, _FAKE_UUID_ONE, _FAKE_UUID_TWO)
    mock_instance.nvcf_helper_get_request_status_with_wait.return_value = None

    # Test with a custom output directory
    custom_out_dir = tmp_path / "custom_output"
    custom_out_dir.mkdir()

    args = ["nvcf", "function", "get-request-status", "--reqid", _FAKE_UUID_ONE, "--out-dir", str(custom_out_dir)]

    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 0

    mock_instance.nvcf_helper_get_request_status_with_wait.side_effect = RuntimeError("mock exception")
    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 1

    mock_instance.id_version.return_value = (None, _FAKE_UUID_ONE, _FAKE_UUID_TWO)
    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 1


@patch("cosmos_curator.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_nvcf_terminate_request(mock_cc: MagicMock) -> None:
    """Test that terminate-request can succeed.

    Args:
        mock_cc: A mock to return a fake helper.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    # Mock the required methods
    mock_instance.id_version.return_value = (True, _FAKE_UUID_ONE, _FAKE_UUID_TWO)
    mock_instance.nvcf_helper_terminate_request.return_value = None

    args = ["nvcf", "function", "terminate-request", "--reqid", _FAKE_UUID_ONE]

    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 0

    mock_instance.nvcf_helper_terminate_request.side_effect = RuntimeError("mock exception")
    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 1

    mock_instance.id_version.return_value = (None, _FAKE_UUID_ONE, _FAKE_UUID_TWO)
    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 1


@patch("cosmos_curator.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_nvcf_get_deployment_detail(mock_cc: MagicMock) -> None:
    """Test that get-deployment-detail can succeed.

    Args:
        mock_cc: A mock to return a fake helper.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    # Mock the required methods
    mock_instance.id_version.return_value = (True, _FAKE_UUID_ONE, _FAKE_UUID_TWO)
    mock_instance.nvcf_helper_get_deployment_detail.return_value = None

    args = ["nvcf", "function", "get-deployment-detail", "--funcid", _FAKE_UUID_ONE, "--version", _FAKE_UUID_TWO]

    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 0

    # Test with exception in helper method
    mock_instance.nvcf_helper_get_deployment_detail.side_effect = RuntimeError("mock exception")
    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 1

    # Test with failed id_version validation
    mock_instance.id_version.return_value = (False, _FAKE_UUID_ONE, _FAKE_UUID_TWO)
    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 1


@patch("cosmos_curator.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_nvcf_delete_function(mock_cc: MagicMock) -> None:
    """Test that delete-function can succeed.

    Args:
        mock_cc: A mock to return a fake helper.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    # Mock the required methods
    mock_instance.id_version.return_value = (True, _FAKE_UUID_ONE, _FAKE_UUID_TWO)
    mock_instance.nvcf_helper_delete_function.return_value = None

    args = ["nvcf", "function", "delete-function", "--funcid", _FAKE_UUID_ONE, "--version", _FAKE_UUID_TWO]

    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 0

    # Test with exception in helper method
    mock_instance.nvcf_helper_delete_function.side_effect = RuntimeError("mock exception")
    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 1

    # Test with failed id_version validation
    mock_instance.id_version.return_value = (False, _FAKE_UUID_ONE, _FAKE_UUID_TWO)
    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 1


@patch("cosmos_curator.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_nvcf_s3_config_function(mock_cc: MagicMock, tmp_path: Path) -> None:
    """Test that s3-config can succeed.

    Args:
        tmp_path: A temporary path.
        mock_cc: A mock to return a fake helper.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    # Create a temporary data file
    fname = tmp_path / "fake.json"
    fname.touch()

    # Mock the required methods
    mock_instance.id_version.return_value = (True, _FAKE_UUID_ONE, _FAKE_UUID_TWO)
    mock_instance.nvcf_helper_s3cred_function.return_value = None

    args = [
        "nvcf",
        "function",
        "s3cred-function",
        "--funcid",
        _FAKE_UUID_ONE,
        "--version",
        _FAKE_UUID_TWO,
        "--s3credfile",
        str(fname),
    ]

    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 0

    # Test with exception in helper method
    mock_instance.nvcf_helper_s3cred_function.side_effect = RuntimeError("mock exception")
    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 1

    # Test with failed id_version validation
    mock_instance.id_version.return_value = (False, _FAKE_UUID_ONE, _FAKE_UUID_TWO)
    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 1


@patch("cosmos_curator.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_nvcf_undeploy_function(mock_cc: MagicMock) -> None:
    """Test that undeploy-function can succeed.

    Args:
        mock_cc: A mock to return a fake helper.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    # Mock the required methods
    mock_instance.id_version.return_value = (True, _FAKE_UUID_ONE, _FAKE_UUID_TWO)
    mock_instance.nvcf_helper_undeploy_function.return_value = ("test-function", "undeployed")

    args = ["nvcf", "function", "undeploy-function", "--funcid", _FAKE_UUID_ONE, "--version", _FAKE_UUID_TWO]

    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 0

    # Test with exception in helper method
    mock_instance.nvcf_helper_undeploy_function.side_effect = RuntimeError("mock exception")
    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 1

    # Test with failed id_version validation
    mock_instance.id_version.return_value = (False, _FAKE_UUID_ONE, _FAKE_UUID_TWO)
    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 1


@patch("cosmos_curator.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_import_function(mock_cc: MagicMock) -> None:
    """Test that import-function can succeed.

    Args:
        mock_cc: A mock to return a fake helper.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    args = [
        "nvcf",
        "function",
        "import-function",
        "--funcid",
        _FAKE_UUID_ONE,
        "--version",
        _FAKE_UUID_TWO,
        "--name",
        "test-function",
    ]

    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code == 0
