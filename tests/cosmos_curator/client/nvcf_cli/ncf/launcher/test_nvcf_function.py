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

"""Test the NVCF Helper Module."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from cosmos_curator.client.nvcf_cli.ncf.launcher.nvcf_function import (
    DEFAULT_BASE_NGC_URL,
    DEFAULT_BASE_NVCF_URL,
    DEFAULT_NVCF_TIMEOUT,
    NvcfFunction,
    NvcfFunctionAlreadyDeployedError,
    NvcfFunctionStatus,
)


@pytest.fixture
def nvcf_function() -> NvcfFunction:
    """Create a test NvcfFunction instance."""
    return NvcfFunction(
        funcid="test-funcid",
        version="test-version",
        key="test-key",
        org="test-org",
        team="test-team",
    )


@pytest.mark.parametrize(
    ("funcid", "version", "key", "org", "team", "kwargs", "expected_url", "expected_nvcf_url", "expected_timeout"),
    [
        # Test with default URLs and timeout
        (
            "test-funcid",
            "test-version",
            "test-key",
            "test-org",
            "test-team",
            {},
            DEFAULT_BASE_NGC_URL,
            DEFAULT_BASE_NVCF_URL,
            DEFAULT_NVCF_TIMEOUT,
        ),
        # Test with custom URLs and timeout
        (
            "test-funcid",
            "test-version",
            "test-key",
            "test-org",
            "test-team",
            {
                "ncg_url": "https://custom-ncg.example.com",
                "nvcf_url": "https://custom-nvcf.example.com",
                "timeout": 60,
            },
            "https://custom-ncg.example.com",
            "https://custom-nvcf.example.com",
            60,
        ),
    ],
)
def test_nvcf_function_init(  # noqa: PLR0913
    funcid: str,
    version: str,
    key: str,
    org: str,
    team: str,
    kwargs: dict[str, Any],
    expected_url: str,
    expected_nvcf_url: str,
    expected_timeout: int,
) -> None:
    """Test NvcfFunction initialization with various parameter combinations."""
    nvcf_function = NvcfFunction(
        funcid=funcid,
        version=version,
        key=key,
        org=org,
        team=team,
        **kwargs,
    )

    assert nvcf_function.funcid == funcid
    assert nvcf_function.version == version
    assert nvcf_function.key == key
    assert nvcf_function.org == org
    assert nvcf_function.team == team
    assert nvcf_function.url == expected_url
    assert nvcf_function.nvcf_url == expected_nvcf_url
    assert nvcf_function.timeout == expected_timeout


@patch.object(NvcfFunction, "nvcf_helper_deploy_function")
def test_nvcf_function_deploy_private(mock_deploy: MagicMock, nvcf_function: NvcfFunction, tmp_path: Path) -> None:
    """Test the private _deploy method."""
    backend = "test-backend"
    gpu = "H100"
    instance_type = "test-instance"
    deploy_config = tmp_path / "deploy.json"
    num_nodes = 2
    max_concurrency = 4

    nvcf_function._deploy(backend, gpu, instance_type, deploy_config, num_nodes, max_concurrency)

    mock_deploy.assert_called_once_with(
        funcid="test-funcid",
        version="test-version",
        backend=backend,
        gpu=gpu,
        instance=instance_type,
        min_instances=1,
        max_instances=1,
        max_concurrency=max_concurrency,
        data_file=str(deploy_config),
        instance_count=num_nodes,
    )


@patch.object(NvcfFunction, "nvcf_helper_undeploy_function")
def test_nvcf_function_undeploy_private(mock_undeploy: MagicMock, nvcf_function: NvcfFunction) -> None:
    """Test the private _undeploy method."""
    mock_undeploy.return_value = ("test-function", "INACTIVE")

    nvcf_function._undeploy()

    mock_undeploy.assert_called_once_with(
        funcid="test-funcid",
        version="test-version",
        graceful=True,
    )


@pytest.mark.parametrize(
    ("mock_return", "mock_exception", "expected_status", "should_raise"),
    [
        # Happy path: ACTIVE status
        ({"Status": "ACTIVE"}, None, NvcfFunctionStatus.ACTIVE, False),
        # Happy path: INACTIVE status
        ({"Status": "INACTIVE"}, None, NvcfFunctionStatus.INACTIVE, False),
        # Happy path: any other status (treated as INACTIVE)
        ({"Status": "PENDING"}, None, NvcfFunctionStatus.INACTIVE, False),
        # Error path: Deployment not found
        (None, RuntimeError("Deployment not found"), NvcfFunctionStatus.NOT_FOUND, False),
        # Error path: Deployment not found with additional text
        (None, RuntimeError("Error: Deployment not found for function"), NvcfFunctionStatus.NOT_FOUND, False),
        # Error path: Other RuntimeError should be re-raised
        (None, RuntimeError("Some other error"), None, True),
    ],
)
@patch.object(NvcfFunction, "nvcf_helper_get_deployment_detail")
def test_nvcf_function_get_status(  # noqa: PLR0913
    mock_get_detail: MagicMock,
    nvcf_function: NvcfFunction,
    mock_return: dict[str, str] | None,
    mock_exception: Exception | None,
    expected_status: NvcfFunctionStatus | None,
    *,
    should_raise: bool,
) -> None:
    """Test get_status method covering all possible code paths."""
    if mock_exception:
        mock_get_detail.side_effect = mock_exception
    else:
        mock_get_detail.return_value = mock_return

    if should_raise:
        with pytest.raises(RuntimeError, match="Some other error"):
            nvcf_function.get_status()
    else:
        status = nvcf_function.get_status()
        assert status == expected_status

    mock_get_detail.assert_called_once_with(
        funcid="test-funcid",
        version="test-version",
    )


@pytest.mark.parametrize(
    ("invoke_kwargs", "expected_wait", "expected_retry_cnt", "expected_retry_delay"),
    [
        # Test with default parameters
        ({}, True, 2, 3),
        # Test with custom parameters
        ({"wait": False, "retry_cnt": 5, "retry_delay": 10}, False, 5, 10),
    ],
)
@patch.object(NvcfFunction, "nvcf_helper_invoke_wait_retry_function")
def test_nvcf_function_invoke(  # noqa: PLR0913
    mock_invoke: MagicMock,
    nvcf_function: NvcfFunction,
    tmp_path: Path,
    invoke_kwargs: dict[str, Any],
    *,
    expected_wait: bool,
    expected_retry_cnt: int,
    expected_retry_delay: int,
) -> None:
    """Test invoke method with various parameter combinations."""
    data_file = tmp_path / "invoke.json"
    s3_config = "base64-encoded-config"

    nvcf_function.invoke(
        data_file=data_file,
        s3_config=s3_config,
        out_dir=tmp_path,
        **invoke_kwargs,
    )

    mock_invoke.assert_called_once_with(
        funcid="test-funcid",
        version="test-version",
        data_file=str(data_file),
        s3_config=s3_config,
        ddir=str(tmp_path),
        wait=expected_wait,
        retry_cnt=expected_retry_cnt,
        retry_delay=expected_retry_delay,
        legacy_cf=False,
    )


@patch.object(NvcfFunction, "get_status")
@patch.object(NvcfFunction, "_deploy")
@patch.object(NvcfFunction, "_undeploy")
@patch("time.sleep")
def test_nvcf_function_deploy_context_manager_success(  # noqa: PLR0913
    mock_sleep: MagicMock,
    mock_undeploy: MagicMock,
    mock_deploy: MagicMock,
    mock_get_status: MagicMock,
    nvcf_function: NvcfFunction,
    tmp_path: Path,
) -> None:
    """Test deploy context manager with successful deployment and undeployment."""
    # Setup status progression: INACTIVE -> ACTIVE -> INACTIVE
    mock_get_status.side_effect = [
        NvcfFunctionStatus.INACTIVE,  # Initial check
        NvcfFunctionStatus.INACTIVE,  # First wait loop
        NvcfFunctionStatus.ACTIVE,  # Deploy complete
        NvcfFunctionStatus.ACTIVE,  # First undeploy wait loop
        NvcfFunctionStatus.INACTIVE,  # Undeploy complete
    ]

    backend = "test-backend"
    gpu = "H100"
    instance_type = "test-instance"
    deploy_config = tmp_path / "deploy.json"
    num_nodes = 2
    max_concurrency = 4

    with nvcf_function.deploy(backend, gpu, instance_type, deploy_config, num_nodes, max_concurrency):
        # This is where the user code would run
        pass

    mock_deploy.assert_called_once_with(backend, gpu, instance_type, deploy_config, num_nodes, max_concurrency)
    mock_undeploy.assert_called_once()
    assert mock_get_status.call_count == 5
    assert mock_sleep.call_count == 2


@patch.object(NvcfFunction, "get_status")
def test_nvcf_function_deploy_context_manager_already_deployed(
    mock_get_status: MagicMock,
    nvcf_function: NvcfFunction,
    tmp_path: Path,
) -> None:
    """Test deploy context manager when function is already deployed."""
    mock_get_status.return_value = NvcfFunctionStatus.ACTIVE

    backend = "test-backend"
    gpu = "H100"
    instance_type = "test-instance"
    deploy_config = tmp_path / "deploy.json"
    num_nodes = 2
    max_concurrency = 4

    with (
        pytest.raises(NvcfFunctionAlreadyDeployedError, match="already deployed"),
        nvcf_function.deploy(backend, gpu, instance_type, deploy_config, num_nodes, max_concurrency),
    ):
        pass


@patch.object(NvcfFunction, "get_status")
@patch.object(NvcfFunction, "_deploy")
@patch.object(NvcfFunction, "_undeploy")
@patch("time.sleep")
def test_nvcf_function_deploy_context_manager_with_exception(  # noqa: PLR0913
    mock_sleep: MagicMock,  # noqa: ARG001  (ensures time.sleep is not called)
    mock_undeploy: MagicMock,
    mock_deploy: MagicMock,
    mock_get_status: MagicMock,
    nvcf_function: NvcfFunction,
    tmp_path: Path,
) -> None:
    """Verify undeploy is called when an exception occurs."""
    # Setup status progression
    mock_get_status.side_effect = [
        NvcfFunctionStatus.INACTIVE,  # Initial check
        NvcfFunctionStatus.ACTIVE,  # Deploy complete
        NvcfFunctionStatus.INACTIVE,  # Undeploy complete
    ]

    backend = "test-backend"
    gpu = "H100"
    instance_type = "test-instance"
    deploy_config = tmp_path / "deploy.json"
    num_nodes = 2
    max_concurrency = 4

    msg = "test exception"
    with (
        pytest.raises(ValueError, match=msg),
        nvcf_function.deploy(backend, gpu, instance_type, deploy_config, num_nodes, max_concurrency),
    ):
        raise ValueError(msg)

    mock_deploy.assert_called_once()
    mock_undeploy.assert_called_once()


@patch.object(NvcfFunction, "get_status")
@patch.object(NvcfFunction, "_deploy")
@patch.object(NvcfFunction, "_undeploy")
@patch("time.sleep")
def test_nvcf_function_deploy_context_manager_deployment_timeout(  # noqa: PLR0913
    # patches
    mock_sleep: MagicMock,
    mock_undeploy: MagicMock,  # noqa: ARG001  (ensures NvcfFunction._undeploy is not called)
    mock_deploy: MagicMock,  # noqa: ARG001  (ensures NvcfFunction._deploy is not called)
    mock_get_status: MagicMock,
    # fixtures
    nvcf_function: NvcfFunction,
    tmp_path: Path,
) -> None:
    """Test deploy context manager when deployment takes multiple attempts."""
    # Setup status progression with multiple INACTIVE states before ACTIVE
    mock_get_status.side_effect = [
        NvcfFunctionStatus.INACTIVE,  # Initial check
        NvcfFunctionStatus.INACTIVE,  # First wait loop
        NvcfFunctionStatus.INACTIVE,  # Second wait loop
        NvcfFunctionStatus.ACTIVE,  # Deploy complete
        NvcfFunctionStatus.INACTIVE,  # Undeploy complete
    ]

    backend = "test-backend"
    gpu = "H100"
    instance_type = "test-instance"
    deploy_config = tmp_path / "deploy.json"
    num_nodes = 2
    max_concurrency = 4

    with nvcf_function.deploy(backend, gpu, instance_type, deploy_config, num_nodes, max_concurrency):
        pass

    assert mock_get_status.call_count == 5
    assert mock_sleep.call_count == 2
