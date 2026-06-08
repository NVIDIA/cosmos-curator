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

import json
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from _pytest.monkeypatch import MonkeyPatch
from rich.console import Console
from rich.table import Table

from cosmos_curator.client.nvcf_cli.ncf.common import NotFoundError, NVCFResponse
from cosmos_curator.client.nvcf_cli.ncf.launcher.nvcf_helper import (
    NvcfHelper,
    _extract_nvcf_error_details,
    _raise_runtime_err,
    _raise_timeout_err,
)


def test_raise_runtime_err() -> None:
    """Test that _raise_runtime_err raises a RuntimeError with the given message."""
    with pytest.raises(RuntimeError):
        _raise_runtime_err("test message")

    with pytest.raises(RuntimeError):
        _raise_runtime_err({"error": "test message"})


def test_raise_timeout_err() -> None:
    """Test that _raise_timeout_err raises a TimeoutError with the given message."""
    with pytest.raises(TimeoutError):
        _raise_timeout_err("test message")

    with pytest.raises(TimeoutError):
        _raise_timeout_err({"error": "test message"})


def test_load_ids(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that load_ids returns the expected dictionary."""
    config_dir = tmp_path / ".config/cosmos_curator"
    config_dir.mkdir(parents=True, exist_ok=True)
    fname = config_dir / "funcid.json"
    with Path.open(fname, "w") as f:
        json.dump({"name": "test_name", "id": "test_id", "version": "test_version"}, f)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    assert nvcf_helper.load_ids() == {"name": "test_name", "id": "test_id", "version": "test_version"}


def test_store_ids(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that store_ids stores the given dictionary to the file."""
    config_dir = tmp_path / ".config/cosmos_curator"
    config_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.store_ids({"name": "test_name", "id": "test_id", "version": "test_version"})
    assert nvcf_helper.load_ids() == {"name": "test_name", "id": "test_id", "version": "test_version"}


def test_cleanup_ids(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that cleanup_ids removes the file."""
    config_dir = tmp_path / ".config/cosmos_curator"
    config_dir.mkdir(parents=True, exist_ok=True)
    fname = config_dir / "funcid.json"
    with Path.open(fname, "w") as f:
        json.dump({"name": "test_name", "id": "test_id", "version": "test_version"}, f)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.cleanup_ids()
    assert not fname.exists()


def test_id_version(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that id_version returns the expected tuple."""
    config_dir = tmp_path / ".config/cosmos_curator"
    config_dir.mkdir(parents=True, exist_ok=True)
    fname = config_dir / "funcid.json"
    Path.open(fname, "w")

    # Check empty file first
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    assert nvcf_helper.id_version(None, None) == (False, None, None)
    assert nvcf_helper.id_version(None, "test_version") == (False, None, "test_version")
    assert nvcf_helper.id_version("test_id", "test_version") == (True, "test_id", "test_version")

    # File file with fake data
    with Path.open(fname, "w") as f:
        json.dump({"name": "test_name", "id": "test_id", "version": "test_version"}, f)

    assert nvcf_helper.id_version(None, "test_version") == (True, "test_id", "test_version")
    assert nvcf_helper.id_version("test_id", "test_version") == (True, "test_id", "test_version")


def test_nvcf_helper_list_clusters_success(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that nvcf_helper_list_clusters returns the expected table on success.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.

    Returns:
        None

    """
    # Mock the response data
    mock_response_data = {
        "status": 200,
        "clusterGroups": [
            {
                "name": "test-backend",
                "gpus": [
                    {"name": "H100", "instanceTypes": [{"name": "instance-1"}, {"name": "instance-2"}]},
                    {"name": "H100", "instanceTypes": [{"name": "instance-3"}]},
                ],
                "clusters": [{"name": "cluster-1"}, {"name": "cluster-2"}],
            },
            {
                "name": "test-backend-2",
                "gpus": [{"name": "H100", "instanceTypes": [{"name": "instance-4"}]}],
                "clusters": [{"name": "cluster-3"}],
            },
        ],
    }

    # Mock the response
    mock_response = NVCFResponse(mock_response_data)

    mock_ncg_client = MagicMock()
    mock_ncg_client.get.return_value = mock_response

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.ncg_api_hdl = mock_ncg_client

    # Get the results
    result = nvcf_helper.nvcf_helper_list_clusters()

    # Asserts for testing the result
    assert result is not None
    assert isinstance(result, Table)
    assert result.title == "Cluster Groups"

    columns = [col.header for col in result.columns]
    assert "Backend Name" in columns
    assert "GPU-Types Inst-Types" in columns
    assert "Clusters" in columns

    mock_ncg_client.get.assert_called_once_with("/v2/nvcf/clusterGroups")

    output_str = StringIO()
    console = Console(file=output_str, width=400, record=True)
    console.print(result)

    # Asserts for testing content of table
    assert "test-backend" in output_str.getvalue()
    assert "test-backend-2" in output_str.getvalue()
    assert "H100" in output_str.getvalue()
    assert "instance-1" in output_str.getvalue()
    assert "instance-2" in output_str.getvalue()
    assert "instance-3" in output_str.getvalue()
    assert "instance-4" in output_str.getvalue()
    assert "cluster-1" in output_str.getvalue()
    assert "cluster-3" in output_str.getvalue()


def test_nvcf_helper_list_clusters_failure(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that nvcf_helper_list_clusters returns None on failure.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.

    Returns:
        None

    """
    mock_ncg_client = MagicMock()
    mock_ncg_client.get.return_value = None

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.ncg_api_hdl = mock_ncg_client
    with pytest.raises(RuntimeError):
        nvcf_helper.nvcf_helper_list_clusters()

    mock_response_data = {
        "status": 500,
    }
    mock_response = NVCFResponse(mock_response_data)
    mock_ncg_client.get.return_value = mock_response
    nvcf_helper.ncg_api_hdl = mock_ncg_client

    with pytest.raises(RuntimeError):
        nvcf_helper.nvcf_helper_list_clusters()


def test_nvcf_helper_list_functions_success(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that nvcf_helper_list_clusters returns the expected table on success.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.

    Returns:
        None

    """
    # Mock the response data
    mock_response_data = {
        "status": 200,
        "functions": [
            {
                "name": "test-function",
                "status": "active",
                "id": "test-id",
                "versionId": "test-version",
                "containerImage": "test-image",
                "inferenceUrl": "test-endpoint",
            },
            {
                "name": "test-function-2",
                "status": "active",
                "id": "test-id-2",
                "versionId": "test-version-2",
                "containerImage": "test-image-2",
                "inferenceUrl": "test-endpoint-2",
            },
        ],
    }

    # Mock the response
    mock_response = NVCFResponse(mock_response_data)

    mock_ncg_client = MagicMock()
    mock_ncg_client.get.return_value = mock_response

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.ncg_api_hdl = mock_ncg_client

    # Get the results
    result = nvcf_helper.nvcf_helper_list_functions()

    # Asserts for testing the result
    assert result is not None
    assert isinstance(result, Table)
    assert result.title == "Functions"

    columns = [col.header for col in result.columns]
    assert "Name" in columns
    assert "Status" in columns
    assert "Id" in columns
    assert "Version" in columns
    assert "Image" in columns
    assert "Endpoint" in columns

    mock_ncg_client.get.assert_called_once_with("/v2/nvcf/functions")

    output_str = StringIO()
    console = Console(file=output_str, width=400, record=True)
    console.print(result)

    # Asserts for testing content of table
    assert "test-function" in output_str.getvalue()
    assert "test-function-2" in output_str.getvalue()
    assert "test-id" in output_str.getvalue()
    assert "test-id-2" in output_str.getvalue()
    assert "test-version" in output_str.getvalue()
    assert "test-version-2" in output_str.getvalue()
    assert "test-image" in output_str.getvalue()
    assert "test-image-2" in output_str.getvalue()
    assert "None:test-endpoint" in output_str.getvalue()
    assert "None:test-endpoint-2" in output_str.getvalue()


def test_nvcf_helper_list_functions_failure(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that nvcf_helper_list_functions returns None on failure.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.

    Returns:
        None

    """
    mock_ncg_client = MagicMock()
    mock_ncg_client.get.return_value = None

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.ncg_api_hdl = mock_ncg_client
    with pytest.raises(RuntimeError):
        nvcf_helper.nvcf_helper_list_functions()

    mock_response_data = {
        "status": 500,
    }
    mock_response = NVCFResponse(mock_response_data)
    mock_ncg_client.get.return_value = mock_response
    nvcf_helper.ncg_api_hdl = mock_ncg_client

    with pytest.raises(RuntimeError):
        nvcf_helper.nvcf_helper_list_functions()


def test_nvcf_helper_list_function_detail(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that nvcf_helper_list_function_detail returns the expected table on success.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.

    Returns:
        None

    """
    # Mock the response data
    mock_response_data = {
        "status": 200,
        "functions": [
            {
                "name": "test-function",
                "status": "active",
                "id": "test-id",
                "versionId": "test-version",
                "containerImage": "test-image",
                "inferenceUrl": "test-endpoint",
            },
            {
                "name": "test-function-2",
                "status": "active",
                "id": "test-id-2",
                "versionId": "test-version-2",
                "containerImage": "test-image-2",
                "inferenceUrl": "test-endpoint-2",
            },
        ],
    }

    # Mock the response
    mock_response = NVCFResponse(mock_response_data)

    mock_ncg_client = MagicMock()
    mock_ncg_client.get.return_value = mock_response

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.ncg_api_hdl = mock_ncg_client

    # Get the results
    result = nvcf_helper.nvcf_helper_list_function_detail("test-function")

    # Asserts for testing the result
    assert result is not None

    mock_ncg_client.get.assert_called_once_with("/v2/nvcf/functions")

    assert result is not None
    assert len(result) == 1
    assert result[0]["Status"] == "active"
    assert result[0]["Id"] == "test-id"
    assert result[0]["Version"] == "test-version"
    assert result[0]["Image"] == "test-image"
    assert result[0]["Endpoint"] == "None:test-endpoint"

    with pytest.raises(NotFoundError):
        nvcf_helper.nvcf_helper_list_function_detail("test-function-not-here")


def test_nvcf_helper_list_function_detail_failures(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that nvcf_helper_list_function_detail returns the expected table on success.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.

    Returns:
        None

    """
    # Mock the response
    mock_response = None

    mock_ncg_client = MagicMock()
    mock_ncg_client.get.return_value = mock_response

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.ncg_api_hdl = mock_ncg_client

    # Get the results for None response
    with pytest.raises(RuntimeError):
        nvcf_helper.nvcf_helper_list_function_detail("test-function")

    # Get the results for error response
    mock_response_data = {
        "status": 500,
    }
    mock_response = NVCFResponse(mock_response_data)
    mock_ncg_client.get.return_value = mock_response
    nvcf_helper.ncg_api_hdl = mock_ncg_client

    with pytest.raises(RuntimeError):
        nvcf_helper.nvcf_helper_list_function_detail("test-function")


def test_nvcf_helper_create_function_no_data(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that nvcf_helper_create_function returns the expected dictionary on success.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.

    Returns:
        None

    """
    mock_ncg_client = MagicMock()
    mock_ncg_client.post.return_value = NVCFResponse(
        {"status": 200, "function": {"id": "test-id", "versionId": "test-version"}}
    )

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.ncg_api_hdl = mock_ncg_client

    result = nvcf_helper.nvcf_helper_create_function(
        name="test-function",
        image="test-image",
        inference_ep="test-endpoint",
        inference_port=8080,
        health_ep="test-health-endpoint",
        health_port=8081,
        args="test-args",
        data_file=None,
        helm_chart="test-helm-chart",
        helm_service_name="test-helm-service-name",
    )
    call_data = {
        "name": "test-function",
        "inferenceUrl": "test-endpoint",
        "inferencePort": 8080,
        "health": {
            "protocol": "HTTP",
            "uri": "test-health-endpoint",
            "port": 8081,
            "timeout": "PT10S",
            "expectedStatusCode": 200,
        },
        "functionType": "DEFAULT",
        "description": "Video Curation Service",
        "apiBodyFormat": "PREDICT_V2",
        "helmChart": None,
        "helmChartServiceName": None,
        "containerImage": "test-image",
        "containerArgs": "test-args",
    }
    mock_ncg_client.post.assert_called_once_with("/v2/nvcf/functions", data=call_data)

    assert result == {"id": "test-id", "version": "test-version"}


def test_nvcf_helper_create_function_with_data(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that nvcf_helper_create_function returns the expected dictionary on success.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.

    Returns:
        None

    """
    mock_ncg_client = MagicMock()
    mock_ncg_client.post.return_value = NVCFResponse(
        {"status": 200, "function": {"id": "test-id", "versionId": "test-version"}}
    )

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.ncg_api_hdl = mock_ncg_client

    tmp_dir = tmp_path / "test-data"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = tmp_dir / "test-data.json"
    with Path.open(tmp_file, "w") as f:
        json.dump(
            {
                "models": ["test-model"],
                "tags": ["test-tag"],
                "resources": ["test-resource"],
                "secrets": [{"key": "test-secret-key", "value": "test-secret-value"}],
                "envs": ["test-env"],
            },
            f,
        )

    result = nvcf_helper.nvcf_helper_create_function(
        name="test-function",
        image="test-image",
        inference_ep="test-endpoint",
        inference_port=8080,
        health_ep="test-health-endpoint",
        health_port=8081,
        args="test-args",
        data_file=str(tmp_file),
        helm_chart="test-helm-chart",
        helm_service_name="test-helm-service-name",
    )

    call_data = {
        "name": "test-function",
        "inferenceUrl": "test-endpoint",
        "inferencePort": 8080,
        "health": {
            "protocol": "HTTP",
            "uri": "test-health-endpoint",
            "port": 8081,
            "timeout": "PT10S",
            "expectedStatusCode": 200,
        },
        "functionType": "DEFAULT",
        "description": "Video Curation Service",
        "apiBodyFormat": "PREDICT_V2",
        "helmChart": None,
        "helmChartServiceName": None,
        "containerImage": "test-image",
        "containerArgs": "test-args",
        "models": ["test-model"],
        "tags": ["test-tag"],
        "resources": ["test-resource"],
        "secrets": [{"key": "test-secret-key", "value": "test-secret-value"}],
        "containerEnvironment": ["test-env"],
    }

    mock_ncg_client.post.assert_called_once_with("/v2/nvcf/functions", data=call_data)
    assert result == {"id": "test-id", "version": "test-version"}


def test_nvcf_helper_create_function_fail(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that nvcf_helper_create_function returns the expected dictionary on success.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.

    Returns:
        None

    """
    mock_ncg_client = MagicMock()
    mock_ncg_client.post.return_value = None

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.ncg_api_hdl = mock_ncg_client

    with pytest.raises(RuntimeError):
        nvcf_helper.nvcf_helper_create_function(
            name="test-function",
            image="test-image",
            inference_ep="test-endpoint",
            inference_port=8080,
            health_ep="test-health-endpoint",
            health_port=8081,
            args="test-args",
            data_file=None,
            helm_chart="test-helm-chart",
            helm_service_name="test-helm-service-name",
        )

    mock_ncg_client.post.return_value = NVCFResponse({"status": 500})
    nvcf_helper.ncg_api_hdl = mock_ncg_client

    with pytest.raises(RuntimeError):
        nvcf_helper.nvcf_helper_create_function(
            name="test-function",
            image="test-image",
            inference_ep="test-endpoint",
            inference_port=8080,
            health_ep="test-health-endpoint",
            health_port=8081,
            args="test-args",
            data_file=None,
            helm_chart="test-helm-chart",
            helm_service_name="test-helm-service-name",
        )


def test_nvcf_helper_deploy_function_no_data(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that nvcf_helper_deploy_function returns the expected dictionary on success.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.

    Returns:
        None

    """
    mock_ncg_client = MagicMock()
    mock_ncg_client.post.return_value = NVCFResponse(
        {"status": 200, "deployment": {"id": "test-id", "version": "test-version"}}
    )

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.ncg_api_hdl = mock_ncg_client

    result = nvcf_helper.nvcf_helper_deploy_function(
        funcid="test-id",
        version="test-version",
        backend="test-backend",
        gpu="test-gpu",
        instance="test-instance",
        data_file=None,
        min_instances=1,
        max_instances=2,
        max_concurrency=3,
    )

    assert result is not None
    assert result["id"] == "test-id"
    assert result["version"] == "test-version"
    assert result["status"] == ""
    assert result["errors"] == []

    call_data = {
        "deploymentSpecifications": [
            {
                "gpu": "test-gpu",
                "clusters": ["test-backend"],
                "maxInstances": 2,
                "minInstances": 1,
                "instanceType": "test-instance",
                "instanceCount": 1,
                "maxRequestConcurrency": 3,
                "preferredOrder": 99,
            }
        ]
    }
    mock_ncg_client.post.assert_called_once_with(
        "/v2/nvcf/deployments/functions/test-id/versions/test-version", data=call_data
    )


def test_nvcf_helper_deploy_function_with_data(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that nvcf_helper_deploy_function returns the expected dictionary on success.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.

    Returns:
        None

    """
    mock_ncg_client = MagicMock()
    mock_ncg_client.post.return_value = NVCFResponse(
        {"status": 200, "deployment": {"id": "test-id", "version": "test-version"}}
    )

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.ncg_api_hdl = mock_ncg_client

    tmp_dir = tmp_path / "test-data"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = tmp_dir / "test-data.json"
    with Path.open(tmp_file, "w") as f:
        json.dump(
            {
                "availabilityZones": ["test-availability-zone"],
                "configuration": {
                    "replicas": 1,
                    "metrics": {
                        "extraExternalLabels": {
                            "backend": "test-backend",
                            "function_id": "test-id",
                            "version_id": "test-version",
                            "gpu": "test-gpu",
                            "org": "test-org",
                        }
                    },
                },
                "clusters": ["test-backend"],
                "regions": ["test-region"],
                "attributes": {"test-attribute": "test-attribute-value"},
            },
            f,
        )

    result = nvcf_helper.nvcf_helper_deploy_function(
        funcid="test-id",
        version="test-version",
        backend="test-backend",
        gpu="test-gpu",
        instance="test-instance",
        data_file=str(tmp_file),
        min_instances=1,
        max_instances=2,
        max_concurrency=3,
    )

    call_data = {
        "deploymentSpecifications": [
            {
                "gpu": "test-gpu",
                "maxInstances": 2,
                "minInstances": 1,
                "instanceType": "test-instance",
                "instanceCount": 1,
                "maxRequestConcurrency": 3,
                "preferredOrder": 99,
                "availabilityZones": ["test-availability-zone"],
                "configuration": {
                    "replicas": 1,
                    "metrics": {
                        "extraExternalLabels": {
                            "backend": "test-backend",
                            "function_id": "test-id",
                            "version_id": "test-version",
                            "gpu": "test-gpu",
                            "org": "",
                        }
                    },
                },
                "clusters": ["test-backend"],
                "regions": ["test-region"],
                "attributes": {"test-attribute": "test-attribute-value"},
            }
        ]
    }

    assert result is not None

    mock_ncg_client.post.assert_called_once_with(
        "/v2/nvcf/deployments/functions/test-id/versions/test-version", data=call_data
    )


def test_nvcf_helper_deploy_function_fail(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that nvcf_helper_deploy_function returns the expected dictionary on success.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.

    Returns:
        None

    """
    mock_ncg_client = MagicMock()
    mock_ncg_client.post.return_value = None

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.ncg_api_hdl = mock_ncg_client

    with pytest.raises(RuntimeError):
        nvcf_helper.nvcf_helper_deploy_function(
            funcid="test-id",
            version="test-version",
            backend="test-backend",
            gpu="test-gpu",
            instance="test-instance",
            data_file=None,
            min_instances=1,
            max_instances=2,
            max_concurrency=3,
        )

    mock_ncg_client.post.return_value = NVCFResponse({"status": 500})
    nvcf_helper.ncg_api_hdl = mock_ncg_client

    with pytest.raises(RuntimeError):
        nvcf_helper.nvcf_helper_deploy_function(
            funcid="test-id",
            version="test-version",
            backend="test-backend",
            gpu="test-gpu",
            instance="test-instance",
            data_file=None,
            min_instances=1,
            max_instances=2,
            max_concurrency=3,
        )


def test_nvcf_helper_s3cred_function(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that nvcf_helper_s3cred_function returns the expected dictionary on success.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.

    Returns:
        None

    """
    mock_ncg_client = MagicMock()
    mock_ncg_client.put.return_value = None

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.ncg_api_hdl = mock_ncg_client

    tmp_dir = tmp_path / "test-data"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = tmp_dir / "test-data.json"
    with Path.open(tmp_file, "w") as f:
        json.dump(
            {
                "test-key": "test-value",
            },
            f,
        )

    with pytest.raises(RuntimeError):
        nvcf_helper.nvcf_helper_s3cred_function(
            funcid="test-id",
            version="test-version",
            s3credfile=str(tmp_file),
        )

    mock_ncg_client.put.return_value = NVCFResponse({"status": 500})

    with pytest.raises(RuntimeError):
        nvcf_helper.nvcf_helper_s3cred_function(
            funcid="test-id",
            version="test-version",
            s3credfile=str(tmp_file),
        )

    mock_ncg_client.put.return_value = NVCFResponse({"status": 200})
    nvcf_helper.ncg_api_hdl = mock_ncg_client

    nvcf_helper.nvcf_helper_s3cred_function(
        funcid="test-id",
        version="test-version",
        s3credfile=str(tmp_file),
    )

    mock_ncg_client.put.assert_called_with(
        "/v2/orgs//nvcf/secrets/functions/test-id/versions/test-version", data={"test-key": "test-value"}
    )


def test_nvcf_helper_invoke_batch_fail(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that nvcf_helper_invoke_batch fails in the correct places.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.

    Returns:
        None

    """
    mock_ncg_client = MagicMock()
    mock_ncg_client.post.return_value = None

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.ncg_api_hdl = mock_ncg_client

    tmp_dir = tmp_path / "test-data"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = tmp_dir / "test-data.json"

    with pytest.raises(ValueError):  # noqa: PT011
        nvcf_helper.nvcf_helper_invoke_batch(
            data_file=str(tmp_file),
            id_file="",
            job_variant_file="",
            ddir="",
            s3_config=None,
            legacy_cf=False,
            retry_cnt=2,
            retry_delay=300,
        )


@pytest.mark.parametrize(
    ("response", "exception"),
    [
        (None, RuntimeError),
        (NVCFResponse({"status": 500}), RuntimeError),
        (NVCFResponse({"status": 400, "timeout": True}), TimeoutError),
        (NVCFResponse({"status": 400, "detail": "test-detail"}), RuntimeError),
        (NVCFResponse({"status": 200, "headers": {}}), RuntimeError),
    ],
)
def test_nvcf_helper_invoke_function_failures(
    monkeypatch: MonkeyPatch, tmp_path: Path, response: NVCFResponse | None, exception: type[Exception]
) -> None:
    """Test that nvcf_helper_invoke_function fails in the correct places.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.
        response: The response object.
        exception: The exception to expect.

    Returns:
        None

    """
    mock_nvcf_client = MagicMock()
    mock_nvcf_client.post.return_value = response

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.nvcf_api_hdl = mock_nvcf_client

    with pytest.raises(exception):
        nvcf_helper.nvcf_helper_invoke_function(
            funcid="test-id",
            ddir="",
            version=None,
            data_file=None,
            prompt_file=None,
            asset_id=None,
            s3_config=None,
        )


def test_nvcf_helper_invoke_function_success(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that nvcf_helper_invoke_function returns the expected dictionary on success.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.

    Returns:
        None

    """
    mock_nvcf_client = MagicMock()
    mock_nvcf_client.post.return_value = NVCFResponse(
        {
            "status": 200,
            "headers": {"test-header": "test-value", "reqid": "test-reqid", "pct": "test-pct", "status": "test-status"},
        }
    )

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.nvcf_api_hdl = mock_nvcf_client

    tmp_dir = tmp_path / "test-data"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = tmp_dir / "test-data.json"
    with Path.open(tmp_file, "w") as f:
        json.dump(
            {
                "test-key": "test-value",
                "args": {
                    "captioning_prompt_text": "A photo of a cat",
                },
            },
            f,
        )

    tmp_prompt_file = tmp_dir / "test-prompt.json"
    with Path.open(tmp_prompt_file, "w") as f:
        json.dump(
            {
                "test-prompt": "test-prompt",
            },
            f,
        )

    result = nvcf_helper.nvcf_helper_invoke_function(
        funcid="test-id",
        ddir=str(tmp_dir),
        version=None,
        data_file=str(tmp_file),
        prompt_file=str(tmp_prompt_file),
        asset_id=None,
        s3_config=None,
    )

    assert result is not None
    assert result["status"] == "test-status"
    assert result["reqid"] == "test-reqid"
    assert mock_nvcf_client.post.call_args[0][0] == "/v2/nvcf/pexec/functions/test-id"
    assert mock_nvcf_client.post.call_args[1]["extra_head"] is None
    assert mock_nvcf_client.post.call_args[1]["addl_headers"]
    assert mock_nvcf_client.post.call_args[1]["enable_504"]

    # Test with location
    mock_nvcf_client.post.return_value = NVCFResponse(
        {
            "status": 200,
            "headers": {
                "test-header": "test-value",
                "reqid": "test-reqid",
                "pct": "test-pct",
                "status": "test-status",
                "location": "test-location",
            },
        }
    )

    mock_nvcf_client.download.return_value = None

    result = nvcf_helper.nvcf_helper_invoke_function(
        funcid="test-id",
        ddir=str(tmp_dir),
        version=None,
        data_file=str(tmp_file),
        prompt_file=None,
        asset_id=None,
        s3_config=None,
    )

    mock_nvcf_client.download.assert_called_once()


@pytest.mark.parametrize(
    ("response", "exception"),
    [
        (None, RuntimeError),
        (NVCFResponse({"status": 500}), RuntimeError),
        (NVCFResponse({"status": 400, "timeout": True}), TimeoutError),
        (NVCFResponse({"status": 400, "detail": "test-detail"}), RuntimeError),
    ],
)
def test_nvcf_helper_get_request_status_failures(
    monkeypatch: MonkeyPatch, tmp_path: Path, response: NVCFResponse | None, exception: type[Exception]
) -> None:
    """Test that nvcf_helper_get_request_status fails in the correct places.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.
        response: The response object.
        exception: The exception to expect.

    Returns:
        None

    """
    mock_nvcf_client = MagicMock()
    mock_nvcf_client.get.return_value = response

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.nvcf_api_hdl = mock_nvcf_client

    with pytest.raises(exception):
        nvcf_helper.nvcf_helper_get_request_status(reqid="test-reqid", ddir="")


def test_nvcf_helper_get_request_status_success(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that nvcf_helper_get_request_status returns the expected dictionary on success.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.

    Returns:
        None

    """
    mock_nvcf_client = MagicMock()
    mock_nvcf_client.get.return_value = NVCFResponse(
        {
            "status": 200,
            "headers": {
                "test-header": "test-value",
                "reqid": "test-reqid",
                "pct": "test-pct",
                "status": "test-status",
                "location": "test-location",
            },
        }
    )

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.nvcf_api_hdl = mock_nvcf_client
    mock_nvcf_client.download.return_value = None

    result = nvcf_helper.nvcf_helper_get_request_status(reqid="test-reqid", ddir=str(tmp_path))
    assert result is not None
    assert result["status"] == "test-status"
    assert result["reqid"] == "test-reqid"
    assert mock_nvcf_client.get.call_args[0][0] == "/v2/nvcf/pexec/status/test-reqid"
    mock_nvcf_client.download.assert_called_once()


def test_nvcf_helper_invoke_batch_success(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that nvcf_helper_invoke_batch runs successfully.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.

    Returns:
        None

    """
    mock_nvcf_client = MagicMock()
    mock_nvcf_client.post.return_value = NVCFResponse(
        {
            "status": 200,
            "headers": {
                "test-header": "test-value",
                "reqid": "test-reqid",
                "pct": "test-pct",
                "status": "test-status",
                "location": "test-location",
            },
        }
    )

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.nvcf_api_hdl = mock_nvcf_client

    tmp_dir = tmp_path / "test-data"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = tmp_dir / "test-data.json"
    with Path.open(tmp_file, "w") as f:
        json.dump(
            {
                "test-key": "test-value",
            },
            f,
        )

    tmp_id_file = tmp_dir / "test-id.json"
    with Path.open(tmp_id_file, "w") as f:
        json.dump(
            [{"func": "function-id-1", "vers": "version-1"}, {"func": "function-id-2", "vers": "version-2"}],
            f,
        )

    tmp_job_variant_file = tmp_dir / "test-job-variant.json"
    with Path.open(tmp_job_variant_file, "w") as f:
        json.dump(
            [
                {"input_file": "video1.mp4", "output_format": "mp4", "quality": "high"},
                {"input_file": "video2.mp4", "output_format": "avi", "quality": "medium"},
                {"input_file": "video3.mp4", "output_format": "mp4", "quality": "low"},
            ],
            f,
        )

    nvcf_helper.nvcf_helper_invoke_batch(
        data_file=str(tmp_file),
        id_file=str(tmp_id_file),
        job_variant_file=str(tmp_job_variant_file),
        ddir=str(tmp_dir),
    )


@pytest.mark.parametrize(
    ("response", "exception"),
    [
        (NVCFResponse({"status": 500}), RuntimeError),
        (NVCFResponse({"status": 400, "timeout": True}), TimeoutError),
        (NVCFResponse({"status": 400, "detail": "test-detail"}), RuntimeError),
    ],
)
def test_nvcf_helper_get_request_status_with_wait_get_failures(
    monkeypatch: MonkeyPatch, tmp_path: Path, response: NVCFResponse | None, exception: type[Exception]
) -> None:
    """Test that nvcf_helper_get_request_status_with_wait fails in the correct places.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.
        response: The response object.
        exception: The exception to expect.

    Returns:
        None

    """
    mock_nvcf_client = MagicMock()
    mock_nvcf_client.get.return_value = response

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.nvcf_api_hdl = mock_nvcf_client

    with pytest.raises(exception):
        nvcf_helper.nvcf_helper_get_request_status_with_wait(reqid="test-reqid", ddir="")


def test_nvcf_helper_get_request_status_with_wait_success(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that nvcf_helper_get_request_status_with_wait functions on success.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.

    """
    mock_nvcf_client = MagicMock()
    mock_nvcf_client.get.return_value = NVCFResponse(
        {
            "status": 200,
            "headers": {
                "test-header": "test-value",
                "reqid": "test-reqid",
                "pct": "test-pct",
                "status": "test-status",
                "location": "test-location",
                "zip": "test-zip",
            },
        }
    )

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.nvcf_api_hdl = mock_nvcf_client

    # Happy Path
    nvcf_helper.nvcf_helper_get_request_status_with_wait(reqid="test-reqid", ddir=str(tmp_path), wait=False)
    mock_nvcf_client.get.assert_called_once()


@pytest.mark.parametrize(
    ("response", "exception"),
    [
        (None, RuntimeError),
        (NVCFResponse({"status": 500, "detail": "test-detail"}), RuntimeError),
        (NVCFResponse({"status": 400, "timeout": True}), TimeoutError),
    ],
)
def test_nvcf_helper_terminate_request_failures(
    monkeypatch: MonkeyPatch, tmp_path: Path, response: NVCFResponse | None, exception: type[Exception]
) -> None:
    """Test that nvcf_helper_terminate_request fails in the correct places.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.
        response: The response object.
        exception: The exception to expect.

    """
    mock_nvcf_client = MagicMock()
    mock_nvcf_client.post.return_value = response

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.nvcf_api_hdl = mock_nvcf_client

    with pytest.raises(exception):
        nvcf_helper.nvcf_helper_terminate_request(reqid="test-reqid", funcid="test-funcid", version="test-version")


def test_nvcf_helper_terminate_request_success(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that nvcf_helper_terminate_request functions on success.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.

    """
    mock_nvcf_client = MagicMock()
    mock_nvcf_client.post.return_value = NVCFResponse(
        {
            "status": 200,
            "headers": {
                "test-header": "test-value",
                "reqid": "test-reqid",
                "pct": "test-pct",
                "status": "test-status",
                "location": "test-location",
            },
        }
    )

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.nvcf_api_hdl = mock_nvcf_client

    result = nvcf_helper.nvcf_helper_terminate_request(reqid="test-reqid", funcid="test-funcid", version="test-version")
    assert result == {"reqid": "test-reqid could not get termination status, please check logs"}


@pytest.mark.parametrize(
    ("response", "exception"),
    [
        (None, RuntimeError),
        (NVCFResponse({"status": 500, "detail": "test-detail"}), RuntimeError),
        (NVCFResponse({}), RuntimeError),
    ],
)
def test_nvcf_helper_delete_function(
    monkeypatch: MonkeyPatch, tmp_path: Path, response: NVCFResponse | None, exception: type[Exception]
) -> None:
    """Test that nvcf_helper_delete_function fails in the correct places.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.
        response: The response object.
        exception: The exception to expect.

    """
    mock_ncg_client = MagicMock()
    mock_ncg_client.delete.return_value = response

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.ncg_api_hdl = mock_ncg_client

    with pytest.raises(exception):
        nvcf_helper.nvcf_helper_delete_function(funcid="test-funcid", version="test-version")


@pytest.mark.parametrize(
    ("response", "exception"),
    [
        (None, RuntimeError),
        (NVCFResponse({"status": 500, "detail": "test-detail"}), RuntimeError),
        (NVCFResponse({}), RuntimeError),
    ],
)
def test_nvcf_helper_get_deployment_detail_failures(
    monkeypatch: MonkeyPatch, tmp_path: Path, response: NVCFResponse | None, exception: type[Exception]
) -> None:
    """Test that nvcf_helper_get_deployment_detail fails in the correct places.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.
        response: The response object.
        exception: The exception to expect.


    """
    mock_ncg_client = MagicMock()
    mock_ncg_client.get.return_value = response

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.ncg_api_hdl = mock_ncg_client

    with pytest.raises(exception):
        nvcf_helper.nvcf_helper_get_deployment_detail(funcid="test-funcid", version="test-version")


def test_nvcf_helper_get_deployment_detail_success(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that nvcf_helper_get_deployment_detail functions on success.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.

    """
    mock_ncg_client = MagicMock()
    mock_ncg_client.get.return_value = NVCFResponse(
        {
            "status": 200,
            "deployment": {
                "functionName": "test-function-name",
                "functionId": "test-function-id",
                "functionVersionId": "test-version",
                "functionStatus": "test-status",
                "sisRequestId": "test-sis-request-status",
                "error": "test-error",
            },
        }
    )

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.ncg_api_hdl = mock_ncg_client

    result = nvcf_helper.nvcf_helper_get_deployment_detail(funcid="test-funcid", version="test-version")
    assert result == {
        "Name": "test-function-name",
        "Id": "test-function-id",
        "Version": "test-version",
        "Status": "test-status",
        "Detail": [],
    }


def test_nvcf_helper_get_deployment_detail_unauthorized(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """A 401 with no detail (e.g. expired API key) must raise a coded auth error, not "None".

    Regression test: previously get_detail() stringified the missing detail to the literal "None",
    so this surfaced as ``RuntimeError: None`` and masked an expired PERF_NGC_NVCF_API_KEY.
    """
    mock_ncg_client = MagicMock()
    # Shape mirrors NvcfClient._handle_client_error for a 401 body without a "detail" field.
    mock_ncg_client.get.return_value = NVCFResponse({"status": 401, "issue": {"status": 401, "title": "Unauthorized"}})

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.ncg_api_hdl = mock_ncg_client

    with pytest.raises(RuntimeError) as exc_info:
        nvcf_helper.nvcf_helper_get_deployment_detail(funcid="test-funcid", version="test-version")

    message = str(exc_info.value)
    assert message != "None"
    assert "HTTP 401" in message
    assert "check API key" in message


@pytest.mark.parametrize(
    ("response", "exception"),
    [
        (None, RuntimeError),
        (NVCFResponse({"status": 500, "detail": "test-detail"}), RuntimeError),
        (NVCFResponse({}), RuntimeError),
    ],
)
def test_nvcf_helper_undeploy_function_failures(
    monkeypatch: MonkeyPatch, tmp_path: Path, response: NVCFResponse | None, exception: type[Exception]
) -> None:
    """Test that nvcf_helper_undeploy_function fails in the correct places.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.
        response: The response object.
        exception: The exception to expect.

    """
    mock_ncg_client = MagicMock()
    mock_ncg_client.delete.return_value = response

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.ncg_api_hdl = mock_ncg_client

    with pytest.raises(exception):
        nvcf_helper.nvcf_helper_undeploy_function(funcid="test-funcid", version="test-version", graceful=False)


def test_nvcf_helper_undeploy_function_success(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that nvcf_helper_undeploy_function functions on success.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.

    """
    mock_ncg_client = MagicMock()
    mock_ncg_client.delete.return_value = NVCFResponse(
        {
            "status": 200,
            "function": {
                "name": "test-function-name",
                "status": "test-status",
            },
        }
    )

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.ncg_api_hdl = mock_ncg_client

    result = nvcf_helper.nvcf_helper_undeploy_function(funcid="test-funcid", version="test-version", graceful=False)
    assert result == ("test-function-name", "test-status")


@pytest.mark.parametrize(
    ("response", "exception"),
    [
        (None, RuntimeError),
        (NVCFResponse({"status": 500, "detail": "test-detail"}), RuntimeError),
        (NVCFResponse({"status": 400, "timeout": True}), TimeoutError),
    ],
)
def test_nvcf_helper_get_request_status_new_failures(
    monkeypatch: MonkeyPatch, tmp_path: Path, response: NVCFResponse | None, exception: type[Exception]
) -> None:
    """Test that nvcf_helper_get_request_status_new fails in the correct places.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.
        response: The response object.
        exception: The exception to expect.

    """
    mock_nvcf_client = MagicMock()
    mock_nvcf_client.post.return_value = response

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.nvcf_api_hdl = mock_nvcf_client

    with pytest.raises(exception):
        nvcf_helper._nvcf_helper_get_request_status_new(
            reqid="test-reqid", funcid="test-funcid", version="test-version"
        )


def test_nvcf_helper_get_request_status_new_success(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that _nvcf_helper_get_request_status_new functions on success.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The temporary path object.

    """
    mock_nvcf_client = MagicMock()
    mock_nvcf_client.post.return_value = NVCFResponse(
        {
            "status": 200,
            "headers": {
                "test-header": "test-value",
                "reqid": "test-reqid",
                "pct": 100,
                "status": "test-status",
            },
        }
    )

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", team="", timeout=15)
    nvcf_helper.nvcf_api_hdl = mock_nvcf_client

    result = nvcf_helper._nvcf_helper_get_request_status_new(
        reqid="test-reqid", funcid="test-funcid", version="test-version"
    )
    assert result is not None
    assert result["reqid"] == "test-reqid"
    assert result["status"] == "test-status"


@pytest.mark.parametrize(
    ("status", "reason", "body", "context", "expected_content"),
    [
        (
            401,
            "Unauthorized",
            {
                "requestStatus": {
                    "requestId": "12345678-109856",
                    "statusCode": "UNAUTHORIZED",
                    "statusDescription": "Authentication Failed",
                }
            },
            "function with name 'test-function'",
            ["Details: Authentication Failed"],
        ),
        (
            400,
            "Bad Request",
            {
                "requestStatus": {
                    "requestId": "abcd1234-567890",
                    "statusCode": "INVALID_REQUEST",
                    "statusDescription": "GPU type L4 and instance type AWS.L40.foo not found in cluster vfm-eks",
                }
            },
            "function with name 'test-function'",
            ["Details: GPU type L4 and instance type AWS.L40.foo not found in cluster vfm-eks"],
        ),
        (
            400,
            "Bad Request",
            {
                "cause": (
                    "403 FORBIDDEN, ProblemDetail[type='urn:kaizen:problem-details:forbidden', "
                    "title='Forbidden', status=403, detail='From upstream endpoint "
                    "'https://api.ngc.nvidia.com/v2/org/example-org/team/dev/helm-charts/"
                    "cosmos-curator/versions/2.1.1': "
                    '{"requestStatus":{"statusCode":"FORBIDDEN","statusDescription":"Access Denied",'
                    '"requestId":"abcd1234-365006"}}\', '
                    "instance='null', properties='null']"
                ),
                "detail": "Function '12345678-1234-5678-9abc-123456789abc': Invalid artifact provided",
                "instance": "/v2/nvcf/accounts/test-account-id/functions",
                "status": 400,
                "title": "Bad Request",
                "type": "urn:kaizen:problem-details:bad-request",
            },
            "function with name 'test-function'",
            ["Function '12345678-1234-5678-9abc-123456789abc': Invalid artifact provided"],
        ),
        (
            400,
            "Bad Request",
            {
                "cause": (
                    "Account 'test-account-id', Function '11111111-2222-3333-4444-555555555555', "
                    "Version 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee': Missing CONTAINER registry for hostname 'foo.tgz'"
                ),
                "detail": (
                    "Account 'test-account-id', Function '11111111-2222-3333-4444-555555555555', "
                    "Version 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee': Missing CONTAINER registry for hostname 'foo.tgz'"
                ),
                "instance": "/v2/nvcf/accounts/test-account-id/functions",
                "status": 400,
                "title": "Bad Request",
                "type": "urn:kaizen:problem-details:bad-request",
            },
            "function with name 'test-function'",
            ["Account 'test-account-id'", "Missing CONTAINER registry"],
        ),
        (
            400,
            "Bad Request",
            {
                "detail": (
                    "JSON parse error: Cannot deserialize value of type "
                    "`java.util.HashSet<com.nvidia.kaizen.nvcf.rest.function.management.dto.ArtifactDto>` "
                    "from String value (token `JsonToken.VALUE_STRING`)"
                ),
                "instance": "/v2/nvcf/accounts/test-account-id/functions",
                "status": 400,
                "title": "Bad Request",
                "type": "about:blank",
            },
            "function with name 'test-function'",
            ["Details: JSON parse error", "ArtifactDto"],
        ),
        (
            400,
            "Bad Request",
            {
                "detail": (
                    "Function id '12345678-1234-5678-9abc-123456789abc', "
                    "version 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee': The configuration field in Gpu specification "
                    "should be empty for container based functions."
                ),
                "instance": (
                    "/v2/nvcf/accounts/test-account-id/deployments/functions/"
                    "12345678-1234-5678-9abc-123456789abc/versions/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
                ),
                "status": 400,
                "title": "Bad Request",
                "type": "urn:kaizen:problem-details:bad-request",
            },
            "function with name 'test-function'",
            [
                "Function id '12345678-1234-5678-9abc-123456789abc'",
                "configuration field in Gpu specification should be empty",
            ],
        ),
        (
            400,
            "Bad Request",
            {
                "detail": (
                    "Function id '11111111-2222-3333-4444-555555555555', "
                    "version 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee': Status DEPLOYING, "
                    "use PUT to update current deployment"
                ),
                "instance": (
                    "/v2/nvcf/accounts/test-account-id/deployments/functions/"
                    "11111111-2222-3333-4444-555555555555/versions/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
                ),
                "status": 400,
                "title": "Bad Request",
                "type": "urn:kaizen:problem-details:bad-request",
            },
            "function with name 'test-function'",
            ["Function id '11111111-2222-3333-4444-555555555555'", "Status DEPLOYING, use PUT to update"],
        ),
        (
            400,
            "Bad Request",
            {
                "detail": (
                    "Function id '11111111-2222-3333-4444-555555555555', "
                    "version 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee': Function already has a deployment with errors"
                ),
                "instance": (
                    "/v2/nvcf/accounts/test-account-id/deployments/functions/"
                    "11111111-2222-3333-4444-555555555555/versions/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
                ),
                "status": 400,
                "title": "Bad Request",
                "type": "urn:kaizen:problem-details:bad-request",
            },
            (
                "Function with Id '11111111-2222-3333-4444-555555555555' "
                "and version 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'"
            ),
            ["Function id '11111111-2222-3333-4444-555555555555'", "Function already has a deployment with errors"],
        ),
        (
            404,
            "Not Found",
            {
                "detail": (
                    "Function id '12345678-1234-5678-9abc-123456789abc': "
                    "Version 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee' not found"
                ),
                "instance": (
                    "/v2/nvcf/accounts/test-account-id/functions/"
                    "12345678-1234-5678-9abc-123456789abc/versions/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
                ),
                "status": 404,
                "title": "Not Found",
                "type": "urn:kaizen:problem-details:not-found",
            },
            "function with ID '12345678-1234-5678-9abc-123456789abc'",
            [
                (
                    "Function id '12345678-1234-5678-9abc-123456789abc': "
                    "Version 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee' not found"
                )
            ],
        ),
        (
            409,
            "Conflict",
            {
                "cause": (
                    "409 CONFLICT, ProblemDetail[type='urn:kaizen:problem-details:exists', "
                    "title='Conflict', status=409, detail='From upstream endpoint "
                    "'https://spot.gdn.nvidia.com/v1/si': "
                    '{"error":"There are no available clusters with capacity for  L4 GPU or '
                    'AWS.GPU.L4_1x instance type"}'
                    "', instance='null', properties='null']"
                ),
                "detail": (
                    "Function id '11111111-2222-3333-4444-555555555555', "
                    "version 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee': Failed to deploy, "
                    "reverting state to 'INACTIVE': 409 CONFLICT, "
                    "ProblemDetail[type='urn:kaizen:problem-details:exists', "
                    "title='Conflict', status=409, detail='From upstream endpoint 'https://spot.gdn.nvidia.com/v1/si': "
                    '{"error":"There are no available clusters with capacity for  L4 GPU or '
                    'AWS.GPU.L4_1x instance type"}'
                    "', instance='null', properties='null']"
                ),
                "instance": (
                    "/v2/nvcf/accounts/test-account-id/deployments/functions/"
                    "11111111-2222-3333-4444-555555555555/versions/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
                ),
                "status": 409,
                "title": "Conflict",
                "type": "urn:kaizen:problem-details:exists",
            },
            "function with ID '11111111-2222-3333-4444-555555555555'",
            ["Function id '11111111-2222-3333-4444-555555555555'", "Failed to deploy, reverting state to 'INACTIVE'"],
        ),
        (
            409,
            "Conflict",
            {
                "issue": {
                    "type": "urn:kaizen:problem-details:exists",
                    "title": "Conflict",
                    "detail": (
                        "Function id '11111111-2222-3333-4444-555555555555', "
                        "version 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee': Failed to deploy, "
                        "reverting state to 'INACTIVE': 409 CONFLICT, "
                        "ProblemDetail[type='urn:kaizen:problem-details:exists', "
                        "title='Conflict', status=409, detail='From upstream endpoint "
                        "'https://spot.gdn.nvidia.com/v1/si': "
                        '{"error":"There are no available clusters with capacity for  L4 GPU or '
                        'AWS.GPU.L4_1x instance type"}'
                        "', instance='null', properties='null']"
                    ),
                    "instance": (
                        "/v2/nvcf/accounts/test-account-id/deployments/functions/"
                        "11111111-2222-3333-4444-555555555555/versions/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
                    ),
                    "cause": (
                        "409 CONFLICT, ProblemDetail[type='urn:kaizen:problem-details:exists', "
                        "title='Conflict', status=409, detail='From upstream endpoint "
                        "'https://spot.gdn.nvidia.com/v1/si': "
                        '{"error":"There are no available clusters with capacity for  L4 GPU or '
                        'AWS.GPU.L4_1x instance type"}'
                        "', instance='null', properties='null']"
                    ),
                },
                "status": 409,
            },
            "function with ID '11111111-2222-3333-4444-555555555555'",
            ["Function id '11111111-2222-3333-4444-555555555555'", "Failed to deploy, reverting state to 'INACTIVE'"],
        ),
    ],
)
def test_extract_nvcf_error_details(
    status: int, reason: str, body: dict[str, Any], context: str, expected_content: list[str]
) -> None:
    """Test _extract_nvcf_error_details with various response formats."""
    # Construct response_data from the separate components
    response_data = {"status": status, "reason": reason, **body}
    mock_response = NVCFResponse(response_data)
    funcid = body.get("detail", "").split("'")[1] if "Function id" in body.get("detail", "") else None
    result = _extract_nvcf_error_details(mock_response, context, funcid=funcid)

    for expected in expected_content:
        assert expected in result
