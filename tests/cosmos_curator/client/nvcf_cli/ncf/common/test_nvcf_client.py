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
"""Test nvcf response and nvcf client functionality."""

import json
from unittest.mock import MagicMock

import pytest

from cosmos_curator.client.nvcf_cli.ncf.common.nvcf_client import (  # type: ignore[import-untyped]
    NvcfClient,
    NVCFResponse,
)

_OK = 200
_NOT_FOUND = 404
_UNAUTHORIZED = 401
_FORBIDDEN = 403
_PRECONDITION_FAILED = 412
_TOO_MANY_REQUESTS = 429
_INTERNAL_SERVER_ERROR = 500
_TIMEOUT = 504


def test_nvcf_response_properties() -> None:
    """Test nvcf client properties."""
    response = NVCFResponse({"status": 200, "body": "test"})
    assert response.status == _OK
    assert response.has_status
    assert not response.is_not_found
    assert not response.is_not_authorized
    assert not response.is_error
    assert not response.is_timeout

    response = NVCFResponse({"status": 404, "body": "test"})
    assert response.status == _NOT_FOUND
    assert response.has_status
    assert response.is_not_found
    assert not response.is_not_authorized
    assert response.is_error

    response = NVCFResponse({"status": 401, "body": "test"})
    assert response.status == _UNAUTHORIZED
    assert response.has_status
    assert not response.is_not_found
    assert response.is_not_authorized
    assert response.is_error

    response = NVCFResponse()
    assert response.status == _INTERNAL_SERVER_ERROR


def test_nvcf_response_get_term_status() -> None:
    """Test nvcf response get term status."""
    response = NVCFResponse({"term-status": "test"})
    assert response.get_term_status() == "test"

    response = NVCFResponse({"body-status": "test"})
    assert response.get_term_status() == "test"

    response = NVCFResponse()
    assert response.get_term_status() == ""


def test_nvcf_response_get_detail() -> None:
    """Test nvcf response get detail."""
    response = NVCFResponse({"detail": "test"})
    assert response.get_detail() == "test"

    response = NVCFResponse({"issue": {"detail": "test"}})
    assert response.get_detail() == "test"

    response = NVCFResponse()
    assert response.get_detail() is None


def test_nvcf_response_get_error() -> None:
    """Test nvcf response get error."""
    response = NVCFResponse({"status": 400})
    assert response.get_error("test") == "invalid request for test"
    response = NVCFResponse({"status": 401})
    assert response.get_error("test") == "operation not authorized for test"
    response = NVCFResponse({"status": 403})
    assert response.get_error("test") == "operation not allowed for test"
    response = NVCFResponse({"status": 404})
    assert response.get_error("test") == "test not found"
    response = NVCFResponse({"status": 412})
    assert response.get_error("test") == "test precondition failed"
    response = NVCFResponse({"status": 429})
    assert response.get_error("test") == "test too many requests"
    response = NVCFResponse({"status": 500})
    assert response.get_error("test") == "unknown error occurred: status=500"

    response = NVCFResponse()
    assert response.get_error("test") == "unexpected empty response"

    response = NVCFResponse({"status": 500, "requestStatus": {"statusDescription": "extra"}})
    assert "requestStatus::statusDescription='extra': extra" in response.get_error("test")


def test_get_success() -> None:
    """Test nvcf client get."""
    client = NvcfClient(logger=MagicMock(), url="testurl")
    assert client is not None

    # Mock the session's get method
    mock_response = MagicMock()
    mock_response.status_code = _OK
    mock_response.json.return_value = {"status": _OK}
    mock_response.text = json.dumps({"status": _OK})
    mock_response.headers = {}

    client.ses.get = MagicMock(return_value=mock_response)

    response = client.get("/test")
    assert response is not None
    assert response.status == _OK


def test_client_error() -> None:
    """Test nvcf client get error."""
    client = NvcfClient(logger=MagicMock(), url="")
    assert client is not None

    # Mock the session's get method
    mock_response = MagicMock()
    mock_response.status_code = _UNAUTHORIZED
    mock_response.json.return_value = {"status": _UNAUTHORIZED}
    mock_response.text = json.dumps({"status": _UNAUTHORIZED})

    client.ses.get = MagicMock(return_value=mock_response)
    client._handle_client_error = MagicMock(return_value=NVCFResponse({"status": _UNAUTHORIZED}))

    response = client.get("/test")
    assert response is not None
    assert response.status == _UNAUTHORIZED
    client._handle_client_error.assert_called()


def test_server_error() -> None:
    """Test nvcf client get server error."""
    client = NvcfClient(logger=MagicMock(), url="testurl")
    assert client is not None

    # Mock the session's get method
    mock_response = MagicMock()
    mock_response.status_code = _INTERNAL_SERVER_ERROR
    mock_response.json.return_value = {"status": _INTERNAL_SERVER_ERROR}
    mock_response.text = json.dumps({"status": _INTERNAL_SERVER_ERROR})

    client.ses.get = MagicMock(return_value=mock_response)

    with pytest.raises(RuntimeError):
        client.get("/test")


def test_get_timeout() -> None:
    """Test nvcf client timeout response."""
    client = NvcfClient(logger=MagicMock(), url="testurl")
    assert client is not None

    # Mock the session's get method
    mock_response = MagicMock()
    mock_response.status_code = _TIMEOUT
    mock_response.json.return_value = {"status": _TIMEOUT}

    client.ses.get = MagicMock(return_value=mock_response)

    response = client.get("/test")
    assert response is not None
    assert response.status == _TIMEOUT
    assert response.is_timeout


def test_nvcf_client_post() -> None:
    """Test nvcf client post."""
    client = NvcfClient(logger=MagicMock(), url="testurl")
    assert client is not None

    # Mock the session's post method
    mock_response = MagicMock()
    mock_response.status_code = _OK
    mock_response.json.return_value = {"status": _OK}
    mock_response.text = json.dumps({"status": _OK})
    mock_response.headers = {}

    client.ses.post = MagicMock(return_value=mock_response)

    response = client.post("/test", data={"test": "test"})
    assert response is not None
    assert response.status == _OK

    response = client.post("/test", data={"test": "test"}, timeout=1)
    assert response is not None
    assert response.status == _OK

    response = client.post("/test", data={"test": "test"}, timeout=1, enable_504=True)
    assert response is not None
    assert response.status == _OK


def test_nvcf_client_put() -> None:
    """Test nvcf client put."""
    client = NvcfClient(logger=MagicMock(), url="testurl")
    assert client is not None

    # Mock the session's put method
    mock_response = MagicMock()
    mock_response.status_code = _OK
    mock_response.json.return_value = {"status": _OK}
    mock_response.text = json.dumps({"status": _OK})
    mock_response.headers = {}

    client.ses.put = MagicMock(return_value=mock_response)

    response = client.put("/test", data={"test": "test"})
    assert response is not None
    assert response.status == _OK


def test_nvcf_client_delete() -> None:
    """Test nvcf client delete."""
    client = NvcfClient(logger=MagicMock(), url="testurl")
    assert client is not None

    # Mock the session's delete method
    mock_response = MagicMock()
    mock_response.status_code = _OK
    mock_response.json.return_value = {"status": _OK}
    mock_response.text = json.dumps({"status": _OK})
    mock_response.headers = {}

    client.ses.delete = MagicMock(return_value=mock_response)

    response = client.delete("/test")
    assert response is not None
    assert response.status == _OK


def test_nvcf_client_is_binary_content() -> None:
    """Test nvcf client _is_binary_content method."""
    client = NvcfClient(logger=MagicMock(), url="testurl")

    # Test binary content types
    binary_content_types = [
        "image/jpeg",
        "image/png",
        "image/gif",
        "application/zip",
        "application/x-zip-compressed",
        "application/pdf",
        "application/octet-stream",
        "binary/octet-stream",
        "IMAGE/JPEG",  # Test case insensitivity
        "APPLICATION/ZIP",
    ]

    for content_type in binary_content_types:
        mock_response = MagicMock()
        mock_response.headers = {"content-type": content_type}
        assert client._is_binary_content(mock_response) is True

    # Test non-binary content types
    non_binary_content_types = [
        "application/json",
        "text/plain",
        "text/html",
        "application/xml",
        "text/xml",
        "application/javascript",
        "",  # Empty content type
        None,  # No content type
    ]

    for content_type in non_binary_content_types:
        mock_response = MagicMock()
        if content_type is None:
            mock_response.headers = {}
        else:
            mock_response.headers = {"content-type": content_type}
        assert client._is_binary_content(mock_response) is False


def test_nvcf_client_handle_binary_response() -> None:
    """Test nvcf client _handle_binary_response method."""
    client = NvcfClient(logger=MagicMock(), url="testurl")

    # Test successful binary response handling
    test_content = b"test binary content"
    mock_response = MagicMock()
    mock_response.status_code = _OK
    mock_response.content = test_content

    retval = {}
    client._handle_binary_response(mock_response, retval)

    # Verify the response was processed correctly
    assert retval["status"] == _OK
    assert "headers" in retval
    assert "zip" in retval
    assert retval["zip"].read() == test_content

    # Test with existing status and headers
    retval = {"status": _OK, "headers": {"existing": "header"}}
    client._handle_binary_response(mock_response, retval)

    # Verify existing values are preserved
    assert retval["status"] == _OK  # Should not be overwritten
    assert retval["headers"]["existing"] == "header"  # Should be preserved
    assert "zip" in retval


def test_nvcf_client_handle_binary_response_error() -> None:
    """Test nvcf client _handle_binary_response method with error."""
    client = NvcfClient(logger=MagicMock(), url="testurl")

    mock_response = MagicMock()
    mock_response.status_code = _OK

    class ProblematicContent:
        def __getattr__(self, name: str) -> None:
            if name == "__iter__":
                raise ValueError
            raise AttributeError

    mock_response.content = ProblematicContent()

    retval = {}

    with pytest.raises(ValueError, match="Failed to process binary response"):
        client._handle_binary_response(mock_response, retval)


def test_nvcf_client_handle_binary_response_empty_content() -> None:
    """Test nvcf client _handle_binary_response method with empty content."""
    client = NvcfClient(logger=MagicMock(), url="testurl")

    mock_response = MagicMock()
    mock_response.status_code = _OK
    mock_response.content = b""

    retval = {}
    client._handle_binary_response(mock_response, retval)

    assert retval["status"] == _OK
    assert "headers" in retval
    assert "zip" in retval
    assert retval["zip"].read() == b""


def test_nvcf_client_handle_binary_response_large_content() -> None:
    """Test nvcf client _handle_binary_response method with large content."""
    client = NvcfClient(logger=MagicMock(), url="testurl")

    large_content = b"x" * 1024 * 1024  # 1MB of data
    mock_response = MagicMock()
    mock_response.status_code = _OK
    mock_response.content = large_content

    retval = {}
    client._handle_binary_response(mock_response, retval)

    assert retval["status"] == _OK
    assert "headers" in retval
    assert "zip" in retval
    assert retval["zip"].read() == large_content


def test_nvcf_client_parse_json_response_success() -> None:
    """Test nvcf client _parse_json_response method with successful JSON."""
    client = NvcfClient(logger=MagicMock(), url="testurl")

    # Test successful JSON parsing
    test_data = {"key": "value", "number": 200, "list": [1, 2, 3]}
    mock_response = MagicMock()
    mock_response.status_code = _OK
    mock_response.text = '{"key": "value", "number": 200, "list": [1, 2, 3]}'
    mock_response.json.return_value = test_data

    retval = {}
    result = client._parse_json_response(mock_response, retval, ignore_if_not_json=False)

    # Verify the response was processed correctly
    assert retval["status"] == _OK
    assert "headers" in retval
    assert retval["key"] == "value"
    assert retval["number"] == _OK
    assert retval["list"] == [1, 2, 3]
    assert result == test_data


def test_nvcf_client_handle_client_error() -> None:
    """Test nvcf client _handle_client_error method."""
    client = NvcfClient(logger=MagicMock(), url="testurl")

    # Create a mock Response object
    mock_response = MagicMock()
    mock_response.status_code = _UNAUTHORIZED
    mock_response.reason = "Unauthorized"

    # Test with None data
    result = client._handle_client_error(mock_response, data=None)
    assert result["status"] == _UNAUTHORIZED
    assert result["issue"]["status"] == _UNAUTHORIZED
    assert result["issue"]["detail"] == "Unauthorized"
    assert result["issue"]["title"] == "Unauthorized"


def test_nvcf_client_handle_server_error() -> None:
    """Test nvcf client _handle_server_error method."""
    client = NvcfClient(logger=MagicMock(), url="testurl")

    # Test with JSON response containing detail
    mock_response = MagicMock()
    mock_response.status_code = _INTERNAL_SERVER_ERROR
    mock_response.headers = {}
    mock_response.json.return_value = {"detail": "Server error\\nMore details\\nFinal line"}
    mock_response.text = "Server error\\nMore details\\nFinal line"

    with pytest.raises(RuntimeError):
        client._handle_server_error(mock_response)


def test_nvcf_client_add_response_headers() -> None:
    """Test nvcf client _add_response_headers method."""
    client = NvcfClient(logger=MagicMock(), url="testurl")

    # Test with JSON response containing detail
    mock_response = MagicMock()
    mock_response.status_code = _OK
    mock_response.headers = {
        "nvcf-reqid": "test",
        "nvcf-status": "test-status",
        "location": "test-location",
        "CURATOR-PIPELINE-STATUS": "test-pipeline-status",
        "CURATOR-PIPELINE-PERCENT-COMPLETE": "100",
    }

    retval = {}
    client._add_response_headers(mock_response, retval)

    assert retval["status"] == _OK
    assert retval["headers"]["reqid"] == "test"
    assert retval["headers"]["status"] == "test-status"
    assert retval["headers"]["location"] == "test-location"
    assert retval["invoke-based-status"] == "test-pipeline-status"
    assert retval["headers"]["pct"] == "100"
