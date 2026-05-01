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
"""Test benchmarks/cloudevent.py."""

from contextlib import AbstractContextManager, nullcontext
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.cloudevent import make_cloudevent, push_cloudevent
from benchmarks.secrets import KratosSecrets


@patch("benchmarks.cloudevent.requests.post")
@pytest.mark.parametrize(
    ("should_raise_error", "raises"),
    [
        (False, nullcontext()),  # Success case
        (True, pytest.raises(Exception, match="RetryError")),  # Error case
    ],
)
def test_push_cloudevent(
    mock_post: MagicMock, *, should_raise_error: bool, raises: AbstractContextManager[Any]
) -> None:
    """Test push_cloudevent function for both success and error scenarios."""
    test_cloudevent = {
        "specversion": "1.0",
        "id": "test-id",
        "source": "test-source",
        "type": "test-type",
        "data": {"key": "value"},
    }
    test_endpoint = "https://example.com/cloudevents"
    test_bearer_token = "test_bearer_token"  # noqa: S105
    test_secrets = KratosSecrets(api_key="test_api_key", bearer_token=test_bearer_token)

    # Mock
    mock_response = MagicMock()
    if should_raise_error:
        mock_response.raise_for_status.side_effect = Exception("RetryError")
    mock_post.return_value = mock_response

    # Act
    with raises:
        push_cloudevent(test_cloudevent, test_endpoint, test_secrets)

    # Assert
    # post may be called more than once due to retries
    assert mock_post.call_count >= 1


@patch("benchmarks.cloudevent.datetime")
@patch("benchmarks.cloudevent.uuid")
def test_make_cloudevent(mock_uuid: MagicMock, mock_datetime: MagicMock) -> None:
    """Test make_cloudevent function."""
    test_data = {"metric": "value", "count": 42}
    test_uuid_id = "test-uuid-id"
    test_uuid_source = "test-uuid-source"
    test_timestamp = "2023-01-01T12:00:00.000000Z"

    # Mock UUID generation
    mock_uuid.uuid4.side_effect = [
        MagicMock(__str__=lambda _: test_uuid_id),  # First call for id
        MagicMock(__str__=lambda _: test_uuid_source),  # Second call for source
    ]

    # Mock datetime
    mock_now = MagicMock()
    mock_now.strftime.return_value = test_timestamp
    mock_datetime.now.return_value = mock_now

    # Act
    result = make_cloudevent(test_data)

    # Assert
    expected_cloudevent = {
        "specversion": "1.0",
        "id": test_uuid_id,
        "time": test_timestamp,
        "source": f"cosmos-curator-{test_uuid_source}",
        "type": "performance-benchmark",
        "subject": "nvcf-performance-metrics",
        "data": test_data,
    }

    assert result == expected_cloudevent
