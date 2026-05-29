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
"""Test benchmarks/secrets.py."""

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.secrets import _get_bearer_token, _get_secrets_from_env


@patch("benchmarks.secrets.os.environ")
@patch("benchmarks.secrets.os.getenv")
@pytest.mark.parametrize(
    ("env_vars", "getenv_side_effect", "environ_values", "raises", "expected_result"),
    [
        # Success case - all variables set
        (
            {"api_key": "API_KEY_ENV", "secret": "SECRET_ENV", "region": "REGION_ENV"},
            {
                "API_KEY_ENV": "test_api_key",
                "SECRET_ENV": "test_secret",
                "REGION_ENV": "us-east-1",
            }.get,
            {"API_KEY_ENV": "test_api_key", "SECRET_ENV": "test_secret", "REGION_ENV": "us-east-1"},
            nullcontext(),
            {"api_key": "test_api_key", "secret": "test_secret", "region": "us-east-1"},
        ),
        # Single missing variable
        (
            {"api_key": "API_KEY_ENV", "secret": "SECRET_ENV"},
            {"API_KEY_ENV": "test_api_key", "SECRET_ENV": None}.get,
            None,
            pytest.raises(ValueError, match="Environment variables secret are not set"),
            None,
        ),
        # Multiple missing variables
        (
            {"api_key": "API_KEY_ENV", "secret": "SECRET_ENV", "region": "REGION_ENV"},
            {"API_KEY_ENV": "test_api_key", "SECRET_ENV": None, "REGION_ENV": None}.get,
            None,
            pytest.raises(ValueError, match="Environment variables secret, region are not set"),
            None,
        ),
        # All variables missing
        (
            {"api_key": "API_KEY_ENV", "secret": "SECRET_ENV"},
            lambda _: None,
            None,
            pytest.raises(ValueError, match="Environment variables api_key, secret are not set"),
            None,
        ),
    ],
)
def test_get_secrets_from_env(  # noqa: PLR0913
    mock_getenv: MagicMock,
    mock_environ: MagicMock,
    env_vars: dict[str, str],
    getenv_side_effect: Callable[[str], str | None],
    environ_values: dict[str, str] | None,
    raises: AbstractContextManager[Any],
    expected_result: dict[str, str] | None,
) -> None:
    """Test _get_secrets_from_env function for various scenarios."""
    # Mock os.getenv
    mock_getenv.side_effect = getenv_side_effect

    # Mock os.environ only if we have values to return (success case)
    if environ_values:
        mock_environ.__getitem__ = lambda _, env_name: environ_values[env_name]

    # Act & Assert
    with raises:
        result = _get_secrets_from_env(env_vars)
        if expected_result is not None:
            assert result == expected_result


@patch("benchmarks.secrets.requests.post")
@pytest.mark.parametrize(
    ("num_failures", "should_succeed", "raises"),
    [
        (0, True, nullcontext()),  # Success on first attempt
        (1, True, nullcontext()),  # Fail once, then succeed
        (2, True, nullcontext()),  # Fail twice, then succeed
        (3, False, pytest.raises(Exception, match="RetryError")),  # Fail all attempts
    ],
)
def test_get_bearer_token(
    mock_post: MagicMock,
    *,
    num_failures: int,
    should_succeed: bool,
    raises: AbstractContextManager[Any],
) -> None:
    """Test get_bearer_token function with various failure/success scenarios."""
    # Arrange
    test_url = "https://example.com/token"
    test_api_key = "test_api_key"
    test_bearer_token = "test_bearer_token"  # noqa: S105

    # Create failure responses
    mock_responses = []
    for _ in range(num_failures):
        mock_response_fail = MagicMock()
        mock_response_fail.raise_for_status.side_effect = Exception("HTTP Error")
        mock_responses.append(mock_response_fail)

    # Add success response if needed
    if should_succeed:
        mock_response_success = MagicMock()
        mock_response_success.json.return_value = {"access_token": test_bearer_token}
        mock_responses.append(mock_response_success)
    else:
        # For failure case, return the same failing response
        mock_response_fail = MagicMock()
        mock_response_fail.raise_for_status.side_effect = Exception("HTTP Error")
        mock_post.return_value = mock_response_fail

    if should_succeed:
        mock_post.side_effect = mock_responses

    # Act & Assert
    with raises:
        result = _get_bearer_token(test_url, test_api_key)

        if should_succeed:
            assert result == test_bearer_token
