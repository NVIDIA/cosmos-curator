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

"""Test the validation functions."""

import socket
import uuid

import pytest
from click import BadParameter

from cosmos_curator.client.utils.validations import (  # type: ignore[import-untyped]
    validate_address,
    validate_in,
    validate_positive_integer,
    validate_uuid,
)


def test_validate_uuid() -> None:
    """Test the validate_uuid function."""
    fake_uuid = uuid.uuid4()

    # test valid uuid
    assert validate_uuid(str(fake_uuid)) == str(fake_uuid)

    # test None
    assert validate_uuid(None) is None

    # test invalid uuid
    with pytest.raises(BadParameter):
        validate_uuid("not-a-uuid")


def test_validate_positive_integer() -> None:
    """Test the validate_positive_integer function."""
    # test valid positive integer
    assert validate_positive_integer(1) == 1

    # test 0
    with pytest.raises(BadParameter):
        validate_positive_integer(0)

    # test negative integer
    with pytest.raises(BadParameter):
        validate_positive_integer(-1)


def test_validate_in() -> None:
    """Test the validate_in function."""
    r = range(1, 10)
    VALID_VALUE = 5
    assert validate_in(r)(VALID_VALUE) == VALID_VALUE
    with pytest.raises(BadParameter):
        validate_in(r)(0)
    with pytest.raises(BadParameter):
        validate_in(r)(10)
    with pytest.raises(BadParameter):
        validate_in(r, [5])(VALID_VALUE)


def test_validate_address() -> None:
    """Test the validate_address function."""
    # test valid address
    assert validate_address("127.0.0.1") == "127.0.0.1"
    assert validate_address("localhost") == "localhost"
    assert validate_address(socket.gethostname()) == socket.gethostname()

    # test invalid address
    with pytest.raises(BadParameter):
        validate_address("not-an-address")
