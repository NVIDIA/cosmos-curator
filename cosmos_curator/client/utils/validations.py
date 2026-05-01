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

"""Validate input parameters and data for NVCF operations.

This module provides validation functions for various input parameters used in NVCF operations,
including UUID validation, integer range validation, and address validation.
"""

import ipaddress
import socket
import uuid
from collections.abc import Callable, Iterable

from click import BadParameter


def validate_uuid(value: str | None) -> str | None:
    """Validate and format a UUID string.

    Args:
        value: The UUID string to validate

    Returns:
        The validated UUID string in standard format

    Raises:
        BadParameter: If the input is not a valid UUID

    """
    if value is None:
        return None

    try:
        return str(uuid.UUID(hex=value))

    except ValueError as e:
        error_msg = f"invalid UUID '{value}'"
        raise BadParameter(error_msg) from e


def validate_positive_integer(x: int) -> int:
    """Validate that a number is a positive integer.

    Args:
        x: The integer to validate

    Returns:
        The validated positive integer

    Raises:
        BadParameter: If the input is not a positive integer

    """
    if x <= 0:
        error_msg = f"'{x}' is not a positive integer."
        raise BadParameter(error_msg)

    return x


def validate_in(r: range, excludes: Iterable[int] = ()) -> Callable[[int], int]:
    """Create a validator function that checks if a number is within a range and not in excluded values.

    Args:
        r: The range to validate against
        excludes: Iterable of values that should be excluded from the range

    Returns:
        A validator function that takes an integer and returns it if valid

    Raises:
        BadParameter: If the input is in the excluded values or outside the range

    """

    def __validate(x: int) -> int:
        if x in excludes:
            error_msg = f"'{x}' is not allowed."
            raise BadParameter(error_msg)

        if x not in r:
            error_msg = f"'{x}' is not between {r.start} and {r.stop - 1}."
            raise BadParameter(error_msg)
        return x

    return __validate


def validate_address(value: str) -> str:
    """Validate a network address or hostname.

    Args:
        value: The address to validate

    Returns:
        The validated address

    Raises:
        BadParameter: If the input is not a valid address

    """
    if value.lower() == "localhost" or value.lower() == socket.gethostname().lower():
        return value

    try:
        ipaddress.ip_address(value)

    except ValueError as e:
        error_msg = f"Invalid address: {value}"
        raise BadParameter(error_msg) from e
    else:
        return value
