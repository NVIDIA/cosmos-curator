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

"""Handle and manage error conditions in NVCF operations.

This module defines custom exception classes for handling various error conditions
that may occur during NVIDIA Cloud Function operations.
"""

from typing import overload


class NotFoundError(Exception):
    """Exception raised when a requested resource is not found.

    This exception can be initialized in two ways:
    1. With a direct error message
    2. With an object kind and identifier
    """

    @overload
    def __init__(self, _message: str) -> None: ...

    @overload
    def __init__(self, _obj_kind: str, **kwargs: str) -> None: ...

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Initialize the NotFoundError exception.

        Args:
            *args: Positional arguments. If only one is provided, it's treated as a direct message.
            **kwargs: Keyword arguments. If provided, should contain exactly one key-value pair
                     representing the identifier type and value.

        """
        if not kwargs:
            super().__init__(next(iter([*args, "not found"])))

        elif len(kwargs) == 1:
            id_kind, obj_id = next(iter(kwargs.items()))
            error_msg = f"{args[0]} with {id_kind} '{obj_id}' not found"
            super().__init__(error_msg)

        else:
            error_msg = f"Expected 0 or 1 keyword arguments, got {len(kwargs)}"
            raise ValueError(error_msg)
