# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""UUID utility functions for validation and manipulation."""

import uuid


def is_uuid(value: str) -> bool:
    """Check if a string is a valid UUID.

    Args:
        value: String to validate

    Returns:
        True if value is a valid UUID, False otherwise

    """
    try:
        uuid.UUID(value)
    except (ValueError, TypeError):
        return False
    else:
        return True
