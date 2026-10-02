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

"""Shared grammar for episode result keys used in versioned Lance field names."""

from re import fullmatch
from typing import Final

RESULT_KEY_PATTERN: Final = r"^[a-z][a-z0-9_]{0,47}$"


def is_valid_result_key(value: str) -> bool:
    """Return whether a result key is one safe deterministic field-name segment."""
    return fullmatch(RESULT_KEY_PATTERN, value) is not None
