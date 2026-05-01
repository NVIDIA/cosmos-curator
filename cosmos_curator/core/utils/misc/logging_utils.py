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

"""Loguru logging helpers for tagged and prefixed log output."""

from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    import loguru


def make_tagged_logger(tag: str) -> "loguru.Logger":
    """Create a loguru logger that auto-prepends *tag* to every message."""
    normalized = tag.strip() if tag else ""

    def _prepend_tag(record: dict[str, Any]) -> None:
        if normalized:
            record["message"] = f"{normalized} {record['message']}"

    return logger.patch(_prepend_tag)  # type: ignore[arg-type]
