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
"""Shared helpers for container launch CLIs."""

import typer

_MIN_MOUNT_PARTS = 2
_MAX_MOUNT_PARTS = 3


SLIM_IMAGE_WARMUP_COMMAND = (
    'if [ -n "${COSMOS_CURATOR_SLIM_ENVS:-}" ]; then '
    'echo "Installing pixi environments: ${COSMOS_CURATOR_SLIM_ENVS}" && '
    "pixi install --frozen -e ${COSMOS_CURATOR_SLIM_ENVS//,/ -e }; "
    "fi"
)


def parse_extra_mounts(raw: str, *, description: str = "mount", max_parts: int = _MAX_MOUNT_PARTS) -> list[str]:
    """Parse comma-separated container mount specifications.

    Each entry must be in ``HOST_PATH:CONTAINER_PATH`` or
    ``HOST_PATH:CONTAINER_PATH:MODE`` format.
    """
    if not raw:
        return []

    mounts: list[str] = []
    for raw_entry in raw.split(","):
        stripped = raw_entry.strip()
        if not stripped:
            continue
        parts = stripped.split(":")
        if len(parts) < _MIN_MOUNT_PARTS or len(parts) > max_parts:
            expected = "HOST_PATH:CONTAINER_PATH or HOST_PATH:CONTAINER_PATH:MODE"
            if max_parts > _MAX_MOUNT_PARTS:
                expected += " or HOST_PATH:CONTAINER_PATH:FSTYPE:OPTIONS"
            msg = f"Invalid {description} '{stripped}'. Expected {expected}"
            raise typer.BadParameter(msg)
        mounts.append(stripped)
    return mounts


def command_contains(command: str | list[str], needle: str) -> bool:
    """Return whether a shell command string or argv list contains a command token substring."""
    if isinstance(command, str):
        return needle in command
    return any(needle in arg for arg in command)
