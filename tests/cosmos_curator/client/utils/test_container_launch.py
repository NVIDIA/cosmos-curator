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
"""Tests for shared container launch helpers."""

from cosmos_curator.client.utils.container_launch import command_contains


def test_command_contains_searches_shell_command_string() -> None:
    """A shell command string should match command substrings."""
    assert command_contains(
        "pixi run --as-is python -m cosmos_curator.core.managers.model_cli download",
        "model_cli",
    )


def test_command_contains_searches_argv_entries() -> None:
    """An argv list should match substrings within individual entries."""
    assert command_contains(
        ["pixi", "run", "--as-is", "python", "-m", "cosmos_curator.core.managers.model_cli", "download"],
        "model_cli",
    )


def test_command_contains_rejects_absent_substring() -> None:
    """Commands without the substring should not match."""
    assert not command_contains(["python", "-m", "cosmos_curator.pipelines.examples.hello_world_pipeline"], "model_cli")
