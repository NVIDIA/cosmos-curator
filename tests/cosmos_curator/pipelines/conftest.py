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
"""Pytest configuration and fixtures for curator tests.

This module provides custom pytest functionality, including environment-specific
test filtering for running tests only in their respective conda environments.
"""

import os

import pytest
import torch

from cosmos_curator.core.interfaces.runner_interface import RunnerInterface
from tests.utils.sequential_runner import SequentialRunner


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """When the user runs `pytest -m env`, filter tests by environment.

    `@pytest.mark.env("<name>")` argument must matche the current
    CONDA_DEFAULT_ENV.  Everything else is deselected.
    """
    # Was the user's command-line marker expression exactly `env`?
    if (config.option.markexpr or "").strip() != "env":
        # User didn't ask for the env filter; do nothing special.
        return

    current_env = os.environ.get("CONDA_DEFAULT_ENV")
    if not current_env:
        # No active conda env ⇒ deselect all tests with the marker.
        marked_tests = [item for item in items if item.get_closest_marker("env") is not None]
        if marked_tests:
            config.hook.pytest_deselected(items=marked_tests)
            items[:] = [item for item in items if item.get_closest_marker("env") is None]
        return
    current_env = current_env.removeprefix("cosmos-curator:")

    selected, deselected = [], []
    for item in items:
        m = item.get_closest_marker("env")
        if m is None:
            # Test is not marked with @pytest.mark.env → drop it
            deselected.append(item)
            continue

        # Accept the test only if its first positional arg equals the env
        if m.args and m.args[0] == current_env:
            selected.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


def _get_available_gpus() -> int:
    """Get the number of available GPUs dynamically."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def pytest_configure_node(node) -> None:  # noqa: ANN001
    """Configure pytest-xdist worker setup with improved GPU isolation.

    This hook is called on the main process to configure each worker node.
    It assigns each worker to a specific GPU to avoid memory conflicts and
    ensures proper GPU memory management.

    Args:
        node: The pytest-xdist worker node object

    """
    # Extract worker ID from node name (e.g., "gw0", "gw1")
    worker_id = node.gateway.id
    num_gpus = _get_available_gpus()

    if num_gpus == 0:
        node.workerinput["cuda_visible_devices"] = ""
        return

    # Assign GPU based on worker ID
    gpu_id = int(worker_id.replace("gw", "")) % num_gpus

    # Pass the GPU assignment to the worker process
    node.workerinput["cuda_visible_devices"] = str(gpu_id)
    node.workerinput["worker_id"] = worker_id
    node.workerinput["num_gpus"] = num_gpus


def pytest_configure(config) -> None:  # noqa: ANN001
    """Configure pytest for GPU distribution when running with xdist.

    This hook is called once per worker process and sets the CUDA_VISIBLE_DEVICES
    environment variable based on the worker configuration.

    Args:
        config: The pytest configuration object

    """
    # Only apply GPU distribution when running with xdist
    if hasattr(config, "workerinput"):
        cuda_device = config.workerinput.get("cuda_visible_devices")

        if cuda_device is not None and cuda_device != "":
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device


@pytest.fixture(scope="session")
def sequential_runner() -> RunnerInterface:
    """Provide a SequentialRunner instance for testing without Ray overhead.

    This fixture is available for tests that want to avoid Ray initialization.
    Both unit tests and integration tests (marked with @pytest.mark.env) should
    use SequentialRunner to avoid starting Ray clusters during testing.

    Returns:
        RunnerInterface: A runner that executes stages sequentially for testing.

    """
    return SequentialRunner()
