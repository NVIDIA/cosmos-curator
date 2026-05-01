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
"""Provide utilities for getting hardware info."""

import pathlib
import shutil
import socket

import attrs
from loguru import logger

try:
    import gpustat  # type: ignore[import-untyped]

    HAS_PYNVML = True
except Exception:  # noqa: BLE001
    HAS_PYNVML = False


@attrs.define()
class GPUInfo:
    """Hold various information about available GPUs."""

    id: int
    name: str
    load: float
    memory_used: int
    memory_total: int
    temperature: int


def get_gpu_infos() -> list[GPUInfo]:
    """Get all GPU information."""
    # gpustat will bug out if the machine does not have access to CUDA.
    # We use a quick hack to see if cuda is available to avoid calling gpustat
    if not HAS_PYNVML:
        return []

    gpu_stats = gpustat.GPUStatCollection.new_query()
    gpu_info_list = []

    for gpu in gpu_stats:
        gpu_info = GPUInfo(
            id=gpu.entry["index"],
            name=gpu.entry["name"],
            load=gpu.entry["utilization.gpu"],
            memory_used=gpu.entry["memory.used"],
            memory_total=gpu.entry["memory.total"],
            temperature=gpu.entry["temperature.gpu"],
        )
        gpu_info_list.append(gpu_info)

    return gpu_info_list


def print_disk_path_info(disk_path: pathlib.Path) -> None:
    """Print disk usage information for the given path.

    Args:
        disk_path: The path to the disk to check.

    """
    hostname = socket.gethostname()
    try:
        disk_info = shutil.disk_usage(disk_path)
        logger.info(
            f"node {hostname}:{disk_path} total={disk_info.total / (2**30):.0f}GB free={disk_info.free / (2**30):.0f}GB"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to get disk usage for {hostname}:{disk_path}: {e!s}")
