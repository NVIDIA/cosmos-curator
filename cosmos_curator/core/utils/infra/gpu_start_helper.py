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

"""GPU helper."""

import contextlib
import gc
import os
import time

import pynvml  # type: ignore[import-untyped]
import torch
from loguru import logger

from cosmos_curator.core.utils.infra.tracing import TracedSpan

_START_UP_RETRIES = 12
_START_UP_RETRY_INTERVAL_S = 90


class GpuNotCleanError(RuntimeError):
    """Raised when a GPU is not clean enough at stage startup to load the model.

    The actor pool's ``num_setup_attempts_python`` retry loop catches this and re-spawns the
    actor on a fresh placement, which is the only thing that can recover from a leaked CUDA
    context squatting on the assigned GPU.
    """


def _required_free_fraction(num_gpus: float, expected_free_fraction: float | None) -> float:
    """Compute the minimum acceptable free-memory fraction for the cleanliness check.

    When the caller knows what fraction the downstream loader will require (e.g. vLLM's
    ``gpu_memory_utilization``), it passes that as ``expected_free_fraction`` so we
    fail-fast here with a structured error rather than letting the loader fail with a
    cryptic message a few seconds later.

    Default behavior (``expected_free_fraction is None``) preserves the original
    ``min(1.0, num_gpus) * 0.9`` heuristic so non-vLLM stages keep their existing
    looser tolerance.
    """
    if expected_free_fraction is None:
        return min(1.0, num_gpus) * 0.9
    # Cap at 1.0 in case a caller passes ``utilization + headroom`` that sums >1.0
    return min(1.0, expected_free_fraction)


def _check_one_gpu(
    idx: int,
    log_line_prefix: str,
    *,
    check_mem: bool,
    num_gpus: float,
    expected_free_fraction: float | None,
) -> bool:
    """Dump info for a single GPU. Returns True if the GPU is clean enough to start on."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_mem = mem_info.total
    used_mem = mem_info.used
    log_line = (
        f"{log_line_prefix}: GPU-{idx} total_mem={total_mem / (1024**3):.0f}GB used_mem={used_mem / (1024**3):.0f}GB "
    )
    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    for proc in processes:
        logger.info(
            f"{log_line_prefix}: GPU-{idx} pid={proc.pid} used_mem={proc.usedGpuMemory / (1024**3):.0f} GB",
        )
        TracedSpan.current().add_event(
            "gpu_info",
            attributes={
                "gpu_idx": idx,
                "pid": proc.pid,
                "used_mem_gb": proc.usedGpuMemory / (1024**3),
                "total_mem_gb": total_mem / (1024**3),
            },
        )
    fraction_free = 1.0 - used_mem / (total_mem + 1e-6)
    fraction_lower_bound = _required_free_fraction(num_gpus, expected_free_fraction)
    if check_mem and fraction_free < fraction_lower_bound:
        log_line += f"memory usage: {fraction_free=:.3f} < {fraction_lower_bound=:.3f}"
        # Ghost-memory case: device-level memory is held but no compute process is attributed.
        # Usually means a leaked CUDA context from a process that has already exited (driver
        # didn't reclaim) or a process in a different PID namespace. Retries probably won't
        # clear it on this GPU.
        if not processes:
            log_line += " (no compute processes attributed - likely ghost CUDA context)"
        logger.warning(log_line)
        return False

    logger.info(log_line)
    return True


def _dump_gpu_info(
    stage_name: str,
    prefix: str,
    *,
    check_mem: bool = False,
    num_gpus: float = 0.0,
    expected_free_fraction: float | None = None,
) -> None:
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible_devices is None:
        logger.warning(f"{stage_name}-{prefix}: CUDA_VISIBLE_DEVICES is not set ?")
        return

    log_line_prefix = f"{stage_name}-{prefix}"
    try:
        gpus = [int(x) for x in cuda_visible_devices.split(",")]
        pynvml.nvmlInit()
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error getting device info w/ pynvml: {e}")
        return

    try:
        all_clean = True
        for attempt in range(1, _START_UP_RETRIES + 2):
            all_clean = all(
                _check_one_gpu(
                    idx,
                    log_line_prefix,
                    check_mem=check_mem,
                    num_gpus=num_gpus,
                    expected_free_fraction=expected_free_fraction,
                )
                for idx in gpus
            )
            if not check_mem or all_clean:
                break
            if attempt <= _START_UP_RETRIES:
                time.sleep(_START_UP_RETRY_INTERVAL_S)

        if check_mem:
            if all_clean:
                logger.info(f"{log_line_prefix} is clean to start")
            else:
                msg = (
                    f"{log_line_prefix} is NOT clean to start after {_START_UP_RETRIES} retries; "
                    f"failing setup so the actor pool can re-spawn this worker on a different GPU"
                )
                logger.error(msg)
                raise GpuNotCleanError(msg)
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


def gpu_stage_startup(
    stage_name: str,
    num_gpus: float,
    *,
    pre_setup: bool,
    expected_free_fraction: float | None = None,
) -> None:
    """Set up a stage worker with the given number of GPUs.

    Args:
        stage_name: The name of the stage.
        num_gpus: The number of GPUs to use.
        pre_setup: Whether this is called before or after stage setup.
        expected_free_fraction: Optional minimum free-memory fraction the stage's
            downstream loader (e.g. vLLM's ``gpu_memory_utilization``) will require.
            When provided, the cleanliness check uses this value instead of the
            default 0.9 heuristic - tightening the bar so we fail-fast with a
            structured ``GpuNotCleanError`` rather than letting the loader fail
            seconds later with a cryptic memory error. Pass the loader's
            requirement plus a small headroom (e.g. ``+0.01``) to absorb
            measurement noise. Ignored when ``pre_setup`` is False.

    """
    if pre_setup:
        logger.info(f"Setup {stage_name} worker (pid={os.getpid()}) with {num_gpus} GPUs")
    logline_prefix = "startup" if pre_setup else "post-setup"
    _dump_gpu_info(
        stage_name,
        logline_prefix,
        check_mem=pre_setup,
        num_gpus=num_gpus,
        expected_free_fraction=expected_free_fraction,
    )


def gpu_stage_cleanup(stage_name: str) -> None:
    """Clean up a stage worker.

    Args:
        stage_name: The name of the stage.

    """
    logger.info(f"Cleanup {stage_name} worker (pid={os.getpid()})")
    gc.collect()
    torch.cuda.empty_cache()
    _dump_gpu_info(stage_name, "cleanup")
