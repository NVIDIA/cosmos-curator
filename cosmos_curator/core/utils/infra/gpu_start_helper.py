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

import gc
import os
import time

import pynvml  # type: ignore[import-untyped]
import torch
from loguru import logger

from cosmos_curator.core.utils.infra.tracing import TracedSpan

_START_UP_RETRIES = 3
_START_UP_RETRY_INTERVAL_S = 120


def _dump_gpu_info(
    stage_name: str,
    prefix: str,
    *,
    check_mem: bool = False,
    num_gpus: float = 0.0,
) -> None:
    try:
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible_devices is not None:
            log_line_prefix = f"{stage_name}-{prefix}"
            gpus = [int(x) for x in cuda_visible_devices.split(",")]
            pynvml.nvmlInit()
            num_retries = 0
            while num_retries <= _START_UP_RETRIES:
                num_retries += 1
                all_clean = True
                for idx in gpus:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    # get memory info
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_mem = mem_info.total
                    used_mem = mem_info.used
                    log_line = (
                        f"{log_line_prefix}: GPU-{idx} "
                        f"total_mem={total_mem / (1024**3):.0f}GB "
                        f"used_mem={used_mem / (1024**3):.0f}GB "
                    )
                    # dump individual process info
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    for proc in processes:
                        logger.info(
                            f"{log_line_prefix}: GPU-{idx} pid={proc.pid} "
                            f"used_mem={proc.usedGpuMemory / (1024**3):.0f} GB",
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
                    # check memory usage
                    fraction_free = 1.0 - used_mem / (total_mem + 1e-6)
                    fraction_lower_bound = min(1.0, num_gpus) * 0.9
                    if check_mem and fraction_free < fraction_lower_bound:
                        all_clean = False
                        log_line += f"memory usage: {fraction_free=:.2f} < {fraction_lower_bound=:.2f}"
                        logger.warning(log_line)
                    else:
                        logger.info(log_line)
                if not check_mem:
                    break
                if all_clean:
                    logger.info(f"{log_line_prefix} is clean to start")
                    break
                if num_retries <= _START_UP_RETRIES:
                    time.sleep(_START_UP_RETRY_INTERVAL_S)
                    continue
                logger.error(f"{log_line_prefix} is NOT clean to start")
            pynvml.nvmlShutdown()
        else:
            logger.warning(f"{stage_name}-{prefix}: CUDA_VISIBLE_DEVICES is not set ?")
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error getting device info w/ pynvml: {e}")


def gpu_stage_startup(stage_name: str, num_gpus: float, *, pre_setup: bool) -> None:
    """Set up a stage worker with the given number of GPUs.

    Args:
        stage_name: The name of the stage.
        num_gpus: The number of GPUs to use.
        pre_setup: Whether this is called before or after stage setup.

    """
    if pre_setup:
        logger.info(f"Setup {stage_name} worker (pid={os.getpid()}) with {num_gpus} GPUs")
    logline_prefix = "startup" if pre_setup else "post-setup"
    _dump_gpu_info(stage_name, logline_prefix, check_mem=pre_setup, num_gpus=num_gpus)


def gpu_stage_cleanup(stage_name: str) -> None:
    """Clean up a stage worker.

    Args:
        stage_name: The name of the stage.

    """
    logger.info(f"Cleanup {stage_name} worker (pid={os.getpid()})")
    gc.collect()
    torch.cuda.empty_cache()
    _dump_gpu_info(stage_name, "cleanup")
