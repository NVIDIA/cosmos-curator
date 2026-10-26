# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run recovery scenarios in a fresh driver process that owns its small local Ray clusters.

The enrichment and append fresh-cluster scenarios restart Ray between attempts;
running them in a child process keeps those restarts from replacing the suite's
shared cluster.
"""

import subprocess
import sys

import ray
from coverage import Coverage
from ray.data.context import DataContext

from cosmos_curator.core.utils import environment

# Own both Ray clusters in a child process so restarting them cannot replace the
# suite's shared cluster. Save child coverage alongside pytest-cov's data when enabled.
_DRIVER = """
import importlib
import sys

import coverage

module, function, data_file, *args = sys.argv[1:]
cov = coverage.Coverage(data_file=data_file, data_suffix=True) if data_file else None
if cov is not None:
    cov.start()
try:
    getattr(importlib.import_module(module), function)(*args)
finally:
    if cov is not None:
        cov.stop()
        cov.save()
"""


def run_in_fresh_driver(target: str, *args: str, timeout: float = 240) -> None:
    """Run ``module:function`` with string arguments in a new driver process that owns its Ray clusters."""
    module, function = target.split(":")
    active_coverage = Coverage.current()
    data_file = active_coverage.get_option("run:data_file") if active_coverage is not None else ""
    result = subprocess.run(  # noqa: S603 -- fixed driver with test-owned arguments
        [sys.executable, "-c", _DRIVER, module, function, str(data_file or ""), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def start_local_cluster(num_cpus: int, *, io_slots: int = 1) -> None:
    """Start a small local cluster, optionally without IO slots to prove a plan needs none."""
    ray.init(
        address="local",
        num_cpus=num_cpus,
        num_gpus=0,
        resources={environment.CURATOR_IO_RESOURCE_NAME: io_slots} if io_slots else None,
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=128 * 1024**2,
    )
    DataContext.get_current().enable_progress_bars = False
