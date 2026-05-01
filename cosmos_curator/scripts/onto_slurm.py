#!/usr/bin/env python3
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
"""Start either a Ray head node or worker node.

The head node is started if the hostname matches the head node hostname,
otherwise a worker node is started.

# Using the CLI instead of `ray.init()`

It is preferable to call `ray.init()` to start the head and worker nodes.
However, we need control over the Ray configuration that is not offered by the
public `ray.init()` API.

Those parameters are available via the Ray CLI, so that is used instead.

There are two challenges with using the Ray CLI:

1. Streaming the output from the Ray CLI as it arrives.
2. Handling the case where the Ray CLI exits with a non-zero exit code in
   spite of success.

To handle the first challenge, `run_subprocess` uses `asyncio` to stream the
output as it arrives.

To handle the second challenge, the `start_ray` function catches the
exception, and if fewer seconds have elapsed than specified by the
`RAY_STOP_RETRIES_AFTER`, then the exception will be re-raised. Otherwise, the
exception is suppressed.

Without this, the retry decorator would interpret the bogus non-zero exit code
as a failure, and would attempt to restart the Ray worker after the Ray head
node has requested a shutdown. In practice, this can take 20-30 minutes to
complete.
"""

import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Self, TextIO

import attrs
import tenacity

logger = logging.getLogger(__name__)

# Default values for Ray configuration
_RAY_GCS_SERVER_PORT = 6379
_RAY_DASHBOARD_HOST = "0.0.0.0"  # noqa: S104
_RAY_DASHBOARD_PORT = 8265
_RAY_OBJECT_MANAGER_PORT = 8076
_RAY_NODE_MANAGER_PORT = 8077
_RAY_DASHBOARD_AGENT_GRPC_PORT = 52366
_RAY_RUNTIME_ENV_AGENT_PORT = 20267
_RAY_METRICS_EXPORT_PORT = 9002
_RAY_OBJECT_SPILL_PATH = Path(tempfile.gettempdir()) / "ray_spill"
_RAY_OBJECT_SPILL_BUFFER_SIZE = 1_000_000
_RAY_LOCAL_FS_CAPACITY_THRESHOLD = 0.90


@attrs.define
class SlurmEnv:
    """Environment variables and configuration for SLURM cluster execution."""

    num_nodes: int
    head_node: str
    nodename: str
    procid: int
    stop_retries_after: int

    def is_head_node(self) -> bool:
        """Check if the current node is the head node."""
        return self.nodename == self.head_node

    @classmethod
    def from_env(cls) -> Self:
        """Extract the environment variables SLURM_NNODES, HEAD_NODE_ADDR, and SLURMD_NODENAME.

        Returns:
            A dictionary containing the environment variables.

        Raises:
            ValueError: If any of the environment variables are not set.

        """
        REQUIRED_VARS = ["SLURM_NNODES", "HEAD_NODE_ADDR", "SLURMD_NODENAME", "RAY_STOP_RETRIES_AFTER", "SLURM_PROCID"]
        for var in REQUIRED_VARS:
            if var not in os.environ:
                error_message = f"Error: environment variable {var} is not set."
                raise ValueError(error_message)

        return cls(
            num_nodes=int(os.environ["SLURM_NNODES"]),
            head_node=os.environ["HEAD_NODE_ADDR"],
            nodename=os.environ["SLURMD_NODENAME"],
            procid=int(os.environ["SLURM_PROCID"]),
            stop_retries_after=int(os.environ["RAY_STOP_RETRIES_AFTER"]),
        )


@attrs.define
class RayObjectSpillingParams:
    """Parameters for Ray object spilling."""

    directory_path: Path = attrs.field(default=_RAY_OBJECT_SPILL_PATH)
    buffer_size: int = attrs.field(default=_RAY_OBJECT_SPILL_BUFFER_SIZE)


# Add a custom JSON encoder for Path objects
class PathJSONEncoder(json.JSONEncoder):
    """JSON Encoder that properly serializes Path objects to strings."""

    def default(self, obj: object) -> object:
        """Convert Path objects to strings during JSON serialization.

        Args:
            obj: Object to serialize

        Returns:
            JSON serializable representation of the object

        """
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


@attrs.define
class RayObjectSpillingConfig:
    """Configuration for Ray object spilling."""

    type: str = "filesystem"
    params: RayObjectSpillingParams = attrs.field(default=RayObjectSpillingParams())

    def to_json(self) -> str:
        """Convert the configuration to a JSON string."""
        return json.dumps(attrs.asdict(self), indent=None, cls=PathJSONEncoder)


@attrs.define
class RaySystemConfig:
    """Configuration for Ray system."""

    local_fs_capacity_threshold: float = _RAY_LOCAL_FS_CAPACITY_THRESHOLD
    object_spilling_config: RayObjectSpillingConfig = attrs.field(default=RayObjectSpillingConfig())

    def to_json(self) -> str:
        """Convert the configuration to a JSON string."""
        d = {
            "local_fs_capacity_threshold": self.local_fs_capacity_threshold,
            "object_spilling_config": self.object_spilling_config.to_json(),
        }
        return json.dumps(d, indent=None, cls=PathJSONEncoder)


@attrs.define
class RayConfig:
    """Configuration for Ray."""

    gcs_server_port: int = _RAY_GCS_SERVER_PORT
    object_manager_port: int = _RAY_OBJECT_MANAGER_PORT
    node_manager_port: int = _RAY_NODE_MANAGER_PORT
    runtime_env_agent_port: int = _RAY_RUNTIME_ENV_AGENT_PORT
    metrics_export_port: int = _RAY_METRICS_EXPORT_PORT
    dashboard_host: str = _RAY_DASHBOARD_HOST
    dashboard_port: int = _RAY_DASHBOARD_PORT
    dashboard_agent_grpc_port: int = _RAY_DASHBOARD_AGENT_GRPC_PORT
    system_config: RaySystemConfig = attrs.field(default=RaySystemConfig())

    @classmethod
    def from_env(cls) -> Self:
        """Create a Ray configuration from environment variables."""
        return cls(
            gcs_server_port=int(os.environ.get("RAY_GCS_SERVER_PORT", str(_RAY_GCS_SERVER_PORT))),
            dashboard_host=os.environ.get("RAY_DASHBOARD_HOST", _RAY_DASHBOARD_HOST),
            dashboard_port=int(os.environ.get("RAY_DASHBOARD_PORT", str(_RAY_DASHBOARD_PORT))),
            object_manager_port=int(os.environ.get("RAY_OBJECT_MANAGER_PORT", str(_RAY_OBJECT_MANAGER_PORT))),
            node_manager_port=int(os.environ.get("RAY_NODE_MANAGER_PORT", str(_RAY_NODE_MANAGER_PORT))),
            dashboard_agent_grpc_port=int(
                os.environ.get("RAY_DASHBOARD_AGENT_GRPC_PORT", str(_RAY_DASHBOARD_AGENT_GRPC_PORT))
            ),
            runtime_env_agent_port=int(os.environ.get("RAY_RUNTIME_ENV_AGENT_PORT", str(_RAY_RUNTIME_ENV_AGENT_PORT))),
            metrics_export_port=int(os.environ.get("XENNA_RAY_METRICS_PORT", str(_RAY_METRICS_EXPORT_PORT))),
        )


def hostname() -> str:
    """Get the name of the current host system."""
    return socket.gethostname()


def get_ray_command(config: RayConfig, head_node: str | None = None) -> list[str]:
    """Get the command to start Ray.

    Args:
        config (RayConfig): The Ray configuration.
        head_node (str | None): The address of the head node. If None, then
            start the head node, otherwise, start a worker node and connect
            to the specified head node.

    Returns:
        list[str]: The command to start Ray.

    """
    if head_node is None:
        system_config_str = config.system_config.to_json()
        return [
            "ray",
            "start",
            "--head",
            "--node-ip-address",
            hostname(),
            "--port",
            str(config.gcs_server_port),
            "--object-manager-port",
            str(config.object_manager_port),
            "--node-manager-port",
            str(config.node_manager_port),
            "--system-config",
            system_config_str,
            "--runtime-env-agent-port",
            f"{config.runtime_env_agent_port}",
            "--metrics-export-port",
            str(config.metrics_export_port),
            "--dashboard-host",
            str(config.dashboard_host),
            "--dashboard-port",
            str(config.dashboard_port),
            "--dashboard-agent-grpc-port",
            str(config.dashboard_agent_grpc_port),
            "--disable-usage-stats",
        ]

    return [
        "ray",
        "start",
        "--block",
        "--address",
        f"{head_node}:{config.gcs_server_port}",
        "--node-ip-address",
        hostname(),
        "--object-manager-port",
        str(config.object_manager_port),
        "--node-manager-port",
        str(config.node_manager_port),
        "--runtime-env-agent-port",
        str(config.runtime_env_agent_port),
        "--metrics-export-port",
        str(config.metrics_export_port),
        "--dashboard-agent-grpc-port",
        str(config.dashboard_agent_grpc_port),
        "--disable-usage-stats",
    ]


async def stream_output(pipe: asyncio.StreamReader, file: TextIO = sys.stdout, *, flush: bool = False) -> None:
    """Stream output from a subprocess pipe and write it to the specified file.

    Args:
        pipe: The pipe to read from
        file: The file object to write to (default: sys.stdout)
        flush: If True, flush the file object whenever new data appears

    """
    while True:
        line = await pipe.readline()
        if not line:
            break
        print(line.decode("utf-8").strip(), file=file, flush=flush)
    file.flush()


async def run_subprocess_async(command: list[str], *, flush: bool = True) -> None:
    """Run a subprocess asynchronously, logs its output in real-time, and raise an exception if it fails.

    Args:
        command (list[str]): The command to run as a subprocess.
        flush (bool, optional): If True, flush stdout and stderr whenever new data appears

    Raises:
        SubprocessError: If the subprocess exits with a non-zero exit code.
        asyncio.TimeoutError: If the subprocess takes longer than the specified timeout.

    """
    cmd_str = " ".join(command)

    logger.info("Starting `%s`", cmd_str)
    process = await asyncio.create_subprocess_exec(*command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if process.stdout is None:
        error_message = "Unexpected condition: process.stdout is None, this should never happen."
        raise RuntimeError(error_message)

    if process.stderr is None:
        error_message = "Unexpected condition: process.stderr is None, this should never happen."
        raise RuntimeError(error_message)

    # Run stream_output for both stdout and stderr concurrently.
    await asyncio.gather(
        stream_output(process.stdout, sys.stdout, flush=flush),
        stream_output(process.stderr, sys.stderr, flush=flush),
    )

    await asyncio.wait_for(process.wait(), timeout=None)

    if process.returncode != 0:
        returncode = -1 if process.returncode is None else process.returncode
        raise subprocess.CalledProcessError(returncode, cmd_str)


@tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(5))
def get_ray_worker_count() -> int:
    """Return the number of Ray workers currently running in the cluster.

    Returns:
        int: The number of active Ray workers.

    """
    # Ruff complains that subprocess.run is being called without shell=True,
    # but if shell=True is added, Ruff complains that subprocess.run is being
    # called with shell=True.
    command = ["ray", "status"]
    result = subprocess.run(command, check=True, text=True, capture_output=True)  # noqa: S603

    count_nodes = False
    worker_count = 0

    for line in result.stdout.splitlines():
        if "Active:" in line:
            count_nodes = True
        elif "Pending:" in line:
            break
        elif count_nodes and "node_" in line:
            worker_count += 1

    return worker_count


def display_nvidia_smi() -> None:
    """Display the NVIDIA SMI output for the current host directly to stdout.

    Returns:
        None

    """
    logger.info("NVIDIA SMI for %s", hostname())
    asyncio.run(run_subprocess_async(["nvidia-smi"]))


def start_ray(
    config: RayConfig,
    head_node: str | None = None,
    stop_retries_after: int = 600,
    retry_wait_seconds: int = 2,
    retry_attempts: int = 5,
) -> None:
    """Start the Ray head node with retries.

    Args:
        config (RayConfig): The Ray configuration.
        head_node (str | None): The address of the head node. If None, then
            start the head node, otherwise, start a worker node and connect
            to the specified head node.
        stop_retries_after (int): The number of seconds after which to stop retrying.
        retry_wait_seconds (int): The number of seconds to wait between retry attempts.
        retry_attempts (int): The maximum number of retry attempts.

    Returns:
        None

    """
    no_retries_after = time.time() + stop_retries_after

    @tenacity.retry(wait=tenacity.wait_fixed(retry_wait_seconds), stop=tenacity.stop_after_attempt(retry_attempts))
    def _start_ray_with_retry(command: list[str]) -> None:
        try:
            asyncio.run(run_subprocess_async(command))
        except Exception:
            # This suppresses the exception in this case, which is by design.
            # See implementation notes above.
            if no_retries_after >= time.time():
                raise

    command = get_ray_command(config, head_node)
    _start_ray_with_retry(command)

    if head_node is None:
        logger.info("Ray head node started at %s:%s", hostname(), config.gcs_server_port)
    else:
        logger.info("Ray worker node started successfully.")


def wait_for_workers(num_workers: int, max_wait_seconds: int = 600) -> None:
    """Wait for worker nodes to connect.

    Args:
        num_workers (int): The number of workers to wait for
        max_wait_seconds (int): The maximum number of seconds to wait for workers to connect

    Returns:
        None

    """
    waiting = True
    ready_set = False
    end_time = time.time() + max_wait_seconds

    while waiting:
        worker_count = get_ray_worker_count()
        logger.info("Current workers ready: %s", worker_count)
        if worker_count == -1:
            logger.info("Ray cluster status not available. Waiting for cluster.")
        elif worker_count == 1 and not ready_set:
            logger.info("Ray cluster is ready. Setting head node status to ready.")
            ready_set = True
        elif worker_count >= num_workers:
            logger.info("Enough workers connected. Proceeding to start the Cosmos Curator server.")
            waiting = False

        if time.time() >= end_time:
            error_message = f"Waited maximum time of {max_wait_seconds} seconds for {num_workers} workers to connect"
            raise TimeoutError(error_message)

        if waiting:
            logger.info("Waiting for workers to connect...")
            time.sleep(10)


def main() -> None:
    """Run the main entry point for the Ray on SLURM application.

    Returns:
        None

    """
    host = hostname()
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s - {host} - %(levelname)s - %(message)s")
    slurm_env = SlurmEnv.from_env()
    ray_config = RayConfig.from_env()
    display_nvidia_smi()

    logger.info("slurm_env: %s", slurm_env)

    if slurm_env.is_head_node():
        if slurm_env.procid == 0:
            start_ray(ray_config, stop_retries_after=slurm_env.stop_retries_after)
            wait_for_workers(slurm_env.num_nodes)

            # Run the user supplied command with pixi
            cmd = sys.argv[1:]
            logger.info("Running: %s", " ".join(cmd))
            asyncio.run(run_subprocess_async(cmd))

            # It's not clear why this is needed. If we get to this point, ray is stopping on its own.
            logger.info("Stopping Ray")
            asyncio.run(run_subprocess_async(["ray", "stop"]))
            logger.info("Sleeping for 30 seconds")
            time.sleep(30)
        else:
            # This is a sanity check to ensure that only one task is scheduled on each node.
            # It's possible to run multiple tasks per node with slurm. This is not expected
            # to work correctly with Ray.
            msg = "Multiple slurm tasks per node detected. This is not supported."
            raise RuntimeError(msg)
    else:
        start_ray(ray_config, slurm_env.head_node, stop_retries_after=slurm_env.stop_retries_after)

    logger.info("Done")


if __name__ == "__main__":
    main()
