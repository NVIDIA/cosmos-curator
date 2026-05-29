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

"""Ray Cluster and NVCF Server Management Script.

This script manages the lifecycle of a Ray cluster and NVCF server in a Kubernetes/NVCF environment.
It handles both head node and worker node initialization, process monitoring, and health checks.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger: logging.Logger = logging.getLogger(__name__)

# Extract Ray ports from environment variables or use defaults
GCS_SERVER_PORT: str = os.environ.get("RAY_GCS_SERVER_PORT", "6379")
DASHBOARD_PORT: str = os.environ.get("RAY_DASHBOARD_PORT", "8265")
OBJECT_MANAGER_PORT: str = os.environ.get("RAY_OBJECT_MANAGER_PORT", "8076")
NODE_MANAGER_PORT: str = os.environ.get("RAY_NODE_MANAGER_PORT", "8077")
DASHBOARD_AGENT_PORT: str = os.environ.get("RAY_DASHBOARD_AGENT_PORT", "52365")
METRICS_PORT: str = os.environ.get("XENNA_RAY_METRICS_PORT", "9002")
POD_NAME: str = os.environ.get("POD_NAME", "nvcf-container-0")
HEADLESS_SERVICE_NAME: str = os.environ.get("HEADLESS_SERVICE_NAME", "nvcf-container-0.svc")
NODES_PER_INSTANCE: int = int(os.environ.get("NODES_PER_INSTANCE", "1"))

# Set environment variables
os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

# Required environment variables
REQUIRED_VARS: set[str] = {"NODES_PER_INSTANCE", "POD_NAME", "HEADLESS_SERVICE_NAME"}


def check_required_vars() -> None:
    """Verify that all required environment variables are set.

    Raises:
        SystemExit: If any required variable is not set.

    """
    missing_vars: list[str] = [var for var in REQUIRED_VARS if not os.environ.get(var)]
    if missing_vars:
        logger.error("Error: Missing required environment variables: %s", ", ".join(missing_vars))
        sys.exit(1)


def get_ray_worker_count() -> int:
    """Count the number of active Ray worker nodes.

    Returns:
        int: Number of active workers. Returns -1 if the status cannot be determined.

    """
    if not check_command_exists("ray"):
        logger.error("Ray command not found in system PATH")
        return -1

    try:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            ["ray", "status"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        output: str = result.stdout

        if "Active:" in output:
            active_section: str = output.split("Active:")[1].split("Pending:", maxsplit=1)[0]
            worker_count: int = sum(1 for line in active_section.splitlines() if "node_" in line)
            return max(0, worker_count)
    except subprocess.TimeoutExpired:
        logger.exception("Ray status command timed out")
        return -1
    except Exception:
        logger.exception("Error getting Ray worker count: ")
        return -1
    else:
        return -1


def check_command_exists(command: str) -> bool:
    """Check if a command exists in the system PATH.

    Args:
        command (str): The command to check.

    Returns:
        bool: True if the command exists, False otherwise.

    """
    return shutil.which(command) is not None


def display_nvidia_smi() -> None:
    """Display NVIDIA SMI information for the current pod."""
    logger.info("NVIDIA SMI for Pod: %s", POD_NAME)
    if check_command_exists("nvidia-smi"):
        try:
            result: subprocess.CompletedProcess[str] = subprocess.run(
                ["nvidia-smi"],  # noqa: S607
                capture_output=True,
                text=True,
                check=False,
            )
            logger.info(result.stdout)
        except Exception:
            logger.exception("Error running nvidia-smi: ")
    else:
        logger.warning("nvidia-smi not in container")


def start_nvcf_server() -> subprocess.Popen[str] | None:
    """Start the NVCF server using uvicorn.

    Returns:
        Optional[subprocess.Popen[str]]: Process object for the started NVCF server,
        or None if the server couldn't be started.

    """
    workdir: str = "/opt/cosmos-curator"
    uvicorn_path: str = os.environ.get("UVICORN_PATH", "/opt/cosmos-curator/.pixi/envs/default/bin/uvicorn")
    app_command: str = os.environ.get("APP_COMMAND", "cosmos_curator.core.cf.nvcf_main:app")

    if not Path(uvicorn_path).exists():
        logger.error("Uvicorn not found at %s", uvicorn_path)
        return None

    logger.info("Starting NVCF app endpoint")
    try:
        return subprocess.Popen(  # noqa: S603
            [
                uvicorn_path,
                app_command,
                "--host",
                "0.0.0.0",  # noqa: S104
                "--port",
                "8000",
                "--workers",
                "2",
                "--limit-concurrency",
                "2",
            ],
            text=True,
            cwd=workdir,
        )
    except Exception:
        logger.exception("Failed to start NVCF server: ")
        return None


def start_ray_head() -> subprocess.Popen[str] | None:
    """Start the Ray head node with configured ports and settings.

    Returns:
        Optional[subprocess.Popen[str]]: Process object for the started Ray head node,
        or None if the node couldn't be started.

    """
    if not check_command_exists("ray"):
        logger.error("Ray command not found in system PATH")
        return None

    logger.info("Starting Ray head node")
    try:
        hostname_ip: str = subprocess.check_output(
            ["hostname", "-i"],  # noqa: S607
            text=True,
            timeout=5,
        ).strip()

        sys_cfg = {
            "local_fs_capacity_threshold": 0.90,
            "object_spilling_config": json.dumps(
                {
                    "type": "filesystem",
                    "params": {"directory_path": "/config/tmp/ray_spill", "buffer_size": 1000000},
                }
            ),
        }
        ray_cmd: list[str] = [
            "ray",
            "start",
            "--head",
            "--node-ip-address",
            hostname_ip,
            "--port",
            GCS_SERVER_PORT,
            "--object-manager-port",
            OBJECT_MANAGER_PORT,
            "--node-manager-port",
            NODE_MANAGER_PORT,
            "--system-config",
            json.dumps(sys_cfg),
            "--metrics-export-port",
            METRICS_PORT,
            "--dashboard-agent-listen-port",
            DASHBOARD_AGENT_PORT,
        ]
        return subprocess.Popen(  # noqa: S603
            ray_cmd,
            text=True,
        )
    except subprocess.TimeoutExpired:
        logger.exception("Timeout while getting hostname IP")
        return None
    except Exception:
        logger.exception("Failed to start Ray head node: ")
        return None


def start_ray_worker(head_pod: str) -> bool:
    """Start a Ray worker node and connect it to the head node.

    Args:
        head_pod (str): The address of the head pod.

    Returns:
        bool: True if the worker node started successfully, False otherwise.

    """
    if not check_command_exists("ray"):
        logger.error("Ray command not found in system PATH")
        return False

    logger.info("Starting Ray worker node and connecting to head at %s", head_pod)
    try:
        hostname_ip: str = subprocess.check_output(
            ["hostname", "-i"],  # noqa: S607
            text=True,
            timeout=5,
        ).strip()
        _: subprocess.CompletedProcess[str] = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "ray",
                "start",
                "--address",
                f"{head_pod}:{GCS_SERVER_PORT}",
                "--node-ip-address",
                hostname_ip,
                "--object-manager-port",
                OBJECT_MANAGER_PORT,
                "--node-manager-port",
                NODE_MANAGER_PORT,
                "--metrics-export-port",
                METRICS_PORT,
                "--dashboard-agent-listen-port",
                DASHBOARD_AGENT_PORT,
            ],
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        logger.exception("Error starting Ray worker node: ")
        return False
    except Exception:
        logger.exception("Failed to start Ray worker node: ")
        return False
    else:
        logger.info("Ray worker node started successfully")
        return True


def ray_is_alive(host: str = "localhost", port: str = DASHBOARD_AGENT_PORT, timeout: int = 10) -> bool:
    """Check if Ray dashboard is alive by querying the /healthz endpoint.

    Args:
        host (str): Host of the Ray head node.
        port (int): Port where Ray dashboard is running (default is 8265).
        timeout (int): Timeout in seconds for the HTTP request.

    Returns:
        bool: True if Ray is alive, False otherwise.

    """
    url: str = f"http://{host}:{port}/api/local_raylet_healthz"
    http_ok: int = 200
    try:
        response = requests.get(url, timeout=timeout)
    except (requests.ConnectionError, requests.Timeout, ValueError):
        return False
    else:
        return response.status_code == http_ok


def monitor_processes(processes: dict[str, subprocess.Popen[str] | None]) -> None:
    """Monitor the health of running processes and handle failures.

    Args:
        processes (Dict[str, Optional[subprocess.Popen[str]]]): Dictionary of process names to their Popen objects.

    Raises:
        RuntimeError: If any of the process dies or cannot contact ray.

    Note:
        This function runs indefinitely until a process fails or is interrupted.
        It will exit with status code 1 if any process fails unexpectedly.

    """
    logger.info("Entering monitoring loop for %d processes", len(processes.keys()))
    err_msg: str = ""
    timeout: int = 10
    while True:
        for name, process in processes.items():
            if process is None:
                logger.warning("Process %s is still not ready...", name)
                continue
            # when we start ray, the pid captured is the pid of launcher
            # we need to check healthz of ray
            if name == "ray":
                if not ray_is_alive(timeout=timeout):
                    err_msg = f"Ray Cluster did not respond in {timeout} secs."
                    raise RuntimeError(err_msg)
            elif process.poll() is not None:
                err_msg = f"Process {name} with PID {process.pid} has exited unexpectedly."
                raise RuntimeError(err_msg)
        time.sleep(timeout)


def startup(replica: str, head_pod: str) -> dict[str, subprocess.Popen[str] | None]:
    """Startup the Ray cluster and uvicorn server.

    The function:
        1. Starts Ray head and or worker node based on replica number
        2. Starts NVCF server

    Args:
        replica (str): The replica-id "0" or "1" or ...
        head_pod (str): Head pod name


    Returns:
        dict[str, subprocess.Popen[str] | None]

    """
    processes: dict[str, subprocess.Popen[str] | None] = {}

    err_msg: str = ""
    if replica == "0":
        # Start Ray head node
        ray_process: subprocess.Popen[str] | None = start_ray_head()
        if ray_process is None:
            err_msg = "Failed to start Ray head node"
            raise RuntimeError(err_msg)
        processes["ray"] = ray_process
        logger.info("Ray head node started with PID %d", ray_process.pid)

        ready_set: bool = False
        while True:
            worker_count: int = get_ray_worker_count()
            logger.info("Current workers ready: %d / %d", worker_count, NODES_PER_INSTANCE)

            if worker_count == -1:
                logger.warning("Ray cluster status not available. Waiting for cluster.")
                time.sleep(5)
                continue

            if worker_count == 1 and not ready_set:
                logger.info("Ray cluster is ready. Setting head node pod status to ready.")
                Path(f"{tempfile.gettempdir()}/is_ready").touch()
                ready_set = True

            if worker_count >= NODES_PER_INSTANCE:
                logger.info("Enough workers connected. Proceeding to start the Cosmos Curator server.")
                break

            logger.info("Waiting for workers to connect...")
            time.sleep(10)

        # Start NVCF server
        nvcf_process: subprocess.Popen[str] | None = start_nvcf_server()
        if nvcf_process is None:
            err_msg = "Failed to start NVCF server"
            raise RuntimeError(err_msg)
        processes["nvcf"] = nvcf_process

    else:
        # Wait for Ray head node to be ready
        time.sleep(60)

        # Start Ray worker node
        if not start_ray_worker(head_pod):
            err_msg = "Failed to start Ray worker node"
            raise RuntimeError(err_msg)
    return processes


def main() -> None:
    """Orchestrate the Ray cluster and NVCF server setup and monitoring.

    The function:
    1. Validates environment variables if no --helm is False
    2. Calls startup to start ray and uvicorn
    3. Monitors all processes
    4. Handles graceful shutdown on interruption
    """
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process the helm flag.")

    # Define the --helm argument as optional, with default True
    parser.add_argument(
        "--helm",
        type=str,
        nargs="?",  # Argument is optional (can be absent)
        const="True",  # If the argument is passed without a value, treat it as "True"
        default="True",  # Default value is "True"
        choices=["True", "False"],  # Only "True" or "False" are allowed
        help="Enable or disable Helm configuration. If not passed, defaults to True.",
    )

    # Parse the arguments
    args = parser.parse_args()
    is_helm = args.helm == "True"

    if is_helm:
        # Check required environment variables
        check_required_vars()
        # Set environment variables
        os.environ["NVCF_MULTI_NODE"] = "true"
    else:
        os.environ["NVCF_MULTI_NODE"] = "false"
        os.environ["NVCF_SINGLE_NODE"] = os.environ.get("NVCF_SINGLE_NODE", "true")
        os.environ["NVCF_REQUEST_STATUS"] = os.environ.get("NVCF_REQUEST_STATUS", "true")

    # Extract replica information
    replica: str = POD_NAME.rsplit("-", maxsplit=1)[-1]
    replica_group: str = "-".join(POD_NAME.split("-")[:-1])
    head_pod: str = f"{replica_group}-0.{HEADLESS_SERVICE_NAME}"

    # Display NVIDIA SMI information
    display_nvidia_smi()

    # Call the startup to start headnode/worker/nvcf
    try:
        processes = startup(replica, head_pod)
    except Exception:
        logger.exception("Startup Failure: ")
        sys.exit(1)

    try:
        monitor_processes(processes)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        for process in processes.values():
            if process is not None:
                process.terminate()
        sys.exit(0)
    except Exception:
        logger.exception("Abnormal Termination. Shutting down..")
        for process in processes.values():
            if process is not None:
                process.terminate()
        sys.exit(1)


if __name__ == "__main__":
    main()
