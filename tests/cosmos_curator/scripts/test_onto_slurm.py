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
"""Test the start_ray module."""

import builtins
import os
import socket
import subprocess
import sys
from contextlib import AbstractContextManager, nullcontext
from io import StringIO
from pathlib import Path
from typing import Any, TextIO, TypedDict
from unittest.mock import AsyncMock, MagicMock, patch

import attrs
import pytest
import tenacity

from cosmos_curator.scripts.onto_slurm import (
    _RAY_DASHBOARD_AGENT_GRPC_PORT,
    _RAY_DASHBOARD_HOST,
    _RAY_DASHBOARD_PORT,
    _RAY_GCS_SERVER_PORT,
    _RAY_METRICS_EXPORT_PORT,
    _RAY_NODE_MANAGER_PORT,
    _RAY_OBJECT_MANAGER_PORT,
    _RAY_RUNTIME_ENV_AGENT_PORT,
    RayConfig,
    RayObjectSpillingConfig,
    RayObjectSpillingParams,
    RaySystemConfig,
    SlurmEnv,
    display_nvidia_smi,
    get_ray_command,
    get_ray_worker_count,
    hostname,
    main,
    run_subprocess_async,
    start_ray,
    stream_output,
    wait_for_workers,
)


class TestHostname:
    """Test the hostname function."""

    def test_hostname(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test the hostname function.

        Args:
            monkeypatch: The monkeypatch object.

        """
        monkeypatch.setattr(socket, "gethostname", lambda: "test-hostname")
        assert hostname() == "test-hostname"


class TestSlurmEnv:
    """SlurmEnv tests."""

    def test_from_env_with_required_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test from_env with required vars.

        Args:
            monkeypatch: The monkeypatch object.

        """
        SLURM_NNODES = 4
        RAY_STOP_RETRIES_AFTER = 2
        SLURM_PROCID = 1
        # Set environment variables
        monkeypatch.setenv("SLURM_NNODES", str(SLURM_NNODES))
        monkeypatch.setenv("HEAD_NODE_ADDR", "head-node")
        monkeypatch.setenv("SLURMD_NODENAME", "worker1")
        monkeypatch.setenv("RAY_STOP_RETRIES_AFTER", str(RAY_STOP_RETRIES_AFTER))
        monkeypatch.setenv("SLURM_PROCID", str(SLURM_PROCID))
        slurm_env = SlurmEnv.from_env()

        assert slurm_env.num_nodes == SLURM_NNODES
        assert slurm_env.head_node == "head-node"
        assert slurm_env.nodename == "worker1"
        assert slurm_env.stop_retries_after == RAY_STOP_RETRIES_AFTER
        assert slurm_env.procid == SLURM_PROCID
        assert slurm_env.is_head_node() is False

    def test_from_env_missing_required_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test from_env with missing required vars.

        Args:
            monkeypatch: The monkeypatch object.

        """
        for var in ["SLURM_NNODES", "HEAD_NODE_ADDR", "SLURMD_NODENAME", "RAY_STOP_RETRIES_AFTER", "SLURM_PROCID"]:
            if var in os.environ:
                monkeypatch.delenv(var)

        with pytest.raises(ValueError, match=r"Error: environment variable .* is not set"):
            SlurmEnv.from_env()


class TestRayConfig:
    """RayConfig class tests."""

    def test_config_classes_serialization(self) -> None:
        """Test config classes serialization."""
        # Test constants
        TEST_SPILL_BUFFER_SIZE = 2000000

        params = RayObjectSpillingParams(directory_path=Path("/test/path"), buffer_size=TEST_SPILL_BUFFER_SIZE)
        assert params.directory_path == Path("/test/path")
        assert params.buffer_size == TEST_SPILL_BUFFER_SIZE

        config = RayObjectSpillingConfig(type="filesystem", params=params)
        config_json = config.to_json()
        assert '"type": "filesystem"' in config_json
        assert '"/test/path"' in config_json

        sys_config = RaySystemConfig(local_fs_capacity_threshold=0.85, object_spilling_config=config)
        sys_config_json = sys_config.to_json()
        assert '"local_fs_capacity_threshold": 0.85' in sys_config_json
        assert '"object_spilling_config"' in sys_config_json

    def test_from_env_with_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test from_env with default values.

        Args:
            monkeypatch: The monkeypatch object.

        """
        # Clear any existing environment variables
        for var in [
            "RAY_GCS_SERVER_PORT",
            "RAY_DASHBOARD_HOST",
            "RAY_DASHBOARD_PORT",
            "RAY_OBJECT_MANAGER_PORT",
            "RAY_NODE_MANAGER_PORT",
            "RAY_DASHBOARD_AGENT_GRPC_PORT",
            "RAY_RUNTIME_ENV_AGENT_PORT",
            "XENNA_RAY_METRICS_PORT",
        ]:
            if var in os.environ:
                monkeypatch.delenv(var)

        config = RayConfig.from_env()

        assert config.gcs_server_port == _RAY_GCS_SERVER_PORT
        assert config.dashboard_host == _RAY_DASHBOARD_HOST
        assert config.dashboard_port == _RAY_DASHBOARD_PORT
        assert config.object_manager_port == _RAY_OBJECT_MANAGER_PORT
        assert config.node_manager_port == _RAY_NODE_MANAGER_PORT
        assert config.dashboard_agent_grpc_port == _RAY_DASHBOARD_AGENT_GRPC_PORT
        assert config.runtime_env_agent_port == _RAY_RUNTIME_ENV_AGENT_PORT
        assert config.metrics_export_port == _RAY_METRICS_EXPORT_PORT

    def test_from_env_with_custom_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test from_env with custom values.

        Args:
            monkeypatch: The monkeypatch object.

        """
        monkeypatch.setenv("RAY_GCS_SERVER_PORT", "1234")
        monkeypatch.setenv("RAY_DASHBOARD_HOST", "localhost")
        monkeypatch.setenv("RAY_DASHBOARD_PORT", "5678")
        monkeypatch.setenv("RAY_OBJECT_MANAGER_PORT", "9012")
        monkeypatch.setenv("RAY_NODE_MANAGER_PORT", "3456")
        monkeypatch.setenv("RAY_DASHBOARD_AGENT_GRPC_PORT", "7890")
        monkeypatch.setenv("RAY_RUNTIME_ENV_AGENT_PORT", "1245")
        monkeypatch.setenv("XENNA_RAY_METRICS_PORT", "6789")
        config = RayConfig.from_env()

        assert config.gcs_server_port == 1234
        assert config.dashboard_host == "localhost"
        assert config.dashboard_port == 5678
        assert config.object_manager_port == 9012
        assert config.node_manager_port == 3456
        assert config.dashboard_agent_grpc_port == 7890
        assert config.runtime_env_agent_port == 1245
        assert config.metrics_export_port == 6789


class TestGetRayCommand:
    """Test get_ray_command function."""

    @pytest.mark.parametrize(
        ("head_node", "expected_flags", "skip_values"),
        [
            (
                None,  # Head node test case
                [
                    "--head",
                    "--node-ip-address",
                    "--port",
                    "--object-manager-port",
                    "--node-manager-port",
                    "--runtime-env-agent-port",
                    "--metrics-export-port",
                    "--dashboard-host",
                    "--dashboard-port",
                    "--dashboard-agent-grpc-port",
                    "--disable-usage-stats",
                ],
                [],  # No values to skip for head node
            ),
            (
                "head-node",  # Worker node test case, connect to head-node
                [
                    "--block",
                    "--address",
                    "--node-ip-address",
                    "--object-manager-port",
                    "--node-manager-port",
                    "--runtime-env-agent-port",
                    "--metrics-export-port",
                    "--dashboard-agent-grpc-port",
                    "--disable-usage-stats",
                ],
                ["dashboard_port", "dashboard_host"],  # Skip these values for worker nodes
            ),
        ],
    )
    def test_get_ray_command(
        self, monkeypatch: pytest.MonkeyPatch, head_node: str | None, expected_flags: list[str], skip_values: list[str]
    ) -> None:
        """Test get_ray_command.

        Args:
            monkeypatch: The monkeypatch object.
            head_node: The head node.
            expected_flags: The expected flags.
            skip_values: The values to skip.

        """
        host_name = "test-host"
        monkeypatch.setattr("cosmos_curator.scripts.onto_slurm.hostname", lambda: host_name)
        config = RayConfig()
        command = get_ray_command(config, head_node=head_node)

        # Basic command verification
        assert command[0] == "ray"
        assert command[1] == "start"

        # Verify expected flags are present
        for flag in expected_flags:
            assert flag in command, f"Flag {flag} not found in command"

        # Verify all config values appear in the command
        expected_values = [
            (k, v) for k, v in attrs.asdict(config).items() if k != "system_config" and k not in skip_values
        ]

        for key, value in expected_values:
            if isinstance(value, (int, float)):
                str_value = str(value)
                expected_value_found = False
                for cmd_part in command:
                    if str_value in cmd_part:
                        expected_value_found = True
                        break

                assert expected_value_found, f"Value {str_value} from {key} not found in command"

        # Additional worker-specific checks
        if head_node:
            assert f"{head_node}:{config.gcs_server_port}" in command
            assert host_name in command


class StreamOutputTestCase(TypedDict):
    """Defines test case structure for stream output tests with type checking support."""

    name: str
    input_lines: list[bytes]
    flush: bool
    expected_args: list[str]
    expected_flush: bool


class TestStreamOutput:
    """Test stream_output function."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "test_case",
        [
            {
                "name": "standard_with_flush",
                "input_lines": [b"line 1\n", b"line 2\n", b""],
                "flush": True,
                "expected_args": ["line 1", "line 2"],
                "expected_flush": True,
            },
            {
                "name": "standard_without_flush",
                "input_lines": [b"line 3\n", b"line 4\n", b""],
                "flush": False,
                "expected_args": ["line 3", "line 4"],
                "expected_flush": False,
            },
            {
                "name": "strip_newlines",
                "input_lines": [b"line with trailing newline\n", b"line without newline", b""],
                "flush": False,
                "expected_args": ["line with trailing newline", "line without newline"],
                "expected_flush": False,
            },
            {
                "name": "empty_input",
                "input_lines": [b""],
                "flush": False,
                "expected_args": [],
                "expected_flush": False,
            },
        ],
    )
    async def test_stream_output(self, test_case: StreamOutputTestCase) -> None:
        """Test stream_output function.

        Args:
            test_case: The test case.

        """
        mock_stream = AsyncMock()
        mock_stream.readline = AsyncMock(side_effect=test_case["input_lines"])
        output_file = StringIO()

        # Store the original print function
        original_print = builtins.print

        # Define a side_effect function that writes to the output file using the original print
        def _custom_print(message: str, file: TextIO | None = None, *, flush: bool = False) -> None:
            if file is output_file:
                original_print(message, file=file, flush=flush)

        with patch("builtins.print", side_effect=_custom_print) as mock_print:
            await stream_output(mock_stream, output_file, flush=test_case["flush"])
            assert mock_print.call_count == len(test_case["expected_args"])

            # Verify that the input lines are passed to print in the correct order
            actual_args = [call.args[0] for call in mock_print.call_args_list]
            for expected, actual in zip(test_case["expected_args"], actual_args, strict=True):
                assert actual == expected

            # Verify that the flush argument is passed correctly
            for call in mock_print.call_args_list:
                assert call.kwargs["flush"] is test_case["expected_flush"]

        # Verify the content of the output file matches expectations
        if test_case["name"] == "empty_input":
            assert output_file.getvalue() == "", "Expected empty output file but found content"
        else:
            # The print function adds newlines, so we need to account for that in our expectation
            expected_file_content = "".join(f"{arg}\n" for arg in test_case["expected_args"])
            actual_content = output_file.getvalue()
            assert actual_content == expected_file_content, (
                f"Expected file content '{expected_file_content}' but got '{actual_content}'"
            )


class TestRunSubprocessAsync:
    """Test run_subprocess_async function."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("flush", [False, True])
    async def test_run_subprocess_async(self, *, flush: bool) -> None:
        """Test run_subprocess with different flush parameter values."""
        # Setup mocks
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.wait = AsyncMock(return_value=0)

        # Setup the mock for create_subprocess_exec
        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_create,
            patch("cosmos_curator.scripts.onto_slurm.stream_output") as mock_stream,
            patch("cosmos_curator.scripts.onto_slurm.logger") as mock_logger,
        ):
            # Call the function
            command = ["test", "command"]
            await run_subprocess_async(command, flush=flush)

            # Verify mocks were called correctly
            mock_create.assert_called_once_with(*command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Verify stream_output was called twice (once for stdout, once for stderr)
            EXPECTED_CALL_COUNT = 2
            assert mock_stream.call_count == EXPECTED_CALL_COUNT
            mock_stream.assert_any_call(mock_process.stdout, sys.stdout, flush=flush)
            mock_stream.assert_any_call(mock_process.stderr, sys.stderr, flush=flush)

            # Verify wait was called
            mock_process.wait.assert_called_once()

            # Verify logging occurred
            # Check that logger.info was called at least once
            assert mock_logger.info.call_count >= 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("stdout", "stderr", "error_message"),
        [
            (None, AsyncMock(), "Unexpected condition: process.stdout is None"),
            (AsyncMock(), None, "Unexpected condition: process.stderr is None"),
        ],
    )
    async def test_run_subprocess_async_stdout_stderr_none(
        self, stdout: AsyncMock | None, stderr: AsyncMock | None, error_message: str
    ) -> None:
        """Test run_subprocess handling of None for stdout/stderr."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.stdout = stdout
        mock_process.stderr = stderr
        mock_process.wait = AsyncMock(return_value=0)

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_process),
            pytest.raises(RuntimeError, match=error_message),
        ):
            await run_subprocess_async(["test", "command"])

    @pytest.mark.asyncio
    async def test_run_subprocess_async_nonzero_exit(self) -> None:
        """Test run_subprocess with non-zero exit code."""
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.wait = AsyncMock(return_value=1)

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_process),
            patch("cosmos_curator.scripts.onto_slurm.stream_output"),
        ):
            command = ["test", "command"]

            with pytest.raises(subprocess.CalledProcessError) as exc_info:
                await run_subprocess_async(command)

            assert exc_info.value.returncode == 1
            assert exc_info.value.cmd == "test command"


class TestGetRayWorkerCount:
    """Test get_ray_worker_count function."""

    @pytest.mark.parametrize("num_workers", [0, 1, 2, 3])
    def test_get_ray_worker_count(self, num_workers: int, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_ray_worker_count function.

        Args:
            num_workers: The number of workers.
            monkeypatch: The monkeypatch object.

        """
        mock_output = """
Status
======
Ray runtime started.
Local node IP: 192.168.1.1

Active:
"""

        for i in range(num_workers):
            mock_output += f" * node_{i} (192.168.1.{i})\n"

        mock_output += """
Pending:
 (none)
"""
        mock_result = subprocess.CompletedProcess(args=["ray", "status"], returncode=0, stdout=mock_output, stderr="")
        monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: mock_result)
        assert get_ray_worker_count() == num_workers


class TestDisplayNvidiaSmi:
    """Test display_nvidia_smi function."""

    def test_display_nvidia_smi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that display_nvidia_smi logs correctly and calls run_subprocess with the right command.

        Args:
            monkeypatch: The monkeypatch object.

        """
        # Mock hostname
        test_hostname = "test-gpu-node"
        monkeypatch.setattr("cosmos_curator.scripts.onto_slurm.hostname", lambda: test_hostname)

        # Mock asyncio.run to avoid actually running the subprocess
        mock_asyncio_run = MagicMock()
        mock_asyncio_run.side_effect = lambda _: None  # Just run the coroutine's side effects
        monkeypatch.setattr("asyncio.run", mock_asyncio_run)

        # Mock run_subprocess to verify it's called with the right command
        mock_run_subprocess_async = MagicMock()
        monkeypatch.setattr("cosmos_curator.scripts.onto_slurm.run_subprocess_async", mock_run_subprocess_async)

        # Mock logging.info to verify it's called with the expected message
        with patch("cosmos_curator.scripts.onto_slurm.logger") as mock_logger:
            display_nvidia_smi()
            mock_logger.info.assert_called_once()
            mock_asyncio_run.assert_called_once()
            mock_run_subprocess_async.assert_called_once_with(["nvidia-smi"])


class TestStartRay:
    """Test start_ray function."""

    @pytest.mark.parametrize(
        ("head_node", "num_simulated_failures", "always_fail"),
        [
            (None, 0, False),
            ("head-node.example", 0, False),
            (None, 2, False),
            ("head-node.example", 2, False),
            (None, 0, True),
            ("head-node.example", 0, True),
        ],
    )
    def test_start_ray(
        self,
        monkeypatch: pytest.MonkeyPatch,
        head_node: str | None,
        num_simulated_failures: int,
        *,
        always_fail: bool,
    ) -> None:
        """Test that start_ray correctly calls run_subprocess.

        Args:
            monkeypatch: The monkeypatch object.
            head_node: The head node.
            num_simulated_failures: The number of simulated failures.
            always_fail: Whether to always fail.

        """
        # Mock hostname() so that it isn't called in start_ray
        test_hostname = "test-ray-node"
        monkeypatch.setattr("cosmos_curator.scripts.onto_slurm.hostname", lambda: test_hostname)

        mock_command = ["ray", "start", "--test-flag"]
        monkeypatch.setattr(
            "cosmos_curator.scripts.onto_slurm.get_ray_command", lambda _config, _head_node: mock_command
        )

        mock_run_subprocess_async = MagicMock()
        monkeypatch.setattr("cosmos_curator.scripts.onto_slurm.run_subprocess_async", mock_run_subprocess_async)

        # Mock asyncio.run to fail N times and then succeed
        call_count = 0

        def mock_run_and_fail_n_times(*_args: object) -> None:
            nonlocal call_count
            call_count += 1
            if call_count <= num_simulated_failures:  # Fail the first N-1 times
                raise subprocess.CalledProcessError(1, "ray start")
            # Success on the Nth attempt

        def mock_run_and_always_fail(*_args: object) -> None:
            nonlocal call_count
            call_count += 1
            raise subprocess.CalledProcessError(1, "ray start")

        mock_asyncio_run = MagicMock(side_effect=mock_run_and_fail_n_times)
        if always_fail:
            mock_asyncio_run = MagicMock(side_effect=mock_run_and_always_fail)

        monkeypatch.setattr("asyncio.run", mock_asyncio_run)

        ray_config = RayConfig()

        raises = pytest.raises(tenacity.RetryError) if always_fail else nullcontext()

        with patch("cosmos_curator.scripts.onto_slurm.logger") as mock_logger:
            # Call with short retry parameters for fast testing
            retry_attempts = 5
            with raises:
                start_ray(
                    ray_config, head_node, stop_retries_after=2, retry_wait_seconds=0, retry_attempts=retry_attempts
                )

            expected_call_count = retry_attempts if always_fail else num_simulated_failures + 1
            assert call_count == expected_call_count
            assert mock_run_subprocess_async.call_count == call_count

            # Verify expected logging
            if not always_fail:
                assert len(mock_logger.info.call_args_list) >= 1


class TestWaitForWorkers:
    """Test wait_for_workers function."""

    @pytest.mark.parametrize(
        ("worker_count_values"),
        [
            # Normal case: Start with 1 worker, then get 3 workers after one check
            ([1, 3]),
            # Case: Already have enough workers at start (immediate return)
            ([3]),
            # Case: Cluster not available initially, then has 1 worker, then 3
            ([-1, 1, 3]),
            # Case: Multiple status checks before reaching target
            ([1, 1, 2, 3]),
        ],
    )
    def test_wait_for_workers(self, monkeypatch: pytest.MonkeyPatch, worker_count_values: list[int]) -> None:
        """Test wait_for_workers with different worker count scenarios.

        Args:
            monkeypatch: The monkeypatch object.
            worker_count_values: The worker count values.

        """
        # Mock get_ray_worker_count to return the sequence of values we want to test
        mock_get_worker_count = MagicMock(side_effect=worker_count_values)
        monkeypatch.setattr("cosmos_curator.scripts.onto_slurm.get_ray_worker_count", mock_get_worker_count)

        # Mock sleep to avoid actually sleeping
        mock_sleep = MagicMock()
        monkeypatch.setattr("time.sleep", mock_sleep)

        num_workers = worker_count_values[-1]
        expected_sleep_calls = len(worker_count_values) - 1 if worker_count_values else 0

        with patch("cosmos_curator.scripts.onto_slurm.logger") as mock_logger:
            wait_for_workers(num_workers)
            # Verify get_ray_worker_count was called the expected number of times
            assert mock_get_worker_count.call_count == len(worker_count_values)

            # Verify that sleep was called the expected number of times
            assert mock_sleep.call_count == expected_sleep_calls

            # Verify appropriate log messages based on worker counts
            log_messages = [args[0][0] for args in mock_logger.info.call_args_list]
            assert any("Current workers ready" in msg for msg in log_messages)
            if -1 in worker_count_values:
                assert any("Ray cluster status not available" in msg for msg in log_messages)
            if 1 in worker_count_values:
                assert any("Ray cluster is ready" in msg for msg in log_messages)
            assert any("Enough workers connected" in msg for msg in log_messages)

    def test_wait_for_workers_never_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test wait_for_workers stopping after max iterations for safety.

        Args:
            monkeypatch: The monkeypatch object.

        """
        # Always return 1 worker (never enough)
        monkeypatch.setattr("cosmos_curator.scripts.onto_slurm.get_ray_worker_count", lambda: -1)

        # Mock sleep to avoid actually sleeping
        mock_sleep = MagicMock()
        monkeypatch.setattr("time.sleep", mock_sleep)

        with pytest.raises(TimeoutError):
            wait_for_workers(5, max_wait_seconds=0)


class TestMain:
    """Test main function."""

    @pytest.mark.parametrize(
        ("is_head_node", "slurm_procid", "raises"),
        [
            (True, 0, nullcontext()),  # Test head node path
            (False, 1, nullcontext()),  # Test worker node path
            (True, 1, pytest.raises(RuntimeError)),  # Head node, multiple tasks, should raise
        ],
    )
    def test_main(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        is_head_node: bool,
        slurm_procid: int,
        raises: AbstractContextManager[Any],
    ) -> None:
        """Test the main function happy path for head node and worker node scenarios.

        Args:
            monkeypatch: The monkeypatch object.
            is_head_node: Whether the node is a head node.
            slurm_procid: The SLURM procid.
            raises: The expected exception to raise.

        """
        test_hostname = "test-hostname"
        test_num_nodes = 3

        mock_slurm_env = MagicMock()
        mock_slurm_env.num_nodes = test_num_nodes
        mock_slurm_env.head_node = test_hostname if is_head_node else "head-node"
        mock_slurm_env.nodename = test_hostname
        mock_slurm_env.stop_retries_after = 2
        mock_slurm_env.is_head_node.return_value = is_head_node
        mock_slurm_env.procid = slurm_procid

        mock_ray_config = MagicMock()
        mock_ray_config.gcs_server_port = _RAY_GCS_SERVER_PORT

        mock_run_subprocess_async = MagicMock()
        mock_display_nvidia_smi = MagicMock()
        mock_start_ray = MagicMock()
        mock_wait_for_workers = MagicMock()
        mock_asyncio_run = MagicMock()
        mock_sleep = MagicMock()

        monkeypatch.setattr("cosmos_curator.scripts.onto_slurm.hostname", lambda: test_hostname)
        monkeypatch.setattr("cosmos_curator.scripts.onto_slurm.SlurmEnv.from_env", lambda: mock_slurm_env)
        monkeypatch.setattr("cosmos_curator.scripts.onto_slurm.RayConfig.from_env", lambda: mock_ray_config)
        monkeypatch.setattr("cosmos_curator.scripts.onto_slurm.run_subprocess_async", mock_run_subprocess_async)
        monkeypatch.setattr("cosmos_curator.scripts.onto_slurm.display_nvidia_smi", mock_display_nvidia_smi)
        monkeypatch.setattr("cosmos_curator.scripts.onto_slurm.start_ray", mock_start_ray)
        monkeypatch.setattr("cosmos_curator.scripts.onto_slurm.wait_for_workers", mock_wait_for_workers)
        monkeypatch.setattr("asyncio.run", mock_asyncio_run)
        monkeypatch.setattr("time.sleep", mock_sleep)

        with patch("logging.basicConfig"), patch("cosmos_curator.scripts.onto_slurm.logger"), raises:
            main()
            mock_display_nvidia_smi.assert_called_once()
            mock_start_ray.assert_called_once()

            if is_head_node:
                mock_wait_for_workers.assert_called_once_with(test_num_nodes)
                MIN_EXPECTED_CALL_COUNT = 2
                assert mock_asyncio_run.call_count >= MIN_EXPECTED_CALL_COUNT  # conda config, command, and ray stop
                mock_sleep.assert_called_once_with(30)
                mock_start_ray.assert_called_once_with(
                    mock_ray_config, stop_retries_after=mock_slurm_env.stop_retries_after
                )
            else:
                mock_wait_for_workers.assert_not_called()
                mock_start_ray.assert_called_once_with(
                    mock_ray_config, mock_slurm_env.head_node, stop_retries_after=mock_slurm_env.stop_retries_after
                )
                mock_sleep.assert_not_called()
