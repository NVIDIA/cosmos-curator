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
"""Test nvcf_main functionality."""

import argparse
import io
import itertools
import json
import queue
import tempfile
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cosmos_curator.core.cf import nvcf_main

# HTTP status codes
HTTP_OK = 200
HTTP_PRECONDITION_FAILED = 412
HTTP_UNPROCESSABLE_ENTITY = 422
HTTP_INTERNAL_SERVER_ERROR = 500

# Test constants
EXPECTED_JOBS = 2
EXPECTED_ASSETS = 3
PROGRESS_42_5 = 42.5
PROGRESS_25_5 = 25.5
PROGRESS_50_0 = 50.0
PROGRESS_60_0 = 60.0
PROGRESS_75_0 = 75.0
PROGRESS_33_3 = 33.3
RETURN_CODE_1 = 1


@pytest.fixture
def test_client() -> Iterator[TestClient]:
    """Yield a fresh FastAPI TestClient for each test to avoid shared state."""
    app = FastAPI()
    app.include_router(nvcf_main.app.router)
    nvcf_main.setup_pipeline_middleware(app)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def mock_request_id() -> str:
    """Create a mock request ID."""
    return "test-request-id-12345"


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_progress_file(self, mock_request_id: str) -> None:
        """Test _get_progress_file returns correct path."""
        progress_file = nvcf_main._get_progress_file(mock_request_id)
        assert progress_file.name == f"progress_{mock_request_id}.json"
        assert progress_file.parent == Path(tempfile.gettempdir())

    def test_get_log_file(self, mock_request_id: str) -> None:
        """Test _get_log_file returns correct path."""
        log_file = nvcf_main._get_log_file(mock_request_id)
        assert log_file.name == f"logs_{mock_request_id}.txt"
        assert log_file.parent == Path(tempfile.gettempdir())

    def test_get_done_file(self, mock_request_id: str) -> None:
        """Test _get_done_file returns correct path."""
        done_file = nvcf_main._get_done_file(mock_request_id)
        assert done_file.name == f"done_{mock_request_id}.txt"
        assert done_file.parent == Path(tempfile.gettempdir())

    def test_get_failed_file(self, mock_request_id: str) -> None:
        """Test _get_failed_file returns correct path."""
        failed_file = nvcf_main._get_failed_file(mock_request_id)
        assert failed_file.name == f"failed_{mock_request_id}.txt"
        assert failed_file.parent == Path(tempfile.gettempdir())

    def test_value_error(self) -> None:
        """Test _value_error raises ValueError."""
        with pytest.raises(ValueError, match="test error"):
            nvcf_main._value_error("test error")


class TestRequestStatus:
    """Test request status functions."""

    def test_get_request_status_not_found(self, mock_request_id: str, tmp_path: Path) -> None:
        """Test _get_request_status returns not-found when files don't exist."""
        with patch("cosmos_curator.core.cf.nvcf_main._get_progress_file") as mock_progress:
            mock_progress.return_value = tmp_path / "nonexistent.json"
            with patch("cosmos_curator.core.cf.nvcf_main._get_done_file") as mock_done:
                mock_done.return_value = tmp_path / "nonexistent_done.txt"
                with patch("cosmos_curator.core.cf.nvcf_main._get_failed_file") as mock_failed:
                    mock_failed.return_value = tmp_path / "nonexistent_failed.txt"
                    status = nvcf_main._get_request_status(mock_request_id)
                    assert status == "not-found"

    def test_get_request_status_running(self, mock_request_id: str, tmp_path: Path) -> None:
        """Test _get_request_status returns running when progress file exists."""
        progress_file = tmp_path / f"progress_{mock_request_id}.json"
        progress_file.touch()

        with patch("cosmos_curator.core.cf.nvcf_main._get_progress_file") as mock_progress:
            mock_progress.return_value = progress_file
            with patch("cosmos_curator.core.cf.nvcf_main._get_done_file") as mock_done:
                mock_done.return_value = tmp_path / "done.txt"
                with patch("cosmos_curator.core.cf.nvcf_main._get_failed_file") as mock_failed:
                    mock_failed.return_value = tmp_path / "failed.txt"
                    status = nvcf_main._get_request_status(mock_request_id)
                    assert status == "running"

    def test_get_request_status_done(self, mock_request_id: str, tmp_path: Path) -> None:
        """Test _get_request_status returns done when done file exists."""
        progress_file = tmp_path / f"progress_{mock_request_id}.json"
        progress_file.touch()
        done_file = tmp_path / f"done_{mock_request_id}.txt"
        done_file.touch()

        with patch("cosmos_curator.core.cf.nvcf_main._get_progress_file") as mock_progress:
            mock_progress.return_value = progress_file
            with patch("cosmos_curator.core.cf.nvcf_main._get_done_file") as mock_done:
                mock_done.return_value = done_file
                with patch("cosmos_curator.core.cf.nvcf_main._get_failed_file") as mock_failed:
                    mock_failed.return_value = tmp_path / "failed.txt"
                    status = nvcf_main._get_request_status(mock_request_id)
                    assert status == "done"

    def test_get_request_status_failed(self, mock_request_id: str, tmp_path: Path) -> None:
        """Test _get_request_status returns failed when failed file exists."""
        progress_file = tmp_path / f"progress_{mock_request_id}.json"
        progress_file.touch()
        failed_file = tmp_path / f"failed_{mock_request_id}.txt"
        failed_file.touch()

        with patch("cosmos_curator.core.cf.nvcf_main._get_progress_file") as mock_progress:
            mock_progress.return_value = progress_file
            with patch("cosmos_curator.core.cf.nvcf_main._get_done_file") as mock_done:
                mock_done.return_value = tmp_path / "done.txt"
                with patch("cosmos_curator.core.cf.nvcf_main._get_failed_file") as mock_failed:
                    mock_failed.return_value = failed_file
                    status = nvcf_main._get_request_status(mock_request_id)
                    assert status == "failed"


class TestFileOperations:
    """Test file read/write operations."""

    def test_read_progress_and_log_files(self, mock_request_id: str, tmp_path: Path) -> None:
        """Test _read_progress_and_log_files reads files correctly."""
        progress_file = tmp_path / f"progress_{mock_request_id}.json"
        progress_file.write_text(json.dumps({"progress": PROGRESS_42_5}))

        log_file = tmp_path / f"logs_{mock_request_id}.txt"
        log_file.write_text("test log line 1\ntest log line 2\n")

        with patch("cosmos_curator.core.cf.nvcf_main._get_progress_file") as mock_progress:
            mock_progress.return_value = progress_file
            with patch("cosmos_curator.core.cf.nvcf_main._get_log_file") as mock_log:
                mock_log.return_value = log_file
                with patch("cosmos_curator.core.cf.nvcf_main._list_all_jobs") as mock_jobs:
                    mock_jobs.return_value = {}
                    progress_pct, log_lines = nvcf_main._read_progress_and_log_files(mock_request_id)

                    assert progress_pct == pytest.approx(PROGRESS_42_5)
                    assert "test log line 1" in log_lines
                    assert "test log line 2" in log_lines

    def test_read_progress_and_log_files_no_request_id(self) -> None:
        """Test _read_progress_and_log_files with no request_id."""
        with patch("cosmos_curator.core.cf.nvcf_main._list_all_jobs") as mock_jobs:
            mock_jobs.return_value = {}
            progress_pct, log_lines = nvcf_main._read_progress_and_log_files(None)

            assert progress_pct == 0.0
            assert "_read_progress_and_log_files called without req_id" in log_lines

    def test_write_progress_and_log_files(self, mock_request_id: str, tmp_path: Path) -> None:
        """Test _write_progress_and_log_files writes files correctly."""
        progress_file = tmp_path / f"progress_{mock_request_id}.json"
        log_file = tmp_path / f"logs_{mock_request_id}.txt"

        with patch("cosmos_curator.core.cf.nvcf_main._get_progress_file") as mock_progress:
            mock_progress.return_value = progress_file
            with patch("cosmos_curator.core.cf.nvcf_main._get_log_file") as mock_log:
                mock_log.return_value = log_file
                with patch("cosmos_curator.core.cf.nvcf_main.get_pipeline_progress") as mock_pct:
                    mock_pct.return_value = PROGRESS_75_0

                    buffer = ["log line 1\n", "log line 2\n"]
                    nvcf_main._write_progress_and_log_files(
                        req_id=mock_request_id,
                        buffer=buffer,
                        write_progress=True,
                        write_log=True,
                    )

                    # Verify progress file
                    progress_data = json.loads(progress_file.read_text())
                    assert progress_data["progress"] == pytest.approx(PROGRESS_75_0)

                    # Verify log file
                    log_content = log_file.read_text()
                    assert "log line 1" in log_content
                    assert "log line 2" in log_content


class TestRayJobOperations:
    """Test Ray job operations."""

    def test_list_all_jobs_success(self) -> None:
        """Test _list_all_jobs returns jobs correctly."""
        mock_response = MagicMock()
        mock_response.status_code = HTTP_OK
        mock_response.json.return_value = [
            {"type": "SUBMISSION", "submission_id": "job-1", "status": "RUNNING", "message": "Running"},
            {"type": "DRIVER", "job_id": "job-2", "status": "SUCCEEDED", "message": "Done"},
        ]

        with patch("requests.get") as mock_get:
            mock_get.return_value = mock_response
            jobs = nvcf_main._list_all_jobs()

            assert len(jobs) == EXPECTED_JOBS
            assert jobs["job-1"] == ("SUBMISSION", "RUNNING", "Running")
            assert jobs["job-2"] == ("DRIVER", "SUCCEEDED", "Done")

    def test_list_all_jobs_failure(self) -> None:
        """Test _list_all_jobs raises ValueError on failure."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection error")

            with pytest.raises(ValueError, match="Failed to get job status"):
                nvcf_main._list_all_jobs()

    def test_wait_for_stop_success(self) -> None:
        """Test _wait_for_stop returns True when job stops."""
        running_resp = MagicMock()
        running_resp.status_code = HTTP_OK
        running_resp.json.return_value = {"status": "RUNNING"}
        running_resp.text = '{"status": "RUNNING"}'

        stopped_resp = MagicMock()
        stopped_resp.status_code = HTTP_OK
        stopped_resp.json.return_value = {"status": "STOPPED"}
        stopped_resp.text = '{"status": "STOPPED"}'

        with (
            patch("requests.get", side_effect=[running_resp, stopped_resp]),
            patch("time.sleep"),
        ):
            result = nvcf_main._wait_for_stop("job-123")

        assert result is True

    def test_wait_for_stop_timeout(self) -> None:
        """Test _wait_for_stop returns False when job doesn't stop."""
        running_responses = []
        for _ in range(5):
            resp = MagicMock()
            resp.status_code = HTTP_OK
            resp.json.return_value = {"status": "RUNNING"}
            resp.text = '{"status": "RUNNING"}'
            running_responses.append(resp)

        with (
            patch("requests.get", side_effect=running_responses),
            patch("time.sleep"),
        ):
            result = nvcf_main._wait_for_stop("job-123")

        assert result is False

    def test_stop_ray_job_success(self) -> None:
        """Test _stop_ray_job successfully stops a job."""
        mock_response = MagicMock()
        mock_response.status_code = HTTP_OK
        mock_response.text = '{"stopped": true}'

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response
            with patch("cosmos_curator.core.cf.nvcf_main._wait_for_stop") as mock_wait:
                mock_wait.return_value = True

                success, msg = nvcf_main._stop_ray_job("job-123")

                assert success is True
                assert '{"stopped": true}' in msg

    def test_stop_ray_job_failure(self) -> None:
        """Test _stop_ray_job handles failure correctly."""
        mock_response = MagicMock()
        mock_response.status_code = HTTP_INTERNAL_SERVER_ERROR
        mock_response.text = "Internal error"

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response

            success, msg = nvcf_main._stop_ray_job("job-123")

            assert success is False
            assert msg == "Failed to stop"

    def test_terminate_last_job_success(self, mock_request_id: str, tmp_path: Path) -> None:
        """Test _terminate_last_job successfully terminates a job."""
        failed_file = tmp_path / f"failed_{mock_request_id}.txt"

        mock_jobs = {"job-1": ("SUBMISSION", "RUNNING", "msg")}

        with patch("cosmos_curator.core.cf.nvcf_main._get_request_status") as mock_status:
            mock_status.return_value = "running"
            with patch("cosmos_curator.core.cf.nvcf_main._list_all_jobs") as mock_list:
                mock_list.return_value = mock_jobs
                with patch("cosmos_curator.core.cf.nvcf_main._stop_ray_job") as mock_stop:
                    mock_stop.return_value = (True, "stopped")
                    with patch("requests.delete") as mock_delete:
                        mock_delete.return_value = MagicMock(status_code=HTTP_OK, text='{"deleted": true}')
                        with patch("cosmos_curator.core.cf.nvcf_main._get_failed_file") as mock_failed:
                            mock_failed.return_value = failed_file

                            success, msg = nvcf_main._terminate_last_job(mock_request_id)

                            assert success is True
                            assert "successfully deleted" in msg

    def test_terminate_last_job_not_running(self, mock_request_id: str) -> None:
        """Test _terminate_last_job raises error when request not running."""
        with patch("cosmos_curator.core.cf.nvcf_main._get_request_status") as mock_status:
            mock_status.return_value = "done"

            with pytest.raises(ValueError, match="status is done"):
                nvcf_main._terminate_last_job(mock_request_id)

    def test_terminate_last_job_force_terminate(self, tmp_path: Path) -> None:
        """Test _terminate_last_job with special force terminate request ID."""
        failed_file = tmp_path / "failed_force.txt"
        mock_jobs = {"job-1": ("SUBMISSION", "RUNNING", "msg")}

        with patch("cosmos_curator.core.cf.nvcf_main._list_all_jobs") as mock_list:
            mock_list.return_value = mock_jobs
            with patch("cosmos_curator.core.cf.nvcf_main._stop_ray_job") as mock_stop:
                mock_stop.return_value = (True, "stopped")
                with patch("requests.delete") as mock_delete:
                    mock_delete.return_value = MagicMock(status_code=HTTP_OK, text='{"deleted": true}')
                    with patch("cosmos_curator.core.cf.nvcf_main._get_failed_file") as mock_failed:
                        mock_failed.return_value = failed_file

                        success, _msg = nvcf_main._terminate_last_job(nvcf_main._FORCE_TERMINATE_REQUEST_ID)

                        assert success is True


class TestAssetHelpers:
    """Test asset helper functions."""

    def test_get_asset_paths(self, tmp_path: Path) -> None:
        """Test get_asset_paths extracts paths correctly."""
        asset_dir = tmp_path / "assets"
        asset_dir.mkdir()
        for asset in ("asset1", "asset2", "asset3"):
            (asset_dir / asset).write_text("dummy")

        mock_request = MagicMock()
        mock_request.headers.get.side_effect = lambda key, default=None: {
            "NVCF-ASSET-DIR": str(asset_dir),
            "NVCF-FUNCTION-ASSET-IDS": "asset1,asset2,asset3",
        }.get(key, default)

        paths = nvcf_main.get_asset_paths(mock_request)

        assert len(paths) == EXPECTED_ASSETS
        assert str(asset_dir / "asset1") in paths
        assert str(asset_dir / "asset2") in paths
        assert str(asset_dir / "asset3") in paths

    def test_get_asset_paths_empty(self) -> None:
        """Test get_asset_paths returns empty list when no assets."""
        mock_request = MagicMock()
        mock_request.headers.get.side_effect = lambda _key, default=None: default

        paths = nvcf_main.get_asset_paths(mock_request)

        assert len(paths) == 0

    def test_input_assets_present(self) -> None:
        """Test input_assets_present returns True when assets exist."""
        mock_request = MagicMock()

        with patch("cosmos_curator.core.cf.nvcf_main.get_asset_paths") as mock_paths:
            mock_paths.return_value = ["/tmp/asset1"]  # noqa: S108
            assert nvcf_main.input_assets_present(mock_request) is True

            mock_paths.return_value = []
            assert nvcf_main.input_assets_present(mock_request) is False

    def test_get_asset_output_path(self, tmp_path: Path) -> None:
        """Test get_asset_output_path returns correct path."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mock_request = MagicMock()
        mock_request.headers.get.return_value = str(output_dir)

        path = nvcf_main.get_asset_output_path(mock_request)

        assert path == str(output_dir)

    def test_get_asset_output_path_fallback(self) -> None:
        """Test get_asset_output_path creates fallback directory."""
        mock_request = MagicMock()
        mock_request.headers.get.return_value = None

        path = nvcf_main.get_asset_output_path(mock_request)

        assert path is not None
        assert "nvcf_output" in path

    def test_get_asset_input_dir(self) -> None:
        """Test get_asset_input_dir returns correct directory."""
        mock_request = MagicMock()
        mock_request.headers.get.return_value = "/tmp/input"  # noqa: S108

        input_dir = nvcf_main.get_asset_input_dir(mock_request)

        assert input_dir == "/tmp/input"  # noqa: S108


class TestPipelineProgress:
    """Test pipeline progress functions."""

    def test_get_pipeline_progress_success(self) -> None:
        """Test get_pipeline_progress returns progress correctly."""
        mock_response = MagicMock()
        mock_response.status_code = HTTP_OK
        mock_response.text = (
            "# HELP ray_pipeline_progress Pipeline progress\n"
            "# TYPE ray_pipeline_progress gauge\n"
            "ray_pipeline_progress 0.75\n"
        )

        with patch("requests.get") as mock_get:
            mock_get.return_value = mock_response
            progress = nvcf_main.get_pipeline_progress()

            assert progress == pytest.approx(PROGRESS_75_0)

    def test_get_pipeline_progress_failure(self) -> None:
        """Test get_pipeline_progress returns 0 on failure."""
        mock_response = MagicMock()
        mock_response.status_code = HTTP_INTERNAL_SERVER_ERROR

        with patch("requests.get") as mock_get:
            mock_get.return_value = mock_response
            progress = nvcf_main.get_pipeline_progress()

            assert progress == 0.0


class TestMiddleware:
    """Test PipelineLockMiddleware."""

    def test_middleware_creates_zip_file(self, mock_request_id: str) -> None:
        """Test middleware _create_zip_file creates proper zip."""
        buffer = nvcf_main.PipelineLockMiddleware._create_zip_file(mock_request_id, PROGRESS_50_0, "test log content")

        assert isinstance(buffer, io.BytesIO)

        # Verify zip contents
        with zipfile.ZipFile(buffer, "r") as zp:
            namelist = zp.namelist()
            assert f"progress-{mock_request_id}.json" in namelist
            assert f"log-{mock_request_id}.txt" in namelist

            # Check progress content
            progress_content = zp.read(f"progress-{mock_request_id}.json")
            progress_data = json.loads(progress_content)
            assert progress_data["progress"] == pytest.approx(PROGRESS_50_0)

            # Check log content
            log_content = zp.read(f"log-{mock_request_id}.txt").decode()
            assert log_content == "test log content"


class TestFastAPIEndpoints:
    """Test FastAPI endpoints."""

    def test_health_check(self, test_client: TestClient) -> None:
        """Test /health endpoint returns healthy status."""
        response = test_client.get("/health")

        assert response.status_code == HTTP_OK
        assert response.json() == {"status": "healthy"}

    def test_get_logs_missing_request_id(self, test_client: TestClient) -> None:
        """Test /v1/logs endpoint returns error when request_id missing."""
        response = test_client.get("/v1/logs")

        assert response.status_code == HTTP_UNPROCESSABLE_ENTITY

    def test_get_logs_success(self, test_client: TestClient, mock_request_id: str, tmp_path: Path) -> None:
        """Test /v1/logs endpoint returns logs successfully."""
        progress_file = tmp_path / f"progress_{mock_request_id}.json"
        progress_file.write_text(json.dumps({"progress": PROGRESS_25_5}))

        log_file = tmp_path / f"logs_{mock_request_id}.txt"
        log_file.write_text("log content here")

        with patch("cosmos_curator.core.cf.nvcf_main._read_progress_and_log_files") as mock_read:
            mock_read.return_value = (PROGRESS_25_5, "log content here")
            with patch("cosmos_curator.core.cf.nvcf_main._get_request_status") as mock_status:
                mock_status.return_value = "running"

                response = test_client.get(f"/v1/logs?request_id={mock_request_id}")

                assert response.status_code == HTTP_OK
                data = response.json()
                assert data["progress"] == pytest.approx(PROGRESS_25_5)
                assert data["status"] == "running"
                assert data["logs"] == "log content here"

    def test_get_progress_success(self, test_client: TestClient, mock_request_id: str) -> None:
        """Test /v1/progress endpoint returns progress successfully."""
        with patch("cosmos_curator.core.cf.nvcf_main._read_progress_and_log_files") as mock_read:
            mock_read.return_value = (PROGRESS_60_0, "logs")
            with patch("cosmos_curator.core.cf.nvcf_main._get_request_status") as mock_status:
                mock_status.return_value = "running"

                response = test_client.get(f"/v1/progress?request_id={mock_request_id}")

                assert response.status_code == HTTP_OK
                data = response.json()
                assert data["progress"] == pytest.approx(PROGRESS_60_0)
                assert data["status"] == "running"

    def test_run_pipeline_termination_request(self, test_client: TestClient, mock_request_id: str) -> None:
        """Test /v1/run_pipeline handles termination request."""
        with patch("cosmos_curator.core.cf.nvcf_main._terminate_last_job") as mock_terminate:
            mock_terminate.return_value = (True, "terminated successfully")

            response = test_client.post(
                "/v1/run_pipeline",
                headers={
                    "CURATOR-REQ-TERMINATE": "true",
                    "CURATOR-NVCF-REQID": mock_request_id,
                },
            )

            assert response.status_code == HTTP_OK
            data = response.json()
            assert "terminated successfully" in data["status"]

    def test_run_pipeline_termination_request_failure(self, test_client: TestClient, mock_request_id: str) -> None:
        """Test /v1/run_pipeline handles termination failure."""
        with patch("cosmos_curator.core.cf.nvcf_main._terminate_last_job") as mock_terminate:
            mock_terminate.return_value = (False, "could not terminate")

            response = test_client.post(
                "/v1/run_pipeline",
                headers={
                    "CURATOR-REQ-TERMINATE": "true",
                    "CURATOR-NVCF-REQID": mock_request_id,
                },
            )

            assert response.status_code == HTTP_PRECONDITION_FAILED
            data = response.json()
            assert "could not be terminated" in data["error"]

    def test_run_pipeline_status_check(self, test_client: TestClient, mock_request_id: str) -> None:
        """Test /v1/run_pipeline handles status check request."""
        with patch("cosmos_curator.core.cf.nvcf_main._read_progress_and_log_files") as mock_read:
            mock_read.return_value = (PROGRESS_33_3, "status log")
            with patch("cosmos_curator.core.cf.nvcf_main._get_request_status") as mock_status:
                mock_status.return_value = "running"
                with patch.dict(nvcf_main.using_nvcf_status, {"get_req_sts": False}):
                    response = test_client.post(
                        "/v1/run_pipeline",
                        headers={
                            "CURATOR-STATUS-CHECK": "true",
                            "CURATOR-NVCF-REQID": mock_request_id,
                        },
                    )

                    assert response.status_code == HTTP_OK
                    assert response.headers["CURATOR-PIPELINE-STATUS"] == "running"
                    assert "CURATOR-PIPELINE-PERCENT-COMPLETE" in response.headers


class TestProcessExecution:
    """Test process execution functions."""

    def test_do_run_process_success(self) -> None:
        """Test _do_run_process executes successfully."""
        mock_process = MagicMock()
        mock_process.stdout = io.StringIO("line 1\nline 2\n")
        mock_process.poll.side_effect = itertools.chain([None, None], itertools.repeat(0))
        mock_process.returncode = 0

        log_queue: queue.Queue[str] = queue.Queue()
        ipc_status = SimpleNamespace(value=True)

        with (
            patch("subprocess.Popen") as mock_popen,
            patch("builtins.print") as mock_print,
        ):
            mock_popen.return_value = mock_process

            nvcf_main._do_run_process(["echo", "test"], log_queue, ipc_status)  # type: ignore[arg-type]

        logged_lines: list[str] = []
        while not log_queue.empty():
            logged_lines.append(log_queue.get_nowait())

        assert logged_lines == ["line 1\n", "line 2\n"]
        assert ipc_status.value is True
        mock_print.assert_has_calls(
            [
                call("line 1\n", end="", flush=True),
                call("line 2\n", end="", flush=True),
            ]
        )

    def test_do_run_process_failure(self) -> None:
        """Test _do_run_process handles failure correctly."""
        mock_process = MagicMock()
        mock_process.stdout = io.StringIO("")
        mock_process.poll.return_value = RETURN_CODE_1
        mock_process.returncode = RETURN_CODE_1

        log_queue: queue.Queue[str] = queue.Queue()
        ipc_status = SimpleNamespace(value=True)

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = mock_process

            with pytest.raises(RuntimeError, match="Process failed with return code 1"):
                nvcf_main._do_run_process(["false"], log_queue, ipc_status)  # type: ignore[arg-type]

    def test_setup_request(self, mock_request_id: str, tmp_path: Path) -> None:
        """Test _setup_request initializes request tracking."""
        log_queue: queue.Queue[str] = queue.Queue()
        ipc_status = SimpleNamespace(value=False)
        logs: list[str] = []

        with patch("cosmos_curator.core.cf.nvcf_main._get_progress_file") as mock_progress:
            mock_progress.return_value = tmp_path / f"progress_{mock_request_id}.json"
            with patch("cosmos_curator.core.cf.nvcf_main._get_log_file") as mock_log:
                mock_log.return_value = tmp_path / f"logs_{mock_request_id}.txt"
                with patch("cosmos_curator.core.cf.nvcf_main.update_progress") as mock_update_progress:

                    def _patched_update_progress(
                        _output_dir: str,
                        stop_event: threading.Event,
                        *_: object,
                    ) -> None:
                        stop_event.wait(timeout=0.01)

                    mock_update_progress.side_effect = _patched_update_progress

                    progress_thread, stop_event = nvcf_main._setup_request(
                        str(tmp_path),
                        log_queue,  # type: ignore[arg-type]
                        ipc_status,  # type: ignore[arg-type]
                        mock_request_id,
                        logs,
                    )

                    # Verify thread started
                    assert progress_thread.is_alive()
                    assert not stop_event.is_set()
                    assert ipc_status.value is True
                    assert mock_update_progress.called

                    # Clean up
                    stop_event.set()
                    progress_thread.join()
                    assert not progress_thread.is_alive()


class TestExecutePipeline:
    """Test execute_pipeline function."""

    def test_execute_pipeline(self, mock_request_id: str) -> None:
        """Test execute_pipeline executes correctly."""
        log_queue: queue.Queue[str] = queue.Queue()
        ipc_status = SimpleNamespace(value=True)
        stop_event = threading.Event()

        mock_func = MagicMock()
        mock_wrapper = MagicMock()
        pipeline_args = argparse.Namespace(test_arg="value")

        with patch("cosmos_curator.core.cf.nvcf_main.ProcessPoolExecutor") as mock_executor_cls:
            mock_executor = MagicMock()
            mock_future = MagicMock()
            mock_future.result.return_value = None
            mock_executor.submit.return_value = mock_future
            mock_executor.__enter__.return_value = mock_executor
            mock_executor.__exit__.return_value = None
            mock_executor_cls.return_value = mock_executor

            nvcf_main.execute_pipeline(
                mock_wrapper,
                mock_func,
                mock_request_id,
                pipeline_args,
                log_queue,  # type: ignore[arg-type]
                ipc_status,  # type: ignore[arg-type]
                stop_event,
            )

            # Verify executor was used
            mock_executor.submit.assert_called_once()
            assert stop_event.is_set()


def test_setup_pipeline_middleware() -> None:
    """Test setup_pipeline_middleware configures middleware correctly."""
    app = FastAPI()

    with patch("cosmos_curator.core.cf.nvcf_main.is_nvcf_helm_deployment") as mock_helm:
        mock_helm.return_value = False
        with patch("cosmos_curator.core.cf.nvcf_main.is_nvcf_container_deployment") as mock_container:
            mock_container.return_value = True

            nvcf_main.setup_pipeline_middleware(app)

            # Verify middleware was added (checking middleware stack)
            assert len(app.user_middleware) > 0
