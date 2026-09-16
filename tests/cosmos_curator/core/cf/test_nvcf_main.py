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
import asyncio
import io
import itertools
import json
import queue
import tempfile
import threading
import uuid
import zipfile
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, call, patch

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from cosmos_curator.core.cf import nvcf_main
from cosmos_curator.core.utils.misc.stage_replay import add_stage_replay_args

if TYPE_CHECKING:
    from collections.abc import Callable

# HTTP status codes
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
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

    def test_request_id_rejects_path_separators(self) -> None:
        """Reject request IDs before using them in temp-file paths."""
        with pytest.raises(ValueError, match="request_id"):
            nvcf_main._get_progress_file("../escape")

    def test_observability_env_defaults_fill_missing_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Chart-level observability overrides apply when invoke payload omits fields."""
        monkeypatch.setenv("COSMOS_CURATOR_OTLP_METRICS_PUSH", "true")
        monkeypatch.setenv("COSMOS_CURATOR_OTLP_METRICS_PUSH_INTERVAL", "17")
        monkeypatch.setenv("COSMOS_CURATOR_PROFILE_TRACING", "true")
        monkeypatch.setenv("COSMOS_CURATOR_PROFILE_TRACING_SAMPLING", "0.5")
        monkeypatch.setenv("COSMOS_CURATOR_OTLP_RUN_ATTRIBUTES_VALUES", '{"function_id":"test-function"}')

        args = argparse.Namespace(input_video_path="s3://in")

        nvcf_main._apply_observability_env_defaults(args)

        assert args.otlp_metrics_push is True
        assert args.otlp_metrics_push_interval == 17
        assert args.profile_tracing is True
        assert args.profile_tracing_sampling == 0.5
        assert args.otlp_run_attributes_map == {"function_id": "test-function"}

    def test_observability_env_defaults_do_not_duplicate_pipeline_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Absent env overrides leave missing fields to existing pipeline defaults."""
        monkeypatch.delenv("COSMOS_CURATOR_OTLP_METRICS_PUSH", raising=False)
        monkeypatch.delenv("COSMOS_CURATOR_OTLP_METRICS_PUSH_INTERVAL", raising=False)
        monkeypatch.delenv("COSMOS_CURATOR_PROFILE_TRACING", raising=False)
        monkeypatch.delenv("COSMOS_CURATOR_PROFILE_TRACING_SAMPLING", raising=False)
        monkeypatch.delenv("COSMOS_CURATOR_OTLP_RUN_ATTRIBUTES_VALUES", raising=False)

        args = argparse.Namespace(input_video_path="s3://in")

        nvcf_main._apply_observability_env_defaults(args)

        assert not hasattr(args, "otlp_metrics_push")
        assert not hasattr(args, "otlp_metrics_push_interval")
        assert not hasattr(args, "profile_tracing")
        assert not hasattr(args, "profile_tracing_sampling")
        assert not hasattr(args, "otlp_run_attributes_map")

    def test_observability_env_defaults_preserve_invoke_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit invoke payload args remain authoritative over env defaults."""
        monkeypatch.setenv("COSMOS_CURATOR_OTLP_METRICS_PUSH", "true")
        monkeypatch.setenv("COSMOS_CURATOR_OTLP_METRICS_PUSH_INTERVAL", "17")
        monkeypatch.setenv("COSMOS_CURATOR_PROFILE_TRACING", "true")
        monkeypatch.setenv("COSMOS_CURATOR_PROFILE_TRACING_SAMPLING", "0.5")
        monkeypatch.setenv("COSMOS_CURATOR_OTLP_RUN_ATTRIBUTES_VALUES", '{"function_id":"test-function"}')

        args = argparse.Namespace(
            otlp_metrics_push=False,
            otlp_metrics_push_interval=9,
            profile_tracing=False,
            profile_tracing_sampling=0.25,
            otlp_run_attributes_map={"function_id": "invoke-function"},
        )

        nvcf_main._apply_observability_env_defaults(args)

        assert args.otlp_metrics_push is False
        assert args.otlp_metrics_push_interval == 9
        assert args.profile_tracing is False
        assert args.profile_tracing_sampling == 0.25
        assert args.otlp_run_attributes_map == {"function_id": "invoke-function"}

    @pytest.mark.parametrize(
        ("name", "value", "attr_name"),
        [
            ("COSMOS_CURATOR_OTLP_METRICS_PUSH_INTERVAL", "0", "otlp_metrics_push_interval"),
            ("COSMOS_CURATOR_PROFILE_TRACING_SAMPLING", "-0.1", "profile_tracing_sampling"),
            ("COSMOS_CURATOR_PROFILE_TRACING_SAMPLING", "1.1", "profile_tracing_sampling"),
            ("COSMOS_CURATOR_PROFILE_TRACING_SAMPLING", "nan", "profile_tracing_sampling"),
        ],
    )
    def test_observability_env_defaults_ignore_out_of_range_numbers(
        self,
        monkeypatch: pytest.MonkeyPatch,
        name: str,
        value: str,
        attr_name: str,
    ) -> None:
        """Chart overrides outside the CLI validators' bounds remain unset."""
        monkeypatch.setenv(name, value)
        args = argparse.Namespace()

        nvcf_main._apply_observability_env_defaults(args)

        assert not hasattr(args, attr_name)

    @pytest.mark.parametrize("value", ["not-json", "[]"])
    def test_observability_env_defaults_ignore_invalid_run_attributes(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """Invalid chart-injected run attributes do not reach profiling args."""
        monkeypatch.setenv("COSMOS_CURATOR_OTLP_RUN_ATTRIBUTES_VALUES", value)

        args = argparse.Namespace()

        nvcf_main._apply_observability_env_defaults(args)

        assert not hasattr(args, "otlp_run_attributes_map")

    def test_observability_env_defaults_preserve_explicit_empty_run_attributes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicitly empty chart value clears run attributes."""
        monkeypatch.setenv("COSMOS_CURATOR_OTLP_RUN_ATTRIBUTES_VALUES", "")

        args = argparse.Namespace()

        nvcf_main._apply_observability_env_defaults(args)

        assert args.otlp_run_attributes_map == {}

    def test_observability_env_defaults_warn_for_empty_run_attribute_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dropped empty labels are reported without logging their values."""
        monkeypatch.setenv(
            "COSMOS_CURATOR_OTLP_RUN_ATTRIBUTES_VALUES",
            '{"function_id":"test-function","empty":"","missing":null}',
        )

        with patch.object(nvcf_main.logger, "warning") as mock_warning:
            args = argparse.Namespace()
            nvcf_main._apply_observability_env_defaults(args)

        assert args.otlp_run_attributes_map == {"function_id": "test-function"}
        mock_warning.assert_called_once_with(
            "Ignoring run attributes with invalid values from COSMOS_CURATOR_OTLP_RUN_ATTRIBUTES_VALUES: empty, missing"
        )


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
                mock_read.assert_called_once_with(mock_request_id, read_log=False)

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

    def test_run_pipeline_status_check_reads_nvcf_status_progress_file(
        self, test_client: TestClient, mock_request_id: str, tmp_path: Path
    ) -> None:
        """Test status check reads output-dir progress when present."""
        progress_file = tmp_path / "progress"
        progress_file.write_text(
            json.dumps({"partialResponse": {"progress": "42.25", "logs": "nvcf status log"}}), encoding="utf-8"
        )
        output_dir_file = tmp_path / f"request_output_dir_{mock_request_id}.txt"

        with patch("tempfile.gettempdir", return_value=str(tmp_path)):
            nvcf_main._write_request_output_dir(mock_request_id, str(tmp_path))
            nvcf_main._mark_direct_request(mock_request_id)
            assert output_dir_file.exists()
            with (
                patch("cosmos_curator.core.cf.nvcf_main._get_request_status", return_value="running"),
                patch.dict(nvcf_main.using_nvcf_status, {"get_req_sts": True}),
            ):
                response = test_client.post(
                    "/v1/run_pipeline",
                    headers={
                        "CURATOR-STATUS-CHECK": "true",
                        "CURATOR-NVCF-REQID": mock_request_id,
                    },
                )

        assert response.status_code == HTTP_OK
        assert response.headers["CURATOR-PIPELINE-PERCENT-COMPLETE"] == "42.25"
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            assert zf.read(f"log-{mock_request_id}.txt").decode().startswith("nvcf status log")

    def test_run_pipeline_status_check_handles_unmarked_request_when_nvcf_status_is_enabled(
        self, mock_request_id: str
    ) -> None:
        """Test status probes use the full endpoint even for pexec-enqueued requests."""
        middleware = nvcf_main.PipelineLockMiddleware(FastAPI())
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/run_pipeline"),
            headers={"CURATOR-STATUS-CHECK": "true", "CURATOR-NVCF-REQID": mock_request_id},
        )

        async def call_next(_request: object) -> JSONResponse:
            return JSONResponse(status_code=HTTP_OK, content={"passed_through": True})

        with (
            patch.dict(nvcf_main.using_nvcf_status, {"get_req_sts": True}),
            patch.object(
                nvcf_main.PipelineLockMiddleware,
                "_handle_status_check",
                return_value=JSONResponse(status_code=HTTP_OK, content={"handled": True}),
            ) as mock_status_check,
        ):
            response = asyncio.run(middleware.dispatch(request, call_next))  # type: ignore[arg-type]

        assert response.status_code == HTTP_OK
        assert json.loads(response.body) == {"handled": True}
        mock_status_check.assert_called_once()

    def test_run_pipeline_status_check_handles_marked_direct_request_when_nvcf_status_is_enabled(
        self, mock_request_id: str, tmp_path: Path
    ) -> None:
        """Test direct status probes are handled after direct invocation creates the request."""
        middleware = nvcf_main.PipelineLockMiddleware(FastAPI())
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/run_pipeline"),
            headers={"CURATOR-STATUS-CHECK": "true", "CURATOR-NVCF-REQID": mock_request_id},
        )

        async def call_next(_request: object) -> JSONResponse:
            return JSONResponse(status_code=HTTP_OK, content={"passed_through": True})

        with (
            patch("tempfile.gettempdir", return_value=str(tmp_path)),
            patch.dict(nvcf_main.using_nvcf_status, {"get_req_sts": True}),
            patch.object(
                nvcf_main.PipelineLockMiddleware,
                "_handle_status_check",
                return_value=JSONResponse(status_code=HTTP_OK, content={"handled": True}),
            ) as mock_status_check,
        ):
            nvcf_main._mark_direct_request(mock_request_id)
            response = asyncio.run(middleware.dispatch(request, call_next))  # type: ignore[arg-type]

        assert response.status_code == HTTP_OK
        assert json.loads(response.body) == {"handled": True}
        mock_status_check.assert_called_once()

    def test_run_pipeline_rechecks_busy_status_after_acquiring_lock(self) -> None:
        """Test a racing invoke is rejected if busy marker appears after the first check."""
        middleware = nvcf_main.PipelineLockMiddleware(FastAPI())
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/run_pipeline"), headers={})

        async def call_next(_request: object) -> JSONResponse:
            return JSONResponse(status_code=HTTP_OK, content={"accepted": True})

        with patch.object(middleware, "_is_pipeline_busy", side_effect=[False, True]) as mock_busy:
            response = asyncio.run(middleware.dispatch(request, call_next))  # type: ignore[arg-type]

        assert response.status_code == 429
        assert json.loads(response.body) == {"error": "Pipeline is currently busy. Please try again later."}
        assert mock_busy.call_count == 2

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

    def test_run_pipeline_status_check_can_skip_logs(self, test_client: TestClient, mock_request_id: str) -> None:
        """Test status checks can return only status/progress headers."""
        with patch("cosmos_curator.core.cf.nvcf_main._read_progress_and_log_files") as mock_read:
            mock_read.return_value = (PROGRESS_33_3, "")
            with patch("cosmos_curator.core.cf.nvcf_main._get_request_status") as mock_status:
                mock_status.return_value = "running"
                with patch.dict(nvcf_main.using_nvcf_status, {"get_req_sts": False}):
                    response = test_client.post(
                        "/v1/run_pipeline",
                        headers={
                            "CURATOR-STATUS-CHECK": "true",
                            "CURATOR-STATUS-INCLUDE-LOGS": "false",
                            "CURATOR-NVCF-REQID": mock_request_id,
                        },
                    )

        assert response.status_code == HTTP_OK
        assert response.content == b""
        assert response.headers["CURATOR-PIPELINE-STATUS"] == "running"
        assert response.headers["CURATOR-PIPELINE-PERCENT-COMPLETE"] == "33.30"
        assert response.headers.get("content-type") != "application/zip"
        mock_read.assert_called_once_with(mock_request_id, read_log=False)

    def test_run_pipeline_pexec_request_runs_synchronously(self, test_client: TestClient, mock_request_id: str) -> None:
        """Test pexec-shaped requests keep the synchronous request lifecycle."""
        fake_manager = MagicMock()
        fake_manager.Value.return_value = SimpleNamespace(value=False)
        fake_manager.Queue.return_value = queue.Queue()
        fake_manager.list.return_value = []
        fake_thread = MagicMock()
        fake_stop_event = threading.Event()

        with (
            patch("cosmos_curator.core.cf.nvcf_main.Manager", return_value=fake_manager),
            patch("cosmos_curator.core.cf.nvcf_main._setup_request", return_value=(fake_thread, fake_stop_event)),
            patch("cosmos_curator.core.cf.nvcf_main.execute_pipeline") as mock_run,
            patch("cosmos_curator.core.cf.nvcf_main.gather_and_upload_outputs"),
            patch("cosmos_curator.core.cf.nvcf_main.cleanup_server_input_workspace") as mock_input_cleanup,
        ):
            response = test_client.post(
                "/v1/run_pipeline",
                headers={"NVCF-REQID": mock_request_id},
                json={"pipeline": "split", "args": {"input_video_path": "s3://bucket/in", "output_clip_path": "/out"}},
            )

            assert response.status_code == HTTP_OK
            assert response.json()["message"] == "Pipeline executed successfully"
            mock_run.assert_called_once()
            # Input-workspace cleanup must run unconditionally, success included --
            # nothing gates it on ipc_status the way output cleanup/upload is gated.
            mock_input_cleanup.assert_called_once()

    def test_run_pipeline_rejects_debug_pipeline_args(
        self,
        test_client: TestClient,
        mock_request_id: str,
    ) -> None:
        """--stage-replay and friends are only needed for debugging; reject for remote invokes."""
        debug_arg_name = "stage_replay"
        fake_manager = MagicMock()
        fake_manager.Value.return_value = SimpleNamespace(value=False)
        fake_manager.Queue.return_value = queue.Queue()
        fake_manager.list.return_value = []
        fake_thread = MagicMock()
        fake_stop_event = threading.Event()

        with (
            patch("cosmos_curator.core.cf.nvcf_main.Manager", return_value=fake_manager),
            patch("cosmos_curator.core.cf.nvcf_main._setup_request", return_value=(fake_thread, fake_stop_event)),
            patch("cosmos_curator.core.cf.nvcf_main.execute_pipeline") as mock_run,
            patch("cosmos_curator.core.cf.nvcf_main.gather_and_upload_outputs") as mock_upload,
        ):
            response = test_client.post(
                "/v1/run_pipeline",
                headers={"NVCF-REQID": mock_request_id},
                json={
                    "pipeline": "split",
                    "args": {
                        "input_video_path": "s3://bucket/in",
                        "output_clip_path": "/out",
                        debug_arg_name: ["SomeStage"],
                    },
                },
            )

            assert response.status_code == HTTP_BAD_REQUEST
            assert debug_arg_name in response.json()["error"]
            mock_run.assert_not_called()
            mock_upload.assert_not_called()

    def test_rejected_debug_pipeline_args_cover_every_stage_replay_flag(self) -> None:
        """The blocklist must track stage_replay.py's real args, not a hand-copied list."""
        parser = argparse.ArgumentParser()
        add_stage_replay_args(parser)
        all_dests = {action.dest for action in parser._actions if action.dest != "help"}

        assert all_dests == nvcf_main._NVCF_REJECTED_ARG_NAMES

    def test_run_pipeline_rejects_allow_unsafe_clip_location(
        self,
        test_client: TestClient,
        mock_request_id: str,
    ) -> None:
        """Reject the trusted-CLI-only clip-location compatibility option for remote invocations."""
        fake_manager = MagicMock()
        fake_manager.Value.return_value = SimpleNamespace(value=False)
        fake_manager.Queue.return_value = queue.Queue()
        fake_manager.list.return_value = []
        fake_thread = MagicMock()
        fake_stop_event = threading.Event()

        with (
            patch("cosmos_curator.core.cf.nvcf_main.Manager", return_value=fake_manager),
            patch("cosmos_curator.core.cf.nvcf_main._setup_request", return_value=(fake_thread, fake_stop_event)),
            patch("cosmos_curator.core.cf.nvcf_main.execute_pipeline") as mock_run,
            patch("cosmos_curator.core.cf.nvcf_main.gather_and_upload_outputs") as mock_upload,
        ):
            response = test_client.post(
                "/v1/run_pipeline",
                headers={"NVCF-REQID": mock_request_id},
                json={
                    "pipeline": "shard",
                    "args": {
                        "input_clip_path": "s3://bucket/in",
                        "output_dataset_path": "s3://bucket/out",
                        "allow_unsafe_clip_location": True,
                    },
                },
            )

            assert response.status_code == HTTP_BAD_REQUEST
            assert "allow_unsafe_clip_location" in response.json()["error"]
            mock_run.assert_not_called()
            mock_upload.assert_not_called()

    def test_run_pipeline_pexec_ray_job_failure_is_logged_once_without_traceback(
        self, test_client: TestClient, mock_request_id: str
    ) -> None:
        """Synchronous Ray job failures return without duplicating captured diagnostics."""
        fake_manager = MagicMock()
        fake_manager.Value.return_value = SimpleNamespace(value=False)
        fake_manager.Queue.return_value = queue.Queue()
        fake_manager.list.return_value = []
        fake_thread = MagicMock()
        fake_stop_event = threading.Event()
        message = "Ray job failed with return code 1"

        with (
            patch("cosmos_curator.core.cf.nvcf_main.Manager", return_value=fake_manager),
            patch("cosmos_curator.core.cf.nvcf_main._setup_request", return_value=(fake_thread, fake_stop_event)),
            patch(
                "cosmos_curator.core.cf.nvcf_main.execute_pipeline",
                side_effect=nvcf_main.RayJobLoggedError(message),
            ),
            patch(
                "cosmos_curator.core.cf.nvcf_main._read_progress_and_log_files",
                return_value=(None, "driver traceback was logged\n"),
            ),
            patch("cosmos_curator.core.cf.nvcf_main.logger.error") as mock_error,
            patch.dict(nvcf_main.using_nvcf_status, {"get_req_sts": False}),
        ):
            response = test_client.post(
                "/v1/run_pipeline",
                headers={"NVCF-REQID": mock_request_id},
                json={"pipeline": "split", "args": {"input_video_path": "s3://bucket/in", "output_clip_path": "/out"}},
            )

        assert response.status_code == HTTP_INTERNAL_SERVER_ERROR
        error_details = response.json()["error"]
        assert f"exception: {message}" in error_details
        assert "exception_type: RayJobLoggedError" in error_details
        assert "logs: driver traceback was logged\n" in error_details
        assert "traceback:" not in error_details
        mock_error.assert_called_once_with(f"Pipeline failed for request {mock_request_id}; details in Ray job log")

    def test_run_pipeline_pexec_failure_does_not_upload_outputs(
        self, test_client: TestClient, mock_request_id: str
    ) -> None:
        """A synchronous failure must not trigger an upload, even with the optimistic ipc_status default."""
        fake_manager = MagicMock()
        # Mirrors _setup_request's real "assume success at start" default; that
        # default is what must not survive a failed run.
        fake_ipc_status = SimpleNamespace(value=True)
        fake_manager.Value.return_value = fake_ipc_status
        fake_manager.Queue.return_value = queue.Queue()
        fake_manager.list.return_value = []
        fake_thread = MagicMock()
        fake_stop_event = threading.Event()

        with (
            patch("cosmos_curator.core.cf.nvcf_main.Manager", return_value=fake_manager),
            patch("cosmos_curator.core.cf.nvcf_main._setup_request", return_value=(fake_thread, fake_stop_event)),
            patch(
                "cosmos_curator.core.cf.nvcf_main.execute_pipeline",
                side_effect=RuntimeError("boom"),
            ),
            patch("cosmos_curator.core.cf.nvcf_main.gather_and_upload_outputs") as mock_upload,
            patch("cosmos_curator.core.cf.nvcf_main.cleanup_server_output_workspace") as mock_cleanup,
            patch("cosmos_curator.core.cf.nvcf_main.cleanup_server_input_workspace") as mock_input_cleanup,
            patch.dict(nvcf_main.using_nvcf_status, {"get_req_sts": False}),
        ):
            response = test_client.post(
                "/v1/run_pipeline",
                headers={"NVCF-REQID": mock_request_id},
                json={"pipeline": "split", "args": {"input_video_path": "s3://bucket/in", "output_clip_path": "/out"}},
            )

        assert response.status_code == HTTP_INTERNAL_SERVER_ERROR
        assert fake_ipc_status.value is False
        mock_upload.assert_not_called()
        mock_cleanup.assert_called_once()
        assert mock_cleanup.call_args.args[0] == "split"
        # Input-workspace cleanup must also run on failure -- it isn't gated on
        # ipc_status at all, unlike the output-side upload/cleanup choice above.
        mock_input_cleanup.assert_called_once()

    def test_run_pipeline_rejects_local_input_video_path(self, test_client: TestClient, mock_request_id: str) -> None:
        """A caller-supplied local input_video_path must be rejected before dispatch."""
        fake_manager = MagicMock()
        fake_manager.Value.return_value = SimpleNamespace(value=True)
        fake_manager.Queue.return_value = queue.Queue()
        fake_manager.list.return_value = []
        fake_thread = MagicMock()
        fake_stop_event = threading.Event()

        with (
            patch("cosmos_curator.core.cf.nvcf_main.Manager", return_value=fake_manager),
            patch("cosmos_curator.core.cf.nvcf_main._setup_request", return_value=(fake_thread, fake_stop_event)),
            patch("cosmos_curator.core.cf.nvcf_main.execute_pipeline") as mock_run,
            patch.dict(nvcf_main.using_nvcf_status, {"get_req_sts": False}),
        ):
            response = test_client.post(
                "/v1/run_pipeline",
                headers={"NVCF-REQID": mock_request_id},
                json={"pipeline": "split", "args": {"input_video_path": "/var/secrets", "output_clip_path": "/out"}},
            )

        assert response.status_code == HTTP_INTERNAL_SERVER_ERROR
        assert "input_video_path" in response.json()["error"]
        mock_run.assert_not_called()

    def test_run_pipeline_direct_request_without_nvcf_request_id_uses_generated_fallback(
        self, test_client: TestClient
    ) -> None:
        """Test direct requests preserve the historical generated request ID fallback."""
        fake_manager = MagicMock()
        ipc_status = SimpleNamespace(value=False)
        fake_manager.Value.return_value = ipc_status
        fake_manager.Queue.return_value = queue.Queue()
        fake_manager.list.return_value = []
        fake_stop_event = threading.Event()
        executed = threading.Event()
        joined = threading.Event()

        class FakeProgressThread:
            def join(self) -> None:
                joined.set()

        def fake_execute(*_args: object) -> None:
            ipc_status.value = True
            executed.set()

        with (
            patch("cosmos_curator.core.cf.nvcf_main.Manager", return_value=fake_manager),
            patch(
                "cosmos_curator.core.cf.nvcf_main._setup_request",
                return_value=(FakeProgressThread(), fake_stop_event),
            ),
            patch("cosmos_curator.core.cf.nvcf_main.execute_pipeline", side_effect=fake_execute),
            patch("cosmos_curator.core.cf.nvcf_main.gather_and_upload_outputs"),
        ):
            response = test_client.post(
                "/v1/run_pipeline",
                headers={"CURATOR-DIRECT-MODE": "true"},
                json={"pipeline": "split", "args": {"input_video_path": "s3://bucket/in", "output_clip_path": "/out"}},
            )
            assert executed.wait(timeout=1)
            assert joined.wait(timeout=1)

        assert response.status_code == HTTP_OK
        response_body = response.json()
        assert uuid.UUID(response_body["reqid"])
        assert response_body["status"] == "in-progress"
        assert "nvcf-status" not in response.headers
        assert "nvcf-reqid" not in response.headers

    def test_run_pipeline_direct_request_returns_request_id(
        self, test_client: TestClient, mock_request_id: str
    ) -> None:
        """Test direct requests return quickly and upload before publishing terminal status."""
        fake_manager = MagicMock()
        ipc_status = SimpleNamespace(value=False)
        fake_manager.Value.return_value = ipc_status
        fake_manager.Queue.return_value = queue.Queue()
        fake_manager.list.return_value = []
        fake_stop_event = threading.Event()
        executed = threading.Event()
        joined = threading.Event()
        events: list[str] = []

        class FakeProgressThread:
            def join(self) -> None:
                events.append("join")
                joined.set()

        def fake_execute(*_args: object) -> None:
            ipc_status.value = True
            events.append("execute")
            executed.set()

        def fake_upload(*_args: object) -> None:
            events.append("upload")

        def fake_stop() -> None:
            events.append("stop")
            threading.Event.set(fake_stop_event)

        with (
            patch("cosmos_curator.core.cf.nvcf_main.Manager", return_value=fake_manager),
            patch(
                "cosmos_curator.core.cf.nvcf_main._setup_request",
                return_value=(FakeProgressThread(), fake_stop_event),
            ),
            patch("cosmos_curator.core.cf.nvcf_main.execute_pipeline", side_effect=fake_execute),
            patch("cosmos_curator.core.cf.nvcf_main.gather_and_upload_outputs", side_effect=fake_upload),
            patch.object(fake_stop_event, "set", side_effect=fake_stop),
        ):
            response = test_client.post(
                "/v1/run_pipeline",
                headers={"CURATOR-DIRECT-MODE": "true", "NVCF-REQID": mock_request_id},
                json={
                    "pipeline": "split",
                    "args": {
                        "input_video_path": "s3://bucket/in",
                        "output_clip_path": "/out",
                        "output_presigned_s3_url": "https://example.test/output.zip",
                    },
                },
            )
            assert executed.wait(timeout=1)
            assert joined.wait(timeout=1)

            assert response.status_code == HTTP_OK
            assert response.json()["reqid"] == mock_request_id
            assert response.json()["status"] == "in-progress"
            assert "nvcf-status" not in response.headers
            assert "nvcf-reqid" not in response.headers
            assert response.headers["CURATOR-PIPELINE-STATUS"] == "running"
            assert response.headers["CURATOR-PIPELINE-PERCENT-COMPLETE"] == "0.00"
            assert events == ["execute", "upload", "stop", "join"]

    def test_run_pipeline_direct_upload_failure_marks_request_failed(
        self, test_client: TestClient, mock_request_id: str
    ) -> None:
        """Test direct upload failures are visible before terminal progress status is written."""
        fake_manager = MagicMock()
        ipc_status = SimpleNamespace(value=False)
        fake_manager.Value.return_value = ipc_status
        fake_manager.Queue.return_value = queue.Queue()
        fake_manager.list.return_value = []
        fake_stop_event = threading.Event()
        executed = threading.Event()
        joined = threading.Event()
        events: list[str] = []

        class FakeProgressThread:
            def join(self) -> None:
                events.append(f"join:{ipc_status.value}")
                joined.set()

        def fake_execute(*_args: object) -> None:
            ipc_status.value = True
            events.append("execute")
            executed.set()

        def fake_upload(*_args: object) -> None:
            events.append("upload")
            msg = "upload failed"
            raise RuntimeError(msg)

        def fake_stop() -> None:
            events.append(f"stop:{ipc_status.value}")
            threading.Event.set(fake_stop_event)

        with (
            patch("cosmos_curator.core.cf.nvcf_main.Manager", return_value=fake_manager),
            patch(
                "cosmos_curator.core.cf.nvcf_main._setup_request",
                return_value=(FakeProgressThread(), fake_stop_event),
            ),
            patch("cosmos_curator.core.cf.nvcf_main.execute_pipeline", side_effect=fake_execute),
            patch("cosmos_curator.core.cf.nvcf_main.gather_and_upload_outputs", side_effect=fake_upload),
            patch.object(fake_stop_event, "set", side_effect=fake_stop),
        ):
            response = test_client.post(
                "/v1/run_pipeline",
                headers={"CURATOR-DIRECT-MODE": "true", "NVCF-REQID": mock_request_id},
                json={
                    "pipeline": "split",
                    "args": {
                        "input_video_path": "s3://bucket/in",
                        "output_clip_path": "/out",
                        "output_presigned_s3_url": "https://example.test/output.zip",
                    },
                },
            )
            assert executed.wait(timeout=1)
            assert joined.wait(timeout=1)

            assert response.status_code == HTTP_OK
            assert events == ["execute", "upload", "stop:False", "join:False"]
            assert ipc_status.value is False

    def test_run_pipeline_direct_ray_job_failure_is_logged_once_without_traceback(
        self, test_client: TestClient, mock_request_id: str
    ) -> None:
        """Direct Ray job failures do not duplicate diagnostics captured from the CLI."""
        fake_manager = MagicMock()
        ipc_status = SimpleNamespace(value=False)
        fake_manager.Value.return_value = ipc_status
        fake_manager.Queue.return_value = queue.Queue()
        fake_manager.list.return_value = []
        fake_stop_event = threading.Event()
        executed = threading.Event()
        joined = threading.Event()

        class FakeProgressThread:
            def join(self) -> None:
                joined.set()

        def fake_execute(*_args: object) -> None:
            executed.set()
            message = "Ray job failed with return code 1"
            raise nvcf_main.RayJobLoggedError(message)

        with (
            patch("cosmos_curator.core.cf.nvcf_main.Manager", return_value=fake_manager),
            patch(
                "cosmos_curator.core.cf.nvcf_main._setup_request",
                return_value=(FakeProgressThread(), fake_stop_event),
            ),
            patch("cosmos_curator.core.cf.nvcf_main.execute_pipeline", side_effect=fake_execute),
            patch("cosmos_curator.core.cf.nvcf_main.gather_and_upload_outputs") as mock_upload,
            patch("cosmos_curator.core.cf.nvcf_main.logger.error") as mock_error,
            patch("cosmos_curator.core.cf.nvcf_main.logger.exception") as mock_exception,
        ):
            response = test_client.post(
                "/v1/run_pipeline",
                headers={"CURATOR-DIRECT-MODE": "true", "NVCF-REQID": mock_request_id},
                json={"pipeline": "split", "args": {"input_video_path": "s3://bucket/in", "output_clip_path": "/out"}},
            )
            assert executed.wait(timeout=1)
            assert joined.wait(timeout=1)

        assert response.status_code == HTTP_OK
        assert ipc_status.value is False
        mock_upload.assert_not_called()
        mock_error.assert_called_once_with(f"Pipeline failed for request {mock_request_id}; details in Ray job log")
        mock_exception.assert_not_called()

    def test_run_pipeline_direct_non_presigned_request_returns_request_id(
        self, test_client: TestClient, mock_request_id: str
    ) -> None:
        """Test direct non-presigned requests do not add a separate upload step."""
        fake_manager = MagicMock()
        fake_manager.Value.return_value = SimpleNamespace(value=False)
        fake_manager.Queue.return_value = queue.Queue()
        fake_manager.list.return_value = []
        fake_stop_event = threading.Event()
        executed = threading.Event()
        joined = threading.Event()
        events: list[str] = []

        class FakeProgressThread:
            def join(self) -> None:
                events.append("join")
                joined.set()

        def fake_execute(*_args: object) -> None:
            events.append("execute")
            executed.set()

        def fake_stop() -> None:
            events.append("stop")
            threading.Event.set(fake_stop_event)

        with (
            patch("cosmos_curator.core.cf.nvcf_main.Manager", return_value=fake_manager),
            patch(
                "cosmos_curator.core.cf.nvcf_main._setup_request",
                return_value=(FakeProgressThread(), fake_stop_event),
            ),
            patch("cosmos_curator.core.cf.nvcf_main.execute_pipeline", side_effect=fake_execute),
            patch.object(fake_stop_event, "set", side_effect=fake_stop),
        ):
            response = test_client.post(
                "/v1/run_pipeline",
                headers={"CURATOR-DIRECT-MODE": "true", "NVCF-REQID": mock_request_id},
                json={
                    "pipeline": "split",
                    "args": {"input_video_path": "s3://bucket/in", "output_clip_path": "/out"},
                },
            )
            assert executed.wait(timeout=1)
            assert joined.wait(timeout=1)

            assert response.status_code == HTTP_OK
            assert response.json()["reqid"] == mock_request_id
            assert response.json()["status"] == "in-progress"
            assert "nvcf-status" not in response.headers
            assert "nvcf-reqid" not in response.headers
            assert events == ["execute", "stop", "join"]

    @pytest.mark.parametrize(
        ("method", "path", "request_kwargs"),
        [
            ("GET", "/v1/logs", {"params": {"request_id": "../escape"}}),
            ("GET", "/v1/progress", {"params": {"request_id": "../escape"}}),
            (
                "POST",
                "/v1/run_pipeline",
                {"headers": {"CURATOR-STATUS-CHECK": "true", "CURATOR-NVCF-REQID": "../escape"}},
            ),
            (
                "POST",
                "/v1/run_pipeline",
                {"headers": {"CURATOR-REQ-TERMINATE": "true", "CURATOR-NVCF-REQID": "../escape"}},
            ),
        ],
    )
    def test_request_id_entrypoints_reject_invalid_request_ids(
        self, test_client: TestClient, method: str, path: str, request_kwargs: dict[str, object]
    ) -> None:
        """Test externally supplied request IDs are rejected at the HTTP boundary."""
        with patch.dict(nvcf_main.using_nvcf_status, {"get_req_sts": False}):
            response = test_client.request(method, path, **request_kwargs)

            assert response.status_code == HTTP_BAD_REQUEST
            assert "request_id" in response.json()["error"]

    def test_run_pipeline_dispatches_annotate(
        self, test_client: TestClient, mock_request_id: str, tmp_path: Path
    ) -> None:
        """Test /v1/run_pipeline routes image annotate invokes to the public entry point."""
        fake_manager = MagicMock()
        fake_manager.Value.return_value = SimpleNamespace(value=False)
        fake_manager.Queue.return_value = queue.Queue()
        fake_manager.list.return_value = []
        fake_thread = MagicMock()
        fake_stop_event = threading.Event()
        captured: dict[str, object] = {}

        def fake_annotate(args: argparse.Namespace) -> None:
            captured["pipeline_args"] = args

        def fake_execute_pipeline(
            _wrapper: object,
            func: "Callable[[argparse.Namespace], None]",
            request_id: str,
            pipeline_args: argparse.Namespace,
            *_args: object,
        ) -> None:
            captured["request_id"] = request_id
            func(pipeline_args)

        with (
            patch.dict(nvcf_main.using_nvcf_status, {"get_req_sts": False}),
            patch("cosmos_curator.core.cf.nvcf_main.Manager", return_value=fake_manager),
            patch("cosmos_curator.core.cf.nvcf_main._setup_request", return_value=(fake_thread, fake_stop_event)),
            patch("cosmos_curator.core.cf.nvcf_main.get_nvcf_output_path", return_value=str(tmp_path / "out")),
            patch("cosmos_curator.core.cf.nvcf_main.execute_pipeline", side_effect=fake_execute_pipeline),
            patch("cosmos_curator.core.cf.nvcf_main.nvcf_run_annotate", side_effect=fake_annotate),
            patch("cosmos_curator.core.cf.nvcf_main.gather_and_upload_outputs"),
        ):
            response = test_client.post(
                "/v1/run_pipeline",
                headers={"NVCF-REQID": mock_request_id},
                json={
                    "pipeline": "annotate",
                    "args": {
                        "input_image_path": "s3://bucket/in",
                        "output_path": str(tmp_path / "out"),
                        "limit": 1,
                    },
                },
            )

        assert response.status_code == HTTP_OK
        assert captured["request_id"] == mock_request_id
        pipeline_args = captured["pipeline_args"]
        assert isinstance(pipeline_args, argparse.Namespace)
        assert pipeline_args.input_image_path == "s3://bucket/in"
        assert pipeline_args.output_path == str(tmp_path / "out")
        assert pipeline_args.limit == 1
        assert fake_stop_event.is_set()
        fake_thread.join.assert_called_once()


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

    def test_do_run_process_raises_logged_ray_job_error(self) -> None:
        """Ray CLI failures already have their diagnostics in the captured output."""
        mock_process = MagicMock()
        mock_process.stdout = io.StringIO("Ray job failed\n")
        mock_process.poll.return_value = RETURN_CODE_1
        mock_process.returncode = RETURN_CODE_1

        log_queue: queue.Queue[str] = queue.Queue()
        ipc_status = SimpleNamespace(value=True)

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = mock_process

            with pytest.raises(nvcf_main.RayJobLoggedError, match="Ray job failed with return code 1"):
                nvcf_main._do_run_process(["false"], log_queue, ipc_status)  # type: ignore[arg-type]

        assert log_queue.get_nowait() == "Ray job failed\n"
        assert ipc_status.value is False

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
            )

            # Verify executor was used
            mock_executor.submit.assert_called_once()
            assert not stop_event.is_set()


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
