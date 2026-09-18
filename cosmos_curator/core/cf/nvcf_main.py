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
"""Main entry point for the NVCF video pipeline."""

import argparse
import ctypes
import io
import json
import math
import multiprocessing
import os
import pathlib
import pickle
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import zipfile
from collections.abc import Callable, Mapping, MutableSequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from multiprocessing.sharedctypes import Synchronized
from queue import Empty
from typing import Any, cast

import aiofiles
import filelock
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from prometheus_client.parser import text_string_to_metric_families
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp

from cosmos_curator.core.cf.nvcf_utils import (
    create_s3_profile,
    is_nvcf_container_deployment,
    is_nvcf_helm_deployment,
    remove_s3_profile,
)
from cosmos_curator.core.utils.environment import CONTAINER_PATHS_CODE_DIR
from cosmos_curator.core.utils.misc.file_lock import file_lock
from cosmos_curator.core.utils.storage.presigned_s3_zip import (
    cleanup_server_input_workspace,
    cleanup_server_output_workspace,
    gather_and_upload_outputs,
    handle_presigned_urls,
    validate_local_input_paths,
)
from cosmos_curator.pipelines.image.annotate_pipeline import nvcf_run_annotate
from cosmos_curator.pipelines.video.dedup_pipeline import nvcf_run_semdedup
from cosmos_curator.pipelines.video.sharding_pipeline import nvcf_run_shard
from cosmos_curator.pipelines.video.splitting_pipeline import nvcf_run_split

_LOG_RDWR_LOCK_FILE = pathlib.Path(tempfile.gettempdir()) / "pipeline_status.lock"
_PIPELINE_LOCK_FILE = pathlib.Path(tempfile.gettempdir()) / "pipeline.lock"
_PIPELINE_STATUS_FILE = pathlib.Path(tempfile.gettempdir()) / "pipeline_status"
_FORCE_TERMINATE_REQUEST_ID = "12345678-1234-1234-1234-123456789abc"
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_REQUEST_OUTPUT_DIR_PATTERN = "request_output_dir_{req_id}.txt"
_DIRECT_REQUEST_PATTERN = "direct_request_{req_id}"
_CURATOR_DIRECT_MODE_HEADER = "CURATOR-DIRECT-MODE"
_CURATOR_STATUS_INCLUDE_LOGS_HEADER = "CURATOR-STATUS-INCLUDE-LOGS"
_RAY_DASHBOARD = f"http://127.0.0.1:{os.getenv('RAY_DASHBOARD_PORT', '8265')}"
_METRICS_PORT = 9002
_RAY_JOB_ENTRYPOINT_FAILURE_MESSAGE = "Ray job entrypoint failed"


def _ray_dashboard_headers() -> dict[str, str]:
    """Build the Authorization header for Ray dashboard API calls when token auth is enabled.

    Plain requests calls do not pick up RAY_AUTH_TOKEN_PATH (only Ray's own CLI/SDK
    do). Mirrors Ray's lookup precedence: the RAY_AUTH_TOKEN_PATH file first, then
    the RAY_AUTH_TOKEN env var. Returns an empty dict when no token is configured.
    """
    token = ""
    token_path = os.getenv("RAY_AUTH_TOKEN_PATH", "")
    if token_path:
        try:
            token = pathlib.Path(token_path).read_text(encoding="utf-8").strip()
        except OSError:
            logger.warning(f"RAY_AUTH_TOKEN_PATH is set but unreadable: {token_path}")
    if not token:
        token = os.getenv("RAY_AUTH_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


# This is evaluated at startup and used to decide if the logs/progress can be sent
# using get-request-status
using_nvcf_status: dict[str, bool] = {"get_req_sts": False}

# Only needed for debugging; reject for remote invokes.
_NVCF_REJECTED_ARG_NAMES = frozenset(
    {
        "stage_save",
        "stage_save_sample_rate",
        "stage_replay",
        "stage_compare",
        "stage_compare_path",
        "stage_compare_atol",
        "stage_compare_pass_threshold",
        "stage_compare_backend",
    },
)

# Compatibility options meant only for trusted local CLI invocations are kept separate from
# _NVCF_REJECTED_ARG_NAMES because that set tracks stage_replay.py's args exactly (see
# test_rejected_debug_pipeline_args_cover_every_stage_replay_flag).
_NVCF_REJECTED_SECURITY_ARG_NAMES = frozenset(
    {
        # Skips input-dataset containment checks for clip locations.
        "allow_unsafe_clip_location",
    },
)


def _validate_no_debug_pipeline_args(args: dict[str, Any]) -> None:
    """Reject NVCF invoke args that are only meant for trusted CLI use.

    Args:
        args: The raw ``args`` mapping from the invoke JSON payload, before it is
            spread into an ``argparse.Namespace``.

    Raises:
        ValueError: If any rejected debug/replay or security-escape-hatch arg name is present.

    """
    rejected = (_NVCF_REJECTED_ARG_NAMES | _NVCF_REJECTED_SECURITY_ARG_NAMES) & args.keys()
    if rejected:
        error_msg = f"Unsupported NVCF invoke args: {', '.join(sorted(rejected))}"
        raise ValueError(error_msg)


def _validate_no_internal_marker_args(args: dict[str, Any]) -> None:
    """Reject NVCF invoke args that collide with server-internal bookkeeping attrs.

    ``argparse.Namespace(**args)`` spreads the raw invoke JSON onto the Namespace
    with no allowlist, so any key the caller supplies becomes a real attribute --
    including ones this codebase later reads back and trusts as if only server code
    could have set them (e.g. ``_curator_server_output_workspace``, which gates
    whether a path gets zipped and uploaded to a caller-supplied URL). A caller who
    supplies that exact key can set both sides of that check to the same value and
    defeat it entirely.

    No legitimate CLI-derived arg name starts with ``_`` (argparse turns
    ``--foo-bar`` into ``foo_bar``, never a leading underscore), so rejecting the
    whole shape closes this off structurally instead of naming each internal
    marker attr as it's added.

    Args:
        args: The raw ``args`` mapping from the invoke JSON payload, before it is
            spread into an ``argparse.Namespace``.

    Raises:
        ValueError: If any key starts with an underscore.

    """
    rejected = {key for key in args if key.startswith("_")}
    if rejected:
        error_msg = f"Unsupported NVCF invoke args: {', '.join(sorted(rejected))}"
        raise ValueError(error_msg)


class RayJobLoggedError(RuntimeError):
    """Raised when a failed Ray CLI command already emitted captured diagnostics."""


def _cleanup_pipeline_lock_files() -> None:
    """Clean up stale pipeline lock files."""
    lock_files = [
        _PIPELINE_STATUS_FILE,
        _PIPELINE_LOCK_FILE,
        _LOG_RDWR_LOCK_FILE,
    ]
    for lock_file in lock_files:
        try:
            if lock_file.exists():
                lock_file.unlink()
                logger.debug(f"Cleaned up {lock_file.name}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to clean up {lock_file.name}: {e}")


def _value_error(msg: str) -> None:
    raise ValueError(msg)


def _validate_request_id(req_id: str) -> str:
    if not _REQUEST_ID_PATTERN.fullmatch(req_id):
        error_msg = "request_id must contain only letters, numbers, dots, underscores, or hyphens"
        raise ValueError(error_msg)
    return req_id


def _header_bool(headers: Mapping[str, str], name: str, *, default: bool) -> bool:
    """Parse common boolean header values, falling back to default when absent."""
    raw_value = headers.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


def _env_bool(name: str) -> bool | None:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return None
    normalized = raw_value.strip().lower()
    if normalized == "":
        return None
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning(f"Ignoring invalid boolean env var {name}={raw_value!r}")
    return None


def _env_int(name: str, *, minimum: int | None = None) -> int | None:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return None
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning(f"Ignoring invalid integer env var {name}={raw_value!r}")
        return None
    if minimum is not None and value < minimum:
        logger.warning(f"Ignoring out-of-range integer env var {name}={raw_value!r}")
        return None
    return value


def _env_float(name: str, *, minimum: float | None = None, maximum: float | None = None) -> float | None:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return None
    try:
        value = float(raw_value)
    except ValueError:
        logger.warning(f"Ignoring invalid float env var {name}={raw_value!r}")
        return None
    if (
        not math.isfinite(value)
        or (minimum is not None and value < minimum)
        or (maximum is not None and value > maximum)
    ):
        logger.warning(f"Ignoring out-of-range float env var {name}={raw_value!r}")
        return None
    return value


def _env_json_str_map(name: str) -> dict[str, str] | None:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return None
    if not raw_value:
        return {}
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        logger.warning(f"Ignoring invalid JSON env var {name}")
        return None
    if not isinstance(parsed, dict):
        logger.warning(f"Ignoring non-object JSON env var {name}")
        return None
    empty_value_keys = sorted(str(k) for k, v in parsed.items() if v in (None, ""))
    if empty_value_keys:
        logger.warning(f"Ignoring run attributes with invalid values from {name}: {', '.join(empty_value_keys)}")
    return {str(k): str(v) for k, v in parsed.items() if isinstance(k, str) and k and v not in (None, "")}


def _set_env_default(args: argparse.Namespace, attr_name: str, value_fn: Callable[[], object | None]) -> None:
    if hasattr(args, attr_name):
        return
    value = value_fn()
    if value is not None:
        setattr(args, attr_name, value)


def _apply_observability_env_defaults(args: argparse.Namespace) -> None:
    """Apply chart defaults without overriding invoke args and mark the result normalized.

    ``profiling._apply_profiling_config`` uses the marker to distinguish an
    explicit NVCF ``profile_tracing=False`` from an ordinary CLI parser default.
    """
    env_otlp_metrics_push = "COSMOS_CURATOR_OTLP_METRICS_PUSH"
    env_otlp_metrics_push_interval = "COSMOS_CURATOR_OTLP_METRICS_PUSH_INTERVAL"
    env_profile_tracing = "COSMOS_CURATOR_PROFILE_TRACING"
    env_profile_tracing_sampling = "COSMOS_CURATOR_PROFILE_TRACING_SAMPLING"
    env_otlp_run_attributes_values = "COSMOS_CURATOR_OTLP_RUN_ATTRIBUTES_VALUES"

    _set_env_default(args, "otlp_metrics_push", lambda: _env_bool(env_otlp_metrics_push))
    _set_env_default(
        args,
        "otlp_metrics_push_interval",
        lambda: _env_int(env_otlp_metrics_push_interval, minimum=1),
    )
    _set_env_default(args, "profile_tracing", lambda: _env_bool(env_profile_tracing))
    _set_env_default(
        args,
        "profile_tracing_sampling",
        lambda: _env_float(env_profile_tracing_sampling, minimum=0.0, maximum=1.0),
    )
    _set_env_default(args, "otlp_run_attributes_map", lambda: _env_json_str_map(env_otlp_run_attributes_values))
    args.observability_env_defaults_applied = True


def _setup_request(
    output_dir: str,
    log_queue: multiprocessing.Queue,  # type: ignore[type-arg]
    ipc_status: Synchronized,  # type: ignore[type-arg]
    req_id: str,
    logs: list[str] | MutableSequence[Any],
) -> tuple[threading.Thread, threading.Event]:
    # Assume success at start
    ipc_status.value = True

    progress_file = _get_progress_file(req_id)
    log_file = _get_log_file(req_id)
    with progress_file.open("w") as fp:
        data = {"progress": 0.0}
        json.dump(data, fp)
    pathlib.Path(log_file).touch()
    stop_event = threading.Event()
    stop_event.clear()
    progress_thread = threading.Thread(
        target=update_progress,
        args=(output_dir, stop_event, log_queue, ipc_status, req_id, logs),
        daemon=True,
    )
    progress_thread.start()
    return progress_thread, stop_event


def _get_request_status(req_id: str) -> str:
    is_found: bool = False
    is_done: bool = False
    is_failed: bool = False
    with file_lock(_LOG_RDWR_LOCK_FILE):
        is_found = _get_progress_file(req_id).exists()
        is_done = _get_done_file(req_id).exists()
        is_failed = _get_failed_file(req_id).exists()
    # Order of Precedence
    # failed
    # done
    # found
    if is_failed:
        return "failed"
    if is_done:
        return "done"
    if is_found:
        return "running"
    # Should not happen ever
    return "not-found"


def _get_progress_file(req_id: str) -> pathlib.Path:
    return pathlib.Path(tempfile.gettempdir()) / f"progress_{_validate_request_id(req_id)}.json"


def _get_request_output_dir_file(req_id: str) -> pathlib.Path:
    return pathlib.Path(tempfile.gettempdir()) / _REQUEST_OUTPUT_DIR_PATTERN.format(req_id=_validate_request_id(req_id))


def _get_direct_request_file(req_id: str) -> pathlib.Path:
    return pathlib.Path(tempfile.gettempdir()) / _DIRECT_REQUEST_PATTERN.format(req_id=_validate_request_id(req_id))


def _mark_direct_request(req_id: str) -> None:
    _get_direct_request_file(req_id).touch()


def _is_direct_request(req_id: str) -> bool:
    return _get_direct_request_file(req_id).exists()


def _write_request_output_dir(req_id: str, output_dir: str) -> None:
    _get_request_output_dir_file(req_id).write_text(output_dir, encoding="utf-8")


def _read_request_output_dir(req_id: str) -> str | None:
    output_dir_file = _get_request_output_dir_file(req_id)
    if not output_dir_file.exists():
        return None
    output_dir = output_dir_file.read_text(encoding="utf-8").strip()
    return output_dir or None


def _get_log_file(req_id: str) -> pathlib.Path:
    return pathlib.Path(tempfile.gettempdir()) / f"logs_{_validate_request_id(req_id)}.txt"


def _get_done_file(req_id: str) -> pathlib.Path:
    return pathlib.Path(tempfile.gettempdir()) / f"done_{_validate_request_id(req_id)}.txt"


def _get_failed_file(req_id: str) -> pathlib.Path:
    return pathlib.Path(tempfile.gettempdir()) / f"failed_{_validate_request_id(req_id)}.txt"


# returns a dict (job_id/submission_id, (type, status, message))
def _list_all_jobs() -> dict[str, tuple[str, str, str]]:
    jlist: dict[str, tuple[str, str, str]] = {}
    tout: int = 600
    try:
        resp = requests.get(f"{_RAY_DASHBOARD}/api/jobs/", timeout=tout, headers=_ray_dashboard_headers())
        jobs = resp.json()
        id_: str = ""
        type_: str = ""
        status: str = ""
        msg: str = ""
        for job in jobs:
            type_ = job.get("type", "")
            id_ = job.get("submission_id", "") if type_ == "SUBMISSION" else job.get("job_id", "")
            status = job.get("status", "")
            msg = job.get("message", "")
            jlist[id_] = (type_, status, msg)
    except Exception as e:
        error_msg = "Failed to get job status"
        raise ValueError(error_msg) from e
    return jlist


def _read_nvcf_status_progress_file(req_id: str) -> tuple[float, str] | None:
    output_dir = _read_request_output_dir(req_id)
    if output_dir is None:
        return None
    progress_file = pathlib.Path(output_dir) / "progress"
    if not progress_file.exists():
        return None
    with progress_file.open() as fp:
        progress_data = json.load(fp)
    partial_response = progress_data.get("partialResponse", {})
    progress_pct = float(partial_response.get("progress", 0.0))
    log_lines = str(partial_response.get("logs", ""))
    return progress_pct, log_lines


def _read_progress_and_log_files(
    req_id: str | None,
    *,
    read_progress: bool = True,
    read_log: bool = True,
) -> tuple[float, str]:
    log_lines = ""
    # Initialize to a sane default to avoid UnboundLocalError if req_id is None
    progress_pct: float = 0.0
    if req_id is not None:
        nvcf_status_progress = _read_nvcf_status_progress_file(req_id) if using_nvcf_status["get_req_sts"] else None
        if nvcf_status_progress is not None:
            progress_pct, log_lines = nvcf_status_progress
        else:
            # avoid race condition
            with file_lock(_LOG_RDWR_LOCK_FILE):
                # read progress
                if read_progress and _get_progress_file(req_id).exists():
                    with _get_progress_file(req_id).open() as fp:
                        progress_pct = json.load(fp).get("progress", 0)
                # read log and create a zip file in the buffer
                if read_log and _get_log_file(req_id).exists():
                    with _get_log_file(req_id).open() as fp:
                        log_lines = fp.read()
    else:
        log_lines = "_read_progress_and_log_files called without req_id"
    if read_log:
        try:
            jobs = _list_all_jobs()
            # `(job_id/submission_id, (type, status, message))`
            str_jobs = list(jobs.items())
            log_lines += f"================Jobs Status================\n{str_jobs}\n"
            log_lines += "===========================================\n"
        except Exception as e:  # noqa: BLE001
            log_lines += str(e)
    return progress_pct, log_lines


def _wait_for_stop(job_id: str) -> bool:
    count = 0
    retry_cnt: int = 5
    tout: int = 600
    http_ok: int = 200
    while count < retry_cnt:
        resp = requests.get(f"{_RAY_DASHBOARD}/api/jobs/{job_id}", timeout=tout, headers=_ray_dashboard_headers())
        if resp.status_code == http_ok:
            status = resp.json()
            logger.debug(f"{job_id} : stop status : {resp.text}")
            if status.get("status", "") != "RUNNING":
                # Job stopped
                return True

            count += 1
            logger.debug(f"{job_id} : still RUNNING : attempt {count}")
            time.sleep(10)

        else:
            logger.debug(f"{job_id} : could not get job status {resp.text}")
            return False

    logger.debug(f"{job_id} : did not stop")
    return False


def _stop_ray_job(job_id: str) -> tuple[bool, str]:
    count = 0
    retry_cnt: int = 5
    tout: int = 600
    http_ok: int = 200
    while count < retry_cnt:
        resp = requests.post(f"{_RAY_DASHBOARD}/api/jobs/{job_id}/stop", timeout=tout, headers=_ray_dashboard_headers())
        if resp.status_code == http_ok:
            # ideal resp : {"stopped": true}, this is async
            # hence we need to wait for status to change to STOPPED
            status = _wait_for_stop(job_id)
            if status:
                return (status, resp.text)

            # Should not happen
            count += 1
            logger.debug(f"{job_id} : Retry stop : {count}")
            time.sleep(10)
        else:
            logger.debug(f"{job_id} : Failed to stop : {resp.text}")
            return (False, "Failed to stop")

    logger.debug(f"{job_id} : Failed to stop after 5 tries")
    return (False, "Failed to stop")


def _terminate_last_job(request_id: str) -> tuple[bool, str]:
    """Terminate the currently running job.

    The NVCF request_id passed to this function is merely
    used for record keeping and checking if the job is actually
    running by using states maintained using file. There is no
    direct relationship between NVCF request-id and ray-job-id
    There are two types of ray jobs - DRIVER and SUBMISSION.
    Currently ray api cannot terminate DRIVER jobs. Hence to make
    this function work, we need to start the ray pipeline using
    `ray job submit`. See _run_in_process for additional info
    Alternative approach can be to use ray.cancel but would require
    keeping list of remote object references accessible from this
    module. There are two steps in terminating a ray job. First
    it must be stopped, then once the state is set to 'stopped',
    it should be deleted to release all resources held.
    """
    success: bool = False
    tout: int = 600
    http_ok: int = 200

    if request_id != _FORCE_TERMINATE_REQUEST_ID:
        status = _get_request_status(request_id)
        logger.info(f"{request_id} : status is : {status}")
        if status != "running":
            _value_error(f"{request_id} status is {status}")
    else:
        logger.info(f"got special req_id {request_id}, proceeding to terminate last job")

    # List all running ray jobs for the currently running
    # NVCF request
    jobs = _list_all_jobs()
    # `(job_id/submission_id, (type, status, message))`
    msg: str = ""
    ctr: int = 0
    # iterate over the jobs and attempt to terminate
    # RUNNING jobs
    logger.debug(f"{request_id} has the following jobs : {list(jobs.items())}")
    for k, v in jobs.items():
        status = v[1]
        job_type = v[0]

        # ray only supports terminating SUBMISSION jobs
        if status == "RUNNING":
            if job_type == "SUBMISSION":
                # attempt to stop first
                state, m = _stop_ray_job(k)
                logger.debug(f"{request_id} : {k} Stopped : {m}")
                if state:
                    # then delete
                    resp = requests.delete(
                        f"{_RAY_DASHBOARD}/api/jobs/{k}", timeout=tout, headers=_ray_dashboard_headers()
                    )
                    # ideal resp : {"deleted": true}, this is async
                    logger.debug(f"{request_id} : {k} Deleted : {resp.text}")
                    if resp.status_code != http_ok:
                        msg += f"{k} : {resp.text}\n"
                        logger.debug(f"{k} : stopped but no cleanup : {resp.text}")
                        continue

                    msg += f"{k} : Job was successfully deleted\n"
                    logger.debug(f"{k} : Job was successfully deleted")
                    ctr += 1

                else:
                    msg += m
            else:
                msg += f"{k} : Cannot terminate DRIVER type job\n"
                logger.debug(f"{k} : Cannot terminate DRIVER type job")

    # mark it as failed ray job
    if request_id is not None and ctr > 0:
        with file_lock(_LOG_RDWR_LOCK_FILE):
            # this is the only way to notify the update_prgress thread
            pathlib.Path(_get_failed_file(request_id)).touch()
            logger.debug(f"Recorded termination request for req_id : {request_id}")
        success = True
    return (success, msg)


def _write_progress_and_log_files(
    *,
    req_id: str,
    buffer: list[str],
    write_progress: bool = True,
    write_log: bool = True,
) -> None:
    with file_lock(_LOG_RDWR_LOCK_FILE):
        # write progress to file
        if write_progress:
            with _get_progress_file(req_id).open("w") as fp:
                data = {"progress": get_pipeline_progress()}
                json.dump(data, fp)
        # append to log file
        if write_log:
            with _get_log_file(req_id).open("a") as fp:
                fp.writelines(buffer)


class PipelineLockMiddleware(BaseHTTPMiddleware):
    """Middleware to handle pipeline requests."""

    def __init__(self, app: ASGIApp) -> None:
        """Initialize the pipeline lock middleware.

        Args:
            app: The ASGI application to wrap.

        """
        super().__init__(app)
        # Create a file lock in temp directory that's shared between workers
        self._lock_file: str = str(_PIPELINE_LOCK_FILE)
        self._file_lock = filelock.FileLock(self._lock_file)
        self._status_file: pathlib.Path = _PIPELINE_STATUS_FILE

    def _is_pipeline_busy(self) -> bool:
        try:
            return self._status_file.exists()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error checking pipeline status: {e}")
            return False

    @staticmethod
    def _pipeline_busy_response() -> JSONResponse:
        logger.warning(
            "Pipeline is busy (another worker is processing), rejecting request",
        )
        return JSONResponse(
            status_code=429,
            content={
                "error": "Pipeline is currently busy. Please try again later.",
            },
        )

    @staticmethod
    def _create_zip_file(
        req_id: str,
        progress_pct: float,
        log_lines: str,
    ) -> io.BytesIO:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zp:
            zp.writestr(
                f"progress-{req_id}.json",
                json.dumps({"progress": progress_pct}),
            )
            zp.writestr(f"log-{req_id}.txt", log_lines)
        buffer.seek(0)
        return buffer

    @staticmethod
    def _validate_curator_request_id(
        req_id: str | None, missing_message: str
    ) -> tuple[str | None, JSONResponse | None]:
        if req_id is None:
            return None, JSONResponse(
                status_code=400,
                content={"error": missing_message},
                media_type="application/json",
            )
        try:
            return _validate_request_id(req_id), None
        except ValueError as e:
            return None, JSONResponse(status_code=400, content={"error": str(e)})

    def _handle_termination_request(self, request: Request) -> Response:
        req_id, error_response = self._validate_curator_request_id(
            request.headers.get("CURATOR-NVCF-REQID"),
            "Missing request id",
        )
        if error_response is not None:
            return error_response
        assert req_id is not None

        logger.warning(f"Termination request for {req_id} received")
        code = 200
        status = "status"
        try:
            success, msg = _terminate_last_job(req_id)
            if success:
                msg = f"{req_id} terminated successfully"
            else:
                msg = f"{req_id} could not be terminated"
                code = 412
                status = "error"
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            code = 412
            status = "error"
        logger.info(f"Termination state of {req_id} : {msg}")
        hdrs = {
            "CURATOR-TERM-STATUS": msg,
            "ACCESS-CONTROL-EXPOSE-HEADERS": "CURATOR-TERM-STATUS",
        }
        return JSONResponse(
            status_code=code,
            content={status: msg},
            headers=hdrs,
        )

    def _handle_status_check(self, request: Request) -> Response:
        req_id, error_response = self._validate_curator_request_id(
            request.headers.get("CURATOR-NVCF-REQID"),
            "Missing request ID in headers",
        )
        if error_response is not None:
            return error_response
        assert req_id is not None

        logger.info(f"Received status check for request {req_id}")
        include_logs = _header_bool(request.headers, _CURATOR_STATUS_INCLUDE_LOGS_HEADER, default=True)
        progress_pct, log_lines = _read_progress_and_log_files(req_id, read_log=include_logs)
        status_str = _get_request_status(req_id)

        logger.debug(
            f"Progress for request {req_id}: {status_str=} {progress_pct}",
        )
        headers = {
            "CURATOR-PIPELINE-STATUS": status_str,
            "CURATOR-PIPELINE-PERCENT-COMPLETE": f"{progress_pct:.2f}",
            "access-control-expose-headers": ("CURATOR-PIPELINE-STATUS, CURATOR-PIPELINE-PERCENT-COMPLETE"),
        }
        if not include_logs:
            return Response(status_code=200, headers=headers)

        buffer = self._create_zip_file(req_id, progress_pct, log_lines)
        zresponse = StreamingResponse(buffer, media_type="application/zip")
        zresponse.headers["Content-Disposition"] = "attachment; filename=files.zip"
        zresponse.headers.update(headers)
        return zresponse

    async def dispatch(  # noqa: PLR0911
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response | StreamingResponse:
        """Process incoming requests and handle pipeline locking.

        Args:
            request: The incoming request.
            call_next: The next middleware in the chain.

        Returns:
            The response from the next middleware or a custom response.

        """
        if request.url.path == "/v1/run_pipeline":
            if request.headers.get("CURATOR-REQ-TERMINATE", ""):
                return self._handle_termination_request(request)
            # status check request
            if request.headers.get("CURATOR-STATUS-CHECK", ""):
                return self._handle_status_check(request)

            # real pipeline request
            if self._is_pipeline_busy():
                return self._pipeline_busy_response()

            try:
                # Try to acquire lock with timeout
                with self._file_lock.acquire(timeout=1):
                    cleanup_in_background = False
                    try:
                        if self._is_pipeline_busy():
                            return self._pipeline_busy_response()

                        # Create status file to indicate pipeline is running
                        async with aiofiles.open(self._status_file.resolve(), "w", encoding="utf-8") as f:
                            await f.write("busy")

                        logger.info("Pipeline started - acquired lock")
                        response = await call_next(request)
                        if response.headers.get("CURATOR-INVOCATION-MODE") == "direct":
                            cleanup_in_background = True
                            logger.info("Direct pipeline accepted - keeping lock files for background run")
                            return response
                        return response
                    finally:
                        if not cleanup_in_background:
                            # Clean up lock files
                            _cleanup_pipeline_lock_files()
                            logger.info("Pipeline completed - released lock")
            except filelock.Timeout:
                logger.warning("Could not acquire lock, pipeline appears to be busy")
                return self._pipeline_busy_response()

        return await call_next(request)


def setup_pipeline_middleware(app: FastAPI) -> None:
    """Set up the pipeline middleware."""
    # NVCF_REQUEST_STATUS indicates if this container will support using
    # NVCF get-request-status. If this flag is not present, is assumed true
    # when NVCF_SINGLE_NODE is true
    if not is_nvcf_helm_deployment():
        is_single_node: bool = is_nvcf_container_deployment()
        is_nvcf_status: bool = os.environ.get("NVCF_REQUEST_STATUS", "true").lower() == "true"
        using_nvcf_status["get_req_sts"] = is_single_node and is_nvcf_status

    logger.info(f"Starting up using_nvcf_status = {using_nvcf_status['get_req_sts']}")

    # Clean up stale lock files from previous crashed instances
    logger.info("Cleaning up stale pipeline lock files on startup")
    _cleanup_pipeline_lock_files()

    app.add_middleware(PipelineLockMiddleware)


app = FastAPI()
setup_pipeline_middleware(app)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint to check if the service is running."""
    return {"status": "healthy"}


@app.get("/v1/logs")
def get_logs(request_id: str) -> JSONResponse:
    """Get logs for a specific request ID.

    Args:
        request_id: The request ID to get logs for

    Returns:
        JSON object with logs and status

    """
    if not request_id:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing request_id parameter"},
        )
    try:
        request_id = _validate_request_id(request_id)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    try:
        progress_pct, log_lines = _read_progress_and_log_files(request_id)
        status_str = _get_request_status(request_id)

        return JSONResponse(
            status_code=200,
            content={"progress": progress_pct, "status": status_str, "logs": log_lines},
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error retrieving logs for request {request_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Failed to retrieve logs for request {request_id}: {e!s}",
            },
        )


@app.get("/v1/progress")
def get_progress(request_id: str) -> JSONResponse:
    """Get progress percentage for a specific request ID.

    Args:
        request_id: The request ID to get progress for

    Returns:
        JSON object with progress percentage and status

    """
    if not request_id:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing request_id parameter"},
        )
    try:
        request_id = _validate_request_id(request_id)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    try:
        progress_pct, _ = _read_progress_and_log_files(request_id, read_log=False)
        status_str = _get_request_status(request_id)

        return JSONResponse(
            status_code=200,
            content={"progress": progress_pct, "status": status_str},
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error retrieving progress for request {request_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Failed to retrieve progress for request {request_id}: {e!s}",
            },
        )


def get_nvcf_output_path(request: Request) -> str:
    """Get the NVCF large output directory, or a local fallback.

    NVCF-LARGE-OUTPUT-DIR started as the path to a linked asset output
    directory. Keep honoring it for compatibility: it is still used for status
    files and may also be used by callers as a cache/work directory without
    relying on asset linkage.
    """
    output_dir = request.headers.get("NVCF-LARGE-OUTPUT-DIR")

    if output_dir is None or not pathlib.Path(output_dir).exists():
        logger.warning(f"NVCF-LARGE-OUTPUT-DIR {output_dir} does not exist")
        o_dir = pathlib.Path(tempfile.gettempdir()) / "nvcf_output"
        o_dir.mkdir(parents=True, exist_ok=True)
        output_dir = str(o_dir)
    return output_dir


# Curator endpoint to run the video pipeline script
@app.post("/v1/run_pipeline")
async def curate_video(request: Request) -> JSONResponse:  # noqa: C901, PLR0912, PLR0915
    """Curator endpoint to run the video pipeline script."""
    did_init_s3_profile = False
    manager = None
    log_queue = None
    stop_event = None
    progress_thread = None
    nvcf_output_dir = None
    pipeline_args = None
    should_cleanup = True
    request_id: str | None = None
    logs: list[str] | MutableSequence[Any] = []
    # mypy is confused about the type
    ipc_status: Any = None

    try:
        nvcf_output_dir = get_nvcf_output_path(request)
        manager = Manager()

        ipc_status = manager.Value(ctypes.c_bool, value=False)

        log_queue = cast("multiprocessing.Queue", manager.Queue())  # type: ignore[type-arg]
        logs = manager.list() if using_nvcf_status["get_req_sts"] else ["success"]

        nvcf_ncaid = request.headers.get("NVCF-NCAID")
        nvcf_subid = request.headers.get("NVCF-SUBID")
        nvcf_env = request.headers.get("NVCF-ENV")
        logger.info(f"NVCF-NCAID: {nvcf_ncaid}")
        logger.info(f"NVCF-SUBID: {nvcf_subid}")
        logger.info(f"NVCF-ENV: {nvcf_env}")
        logger.info(f"NVCF-LARGE-OUTPUT-DIR: {nvcf_output_dir}")
        logger.info(f"NVCF-Curator node={socket.gethostname()} pid={os.getpid()}")

        request_id = request.headers.get("NVCF-REQID")
        is_direct_invocation = request.headers.get(_CURATOR_DIRECT_MODE_HEADER, "").lower() == "true"
        logger.info(
            f"NVCF request classification: has_nvcf_reqid={request_id is not None}, "
            f"is_direct_invocation={is_direct_invocation}",
        )
        if request_id is None:
            logger.warning("NVCF-REQID is missing, generating fake request-id")
            request_id = str(uuid.uuid4())  # Generate a fake request-id
        try:
            request_id = _validate_request_id(request_id)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

        logger.info(f"Request ID: {request_id}")
        output_dir = nvcf_output_dir if using_nvcf_status["get_req_sts"] else None
        if output_dir is None:
            output_dir = tempfile.gettempdir()  # do not leave as None
        _write_request_output_dir(request_id, output_dir)

        progress_thread, stop_event = _setup_request(
            output_dir,
            log_queue,
            ipc_status,
            request_id,
            logs,
        )

        invoke_args = await request.json()
        pipeline_type = invoke_args.get("pipeline", "unknown")
        raw_args = invoke_args.get("args", {})
        try:
            _validate_no_debug_pipeline_args(raw_args)
            _validate_no_internal_marker_args(raw_args)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        pipeline_args = argparse.Namespace(**raw_args)
        _apply_observability_env_defaults(pipeline_args)

        def prepare_and_run_pipeline() -> None:  # noqa: C901, PLR0912
            nonlocal did_init_s3_profile, pipeline_args

            # Handle any presigned URL processing and ensure we now have a concrete Namespace.
            # For direct invocation this must happen in the background thread, after the
            # request id has been returned to the caller.
            assert isinstance(pipeline_args, argparse.Namespace)
            pipeline_args = handle_presigned_urls(pipeline_type, pipeline_args)

            # At this point `pipeline_args` **must** be a populated Namespace object.
            # Add an explicit runtime assertion so static type-checkers understand this.
            assert isinstance(pipeline_args, argparse.Namespace)

            validate_local_input_paths(pipeline_args)

            if hasattr(pipeline_args, "s3_config"):
                did_init_s3_profile = create_s3_profile(pipeline_args.s3_config)
                # set it to redacted so that it cannot be saved in script file
                # the pipeline script dont need it once the profile is created
                pipeline_args.s3_config = "REDACTED"

            logger.info(f"Launching pipeline {pipeline_type} with args: {pipeline_args}")
            pipeline_func: Callable[[argparse.Namespace], None] | None = None

            if pipeline_type == "split":
                if not getattr(pipeline_args, "output_clip_path", None):
                    pipeline_args.output_clip_path = get_nvcf_output_path(request)

                # Validate that we have either a direct path or a presigned URL for both input and output
                missing_input_path = not getattr(pipeline_args, "input_video_path", None)
                missing_input_url = not getattr(pipeline_args, "input_presigned_s3_url", None)

                missing_output_path = not getattr(pipeline_args, "output_clip_path", None)
                missing_output_url = not getattr(pipeline_args, "output_presigned_s3_url", None)

                if missing_input_path and missing_input_url:
                    _value_error(
                        "Invalid Pipeline args: Either input_video_path or input_presigned_s3_url must be provided",
                    )

                if missing_output_path and missing_output_url:
                    _value_error(
                        "Invalid Pipeline args: Either output_clip_path or output_presigned_s3_url must be provided",
                    )

                pipeline_func = nvcf_run_split
            elif pipeline_type == "shard":
                pipeline_func = nvcf_run_shard
            elif pipeline_type == "annotate":
                if not getattr(pipeline_args, "output_path", None):
                    pipeline_args.output_path = get_nvcf_output_path(request)

                if not getattr(pipeline_args, "input_image_path", None):
                    _value_error("Invalid Pipeline args: input_image_path must be provided")
                if not getattr(pipeline_args, "output_path", None):
                    _value_error("Invalid Pipeline args: output_path must be provided")

                pipeline_func = nvcf_run_annotate
            elif pipeline_type in ("image-caption", "image-embed", "av-split"):
                pass
            elif pipeline_type == "semantic-dedup":
                pipeline_func = nvcf_run_semdedup
            else:
                _value_error(f"Invalid Pipeline type {pipeline_type}")

            if pipeline_func is None:
                return
            execute_pipeline(
                _run_in_process,
                pipeline_func,
                request_id,
                pipeline_args,
                log_queue,
                ipc_status,
            )

        if is_direct_invocation:
            should_cleanup = False
            _mark_direct_request(request_id)

            def run_direct_pipeline() -> None:
                try:
                    logger.info(f"Background direct pipeline starting for request {request_id}")
                    prepare_and_run_pipeline()
                except RayJobLoggedError:
                    logger.error(f"Pipeline failed for request {request_id}; details in Ray job log")
                    ipc_status.value = False
                except Exception as e:  # noqa: BLE001
                    logger.exception(f"Error in background pipeline for request {request_id}: {e}")
                    ipc_status.value = False
                finally:
                    logger.info("Cleaning up after finishing the invoke")
                    if pipeline_args is not None:
                        if ipc_status.value:
                            try:
                                gather_and_upload_outputs(pipeline_type, pipeline_args)
                            except Exception as e:  # noqa: BLE001
                                logger.exception(f"Error uploading pipeline outputs for request {request_id}: {e}")
                                ipc_status.value = False
                        else:
                            # The run failed before ever reaching gather_and_upload_outputs,
                            # so its usual cleanup never ran; remove any server-created
                            # output temp dir directly instead of leaving it behind.
                            cleanup_server_output_workspace(pipeline_type, pipeline_args)
                        # Nothing reads the presigned-input extract dir after the
                        # pipeline function has returned, win or lose -- unlike output,
                        # this isn't gated on success.
                        cleanup_server_input_workspace(pipeline_args)
                    if stop_event and not stop_event.is_set():
                        stop_event.set()
                    if progress_thread:
                        progress_thread.join()
                    if did_init_s3_profile:
                        remove_s3_profile()
                    if manager:
                        manager.shutdown()
                    _cleanup_pipeline_lock_files()
                    logger.info("Background pipeline completed - released lock")

            threading.Thread(target=run_direct_pipeline, daemon=True).start()
            return JSONResponse(
                status_code=200,
                content={"message": "Pipeline accepted", "reqid": request_id, "status": "in-progress"},
                headers={
                    "CURATOR-PIPELINE-STATUS": "running",
                    "CURATOR-PIPELINE-PERCENT-COMPLETE": "0.00",
                    "CURATOR-INVOCATION-MODE": "direct",
                    "access-control-expose-headers": "CURATOR-PIPELINE-STATUS, CURATOR-PIPELINE-PERCENT-COMPLETE",
                },
            )

        prepare_and_run_pipeline()

        return JSONResponse(
            status_code=200,
            content={"message": "Pipeline executed successfully", "logs": "".join(logs)},
        )

    except Exception as e:  # noqa: BLE001
        # A failure here means the pipeline did not produce output; do not let the
        # optimistic default in ipc_status trigger an upload in the finally block.
        if ipc_status is not None:
            ipc_status.value = False
        ray_job_logged_error = isinstance(e, RayJobLoggedError)
        if ray_job_logged_error:
            logger.error(f"Pipeline failed for request {request_id}; details in Ray job log")
        else:
            logger.error(f"Received Exception, waiting for progress thread to finish: {e}")

        if progress_thread and stop_event:
            if not stop_event.is_set():
                stop_event.set()
            progress_thread.join()
            progress_thread = None

        if using_nvcf_status["get_req_sts"]:
            log_lines = "".join(logs)
        else:
            _, log_lines = _read_progress_and_log_files(request_id, read_progress=False)
        log_str = "failed" if log_lines is None or len(log_lines) == 0 else log_lines
        error_dict = {
            "exception": str(e),
            "exception_type": type(e).__name__,
        }
        if not ray_job_logged_error:
            error_dict["traceback"] = traceback.format_exc()
        error_dict["logs"] = log_str
        # flatten it, json.dumps is bad, NVCF adds its own encoding, too many \\
        error_details = "\n".join([f"{k}: {v}" for k, v in error_dict.items()])
        if not ray_job_logged_error:
            logger.error(f"Error in pipeline: {error_details}")
        return JSONResponse(status_code=500, content={"error": error_details})
    finally:
        if should_cleanup:
            logger.info("Cleaning up after finishing the invoke")
            if pipeline_args is not None:
                if ipc_status is not None and ipc_status.value:
                    try:
                        gather_and_upload_outputs(pipeline_type, pipeline_args)
                    except Exception as e:  # noqa: BLE001
                        logger.exception(f"Error uploading pipeline outputs for request {request_id}: {e}")
                        ipc_status.value = False
                else:
                    # The run failed before ever reaching gather_and_upload_outputs, so
                    # its usual cleanup never ran; remove any server-created output temp
                    # dir directly instead of leaving it behind.
                    cleanup_server_output_workspace(pipeline_type, pipeline_args)
                # Nothing reads the presigned-input extract dir after the pipeline
                # function has returned, win or lose -- unlike output, this isn't
                # gated on success.
                cleanup_server_input_workspace(pipeline_args)
            if stop_event and not stop_event.is_set():
                stop_event.set()
            if progress_thread:
                progress_thread.join()
            if did_init_s3_profile:
                remove_s3_profile()
            if manager:
                manager.shutdown()


def _run_in_process(
    func: Callable[..., None],
    request_id: str,
    pipeline_args: argparse.Namespace,
    log_queue: multiprocessing.Queue,  # type: ignore[type-arg]
    ipc_status: Synchronized,  # type: ignore[type-arg]
) -> None:
    """Run the function in a process, capturing its output.

    We need to use `ray job submit` semantics to allow the
    type of job to be SUBMISSION (instead of DRIVER) to allow
    for the job to be terminated. See _terminate_last_job
    for additional info
    """
    # Create the script dynamically
    validated_request_id = _validate_request_id(request_id)
    sfile = pathlib.Path(tempfile.gettempdir()) / f"{validated_request_id}.py"
    with sfile.open("w") as sf:
        sf.write(f"""
import sys
import pickle
import os

# Tag all pipeline logs with the NVCF request id before any cosmos import so that
# loguru/Ray structured logging (CURATOR_RUN_ID) carries it. setdefault keeps an
# externally provided value if one is already set.
os.environ.setdefault("CURATOR_RUN_ID", {validated_request_id!r})

# Set up sys.path
sys.path = pickle.loads({pickle.dumps(sys.path)!r})

from cosmos_xenna.utils import python_log

def run_func():
    mod = __import__('{func.__module__}', fromlist=['{func.__name__}'])
    func = getattr(mod, '{func.__name__}')
    pipeline_args = pickle.loads({pickle.dumps(pipeline_args)!r})
    func(pipeline_args)

try:
    run_func()
except Exception:
    python_log.exception({_RAY_JOB_ENTRYPOINT_FAILURE_MESSAGE!r})
    sys.exit(1)
""")
    # Create a subprocess that runs the function
    cmd = [
        "ray",
        "job",
        "submit",
        "--address",
        f"{_RAY_DASHBOARD}",
        "--",
        sys.executable,
        str(sfile),
    ]
    _do_run_process(cmd, log_queue, ipc_status)


def _do_run_process(
    cmd: list[str],
    log_queue: multiprocessing.Queue,  # type: ignore[type-arg]
    ipc_status: Synchronized,  # type: ignore[type-arg]
) -> None:
    process = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=CONTAINER_PATHS_CODE_DIR,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    # Read output line by line and send to queue
    while True and process is not None and process.stdout is not None:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line.strip():
            log_queue.put(line)
            print(line, end="", flush=True)  # noqa: T201 Print to stdout and flush immediately

    # Check if process failed
    if process.returncode != 0:
        # Failed
        ipc_status.value = False
        error_msg = f"Ray job failed with return code {process.returncode}"
        raise RayJobLoggedError(error_msg)


def execute_pipeline(  # noqa: PLR0913
    wrapper_func: Callable[..., None],
    func: Callable[..., None],
    request_id: str | None,
    pipeline_args: argparse.Namespace,
    log_queue: multiprocessing.Queue,  # type: ignore[type-arg]
    ipc_status: Synchronized,  # type: ignore[type-arg]
) -> None:
    """Run the video pipeline in a separate process and capture its output."""
    if request_id is None:
        request_id = str(uuid.uuid4())
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            wrapper_func,
            func,
            request_id,
            pipeline_args,
            log_queue,
            ipc_status,
        )
        _ = future.result()


def get_pipeline_progress() -> float:
    """Get the progress of the pipeline."""
    progress = 0.0
    tout: int = 600
    http_ok: int = 200
    response = requests.get(f"http://localhost:{_METRICS_PORT}/metrics", timeout=tout)
    if response.status_code == http_ok:
        for item in text_string_to_metric_families(response.text):
            if item.name == "ray_pipeline_progress":
                progress = item.samples[0].value
                break
    if isinstance(progress, float):
        return 100 * progress
    return 0  # type: ignore[unreachable]


def update_progress(  # noqa: C901, PLR0912, PLR0913, PLR0915
    output_dir: str,
    stop_event: threading.Event,
    log_queue: multiprocessing.Queue,  # type: ignore[type-arg]
    ipc_status: Synchronized,  # type: ignore[type-arg]
    request_id: str,
    logs: list[str],
) -> None:
    """Update the progress of the pipeline."""
    progress_file = pathlib.Path(output_dir) / "progress" if using_nvcf_status["get_req_sts"] else None
    buffer = []
    last_update = time.time()

    min_delta_time: int = 5
    while not stop_event.is_set():
        try:
            status = _get_request_status(request_id)
            if status != "running":
                # try to exit normally
                stop_event.set()
        except Exception:  # noqa: BLE001, S110
            # just be silent
            pass
        # Process all available messages
        try:
            while True and not stop_event.is_set():
                try:
                    status = _get_request_status(request_id)
                    if status != "running":
                        # try to exit normally
                        stop_event.set()
                except Exception:  # noqa: BLE001, S110
                    # just be silent
                    pass
                try:
                    line = log_queue.get_nowait()
                    if using_nvcf_status["get_req_sts"]:
                        logs.append(line)
                    buffer.append(line)
                except Empty:
                    break

            current_time = time.time()
            if current_time - last_update >= min_delta_time or stop_event.is_set():
                if using_nvcf_status["get_req_sts"]:
                    pct_complete = get_pipeline_progress()
                    output_data = {
                        "id": request_id,
                        "progress": 0,
                        "partialResponse": {
                            "logs": "".join(buffer) if buffer else "",
                            "progress": f"{pct_complete:.2f}",
                        },
                    }

                    # Write atomically using temporary file
                    assert progress_file is not None
                    temp_file = f"{progress_file!s}.{os.getpid()}.tmp"
                    ptf = pathlib.Path(temp_file)
                    try:
                        with ptf.open("w") as f:
                            json.dump(output_data, f)
                        ptf.replace(progress_file)
                    except Exception as e:  # noqa: BLE001
                        logger.error(f"Failed to write progress file: {e}")
                    finally:
                        if ptf.exists():
                            try:
                                ptf.unlink()
                            finally:
                                pass
                else:
                    _write_progress_and_log_files(req_id=request_id, buffer=buffer)
                buffer.clear()
                last_update = current_time

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error processing log queue: {e}")

        time.sleep(0.1)

    # Drain any remaining messages from the queue
    while True:
        try:
            line = log_queue.get_nowait()
            if using_nvcf_status["get_req_sts"]:
                logs.append(line)
            buffer.append(line)
        except Empty:
            break

    # Final update with any remaining logs
    if buffer:
        if using_nvcf_status["get_req_sts"]:
            pct_complete = get_pipeline_progress()
            output_data = {
                "id": request_id,
                "progress": 0,
                "partialResponse": {
                    "logs": "".join(buffer),
                    "progress": f"{pct_complete:.2f}",
                },
            }
            assert progress_file is not None
            temp_file = f"{progress_file}.{os.getpid()}.tmp"
            ptf = pathlib.Path(temp_file)
            try:
                with ptf.open("w") as f:
                    json.dump(output_data, f)
                ptf.replace(progress_file)
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to write final progress: {e}")
            finally:
                if ptf.exists():
                    try:
                        ptf.unlink()
                    finally:
                        pass
        else:
            _write_progress_and_log_files(req_id=request_id, buffer=buffer, write_progress=False)

    # Drop the done for failed file,
    # this records the request status
    # long after it is done
    with file_lock(_LOG_RDWR_LOCK_FILE):
        if not _get_failed_file(request_id).exists():
            if ipc_status.value:
                pathlib.Path(_get_done_file(request_id)).touch()
            else:
                pathlib.Path(_get_failed_file(request_id)).touch()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=2)  # noqa: S104
