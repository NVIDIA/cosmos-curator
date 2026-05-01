#!/usr/bin/env python3
"""Example script to operate X nvcf functions to run N input files."""

import argparse
import collections
import json
import random
import re
import subprocess
import time
import uuid
from enum import Enum
from pathlib import Path

import attrs
from loguru import logger

# USER MIGHT NEED TO CHANGE THIS
_S3_CREDENTIALS_PATH = Path("~/.aws/credentials").expanduser().as_posix()
_NUM_RETRIES = 5
_RETRY_INTERVAL = 120  # seconds

# For internal use
_NVCF_SPECIAL_REQ_ID = "12345678-1234-1234-1234-123456789abc"
_DRY_RUN_RESPONSE = {
    "invoke": [
        "ReqId: c8223660-f041-4023-9ba0-8b6bb86bdd71, Status: 'in-progress', Completed: None",
    ],
    "status": [
        "[15:29:15] INFO     ReqId: c8223660-f041-4023-9ba0-8b6bb86bdd71, Status: 'in-progress', Completed: None",
        "[15:29:15] INFO     ReqId: c8223660-f041-4023-9ba0-8b6bb86bdd71, Status: 'failed', Completed: 10.0%",
        "[15:29:15] INFO     ReqId: c8223660-f041-4023-9ba0-8b6bb86bdd71, Status: 'fulfilled', Completed: 100.0%",
    ],
    "terminate": [
        "12345678-1234-1234-1234-123456789abc terminated successfully",
    ],
}


def _run_command(command: list[str], cmd_type: str, *, dry_run: bool) -> tuple[int, str | None]:
    logger.debug(f"Running command: {' '.join(command)}")
    if not dry_run:
        try:
            proc = subprocess.Popen(  # noqa: S603
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=False,
            )
            stdout, _ = proc.communicate(timeout=300)
            return proc.returncode, stdout.decode("utf-8")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Command failed: {e}")
            return -1, None
    else:
        dry_run_resp = _DRY_RUN_RESPONSE.get(cmd_type, ["Dry run response not defined"])
        return 0, random.choice(dry_run_resp)  # noqa: S311


class _JobState(Enum):
    IN_PROGRESS = "in-progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@attrs.define
class _InputJob:
    index: int
    input_path: str
    output_path: str


@attrs.define
class _JobStatus:
    state: _JobState
    input_path: str


class _Worker:
    def __init__(self, invoke_template: Path, func_id: str, version_id: str) -> None:
        self._invoke_template = invoke_template
        self._func_id = func_id
        self._version_id = version_id
        self._index: int | None = None
        self._input_path: str | None = None
        self._output_path: str | None = None
        self._req_id: str | None = None
        self._rerun_count = 0

    @property
    def is_idle(self) -> bool:
        _is_idle = self._index is None
        if _is_idle:
            assert self._input_path is None
            assert self._output_path is None
            assert self._req_id is None
            assert self._rerun_count == 0
        return _is_idle

    def _get_nvcf_commands(self) -> list[str]:
        return [
            "cosmos-curator",
            "nvcf",
            "function",
        ]

    def _get_func_id_version(self) -> list[str]:
        return [
            "--funcid",
            self._func_id,
            "--version",
            self._version_id,
        ]

    def _make_invoke_json(self) -> str:
        _job_uuid = uuid.uuid5(uuid.NAMESPACE_URL, self._input_path)
        _invoke_json_path = Path(f"invoke-{self._index:04d}-{_job_uuid}.json")
        _invoke_json = {}
        with self._invoke_template.open("r") as fin:
            _invoke_json = json.load(fin)
            _invoke_json["args"]["input_video_list_json_path"] = self._input_path
            _invoke_json["args"]["output_clip_path"] = self._output_path
        with _invoke_json_path.open("w") as fout:
            json.dump(_invoke_json, fout, indent=2)
        return _invoke_json_path.as_posix()

    def _terminate_last_job(self, *, dry_run: bool) -> None:
        cmd = self._get_nvcf_commands()
        cmd += [
            "terminate-request",
            "--reqid",
            _NVCF_SPECIAL_REQ_ID,
        ]
        cmd += self._get_func_id_version()
        _, output = _run_command(cmd, cmd_type="terminate", dry_run=dry_run)
        logger.debug(f"Terminate output on func {self._func_id}:\n{output}")

    def _submit_job(self, cmd: list[str], *, dry_run: bool) -> None:
        retry_count = 0
        while True:
            _, output = _run_command(cmd, cmd_type="invoke", dry_run=dry_run)
            # find the req_id
            req_id: str | None = None
            if output is not None:
                for line in output.splitlines():
                    if "ReqId:" in line:
                        match = re.search(r"ReqId:\s*([a-f0-9\-]+)", line)
                        req_id = match.group(1) if match is not None else None
                        break
            # check if req_id is valid
            if req_id is not None:
                try:
                    _ = uuid.UUID(req_id)
                    self._req_id = req_id
                    logger.info(
                        f"Submitted job {self._index} at {self._input_path} "
                        f"with req_id {self._req_id} on func {self._func_id}"
                    )
                    break
                except ValueError:
                    logger.warning(f"Got invalid req_id {req_id} on func {self._func_id}")
            # error case
            logger.warning(f"Failed to submit job onto {self._func_id}, output was:\n{output}")
            if retry_count >= _NUM_RETRIES:
                err_msg = "Invoke keeps failing; I can no longer handle"
                raise RuntimeError(err_msg)
            retry_count += 1
            self._terminate_last_job(dry_run=dry_run)
            retry_sleep_time = 2 if dry_run else _RETRY_INTERVAL
            time.sleep(retry_sleep_time)
            continue

    def _launch_job(self, *, dry_run: bool) -> None:
        self._req_id = None
        cmd = self._get_nvcf_commands()
        cmd += [
            "invoke-function",
            "--data-file",
            self._make_invoke_json(),
            "--s3-config-file",
            _S3_CREDENTIALS_PATH,
            "--retry-cnt",
            str(1),
            "--no-wait",
        ]
        cmd += self._get_func_id_version()
        self._submit_job(cmd, dry_run=dry_run)

    def add_job(self, index: int, input_path: str, output: str, *, dry_run: bool) -> None:
        assert self.is_idle, "Worker is not idle"
        self._index = index
        self._input_path = input_path
        self._output_path = output
        # invoke it
        self._launch_job(dry_run=dry_run)

    def _reset(self) -> None:
        self._index = None
        self._input_path = None
        self._output_path = None
        self._req_id = None
        self._rerun_count = 0

    def _check_job_status(self, cmd: list[str], *, dry_run: bool) -> str:
        retry_count = 0
        while True:
            _, output = _run_command(cmd, cmd_type="status", dry_run=dry_run)
            # pass the status
            status = None
            status_line = ""
            if output is not None:
                for line in output.splitlines():
                    if "Status:" in line:
                        match = re.search(r"Status:\s*'([a-zA-Z0-9\-]+)'", line)
                        status = match.group(1) if match is not None else None
                        status_line = line.strip()
                        break
                    if "Could not get request status" in line:
                        status = "not-found"
                        status_line = line.strip()
                        break
            if status is not None:
                if status in ["in-progress", "failed", "fulfilled", "not-found"]:
                    logger.debug(f"Job {self._index} on func {self._func_id} status: {status} / {status_line}")
                    return status
                logger.warning(f"Got unexpected status {status} / {status_line} on func {self._func_id}")
            logger.warning(f"Failed to get job status on func {self._func_id}, output was:\n{output}")
            if retry_count >= _NUM_RETRIES:
                err_msg = "get-request-status keeps failing; I can no longer handle"
                raise RuntimeError(err_msg)
            retry_count += 1
            retry_sleep_time = 2 if dry_run else _RETRY_INTERVAL
            time.sleep(retry_sleep_time)

    def poll_job(self, *, dry_run: bool) -> _JobStatus:
        assert not self.is_idle
        assert self._req_id is not None
        cmd = self._get_nvcf_commands()
        cmd += [
            "get-request-status",
            "--reqid",
            self._req_id,
            "--no-wait",
        ]
        cmd += self._get_func_id_version()
        status = self._check_job_status(cmd, dry_run=dry_run)
        if status == "fulfilled":
            # reset the worker
            logger.info(f"Finished job {self._index} at {self._input_path} on func {self._func_id}")
            job_status = _JobStatus(state=_JobState.SUCCEEDED, input_path=self._input_path)
            self._reset()
            return job_status
        if status in {"failed", "not-found"}:
            logger.info(f"Job {self._index} at {self._input_path} failed or not found on func {self._func_id}")
            self._rerun_count += 1
            if self._rerun_count >= _NUM_RETRIES:
                logger.error(f"Job {self._index} at {self._input_path} failed too many times; giving up")
                job_status = _JobStatus(state=_JobState.FAILED, input_path=self._input_path)
                self._reset()
                return job_status
            # do a blind terminate and then restart the job
            self._terminate_last_job(dry_run=dry_run)
            self._launch_job(dry_run=dry_run)
        return _JobStatus(state=_JobState.IN_PROGRESS, input_path=self._input_path)


def _read_progress_file(progress_file: Path) -> set[str]:
    finished_jobs = set()
    if Path(progress_file).exists():
        with Path(progress_file).open("r") as fp:
            for line in fp:
                if line.strip() == "":
                    continue
                finished_jobs.add(line.strip())
    logger.info(f"Found {len(finished_jobs)} finished jobs")
    return finished_jobs


def _build_input_list(input_json: Path, progress_file: Path) -> dict[str, _InputJob]:
    # read in finished jobs to filter them out
    finished_jobs = _read_progress_file(progress_file)
    # read in input json
    assert Path(input_json).exists()
    pending_jobs = collections.OrderedDict()
    num_total_jobs = 0
    with Path(input_json).open("r") as fp:
        input_list = json.load(fp)
        for index, (input_path, output_path) in enumerate(input_list):
            num_total_jobs += 1
            if input_path in finished_jobs:
                continue
            pending_jobs[input_path] = _InputJob(index + 1, input_path, output_path)
    logger.info(f"About to process {len(pending_jobs)}/{num_total_jobs} jobs")
    return pending_jobs


def _setup_func_workers(func_json: Path, invoke_template: Path) -> list[_Worker]:
    assert Path(func_json).exists()
    assert Path(invoke_template).exists()
    workers = []
    with Path(func_json).open("r") as fp:
        func_list = json.load(fp)
        for func_id, version_id in func_list:
            workers.append(_Worker(invoke_template, func_id, version_id))
    return workers


def _main(args: argparse.Namespace) -> None:
    # build input list
    pending_jobs = _build_input_list(Path(args.input_json), Path(args.output_progress_file))
    # setup nvcf function workers
    workers = _setup_func_workers(Path(args.nvcf_func_json), Path(args.invoke_template))
    # main loop
    while len(pending_jobs) > 0 or any(not worker.is_idle for worker in workers):
        for worker in workers:
            if worker.is_idle:
                if len(pending_jobs) == 0:
                    continue
                _, job = pending_jobs.popitem(last=False)
                worker.add_job(job.index, job.input_path, job.output_path, dry_run=args.dry_run)
            else:
                job_status = worker.poll_job(dry_run=args.dry_run)
                if job_status.state == _JobState.SUCCEEDED:
                    # record the finished job
                    with Path(args.output_progress_file).open("a") as fp:
                        fp.write(job_status.input_path + "\n")
                        fp.flush()
        sleep_time = 1 if args.dry_run else 120
        time.sleep(sleep_time)


def setup_parser() -> argparse.ArgumentParser:
    """Set up the argument parser for the script."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Operate X nvcf functions to run N input files",
    )
    parser.add_argument(
        "--invoke-template",
        type=str,
        required=True,
        help="Template file for invoke.json",
    )
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Input json file with a list of [input-path, output-path] pairs",
    )
    parser.add_argument(
        "--output-progress-file",
        type=str,
        required=True,
        help="Output progress file to track the status of each input file",
    )
    parser.add_argument(
        "--nvcf-func-json",
        type=str,
        required=True,
        help="Function json file with a list of [func-id, version-id] pairs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, only print the commands that would be run, without executing them",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    _main(args)
