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
"""Provide utilities for getting operation info."""

import contextlib
import os
import pathlib
import sys
import tempfile
from collections.abc import Generator

from cosmos_curator.core.utils.environment import (
    LOCAL_DOCKER_ENV_VAR_NAME,
    PIXI_ENVIRONMENT_NAME_VAR_NAME,
    SLURM_RAY_ENV_VAR_NAME,
)

# ANSI escape codes for error highlighting
_ERROR_COLOR = "\033[1;31m"
_RESET_COLOR = "\033[0m"


def is_running_on_slurm() -> bool:
    """Check if the job is running on SLURM."""
    return SLURM_RAY_ENV_VAR_NAME in os.environ


def is_running_on_the_cloud() -> bool:
    """Check if the job is running on the cloud."""
    return LOCAL_DOCKER_ENV_VAR_NAME not in os.environ


def get_tmp_dir() -> pathlib.Path:
    """Retrieve the appropriate temporary directory based on the runtime environment.

    Returns:
        pathlib.Path: Path to the temporary directory.

    """
    if is_running_on_the_cloud():
        if is_running_on_slurm():
            return pathlib.Path(tempfile.gettempdir())
        tmp_dir = pathlib.Path("/config/tmp/")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        return tmp_dir
    return pathlib.Path(tempfile.gettempdir())


@contextlib.contextmanager
def make_temporary_dir(
    *,
    prefix: str | None = None,
    target_dir: pathlib.Path | None = None,
    delete: bool = True,
) -> Generator[pathlib.Path, None, None]:
    """Context manager to create a temporary directory.

    Args:
        prefix (Optional[str], optional): Prefix for the directory name. Defaults to None.
        target_dir (Optional[pathlib.Path], optional): Parent directory for the temporary directory. Defaults to None.
        delete (bool, optional): If True, the directory will be deleted upon exit. Defaults to True.

    Yields:
        Generator[pathlib.Path, None, None]: Path of the created temporary directory.

    """
    if target_dir is None:
        target_dir = get_tmp_dir()
    if prefix is not None:
        prefix = str(prefix)

    # If not set to delete, make the directory and yield its path
    if not delete:
        yield pathlib.Path(tempfile.mkdtemp(prefix=prefix, dir=str(target_dir)))
    else:
        with tempfile.TemporaryDirectory(dir=target_dir, prefix=prefix) as tmp_dir:
            yield pathlib.Path(tmp_dir)


@contextlib.contextmanager
def make_named_temporary_file(
    *,
    prefix: str | None = None,
    suffix: str | None = None,
    delete: bool = True,
    target_dir: pathlib.Path | None = None,
) -> Generator[pathlib.Path, None, None]:
    """Context manager to create a named temporary file.

    Args:
        prefix (Optional[str], optional): Prefix for the file name. Defaults to None.
        suffix (Optional[str], optional): suffix for the file name. Defaults to None.
        delete (bool, optional): If True, the file will be deleted upon exit. Defaults to True.
        target_dir (Optional[pathlib.Path], optional): Directory where the file should be created. Defaults to None.

    Yields:
        Generator[pathlib.Path, None, None]: Path of the created temporary file.

    """
    if target_dir is None:
        target_dir = get_tmp_dir()

    with tempfile.NamedTemporaryFile(dir=target_dir, delete=delete, prefix=prefix, suffix=suffix) as file:
        yield pathlib.Path(file.name)


@contextlib.contextmanager
def make_pipeline_temporary_dir(
    sub_dir: str | None = None,
) -> Generator[pathlib.Path, None, None]:
    """Context manager to create a temporary directory for pipelines."""
    if sub_dir is not None:
        target_dir = get_tmp_dir() / pathlib.Path("ray_pipeline") / pathlib.Path(sub_dir)
    else:
        target_dir = get_tmp_dir() / pathlib.Path("ray_pipeline")
    target_dir.mkdir(parents=True, exist_ok=True)
    with make_temporary_dir(target_dir=target_dir, delete=True) as tdir:
        yield tdir


@contextlib.contextmanager
def make_pipeline_named_temporary_file(
    sub_dir: str | None = None,
    suffix: str | None = None,
) -> Generator[pathlib.Path, None, None]:
    """Context manager to create a named temporary file for pipelines."""
    if sub_dir is not None:
        target_dir = get_tmp_dir() / pathlib.Path("ray_pipeline") / pathlib.Path(sub_dir)
    else:
        target_dir = get_tmp_dir() / pathlib.Path("ray_pipeline")
    target_dir.mkdir(parents=True, exist_ok=True)
    with make_named_temporary_file(delete=True, target_dir=target_dir, suffix=suffix) as file:
        yield file


def is_running_in_pixi_env() -> bool:
    """Check if the job is running in a pixi environment."""
    return PIXI_ENVIRONMENT_NAME_VAR_NAME in os.environ


def check_if_running_in_pixi_env() -> None:
    """Check if the job is running in a pixi environment and exit if not."""
    if not is_running_in_pixi_env():
        error_msg = (
            "Error: Pipelines must be executed inside a Pixi environment.\n"
            "\n"
            "  To use the default environment:\n"
            "    pixi run --as-is python -m <module> [args]\n"
            "  To use a named environment:\n"
            "    pixi run --as-is -e <env_name> python -m <module> [args]\n"
            "\n"
            "Run `pixi run --help` for more details.\n"
        )
        sys.exit(f"{_ERROR_COLOR}{error_msg}{_RESET_COLOR}")
