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
"""Provide utilities for consistent interface across different environments.

See README.md for more info.
"""

import math
import os
import pathlib

from loguru import logger

# Environment variable to tell b/w local and cloud job.
LOCAL_DOCKER_ENV_VAR_NAME = "COSMOS_CURATOR_LOCAL_DOCKER_JOB"
SLURM_RAY_ENV_VAR_NAME = "COSMOS_CURATOR_RAY_SLURM_JOB"

# Driver-side OTLP metrics push: opt-in switch and scrape/push interval (seconds).
# Shared by the NVCF deployment defaults and the run-pipeline entrypoint.
OTLP_METRICS_PUSH_ENV_VAR = "COSMOS_CURATOR_OTLP_METRICS_PUSH"
OTLP_METRICS_PUSH_INTERVAL_ENV_VAR = "COSMOS_CURATOR_OTLP_METRICS_PUSH_INTERVAL"

# Each large-source download reserves one slot to cap concurrency per node
# independently of CPU reservations; capacity follows elastic workers.
# See docs/curator/design/ray-data.md#large-source-downloads-and-curator_io.
CURATOR_IO_RESOURCE_NAME = "curator_io"
CURATOR_IO_SLOTS_PER_NODE_ENV_VAR = "COSMOS_CURATOR_RAY_IO_SLOTS_PER_NODE"
DEFAULT_CURATOR_IO_SLOTS_PER_NODE = 16

# Where pipeline code is located. Overridable so tooling can run against a local checkout
# (e.g. `COSMOS_CURATOR_CODE_DIR=$(pwd) python -m cosmos_curator.core.managers.model_cli ...`)
# instead of requiring a symlink or bind mount at /opt/cosmos-curator.
CONTAINER_PATHS_CODE_DIR = pathlib.Path(os.getenv("COSMOS_CURATOR_CODE_DIR", "/opt/cosmos-curator"))

# Local cosmos_curator config file path.
LOCAL_COSMOS_CURATOR_CONFIG_FILE = pathlib.Path("~/.config/cosmos_curator/config.yaml").expanduser()

# Local workspace path.
_LOCAL_WORKSPACE_PATH_PREFIX = os.getenv("COSMOS_CURATOR_LOCAL_WORKSPACE_PREFIX", "~")
LOCAL_WORKSPACE_PATH = pathlib.Path(_LOCAL_WORKSPACE_PATH_PREFIX).expanduser() / "cosmos_curator_local_workspace"

# Where cosmos_curator config file is located.
CONTAINER_PATHS_COSMOS_CURATOR_CONFIG_FILE = pathlib.Path("/cosmos_curator/config/cosmos_curator.yaml")

# Workspace path to cache/distribute weights & engines
CONTAINER_PATHS_DEFAULT_WORKSPACE_DIR = pathlib.Path("/config")

# Where to put model weights
CONTAINER_PATHS_MODEL_WEIGHT_CACHE_DIR = CONTAINER_PATHS_DEFAULT_WORKSPACE_DIR / "models"

# Where to put TRT LLM engines
CONTAINER_PATHS_ENGINE_CACHE_DIR = CONTAINER_PATHS_DEFAULT_WORKSPACE_DIR / "engines"

# Where to download model weights at runtime if not found in local workspace
MODEL_WEIGHTS_PREFIX = "s3://your_bucket_name/model_weights/"

# S3 & Azure credentials
S3_PROFILE_PATH = pathlib.Path(os.getenv("COSMOS_S3_PROFILE_PATH", "/dev/shm/s3_creds_file"))  # noqa: S108
# Named separately from the resolved path so that a caller re-reading
# COSMOS_AZURE_PROFILE_PATH at runtime has a default to fall back to that is not itself
# an import-time reading of that variable.
DEFAULT_AZURE_PROFILE_PATH = pathlib.Path("/dev/shm/azure_creds_file")  # noqa: S108
AZURE_PROFILE_PATH = pathlib.Path(os.getenv("COSMOS_AZURE_PROFILE_PATH", str(DEFAULT_AZURE_PROFILE_PATH)))

# Local S3 credentials file path.
LOCAL_AWS_CREDENTIALS_FILE = pathlib.Path("~/.aws/credentials").expanduser()

# Local Azure credentials file path
LOCAL_AZURE_CREDENTIALS_FILE = pathlib.Path("~/.azure/credentials").expanduser()

# Environment variable for the pixi environment name.
PIXI_ENVIRONMENT_NAME_VAR_NAME = "PIXI_ENVIRONMENT_NAME"


# Lenient env-var readers: a bad value warns and yields None so the caller
# falls back to its default; a typo in the environment must not fail the run.
def env_bool(name: str) -> bool | None:
    """Parse a boolean env var leniently; an invalid value warns and yields None."""
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


def env_int(name: str, *, minimum: int | None = None) -> int | None:
    """Parse an integer env var leniently; invalid or out-of-range values warn and yield None."""
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


def env_float(name: str, *, minimum: float | None = None, maximum: float | None = None) -> float | None:
    """Parse a float env var leniently; invalid or out-of-bounds values warn and yield None."""
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
