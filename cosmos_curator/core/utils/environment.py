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

import os
import pathlib

# Environment variable to tell b/w local and cloud job.
LOCAL_DOCKER_ENV_VAR_NAME = "COSMOS_CURATOR_LOCAL_DOCKER_JOB"
SLURM_RAY_ENV_VAR_NAME = "COSMOS_CURATOR_RAY_SLURM_JOB"

# Where pipeline code is located.
CONTAINER_PATHS_CODE_DIR = pathlib.Path("/opt/cosmos-curator")

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
AZURE_PROFILE_PATH = pathlib.Path(os.getenv("COSMOS_AZURE_PROFILE_PATH", "/dev/shm/azure_creds_file"))  # noqa: S108

# Local S3 credentials file path.
LOCAL_AWS_CREDENTIALS_FILE = pathlib.Path("~/.aws/credentials").expanduser()

# Local Azure credentials file path
LOCAL_AZURE_CREDENTIALS_FILE = pathlib.Path("~/.azure/credentials").expanduser()

# Environment variable for the pixi environment name.
PIXI_ENVIRONMENT_NAME_VAR_NAME = "PIXI_ENVIRONMENT_NAME"
