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
"""Utilities for dealing with Pixi environments."""

import os

from cosmos_curator.core.utils.environment import PIXI_ENVIRONMENT_NAME_VAR_NAME


def is_running_in_env(env_name: str) -> bool:
    """Check whether Python is running under a given Pixi environment name."""
    return os.environ.get(PIXI_ENVIRONMENT_NAME_VAR_NAME) == env_name


def get_env_name() -> str:
    """Return the name of the current Pixi environment."""
    return os.environ.get(PIXI_ENVIRONMENT_NAME_VAR_NAME, "")
