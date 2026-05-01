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
"""Utilities for dealing with conda environments."""

import os


def is_running_in_env(env_name: str) -> bool:
    """Check whether python is running under a given env name."""
    current_conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    return current_conda_env in (env_name, f"cosmos-curator:{env_name}")


def get_conda_env_name() -> str:
    """Return the name of the current conda environment."""
    return os.environ.get("CONDA_DEFAULT_ENV", "")
