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
"""A Pixi-based runtime environment for Ray."""

from ray.runtime_env import RuntimeEnv


class PixiRuntimeEnv(RuntimeEnv):
    """RuntimeEnv that launches Python inside a Pixi environment.

    This thin wrapper forwards all arguments to :class:`ray.runtime_env.RuntimeEnv`
    but overrides the ``py_executable`` to run ``python`` via ``pixi run --as-is``
    when a Pixi environment name is provided.
    """

    def __init__(self, env_name: str, env_vars: dict[str, str] | None = None) -> None:
        """Create a Pixi-backed Ray runtime environment.

        Parameters
        ----------
        env_name: str
            Name of the Pixi environment to activate. If empty, the default
            Python executable resolution is used.
        env_vars: dict[str, str] | None
            Environment variables to forward into the Ray runtime environment.

        """
        copied_env_vars = None if env_vars is None else dict(env_vars)
        super().__init__(
            env_vars=copied_env_vars,
            py_executable=f"pixi run --as-is -e {env_name} python" if env_name else None,
        )
