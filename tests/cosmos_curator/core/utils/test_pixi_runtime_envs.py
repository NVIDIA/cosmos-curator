# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for Pixi-backed Ray runtime environments."""

from cosmos_curator.core.utils.pixi_runtime_envs import PixiRuntimeEnv, ray_data_gpu_runtime_env


def test_pixi_runtime_env_sets_pixi_executable() -> None:
    """Pixi runtime envs launch Python inside the requested Pixi environment."""
    env = PixiRuntimeEnv("default")

    assert env.get("py_executable") == "pixi run --as-is -e default python"
    assert "env_vars" not in env


def test_ray_data_gpu_runtime_env_enables_ray_cuda_masking() -> None:
    """Ray Data GPU runtime envs restore Ray's CUDA device masking."""
    env = ray_data_gpu_runtime_env("default")

    assert env.get("py_executable") == "pixi run --as-is -e default python"
    assert env["env_vars"]["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "0"


def test_ray_data_gpu_runtime_env_pins_a_node_local_vllm_cache_root() -> None:
    """Ray Data GPU runtime envs keep the vLLM cache off the shared cluster filesystem."""
    env = ray_data_gpu_runtime_env("default")

    assert env["env_vars"]["VLLM_CACHE_ROOT"] == "/tmp/vllm"  # noqa: S108


def test_ray_data_gpu_runtime_env_merges_extra_env_vars() -> None:
    """GPU runtime env callers can pass additional actor environment variables."""
    env = ray_data_gpu_runtime_env("default", env_vars={"EXTRA": "1"})

    assert env["env_vars"]["EXTRA"] == "1"
    assert env["env_vars"]["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "0"


def test_ray_data_gpu_runtime_env_preserves_cuda_masking_override() -> None:
    """GPU runtime env callers cannot accidentally disable Ray CUDA masking."""
    env = ray_data_gpu_runtime_env("default", env_vars={"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"})

    assert env["env_vars"]["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "0"
