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
"""vLLM model ids.

This module provides model ids for a model variant.

Originally, these IDs were included in two places, which causes confusion.
The reason was to solve the problem of the vllm package not being available in
the default conda environment, which is used when building pipelines.

Instead, these IDs have been moved to a separate file, which can be imported
by each plugin, and by vllm_caption_stages.py, in the default conda
environment.

This allows for a location for variant -> model id mapping.
"""

_VLLM_MODELS = {
    "cosmos_r1": "nvidia/Cosmos-Reason1-7B",
    "cosmos_r2": "nvidia/Cosmos-Reason2-8B",
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "nemotron": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
    "qwen3_vl_235b": "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "qwen3_vl_235b_fp8": "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
    "qwen3_5_27b": "Qwen/Qwen3.5-27B-FP8",
    "qwen3_vl_30b": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "qwen3_vl_30b_fp8": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
}


def get_vllm_model_id(model_variant: str) -> str:
    """Get the vLLM model ID for the model variant.

    Args:
        model_variant: The variant of the model.

    Returns:
        The vLLM model ID.

    Raises:
        ValueError: If the model variant is not supported.

    """
    if model_variant not in _VLLM_MODELS:
        msg = f"vLLM model variant {model_variant} not supported"
        raise ValueError(msg)
    return _VLLM_MODELS[model_variant]
