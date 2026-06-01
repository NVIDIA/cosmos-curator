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
"""Tests for vllm_model_ids.get_vllm_model_id."""

from contextlib import AbstractContextManager, nullcontext
from typing import Any

import pytest

from cosmos_curator.models.vllm_model_ids import get_vllm_model_id


@pytest.mark.parametrize(
    ("variant", "expected_model_id", "raises"),
    [
        ("qwen", "Qwen/Qwen2.5-VL-7B-Instruct", nullcontext()),
        ("cosmos_r1", "nvidia/Cosmos-Reason1-7B", nullcontext()),
        ("cosmos_r2", "nvidia/Cosmos-Reason2-8B", nullcontext()),
        ("cosmos3_nano_reasoner", "nvidia/Cosmos3-Nano-Reasoner", nullcontext()),
        ("cosmos3_super_reasoner", "nvidia/Cosmos3-Super-Reasoner", nullcontext()),
        ("qwen3_5_27b", "Qwen/Qwen3.5-27B-FP8", nullcontext()),
        ("qwen3_6_27b", "Qwen/Qwen3.6-27B", nullcontext()),
        ("qwen3_6_27b_fp8", "Qwen/Qwen3.6-27B-FP8", nullcontext()),
        ("unknown", None, pytest.raises(ValueError, match=r"vLLM model variant unknown not supported")),
    ],
)
def test_get_vllm_model_id(variant: str, expected_model_id: str, raises: AbstractContextManager[Any]) -> None:
    """Test get_vllm_model_id."""
    with raises:
        assert get_vllm_model_id(variant) == expected_model_id
