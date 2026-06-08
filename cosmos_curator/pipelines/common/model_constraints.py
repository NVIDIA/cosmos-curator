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

"""Shared pipeline-level model capability constraints."""

import enum


class PreprocessMode(enum.StrEnum):
    """Owner of resize/rescale/normalize before a model consumes visual inputs."""

    CURATOR = "curator"
    MODEL = "model"


MODEL_VARIANTS_REQUIRING_MODEL_PREPROCESS: frozenset[str] = frozenset(
    {
        "cosmos3_nano",
        "cosmos3_super",
        "cosmos_r2",
        "nemotron",
        "qwen3_5_27b",
        "qwen3_6_27b",
        "qwen3_6_27b_fp8",
        "qwen3_vl_30b",
        "qwen3_vl_30b_fp8",
        "qwen3_vl_235b",
        "qwen3_vl_235b_fp8",
    }
)


def resolve_preprocess_mode(
    model_variant: str,
    requested_mode: PreprocessMode | str = PreprocessMode.CURATOR,
) -> PreprocessMode:
    """Return the effective preprocessing mode for a model variant."""
    requested_mode = PreprocessMode(requested_mode)
    if model_variant in MODEL_VARIANTS_REQUIRING_MODEL_PREPROCESS:
        return PreprocessMode.MODEL
    return requested_mode
