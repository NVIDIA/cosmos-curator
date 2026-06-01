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
"""vLLM plugins for Cosmos3-Reasoner vision-language models."""

from cosmos_curator.models.vllm_qwen import VllmQwen3VL


class VllmCosmos3NanoReasonerVL(VllmQwen3VL):
    """Cosmos3-Nano-Reasoner vLLM model variant plugin."""

    @staticmethod
    def model_variant() -> str:
        """Return the model variant name."""
        return "cosmos3_nano_reasoner"


class VllmCosmos3SuperReasonerVL(VllmQwen3VL):
    """Cosmos3-Super-Reasoner vLLM model variant plugin."""

    @staticmethod
    def model_variant() -> str:
        """Return the model variant name."""
        return "cosmos3_super_reasoner"
