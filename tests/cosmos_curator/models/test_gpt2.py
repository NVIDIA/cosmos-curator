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
"""Integration test for the GPT-2 model interface."""

import pytest

from cosmos_curator.core.utils.model import model_utils
from cosmos_curator.models.gpt2 import GPT2


@pytest.mark.env("transformers")
def test_gpt2_generate_text() -> None:
    """Ensure GPT-2 loads real weights and generates text."""
    model = GPT2(max_output_len=32)
    weights_dir = model_utils.get_local_dir_for_weights_name(model.model_id_names[0])
    assert weights_dir.exists(), f"Expected pre-downloaded weights at {weights_dir}"

    model.setup()

    prompt = "Cosmos Curator enables efficient video curation"
    output = model.generate(prompt)
    assert output
