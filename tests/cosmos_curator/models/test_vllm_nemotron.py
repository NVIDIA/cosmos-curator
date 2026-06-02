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
"""Test vllm_nemotron.py."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from cosmos_curator.core.utils.model import pixi_utils
from cosmos_curator.pipelines.video.utils.data_model import VllmCaptionRequest, VllmConfig

if pixi_utils.is_running_in_env("unified"):
    from cosmos_curator.models.vllm_nemotron import VllmNemotronNano12Bv2VL, make_message, make_prompt

    _MODEL_VARIANT = VllmNemotronNano12Bv2VL.model_variant()


@pytest.mark.env("unified")
def test_make_llm_input_nemotron() -> None:
    """Test make_llm_input_nemotron function."""
    # Mock the tokenizer to return a tensor that can be indexed and converted to list
    mock_tensor = torch.tensor([[1, 2, 3, 4, 5]])  # Shape: (1, 5)

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = mock_tensor

    N, C, H, W = 2, 3, 32, 64

    # Create test frames tensor
    frames = torch.rand(N, C, H, W)  # 2 frames, 3 channels, 32x32
    prompt = "Describe the video"

    metadata: dict[str, int | float | list[int] | str] = {
        "fps": 2,
        "duration": 1.0,
        "frames_indices": [0, 1],
        "video_backend": "opencv",
    }

    config = VllmConfig(model_variant="nemotron", video_max_pixels_per_frame=602112)
    result = VllmNemotronNano12Bv2VL.make_llm_input(prompt, frames, metadata, mock_processor, config)

    # Verify structure
    assert "multi_modal_data" in result
    assert "video" in result["multi_modal_data"]
    assert result["prompt_token_ids"] == [1, 2, 3, 4, 5]  # Should be the token IDs as list
    assert result["multi_modal_data"]["video"][0].shape == (N, H, W, C)
    assert result["multi_modal_data"]["video"][1]["fps"] == metadata["fps"]
    assert "mm_processor_kwargs" not in result


@pytest.mark.env("unified")
def test_make_message() -> None:
    """Test make_message function."""
    prompt = "Test prompt"
    message = make_message(prompt)
    assert "role" in message
    assert message["role"] == "user"
    assert "content" in message
    assert isinstance(message["content"], list)
    content = message["content"]
    assert len(content) == 2


@pytest.mark.env("unified")
def test_make_prompt() -> None:
    """Test make_prompt function."""
    # Mock the tokenizer to return a tensor that can be indexed and converted to list
    mock_tensor = torch.tensor([[10, 20, 30, 40]])  # Shape: (1, 4)

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = mock_tensor

    metadata: dict[str, int | float | list[int] | str] = {
        "fps": 2,
        "duration": 1.0,
        "frames_indices": [0, 1],
        "video_backend": "opencv",
    }

    N, C, H, W = 2, 3, 32, 64
    C = 3
    prompt = "Test prompt"
    frames = torch.rand(N, C, H, W)
    message = make_message(prompt)
    result = make_prompt(message, frames, metadata, mock_processor)
    assert result["prompt_token_ids"] == [10, 20, 30, 40]  # Should be the token IDs as list
    assert result["multi_modal_data"]["video"][0].shape == (N, H, W, C)
    assert result["multi_modal_data"]["video"][1]["fps"] == metadata["fps"]


@pytest.mark.env("unified")
def test_make_refined_llm_request_nemotron() -> None:
    """Test refine flow creates a new request preserving video (numpy path) and updating prompt."""
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])

    # Nemotron stores video as (video_np, nemotron_metadata) with numpy (T, H, W, C)
    video_np = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    nemotron_metadata = {
        "total_num_frames": 1,
        "fps": 30.0,
        "duration": 1.0,
        "frames_indices": [0],
        "video_backend": "opencv",
    }
    base_inputs = {
        "prompt_token_ids": [0],
        "multi_modal_data": {"video": (video_np, nemotron_metadata)},
    }
    base_req = VllmCaptionRequest(
        request_id="r1",
        inputs=base_inputs,
        caption="stage1 caption",
    )

    refined = VllmNemotronNano12Bv2VL.make_refined_llm_request(base_req, mock_processor, refine_prompt=None)
    assert "prompt_token_ids" in refined.inputs
    assert "multi_modal_data" in refined.inputs
    assert "video" in refined.inputs["multi_modal_data"]
    refined_video_np, refined_meta = refined.inputs["multi_modal_data"]["video"]
    assert refined_video_np.shape == video_np.shape
    assert refined_meta["fps"] == nemotron_metadata["fps"]
