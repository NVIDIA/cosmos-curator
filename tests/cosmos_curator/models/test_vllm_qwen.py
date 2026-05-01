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
"""Test vllm_qwen.py."""

from unittest.mock import MagicMock

import pytest
import torch

from cosmos_curator.core.utils.model import conda_utils
from cosmos_curator.pipelines.video.utils.data_model import VllmConfig

if conda_utils.is_running_in_env("unified"):
    from cosmos_curator.models.vllm_qwen import (
        VllmQwen,
        VllmQwen3VL,
        VllmQwen7B,
        make_message,
        make_prompt,
    )

    _MODEL_VARIANT = VllmQwen7B.model_variant()


@pytest.mark.env("unified")
def test_make_llm_input_qwen() -> None:
    """Test make_llm_input (video path)."""
    # Mock the tokenizer to return a tensor that can be indexed and converted to list
    mock_tensor = torch.tensor([[1, 2, 3, 4, 5]])  # Shape: (1, 5)

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = mock_tensor

    # Create test frames tensor
    frames = torch.rand(2, 3, 32, 32)  # 2 frames, 3 channels, 32x32
    prompt = "Describe the video"
    metadata = {"fps": 2.0, "duration": 1.0}

    config = VllmConfig(model_variant="qwen")
    result = VllmQwen.make_llm_input(prompt, frames, metadata, mock_processor, config)

    # Verify structure
    assert "multi_modal_data" in result
    assert "video" in result["multi_modal_data"]
    assert result["prompt_token_ids"] == [1, 2, 3, 4, 5]  # Should be the token IDs as list
    assert len(result["multi_modal_data"]["video"]) == 1
    video_frames, video_metadata = result["multi_modal_data"]["video"][0]
    assert video_frames.shape == (2, 3, 32, 32)
    assert video_metadata == metadata


@pytest.mark.env("unified")
def test_make_llm_input_qwen_image() -> None:
    """Test make_llm_input with use_image_input=True (image pipeline path)."""
    mock_tensor = torch.tensor([[1, 2, 3, 4, 5]])
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = mock_tensor

    # Single image: (1, C, H, W)
    frames = torch.rand(1, 3, 32, 32)
    prompt = "Describe the image"
    config = VllmConfig(model_variant="qwen", use_image_input=True)

    result = VllmQwen.make_llm_input(prompt, frames, {}, mock_processor, config)

    assert "multi_modal_data" in result
    assert "image" in result["multi_modal_data"]
    assert "video" not in result["multi_modal_data"]
    assert result["prompt_token_ids"] == [1, 2, 3, 4, 5]
    assert result["multi_modal_data"]["image"].shape == (1, 3, 32, 32)


@pytest.mark.env("unified")
def test_make_llm_input_qwen3vl_image() -> None:
    """Test VllmQwen3VL.make_llm_input with use_image_input=True (image path)."""
    mock_tensor = torch.tensor([[1, 2, 3, 4, 5]])
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = mock_tensor

    frames = torch.rand(1, 3, 32, 32)
    prompt = "Describe the image"
    config = VllmConfig(model_variant="qwen3_vl_30b", use_image_input=True)

    result = VllmQwen3VL.make_llm_input(prompt, frames, {}, mock_processor, config)

    assert "multi_modal_data" in result
    assert "image" in result["multi_modal_data"]
    assert "video" not in result["multi_modal_data"]
    assert result["multi_modal_data"]["image"].shape == (1, 3, 32, 32)


@pytest.mark.env("unified")
def test_make_message() -> None:
    """Test make_message function (video path)."""
    prompt = "Test prompt"
    message = make_message(prompt)
    assert "role" in message
    assert message["role"] == "user"
    assert "content" in message
    assert isinstance(message["content"], list)
    content = message["content"]
    assert len(content) == 2


@pytest.mark.env("unified")
def test_make_message_image() -> None:
    """Test make_message with use_image=True (image pipeline path)."""
    prompt = "Describe the image"
    message = make_message(prompt, use_image=True)
    assert message["role"] == "user"
    content = message["content"]
    assert len(content) == 2
    # First content block should be image type
    assert content[0]["type"] == "image"
    assert content[1]["type"] == "text"
    assert content[1]["text"] == prompt


@pytest.mark.env("unified")
def test_make_prompt() -> None:
    """Test make_prompt function (video path)."""
    # Mock the tokenizer to return a tensor that can be indexed and converted to list
    mock_tensor = torch.tensor([[10, 20, 30, 40]])  # Shape: (1, 4)

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = mock_tensor

    prompt = "Test prompt"
    frames = torch.rand(2, 3, 32, 32)
    metadata = {"fps": 2.0, "duration": 1.0}
    message = make_message(prompt)
    result = make_prompt(message, [(frames, metadata)], mock_processor)
    assert result["prompt_token_ids"] == [10, 20, 30, 40]  # Should be the token IDs as list
    assert len(result["multi_modal_data"]["video"]) == 1
    video_frames, video_metadata = result["multi_modal_data"]["video"][0]
    assert video_frames.shape == (2, 3, 32, 32)
    assert video_metadata == metadata


@pytest.mark.env("unified")
def test_make_llm_input_qwen3vl_video() -> None:
    """Test VllmQwen3VL.make_llm_input uses the same video payload format as base Qwen."""
    mock_tensor = torch.tensor([[1, 2, 3, 4, 5]])
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = mock_tensor

    frames = torch.rand(2, 3, 32, 32)
    metadata = {"fps": 2.0, "duration": 1.0}
    prompt = "Describe the video"
    config = VllmConfig(model_variant="qwen3_vl_30b")

    result = VllmQwen3VL.make_llm_input(prompt, frames, metadata, mock_processor, config)

    assert "multi_modal_data" in result
    assert "video" in result["multi_modal_data"]
    assert len(result["multi_modal_data"]["video"]) == 1
    video_frames, video_metadata = result["multi_modal_data"]["video"][0]
    assert video_frames.shape == (2, 3, 32, 32)
    assert video_metadata == metadata


@pytest.mark.env("unified")
def test_make_prompt_image() -> None:
    """Test make_prompt with use_image=True (image pipeline path)."""
    mock_tensor = torch.tensor([[10, 20, 30, 40]])
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = mock_tensor

    prompt = "Describe the image"
    message = make_message(prompt, use_image=True)
    # Single image: (1, C, H, W)
    image_frames = torch.rand(1, 3, 32, 32)
    result = make_prompt(message, image_frames, mock_processor, use_image=True)

    assert result["prompt_token_ids"] == [10, 20, 30, 40]
    assert "image" in result["multi_modal_data"]
    assert "video" not in result["multi_modal_data"]
    assert result["multi_modal_data"]["image"].shape == (1, 3, 32, 32)
