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
        _strip_qwen3_reasoning,
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


@pytest.mark.env("unified")
@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        # Closing-tag-only output (the actual Qwen3.5-27B-FP8 default behavior we
        # observe: the model streams reasoning content without an opening <think>
        # tag, then emits </think> before the answer).
        pytest.param(
            (
                "The user wants a description.\nLet me look at the frames.\n"
                "</think>\n\nThe video opens with a snowy mountain."
            ),
            "The video opens with a snowy mountain.",
            id="closing-tag-only-qwen3.5-default",
        ),
        # Explicit <think>...</think> wrapper (textbook reasoning format).
        pytest.param(
            "<think>chain of thought here</think>\n\nThe video shows mountains.",
            "The video shows mountains.",
            id="explicit-open-and-close-tags",
        ),
        # No reasoning tags at all (e.g. Qwen2.5-VL output passing through, or a
        # future Qwen3 checkpoint that honors enable_thinking=False at the chat
        # template layer).
        pytest.param(
            "The video begins with a scene set in a snowy, mountainous environment.",
            "The video begins with a scene set in a snowy, mountainous environment.",
            id="no-tags-passthrough",
        ),
        # count=1 behavior: an incidental "</think>" token that appears later in
        # the answer (e.g. in a quoted code block) must NOT truncate the caption.
        pytest.param(
            "<think>reasoning</think>\n\nThe word </think> appears in the answer.",
            "The word </think> appears in the answer.",
            id="count=1-incidental-close-tag-in-answer",
        ),
        # Empty-after-strip: model emitted only a reasoning block and then
        # terminated cleanly. The helper itself returns empty here; the decode()
        # override leaves this path alone (no answer to extract).
        pytest.param(
            "<think>just reasoning, no answer</think>",
            "",
            id="empty-after-strip",
        ),
        # Empty input: degenerate but should not raise.
        pytest.param(
            "",
            "",
            id="empty-input",
        ),
        # Missing-</think> boundary: when the closing tag is absent, the helper
        # returns text unchanged. Disambiguating "thinking off" vs "truncated
        # mid-reasoning" requires finish_reason context, which lives in
        # VllmQwen3VL.decode() rather than this pure-text helper.
        pytest.param(
            "Reasoning content with no closing tag because generation was truncated",
            "Reasoning content with no closing tag because generation was truncated",
            id="missing-close-tag-passthrough",
        ),
        # Multiline reasoning block: re.DOTALL is required for ``.`` to match
        # across newlines. Regression guard.
        pytest.param(
            "<think>line one\nline two\nline three</think>\n\nFinal answer.",
            "Final answer.",
            id="multiline-reasoning-re.DOTALL-regression-guard",
        ),
    ],
)
def test_strip_qwen3_reasoning(raw_text: str, expected: str) -> None:
    """Deterministic coverage for the Qwen3 chain-of-thought strip helper."""
    assert _strip_qwen3_reasoning(raw_text) == expected


@pytest.mark.env("unified")
def test_qwen3vl_decode_returns_empty_on_truncated_reasoning() -> None:
    """Truncation mid-reasoning (finish_reason='length' + no </think>) yields empty.

    Aligns with vLLM's own qwen3_reasoning_parser semantics so the caption pipeline
    can mark the window as failed (via VLLM_UNKNOWN_CAPTION fallthrough in
    vllm_interface._make_window_result) rather than persisting reasoning text as
    a successful caption.
    """
    raw_output = MagicMock()
    raw_output.outputs[0].text = "The user wants a detailed breakdown. Let me think about this..."
    raw_output.outputs[0].finish_reason = "length"

    assert VllmQwen3VL.decode(raw_output) == ""


@pytest.mark.env("unified")
def test_qwen3vl_decode_passes_through_on_clean_finish_without_tags() -> None:
    """Non-truncated output without ``</think>`` is passed through unchanged.

    Covers the future case where ``enable_thinking=False`` is honored at the
    chat-template layer and the model emits no reasoning at all. We must not
    swallow that legitimate caption.
    """
    raw_output = MagicMock()
    raw_output.outputs[0].text = "The video shows snowy mountains."
    raw_output.outputs[0].finish_reason = "stop"

    assert VllmQwen3VL.decode(raw_output) == "The video shows snowy mountains."


@pytest.mark.env("unified")
def test_qwen3vl_decode_strips_reasoning_on_clean_finish() -> None:
    """The happy path: reasoning block before answer, finish_reason='stop'."""
    raw_output = MagicMock()
    raw_output.outputs[0].text = "<think>reasoning</think>\n\nThe video shows snowy mountains."
    raw_output.outputs[0].finish_reason = "stop"

    assert VllmQwen3VL.decode(raw_output) == "The video shows snowy mountains."
