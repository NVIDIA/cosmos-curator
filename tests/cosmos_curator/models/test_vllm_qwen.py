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

from cosmos_curator.core.utils.model import pixi_utils
from cosmos_curator.pipelines.video.utils.data_model import VllmCaptionRequest, VllmConfig, WindowConfig

if pixi_utils.is_running_in_env("default"):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, Qwen2_5_VLProcessor, Qwen2VLImageProcessor, Qwen2VLVideoProcessor
    from vllm.config import ModelConfig, MultiModalConfig
    from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLMultiModalProcessor, Qwen2_5_VLProcessingInfo
    from vllm.model_executor.models.qwen2_vl import Qwen2VLMultiModalDataParser
    from vllm.multimodal.processing import InputProcessingContext

    from cosmos_curator.models.vllm_exceptions import ReasoningOutputTruncatedError
    from cosmos_curator.models.vllm_interface import make_metadata
    from cosmos_curator.models.vllm_qwen import (
        VllmQwen,
        VllmQwen3VL,
        VllmQwen7B,
        _strip_qwen3_reasoning,
        make_message,
        make_prompt,
        qwen3_video_size_kwargs,
    )
    from cosmos_curator.pipelines.video.utils.vision_process import VIDEO_MIN_PIXELS

    _MODEL_VARIANT = VllmQwen7B.model_variant()


def _qwen3_like_apply_chat_template(
    *_: object,
    enable_thinking: bool = True,
    **__: object,
) -> torch.Tensor:
    """Return distinct token IDs for open-thinking vs no-thinking Qwen3 prompts."""
    if enable_thinking:
        return torch.tensor([[99, 100]])
    return torch.tensor([[10, 20, 30, 40]])


@pytest.mark.env("default")
def test_make_llm_input_qwen() -> None:
    """Test make_llm_input (video path)."""
    # Mock the Transformers v5 tensor return shape.
    mock_tensor = torch.tensor([[1, 2, 3, 4, 5]])  # Shape: (1, 5)

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = mock_tensor

    # Create test frames tensor
    frames = torch.rand(2, 3, 32, 32)  # 2 frames, 3 channels, 32x32
    prompt = "Describe the video"
    metadata = {"fps": 2.0, "duration": 1.0, "video_backend": "opencv", "do_sample_frames": False}

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
    assert "mm_processor_kwargs" not in result


@pytest.mark.env("default")
@pytest.mark.parametrize("sampling_fps", [2.0, 4.0])
@pytest.mark.parametrize("refine", [False, True], ids=["initial", "refined"])
def test_qwen_video_input_reaches_hf_processor(sampling_fps: float, *, refine: bool) -> None:
    """Real vLLM/HF processing preserves sampled frames and timing without duplicate kwargs."""
    frames = torch.zeros(4, 3, 56, 56, dtype=torch.uint8)
    metadata = make_metadata([frames], WindowConfig(sampling_fps=sampling_fps))[0]
    prompt_processor = MagicMock()
    prompt_processor.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])
    config = VllmConfig(model_variant="qwen")
    inputs = VllmQwen.make_llm_input("Describe the video", frames, metadata, prompt_processor, config)
    if refine:
        request = VllmCaptionRequest(request_id="r1", inputs=inputs, caption="A video caption")
        inputs = VllmQwen.make_refined_llm_request(request, prompt_processor).inputs

    # A tiny tokenizer avoids downloading weights; video preprocessing and vLLM's
    # metadata extraction and processor invocation use the installed implementations.
    unknown_token = "<unk>"  # noqa: S105 - tokenizer vocabulary marker
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({unknown_token: 0, "<|image_pad|>": 1, "<|video_pad|>": 2}, unk_token=unknown_token),
        ),
        unk_token=unknown_token,
        additional_special_tokens=["<|image_pad|>", "<|video_pad|>"],
    )
    hf_processor = Qwen2_5_VLProcessor(
        image_processor=Qwen2VLImageProcessor(),
        video_processor=Qwen2VLVideoProcessor(cap_pixels_per_frame=True),
        tokenizer=tokenizer,
    )
    model_config = MagicMock(spec=ModelConfig)
    model_config.dtype = torch.float32
    model_config.get_multimodal_config.return_value = MultiModalConfig(
        mm_processor_kwargs={"do_resize": False, "do_sample_frames": False},
    )
    info = MagicMock(spec=Qwen2_5_VLProcessingInfo)
    info.ctx = InputProcessingContext(model_config, tokenizer)
    info.get_data_parser.return_value = Qwen2VLMultiModalDataParser(spatial_merge_size=2, video_needs_metadata=True)
    info.get_hf_processor.return_value = hf_processor
    vllm_processor = Qwen2_5_VLMultiModalProcessor(info, MagicMock())

    mm_items = vllm_processor.data_parser.parse_mm_data(inputs["multi_modal_data"])
    hf_inputs = vllm_processor._get_hf_mm_inputs(mm_items, inputs.get("mm_processor_kwargs", {}))
    processed = vllm_processor._call_hf_processor(hf_inputs.hf_data, hf_inputs.hf_kwargs)

    assert processed["video_grid_thw"].tolist() == [[2, 4, 4]]
    assert processed["second_per_grid_ts"] == pytest.approx([2 / sampling_fps])


@pytest.mark.env("default")
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


@pytest.mark.env("default")
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
    assert "mm_processor_kwargs" not in result


@pytest.mark.env("default")
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


@pytest.mark.env("default")
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


@pytest.mark.env("default")
def test_make_prompt() -> None:
    """Test make_prompt function (video path)."""
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.side_effect = _qwen3_like_apply_chat_template

    prompt = "Test prompt"
    frames = torch.rand(2, 3, 32, 32)
    metadata = {"fps": 2.0, "duration": 1.0}
    message = make_message(prompt)
    result = make_prompt(message, [(frames, metadata)], mock_processor)
    assert result["prompt_token_ids"] == [10, 20, 30, 40]  # Should be the no-thinking token IDs as list
    assert len(result["multi_modal_data"]["video"]) == 1
    video_frames, video_metadata = result["multi_modal_data"]["video"][0]
    assert video_frames.shape == (2, 3, 32, 32)
    assert video_metadata == metadata


@pytest.mark.env("default")
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
    assert "mm_processor_kwargs" not in result


@pytest.mark.env("default")
@pytest.mark.parametrize("num_frames", [2, 3])
def test_qwen3_video_size_kwargs_scales_edges_by_frame_count(num_frames: int) -> None:
    """Qwen3 processor size is expressed as a whole-video budget."""
    result = qwen3_video_size_kwargs(num_frames, 602112)

    assert result == {
        "size": {
            "shortest_edge": num_frames * VIDEO_MIN_PIXELS,
            "longest_edge": num_frames * 602112,
        }
    }


@pytest.mark.env("default")
def test_qwen3_video_size_kwargs_boundary_min_equals_max() -> None:
    """At the lower boundary, the fixed floor and user cap translate to the same whole-video budget."""
    result = qwen3_video_size_kwargs(3, VIDEO_MIN_PIXELS)

    assert result["size"]["shortest_edge"] == result["size"]["longest_edge"]


@pytest.mark.env("default")
def test_make_llm_input_qwen3vl_video_emits_top_level_size() -> None:
    """Qwen3 video inputs carry request-level processor size when the sync cap is set."""
    mock_tensor = torch.tensor([[1, 2, 3, 4, 5]])
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = mock_tensor

    frames = torch.rand(3, 3, 32, 32)
    metadata = {"fps": 2.0, "duration": 1.5}
    config = VllmConfig(model_variant="qwen3_vl_30b", video_max_pixels_per_frame=602112)

    result = VllmQwen3VL.make_llm_input("Describe the video", frames, metadata, mock_processor, config)

    assert result["mm_processor_kwargs"] == {
        "size": {
            "shortest_edge": 3 * VIDEO_MIN_PIXELS,
            "longest_edge": 3 * 602112,
        }
    }
    assert "mm_processor_kwargs" not in result["multi_modal_data"]


@pytest.mark.env("default")
def test_make_llm_input_qwen_does_not_emit_size_when_cap_is_set() -> None:
    """Qwen2.5 keeps timing in the video payload without Qwen3 request-level size."""
    mock_tensor = torch.tensor([[1, 2, 3, 4, 5]])
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = mock_tensor

    frames = torch.rand(2, 3, 32, 32)
    metadata = {"fps": 2.0, "duration": 1.0, "video_backend": "opencv"}
    config = VllmConfig(model_variant="qwen", video_max_pixels_per_frame=602112)

    result = VllmQwen.make_llm_input("Describe the video", frames, metadata, mock_processor, config)

    assert result["multi_modal_data"]["video"][0][1] == metadata
    assert "mm_processor_kwargs" not in result


@pytest.mark.env("default")
def test_make_llm_input_qwen3vl_image_does_not_emit_video_size_when_cap_is_set() -> None:
    """Image inputs do not receive video processor sizing."""
    mock_tensor = torch.tensor([[1, 2, 3, 4, 5]])
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = mock_tensor

    frames = torch.rand(1, 3, 32, 32)
    config = VllmConfig(model_variant="qwen3_vl_30b", use_image_input=True, video_max_pixels_per_frame=602112)

    result = VllmQwen3VL.make_llm_input("Describe the image", frames, {}, mock_processor, config)

    assert "mm_processor_kwargs" not in result


@pytest.mark.env("default")
def test_qwen_stage2_refine_preserves_mm_processor_kwargs() -> None:
    """Stage-2 refinement keeps request-level processor sizing from stage 1."""
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])

    frames = torch.rand(2, 3, 32, 32)
    size_kwargs = qwen3_video_size_kwargs(2, 602112)
    base_req = VllmCaptionRequest(
        request_id="r1",
        inputs={
            "prompt_token_ids": [0],
            "multi_modal_data": {"video": [(frames, {"fps": 2.0})]},
            "mm_processor_kwargs": size_kwargs,
        },
        caption="stage1 caption",
    )

    refined = VllmQwen3VL.make_refined_llm_request(base_req, mock_processor, refine_prompt="Refine: ")

    assert refined.inputs["mm_processor_kwargs"] == size_kwargs


@pytest.mark.env("default")
def test_make_prompt_image() -> None:
    """Test make_prompt with use_image=True (image pipeline path)."""
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.side_effect = _qwen3_like_apply_chat_template

    prompt = "Describe the image"
    message = make_message(prompt, use_image=True)
    # Single image: (1, C, H, W)
    image_frames = torch.rand(1, 3, 32, 32)
    result = make_prompt(message, image_frames, mock_processor, use_image=True)

    assert result["prompt_token_ids"] == [10, 20, 30, 40]
    assert "image" in result["multi_modal_data"]
    assert "video" not in result["multi_modal_data"]
    assert result["multi_modal_data"]["image"].shape == (1, 3, 32, 32)


@pytest.mark.env("default")
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


@pytest.mark.env("default")
def test_qwen3vl_decode_raises_on_truncated_reasoning() -> None:
    """Truncation mid-reasoning is fatal instead of becoming a per-window failure."""
    raw_output = MagicMock()
    raw_output.outputs[0].text = "The user wants a detailed breakdown. Let me think about this..."
    raw_output.outputs[0].finish_reason = "length"

    with pytest.raises(
        ReasoningOutputTruncatedError, match="before a complete answer could be safely extracted"
    ) as exc_info:
        VllmQwen3VL.decode(raw_output)

    msg = str(exc_info.value)
    assert "output_chars=" in msg
    assert "output_sha256_prefix=" in msg
    assert "detailed breakdown" not in msg


@pytest.mark.env("default")
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


@pytest.mark.env("default")
def test_qwen3vl_decode_strips_reasoning_on_clean_finish() -> None:
    """The happy path: reasoning block before answer, finish_reason='stop'."""
    raw_output = MagicMock()
    raw_output.outputs[0].text = "<think>reasoning</think>\n\nThe video shows snowy mountains."
    raw_output.outputs[0].finish_reason = "stop"

    assert VllmQwen3VL.decode(raw_output) == "The video shows snowy mountains."
