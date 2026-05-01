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
"""Test vllm_cosmos_reason1_vl.py."""

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from cosmos_curator.core.utils.model import conda_utils
from cosmos_curator.pipelines.video.utils.data_model import VllmCaptionRequest, VllmConfig

_PLUGIN_CLASSES: tuple[type, ...] = ()

if conda_utils.is_running_in_env("unified"):
    from cosmos_curator.models.vllm_cosmos_reason1_vl import (
        VllmCosmosReason1VL,
        _extract_from_reasoning_format,
        make_message,
        make_prompt,
    )
    from cosmos_curator.models.vllm_cosmos_reason2_vl import VllmCosmosReason2VL

    _PLUGIN_CLASSES = (VllmCosmosReason1VL, VllmCosmosReason2VL)

if not _PLUGIN_CLASSES:
    pytest.skip("Cosmos Reason vLLM tests require unified environment", allow_module_level=True)


@pytest.mark.env("unified")
@pytest.mark.parametrize("plugin_cls", _PLUGIN_CLASSES, ids=lambda cls: cls.model_variant())
def test_make_llm_input_cosmos_reason(plugin_cls: type[VllmCosmosReason1VL]) -> None:
    """Test make_llm_input for Cosmos-Reason plugins."""
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "mocked_reasoning_prompt"

    frames = torch.rand(2, 3, 32, 32)
    prompt = "Describe the video"

    metadata = {"fps": 30, "duration": 1.0}
    config = VllmConfig(model_variant=plugin_cls.model_variant())
    result = plugin_cls.make_llm_input(prompt, frames, metadata, mock_processor, config)

    assert "multi_modal_data" in result
    assert "video" in result["multi_modal_data"]
    assert result["prompt"] == "mocked_reasoning_prompt"
    assert len(result["multi_modal_data"]["video"]) == 1
    # Video is stored as (frames, metadata) tuple
    video_frames, video_metadata = result["multi_modal_data"]["video"][0]
    assert video_frames.shape == (2, 3, 32, 32)
    assert video_metadata == metadata


@pytest.mark.env("unified")
def test_extract_from_reasoning_format() -> None:
    """Test that decode extracts <answer>...</answer> content."""
    text = "<think>some thoughts</think>\n<answer>final caption</answer>"
    assert _extract_from_reasoning_format(text) == "final caption"

    # Fallback if missing tags
    plain = "no tags here"
    assert _extract_from_reasoning_format(plain) == plain


@pytest.mark.env("unified")
def test_make_prompt_uses_chat_template() -> None:
    """Ensure make_prompt uses processor.apply_chat_template and wires video correctly."""
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "chat-prompt"

    frames = torch.rand(1, 3, 16, 16)
    metadata = {"fps": 30, "duration": 1.0}
    result = make_prompt(make_message("hello"), frames, metadata, mock_processor)
    assert result["prompt"] == "chat-prompt"
    # Video is stored as (frames, metadata) tuple
    video_frames, video_metadata = result["multi_modal_data"]["video"][0]
    assert video_frames.shape == (1, 3, 16, 16)
    assert video_metadata == metadata


@pytest.mark.env("unified")
@pytest.mark.parametrize("plugin_cls", _PLUGIN_CLASSES, ids=lambda cls: cls.model_variant())
def test_make_refined_llm_request(plugin_cls: type[VllmCosmosReason1VL]) -> None:
    """Test refine flow creates a new request preserving video and updating prompt."""
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "refined-prompt"

    frames = torch.rand(1, 3, 8, 8)
    metadata = {"fps": 30, "duration": 1.0}
    # Video is stored as (frames, metadata) tuple
    base_inputs = {"prompt": "base", "multi_modal_data": {"video": [(frames, metadata)]}}

    base_req = VllmCaptionRequest(
        request_id="r1",
        inputs=base_inputs,
        caption="stage1 caption",
    )

    refined = plugin_cls.make_refined_llm_request(base_req, mock_processor, refine_prompt=None)
    assert refined.inputs["prompt"] == "refined-prompt"
    refined_frames, refined_metadata = refined.inputs["multi_modal_data"]["video"][0]
    assert refined_frames.shape == (1, 3, 8, 8)
    assert refined_metadata == metadata


@pytest.mark.env("unified")
def test_stage2_refine_prompt_equivalence_with_real_processor() -> None:
    """Integration test: verify refine prompt equivalence using the real processor.

    Fails if model weights are unavailable or processor lacks apply_chat_template.

    This test is used as part of the migration from cosmos_reason1_vl to vllm_cosmos_reason1_vl.

    It is expected that the prompt generated by the chat template provided by the model's processor
    will be the same as the prompt generated by the regex substitution used previously.

    Over the long term, we expect to migrate away from the regex substitution and use the chat template
    provided by the model's processor, making this test obsolete.
    """
    vllm_config = VllmConfig(model_variant="cosmos_r1")
    model_path = Path(str(VllmCosmosReason1VL.model_path(vllm_config)))
    if not model_path.exists():
        pytest.fail("Cosmos-Reason1 weights not available locally; this integration test requires them.")

    processor = VllmCosmosReason1VL.processor(vllm_config)
    if not hasattr(processor, "apply_chat_template"):
        pytest.fail("Processor lacks apply_chat_template; this integration test requires it.")

    frames = torch.rand(1, 3, 8, 8)
    metadata = {"fps": 30, "duration": 1.0}

    # Generate initial prompt via real processor
    initial_inputs = VllmCosmosReason1VL.make_llm_input("initial user text", frames, metadata, processor, vllm_config)
    initial_prompt = initial_inputs["prompt"]

    caption = "stage1 caption"
    refine_prompt = "REFINE:\n"

    pattern = (
        r"(<\|im_start\|>system\s*.*?<\|im_end\|>\s*"
        r"<\|im_start\|>user\s*<\|vision_start\|><\|video_pad\|><\|vision_end\|>\s*)(.*?)(\s*<\|im_end\|>)"
    )
    expected = re.sub(pattern, rf"\1{refine_prompt + caption}\3", initial_prompt, flags=re.DOTALL)

    base_req = VllmCaptionRequest(
        request_id="r1",
        inputs=initial_inputs,
        caption=caption,
    )
    refined = VllmCosmosReason1VL.make_refined_llm_request(base_req, processor, refine_prompt=refine_prompt)

    assert refined.inputs["prompt"] == expected
