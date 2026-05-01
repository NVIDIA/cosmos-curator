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

"""Unit tests for image vLLM caption result normalization."""

import pathlib

import attrs

from cosmos_curator.pipelines.image.captioning.image_vllm_stages import (
    _normalize_vllm_result,
    _scatter_captions,
)
from cosmos_curator.pipelines.image.utils.data_model import Image, ImagePipeTask
from cosmos_curator.pipelines.video.utils.data_model import CaptionOutcome, TokenCounts


@attrs.define
class _FakeVllmWindowResult:
    text: str
    finish_reason: str | None = None
    token_counts: TokenCounts = attrs.field(factory=TokenCounts)


def test_normalize_vllm_result_marks_truncated_output() -> None:
    """Length-limited responses with text should become truncated captions."""
    result = _FakeVllmWindowResult(text="trimmed caption", finish_reason="length")

    normalized = _normalize_vllm_result(result)

    assert normalized.outcome == CaptionOutcome.TRUNCATED
    assert normalized.text == "trimmed caption"
    assert normalized.failure_reason is None


def test_scatter_captions_writes_plain_normalized_fields() -> None:
    """Scatter stores plain caption text, token counts, and status on the image task."""
    fake_path = pathlib.Path("/fake/example.jpg")
    task = ImagePipeTask(
        session_id=str(fake_path),
        image=Image(input_image=fake_path),
    )
    result = _FakeVllmWindowResult(
        text="a dog on a couch",
        finish_reason=None,
        token_counts=TokenCounts(prompt_tokens=7, output_tokens=9),
    )

    _scatter_captions([task], [0], [result], "qwen", verbose=False)

    assert task.image.caption == "a dog on a couch"
    assert task.image.captions == {"qwen": "a dog on a couch"}
    assert task.image.token_counts["qwen"] == TokenCounts(prompt_tokens=7, output_tokens=9)
    assert task.image.caption_status == "success"
    assert task.image.caption_failure_reason is None
    assert task.image.has_caption() is True
