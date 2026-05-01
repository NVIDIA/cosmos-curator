# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Test the av captioning stages."""

import copy
from collections.abc import Collection, Generator
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from cosmos_curator.pipelines.av.captioning.captioning_stages import (
    EnhanceCaptionStage,
    _add_prefix_to_captions,
    _decode_vri_text,
    _filter_prompts,
    _get_frame_counts,
    enhance_captions,
)
from cosmos_curator.pipelines.av.utils.av_data_model import (
    AvClipAnnotationTask,
    CaptionWindow,
    ClipForAnnotation,
)

DEFAULT_BATCH_SIZE = 128
CUSTOM_BATCH_SIZE = 64


def clip0(prompt_type: str) -> ClipForAnnotation:
    """ClipForAnnotation fixture 0."""
    return ClipForAnnotation(
        video_session_name="test-session-clip0",
        clip_session_uuid=UUID("11111111-1111-1111-1111-111111111111"),
        uuid=UUID("22222222-2222-2222-2222-222222222222"),
        camera_id=0,
        span_index=0,
        url="s3://bucket/clip0",
        caption_windows=[
            CaptionWindow(
                start_frame=0,
                end_frame=256,
                captions={prompt_type: ["Original caption 1"]},
            ),
        ],
    )


def clip1(prompt_type: str) -> ClipForAnnotation:
    """ClipForAnnotation fixture 1."""
    return ClipForAnnotation(
        video_session_name="test-session-clip1",
        clip_session_uuid=UUID("33333333-3333-3333-3333-333333333333"),
        uuid=UUID("44444444-4444-4444-4444-444444444444"),
        camera_id=0,
        span_index=1,
        url="s3://bucket/clip1",
        caption_windows=[
            CaptionWindow(
                start_frame=0,
                end_frame=256,
                captions={prompt_type: ["Original caption 2"]},
            ),
        ],
    )


@pytest.fixture
def mock_qwen_lm() -> Generator[MagicMock, None, None]:
    """Mock ChatLM used by EnhanceCaptionStage."""
    with patch("cosmos_curator.pipelines.av.captioning.captioning_stages.ChatLM") as mock:
        mock_instance = MagicMock()
        # Return number of captions based on input length
        mock_instance.generate.side_effect = lambda inputs, *_args, **_kwargs: [
            f"Enhanced caption {i + 1}" for i in range(len(inputs))
        ]
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def sample_task() -> AvClipAnnotationTask:
    """Sample task fixture."""
    return AvClipAnnotationTask(clips=[clip0("road_conditions"), clip1("road_conditions")])


def test_enhance_caption_stage_init(mock_qwen_lm: MagicMock) -> None:
    """Test initialization of EnhanceCaptionStage with default parameters."""
    stage = EnhanceCaptionStage()
    assert stage._batch_size == DEFAULT_BATCH_SIZE
    assert stage._verbose is False
    assert stage._log_stats is False
    assert stage._prompt_variants == ["default"]
    assert stage._prompt_text is None
    assert stage._raw_model == mock_qwen_lm


def test_enhance_caption_stage_init_custom_params(mock_qwen_lm: MagicMock) -> None:
    """Test initialization of EnhanceCaptionStage with custom parameters."""
    stage = EnhanceCaptionStage(
        prompt_variants=["visibility"],
        prompt_text="Custom prompt",
        batch_size=CUSTOM_BATCH_SIZE,
        fp8_enable=True,
        max_output_tokens=1024,
        verbose=True,
        log_stats=True,
    )
    assert stage._batch_size == CUSTOM_BATCH_SIZE
    assert stage._verbose is True
    assert stage._log_stats is True
    assert stage._prompt_variants == ["visibility"]
    assert stage._prompt_text == "Custom prompt"
    assert stage._raw_model == mock_qwen_lm


@pytest.mark.parametrize(
    ("captions", "prompt_variant_key", "prompt_prefixes", "expected_captions"),
    [
        # Single caption with visibility prefix
        (
            ["The road is wet"],
            "visibility",
            {"visibility": "Here is the visibility condition: "},
            ["Here is the visibility condition: The road is wet"],
        ),
        # Multiple captions with road conditions prefix
        (
            ["The road is wet", "The road is dry"],
            "road_conditions",
            {"road_conditions": "Here is the road condition: "},
            [
                "Here is the road condition: The road is wet",
                "Here is the road condition: The road is dry",
            ],
        ),
        # Multiple captions with illumination prefix
        (
            ["The street is dark", "The street is bright"],
            "illumination",
            {"illumination": "Here is the illumination condition: "},
            [
                "Here is the illumination condition: The street is dark",
                "Here is the illumination condition: The street is bright",
            ],
        ),
        # Empty captions list
        (
            [],
            "visibility",
            {"visibility": "Here is the visibility condition: "},
            [],
        ),
        # Default prompt variant (empty prefix)
        (
            ["The scene is clear"],
            "default",
            {"default": ""},
            ["The scene is clear"],
        ),
    ],
)
def test_add_prefix_to_captions(
    captions: list[str],
    prompt_variant_key: str,
    prompt_prefixes: dict[str, str],
    expected_captions: list[str],
) -> None:
    """Test adding prefixes to captions.

    Args:
        captions: List of captions to modify
        prompt_variant_key: Type of prompt (visibility, road_conditions, illumination, default)
        prompt_prefixes: Dictionary mapping prompt variants to their prefix strings
        expected_captions: Expected list of captions with prefixes added

    """
    modified_captions = _add_prefix_to_captions(
        captions=captions,
        prompt_variant_key=prompt_variant_key,
        prompt_prefixes=prompt_prefixes,
    )
    assert modified_captions == expected_captions


@pytest.mark.parametrize(
    (
        "clips",
        "prompt_variant_key",
        "caption_prefixes",
        "prompt_variants",
        "expected_captions",
    ),
    [
        # Single clip with visibility enhancement
        (
            [clip0("visibility")],
            "visibility",
            {"visibility": "Here is the visibility condition: "},
            {"visibility": "Classify the visibility"},
            ["Enhanced caption 1"],
        ),
        # Multiple clips with road conditions enhancement
        (
            [clip0("road_conditions"), clip1("road_conditions")],
            "road_conditions",
            {"road_conditions": "Here is the road condition: "},
            {"road_conditions": "Classify the road condition"},
            ["Enhanced caption 1", "Enhanced caption 2"],
        ),
        # Single clip with illumination enhancement
        (
            [clip0("illumination")],
            "illumination",
            {"illumination": "Here is the illumination condition: "},
            {"illumination": "Classify the lighting"},
            ["Enhanced caption 1"],
        ),
        # Empty clips list
        (
            [],
            "visibility",
            {"visibility": "Here is the visibility condition: "},
            {"visibility": "Classify the visibility"},
            [],
        ),
        # Default prompt variant (empty prefix and prompt)
        (
            [clip0("default")],
            "default",
            {"default": ""},
            {"default": ""},
            ["Enhanced caption 1"],
        ),
    ],
)
def test_enhance_captions(  # noqa: PLR0913
    mock_qwen_lm: MagicMock,
    clips: list[ClipForAnnotation],
    prompt_variant_key: str,
    caption_prefixes: dict[str, str],
    prompt_variants: dict[str, str],
    expected_captions: list[str],
) -> None:
    """Test the enhance_captions function.

    Args:
        mock_qwen_lm: Mock QwenLM instance
        clips: List of clips containing caption windows
        prompt_variant_key: Type of prompt, for example: (visibility,
            road_conditions,illumination, default)
        caption_prefixes: Dictionary of prefixes to add to the captions.
            prompt_variant_key is used to choose the prefix
        prompt_variants: Dictionary of prompts to send to the model.
            prompt_variant_key is used to choose the prompt
        prompt_text: Text of the prompt to send to the model
        expected_captions: Expected list of enhanced captions

    """
    # Create a copy of clips to avoid modifying the test fixtures
    test_clips = [copy.deepcopy(clip) for clip in clips]

    enhance_captions(
        clips=test_clips,
        model=mock_qwen_lm,
        caption_prefixes=caption_prefixes,
        prompt_variant_key=prompt_variant_key,
        prompt_variants=prompt_variants,
        prompt_text=None,
    )
    _PROMPT_COUNT = 2

    # Verify the model was called with the correct inputs
    if clips:
        mock_qwen_lm.generate.assert_called_once()
        call_args = mock_qwen_lm.generate.call_args[0][0]
        # Verify the number of prompts matches the number of clips
        assert len(call_args) == len(clips)

        # Verify each prompt has the correct structure and content
        for i, prompt in enumerate(call_args):
            assert len(prompt) == _PROMPT_COUNT  # system and user messages
            assert prompt[0]["role"] == "system"
            assert prompt[1]["role"] == "user"
            assert prompt[1]["content"].startswith(caption_prefixes[prompt_variant_key])
            assert prompt[1]["content"].endswith(clips[i].caption_windows[0].captions[prompt_variant_key][0])
    else:
        # For empty clips list, verify generate was called with empty list
        mock_qwen_lm.generate.assert_called_once_with([], batch_size=None)

    # Verify the enhanced captions were appended correctly
    for i, clip in enumerate(test_clips):
        if clips:
            _EXPECTED_CAPTION_LEN = 2
            assert len(clip.caption_windows[0].captions[prompt_variant_key]) == _PROMPT_COUNT
            assert clip.caption_windows[0].captions[prompt_variant_key][1] == expected_captions[i]
        else:
            assert len(clip.caption_windows[0].captions[prompt_variant_key]) == 1


@pytest.mark.parametrize(
    (
        "prompt_variants",
        "target_clip_size",
        "front_window_size",
        "expected_frame_counts",
    ),
    [
        # Single default prompt
        (["default"], 256, 57, [256, 57]),
        # Multiple prompts including default
        (["default", "visibility"], 256, 57, [256, 57]),
        # Only non-default prompts
        (["visibility", "road_conditions"], 256, 57, [256]),
        # Empty collection
        ([], 256, 57, []),
        # Empty collection
        ({}.keys(), 256, 57, []),
        # Set input
        ({"default"}, 256, 57, [256, 57]),
        # Dict keys input
        ({"default": "", "visibility": ""}.keys(), 256, 57, [256, 57]),
    ],
)
def test_get_frame_counts(
    prompt_variants: Collection[str],
    target_clip_size: int,
    front_window_size: int,
    expected_frame_counts: list[int],
) -> None:
    """Test the _get_frame_counts function with various input collections.

    Args:
        prompt_variants: Collection of prompt variant strings to test
        target_clip_size: Target size for the clip
        front_window_size: Size of the front window
        expected_frame_counts: Expected list of frame counts to be returned

    """
    assert _get_frame_counts(prompt_variants, target_clip_size, front_window_size) == expected_frame_counts


@pytest.mark.parametrize(
    (
        "frame_count",
        "prompts",
        "target_clip_size",
        "front_window_size",
        "expected_prompts",
        "raises",
    ),
    [
        # Case with target_clip_size - should return all prompts
        (
            256,
            {"default": "default_prompt", "visibility": "visibility_prompt"},
            256,
            57,
            {"default": "default_prompt", "visibility": "visibility_prompt"},
            does_not_raise(),
        ),
        # Case with front_window_size - should return only default prompt
        (
            57,
            {"default": "default_prompt", "visibility": "visibility_prompt"},
            256,
            57,
            {"default": "default_prompt"},
            does_not_raise(),
        ),
        # Case with empty prompts
        (
            256,
            {},
            256,
            57,
            {},
            does_not_raise(),
        ),
        # Case with single prompt
        (
            256,
            {"default": "default_prompt"},
            256,
            57,
            {"default": "default_prompt"},
            does_not_raise(),
        ),
        # Invalid frame count
        (
            100,
            {"default": "default_prompt"},
            256,
            57,
            None,
            pytest.raises(ValueError),  # noqa: PT011
        ),
    ],
)
def test_filter_prompts(  # noqa: PLR0913
    frame_count: int,
    prompts: dict[str, str],
    target_clip_size: int,
    front_window_size: int,
    expected_prompts: dict[str, str] | None,
    raises: AbstractContextManager[Any],
) -> None:
    """Test the _filter_prompts function with various input parameters.

    Args:
        frame_count: Number of frames in the clip
        prompts: Dictionary of prompt variants and their text
        target_clip_size: Target size for the clip
        front_window_size: Size of the front window
        expected_prompts: Expected filtered prompts dictionary, or None if expecting an error
        raises: Expected exception to be raised, or None if expecting success

    """
    with raises:
        result = _filter_prompts(frame_count, prompts, target_clip_size, front_window_size)
        assert result == expected_prompts


@pytest.mark.parametrize(
    ("caption_text", "prompt_variant", "expected_vri_tags", "raises"),
    [
        (
            "**Visibility Condition:** Clear.\n**Road Surface Condition:** Dry.\n**Illumination Condition:** Bright.",
            "vri",
            {"visibility": "clear", "road_condition": "dry", "illumination": "bright"},
            does_not_raise(),
        ),
        (
            "**Visibility Condition:** Clear\n**Road Surface Condition:** Dry\n**Illumination Condition:** Bright",
            "vri",
            {"visibility": "clear", "road_condition": "dry", "illumination": "bright"},
            does_not_raise(),
        ),
        (
            "Total nonsense, some hallucination",
            "vri",
            {},
            pytest.raises(ValueError),  # noqa: PT011
        ),
        (
            "Total nonsense, some hallucination",
            "default",
            {},
            does_not_raise(),
        ),
    ],
)
def test_decode_vri_text(
    caption_text: str, prompt_variant: str, expected_vri_tags: dict[str, str], raises: AbstractContextManager[Any]
) -> None:
    """Test the _decode_vri_text function with various input parameters.

    Args:
        caption_text: Text of the caption to decode
        prompt_variant: Type of prompt to decode
        expected_vri_tags: Expected dictionary of VRI tags
        raises: Expected exception to be raised, or None if expecting success

    """
    with raises:
        assert _decode_vri_text(caption_text, prompt_variant) == expected_vri_tags
