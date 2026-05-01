# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Test AV data model."""

from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any
from uuid import UUID

import pytest

from cosmos_curator.pipelines.av.utils.av_data_model import (
    CaptionWindow,
    ClipForAnnotation,
    append_captions_to_clips,
    get_clip_window_mappings,
    get_last_captions,
)


@pytest.mark.parametrize(
    ("clips", "prompt_variant", "skip_missing", "expected_mappings", "raises"),
    [
        # Single clip with single window with caption
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session-1",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=0,
                    span_index=0,
                    url="test-url-1",
                    caption_windows=[
                        CaptionWindow(0, 100, captions={"visibility": ["caption1"]}),
                    ],
                ),
            ],
            "visibility",
            "captions",
            [(0, 0)],
            does_not_raise(),
        ),
        # Single clip with multiple windows, one without caption
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session-2",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=0,
                    span_index=0,
                    url="test-url-2",
                    caption_windows=[
                        CaptionWindow(0, 100, captions={"visibility": ["caption1"]}),
                        CaptionWindow(0, 200, captions={"visibility": []}),  # Empty captions
                        CaptionWindow(0, 300, captions={"visibility": ["caption3"]}),
                    ],
                ),
            ],
            "visibility",
            "captions",
            [(0, 0), (0, 2)],  # Middle window should be skipped
            does_not_raise(),
        ),
        # Multiple clips with mixed caption states
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session-3",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=0,
                    span_index=0,
                    url="test-url-3",
                    caption_windows=[
                        CaptionWindow(0, 100, captions={"visibility": ["caption1"]}),
                        CaptionWindow(0, 200, captions={"visibility": None}),  # Empty captions dict
                    ],
                ),
                ClipForAnnotation(
                    video_session_name="test-session-3",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=1,
                    span_index=0,
                    url="test-url-4",
                    caption_windows=[
                        CaptionWindow(0, 100, captions={"visibility": ["caption2"]}),
                    ],
                ),
            ],
            "visibility",
            "captions",
            [(0, 0), (1, 0)],  # Second window of first clip should be skipped
            does_not_raise(),
        ),
        # Clip with no windows
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session-5",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=0,
                    span_index=0,
                    url="test-url-5",
                    caption_windows=[],
                ),
            ],
            "visibility",
            "captions",
            [],
            does_not_raise(),
        ),
        # Empty clips list
        (
            [],
            "visibility",
            "captions",
            [],
            does_not_raise(),
        ),
        # Model inputs tests
        # Single clip with single window with model input
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session-6",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=0,
                    span_index=0,
                    url="test-url-1",
                    caption_windows=[
                        CaptionWindow(0, 100, model_input={"visibility": {"input": "test"}}),
                    ],
                ),
            ],
            "visibility",
            "model_input",
            [(0, 0)],
            does_not_raise(),
        ),
        # Single clip with multiple windows, one without model input
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session-7",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=0,
                    span_index=0,
                    url="test-url-2",
                    caption_windows=[
                        CaptionWindow(0, 100, model_input={"visibility": {"input": "test1"}}),
                        CaptionWindow(0, 200, model_input={}),  # Empty model input
                        CaptionWindow(0, 300, model_input={"visibility": {"input": "test3"}}),
                    ],
                ),
            ],
            "visibility",
            "model_input",
            [(0, 0), (0, 2)],  # Middle window should be skipped
            does_not_raise(),
        ),
        # Multiple clips with mixed model input states
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session-8",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=0,
                    span_index=0,
                    url="test-url-3",
                    caption_windows=[
                        CaptionWindow(0, 100, model_input={"visibility": {"input": "test1"}}),
                        CaptionWindow(0, 200, model_input={}),  # Empty model input
                    ],
                ),
                ClipForAnnotation(
                    video_session_name="test-session-8",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=1,
                    span_index=0,
                    url="test-url-4",
                    caption_windows=[
                        CaptionWindow(0, 100, model_input={"visibility": {"input": "test2"}}),
                    ],
                ),
            ],
            "visibility",
            "model_input",
            [(0, 0), (1, 0)],  # Second window of first clip should be skipped
            does_not_raise(),
        ),
        # Invalid skip_missing value
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session-9",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=0,
                    span_index=0,
                    url="test-url-1",
                    caption_windows=[
                        CaptionWindow(0, 100, captions={"visibility": ["caption1"]}),
                    ],
                ),
            ],
            "visibility",
            "invalid_value",
            [],
            pytest.raises(ValueError),  # noqa: PT011
        ),
    ],
)
def test_get_clip_window_mappings(
    clips: list[ClipForAnnotation],
    prompt_variant: str,
    skip_missing: str,
    expected_mappings: list[tuple[int, int]],
    raises: AbstractContextManager[Any],
) -> None:
    """Test getting mappings for windows with captions or model inputs.

    Args:
        clips: List of clips containing caption windows
        prompt_variant: Type of prompt to use (e.g. 'default', 'visibility', etc.)
        skip_missing: Whether to skip clips without captions or model inputs
        expected_mappings: Expected list of (clip_idx, window_idx) tuples for windows with captions
        raises: Expected exception context

    """
    with raises:
        mappings = get_clip_window_mappings(clips, prompt_variant, skip_missing)
        assert mappings == expected_mappings


@pytest.mark.parametrize(
    ("clips", "prompt_variant", "captions", "expected_captions", "raises"),
    [
        # Append single caption to single window
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session-10",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=0,
                    span_index=0,
                    url="test-url-1",
                    caption_windows=[
                        CaptionWindow(0, 100, captions={"default": ["original1-a"]}),
                    ],
                ),
            ],
            "default",
            ["enhanced1-a"],
            [["original1-a", "enhanced1-a"]],
            does_not_raise(),
        ),
        # Append multiple captions to multiple windows
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session-11",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=0,
                    span_index=0,
                    url="test-url-1",
                    caption_windows=[
                        CaptionWindow(0, 100, captions={"default": ["original1-b"]}),
                        CaptionWindow(0, 200, captions={"default": ["original2-b"]}),
                    ],
                ),
            ],
            "default",
            ["enhanced1-b", "enhanced2-b"],
            [["original1-b", "enhanced1-b"], ["original2-b", "enhanced2-b"]],
            does_not_raise(),
        ),
        # Append captions across multiple clips
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session-12",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=0,
                    span_index=0,
                    url="test-url-1",
                    caption_windows=[
                        CaptionWindow(0, 100, captions={"default": ["original1-c"]}),
                    ],
                ),
                ClipForAnnotation(
                    video_session_name="test-session-12",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=1,
                    span_index=0,
                    url="test-url-2",
                    caption_windows=[
                        CaptionWindow(0, 100, captions={"default": ["original2-c"]}),
                    ],
                ),
            ],
            "default",
            ["enhanced1-c", "enhanced2-c"],
            [["original1-c", "enhanced1-c"], ["original2-c", "enhanced2-c"]],
            does_not_raise(),
        ),
        # Empty captions list
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session-13",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=0,
                    span_index=0,
                    url="test-url-1",
                    caption_windows=[
                        CaptionWindow(0, 100, captions={"default": ["original1"]}),
                    ],
                ),
            ],
            "default",
            [],
            [["original1"]],
            pytest.raises(ValueError),  # noqa: PT011
        ),
        # Mappings and captions do not match
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session-14",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=0,
                    span_index=0,
                    url="test-url-1",
                    caption_windows=[
                        CaptionWindow(0, 100, captions={"default": ["original1"]}),
                    ],
                ),
            ],
            "default",
            ["enhanced1-d", "enhanced2-d"],
            [],
            pytest.raises(ValueError),  # noqa: PT011
        ),
    ],
)
def test_append_captions_to_clips(
    clips: list[ClipForAnnotation],
    prompt_variant: str,
    captions: list[str],
    expected_captions: list[list[str]],
    raises: AbstractContextManager[Any],
) -> None:
    """Test appending captions to clip windows.

    Args:
        clips: List of clips containing caption windows
        prompt_variant: Type of prompt to use (e.g. 'default', 'visibility', etc.)
        captions: List of captions to append
        expected_captions: Expected caption chains after appending
        raises: Expected exception context

    """
    with raises:
        mappings = get_clip_window_mappings(clips, prompt_variant, skip_missing="captions")
        append_captions_to_clips(clips, prompt_variant, captions, mappings)

        # Verify each window's caption chain matches expected
        for clip_idx, clip in enumerate(clips):
            for window_idx, window in enumerate(clip.caption_windows):
                if (clip_idx, window_idx) in mappings:
                    expected_idx = mappings.index((clip_idx, window_idx))
                    assert window.captions[prompt_variant] == expected_captions[expected_idx]
                else:
                    # Windows not in mappings should remain unchanged
                    assert window.captions.get(prompt_variant, []) == ["original" + str(window_idx + 1)]


@pytest.mark.parametrize(
    ("clips", "prompt_variant", "expected_captions"),
    [
        # Single clip with single window
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session-15",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=0,
                    span_index=0,
                    url="test-url-1",
                    caption_windows=[
                        CaptionWindow(0, 100, captions={"default": ["caption1", "caption2"]}),
                    ],
                ),
            ],
            "default",
            ["caption2"],  # Should get the last caption
        ),
        # Single clip with multiple windows
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session-16",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=0,
                    span_index=0,
                    url="test-url-2",
                    caption_windows=[
                        CaptionWindow(0, 100, captions={"default": ["caption1", "caption2"]}),
                        CaptionWindow(0, 200, captions={"default": ["caption3", "caption4"]}),
                    ],
                ),
            ],
            "default",
            ["caption2", "caption4"],  # Should get last caption from each window
        ),
        # Multiple clips with multiple windows
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session-17",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=0,
                    span_index=0,
                    url="test-url-3",
                    caption_windows=[
                        CaptionWindow(0, 100, captions={"default": ["caption1", "caption2"]}),
                    ],
                ),
                ClipForAnnotation(
                    video_session_name="test-session-17",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=1,
                    span_index=0,
                    url="test-url-4",
                    caption_windows=[
                        CaptionWindow(0, 100, captions={"default": ["caption3", "caption4"]}),
                    ],
                ),
            ],
            "default",
            [
                "caption2",
                "caption4",
            ],  # Should get last caption from each window across clips
        ),
        # Single caption in chain
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session-18",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=0,
                    span_index=0,
                    url="test-url-6",
                    caption_windows=[
                        CaptionWindow(0, 100, captions={"default": ["caption1"]}),
                    ],
                ),
            ],
            "default",
            ["caption1"],  # Should get the only caption
        ),
    ],
)
def test_get_last_captions(
    clips: list[ClipForAnnotation],
    prompt_variant: str,
    expected_captions: list[str],
) -> None:
    """Test extracting the last caption from each window.

    Args:
        clips: List of clips containing caption windows
        prompt_variant: Type of prompt to use (e.g. 'default', 'visibility', etc.)
        expected_captions: Expected list of last captions from each window

    """
    mappings = get_clip_window_mappings(clips, prompt_variant, "captions")
    last_captions = get_last_captions(clips, prompt_variant, mappings)
    assert last_captions == expected_captions


@pytest.mark.parametrize(
    ("last_caption_only", "attr_white_list", "expected_attrs", "expected_captions"),
    [
        # Default behavior - include all attributes except model_input and t5_xxl_embeddings
        (
            False,
            None,
            ["start_frame", "end_frame", "captions"],
            {"default": ["caption1", "caption2"]},
        ),
        # Last caption only - include only the last caption from each prompt variant
        (
            True,
            None,
            ["start_frame", "end_frame", "captions"],
            {"default": ["caption2"]},
        ),
        # Custom attribute white list - include only specified attributes
        (
            False,
            ["start_frame", "end_frame"],
            ["start_frame", "end_frame"],
            {"default": ["caption1"]},  # Not checking captions as they're not in the white list
        ),
    ],
)
def test_caption_window_to_dict(
    *,
    last_caption_only: bool,
    attr_white_list: list[str] | None,
    expected_attrs: list[str],
    expected_captions: dict[str, list[str]],
) -> None:
    """Test CaptionWindow.to_dict with various parameters.

    Args:
        last_caption_only: Whether to include only the last caption from each prompt variant
        attr_white_list: List of attributes to include in the result
        expected_attrs: List of attributes expected in the result
        expected_captions: Expected captions in the result (if applicable)

    """
    # Create a test caption window with multiple captions
    window = CaptionWindow(
        start_frame=0,
        end_frame=100,
        captions=expected_captions,  # {"default": ["caption1", "caption2"]},
        model_input={"default": {"input": "test input"}},
        t5_xxl_embeddings={},
    )

    result = window.to_dict(last_caption_only=last_caption_only, attr_white_list=attr_white_list)

    # Check that the result has the expected attributes
    for attr in expected_attrs:
        assert attr in result

    # Check that attributes not in the white list are excluded
    all_attrs = [
        "start_frame",
        "end_frame",
        "captions",
        "model_input",
        "t5_xxl_embeddings",
    ]
    for attr in all_attrs:
        if attr not in expected_attrs:
            assert attr not in result

    # Check captions if applicable
    if "captions" in result and expected_captions is not None:
        assert result["captions"] == expected_captions


@pytest.mark.parametrize(
    ("last_caption_only", "attr_white_list", "expected_attrs"),
    [
        # Default behavior - include all attributes except buffer
        (
            False,
            None,
            [
                "video_session_name",
                "clip_session_uuid",
                "uuid",
                "camera_id",
                "url",
                "caption_windows",
                "t5_xxl_embedding_urls",
            ],
        ),
        # Last caption only - include only the last caption from each window
        (
            True,
            None,
            [
                "video_session_name",
                "clip_session_uuid",
                "uuid",
                "camera_id",
                "url",
                "caption_windows",
                "t5_xxl_embedding_urls",
            ],
        ),
        # Custom attribute white list - include only specified attributes
        (
            False,
            ["video_session_name", "clip_session_uuid", "uuid", "camera_id"],
            ["video_session_name", "clip_session_uuid", "uuid", "camera_id"],
        ),
    ],
)
def test_clip_for_annotation_to_dict(
    *,
    last_caption_only: bool,
    attr_white_list: list[str] | None,
    expected_attrs: list[str],
) -> None:
    """Test ClipForAnnotation.to_dict with various parameters.

    Args:
        last_caption_only: Whether to include only the last caption from each window
        attr_white_list: List of attributes to include in the result
        expected_attrs: List of attributes expected in the result

    """
    # Create a test clip with multiple caption windows
    clip = ClipForAnnotation(
        video_session_name="test-session",
        clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
        uuid=UUID("12345678-1234-5678-1234-567812345678"),
        camera_id=1,
        span_index=0,
        url="test-url",
        caption_windows=[
            CaptionWindow(0, 100, captions={"default": ["caption1", "caption2"]}),
            CaptionWindow(0, 200, captions={"default": ["caption3", "caption4"]}),
        ],
        t5_xxl_embedding_urls={"default": "s3://test-bucket/embedding.json"},
    )

    result = clip.to_dict(last_caption_only=last_caption_only, attr_white_list=attr_white_list)

    # Check that the result has the expected attributes
    for attr in expected_attrs:
        assert attr in result

    # Check that attributes not in the white list are excluded
    all_attrs = [
        "session_name",
        "uuid",
        "camera_id",
        "span_index",
        "url",
        "caption_windows",
        "t5_xxl_embedding_urls",
        "buffer",
    ]
    for attr in all_attrs:
        if attr not in expected_attrs:
            assert attr not in result
