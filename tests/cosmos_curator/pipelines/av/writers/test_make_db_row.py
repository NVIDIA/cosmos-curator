# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Test the make_db_row module."""

from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import numpy as np
import pytest

from cosmos_curator.pipelines.av.utils.av_data_model import (
    CaptionWindow,
    ClipForAnnotation,
)

# Skip entire module if sqlalchemy is not available
sqlalchemy = pytest.importorskip("sqlalchemy", reason="sqlalchemy package is not installed")

from cosmos_curator.pipelines.av.utils.postgres_schema import ClipCaption  # noqa: E402
from cosmos_curator.pipelines.av.writers.make_db_row import (  # noqa: E402
    _get_caption_chain_len,
    _make_prompt_types,
    make_clip_caption,
)


@pytest.mark.parametrize(
    ("caption_chain_len", "prompt_type", "expected", "raises"),
    [
        # Test case 1: caption_chain_len = 4
        (
            4,
            "visibility",
            [
                (3, "visibility"),
                (2, "visibility_000"),
                (1, "visibility_001"),
                (0, "visibility_002"),
            ],
            does_not_raise(),
        ),
        # Test case 2: caption_chain_len = 1
        (1, "road_conditions", [(0, "road_conditions")], does_not_raise()),
        # Test case 3: caption_chain_len = 2
        (
            2,
            "illumination",
            [(1, "illumination"), (0, "illumination_000")],
            does_not_raise(),
        ),
        # Test case 4: caption_chain_len = 3
        (3, "test", [(2, "test"), (1, "test_000"), (0, "test_001")], does_not_raise()),
        # Failure cases
        (0, "test", [], pytest.raises(ValueError)),  # noqa: PT011
        (-1, "test", [], pytest.raises(ValueError)),  # noqa: PT011
    ],
)
def test_make_prompt_types(
    caption_chain_len: int,
    prompt_type: str,
    expected: list[tuple[int, str]],
    raises: AbstractContextManager[Any],
) -> None:
    """Test the _make_prompt_types generator function."""
    with raises:
        prompt_types = list(_make_prompt_types(caption_chain_len, prompt_type))
        assert len(prompt_types) == len(expected)
        assert prompt_types == expected


@pytest.mark.parametrize(
    ("clip", "prompt_type", "expected", "raises"),
    [
        # Test case 1: Empty clip (should raise ValueError)
        (
            ClipForAnnotation(
                video_session_name="test-session-1",
                clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                uuid=UUID("12345678-1234-5678-1234-567812345678"),
                camera_id=0,
                span_index=0,
                url="clip-url",
                caption_windows=[],
            ),
            "visibility",
            0,
            pytest.raises(ValueError, match="Clip has no caption windows"),
        ),
        # Test case 2: Single window with captions (should succeed)
        (
            ClipForAnnotation(
                video_session_name="test-session-2",
                clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                uuid=UUID("12345678-1234-5678-1234-567812345678"),
                camera_id=0,
                span_index=0,
                url="clip-url",
                caption_windows=[CaptionWindow(0, 100, captions={"visibility": ["caption1", "caption2"]})],
            ),
            "visibility",
            2,
            does_not_raise(),
        ),
        # Test case 3: Multiple windows with same number of captions (should succeed)
        (
            ClipForAnnotation(
                video_session_name="test-session-3",
                clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                uuid=UUID("12345678-1234-5678-1234-567812345678"),
                camera_id=0,
                span_index=0,
                url="clip-url",
                caption_windows=[
                    CaptionWindow(0, 100, captions={"visibility": ["caption1", "caption2"]}),
                    CaptionWindow(0, 100, captions={"visibility": ["caption3", "caption4"]}),
                ],
            ),
            "visibility",
            2,
            does_not_raise(),
        ),
        # Test case 4: Multiple windows with different number of captions (should raise ValueError)
        (
            ClipForAnnotation(
                video_session_name="test-session-4",
                clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                uuid=UUID("12345678-1234-5678-1234-567812345678"),
                camera_id=0,
                span_index=0,
                url="clip-url",
                caption_windows=[
                    CaptionWindow(0, 100, captions={"visibility": ["caption1", "caption2"]}),
                    CaptionWindow(0, 100, captions={"visibility": ["caption3"]}),
                ],
            ),
            "visibility",
            0,
            pytest.raises(
                ValueError,
                match="Caption windows have caption chains of different lengths",
            ),
        ),
        # Test case 5: Single window with no captions (should succeed)
        (
            ClipForAnnotation(
                video_session_name="test-session-5",
                clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                uuid=UUID("12345678-1234-5678-1234-567812345678"),
                camera_id=0,
                span_index=0,
                url="clip-url",
                caption_windows=[CaptionWindow(0, 100, captions={"visibility": []})],
            ),
            "visibility",
            0,
            does_not_raise(),
        ),
    ],
)
def test_get_caption_chain_len(
    clip: ClipForAnnotation,
    prompt_type: str,
    expected: int,
    raises: AbstractContextManager[Any],
) -> None:
    """Test the _get_caption_chain_len function."""
    with raises:
        result = _get_caption_chain_len(clip.caption_windows, prompt_type)
        assert result == expected


@pytest.mark.parametrize(
    (
        "clips",
        "version",
        "prompt_type",
        "run_uuid",
        "expected_caption_chain_len",
        "expected",
        "raises",
    ),
    [
        # 1 caption in chain, matches expected length
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session-6",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=0,
                    span_index=0,
                    url="clip-url",
                    caption_windows=[
                        CaptionWindow(
                            0,
                            100,
                            captions={"visibility": ["caption1-a"]},
                            t5_xxl_embeddings={"visibility": np.array([0, 1, 2], dtype=np.float32)},
                        ),
                        CaptionWindow(
                            0,
                            57,
                            captions={"visibility": ["caption2-a"]},
                            t5_xxl_embeddings={"visibility": np.array([0, 1, 2], dtype=np.float32)},
                        ),
                    ],
                    t5_xxl_embedding_urls={"visibility": "t5_embedding_url"},
                ),
            ],
            "v3",
            "visibility",
            UUID("12345678-1234-5678-1234-567812345678"),
            1,
            [
                ClipCaption(
                    clip_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    version="v3",
                    prompt_type="visibility",
                    window_start_frame=[0, 0],
                    window_end_frame=[100, 57],
                    window_caption=["caption1-a", "caption2-a"],
                    t5_embedding_url="t5_embedding_url",
                    run_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    datetime=datetime.now(UTC),
                ),
            ],
            does_not_raise(),
        ),
        # 2 captions in chain, matches expected length
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session-7",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=0,
                    span_index=0,
                    url="clip-url",
                    caption_windows=[
                        CaptionWindow(
                            0,
                            100,
                            captions={"visibility": ["caption1-b", "caption2-b"]},
                            t5_xxl_embeddings={"visibility": np.array([0, 1, 2], dtype=np.float32)},
                        ),
                        CaptionWindow(
                            0,
                            57,
                            captions={"visibility": ["caption3-b", "caption4-b"]},
                            t5_xxl_embeddings={"visibility": np.array([0, 1, 2], dtype=np.float32)},
                        ),
                    ],
                    t5_xxl_embedding_urls={"visibility": "t5_embedding_url"},
                ),
            ],
            "v3",
            "visibility",
            UUID("12345678-1234-5678-1234-567812345678"),
            2,
            [
                ClipCaption(
                    clip_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    version="v3",
                    prompt_type="visibility",
                    window_start_frame=[0, 0],
                    window_end_frame=[100, 57],
                    # Note that the captions are in reverse order of the caption chain.
                    # The last caption in the chain is the most important.
                    window_caption=["caption2-b", "caption4-b"],
                    t5_embedding_url="t5_embedding_url",
                    run_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    datetime=datetime.now(UTC),
                ),
                ClipCaption(
                    clip_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    version="v3",
                    prompt_type="visibility_000",
                    window_start_frame=[0, 0],
                    window_end_frame=[100, 57],
                    window_caption=["caption1-b", "caption3-b"],
                    t5_embedding_url="t5_embedding_url",
                    run_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    datetime=datetime.now(UTC),
                ),
            ],
            does_not_raise(),
        ),
        # 1 caption in chain but expected 2 - should yield no results
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session-8",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=0,
                    span_index=0,
                    url="clip-url",
                    caption_windows=[
                        CaptionWindow(
                            0,
                            100,
                            captions={"visibility": ["caption1-c"]},
                            t5_xxl_embeddings={"visibility": np.array([0, 1, 2], dtype=np.float32)},
                        ),
                    ],
                    t5_xxl_embedding_urls={"visibility": "t5_embedding_url"},
                ),
            ],
            "v3",
            "visibility",
            UUID("12345678-1234-5678-1234-567812345678"),
            2,
            [],
            does_not_raise(),
        ),
        # 2 captions in chain but expected 1 - should yield no results
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session-9",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=0,
                    span_index=0,
                    url="clip-url",
                    caption_windows=[
                        CaptionWindow(
                            0,
                            100,
                            captions={"visibility": ["caption1-d", "caption2-d"]},
                            t5_xxl_embeddings={"visibility": np.array([0, 1, 2], dtype=np.float32)},
                        ),
                    ],
                    t5_xxl_embedding_urls={"visibility": "t5_embedding_url"},
                ),
            ],
            "v3",
            "visibility",
            UUID("12345678-1234-5678-1234-567812345678"),
            1,
            [],
            does_not_raise(),
        ),
        # Clip with no t5_embedding when t5_embedding_stage is True, should yield 0 ClipCaptions
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session-11",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=0,
                    span_index=0,
                    url="clip-url",
                    caption_windows=[
                        CaptionWindow(
                            0,
                            100,
                            captions={"visibility": ["caption1-e"]},
                            t5_xxl_embeddings={},
                        ),
                    ],
                ),
            ],
            "v3",
            "visibility",
            UUID("12345678-1234-5678-1234-567812345678"),
            0,
            [],
            does_not_raise(),
        ),
        # Clip with no t5_embedding when t5_embedding_stage is False
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session-12",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=0,
                    span_index=0,
                    url="clip-url",
                    caption_windows=[
                        CaptionWindow(
                            0,
                            100,
                            captions={"visibility": ["caption1-f"]},
                            t5_xxl_embeddings={},
                        ),
                    ],
                ),
            ],
            "v3",
            "visibility",
            UUID("12345678-1234-5678-1234-567812345678"),
            1,
            [
                ClipCaption(
                    clip_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    version="v3",
                    prompt_type="visibility",
                    window_start_frame=[0],
                    window_end_frame=[100],
                    window_caption=["caption1-f"],
                    t5_embedding_url=None,
                    run_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    datetime=datetime.now(UTC),
                ),
            ],
            does_not_raise(),
        ),
        # Clip with inconsistent number of captions, should yield 0 ClipCaptions
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session-13",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    camera_id=0,
                    span_index=0,
                    url="clip-url",
                    caption_windows=[
                        CaptionWindow(
                            0,
                            100,
                            captions={"visibility": ["caption1-g", "caption2-g"]},
                            t5_xxl_embeddings={"visibility": np.array([0, 1, 2], dtype=np.float32)},
                        ),
                        CaptionWindow(
                            0,
                            57,
                            captions={"visibility": ["caption3-g"]},
                            t5_xxl_embeddings={"visibility": np.array([0, 1, 2], dtype=np.float32)},
                        ),
                    ],
                ),
            ],
            "v3",
            "visibility",
            UUID("12345678-1234-5678-1234-567812345678"),
            0,
            [],
            does_not_raise(),
        ),
    ],
)
def test_make_clip_caption(  # noqa: PLR0913
    clips: list[ClipForAnnotation],
    version: str,
    prompt_type: str,
    run_uuid: UUID,
    expected_caption_chain_len: int,
    expected: list[ClipCaption],
    raises: AbstractContextManager[Any],
) -> None:
    """Test the make_clip_caption function.

    Args:
        clips: List of clips to process
        version: Version string
        prompt_type: Type of prompt
        run_uuid: Run UUID
        expected_caption_chain_len: Expected length of caption chains
        expected: Expected ClipCaption objects
        raises: Expected exception context

    """
    with raises:
        results = list(
            make_clip_caption(
                clips=clips,
                version=version,
                prompt_type=prompt_type,
                run_uuid=run_uuid,
                expected_caption_chain_len=expected_caption_chain_len,
            )
        )
        assert len(results) == len(expected)
        for result, exp in zip(results, expected, strict=True):
            assert result.clip_uuid == exp.clip_uuid
            assert result.version == exp.version
            assert result.prompt_type == exp.prompt_type
            assert result.window_start_frame == exp.window_start_frame
            assert result.window_end_frame == exp.window_end_frame
            assert result.window_caption == exp.window_caption
            assert result.t5_embedding_url == exp.t5_embedding_url
            assert result.run_uuid == exp.run_uuid
            # Don't compare datetime as it's automatically set
