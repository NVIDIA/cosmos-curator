# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Test the T5WriterStage class."""

import pathlib
import uuid
from unittest.mock import MagicMock, patch
from uuid import UUID

import numpy as np
import pytest

from cosmos_curator.core.utils.storage import s3_client
from cosmos_curator.pipelines.av.utils.av_data_model import (
    CaptionWindow,
    ClipForAnnotation,
)
from cosmos_curator.pipelines.av.writers.t5_writer_stage import (
    T5WriterStage,
    _get_t5_embedding_url,
    _write_t5_embeddings,
)
from tests.cosmos_curator.pipelines.av.writers.test_utils import (
    create_test_annotation_task,
    run_writer_stage_test,
)


@pytest.mark.parametrize(
    ("clips", "expected_calls", "expected_return"),
    [
        # Test case 1: Single clip with valid embeddings
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("a2345678-1234-5678-1234-567812345678"),
                    camera_id=1,
                    span_index=0,
                    url="test-url",
                    caption_windows=[
                        CaptionWindow(
                            start_frame=0,
                            end_frame=100,
                            t5_xxl_embeddings={"default": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
                        )
                    ],
                )
            ],
            1,  # Expected number of write_bytes calls
            1,  # Expected return value (number of clips processed)
        ),
        # Test case 2: Multiple clips with valid embeddings
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session-1",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("b2345678-1234-5678-1234-567812345678"),
                    camera_id=1,
                    span_index=0,
                    url="test-url-1",
                    caption_windows=[
                        CaptionWindow(
                            start_frame=0,
                            end_frame=100,
                            t5_xxl_embeddings={"default": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
                        )
                    ],
                ),
                ClipForAnnotation(
                    video_session_name="test-session-2",
                    clip_session_uuid=UUID("87654321-4321-8765-4321-876543210987"),
                    uuid=UUID("c7654321-4321-8765-4321-876543210987"),
                    camera_id=2,
                    span_index=0,
                    url="test-url-2",
                    caption_windows=[
                        CaptionWindow(
                            start_frame=0,
                            end_frame=100,
                            t5_xxl_embeddings={"default": np.array([4.0, 5.0, 6.0], dtype=np.float32)},
                        )
                    ],
                ),
            ],
            2,  # Expected number of write_bytes calls
            2,  # Expected return value (number of clips processed)
        ),
        # Test case 3: Clip with missing embeddings
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("d2345678-1234-5678-1234-567812345678"),
                    camera_id=1,
                    span_index=0,
                    url="test-url",
                    caption_windows=[
                        CaptionWindow(
                            start_frame=0,
                            end_frame=100,
                            t5_xxl_embeddings={},
                        )
                    ],
                )
            ],
            0,  # Expected number of write_bytes calls
            0,  # Expected return value (number of clips processed)
        ),
        # Test case 4: Clip with multiple windows, one missing embedding
        (
            [
                ClipForAnnotation(
                    video_session_name="test-session",
                    clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
                    uuid=UUID("e2345678-1234-5678-1234-567812345678"),
                    camera_id=1,
                    span_index=0,
                    url="test-url",
                    caption_windows=[
                        CaptionWindow(
                            start_frame=0,
                            end_frame=100,
                            t5_xxl_embeddings={"default": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
                        ),
                        CaptionWindow(
                            start_frame=100,
                            end_frame=200,
                            t5_xxl_embeddings={},
                        ),
                    ],
                )
            ],
            0,  # Expected number of write_bytes calls
            0,  # Expected return value (number of clips processed)
        ),
        # Test case 5: Empty clips list
        (
            [],
            0,  # Expected number of write_bytes calls
            0,  # Expected return value (number of clips processed)
        ),
    ],
)
def test_write_t5_embeddings(clips: list[ClipForAnnotation], expected_calls: int, expected_return: int) -> None:
    """Test the _write_t5_embeddings function with various scenarios.

    Args:
        clips: List of ClipForAnnotation objects to test with
        expected_calls: Expected number of write_bytes calls
        expected_return: Expected return value from the function

    """
    mock_s3_client = MagicMock()
    mock_write_bytes = MagicMock()
    mock_get_t5_embedding_url = MagicMock(return_value=pathlib.Path("test/path/embedding.bin"))
    mock_pickle_dump = MagicMock()

    output_prefix = "test-prefix"
    clip_prefix = "test-clip-prefix"
    env = "test-env"
    version = "test-version"
    prompt_variant = "default"
    verbose = False

    with (
        patch(
            "cosmos_curator.pipelines.av.writers.t5_writer_stage.write_bytes",
            mock_write_bytes,
        ),
        patch(
            "cosmos_curator.pipelines.av.writers.t5_writer_stage._get_t5_embedding_url",
            mock_get_t5_embedding_url,
        ),
        patch(
            "cosmos_curator.pipelines.av.writers.t5_writer_stage.pickle.dump",
            mock_pickle_dump,
        ),
        patch("cosmos_curator.pipelines.av.writers.t5_writer_stage.io.BytesIO") as mock_bytesio,
    ):
        mock_buffer = MagicMock()
        mock_buffer.getvalue.return_value = b"test-buffer"
        mock_bytesio.return_value = mock_buffer

        result = _write_t5_embeddings(
            mock_s3_client,
            clips,
            output_prefix,
            clip_prefix,
            env,
            version,
            prompt_variant,
            verbose,
        )

        assert result == expected_return
        assert mock_write_bytes.call_count == expected_calls
        assert mock_get_t5_embedding_url.call_count == expected_calls
        assert mock_pickle_dump.call_count == expected_calls

        if expected_calls > 0:
            for clip in clips:
                # Check that get_t5_embedding_url was called with the correct arguments for this clip
                mock_get_t5_embedding_url.assert_any_call(
                    clip.uuid,
                    output_prefix,
                    env,
                    clip_prefix,
                    prompt_variant,
                    version,
                )

                # Check that write_bytes was called with the correct arguments for this clip
                mock_write_bytes.assert_any_call(
                    b"test-buffer",
                    pathlib.Path("test/path/embedding.bin"),
                    f"clip-{clip.uuid}",
                    "unknown",
                    verbose=verbose,
                    client=mock_s3_client,
                    overwrite=True,
                )


@pytest.mark.parametrize(
    ("prefix", "expected_type"),
    [
        ("/local/path", pathlib.Path),
        ("s3://bucket/path", s3_client.S3Prefix),
    ],
)
def test_get_t5_embedding_url(prefix: str, expected_type: type) -> None:
    """Test the _get_t5_embedding_url function returns the correct type based on the prefix.

    Args:
        prefix: The prefix to test (local or S3 path)
        expected_type: The expected return type (pathlib.Path or S3Prefix)

    """
    # Test parameters
    clip_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
    db_type = "test-db"
    clip_prefix = "test-clip"
    prompt_variant = "test-prompt"
    version = "test-version"

    # Call the function with the test parameters
    result = _get_t5_embedding_url(clip_uuid, prefix, db_type, clip_prefix, prompt_variant, version)

    # Verify the return type
    assert isinstance(result, expected_type)


@pytest.mark.parametrize(
    "output_prefix",
    [
        "/test/path",  # Local path
        "s3://bucket/test/path",  # S3 path
    ],
)
def test_t5_writer_stage(output_prefix: str) -> None:
    """Test the T5WriterStage class happy path with both local and S3 paths.

    Args:
        output_prefix: The output prefix to test (local or S3 path)

    """
    env = "test"
    run_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
    version = "v9000"
    prompt_variant = "default"
    verbose = False

    stage = T5WriterStage(
        env=env,
        output_prefix=output_prefix,
        run_id=run_id,
        version=version,
        prompt_variants=[prompt_variant],
        verbose=verbose,
        log_stats=True,
    )
    tasks = [create_test_annotation_task()]

    run_writer_stage_test(
        stage=stage,
        tasks=tasks,
        writer_function_path="cosmos_curator.pipelines.av.writers.t5_writer_stage._write_t5_embeddings",
    )
