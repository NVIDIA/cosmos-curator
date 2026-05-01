# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Test utilities for the AV writer stages."""

import uuid
from unittest.mock import MagicMock, patch
from uuid import UUID

import numpy as np

from cosmos_curator.core.interfaces.stage_interface import CuratorStage
from cosmos_curator.core.utils.storage.s3_client import is_s3path
from cosmos_curator.pipelines.av.utils.av_data_model import (
    AvClipAnnotationTask,
    CaptionWindow,
    ClipForAnnotation,
)


def create_mock_clip(
    clip_uuid: UUID | None = None,
    camera_id: int = 1,
    video_session_name: str = "test_session",
    clip_session_uuid: UUID | None = None,
    url: str | None = None,
) -> MagicMock:
    """Create a mock clip for testing.

    Args:
        clip_uuid: UUID for the clip, or None to generate a random one
        camera_id: Camera ID for the clip
        video_session_name: Session name for the clip
        clip_session_uuid: UUID for the clip session
        url: URL for the clip

    Returns:
        A MagicMock object configured as a ClipForAnnotation

    """
    if clip_uuid is None:
        clip_uuid = uuid.uuid4()

    if clip_session_uuid is None:
        clip_session_uuid = UUID("12345678-1234-5678-1234-567812345678")

    if url is None:
        url = "http://example.com/video.mp4"

    mock_clip = MagicMock(spec=ClipForAnnotation)
    mock_clip.uuid = clip_uuid
    mock_clip.camera_id = camera_id
    mock_clip.video_session_name = video_session_name
    mock_clip.clip_session_uuid = clip_session_uuid
    mock_clip.url = url
    mock_clip.to_dict.return_value = {
        "camera_id": camera_id,
        "caption_windows": ["window1", "window2"],
        "video_session_name": video_session_name,
        "clip_session_uuid": clip_session_uuid,
        "uuid": clip_uuid,
        "extra_field": "should be filtered out",
    }

    return mock_clip


def create_test_annotation_task(*, single_clip: bool = False) -> AvClipAnnotationTask:
    """Create a test task with two clips containing T5 embeddings.

    Returns:
        AvClipAnnotationTask: A test task with two clips.

    """
    # Create test clips with embeddings
    clip1 = ClipForAnnotation(
        video_session_name="test-session-1",
        clip_session_uuid=UUID("12345678-1234-5678-1234-567812345678"),
        uuid=UUID("12345678-1234-5678-1234-567812345678"),
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
    )

    clip2 = ClipForAnnotation(
        video_session_name="test-session-2",
        clip_session_uuid=UUID("87654321-4321-8765-4321-876543210987"),
        uuid=UUID("87654321-4321-8765-4321-876543210987"),
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
    )

    # Create and return a task with the test clips
    if single_clip:
        return AvClipAnnotationTask(clips=[clip1])

    return AvClipAnnotationTask(clips=[clip1, clip2])


def run_writer_stage_test(
    stage: CuratorStage,
    tasks: list[AvClipAnnotationTask],
    writer_function_path: str,
) -> None:
    """Test a writer stage with a common pattern, supporting both S3 and local paths.

    Args:
        stage:  the stage to test
        tasks: The tasks to process, containing clips to be written
        writer_function_path: The path to the writer function to mock
        expected_writer_args: Function that returns the expected positional arguments for the writer function
        expected_writer_kwargs: Optional function that returns the expected keyword arguments for the writer function

    """
    assert hasattr(stage, "_output_prefix"), "Stage must have an _output_prefix attribute"
    assert hasattr(stage, "_log_stats"), "Stage must have a _log_stats attribute"

    # Determine if we're testing S3 or local path based on output_prefix
    output_prefix = stage._output_prefix
    is_s3 = is_s3path(output_prefix)

    # Create mocks for the writer function and S3 client
    with (
        patch(writer_function_path) as mock_writer,
        patch("cosmos_curator.core.utils.storage.s3_client.create_s3_client") as mock_create_s3_client,
        patch("loguru.logger"),
    ):
        # Configure mocks
        mock_s3_client = MagicMock()
        mock_create_s3_client.return_value = mock_s3_client
        mock_writer.return_value = [len(task.clips) for task in tasks]

        stage.stage_setup()

        # Verify S3 client was created correctly if using S3.
        # This is called even if we're not using S3, but none is returned
        if is_s3:
            mock_create_s3_client.assert_called_once_with(
                target_path=output_prefix,
                can_overwrite=True,
            )

        # Call process_data
        result = stage.process_data(tasks)  # type: ignore[arg-type]
        assert len(mock_writer.mock_calls) == len(tasks)
        assert result == tasks

        if stage._log_stats:
            for task in tasks:
                assert stage.__class__.__name__ in task.stage_perf
