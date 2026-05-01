# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Module for creating database row objects.

This module provides functions for creating database row objects from various task types
and data models. These objects are used to write metadata to the database tables.
"""

import hashlib
import uuid
from collections.abc import Generator

import numpy as np
import numpy.typing as npt
from loguru import logger

from cosmos_curator.core.utils.data.ref_resolver import prefetch, resolve_as_ready
from cosmos_curator.pipelines.av.utils.av_data_model import (
    AvSessionTrajectoryTask,
    AvSessionVideoSplitTask,
    CaptionWindow,
    ClipForAnnotation,
)
from cosmos_curator.pipelines.av.utils.postgres_schema import (
    ClipCaption,
    ClippedSession,
    ClipTrajectory,
    SourceData,
    VideoSpan,
)
from cosmos_curator.pipelines.video.utils.decoder_utils import get_frame_count


def make_source_video_session(
    task: AvSessionVideoSplitTask,
) -> SourceData:
    """Create a SourceData row object from a video split task.

    Args:
        task: Video split task containing source video session information

    Returns:
        SourceData object for database insertion

    """
    return SourceData(
        session_name=task.source_video_session_name,
        version=task.source_video_version,
        session_url=task.session_url,
        num_cameras=task.num_cameras,
    )


def make_clipped_session(
    task: AvSessionVideoSplitTask,
    version: str,
    split_algo_name: str,
    encoder: str,
    run_uuid: uuid.UUID,
) -> ClippedSession:
    """Create a ClippedSession row object from a video split task.

    Args:
        task: Video split task containing session information
        version: Version string
        split_algo_name: Name of the algorithm used for splitting
        encoder: Name of the video encoder used
        run_uuid: UUID of the processing run

    Returns:
        ClippedSession object for database insertion

    """
    return ClippedSession(
        source_video_session_name=task.source_video_session_name,
        source_video_version=task.source_video_version,
        session_uuid=task.session_uuid,
        version=version,
        num_cameras=task.num_cameras,
        split_algo_name=split_algo_name,
        encoder=encoder,
        run_uuid=run_uuid,
    )


def _calculate_sha256(buffer: bytes | npt.NDArray[np.uint8]) -> str:
    """Calculate SHA-256 hash of a byte buffer.

    Args:
        buffer: Bytes to hash

    Returns:
        Hexadecimal string of the SHA-256 hash

    """
    return hashlib.sha256(buffer).hexdigest()


def make_video_spans(
    task: AvSessionVideoSplitTask,
    version: str,
    run_uuid: uuid.UUID,
) -> Generator[VideoSpan, None, None]:
    """Create VideoSpan row objects from a video split task.

    Args:
        task: Video split task containing video spans
        version: Version string
        run_uuid: UUID of the processing run

    Yields:
        VideoSpan objects for database insertion

    """
    for video in task.videos:
        prefetch([clip.encoded_data for clip in video.clips])
        for clip, data in resolve_as_ready([(clip, clip.encoded_data) for clip in video.clips]):
            if clip.timestamps_ms is None:
                error = f"Clip {clip.uuid} has no timestamps"
                raise ValueError(error)

            if data is None:
                error = f"Clip {clip.uuid} has no encoded_data"
                raise ValueError(error)

            yield VideoSpan(
                clip_uuid=clip.uuid,
                clip_session_uuid=clip.clip_session_uuid,
                version=version,
                source_video=video.source_video,
                camera_id=video.camera_id,
                session_uuid=task.session_uuid,
                span_index=clip.span_index,
                split_algo_name=task.split_algo_name,
                span_start=clip.span_start,
                span_end=clip.span_end,
                timestamps=clip.timestamps_ms.tobytes(),
                encoder=task.encoder,
                url=clip.url,
                byte_size=clip.encoded_data.nbytes,
                duration=clip.span_end - clip.span_start,
                framerate=video.metadata.framerate,
                num_frames=get_frame_count(data),
                height=video.metadata.height,
                width=video.metadata.width,
                sha256=_calculate_sha256(data),
                run_uuid=run_uuid,
            )


def _get_caption_chain_len(windows: list[CaptionWindow], prompt_type: str) -> int:
    """Get the length of caption chains in a list of caption windows.

    In a clip, there can be multiple caption windows. Each caption window can
    have a chain of captions. This function returns the length of the caption chain
    and verifies that all windows have the same chain length.

    Args:
        windows: List of caption windows to check
        prompt_type: Type of prompt to check captions for

    Returns:
        Length of the caption chains

    Raises:
        ValueError: If windows is empty or windows have different chain lengths

    """
    if not windows:
        error = f"Clip has no caption windows for prompt type {prompt_type}"
        raise ValueError(error)

    lengths = [len(window.captions[prompt_type]) for window in windows]

    if not all(length == lengths[0] for length in lengths):
        error = "Caption windows have caption chains of different lengths"
        raise ValueError(error)

    return lengths[0]


def _make_prompt_types(prompt_chain_len: int, prompt_type: str) -> Generator[tuple[int, str], None, None]:
    """Generate numbered prompt types for a chain of captions.

    The numbering ensures that when sorted alphabetically, the prompts appear
    in the correct order (most important / last one first).

    For example, with prompt_chain_len=4 and prompt_type="visibility":
        (3, "visibility")      # Last caption in the chain (most important)
        (2, "visibility_000")  # Third caption
        (1, "visibility_001")  # Second caption
        (0, "visibility_002")  # Firat caption (least important)

    When these are sorted alphabetically, they appear in the correct order:
        visibility
        visibility_000
        visibility_001
        visibility_002

    Args:
        prompt_chain_len: Number of captions in the chain
        prompt_type: Base name for the prompt type (e.g., "visibility", "road_conditions")

    Yields:
        Tuples of (index, prompt_type) where index is the position in the chain
        and prompt_type is the numbered prompt type string

    Raises:
        ValueError: If prompt_chain_len is not positive

    """
    if prompt_chain_len <= 0:
        error = "prompt_chain_len must be positive"
        raise ValueError(error)

    for i in reversed(range(prompt_chain_len)):
        _prompt_type = prompt_type
        if i < prompt_chain_len - 1:
            count = prompt_chain_len - 2 - i
            _prompt_type += f"_{count:03d}"
        yield i, _prompt_type


def make_clip_caption(  # noqa: PLR0913
    clips: list[ClipForAnnotation],
    version: str,
    prompt_type: str,
    run_uuid: uuid.UUID,
    expected_caption_chain_len: int,
    verbose: bool = False,  # noqa: FBT001, FBT002
) -> Generator[ClipCaption, None, None]:
    """Create ClipCaption row objects from a list of clips.

    Args:
        clips: List of clips with captions
        version: Version string
        prompt_type: Type of prompt for captions
        run_uuid: UUID of the processing run
        expected_caption_chain_len: Expected length of caption chains
        verbose: Whether to enable verbose logging

    Yields:
        ClipCaption objects for database insertion

    Raises:
        ValueError: If clips have inconsistent caption chain lengths

    """
    for clip in clips:
        windows = [window for window in clip.caption_windows if prompt_type in window.captions]

        try:
            caption_chain_len = _get_caption_chain_len(windows, prompt_type)
        except ValueError:
            logger.error(f"Clip {clip.uuid} has inconsistent number of captions across windows")
            continue

        if caption_chain_len != expected_caption_chain_len:
            logger.error(
                f"Clip {clip.uuid} {prompt_type} has fewer captions than expected: "
                f"{caption_chain_len} != {expected_caption_chain_len}"
            )
            continue

        for i, _prompt_type in _make_prompt_types(caption_chain_len, prompt_type):
            if verbose:
                logger.info(f"Making {i}-th row for {clip.uuid} prompt_type={_prompt_type}")
            yield ClipCaption(
                clip_uuid=clip.uuid,
                version=version,
                prompt_type=_prompt_type,
                window_start_frame=[window.start_frame for window in windows],
                window_end_frame=[window.end_frame for window in windows],
                window_caption=[window.captions[prompt_type][i] for window in windows],
                t5_embedding_url=clip.t5_xxl_embedding_urls.get(prompt_type, None),
                run_uuid=run_uuid,
            )


def make_clip_trajectory(
    task: AvSessionTrajectoryTask,
    version: str,
    run_uuid: str,
) -> Generator[ClipTrajectory, None, None]:
    """Create ClipTrajectory row objects from a trajectory task.

    Args:
        task: Trajectory task containing clips with trajectory data
        version: Version string
        run_uuid: UUID of the processing run

    Yields:
        ClipTrajectory objects for database insertion

    Raises:
        ValueError: If a clip has no trajectory data

    """
    for clip in task.clips:
        if clip.trajectory is None:
            logger.error(f"Clip {clip.uuid} has no trajectory")
            continue
        yield ClipTrajectory(
            clip_uuid=clip.uuid,
            version=version,
            trajectory_url=clip.trajectory_url,
            run_uuid=run_uuid,
        )
