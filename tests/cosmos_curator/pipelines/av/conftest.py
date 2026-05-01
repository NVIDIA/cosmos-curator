# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Shared fixtures for AV pipeline tests."""

import uuid
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.pipelines.av.utils.av_data_model import (
    AvSessionVideoSplitTask,
    AvVideo,
    ClipForTranscode,
    VideoMetadata,
)


class RecordingSession:
    """Minimal SQLAlchemy-like session that records bulk operations."""

    def __init__(self) -> None:
        """Initialize tracking state."""
        self.objects: list[Any] | None = None
        self.commit_count = 0

    def bulk_save_objects(self, objects: list[object]) -> None:
        """Record the provided objects."""
        self.objects = list(objects)

    def commit(self) -> None:
        """Record that commit was invoked."""
        self.commit_count += 1


@pytest.fixture
def clip_factory() -> Callable[..., ClipForTranscode]:
    """Create ClipForTranscode instances."""
    _sentinel = object()

    def _make(
        span_index: int,
        *,
        payload: bytes | None | object = _sentinel,
        span_start: float | None = None,
        span_end: float | None = None,
        timestamps_ms: npt.NDArray[np.int64] | None = None,
    ) -> ClipForTranscode:
        raw = cast("bytes | None", f"clip-{span_index}".encode() if payload is _sentinel else payload)
        data = bytes_to_numpy(raw) if raw is not None else None
        timestamps = timestamps_ms if timestamps_ms is not None else np.array([0, 33, 66], dtype=np.int64)
        return ClipForTranscode(
            uuid=uuid.uuid4(),
            clip_session_uuid=uuid.uuid4(),
            span_index=span_index,
            span_start=span_start if span_start is not None else float(span_index),
            span_end=span_end if span_end is not None else float(span_index + 1),
            timestamps_ms=timestamps,
            encoded_data=data,  # type: ignore[assignment]
        )

    return _make


@pytest.fixture
def video_factory(clip_factory: Callable[..., ClipForTranscode]) -> Callable[..., AvVideo]:
    """Create AvVideo instances."""

    def _make(  # noqa: PLR0913
        num_clips: int = 0,
        *,
        camera_id: int = 0,
        metadata: VideoMetadata | None = None,
        source_video: str | None = None,
        encoded_data: npt.NDArray[np.uint8] | None = None,
        timestamps_ms: npt.NDArray[np.int64] | None = None,
        framerate: float | None = None,
        num_frames: int | None = None,
        duration: float | None = None,
        height: int | None = None,
        width: int | None = None,
        size: int | None = None,
        clips: list[ClipForTranscode] | None = None,
    ) -> AvVideo:
        clip_list = (
            clips if clips is not None else [clip_factory(span_index=camera_id * 10 + idx) for idx in range(num_clips)]
        )
        effective_framerate = framerate if framerate is not None else 30.0 + camera_id
        effective_num_frames = num_frames if num_frames is not None else 900
        effective_duration = duration
        if effective_duration is None:
            effective_duration = effective_num_frames / effective_framerate if effective_framerate else 0.0
        video_metadata = metadata or VideoMetadata(
            size=size if size is not None else 4096,
            height=height if height is not None else 720 + camera_id,
            width=width if width is not None else 1280 + camera_id,
            framerate=effective_framerate,
            num_frames=effective_num_frames,
            duration=effective_duration,
        )
        timestamp_array = timestamps_ms
        if timestamp_array is None and effective_framerate:
            frame_interval_ms = int(1_000 / effective_framerate)
            timestamp_array = np.arange(0, effective_num_frames * frame_interval_ms, frame_interval_ms, dtype=np.int64)
        return AvVideo(
            source_video=source_video or f"video-{camera_id}.mp4",
            camera_id=camera_id,
            encoded_data=encoded_data,
            metadata=video_metadata,
            timestamps_ms=timestamp_array,
            clips=clip_list,
        )

    return _make


@pytest.fixture
def split_task_factory(video_factory: Callable[..., AvVideo]) -> Callable[..., AvSessionVideoSplitTask]:
    """Create AvSessionVideoSplitTask instances."""

    def _make(  # noqa: PLR0913
        *,
        clip_counts: tuple[int, ...] = (1,),
        session_name: str = "demo-session/",
        session_url: str | None = None,
        split_algo_name: str = "algo",
        encoder: str = "hevc",
        videos: list[AvVideo] | None = None,
    ) -> AvSessionVideoSplitTask:
        task_videos = (
            videos
            if videos is not None
            else [video_factory(count, camera_id=idx) for idx, count in enumerate(clip_counts)]
        )
        return AvSessionVideoSplitTask(
            source_video_session_name=session_name,
            source_video_version="v1",
            session_uuid=uuid.uuid4(),
            session_url=session_url or f"memory://{uuid.uuid4()}",
            num_cameras=len(task_videos),
            videos=task_videos,
            split_algo_name=split_algo_name,
            encoder=encoder,
        )

    return _make


@pytest.fixture
def recording_session() -> RecordingSession:
    """Provide a RecordingSession for DB writer tests."""
    return RecordingSession()
