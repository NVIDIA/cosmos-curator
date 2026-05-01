# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Tests for clip_writer_stage module."""

import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from cosmos_curator.core.utils.db.database_types import EnvType, PostgresDB
from cosmos_curator.core.utils.storage.s3_client import S3Prefix
from cosmos_curator.pipelines.av.utils.av_data_model import (
    AvClipAnnotationTask,
    AvSessionVideoSplitTask,
    AvVideo,
)
from cosmos_curator.pipelines.av.utils.postgres_schema import ClippedSession
from cosmos_curator.pipelines.av.writers import clip_writer_stage
from cosmos_curator.pipelines.av.writers.clip_writer_stage import ClipWriterStage
from tests.cosmos_curator.pipelines.av.conftest import RecordingSession

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


@pytest.fixture
def fake_video_span(monkeypatch: pytest.MonkeyPatch) -> object:
    """Replace make_video_spans with a lightweight generator for DB tests."""
    span = object()

    def _fake_make_video_spans(*args: object, **kwargs: object) -> Iterator[object]:  # noqa: ARG001
        yield span

    monkeypatch.setattr(clip_writer_stage, "make_video_spans", _fake_make_video_spans)
    return span


def test_get_clip_url_for_local_path(tmp_path: Path) -> None:
    """_get_clip_url should return a filesystem path when prefix is local."""
    stage = ClipWriterStage(
        db=None,
        output_prefix=str(tmp_path / "outputs/"),
        run_id=uuid.uuid4(),
        version="v9000",
        continue_captioning=False,
        caption_chunk_size=1,
    )
    clip_uuid = uuid.uuid4()

    dest = stage._get_clip_url(clip_uuid, encoder="h264")
    assert isinstance(dest, Path)

    expected = tmp_path / "outputs" / "raw_clips" / f"{clip_uuid}.mp4"
    assert dest == expected


def test_get_clip_url_for_db_env_includes_layout() -> None:
    """_get_clip_url should add env, encoder, and version when DB is configured."""
    stage = ClipWriterStage(
        db=PostgresDB(EnvType.DEV, "endpoint", "db", "user", "pass"),
        output_prefix="s3://bucket/root",
        run_id=uuid.uuid4(),
        version="v1",
        continue_captioning=False,
        caption_chunk_size=1,
    )
    clip_uuid = uuid.uuid4()

    dest = stage._get_clip_url(clip_uuid, encoder="vp9")
    assert isinstance(dest, S3Prefix)

    expected = f"s3://bucket/root/dev/raw_clips/vp9/v1/{clip_uuid}.mp4"
    assert dest.path == expected


def test_upload_clips_writes_files(
    tmp_path: Path,
    video_factory: Callable[..., AvVideo],
) -> None:
    """_upload_clips should persist clip payloads and stamp URLs."""
    stage = ClipWriterStage(
        db=None,
        output_prefix=str(tmp_path),
        run_id=uuid.uuid4(),
        version="v1",
        continue_captioning=False,
        caption_chunk_size=2,
    )
    stage.stage_setup()
    video = video_factory(num_clips=2, camera_id=0)

    uploaded = stage._upload_clips(video, encoder="h264")

    assert uploaded == 2
    for clip in video.clips:
        assert clip.url is not None
        clip_path = Path(clip.url)
        assert clip_path.exists()
        assert clip_path.read_bytes() == clip.encoded_data.resolve().tobytes()


def test_upload_clips_requires_payload(
    tmp_path: Path,
    video_factory: Callable[..., AvVideo],
) -> None:
    """_upload_clips should raise when a clip lacks encoded bytes."""
    stage = ClipWriterStage(
        db=None,
        output_prefix=str(tmp_path),
        run_id=uuid.uuid4(),
        version="v1",
        continue_captioning=False,
        caption_chunk_size=2,
    )
    stage.stage_setup()
    video = video_factory(num_clips=1, camera_id=0)
    video.clips[0].encoded_data.drop()

    with pytest.raises(ValueError, match="has no encoded_data"):
        stage._upload_clips(video, encoder="h264")


def test_chunk_clips_for_captioning_respects_chunk_size(
    split_task_factory: Callable[..., AvSessionVideoSplitTask],
) -> None:
    """_chunk_clips_for_captioning should batch clips and propagate metadata."""
    stage = ClipWriterStage(
        db=None,
        output_prefix="/unused",
        run_id=uuid.uuid4(),
        version="v1",
        continue_captioning=True,
        caption_chunk_size=2,
    )
    task = split_task_factory(clip_counts=(3,), session_name="session/")
    for video in task.videos:
        for clip in video.clips:
            clip.url = f"file://{clip.uuid}"

    caption_tasks = stage._chunk_clips_for_captioning([task])

    assert len(caption_tasks) == 2
    assert caption_tasks[0].video_session_name == "session"
    assert caption_tasks[0].num_session_chunks == 2
    assert caption_tasks[0].session_chunk_index == 0
    assert caption_tasks[1].session_chunk_index == 1
    assert len(caption_tasks[0].clips) == 2
    assert len(caption_tasks[1].clips) == 1
    assert caption_tasks[0].source_video_duration_s == task.source_video_duration_s
    assert caption_tasks[1].source_video_duration_s == 0
    last_video = task.videos[-1]
    assert caption_tasks[0].height == last_video.metadata.height
    assert caption_tasks[0].width == last_video.metadata.width
    assert caption_tasks[0].framerate == last_video.metadata.framerate


def test_process_data_without_captioning_clears_buffers(
    tmp_path: Path,
    split_task_factory: Callable[..., AvSessionVideoSplitTask],
) -> None:
    """process_data should drop encoded bytes when captioning is disabled."""
    stage = ClipWriterStage(
        db=None,
        output_prefix=str(tmp_path),
        run_id=uuid.uuid4(),
        version="v1",
        continue_captioning=False,
        caption_chunk_size=2,
    )
    stage.stage_setup()
    task = split_task_factory(clip_counts=(1,))
    clip = task.videos[0].clips[0]
    original_payload = clip.encoded_data.resolve()

    processed = stage.process_data([task])

    assert processed is not None
    assert all(isinstance(item, AvSessionVideoSplitTask) for item in processed)
    split_tasks = [cast("AvSessionVideoSplitTask", item) for item in processed]
    assert split_tasks[0] is task
    assert clip.encoded_data.resolve() is None
    assert clip.timestamps_ms is None
    assert clip.url is not None
    assert Path(clip.url).read_bytes() == original_payload.tobytes()


def test_process_data_with_captioning_returns_annotation_tasks(
    tmp_path: Path,
    split_task_factory: Callable[..., AvSessionVideoSplitTask],
) -> None:
    """process_data should emit AvClipAnnotationTask instances when enabled."""
    stage = ClipWriterStage(
        db=None,
        output_prefix=str(tmp_path),
        run_id=uuid.uuid4(),
        version="v1",
        continue_captioning=True,
        caption_chunk_size=1,
    )
    stage.stage_setup()
    task = split_task_factory(clip_counts=(2,), session_name="session/")
    original_payloads = [clip.encoded_data.resolve() for clip in task.videos[0].clips]

    caption_tasks = stage.process_data([task])

    assert caption_tasks is not None
    assert all(isinstance(task_item, AvClipAnnotationTask) for task_item in caption_tasks)
    annotation_tasks = [cast("AvClipAnnotationTask", item) for item in caption_tasks]
    assert len(annotation_tasks) == 2
    assert annotation_tasks[0].video_session_name == "session"
    assert annotation_tasks[0].session_chunk_index == 0
    assert annotation_tasks[1].session_chunk_index == 1
    assert annotation_tasks[0].source_video_duration_s == task.source_video_duration_s
    assert annotation_tasks[1].source_video_duration_s == 0
    assert annotation_tasks[0].clips[0].url == task.videos[0].clips[0].url
    assert np.array_equal(task.videos[0].clips[0].encoded_data.resolve(), original_payloads[0])
    assert np.array_equal(task.videos[0].clips[1].encoded_data.resolve(), original_payloads[1])


def test_write_data_persists_video_spans_and_clipped_session(
    fake_video_span: object,
    split_task_factory: Callable[..., AvSessionVideoSplitTask],
    recording_session: RecordingSession,
) -> None:
    """write_data should store generated spans and clipped session rows."""
    stage = ClipWriterStage(
        db=None,
        output_prefix="/unused",
        run_id=uuid.uuid4(),
        version="v42",
        continue_captioning=False,
        caption_chunk_size=1,
    )
    task = split_task_factory()

    stage.write_data(cast("Session", recording_session), task)

    assert recording_session.objects is not None
    assert recording_session.objects[0] is fake_video_span
    clipped_session = recording_session.objects[1]
    assert isinstance(clipped_session, ClippedSession)
    assert clipped_session.source_video_session_name == task.source_video_session_name
    assert clipped_session.version == "v42"
    assert recording_session.commit_count == 1


@pytest.mark.parametrize(
    ("field_name", "message"),
    [
        ("split_algo_name", "Split algorithm name is required"),
        ("encoder", "Encoder is required"),
    ],
)
def test_write_data_validates_required_fields(
    field_name: str,
    message: str,
    fake_video_span: object,  # noqa: ARG001
    split_task_factory: Callable[..., AvSessionVideoSplitTask],
) -> None:
    """write_data should raise if required metadata is missing."""
    stage = ClipWriterStage(
        db=None,
        output_prefix="/unused",
        run_id=uuid.uuid4(),
        version="v1",
        continue_captioning=False,
        caption_chunk_size=1,
    )
    task = split_task_factory()
    setattr(task, field_name, None)

    with pytest.raises(ValueError, match=message):
        stage.write_data(cast("Session", RecordingSession()), task)
