# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Module for writing video clips and metadata to storage and database.

This module provides stages for writing video clips to storage (local or S3) and
their metadata to a PostgreSQL database. It includes stages for source video ingestion
and clip writing with support for concurrent uploads.
"""

import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

from loguru import logger
from sqlalchemy.orm import Session

from cosmos_curator.core.utils.data.ref_resolver import prefetch, resolve_as_ready
from cosmos_curator.core.utils.db.database_types import PostgresDB
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.core.utils.misc.grouping import split_by_chunk_size
from cosmos_curator.core.utils.storage import s3_client
from cosmos_curator.core.utils.storage.s3_client import is_s3path
from cosmos_curator.core.utils.storage.writer_utils import write_bytes
from cosmos_curator.pipelines.av.utils.av_data_model import (
    AvClipAnnotationTask,
    AvSessionVideoIngestTask,
    AvSessionVideoSplitTask,
    AvVideo,
    ClipForAnnotation,
)
from cosmos_curator.pipelines.av.utils.postgres_schema import (
    ClippedSession,
    SourceData,
    VideoSpan,
)
from cosmos_curator.pipelines.av.writers.base_writer_stage import BaseWriterStage
from cosmos_curator.pipelines.av.writers.make_db_row import (
    make_clipped_session,
    make_source_video_session,
    make_video_spans,
)


class SourceVideoIngestionStage(BaseWriterStage):
    """Stage for writing source video metadata to database.

    This stage handles writing source video session metadata to a PostgreSQL database.
    """

    def __init__(
        self,
        db: PostgresDB,
        max_retries: int = 1,
        verbose: bool = False,  # noqa: FBT001, FBT002
        log_stats: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize a stage that writes source video metadata to database.

        Args:
            db: PostgreSQL database configuration
            max_retries: Maximum number of retry attempts for database operations
            verbose: Whether to enable verbose logging
            log_stats: Whether to log performance statistics

        """
        super().__init__(
            db=db,
            max_retries=max_retries,
        )
        self._timer = StageTimer(self)
        self._verbose = verbose
        self._log_stats = log_stats

    def stage_setup(self) -> None:
        """Set up database connection."""
        super().stage_setup()

    def write_data(  # type: ignore[override]
        self, session: Session, task: AvSessionVideoIngestTask
    ) -> list[AvSessionVideoIngestTask] | None:
        """Write source video metadata to database.

        Args:
            session: SQLAlchemy database session
            task: Task containing source video metadata to write

        Returns:
            List containing the input task if successful

        Raises:
            SQLAlchemyError: If database write fails

        """
        objects = [make_source_video_session(video_session) for video_session in task.sessions]
        logger.info(f"Writing {len(objects)} rows to {SourceData.__tablename__}.")
        session.bulk_save_objects(objects)
        session.commit()
        return [task]


class ClipWriterStage(BaseWriterStage):
    """Stage for writing video clips to storage and metadata to database.

    This stage handles writing video clips to storage (local or S3) and their metadata
    to a PostgreSQL database. It supports concurrent uploads and optional continuation
    to captioning stages.
    """

    def __init__(  # noqa: PLR0913
        self,
        db: PostgresDB | None,
        output_prefix: str,
        run_id: uuid.UUID,
        version: str,
        continue_captioning: bool,  # noqa: FBT001
        caption_chunk_size: int,
        max_retries: int = 1,
        verbose: bool = False,  # noqa: FBT001, FBT002
        log_stats: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize a stage that writes metadata for a split run.

        Args:
            db: PostgreSQL database configuration, or None if no database writes needed
            output_prefix: Base path for output files
            run_id: Unique identifier for this run
            version: Version string
            continue_captioning: Whether to prepare tasks for captioning stages
            caption_chunk_size: Size of chunks for captioning tasks
            max_retries: Maximum number of retry attempts for database operations
            verbose: Whether to enable verbose logging
            log_stats: Whether to log performance statistics

        """
        if db is not None:
            super().__init__(
                db=db,
                max_retries=max_retries,
            )
        self._db_type = db.env_type.value if db is not None else None
        self._timer = StageTimer(self)
        self._output_prefix = output_prefix.rstrip("/")
        self._run_id = run_id
        self._version = version
        self._continue_captioning = continue_captioning
        self._caption_chunk_size = caption_chunk_size
        self._verbose = verbose
        self._log_stats = log_stats
        self._clip_prefix = "raw_clips"

    def stage_setup(self) -> None:
        """Set up database connection and S3 client."""
        if self._db_type is not None:
            super().stage_setup()
        self._s3_client = s3_client.create_s3_client(target_path=self._output_prefix)

    def _get_clip_url(self, video_span_uuid: uuid.UUID, encoder: str) -> s3_client.S3Prefix | Path:
        """Generate a URL or path for storing video clip data.

        Args:
            video_span_uuid: UUID of the video span
            encoder: Name of the video encoder used

        Returns:
            An S3Prefix object if the output_prefix is an S3 path, otherwise a
            pathlib.Path object

        """
        if self._db_type is None:
            full_path = f"{self._output_prefix}/{self._clip_prefix}/{video_span_uuid}.mp4"
        else:
            full_path = (
                f"{self._output_prefix}/{self._db_type}/{self._clip_prefix}/{encoder}"
                f"/{self._version}/{video_span_uuid}.mp4"
            )
        if is_s3path(self._output_prefix):
            return s3_client.S3Prefix(full_path)
        return Path(full_path)

    def write_data(  # type: ignore[override]
        self, session: Session, task: AvSessionVideoSplitTask
    ) -> list[AvSessionVideoSplitTask] | None:
        """Write video clip metadata to database.

        Args:
            session: SQLAlchemy database session
            task: Task containing video clip metadata to write

        Returns:
            None if successful

        Raises:
            SQLAlchemyError: If database write fails

        """
        objects: list[VideoSpan | ClippedSession] = list(make_video_spans(task, self._version, self._run_id))
        num_clips = len(objects)
        logger.info(f"Writing {num_clips} rows to {VideoSpan.__tablename__}.")

        if task.split_algo_name is None:
            error = "Split algorithm name is required"
            raise ValueError(error)
        if task.encoder is None:
            error = "Encoder is required"
            raise ValueError(error)

        objects.append(make_clipped_session(task, self._version, task.split_algo_name, task.encoder, self._run_id))
        num_video_sessions = len(objects) - num_clips
        logger.info(f"Writing {num_video_sessions} rows to {ClippedSession.__tablename__}.")
        session.bulk_save_objects(objects)
        session.commit()
        return None

    def _upload_clips(self, video: AvVideo, encoder: str) -> int:
        """Upload video clips to storage.

        Args:
            video: Video containing clips to upload
            encoder: Name of the video encoder used

        Returns:
            Number of clips successfully uploaded

        Raises:
            ValueError: If a clip has no encoded_data

        """
        num_uploaded_clips = 0
        prefetch([clip.encoded_data for clip in video.clips])
        for clip, data in resolve_as_ready([(clip, clip.encoded_data) for clip in video.clips]):
            if data is None:
                error = f"Clip-{clip.span_index} from {video.source_video} has no encoded_data"
                raise ValueError(error)
            dest = self._get_clip_url(clip.uuid, encoder)
            clip.url = str(dest)
            write_bytes(
                data,
                dest,
                f"clip-{clip.span_index}",
                video.source_video,
                verbose=self._verbose,
                client=self._s3_client,
            )
            num_uploaded_clips += 1
        logger.info(f"Uploaded {len(video.clips)} clips from {video.source_video}")
        return num_uploaded_clips

    def process_data(  # type: ignore[override]
        self, tasks: list[AvSessionVideoSplitTask]
    ) -> list[AvSessionVideoSplitTask | AvClipAnnotationTask] | None:
        """Process and write video clips with performance tracking.

        This method uploads video clips to storage using multiple threads and optionally
        writes metadata to the database. If continuation to captioning is enabled,
        it prepares captioning tasks.

        Args:
            tasks: Tasks containing video clips to process

        Returns:
            List of captioning tasks if continue_captioning is True, None otherwise

        Raises:
            Exception: If storage upload or database write fails

        """
        processed_tasks = [self._process_data(task) for task in tasks]

        if not self._continue_captioning:
            # Use explicit cast to satisfy the type checker
            return cast("list[AvSessionVideoSplitTask | AvClipAnnotationTask]", processed_tasks)

        # Use explicit cast to satisfy the type checker
        return cast(
            "list[AvSessionVideoSplitTask | AvClipAnnotationTask]", self._chunk_clips_for_captioning(processed_tasks)
        )

    def _chunk_clips_for_captioning(self, tasks: list[AvSessionVideoSplitTask]) -> list[AvClipAnnotationTask]:
        # captioning stages take AvClipAnnotationTask as input
        # we also want to further chunk the pipeline payload

        captioning_clips = [
            ClipForAnnotation(
                video_session_name=task.source_video_session_name.rstrip("/"),
                clip_session_uuid=clip.clip_session_uuid,
                uuid=clip.uuid,
                camera_id=video.camera_id,
                span_index=clip.span_index,
                url=clip.url,  # type: ignore[arg-type]
                encoded_data=clip.encoded_data,
                span_start=clip.span_start,
                span_end=clip.span_end,
            )
            for task in tasks
            for video in task.videos
            for clip in video.clips
        ]

        height: int
        width: int
        framerate: float
        source_video_session_name: str

        for task in tasks:
            if len(task.videos) == 0:
                error = f"Task {task.source_video_session_name} has no videos, cannot continue to captioning"
                raise ValueError(error)
            source_video_session_name = task.source_video_session_name.rstrip("/")

            for video in task.videos:
                height = video.metadata.height  # type: ignore[assignment]
                width = video.metadata.width  # type: ignore[assignment]
                framerate = video.metadata.framerate  # type: ignore[assignment]

        chunks = list(split_by_chunk_size(captioning_clips, self._caption_chunk_size))

        captioning_tasks = []
        for idx, chunked_clips in enumerate(chunks):
            captioning_tasks.append(
                AvClipAnnotationTask(
                    clips=chunked_clips,
                    video_session_name=source_video_session_name,
                    num_session_chunks=len(chunks),
                    session_chunk_index=idx,
                    source_video_duration_s=(task.source_video_duration_s if idx == 0 else 0),
                    height=height,
                    width=width,
                    framerate=framerate,
                )
            )
        logger.info(f"Spawn {len(captioning_tasks)} captioning tasks from {len(task.videos)} videos")
        return captioning_tasks

    def _process_data(self, task: AvSessionVideoSplitTask) -> AvSessionVideoSplitTask:  # noqa: C901
        self._timer.reinit(self, task.get_major_size())

        if task.encoder is None:
            error = "Encoder is required"
            raise ValueError(error)

        # upload clips to s3
        with self._timer.time_process():
            num_uploaded_clips = 0
            try:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(self._upload_clips, video, task.encoder) for video in task.videos]
                    for future in futures:
                        try:
                            num_uploaded_clips += future.result()
                        except Exception as e:
                            logger.error(f"Error uploading clips: {e!s}")
                            raise
            except Exception as e:  # noqa: BLE001
                logger.error(f"S3 uploading failure, skip DB write: {e!s}")
            else:
                # abort if any clip within this session failed
                logger.info(f"Uploaded {num_uploaded_clips} clips from {task.session_url}")
                if self._db_type is not None:
                    # write metadata to postgres
                    try:
                        BaseWriterStage.process_data(self, [task])
                    except Exception as e:  # noqa: BLE001
                        logger.error(f"Error writing to postgres: {e!s}")

            # Clear buffers and timestamps if not continuing to captioning
            for video in task.videos:
                for clip in video.clips:
                    if not self._continue_captioning:
                        clip.encoded_data.drop()
                    clip.timestamps_ms = None

        if self._log_stats:
            stage_name, stage_perf_stats = self._timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats

        return task
