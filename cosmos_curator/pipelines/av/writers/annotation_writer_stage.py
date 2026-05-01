# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Module for writing clip annotations to database and storage.

This module provides stages for writing clip annotations to both a database
and storage (local or S3) in JSON format.
"""

import pathlib
import uuid

from loguru import logger
from sqlalchemy.orm import Session

from cosmos_curator.core.interfaces.stage_interface import CuratorStage
from cosmos_curator.core.utils.db.database_types import PostgresDB
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.core.utils.storage import s3_client
from cosmos_curator.core.utils.storage.s3_client import is_s3path
from cosmos_curator.core.utils.storage.writer_utils import write_json
from cosmos_curator.pipelines.av.utils.av_data_model import (
    AvClipAnnotationTask,
)
from cosmos_curator.pipelines.av.utils.postgres_schema import ClipCaption
from cosmos_curator.pipelines.av.writers.base_writer_stage import BaseWriterStage
from cosmos_curator.pipelines.av.writers.make_db_row import make_clip_caption


class AnnotationDbWriterStage(BaseWriterStage):
    """Stage for writing clip annotations to database.

    This stage handles writing clip annotations and metadata to a PostgreSQL database.
    It supports multiple caption prompt types and chain lengths.
    """

    def __init__(  # noqa: PLR0913
        self,
        db: PostgresDB,
        output_prefix: str,
        run_id: uuid.UUID,
        version: str,
        caption_prompt_types: list[str],
        caption_chain_lens: dict[str, int],
        max_retries: int = 1,
        verbose: bool = False,  # noqa: FBT001, FBT002
        log_stats: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize a stage that writes metadata for a split run.

        Args:
            db: PostgreSQL database configuration
            output_prefix: Base path for output files
            run_id: Unique identifier for this run
            version: Version string
            caption_prompt_types: List of caption prompt types to process
            caption_chain_lens: Dictionary mapping prompt types to chain lengths
            max_retries: Maximum number of retry attempts for database operations
            verbose: Whether to enable verbose logging
            log_stats: Whether to log performance statistics

        """
        super().__init__(
            db=db,
            max_retries=max_retries,
        )
        self._timer = StageTimer(self)
        self._db_type = db.env_type.value
        self._output_prefix = output_prefix.rstrip("/")
        self._run_id = run_id
        self._version = version
        self._caption_prompt_types = caption_prompt_types
        self._verbose = verbose
        self._log_stats = log_stats
        self._clip_prefix = "t5_embeddings"
        self._caption_chain_lens = caption_chain_lens

    def stage_setup(self) -> None:
        """Set up database connection and S3 client."""
        super().stage_setup()
        self._s3_client = s3_client.create_s3_client(
            target_path=self._output_prefix,
            can_overwrite=True,
        )

    def write_data(  # type: ignore[override]
        self, session: Session, task: AvClipAnnotationTask
    ) -> list[AvClipAnnotationTask] | None:
        """Write clip annotations to database.

        Args:
            session: SQLAlchemy database session
            task: Task containing clip annotations to write

        Returns:
            None if successful

        Raises:
            SQLAlchemyError: If database write fails

        """
        objects = [
            obj
            for prompt_type in self._caption_prompt_types
            for obj in make_clip_caption(
                task.clips, self._version, prompt_type, self._run_id, self._caption_chain_lens[prompt_type]
            )
        ]
        logger.info(f"Writing {len(objects)} rows to {ClipCaption.__tablename__}.")
        session.bulk_save_objects(objects)
        session.commit()
        return None

    def process_data(  # type: ignore[override]
        self, tasks: list[AvClipAnnotationTask]
    ) -> list[AvClipAnnotationTask] | None:
        """Process and write clip annotations with performance tracking.

        Args:
            tasks: Tasks containing clip annotations to process

        Returns:
            List containing the input task if successful

        Raises:
            Exception: If database write fails

        """
        return [self._process_data(task) for task in tasks]

    def _process_data(self, task: AvClipAnnotationTask) -> AvClipAnnotationTask:
        self._timer.reinit(self, task.get_major_size())
        # upload clip caption embeddings to s3
        with self._timer.time_process():
            # write metadata to postgres
            try:
                BaseWriterStage.process_data(self, [task])
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error writing to postgres: {e!s}")

        if self._log_stats:
            stage_name, stage_perf_stats = self._timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats
        return task


def _get_json_annotation_url(
    prefix: str,
    clip_uuid: uuid.UUID,
) -> s3_client.S3Prefix | pathlib.Path:
    """Generate a URL or path for storing annotation JSON data.

    Args:
        prefix: Base path for annotations
        clip_uuid: UUID of the specific clip

    Returns:
        An S3Prefix object if the output_prefix is an S3 path, otherwise a
        pathlib.Path object

    """
    full_path = f"{prefix}/metas/{clip_uuid}.json"
    if is_s3path(prefix):
        return s3_client.S3Prefix(full_path)
    return pathlib.Path(full_path)


def _get_json_processed_chunk_url(
    prefix: str,
    video_session_name: str,
    session_chunk_index: int,
) -> s3_client.S3Prefix | pathlib.Path:
    """Generate a URL or path for storing processed chunk JSON data.

    Args:
        prefix: Base path for processed chunks
        video_session_name: Name of the video session
        session_chunk_index: Index of the chunk within the session

    Returns:
        An S3Prefix object if the output_prefix is an S3 path, otherwise a
        pathlib.Path object

    """
    full_path = f"{prefix}/processed_session_chunks/{video_session_name}_{session_chunk_index}.json"
    if is_s3path(prefix):
        return s3_client.S3Prefix(full_path)
    return pathlib.Path(full_path)


def _get_json_processed_session_url(
    prefix: str,
    video_session_name: str,
) -> s3_client.S3Prefix | pathlib.Path:
    """Generate a URL or path for storing processed session JSON data.

    Args:
        prefix: Base path for processed sessions
        video_session_name: Name of the video session

    Returns:
        An S3Prefix object if the output_prefix is an S3 path, otherwise a
        pathlib.Path object

    """
    full_path = f"{prefix}/processed_sessions/{video_session_name}.json"
    if is_s3path(prefix):
        return s3_client.S3Prefix(full_path)
    return pathlib.Path(full_path)


def _annotation_json_writer(
    s3_client: s3_client.S3Client | None,
    task: AvClipAnnotationTask,
    output_prefix: str,
    overwrite: bool,  # noqa: FBT001
    verbose: bool,  # noqa: FBT001
) -> None:
    """Write JSON annotations for clips to storage.

    Args:
        s3_client: Client for S3 operations, or None for local storage
        task: Task containing clip annotations to write
        output_prefix: Base path for output
        overwrite: Whether to overwrite existing files
        verbose: Whether to log verbose output

    """
    # write clip-level metas
    for clip in task.clips:
        # data
        data = clip.to_dict(last_caption_only=True, use_formatted_vri_tags=True)
        data["height"] = task.height
        data["width"] = task.width
        data["framerate"] = task.framerate
        output_url = _get_json_annotation_url(
            output_prefix,
            clip.uuid,
        )
        write_json(
            data,
            output_url,
            f"clip-{clip.uuid}",
            str(output_url),
            verbose=verbose,
            client=s3_client,
            overwrite=overwrite,
        )
    # write video session chunk-level metas
    if task.session_chunk_index == 0:
        data = task.to_dict(video_level=True)
        output_url = _get_json_processed_session_url(
            output_prefix,
            task.video_session_name,  # type: ignore[arg-type]
        )
        write_json(
            data,
            output_url,
            f"session-{task.video_session_name}-chunk-{task.session_chunk_index}",
            str(output_url),
            verbose=verbose,
            client=s3_client,
            overwrite=overwrite,
        )
    # write video session-level metas
    data = task.to_dict()
    output_url = _get_json_processed_chunk_url(
        output_prefix,
        task.video_session_name,  # type: ignore[arg-type]
        task.session_chunk_index,  # type: ignore[arg-type]
    )
    write_json(
        data,
        output_url,
        f"session-{task.video_session_name}",
        str(output_url),
        verbose=verbose,
        client=s3_client,
        overwrite=overwrite,
    )


class AnnotationJsonWriterStage(CuratorStage):
    """Stage for writing clip annotations to JSON files.

    This stage handles writing clip annotations and metadata to JSON files,
    either locally or to S3 storage.
    """

    def __init__(
        self,
        output_prefix: str,
        overwrite: bool = False,  # noqa: FBT001, FBT002
        verbose: bool = False,  # noqa: FBT001, FBT002
        log_stats: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize a stage that writes metadata for a split run.

        Args:
            output_prefix: Base path for output files
            overwrite: Whether to overwrite existing files
            verbose: Whether to enable verbose logging
            log_stats: Whether to log performance statistics

        """
        self._timer = StageTimer(self)
        self._output_prefix = output_prefix.rstrip("/")
        self._overwrite = overwrite
        self._verbose = verbose
        self._log_stats = log_stats

    def stage_setup(self) -> None:
        """Set up S3 client for writing JSON files."""
        super().stage_setup()
        self._s3_client = s3_client.create_s3_client(
            target_path=self._output_prefix,
            can_overwrite=True,
        )

    def process_data(  # type: ignore[override]
        self, tasks: list[AvClipAnnotationTask]
    ) -> list[AvClipAnnotationTask] | None:
        """Process and write clip annotations to JSON files.

        Args:
            tasks: Tasks containing clip annotations to process

        Returns:
            List containing the input task if successful

        """
        return [self._process_data(task) for task in tasks]

    def _process_data(self, task: AvClipAnnotationTask) -> AvClipAnnotationTask:
        self._timer.reinit(self, task.get_major_size())
        with self._timer.time_process():
            try:
                _annotation_json_writer(
                    self._s3_client,
                    task,
                    self._output_prefix,
                    self._overwrite,
                    self._verbose,
                )
            except Exception as e:  # noqa: BLE001
                logger.error(f"Annotation json write failure {e!s}")

        if self._log_stats:
            stage_name, stage_perf_stats = self._timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats
        return task
