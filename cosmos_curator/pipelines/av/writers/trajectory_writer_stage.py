# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Module for writing trajectory data to database and storage.

This module provides a stage for writing trajectory data and metadata to both a database
and storage (local or S3) in binary format.
"""

import io
import pickle
import uuid
from pathlib import Path

from loguru import logger
from sqlalchemy.orm import Session

from cosmos_curator.core.utils.db.database_types import PostgresDB
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.core.utils.storage import s3_client
from cosmos_curator.core.utils.storage.writer_utils import write_bytes
from cosmos_curator.pipelines.av.utils.av_data_model import AvSessionTrajectoryTask
from cosmos_curator.pipelines.av.utils.postgres_schema import ClipTrajectory
from cosmos_curator.pipelines.av.writers.base_writer_stage import BaseWriterStage
from cosmos_curator.pipelines.av.writers.make_db_row import make_clip_trajectory


class TrajectoryWriterStage(BaseWriterStage):
    """Stage for writing trajectory data to database and storage.

    This stage handles writing trajectory data and metadata to a PostgreSQL database
    and binary files in storage (local or S3).

    Args:
        db: PostgreSQL database configuration
        output_prefix: Base path for output files
        run_id: Unique identifier for this run
        version: Version string
        max_retries: Maximum number of retry attempts for database operations
        verbose: Whether to enable verbose logging
        log_stats: Whether to log performance statistics

    """

    def __init__(  # noqa: PLR0913
        self,
        db: PostgresDB,
        output_prefix: str,
        run_id: uuid.UUID,
        version: str,
        max_retries: int = 1,
        verbose: bool = False,  # noqa: FBT001, FBT002
        log_stats: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize a stage that writes metadata for a trajectory run.

        Args:
            db: PostgreSQL database configuration
            output_prefix: Base path for output files
            run_id: Unique identifier for this run
            version: Version string
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
        self._verbose = verbose
        self._log_stats = log_stats
        self._clip_prefix = "trajectory"

    def stage_setup(self) -> None:
        """Set up database connection and S3 client."""
        super().stage_setup()
        self._s3_client = s3_client.create_s3_client(
            target_path=self._output_prefix,
            can_overwrite=True,
        )

    def _get_trajectory_url(self, clip_uuid: uuid.UUID) -> s3_client.S3Prefix | Path:
        """Generate a URL or path for storing trajectory data.

        Args:
            clip_uuid: UUID of the specific clip

        Returns:
            An S3Prefix object if the output_prefix is an S3 path, otherwise a
            pathlib.Path object

        """
        full_path = f"{self._output_prefix}/{self._db_type}/{self._clip_prefix}/{self._version}/{clip_uuid}.bin"
        if s3_client.is_s3path(self._output_prefix):
            return s3_client.S3Prefix(full_path)
        return Path(full_path)

    def write_data(  # type: ignore[override]
        self, session: Session, task: AvSessionTrajectoryTask
    ) -> list[AvSessionTrajectoryTask] | None:
        """Write trajectory metadata to database.

        Args:
            session: SQLAlchemy database session
            task: Task containing trajectory data to write

        Returns:
            None if successful

        Raises:
            SQLAlchemyError: If database write fails

        """
        objects = list(make_clip_trajectory(task, self._version, str(self._run_id)))
        logger.info(f"Writing {len(objects)} rows to {ClipTrajectory.__tablename__}.")
        session.bulk_save_objects(objects)
        session.commit()
        return None

    def process_data(  # type: ignore[override]
        self, tasks: list[AvSessionTrajectoryTask]
    ) -> list[AvSessionTrajectoryTask] | None:
        """Process and write trajectory data with performance tracking.

        This method uploads trajectory data to storage and writes metadata to the database.
        After processing, the trajectory data is cleared from memory.

        Args:
            tasks: Tasks containing trajectory data to process

        Returns:
            List containing the input task if successful

        Raises:
            Exception: If storage upload or database write fails

        """
        return [self._process_data(task) for task in tasks]

    def _process_data(self, task: AvSessionTrajectoryTask) -> AvSessionTrajectoryTask:
        self._timer.reinit(self, task.get_major_size())
        # upload clip trajectory to s3
        with self._timer.time_process():
            num_uploaded_clips = 0
            try:
                for clip in task.clips:
                    trajectory = clip.trajectory
                    if trajectory is None:
                        logger.error(f"Clip {clip.uuid} has no trajectory")
                        continue
                    dest = self._get_trajectory_url(clip.uuid)
                    clip.trajectory_url = str(dest)
                    buffer = io.BytesIO()
                    pickle.dump(trajectory, buffer)
                    write_bytes(
                        buffer.getvalue(),
                        dest,
                        f"clip-{clip.uuid}",
                        "unknown",
                        verbose=self._verbose,
                        client=self._s3_client,
                        overwrite=True,
                    )
                    num_uploaded_clips += 1
            except Exception as e:  # noqa: BLE001
                logger.error(f"S3 uploading failure, skip DB write: {e!s}")
            finally:
                # allow DB write if some clip failed
                logger.info(f"Uploaded {num_uploaded_clips} clips")
                # write metadata to postgres
                try:
                    BaseWriterStage.process_data(self, [task])
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Error writing to postgres: {e!s}")
            # Clear trajectory data from memory
            for clip in task.clips:
                clip.trajectory = None

        if self._log_stats:
            stage_name, stage_perf_stats = self._timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats
        return task
