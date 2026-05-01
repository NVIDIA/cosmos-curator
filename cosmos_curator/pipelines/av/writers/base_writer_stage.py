"""Base class for writing metadata to database.

This module provides a base class for writing metadata to a database with retry logic
and session management.
"""

from abc import abstractmethod
from typing import TypeVar

import psycopg2
from loguru import logger
from sqlalchemy import exc
from sqlalchemy.orm import Session, scoped_session, sessionmaker

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, PipelineTask
from cosmos_curator.core.utils.db.database_types import PostgresDB

# Generic type for the task data
PipelineTaskType = TypeVar("PipelineTaskType", bound=PipelineTask)


class BaseWriterStage(CuratorStage):
    """Base class for database writer stages.

    This class provides common functionality for writing data to a database,
    including session management and retry logic.
    """

    def __init__(
        self,
        db: PostgresDB,
        max_retries: int = 1,
    ) -> None:
        """Initialize a base writer stage.

        Args:
            db: PostgreSQL database configuration
            max_retries: Maximum number of retry attempts for database operations

        """
        self._db = db
        self._max_retries = max_retries
        if self._max_retries < 0:
            error = f"Max retries must be >= 0! Got {self._max_retries}"
            raise ValueError(error)

    def stage_setup(self) -> None:
        """Set up database connection and session factory."""
        self._engine = self._db.make_sa_engine()
        self._scoped_session = scoped_session(sessionmaker(bind=self._engine))

    def reset_session(self) -> None:
        """Reset the database session.

        To do this properly:
        1. Dispose of engine
        2. Create new engine
        3. Remove any scoped sessions
        4. Configure session by binding to newly created engine
        """
        self._engine.dispose()
        self._engine = self._db.make_sa_engine()
        self._scoped_session.remove()
        self._scoped_session.configure(bind=self._engine)

    @abstractmethod
    def write_data(self, session: Session, task: PipelineTaskType) -> list[PipelineTaskType] | None:
        """Write data to the database.

        Args:
            session: SQLAlchemy database session
            task: Task data to write

        Returns:
            List of processed tasks

        Raises:
            NotImplementedError: This method must be implemented by subclasses

        """
        error = "Please implement this method!"
        raise NotImplementedError(error)

    def write_data_batch(self, session: Session, tasks: list[PipelineTaskType]) -> list[PipelineTaskType]:
        """Write data to the database.

        Args:
            session: SQLAlchemy database session
            tasks: Tasks data to write

        Returns:
            List of processed tasks

        """
        output_tasks: list[PipelineTaskType] = []
        for task in tasks:
            written_tasks = self.write_data(session, task)
            if written_tasks is not None:
                output_tasks += written_tasks
        return output_tasks

    def process_data(self, tasks: list[PipelineTaskType]) -> list[PipelineTaskType] | None:
        """Process and write data with retry logic.

        This method attempts to write data using the write_data method,
        with configurable retry attempts on database errors.

        Args:
            tasks: Tasks data to process and write

        Returns:
            Optional list of processed tasks

        Raises:
            SQLAlchemyError: If all retry attempts fail

        """
        attempts = 0
        output_tasks: list[PipelineTaskType] = []
        while attempts <= max(0, self._max_retries):
            session = self._scoped_session()
            try:
                output_tasks += self.write_data_batch(session, tasks)
                break
            except (psycopg2.OperationalError, exc.SQLAlchemyError) as e:
                session.rollback()
                logger.error(f"Attempt {attempts + 1} failed, error: {e}")
                if attempts < self._max_retries:
                    logger.info("Re-setting session and retrying...")
                    self.reset_session()
                else:
                    logger.error(f"Max retries {self._max_retries} reached, failing operation.")
                    raise
            finally:
                session.close()
                self._scoped_session.remove()
            attempts += 1
        return output_tasks
