# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions for Postgres SQL database."""

from collections.abc import Iterable
from typing import Any, Self

import psycopg2
from loguru import logger
from sqlalchemy import exc
from sqlalchemy.orm import scoped_session, sessionmaker

from cosmos_curator.core.utils.db.database_types import EnvType, PostgresDB


class DbRetrier:
    """A Class which can be used to write data to a postgres DB with retries."""

    def __init__(
        self,
        db: PostgresDB,
        max_retries: int = 5,
    ) -> None:
        """Initialize the DbRetrier.

        Args:
            db: The PostgresDB object.
            max_retries: The maximum number of retries.

        """
        self.max_retries = max_retries
        if self.max_retries < 0:
            error_msg = f"Max retries must be >= 0! Got {self.max_retries}"
            raise ValueError(error_msg)
        self.db = db
        self.engine = self.db.make_sa_engine()
        self.scoped_session = scoped_session(sessionmaker(bind=self.engine))

    @classmethod
    def make_from_db_name(cls, etype: EnvType, max_retries: int = 5) -> Self:
        """Create a DbRetrier from a database name.

        Args:
            etype: The environment type.
            max_retries: The maximum number of retries.

        """
        db = PostgresDB.make_from_config(env_type=etype)
        return cls(db, max_retries)

    def reset_session(self) -> None:
        """Reset session.

        To do this properly:
        1. Dispose of engine
        2. Create new engine
        3. Remove any scoped sessions
        4. Configure session by binding to newly created engine
        """
        self.engine.dispose()
        self.engine = self.db.make_sa_engine()
        self.scoped_session.remove()
        self.scoped_session.configure(bind=self.engine)

    def write_data(self, objects: Iterable[Any]) -> None:
        """Will write data from `create_data` method."""
        attempts = 0
        while attempts <= max(0, self.max_retries):
            session = self.scoped_session()
            try:
                session.bulk_save_objects(objects)
                session.commit()
            except (psycopg2.OperationalError, exc.SQLAlchemyError) as e:
                session.rollback()
                logger.error(f"Attempt {attempts + 1} failed, error: {e}")
                if attempts < self.max_retries:
                    logger.info("Re-setting session and retrying ...")
                    self.reset_session()
                else:
                    logger.error(f"Max retries {self.max_retries} reached, failing operation.")
                    raise
            else:
                return
            finally:
                session.close()
                self.scoped_session.remove()
            attempts += 1
