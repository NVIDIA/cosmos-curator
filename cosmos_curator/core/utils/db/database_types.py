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
"""Common types used for databases.

This is not meant to store pipeline schemas, but instead data types used to access/interact with our databases and
common types.
"""

import enum
import time
import urllib.parse
from typing import Any, Self

import attrs
import sqlalchemy
import tenacity
from loguru import logger

from cosmos_curator.core.utils.config import config


class TableSet(enum.Enum):
    """Enumeration of table sets."""

    AV = "av"


class EnvType(enum.Enum):
    """Enumeration of environment types."""

    LOCAL = "local"
    DEV = "dev"
    PROD = "prod"


@attrs.define
class RetryParams:
    """Parameters for configuring retry behavior in database operations.

    Attributes:
        num_attempts: Number of times to retry a failed operation. Defaults to 5.
        wait_time: Time in seconds to wait between retry attempts. Defaults to 3.0.

    """

    num_attempts: int = 5
    wait_time: float = 3.0


@attrs.define
class PostgresDB:
    """A class to represent and manage a PostgreSQL database connection."""

    env_type: EnvType
    endpoint: str
    name: str
    username: str
    password: str

    @staticmethod
    def _make_video_profile_name(env_type: EnvType) -> str:
        """Generate a profile name based on env type.

        Args:
            env_type: An EnvType enum value representing the environment.

        Returns:
            A string representing the profile name.

        Raises:
            ValueError: If the env_type is not supported.

        """
        match env_type:
            case EnvType.LOCAL:
                profile_name = "nvc_local"
            case EnvType.DEV:
                profile_name = "nvc_dev"
            case EnvType.PROD:
                profile_name = "nvc_prd"
            case _:
                error_msg = f"Unknown env_type: {env_type}"  # type: ignore[unreachable]
                raise ValueError(error_msg)
        return profile_name

    @classmethod
    def make_from_config(cls, env_type: EnvType) -> Self:
        """Create a PostgresDB instance from configuration.

        Args:
            env_type: An EnvType enum value representing the environment.

        Returns:
            A PostgresDB instance.

        Raises:
            ValueError: If env_type is unknown.

        """
        c = config.load_config()

        profile_name = cls._make_video_profile_name(env_type)

        match env_type:
            case EnvType.LOCAL:
                db_name = "nemo_video_curator_local"
            case EnvType.DEV:
                db_name = "nemo_video_curator_dev"
            case EnvType.PROD:
                db_name = "nemo_video_curator_prd"
            case _:
                error_msg = f"Unknown env_type: {env_type}"  # type: ignore[unreachable]
                raise ValueError(error_msg)
        postgres_prof = c.get_postgres_profile(profile_name)
        return cls(
            env_type,
            postgres_prof.endpoint,
            db_name,
            postgres_prof.user,
            postgres_prof.password,
        )

    def _make_engine_uri(self) -> str:
        """Generate a SQLAlchemy engine URI for the database connection.

        Returns:
            A string representing the SQLAlchemy engine URI.

        """
        encoded_password = urllib.parse.quote_plus(self.password)
        return f"postgresql://{self.username}:{encoded_password}@{self.endpoint}/{self.name}"

    def make_sa_engine(self) -> sqlalchemy.Engine:
        """Create a SQLAlchemy engine for the database connection.

        Returns:
            A SQLAlchemy Engine instance.

        """
        return sqlalchemy.create_engine(self._make_engine_uri(), pool_pre_ping=True)

    def _make_query(self, query: str, *, verbose: bool = True) -> list[Any]:
        engine = self.make_sa_engine()
        if verbose:
            start_time = time.time()
            logger.info(f"Making query\n{query}")
        with engine.begin() as connection:
            result = connection.execute(sqlalchemy.text(query))
        if verbose:
            end_time = time.time()
            logger.info(f"Query took {end_time - start_time} seconds.")
        return list(result)

    def make_query(self, query: str, retry_params: RetryParams | None = None, *, verbose: bool = False) -> list[Any]:
        """Make a query to Postgres.

        Args:
          query: The query to make.
          retry_params: The parameters for retrying the query.
          verbose: If true, this method will log query timing info.

        Returns:
          A list of Cursor Results.

        """
        if retry_params is not None:

            def make_query_fn() -> list[Any]:
                return self._make_query(query, verbose=verbose)

            return tenacity.retry(
                stop=tenacity.stop_after_attempt(retry_params.num_attempts),
                wait=tenacity.wait_fixed(retry_params.wait_time),
                reraise=True,
            )(make_query_fn)()
        return self._make_query(query, verbose=verbose)
