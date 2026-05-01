# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Utility functions for running AV pipelines."""

import uuid
from typing import Any

from loguru import logger
from sqlalchemy.orm import sessionmaker

from cosmos_curator.core.utils.db.database_types import PostgresDB
from cosmos_curator.pipelines.av.utils import postgres_schema


def add_run_to_postrges(  # noqa: PLR0913
    db: PostgresDB,
    run_uuid: str,
    run_type: str,
    pipeline_version: str,
    description: str | None = None,
    extra: dict[str, Any] | None = None,
    exists_ok: bool = False,  # noqa: FBT001, FBT002
) -> None:
    """Initialize a run entry in the database with a specific run ID.

    Args:
        db: Postgres Db to record to
        run_uuid: The UUID of the run
        run_type: The type of the pipeline
        pipeline_version: The version of the pipeline being run (e.g. v0, v1, etc)
        description: Optional field for a description of the run
        extra: A dict of extra metadata to record. This will be recorded as a JSONB.
        exists_ok: Whether to allow the run to already exist in the database.

    Raises:
        ValueError if run already exists and exist_ok is False.

    """
    logger.info(f"Initializing run with config info {extra}")

    engine = db.make_sa_engine()
    session_maker = sessionmaker(engine)
    with session_maker() as session:
        run_exists = (
            session.query(postgres_schema.Run).filter(postgres_schema.Run.run_uuid == uuid.UUID(run_uuid)).first()
        )
        if run_exists and not exists_ok:
            error = f"Run ID {run_uuid} already present in database. Please set `exists_okay=True` if this is expected."
            raise ValueError(error)
        if run_exists and exists_ok:
            logger.info(f"Run ID {run_uuid} already present in database. Will re-use!")
            return

    # Video and image have slighly different schemas, so we need to switch case here.
    if not run_exists:
        with session_maker() as session:
            run = postgres_schema.Run(
                run_uuid=uuid.UUID(run_uuid),
                run_type=run_type,
                pipeline_version=pipeline_version,
                description=description,
                params=extra,
            )
            session.add(run)
            session.commit()
            logger.info(f"Created new run {run_uuid}.")
