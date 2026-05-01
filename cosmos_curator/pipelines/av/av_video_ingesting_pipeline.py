# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Ray pipeline for ingesting source videos into the database."""

import argparse

from loguru import logger

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.core.utils.config import args_utils
from cosmos_curator.core.utils.db.database_types import EnvType, PostgresDB
from cosmos_curator.core.utils.infra.profiling import profiling_scope
from cosmos_curator.core.utils.storage.storage_utils import verify_path
from cosmos_curator.pipelines.av.pipeline_args import add_common_args
from cosmos_curator.pipelines.av.utils.av_data_model import (
    AvSessionVideoIngestTask,
)
from cosmos_curator.pipelines.av.utils.av_pipe_input import (
    extract_source_video_sessions,
)
from cosmos_curator.pipelines.av.writers.clip_writer_stage import (
    SourceVideoIngestionStage,
)


def ingest(args: argparse.Namespace) -> None:
    """Run the ingestion pipeline with profiling and tracing.

    Public entry point that wraps ``_ingest()`` with ``profiling_scope``
    so that every execution path (CLI, Slurm, NVCF, local launch)
    automatically gets profiling and distributed tracing.

    Args:
        args: The arguments for the pipeline.

    """
    with profiling_scope(args):
        _ingest(args)


def _ingest(args: argparse.Namespace) -> None:
    """Run the ingestion pipeline.

    Args:
        args: The arguments for the pipeline.

    """
    # validate input arguments
    if args.input_prefix is None:
        error = "input_prefix is required"
        raise ValueError(error)
    verify_path(args.input_prefix)

    db = PostgresDB.make_from_config(EnvType(args.db_profile))

    # extract input data
    input_sessions = extract_source_video_sessions(
        db,
        input_path=args.input_prefix,
        version=args.source_version,
        verbose=args.verbose,
    )
    if args.verbose:
        for session in input_sessions:
            logger.info(f"Found input data {session}")

    if len(input_sessions) == 0:
        logger.info("No new input sessions found. Exiting.")
        return

    logger.info(f"Adding {len(input_sessions)} input video sessions")
    input_tasks = [AvSessionVideoIngestTask(sessions=input_sessions)]

    stages: list[CuratorStage | CuratorStageSpec] = [
        SourceVideoIngestionStage(db),
    ]

    if not args.dry_run:
        run_pipeline(input_tasks, stages=stages, args=args)


def _setup_parser(parser: argparse.ArgumentParser) -> None:
    add_common_args(parser, "ingest")


def run_ingest(args: argparse.Namespace) -> None:
    """Run the ingestion pipeline.

    Args:
        args: The arguments for the pipeline.

    """
    args_utils.fill_default_args(args, _setup_parser)
    ingest(args)


def add_ingest_command(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> argparse.ArgumentParser:
    """Add the ingestion command to the parser.

    Args:
        subparsers: The subparsers for the parser.

    Returns:
        The parser.

    """
    parser = subparsers.add_parser(
        "ingest",
        help="ingest source videos into database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(func=run_ingest)
    _setup_parser(parser)
    return parser  # type: ignore[no-any-return]
