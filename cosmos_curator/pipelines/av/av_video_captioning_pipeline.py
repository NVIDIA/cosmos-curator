# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Ray pipeline for video clip captioning.

This pipeline:
- Download clips
- Generate caption for clips
- Generate embedding for captions
"""

import argparse
import time
import uuid

import ray
from loguru import logger

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.core.utils.config import args_utils
from cosmos_curator.core.utils.db.database_types import EnvType, PostgresDB
from cosmos_curator.core.utils.infra import ray_cluster_utils
from cosmos_curator.core.utils.infra.profiling import profiling_scope
from cosmos_curator.core.utils.misc.grouping import split_by_chunk_size
from cosmos_curator.pipelines.av.av_video_pipelines_common import (
    build_caption_pipeline_stages,
)
from cosmos_curator.pipelines.av.downloaders.download_stages import (
    ClipDownloader,
)
from cosmos_curator.pipelines.av.pipeline_args import add_common_args
from cosmos_curator.pipelines.av.utils.av_data_model import (
    AvClipAnnotationTask,
)
from cosmos_curator.pipelines.av.utils.av_pipe_input import (
    extract_clip_caption_tasks,
    read_session_file,
)
from cosmos_curator.pipelines.av.utils.run_utils import add_run_to_postrges
from cosmos_curator.pipelines.av.writers.cosmos_predict2_writer_stage import (
    generate_cosmos_predict2_prefix_cache,
)


def caption(args: argparse.Namespace) -> None:
    """Caption video clips with profiling and tracing.

    Public entry point that wraps ``_caption()`` with ``profiling_scope``
    so that every execution path (CLI, Slurm, NVCF, local launch)
    automatically gets profiling and distributed tracing.

    Args:
        args: Command line arguments

    """
    with profiling_scope(args):
        _caption(args)


def _caption(args: argparse.Namespace) -> None:
    """Caption video clips.

    This function:
    - Reads session file to get list of sessions to process
    - Creates a database instance
    - Extracts input data
    - Runs the pipeline

    Args:
        args: Command line arguments

    """
    zero_start = time.time()
    # it is possible a filename containing intended sessions are passed
    # process only those sessions
    sessions: list[str] = read_session_file(args.session_file)

    limit = 0 if len(sessions) > 0 else args.limit
    # create a database instance; no connection at this point
    db = PostgresDB.make_from_config(EnvType(args.db_profile))
    # extract input data
    input_clips = extract_clip_caption_tasks(
        db,
        camera_format_id=args.camera_format_id,
        prompt_types=args.prompt_types,
        source_version=args.clip_version,
        encoder=args.encoder,
        target_version=args.caption_version,
        sessions=sessions,
        limit=limit,
    )
    if args.limit > 0:
        input_clips = input_clips[: args.limit]
    if args.verbose:
        for clip in input_clips[:4]:
            logger.debug(f"{clip}")
    if len(input_clips) == 0:
        logger.info("No clips to process.")
        return

    input_tasks = [AvClipAnnotationTask(x) for x in split_by_chunk_size(input_clips, args.caption_chunk_size)]

    run_uuid = uuid.uuid4()
    logger.info(f"About to process {len(input_tasks)} tasks with run_id={run_uuid}")

    stages: list[CuratorStage | CuratorStageSpec] = [
        CuratorStageSpec(
            ClipDownloader(
                output_prefix=args.output_prefix,
                verbose=args.verbose,
                log_stats=args.perf_profile,
            ),
            num_workers_per_node=4,
            num_run_attempts_python=5,
        ),
    ]

    stages.extend(
        build_caption_pipeline_stages(
            args=args,
            db=db,
            run_uuid=run_uuid,
        )
    )

    if not args.dry_run:
        add_run_to_postrges(
            db,
            str(run_uuid),
            "caption",
            args.caption_version,
            extra={
                "camera_format_id": args.camera_format_id,
                "clip_version": args.clip_version,
                "prompt_types": args.prompt_types,
                "session_file": (str(args.session_file).removesuffix(".txt") if args.session_file is not None else ""),
                "num_clips": len(input_clips),
            },
        )

    pipeline_start = time.time()
    output_tasks = run_pipeline(
        input_tasks,
        stages=stages,
        args=args,
    )

    # Post-pipeline cache generation (only for cosmos_predict2 format)
    if args.output_format == "cosmos_predict2":
        # cluster is torn down after pipeline run, reinit
        ray_cluster_utils.init_or_connect_to_cluster()
        try:
            logger.info("Generating prefix embeddings cache...")
            cache_future = generate_cosmos_predict2_prefix_cache.remote(
                args.output_prefix,
                args.dataset_name,
                args.camera_format_id,
                args.prompt_types[0],  # Single prompt type guaranteed by validation
                args.prompt_text,
                args.verbose,
            )
            ray.get(cache_future)
            logger.info("Prefix embeddings cache generation completed successfully")
        except Exception as cache_error:  # noqa: BLE001
            logger.warning(f"Failed to generate prefix embeddings cache: {cache_error}")
            logger.info("Dataset generation completed successfully (cache generation failed)")

    if args.perf_profile:
        total_object_size = 0
        for task in output_tasks:
            total_object_size += task.get_major_size()
        logger.info(f"Total object size: {total_object_size:,} bytes")

    input_build_time = (pipeline_start - zero_start) / 60
    pipeline_run_time = (time.time() - pipeline_start) / 60

    logger.info(
        f"Caption pipeline: {input_build_time=:.2f} / "
        f"{pipeline_run_time=:.2f} mins processing "
        f"time for {len(input_clips)} clips"
    )


def _setup_parser(parser: argparse.ArgumentParser) -> None:
    add_common_args(parser, "caption")


def run_caption(args: argparse.Namespace) -> None:
    """Run the video captioning pipeline.

    Args:
        args: Command line arguments

    """
    args_utils.fill_default_args(args, _setup_parser)
    caption(args)


def add_caption_command(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> argparse.ArgumentParser:
    """Add the video captioning pipeline to the command line parser.

    Args:
        subparsers: The subparsers action to add the video captioning pipeline to

    Returns:
        The parser for the video captioning pipeline

    """
    parser = subparsers.add_parser(
        "caption",
        help="Caption clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(func=run_caption)
    _setup_parser(parser)
    return parser  # type: ignore[no-any-return]
