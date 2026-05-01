# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Ray pipeline for sharding video data."""

import argparse
import time
import uuid
from pathlib import Path

from loguru import logger

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.core.utils.config import args_utils
from cosmos_curator.core.utils.db.database_types import EnvType, PostgresDB
from cosmos_curator.core.utils.infra.profiling import profiling_scope
from cosmos_curator.core.utils.misc.grouping import split_by_chunk_size
from cosmos_curator.core.utils.storage.s3_client import S3Prefix, create_s3_client, is_s3path
from cosmos_curator.core.utils.storage.writer_utils import write_json
from cosmos_curator.pipelines.av.pipeline_args import add_common_args
from cosmos_curator.pipelines.av.utils.av_data_model import AvShardingTask
from cosmos_curator.pipelines.av.utils.av_pipe_input import (
    CAPTION_KEYWORDS,
    WINDOWS_PER_CLIP,
    add_trajectory_to_samples,
    extract_sharding_tasks,
    read_session_file,
)
from cosmos_curator.pipelines.av.utils.run_utils import add_run_to_postrges
from cosmos_curator.pipelines.av.writers.dataset_writer_stage import (
    MAX_TARS_PER_PART,
    T5_VARIANTS,
    ClipPackagingStage,
    T5EmbeddingPackagingStageE,
    T5EmbeddingPackagingStageH,
)

_SPLIT_ALGO_NAME = "fixed-stride"


def shard(args: argparse.Namespace) -> None:
    """Run the AV sharding pipeline with profiling and tracing.

    Public entry point that wraps ``_shard()`` with ``profiling_scope``
    so that every execution path (CLI, Slurm, NVCF, local launch)
    automatically gets profiling and distributed tracing.

    Args:
        args: The arguments for the pipeline.

    """
    with profiling_scope(args):
        _shard(args)


def _shard(args: argparse.Namespace) -> None:  # noqa: PLR0912, C901
    """Run the sharding pipeline.

    Args:
        args: The arguments for the pipeline.

    """
    zero_start = time.time()
    # it is possible a filename containing intended sessions are passed
    # process only those sessions
    sessions: list[str] = read_session_file(args.session_file)
    limit = 0 if len(sessions) > 0 else args.limit

    # create a database instance; no connection at this point
    db = PostgresDB.make_from_config(EnvType(args.db_profile))
    # extract input data
    input_clip_sessions = extract_sharding_tasks(
        db,
        camera_format_id=args.camera_format_id,
        clip_version=args.clip_version,
        split_algo_name=_SPLIT_ALGO_NAME,
        encoder=args.encoder,
        caption_version=args.caption_version,
        prompt_type=args.prompt_type,
        sessions=sessions,
        keyword=args.keyword,
        limit=limit,
    )

    if args.include_trajectory:
        input_clip_sessions = add_trajectory_to_samples(
            db,
            input_clip_sessions,
            args.trajectory_version,
        )

    if len(input_clip_sessions) == 0:
        logger.info("No clip-sessions to process.")
        return
    logger.info(f"Found {len(input_clip_sessions)} clip-sessions")

    if args.limit > 0:
        input_clip_sessions = input_clip_sessions[: args.limit]

    if args.dry_run:
        return

    run_uuid = uuid.uuid4()

    t5_packaging_stage: T5EmbeddingPackagingStageE | T5EmbeddingPackagingStageH
    if args.t5_tar_format_variant == "H":
        t5_packaging_stage = T5EmbeddingPackagingStageH(
            db,
            camera_format_id=args.camera_format_id,
            dataset_name=str(run_uuid),
            output_prefix=args.output_prefix,
            verbose=args.verbose,
            log_stats=args.perf_profile,
        )
        chunk_size = t5_packaging_stage.sessions_per_part()
    elif args.t5_tar_format_variant == "E":
        t5_packaging_stage = T5EmbeddingPackagingStageE(
            db,
            camera_format_id=args.camera_format_id,
            dataset_name=str(run_uuid),
            output_prefix=args.output_prefix,
            verbose=args.verbose,
            log_stats=args.perf_profile,
        )
        chunk_size = MAX_TARS_PER_PART
    else:
        error = f"Unknown T5 tar format variant: {args.t5_tar_format_variant}"
        raise ValueError(error)

    input_tasks: list[AvShardingTask] = []
    for samples in split_by_chunk_size(input_clip_sessions, chunk_size):
        part_num = len(input_tasks)
        input_tasks.append(AvShardingTask(part_num, samples))

    logger.info(f"About to process {len(input_tasks)} tasks with run_id={run_uuid}")

    stages: list[CuratorStage | CuratorStageSpec] = [
        CuratorStageSpec(
            ClipPackagingStage(
                db,
                camera_format_id=args.camera_format_id,
                dataset_name=str(run_uuid),
                output_prefix=args.output_prefix,
                verbose=args.verbose,
                log_stats=args.perf_profile,
            ),
            num_workers_per_node=16,
            num_run_attempts_python=5,
        ),
        CuratorStageSpec(
            t5_packaging_stage,
            num_workers_per_node=8,
            num_run_attempts_python=5,
        ),
    ]

    add_run_to_postrges(
        db,
        str(run_uuid),
        "shard",
        "v0",
        extra={
            "camera_format_id": args.camera_format_id,
            "t5_tar_format_variant": args.t5_tar_format_variant,
            "clip_version": args.clip_version,
            "caption_version": args.caption_version,
            "prompt_type": args.prompt_type,
            "session_file": (str(args.session_file).removesuffix(".txt") if args.session_file is not None else ""),
            "keyword": args.keyword if args.keyword is not None else "",
            "num_clip_sessions": len(input_clip_sessions),
            "include_trajectory": args.include_trajectory,
            "trajectory_version": args.trajectory_version,
        },
    )

    pipeline_start = time.time()
    output_tasks = run_pipeline(
        input_tasks,
        stages=stages,
        args=args,
    )

    if args.t5_tar_format_variant == "H":
        if not isinstance(t5_packaging_stage, T5EmbeddingPackagingStageH):
            error_msg = "T5 packaging stage is not a T5EmbeddingPackagingStageH"
            raise TypeError(error_msg)
        # write mapping file
        bin_to_tar_mappings: dict[int, dict[str, str]] = {k: {} for k in range(WINDOWS_PER_CLIP)}
        for task in output_tasks:
            for k in task.tar_mappings:
                for clip_name, tar_url in task.tar_mappings[k].items():
                    bin_to_tar_mappings[k][clip_name] = tar_url
        client = create_s3_client(target_path=args.output_prefix)
        for k, the_mappings in bin_to_tar_mappings.items():
            url = f"{t5_packaging_stage.get_chunk_prefix(T5_VARIANTS[k])}/mapping.json"
            dest = S3Prefix(url) if is_s3path(args.output_prefix) else Path(url)
            write_json(
                the_mappings,
                dest,
                "mappings",
                str(run_uuid),
                verbose=args.verbose,
                client=client,
                backup_and_overwrite=True,
            )

    if args.perf_profile:
        total_object_size = 0
        for task in output_tasks:
            total_object_size += task.get_major_size()
        logger.info(f"Total object size: {total_object_size:,} bytes")

    input_build_time = (pipeline_start - zero_start) / 60
    pipeline_run_time = (time.time() - pipeline_start) / 60

    logger.info(
        f"Shard pipeline: {input_build_time=:.2f} / "
        f"{pipeline_run_time=:.2f} mins processing "
        f"time for {len(input_clip_sessions)} sessions"
    )


def _setup_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--t5-tar-format-variant",
        type=str,
        choices=["H", "E"],
        default="H",
        help="T5 tar format variant",
    )
    parser.add_argument(
        "--keyword",
        type=str,
        choices=CAPTION_KEYWORDS,
        default=None,
        help="Full Pathname of file containing list of sessions",
    )
    parser.add_argument(
        "--include-trajectory",
        action="store_true",
        help="Whether to include GPS trajectory for each clip",
    )
    add_common_args(parser, "shard")


def run_shard(args: argparse.Namespace) -> None:
    """Run the sharding pipeline.

    Args:
        args: The arguments for the pipeline.

    """
    args_utils.fill_default_args(args, _setup_parser)
    shard(args)


def add_shard_command(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> argparse.ArgumentParser:
    """Add the shard command to the parser.

    Args:
        subparsers: The subparsers for the parser.

    Returns:
        The parser.

    """
    parser = subparsers.add_parser(
        "shard",
        help="Shard clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(func=run_shard)
    _setup_parser(parser)
    return parser  # type: ignore[no-any-return]
