# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Ray pipelines.

- Download videos
- Split videos into segments
- Transcode raw videos into clips
"""

import argparse
import time
import uuid

from loguru import logger

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.core.utils.config import args_utils
from cosmos_curator.core.utils.db.database_types import EnvType, PostgresDB
from cosmos_curator.core.utils.infra.profiling import profiling_scope
from cosmos_curator.pipelines.av.av_video_pipelines_common import (
    build_caption_pipeline_stages,
)
from cosmos_curator.pipelines.av.captioning.captioning_stages import VRI_PROMPTS
from cosmos_curator.pipelines.av.clipping.clip_extraction_stages import (
    ClipTranscodingStage,
    FixedStrideExtractorStage,
)
from cosmos_curator.pipelines.av.downloaders.download_stages import VideoDownloader
from cosmos_curator.pipelines.av.pipeline_args import (
    add_common_args,
    validate_choices,
)
from cosmos_curator.pipelines.av.utils.av_pipe_input import (
    extract_session_split_tasks,
    read_session_file,
    write_summary,
)
from cosmos_curator.pipelines.av.utils.run_utils import add_run_to_postrges
from cosmos_curator.pipelines.av.writers.clip_writer_stage import ClipWriterStage


def split(args: argparse.Namespace) -> None:
    """Run the AV split pipeline with profiling and tracing.

    Public entry point that wraps ``_split()`` with ``profiling_scope``
    so that every execution path (CLI, Slurm, NVCF, local launch)
    automatically gets profiling and distributed tracing.

    Args:
        args: The arguments for the pipeline.

    """
    with profiling_scope(args):
        _split(args)


def _split(args: argparse.Namespace) -> None:  # noqa: C901
    """Run the AV split pipeline.

    Args:
        args: The arguments for the pipeline.

    """
    zero_start = time.time()
    # it is possible a filename containing intended sessions are passed
    # process only those sessions
    sessions: list[str] = read_session_file(args.session_file)
    limit = 0 if len(sessions) > 0 else args.limit

    # create a database instance; no connection at this point
    db = PostgresDB.make_from_config(EnvType(args.db_profile)) if args.db_profile is not None else None

    # extract input data
    input_sessions = extract_session_split_tasks(
        db,
        input_prefix=args.input_prefix,
        output_prefix=args.output_prefix,
        source_version=args.source_version,
        encoder=args.encoder,
        target_version=args.clip_version,
        sessions=sessions,
        limit=limit,
    )
    if args.limit > 0:
        input_sessions = input_sessions[: args.limit]
    if len(input_sessions) == 0:
        logger.info("No video sessions to process.")
        return

    run_uuid = uuid.uuid4()
    logger.info(f"About to process {len(input_sessions)} video sessions with run_id={run_uuid}")
    if args.verbose:
        for session in input_sessions[:4]:
            logger.debug(f"{session}")

    stages: list[CuratorStage | CuratorStageSpec] = [
        CuratorStageSpec(
            VideoDownloader(
                output_prefix=args.output_prefix,
                camera_format_id=args.camera_format_id,
                prompt_variants=args.prompt_types,
                verbose=args.verbose,
                log_stats=args.perf_profile,
            ),
            num_workers_per_node=4,
            num_run_attempts_python=5,
        ),
        CuratorStageSpec(
            FixedStrideExtractorStage(
                camera_format_id=args.camera_format_id,
                clip_len_frames=args.fixed_stride_split_frames,
                clip_stride_frames=args.fixed_stride_split_frames,
                limit_clips=args.limit_clips,
                verbose=args.verbose,
                log_stats=args.perf_profile,
            ),
            num_workers_per_node=1,
        ),
        ClipTranscodingStage(
            encoder=args.encoder,
            encoder_threads=args.encoder_threads,
            openh264_bitrate=args.openh264_bitrate,
            encode_batch_size=args.encode_batch_size,
            nb_streams_per_gpu=args.encode_streams_per_gpu,
            verbose=args.verbose,
            log_stats=args.perf_profile,
        ),
    ]

    if not args.dry_run:
        stages.append(
            CuratorStageSpec(
                ClipWriterStage(
                    db,
                    output_prefix=args.output_prefix,
                    run_id=run_uuid,
                    version=args.clip_version,
                    continue_captioning=args.continue_captioning,
                    caption_chunk_size=args.caption_chunk_size,
                    verbose=args.verbose,
                    log_stats=args.perf_profile,
                ),
                num_workers_per_node=8,
                num_run_attempts_python=5,
            )
        )

        # TODO: dry_run with continue_captioning is not supported yet
        if args.continue_captioning:
            stages.extend(
                build_caption_pipeline_stages(
                    args=args,
                    db=db,
                    run_uuid=run_uuid,
                )
            )

    if not args.dry_run and db is not None:
        run_type = "split-caption" if args.continue_captioning else "split"
        extra_info = {
            "camera_format_id": args.camera_format_id,
            "session_file": (str(args.session_file).removesuffix(".txt") if args.session_file is not None else ""),
            "num_sessions": len(input_sessions),
        }
        if args.continue_captioning:
            extra_info["clip_version"] = args.clip_version
            extra_info["caption_version"] = args.caption_version
            extra_info["prompt_types"] = args.prompt_types
        add_run_to_postrges(
            db,
            str(run_uuid),
            run_type,
            args.clip_version,
            extra=extra_info,
        )

    pipeline_start = time.time()
    output_tasks = run_pipeline(
        input_sessions,
        stages=stages,
        args=args,
    )
    if args.perf_profile:
        total_object_size = 0
        for task in output_tasks:
            total_object_size += task.get_major_size()
        logger.info(f"Total object size: {total_object_size:,} bytes")

    input_build_time = (pipeline_start - zero_start) / 60
    pipeline_run_time = (time.time() - pipeline_start) / 60

    total_video_length = write_summary(output_path=args.output_prefix, num_threads=32)

    logger.info(
        f"Split-Transcode pipeline: {input_build_time=:.2f} / "
        f"{pipeline_run_time=:.2f} mins processing "
        f"time for {total_video_length=:.3f} hours of raw videos"
    )


def _setup_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fixed-stride-split-frames",
        type=int,
        default=256,
        help="Duration of clips (in frame count) generated from the fixed stride splitting stage.",
    )
    parser.add_argument(
        "--limit-clips",
        type=int,
        default=0,
        help="Limit the number of clips from each video to generate (for testing).",
    )
    parser.add_argument(
        "--openh264-bitrate",
        type=int,
        default=10,
        help="Bitrate in Mbps for libopenh264 encoder.",
    )
    parser.add_argument(
        "--encoder-threads",
        type=int,
        default=32,
        help="Number of threads to use for CPU encoding.",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=16,
        help="Number of clips to encode in parallel.",
    )
    parser.add_argument(
        "--encode-streams-per-gpu",
        type=int,
        default=1,
        help="Number of concurrent encoding streams per GPU.",
    )
    parser.add_argument(
        "--continue-captioning",
        action="store_true",
        help="Continue captioning after split.",
    )
    add_common_args(parser, "split")


def nvcf_run_av_split(args: argparse.Namespace) -> None:
    """Run the split pipeline.

    Args:
        args: The arguments for the pipeline.

    """
    args_utils.fill_default_args(args, _setup_parser)
    args.prompt_types = validate_choices(
        args.prompt_types,
        {"default", *VRI_PROMPTS},
        "prompt_types",
    )
    cli_run_split(args)


def cli_run_split(args: argparse.Namespace) -> None:
    """Run the split pipeline.

    Args:
        args: The arguments for the pipeline.

    """
    split(args)


def add_split_command(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> argparse.ArgumentParser:
    """Add the split command to the parser.

    Args:
        subparsers: The subparsers for the parser.

    Returns:
        The parser.

    """
    parser = subparsers.add_parser(
        "split",
        help="Split videos into clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(func=cli_run_split)
    _setup_parser(parser)
    return parser  # type: ignore[no-any-return]
