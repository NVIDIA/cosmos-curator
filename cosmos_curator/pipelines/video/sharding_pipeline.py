# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Ray pipelines.

Which:
  - Download splitted & annotated video clips
  - Optionally filter clips based on semantic dedup results
  - Generate T5 embedding for captions
  - Pack clips into webdataset
"""

import argparse
import collections
import pathlib
import time
from collections.abc import Generator, Iterable

from loguru import logger

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.core.utils.config import args_utils
from cosmos_curator.core.utils.dataset import dimensions, webdataset_utils
from cosmos_curator.core.utils.misc import grouping
from cosmos_curator.core.utils.storage.storage_client import StoragePrefix
from cosmos_curator.core.utils.storage.storage_utils import (
    create_path,
    get_directories_relative,
    get_full_path,
    get_storage_client,
    verify_path,
)
from cosmos_curator.pipelines.common_pipeline_settings import (
    composite_from_namespace,
    composite_profiling_scope,
    sync_common_from_namespace,
)
from cosmos_curator.pipelines.pipeline_args import add_common_args
from cosmos_curator.pipelines.video.captioning.captioning_stages import T5StageForShard
from cosmos_curator.pipelines.video.read_write.download_stages import DownloadPackUpload
from cosmos_curator.pipelines.video.read_write.summary_writers import write_shard_summary
from cosmos_curator.pipelines.video.shard_pipeline_settings import (
    MIN_CLIPS_PER_TAR_DEFAULT,
    ShardPipelineSettings,
    add_shard_args,
)
from cosmos_curator.pipelines.video.utils.data_model import (
    ClipSample,
    ShardPipeTask,
)
from cosmos_curator.pipelines.video.utils.video_pipe_input import (
    extract_shard_tasks,
    filter_shard_tasks_by_semantic_dedup,
)


def _group_samples_by_bin(
    samples: Iterable[ClipSample],
) -> dict[dimensions.ResolutionAspectRatioFrames | None, list[ClipSample]]:
    out: dict[dimensions.ResolutionAspectRatioFrames | None, list[ClipSample]] = collections.defaultdict(list)
    bin_spec = dimensions.ResolutionAspectRatioFramesBinsSpec.for_standard_video_datasets()
    count = 0
    for sample in samples:
        framerate = getattr(sample, "framerate", None)
        if framerate and framerate > 0:
            lbin = bin_spec.find_appropriate_bin(
                dimensions.Dimensions(sample.width, sample.height), float(sample.num_frames) / sample.framerate
            )
        else:
            logger.warning(f"Invalid framerate={framerate} for sample; assigning to None bin.")
            lbin = None
        out[lbin].append(sample)
        count += 1
    logger.info(f"Found {count} total samples in {len(out)} bins.")
    return out


def _group_samples_by_size(
    samples: list[ClipSample],
    target_size_bytes: int,
    *,
    drop_small_shards: bool,
    min_clips_per_tar: int = MIN_CLIPS_PER_TAR_DEFAULT,
) -> Generator[list[ClipSample], None, None]:
    current_size = 0
    out: list[ClipSample] = []

    for sample in samples:
        if not out:
            out.append(sample)
            current_size = sample.num_bytes
        elif current_size + sample.num_bytes > target_size_bytes:
            yield out
            out = [sample]
            current_size = sample.num_bytes
        else:
            out.append(sample)
            current_size += sample.num_bytes

    if out and not (drop_small_shards and len(out) < min_clips_per_tar):
        yield out


def _group_samples_into_tasks(  # noqa: PLR0913
    samples: Iterable[ClipSample],
    *,
    drop_small_shards: bool,
    max_tars_per_part: int,
    target_tar_size_bytes: int,
    min_clips_per_tar: int,
    output_path: str,
    output_s3_profile_name: str,
) -> tuple[list[ShardPipeTask], list[StoragePrefix | pathlib.Path], int]:
    tasks: list[ShardPipeTask] = []
    all_bins: list[StoragePrefix | pathlib.Path] = []
    num_dropped_samples: int = 0
    grouped_by_bin = _group_samples_by_bin(samples)
    client_output = get_storage_client(output_path, profile_name=output_s3_profile_name)
    for lbin, binned_samples in grouped_by_bin.items():
        sample_count = len(binned_samples)
        if lbin is None:
            logger.warning(f"Found {sample_count} samples which do not correspond to a lbin. Ignoring them ...")
            num_dropped_samples += sample_count
            continue

        path_for_bin = get_full_path(output_path, lbin.to_path_string())
        logger.info(f"Inspecting bin {lbin} with {sample_count} samples at {path_for_bin}.")
        all_bins.append(path_for_bin)

        path_for_video = get_full_path(path_for_bin, "video")
        part_dirs = get_directories_relative(str(path_for_video), client_output)

        logger.info(f"Current parts under {part_dirs}:")
        for part_dir in part_dirs:
            logger.info(part_dir)

        starting_part_num = (
            max(
                [webdataset_utils.get_part_num_from_path_str(str(x)) for x in part_dirs],
                default=-1,
            )
            + 1
        )
        logger.info(f"Starting part number: {starting_part_num}")

        for part_idx, tar_group in enumerate(
            grouping.split_by_chunk_size(
                _group_samples_by_size(
                    binned_samples,
                    target_tar_size_bytes,
                    drop_small_shards=drop_small_shards,
                    min_clips_per_tar=min_clips_per_tar,
                ),
                max_tars_per_part,
            ),
        ):
            part_num = starting_part_num + part_idx
            for tar_idx, tar_samples in enumerate(tar_group):
                path_for_tar = (
                    webdataset_utils.make_part_path_str(part_num) + "/" + webdataset_utils.make_tar_path_str(tar_idx)
                )
                output_object_video = get_full_path(path_for_video, path_for_tar)
                output_object_metas = get_full_path(path_for_bin, "metas", path_for_tar)
                output_object_t5_xxl = get_full_path(path_for_bin, "t5_xxl", path_for_tar)
                tasks.append(
                    ShardPipeTask(
                        str(path_for_bin),
                        part_num,
                        tar_samples,
                        output_object_video,
                        output_object_metas,
                        output_object_t5_xxl,
                        key_count=0,
                    ),
                )
    logger.info(f"Created {len(tasks)} tasks in {len(all_bins)} shards:")
    for task in tasks:
        logger.info(f"part={task.part_num} output={task.output_tar_video}, samples={len(task.samples)}")
    return tasks, all_bins, num_dropped_samples


def shard(args: argparse.Namespace) -> None:
    """Run the shard pipeline with profiling and tracing.

    Public entry point that wraps ``_shard()`` with ``profiling_scope``
    so that every execution path (CLI, Slurm, NVCF, local launch)
    automatically gets profiling and distributed tracing.

    Args:
        args: Command line arguments.

    """
    settings = composite_from_namespace(ShardPipelineSettings, args)
    with composite_profiling_scope(settings) as profiling_ns:
        _shard(settings, profiling_ns)


def _shard(settings: ShardPipelineSettings, profiling_args_ns: argparse.Namespace) -> None:
    """Run the shard pipeline.

    This function orchestrates the entire pipeline, from input validation to output generation.
    It validates input arguments, builds input data, and executes the pipeline stages.

    Args:
        settings: Validated shard pipeline settings.
        profiling_args_ns: Flat CLI namespace for profiling (same field names as *settings*)

    """
    start_time = time.time()
    # validate input arguments
    output_dataset_path = str(get_full_path(settings.output_dataset_path, settings.annotation_version))
    verify_path(settings.input_clip_path)
    verify_path(settings.output_dataset_path, level=1)
    create_path(settings.output_dataset_path)

    # get input samples
    samples = extract_shard_tasks(
        settings.input_clip_path,
        output_dataset_path,
        settings.common.input_s3_profile_name,
        settings.common.output_s3_profile_name,
        settings.annotation_version,
        verbose=settings.common.verbose,
    )
    logger.info(f"Found {len(samples)} samples under input path {settings.input_clip_path}.")

    if settings.input_semantic_dedup_path is not None:
        samples = filter_shard_tasks_by_semantic_dedup(
            samples,
            settings.input_semantic_dedup_path,
            settings.input_semantic_dedup_s3_profile_name,
            settings.semantic_dedup_epsilon,
            verbose=settings.common.verbose,
        )
        logger.info(f"After semantic deduplication, {len(samples)} samples remain.")

    # Convert target tar size from MB to bytes
    target_tar_size_bytes = settings.target_tar_size_mb * 1024 * 1024

    tasks, all_bins, num_dropped_samples = _group_samples_into_tasks(
        samples,
        drop_small_shards=settings.drop_small_shards,
        max_tars_per_part=settings.max_tars_per_part,
        target_tar_size_bytes=target_tar_size_bytes,
        min_clips_per_tar=settings.min_clips_per_tar,
        output_path=output_dataset_path,
        output_s3_profile_name=settings.common.output_s3_profile_name,
    )
    logger.info(f"Dropped {num_dropped_samples} samples during sharding process.")
    if len(tasks) == 0:
        logger.warning("No tasks to process. Exiting ...")
        return

    stages: list[CuratorStage | CuratorStageSpec] = [
        T5StageForShard(
            caption_fields=[f"{settings.captioning_algorithm}_caption"],
            verbose=settings.common.verbose,
            log_stats=settings.common.perf_profile,
        ),
        CuratorStageSpec(
            DownloadPackUpload(
                input_path=settings.input_clip_path,
                output_path=output_dataset_path,
                input_s3_profile_name=settings.common.input_s3_profile_name,
                output_s3_profile_name=settings.common.output_s3_profile_name,
                verbose=settings.common.verbose,
                log_stats=settings.common.perf_profile,
            ),
            num_workers_per_node=8,
        ),
    ]

    output_packets: list[ShardPipeTask] = run_pipeline(
        tasks,
        stages,
        args=profiling_args_ns,
    )
    sync_common_from_namespace(settings, profiling_args_ns)
    if settings.common.perf_profile:
        total_object_size = 0
        for packet in output_packets:
            total_object_size += packet.get_major_size()
        logger.info(f"Total object size: {total_object_size:,} bytes")

    write_shard_summary(
        output_dataset_path,
        settings.output_dataset_path,
        settings.common.output_s3_profile_name,
        all_bins,
        settings.max_tars_per_part,
        output_packets,
        perf_profile=settings.common.perf_profile,
    )

    elapsed_time = (time.time() - start_time) / 60
    logger.info(f"Embedding-Shard-Webdataset pipeline completed in {elapsed_time:.2f} minutes")


def _setup_parser(parser: argparse.ArgumentParser) -> None:
    """Set up the parser for the shard pipeline.

    Registers shard-only flags then shared flags (same order as settings construction in :func:`shard`).

    Args:
        parser: The parser to add arguments to.

    """
    add_shard_args(parser)
    add_common_args(parser)


def nvcf_run_shard(args: argparse.Namespace) -> None:
    """Run the shard pipeline.

    This function orchestrates the entire pipeline, from input validation to output generation.
    It validates input arguments, builds input data, and executes the pipeline stages.

    Args:
        args: Command line arguments.

    """
    args_utils.fill_default_args(args, _setup_parser)
    cli_run_shard(args)


def cli_run_shard(args: argparse.Namespace) -> None:
    """Run the shard pipeline in CLI mode.

    Args:
        args: Command line arguments.

    """
    shard(args)


def add_shard_command(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> argparse.ArgumentParser:
    """Add shard command to the CLI parser.

    Args:
        subparsers: Subparsers object to add the command to.

    Returns:
        The configured parser for the shard command.

    """
    parser = subparsers.add_parser(
        "shard",
        help="Shard clips into webdatasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(func=cli_run_shard)
    _setup_parser(parser)
    return parser  # type: ignore[no-any-return]
