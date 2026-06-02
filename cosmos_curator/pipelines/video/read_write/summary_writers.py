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
"""Write metadata for clips to DB."""

import pathlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import attrs
from loguru import logger

from cosmos_curator.core.utils.infra.performance_utils import (
    dump_and_write_perf_stats,
)
from cosmos_curator.core.utils.misc import grouping
from cosmos_curator.core.utils.misc.retry_utils import do_with_retries
from cosmos_curator.core.utils.storage import storage_client, storage_utils
from cosmos_curator.core.utils.storage.storage_utils import (
    get_directories_relative,
    get_files_relative,
    get_full_path,
    path_exists,
    read_json_file,
)
from cosmos_curator.core.utils.storage.writer_utils import (
    write_csv,
    write_json,
)
from cosmos_curator.pipelines.video.read_write.metadata_writer_stage import ClipWriterStage
from cosmos_curator.pipelines.video.utils.data_model import (
    CaptionQualityStats,
    ShardPipeTask,
    SplitPipeTask,
)

_SUMMARIZE_NUM_WORKERS = 32
_SUMMARIZE_NUM_CHUNKS = 128
_CAPTION_QUALITY_STATS_FILENAME = "caption_quality_stats.json"


@attrs.define
class ProcessedVideoMetadata:
    """Container for processed video metadata and clip chunk information.

    This class stores video metadata and a list of clip chunks that have been processed
    through the video processing pipeline.
    """

    video_metadata: dict[str, Any] | None = None
    clip_chunks: list[dict[str, Any]] = attrs.Factory(list)


@attrs.define(frozen=True)
class CaptionQualityStatsWriteOptions:
    """Options that gate caption quality stats emission."""

    generate_captions: bool
    caption_quality_stats_enabled: bool
    caption_models: list[str]
    multi_cam: bool


def _aggregate_caption_quality_stats(
    all_video_data: dict[str, ProcessedVideoMetadata],
) -> tuple[CaptionQualityStats | None, str | None]:
    """Aggregate caption stats from chunk metadata already read for summary writing."""
    stats = CaptionQualityStats()
    for input_video, data in all_video_data.items():
        if data.video_metadata is None:
            # Match summary.json: unprocessed inputs are represented as processed=false
            # and contribute no clip or caption-window counters.
            continue

        num_clip_chunks = data.video_metadata.get("num_clip_chunks", 0)
        if len(data.clip_chunks) != num_clip_chunks:
            return None, f"{input_video} has missing clip chunk metadata"

        for clip_chunk in data.clip_chunks:
            chunk_stats = clip_chunk.get("caption_quality_stats")
            if chunk_stats is None:
                return None, f"{input_video} clip chunk is missing caption_quality_stats"
            try:
                stats.combine(CaptionQualityStats.from_dict(chunk_stats))
            except (TypeError, ValueError) as exc:
                return None, f"{input_video} clip chunk has invalid caption_quality_stats: {exc}"

    errors = stats.validation_errors()
    if errors:
        return None, "; ".join(errors)
    return stats, None


def _clear_caption_quality_stats(output_path: str, client_output: storage_client.StorageClient | None) -> None:
    def _delete_if_exists() -> None:
        dest = get_full_path(output_path, _CAPTION_QUALITY_STATS_FILENAME)
        if not path_exists(dest, client_output):
            return

        if isinstance(dest, storage_client.StoragePrefix):
            assert client_output is not None
            client_output.delete_object(dest)
        else:
            pathlib.Path(dest).unlink(missing_ok=True)

    do_with_retries(_delete_if_exists)


def _write_caption_quality_stats(
    *,
    output_path: str,
    client_output: storage_client.StorageClient | None,
    all_video_data: dict[str, ProcessedVideoMetadata],
    options: CaptionQualityStatsWriteOptions,
) -> None:
    if not options.generate_captions or not options.caption_quality_stats_enabled:
        _clear_caption_quality_stats(output_path, client_output)
        return

    if options.multi_cam:
        logger.warning("Skipping caption_quality_stats.json: multi-camera caption quality aggregation is not supported")
        _clear_caption_quality_stats(output_path, client_output)
        return

    if len(options.caption_models) != 1:
        logger.warning(
            "Skipping caption_quality_stats.json: expected exactly one caption model, got {}",
            len(options.caption_models),
        )
        _clear_caption_quality_stats(output_path, client_output)
        return

    caption_quality_stats, error = _aggregate_caption_quality_stats(all_video_data)
    if caption_quality_stats is None:
        logger.warning(f"Skipping caption_quality_stats.json: {error}")
        _clear_caption_quality_stats(output_path, client_output)
        return

    caption_quality_stats_dest = get_full_path(output_path, _CAPTION_QUALITY_STATS_FILENAME)

    def _write_artifact() -> None:
        write_json(
            caption_quality_stats.to_dict(include_schema=True),
            caption_quality_stats_dest,
            "caption quality stats",
            "all videos",
            verbose=True,
            client=client_output,
            backup_and_overwrite=True,
        )
        logger.info(f"Wrote caption quality stats to {caption_quality_stats_dest}")

    do_with_retries(_write_artifact)


def _worker_read_video_metadata(  # noqa: C901
    output_path: str,
    output_s3_profile_name: str | None,
    input_videos_relative: list[str],
    limit: int,
    *,
    verbose: bool = False,
) -> dict[str, ProcessedVideoMetadata]:
    all_video_data: dict[str, ProcessedVideoMetadata] = {}
    profile_name = output_s3_profile_name or "default"
    client = storage_utils.get_storage_client(output_path, profile_name=profile_name)
    for input_video in input_videos_relative:
        video_metadata_path = get_full_path(
            ClipWriterStage.get_output_path_processed_videos(output_path),
            f"{input_video}.json",
        )

        def func_to_call(
            video_metadata_path: storage_client.StoragePrefix | pathlib.Path = video_metadata_path,
        ) -> dict[str, Any]:
            if verbose:
                logger.info(f"Reading video metadata from {video_metadata_path} ...")
            return read_json_file(video_metadata_path, client)

        all_video_data[input_video] = ProcessedVideoMetadata()
        if not path_exists(video_metadata_path, client):
            if limit == 0:
                logger.error(f"video process-record {input_video} not found ???")
        else:
            data = do_with_retries(func_to_call)
            all_video_data[input_video].video_metadata = data

            num_clip_chunks = data.get("num_clip_chunks", 0)
            if num_clip_chunks == 0:
                logger.warning(f"video process-record {input_video} has no clip")
                continue
            for idx in range(num_clip_chunks):
                clip_chunk_path = get_full_path(
                    ClipWriterStage.get_output_path_processed_clip_chunks(output_path),
                    f"{input_video}_{idx}.json",
                )

                def func_to_call2(
                    clip_chunk_path: storage_client.StoragePrefix | pathlib.Path = clip_chunk_path,
                ) -> dict[str, Any]:
                    if verbose:
                        logger.info(f"Reading clip chunk from {clip_chunk_path} ...")
                    return read_json_file(clip_chunk_path, client)

                if not path_exists(clip_chunk_path, client):
                    logger.error(f"clip chunk record {clip_chunk_path} not found ???")
                    continue

                all_video_data[input_video].clip_chunks.append(do_with_retries(func_to_call2))
    return all_video_data


def _write_split_result_summary(  # noqa: PLR0913, C901
    input_path: str,
    input_videos_relative: list[str],
    num_input_videos_selected: int,
    output_path: str,
    output_s3_profile_name: str,
    *,
    embedding_algorithm: str,
    limit: int,
    pipeline_run_time: float = 0.0,
    write_all_caption_json: bool = False,
    video_bytes: int = 0,
    multi_cam: bool = False,
    num_remuxed_videos: int = 0,
    generate_captions: bool = False,
    caption_quality_stats_enabled: bool = True,
    caption_models: list[str] | None = None,
) -> None:
    logger.info(f"Starting to summarize data in {output_path} ...")
    client_output = storage_utils.get_storage_client(
        output_path,
        profile_name=output_s3_profile_name,
        can_overwrite=True,
        can_delete=True,
    )

    if not multi_cam:
        processed_sessions = get_files_relative(
            ClipWriterStage.get_output_path_processed_videos(output_path), client_output
        )
        logger.info(f"Summarize: found {len(processed_sessions)} processed videos")
    else:
        processed_sessions = get_directories_relative(
            ClipWriterStage.get_output_path_processed_videos(output_path), client_output
        )
        logger.info(f"Summarize: found {len(processed_sessions)} processed sessions")

    clip_stats_keys = [
        "num_clips_filtered_by_motion",
        "num_clips_filtered_by_aesthetic",
        "num_clips_filtered_by_qwen_classifier",
        "num_clips_filtered_by_qwen_semantic",
        "num_clips_filtered_by_artificial_text",
        "num_clips_passed",
        "num_clips_transcoded",
        "num_clips_with_embeddings",
        "num_clips_with_caption",
        "num_caption_windows",
        "num_clips_with_webp",
    ]
    summary_data: dict[str, Any] = {
        "num_input_videos": len(input_videos_relative),
        "num_input_videos_selected": num_input_videos_selected,
        "num_processed_videos": len(processed_sessions),
        "embedding_algorithm": embedding_algorithm,
        "total_video_duration": 0,
        "total_clip_duration": 0,
        "max_clip_duration": 0,
        "pipeline_run_time": pipeline_run_time,
        "total_video_bytes": video_bytes,
        "num_remuxed_videos": num_remuxed_videos,
        "total_prompt_tokens": 0,
        "total_output_tokens": 0,
    }

    for key in clip_stats_keys:
        summary_data[f"total_{key}"] = 0

    all_video_data = _read_all_video_metadata_parallel(
        output_path,
        output_s3_profile_name,
        input_videos_relative,
        limit,
    )
    logger.info(f"Summarize: read metadata for {len(all_video_data)} videos")
    for input_video, data in all_video_data.items():
        summary_data[input_video] = {
            "source_video": str(get_full_path(input_path, input_video)),
        }
        if data.video_metadata is None:
            summary_data[input_video]["processed"] = False
            continue

        summary_data[input_video]["video_uuid"] = data.video_metadata.get("video_uuid", "N/A")
        summary_data[input_video]["num_clip_chunks"] = len(data.clip_chunks)
        summary_data[input_video]["num_total_clips"] = data.video_metadata.get("num_total_clips", "N/A")
        summary_data[input_video]["clips"] = []
        summary_data[input_video]["filtered_clips"] = []
        for key in clip_stats_keys:
            summary_data[input_video][key] = 0
        for clip_chunk in data.clip_chunks:
            for key in clip_stats_keys:
                summary_data[input_video][key] += clip_chunk.get(key, 0)
                summary_data[f"total_{key}"] += clip_chunk.get(key, 0)
            summary_data[input_video]["clips"].extend(clip_chunk.get("clips", []))
            summary_data[input_video]["filtered_clips"].extend(clip_chunk.get("filtered_clips", []))
            summary_data["total_clip_duration"] += clip_chunk.get("total_clip_duration", 0)
            summary_data["max_clip_duration"] = max(
                summary_data["max_clip_duration"],
                clip_chunk.get("max_clip_duration", 0),
            )
            summary_data["total_prompt_tokens"] += clip_chunk.get("total_prompt_tokens", 0)
            summary_data["total_output_tokens"] += clip_chunk.get("total_output_tokens", 0)
        summary_data["total_video_duration"] += data.video_metadata.get("duration", 0)

    # Compute and log captioning throughput metrics
    total_output = summary_data["total_output_tokens"]
    total_prompt = summary_data["total_prompt_tokens"]
    num_caption_windows = summary_data.get("total_num_caption_windows", 0)
    if total_output > 0 and pipeline_run_time > 0:
        tokens_per_s = round(total_output / (pipeline_run_time * 60), 1)
        summary_data["output_tokens_per_s"] = tokens_per_s
        avg_prompt = total_prompt // num_caption_windows if num_caption_windows else 0
        avg_output = total_output // num_caption_windows if num_caption_windows else 0
        logger.info(
            "\n"
            "  Captioning throughput\n"
            "  -------------------------------------\n"
            "  total prompt tokens:      {:>10,}\n"
            "  total output tokens:      {:>10,}\n"
            "  total caption windows:    {:>10,}\n"
            "  avg prompt tokens/window: {:>10,}\n"
            "  avg output tokens/window: {:>10,}\n"
            "  output tokens/s:          {:>10,.1f}\n"
            "  -------------------------------------",
            total_prompt,
            total_output,
            num_caption_windows,
            avg_prompt,
            avg_output,
            tokens_per_s,
        )

    def func_write_summary() -> None:
        summary_dest = get_full_path(output_path, "summary.json")
        write_json(
            summary_data,
            summary_dest,
            "summary",
            "all videos",
            verbose=True,
            client=client_output,
            backup_and_overwrite=True,
        )
        logger.info(f"Wrote summary to {summary_dest}")

    do_with_retries(func_write_summary)

    _write_caption_quality_stats(
        output_path=output_path,
        client_output=client_output,
        all_video_data=all_video_data,
        options=CaptionQualityStatsWriteOptions(
            generate_captions=generate_captions,
            caption_quality_stats_enabled=caption_quality_stats_enabled,
            caption_models=caption_models if caption_models is not None else [],
            multi_cam=multi_cam,
        ),
    )

    if write_all_caption_json:
        _write_all_window_captions(
            output_path=output_path,
            client_output=client_output,
            all_video_data=all_video_data,
        )


def write_split_summary(  # noqa: PLR0913
    input_path: str,
    input_videos_relative: list[str],
    num_input_videos_selected: int,
    output_path: str,
    output_s3_profile_name: str,
    output_tasks: list[SplitPipeTask],
    embedding_algorithm: str,
    limit: int,
    *,
    perf_profile: bool = False,
    pipeline_run_time: float = 0.0,
    write_all_caption_json: bool = False,
    multi_cam: bool = False,
    generate_captions: bool = False,
    caption_quality_stats_enabled: bool = True,
    caption_models: list[str] | None = None,
) -> None:
    """Write summary of split pipeline results including job stats and performance metrics.

    Args:
        input_path: Path to input videos.
        input_videos_relative: List of relative paths to input videos.
        num_input_videos_selected: Number of input videos selected for this run after applying skip/limit semantics.
        output_path: Path to write output files.
        output_s3_profile_name: S3 profile name for output.
        output_tasks: List of completed split pipeline tasks.
        embedding_algorithm: Name of the embedding algorithm used.
        limit: Maximum number of videos to process.
        perf_profile: Whether to write performance statistics.
        pipeline_run_time: Total runtime of the pipeline in minutes.
        write_all_caption_json: Whether to write all caption JSON file.
        multi_cam: Whether the pipeline is running in multi-cam mode.
        generate_captions: Whether subject caption generation was enabled.
        caption_quality_stats_enabled: Whether to emit the run-level caption quality stats artifact.
        caption_models: Configured subject caption fields.

    """
    # dump and write job summary
    video_bytes = sum(v.metadata.size for task in output_tasks for v in task.videos if v.metadata.size is not None)
    num_remuxed = sum(v.was_remuxed for task in output_tasks for v in task.videos if v.clip_chunk_index == 0)

    _write_split_result_summary(
        input_path,
        input_videos_relative,
        num_input_videos_selected,
        output_path,
        output_s3_profile_name,
        embedding_algorithm=embedding_algorithm,
        limit=limit,
        pipeline_run_time=pipeline_run_time,
        write_all_caption_json=write_all_caption_json,
        video_bytes=video_bytes,
        multi_cam=multi_cam,
        num_remuxed_videos=num_remuxed,
        generate_captions=generate_captions,
        caption_quality_stats_enabled=caption_quality_stats_enabled,
        caption_models=caption_models,
    )
    # dump and write performance stats
    if perf_profile:
        dump_and_write_perf_stats(
            [task.stage_perf for task in output_tasks],
            output_path,
            output_s3_profile_name,
        )


def write_shard_summary(  # noqa: PLR0913
    output_path: str,
    raw_output_path: str,
    output_s3_profile_name: str,
    all_bins: list[storage_client.StoragePrefix | pathlib.Path],
    max_tars_per_part: int,
    output_tasks: list[ShardPipeTask],
    *,
    perf_profile: bool = False,
) -> None:
    """Write summary of shard pipeline results including job stats and performance metrics.

    Args:
        output_path: Path to write output files.
        raw_output_path: Path to write raw output files.
        output_s3_profile_name: S3 profile name for output.
        all_bins: List of storage prefixes for all bins.
        max_tars_per_part: Maximum number of tars per part.
        output_tasks: List of completed shard pipeline tasks.
        perf_profile: Whether to write performance statistics.

    """
    # dump and write job summary
    client = storage_utils.get_storage_client(output_path, profile_name=output_s3_profile_name)
    # get per-bin total key count
    key_counts = {}
    for task in output_tasks:
        if task.bin_path not in key_counts:
            key_counts[task.bin_path] = 0
        key_counts[task.bin_path] += task.key_count
    # write wdinfo.json for each bin
    all_wdinfo = []
    for lbin in all_bins:
        bin_prefix = lbin.prefix if isinstance(lbin, storage_client.StoragePrefix) else str(lbin)
        dest = get_full_path(lbin, "wdinfo.json")
        video_path = str(get_full_path(lbin, "video"))
        data = {
            "data_keys": ["video", "t5_xxl", "metas"],
            "chunk_size": max_tars_per_part,
            "data_list": [str(x) for x in get_files_relative(video_path, client)],
            "root": bin_prefix,
            "total_key_count": key_counts[str(lbin)],
        }

        def write_wdinfo(
            data: dict = data,  # type: ignore[type-arg]
            dest: storage_client.StoragePrefix | pathlib.Path = dest,
            lbin: storage_client.StoragePrefix | pathlib.Path = lbin,
        ) -> None:
            write_json(data, dest, "wdinfo", str(lbin), verbose=True, client=client)

        do_with_retries(write_wdinfo)
        all_wdinfo.append([str(dest)])

    # write a top-level csv
    def write_wdinfo_list() -> None:
        summary_dest = get_full_path(output_path, "wdinfo_list.csv")
        write_csv(
            summary_dest,
            "wdinfo list",
            "all bins",
            all_wdinfo,
            verbose=True,
            client=client,
        )

    do_with_retries(write_wdinfo_list)
    # dump and write performance stats
    if perf_profile:
        dump_and_write_perf_stats(
            [task.stage_perf for task in output_tasks],
            raw_output_path,
            output_s3_profile_name,
        )


def _read_all_video_metadata_parallel(
    output_path: str,
    output_s3_profile_name: str | None,
    input_videos_relative: list[str],
    limit: int,
) -> dict[str, ProcessedVideoMetadata]:
    """Read per-video metadata using a thread pool for better IO throughput."""
    all_video_data: dict[str, ProcessedVideoMetadata] = {}
    futures = []
    with ThreadPoolExecutor(max_workers=_SUMMARIZE_NUM_WORKERS) as executor:
        futures = [
            executor.submit(
                _worker_read_video_metadata,
                output_path,
                output_s3_profile_name,
                chunk,
                limit,
            )
            for chunk in grouping.split_into_n_chunks(input_videos_relative, _SUMMARIZE_NUM_CHUNKS)
        ]
        for fut in futures:
            all_video_data.update(fut.result())
    return all_video_data


def _write_all_window_captions(  # noqa: PLR0913
    *,
    output_path: str,
    client_output: storage_client.StorageClient | None,
    all_video_data: dict[str, ProcessedVideoMetadata] | None = None,
    output_s3_profile_name: str | None = None,
    input_videos_relative: list[str] | None = None,
    limit: int | None = None,
) -> None:
    """Gather all window captions data and write it to a JSON file."""
    if all_video_data is None:
        # this the hacky managed service zip upload/download path
        assert input_videos_relative is not None
        assert limit is not None
        all_video_data = _read_all_video_metadata_parallel(
            output_path,
            output_s3_profile_name,
            input_videos_relative,
            limit,
        )

    all_window_captions_data: dict[str, Any] = {}
    for input_video, data in all_video_data.items():
        if data.video_metadata is None:
            continue
        all_window_captions_data[input_video] = {}
        for clip_chunk in data.clip_chunks:
            for clip_id, clip_data in clip_chunk.get("all_windows", {}).items():
                if clip_id not in all_window_captions_data[input_video]:
                    all_window_captions_data[input_video][clip_id] = {}
                all_window_captions_data[input_video][clip_id].update(clip_data)

    def _write() -> None:
        dest = get_full_path(output_path, "v0", "all_window_captions.json")
        write_json(
            all_window_captions_data,
            dest,
            "all window captions",
            "all videos",
            verbose=True,
            client=client_output,
            backup_and_overwrite=True,
        )
        logger.info(f"Wrote all window captions to {dest}")

    do_with_retries(_write)
