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

"""Video Pipe Input."""

import json
import pathlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from loguru import logger
from six import BytesIO

from cosmos_curator.core.utils.misc.uuid_utils import is_uuid
from cosmos_curator.core.utils.storage.storage_client import StorageClient, StoragePrefix
from cosmos_curator.core.utils.storage.storage_utils import (
    get_directories_relative,
    get_files_relative,
    get_full_path,
    get_storage_client,
    is_parquet_file,
    path_exists,
    read_bytes,
    read_json_file,
)
from cosmos_curator.pipelines.video.utils.data_model import ClipSample, SplitPipeTask, Video


def _check_output_path(output_path: str, client: StorageClient | None) -> None:
    if output_path is not None and path_exists(get_full_path(output_path, "summary.json"), client):
        logger.warning(f"Output path {output_path} already concluded with a summary.json file")


def _find_processed_video_jsons(
    output_video_path: str,
    client: StorageClient | None,
    *,
    verbose: bool,
) -> list[str]:
    processed_video_jsons = get_files_relative(output_video_path, client)
    logger.info(f"Found {len(processed_video_jsons)} processed videos in {output_video_path}")
    if verbose:
        for video in processed_video_jsons:
            logger.debug(video)
    return processed_video_jsons


def _worker_verify_processed_video(
    processed_video_json: str,
    output_video_path: str,
    output_clip_chunk_path: str,
    client: StorageClient | None,
) -> str | None:
    # read the processed video json
    processed_video_json_path = get_full_path(output_video_path, processed_video_json)
    try:
        data = read_json_file(processed_video_json_path, client)
        num_clip_chunks = int(data["num_clip_chunks"])
        for idx in range(num_clip_chunks):
            processed_clip_chunk_path = processed_video_json.removesuffix(".json") + f"_{idx}.json"
            clip_chunk_path = get_full_path(output_clip_chunk_path, processed_clip_chunk_path)
            if not path_exists(clip_chunk_path, client):
                logger.debug(f"Semi-processed video {processed_video_json} missing chunk-{idx}")
                return None
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to read processed video json {processed_video_json_path}: {e}")
        return None
    else:
        return processed_video_json


def _read_video_list_json(
    input_path: str,
    input_video_list_json_path: str,
    input_video_list_s3_profile_name: str,
) -> list[str]:
    input_videos = []
    client = get_storage_client(input_video_list_json_path, profile_name=input_video_list_s3_profile_name)
    try:
        data = read_json_file(input_video_list_json_path, client)
        listed_input_videos = [str(x) for x in data]
    except Exception as e:
        logger.exception(f"Failed to read input video list from {input_video_list_json_path}: {e}")
        raise

    for video_path in listed_input_videos:
        _input_path = input_path.rstrip("/") + "/"
        if not video_path.startswith(_input_path):
            error_msg = f"Input video {video_path} is not in {_input_path}"
            logger.exception(error_msg)
            raise ValueError(error_msg)
        input_videos.append(video_path[len(_input_path) :])

    return input_videos


def extract_single_cam_split_tasks(  # noqa: PLR0913
    input_path: str,
    input_video_list_json_path: str | None,
    output_path: str,
    output_video_path: str,
    output_clip_chunk_path: str,
    input_s3_profile_name: str,
    input_video_list_s3_profile_name: str,
    output_s3_profile_name: str,
    limit: int = 0,
    *,
    verbose: bool = False,
) -> tuple[list[Video], list[str], int]:
    """Extract list of input video paths from the input S3 or local path."""
    client_input = get_storage_client(input_path, profile_name=input_s3_profile_name)
    client_output = get_storage_client(output_path, profile_name=output_s3_profile_name)

    _check_output_path(output_path, client_output)

    # find already processed videos
    processed_video_jsons = _find_processed_video_jsons(output_video_path, client_output, verbose=verbose)
    # verify if all chunks are processed
    fully_processed_video_jsons = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(
                _worker_verify_processed_video,
                processed_video_json,
                output_video_path,
                output_clip_chunk_path,
                client_output,
            )
            for processed_video_json in processed_video_jsons
        ]
        for future in futures:
            processed_video_json = future.result()
            if processed_video_json is not None:
                fully_processed_video_jsons.append(processed_video_json)
    # build the final set of processed videos
    processed_videos = {x.removesuffix(".json") for x in fully_processed_video_jsons}
    logger.info(f"Found {len(processed_videos)} fully processed videos in {output_video_path}")

    # input
    if input_video_list_json_path is not None:
        input_videos = _read_video_list_json(input_path, input_video_list_json_path, input_video_list_s3_profile_name)
    else:
        _limit = 0 if limit == 0 else len(processed_videos) + limit
        input_videos = get_files_relative(input_path, client_input, _limit)

    # apply filter func
    all_videos = list(input_videos)
    logger.info(f"Found {len(all_videos)} input videos in {input_path}")
    if verbose:
        for video in all_videos:
            logger.debug(video)

    # remove already processed videos
    skipped_videos = [x for x in all_videos if x in processed_videos]
    if skipped_videos and verbose:
        logger.info(f"Skipping {len(skipped_videos)} already-processed video(s)")

    raw_videos = [x for x in all_videos if x not in processed_videos]
    # apply limit
    if limit > 0:
        raw_videos = raw_videos[:limit]
    # prepare the final list of videos (single-cam: relative_path="" so clips go to clips/{uuid}.mp4)
    return (
        [Video(get_full_path(input_path, x), relative_path="") for x in raw_videos],
        all_videos,
        len(processed_videos),
    )


def _order_video_paths(
    paths: list[str],
    video_extensions: set[str],
    primary_camera_keyword: str,
) -> list[str]:
    """Return video paths with primary (path containing keyword) first, rest sorted."""
    video_paths = sorted(p for p in paths if any(p.endswith(ext) for ext in video_extensions))

    # If no video files, return empty list (caller will handle)
    if not video_paths:
        return []

    primary = [p for p in video_paths if primary_camera_keyword in p]

    if len(primary) > 1:
        msg = f"Multiple primary cameras found: {primary=}, need distinct primary camera to run multicam pipeline"
        raise ValueError(msg)

    if len(primary) == 0:
        msg = (
            f"No primary camera found with keyword {primary_camera_keyword}, "
            "need distinct primary camera to run multicam pipeline"
        )
        raise ValueError(msg)

    rest = [p for p in video_paths if p not in primary]
    return primary + sorted(rest)


def _multi_cam_session_to_split_task(  # noqa: PLR0913
    session_id: str,
    sessions_prefix: str,
    client: StorageClient | None,
    video_extensions: set[str],
    primary_camera_keyword: str,
    *,
    verbose: bool = False,
) -> SplitPipeTask | None:
    """Build a SplitPipeTask for one multi-cam session, or None if the session has no video files."""
    session_prefix = str(get_full_path(sessions_prefix, session_id))
    session_files = get_files_relative(session_prefix, client)
    video_paths = _order_video_paths(session_files, video_extensions, primary_camera_keyword)
    if not video_paths:
        if verbose:
            logger.debug(f"Session {session_id} has no video files, skipping")
        return None
    videos = [
        Video(get_full_path(sessions_prefix, session_id, p), relative_path=str(pathlib.Path(p).with_suffix("")))
        for p in video_paths
    ]
    if verbose:
        logger.debug(f"Session {session_id}: {len(videos)} videos (primary first)")
    return SplitPipeTask(videos=videos, session_id=session_id)


def extract_multi_cam_split_tasks(  # noqa: PLR0913
    sessions_prefix: str,
    primary_camera_keyword: str,
    video_extensions: set[str],
    input_s3_profile_name: str,
    limit: int = 0,
    *,
    verbose: bool = False,
) -> list[SplitPipeTask]:
    """Extract one SplitPipeTask per multi-cam session from a sessions prefix.

    Lists direct children of sessions_prefix that are valid UUIDs (session dirs),
    discovers video files by extension under each session, orders cameras so the
    primary (path containing primary_camera_keyword) is first, and returns one
    task per session

    Args:
        sessions_prefix: Path (local or S3) under which each UUID-named subdir is a session.
        primary_camera_keyword: String to identify primary camera; matching path is placed at slot 0.
        video_extensions: Set of extensions (e.g. {".mp4", ".h264"}) to discover.
        input_s3_profile_name: S3 profile for remote paths.
        limit: Max number of sessions to return (0 = no limit).
        verbose: Log discovered paths.

    Returns:
        List of SplitPipeTask, one per session, each with videos=[Video(...), ...], is_multi_cam=True.

    """
    client = get_storage_client(sessions_prefix, profile_name=input_s3_profile_name)
    session_ids = [name for name in get_directories_relative(sessions_prefix, client) if is_uuid(name)]

    tasks: list[SplitPipeTask] = []
    for session_id in sorted(session_ids):
        task = _multi_cam_session_to_split_task(
            session_id,
            sessions_prefix,
            client,
            video_extensions,
            primary_camera_keyword,
            verbose=verbose,
        )
        if task is not None:
            tasks.append(task)
        if limit > 0 and len(tasks) >= limit:
            break

    logger.info(f"Extracted {len(tasks)} session tasks from {sessions_prefix}")
    return tasks


def format_session_videos_tree(
    tasks: list[SplitPipeTask],
    input_prefix: str,
    limit: int = 3,
) -> str:
    """Format a tree view of multi-cam session tasks organized by session id.

    Returns a tree-formatted string with the input prefix as the root, with each session
    task shown as a child node. Videos within each session are organized by subdirectories
    (e.g., recorder directories).

    This is useful for debugging and visualizing the input data.

    Args:
        tasks: List of SplitPipeTask instances to display.
        input_prefix: Base path to display as root (e.g., "s3://bucket/path" or local path).
        limit: Maximum number of tasks to display, set to 0 to display all tasks.

    Returns:
        A multi-line string containing the formatted tree view.

    Example output:
        s3://hyperion8/samples-trimmed/
        ├── SplitPipeTask(0c99dbb9-646e-44b8-9583-2448310cd6a6)
        │   ├── recorder-00/
        │   │   ├── camera_front_tele_30fov.mp4
        │   │   └── camera_front_wide_120fov.mp4
        │   └── recorder-10/
        │       ├── camera_front_fisheye_200fov.mp4
        │       └── camera_rear_left_70fov.mp4
        └── SplitPipeTask(a778bacd-6d5e-4401-b060-f4281c28d4c6)
            ├── recorder-00/
            │   └── camera_front_wide_120fov.mp4
            └── recorder-10/
                ├── camera_rear_fisheye_200fov.mp4
                └── camera_rear_left_70fov.mp4

    """
    limit = len(tasks) if limit == 0 else limit
    lines = []

    # Add root path
    lines.append("")
    lines.append(f"{input_prefix}/")

    # Process each task
    for task_idx, task in enumerate(tasks[:limit]):
        is_last_task = task_idx == len(tasks[:limit]) - 1
        task_prefix = "└── " if is_last_task else "├── "
        subtree_prefix = "    " if is_last_task else "│   "

        # Strip base path and clean up path separators
        paths = [str(video.input_video).replace(str(input_prefix), "").lstrip("/\\") for video in task.videos]

        # Extract session ID from paths (each path is session_id/{path parts...})
        session_ids = [path.split("/")[0] for path in paths]
        paths = [path.replace(session_id + "/", "") for path, session_id in zip(paths, session_ids, strict=True)]
        session_ids_set = set(session_ids)

        if len(session_ids_set) != 1:
            msg = f"Multiple session ids found in task: {session_ids_set}"
            raise ValueError(msg)

        session_id = session_ids_set.pop()
        lines.append(f"{task_prefix}SplitPipeTask({session_id})")

        # Organize paths by directory for tree display
        dirs_to_files: dict[str, list[str]] = defaultdict(list)
        for path in paths:
            if "/" in path:
                dir_name, file_name = path.rsplit("/", 1)
                dirs_to_files[dir_name].append(file_name)
            else:
                dirs_to_files[""].append(path)

        # Build tree format for this task's videos
        sorted_dirs = sorted(dirs_to_files.keys())
        for i, dir_name in enumerate(sorted_dirs):
            is_last_dir = i == len(sorted_dirs) - 1
            dir_prefix = "└── " if is_last_dir else "├── "
            file_prefix = "    " if is_last_dir else "│   "

            if dir_name:
                lines.append(f"{subtree_prefix}{dir_prefix}{dir_name}/")
                files = sorted(dirs_to_files[dir_name])
                for j, file_name in enumerate(files):
                    is_last_file = j == len(files) - 1
                    file_symbol = "└── " if is_last_file else "├── "
                    lines.append(f"{subtree_prefix}{file_prefix}{file_symbol}{file_name}")
            else:
                # Files in root
                files = sorted(dirs_to_files[dir_name])
                for j, file_name in enumerate(files):
                    is_last_file = j == len(files) - 1 and is_last_dir
                    file_symbol = "└── " if is_last_file else "├── "
                    lines.append(f"{subtree_prefix}{file_symbol}{file_name}")

    return "\n".join(lines)


def _get_clip_metadata_paths_from_summary(
    input_path: str,
    client: StorageClient | None,
    annotate_version: str,
) -> list[StoragePrefix | pathlib.Path]:
    # TODO: handle chunked jsonl metadata case
    clip_metadata_paths: list[StoragePrefix | pathlib.Path] = []
    try:
        summary_file_path = get_full_path(input_path, "summary.json")
        logger.info(f"Reading split-annotate pipeline summary from {summary_file_path}")
        summary_data = read_json_file(summary_file_path, client, max_attempts=2)
        for v in summary_data.values():
            if isinstance(v, dict) and "clips" in v:
                for clip_uuid in v["clips"]:
                    clip_metadata_path = get_full_path(input_path, "metas", annotate_version, f"{clip_uuid}.json")
                    clip_metadata_paths.append(clip_metadata_path)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error(f"Failed to parse summary from {input_path}: {e}")
        return []
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to read summary file from {input_path}: {e}")
        return []
    else:
        return sorted(clip_metadata_paths, key=str)


def _worker_read_clip_metadata(
    clip_metadata_path: StoragePrefix | pathlib.Path,
    client: StorageClient | None,
    *,
    verbose: bool = False,
) -> ClipSample | None:
    try:
        clip_metadata = read_json_file(clip_metadata_path, client)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to read clip metadata from {clip_metadata_path}: {e}")
        return None
    else:
        if clip_metadata["valid"]:
            return ClipSample(
                uuid=str(clip_metadata["span_uuid"]),
                width=clip_metadata["width"],
                height=clip_metadata["height"],
                num_frames=clip_metadata["num_frames"],
                framerate=clip_metadata["framerate"],
                num_bytes=clip_metadata["num_bytes"],
                clip_location=get_full_path(clip_metadata["clip_location"]),
                clip_metadata=clip_metadata,
            )
        # the clip is marked invalid by the split-annotate pipeline
        logger.warning(f"Clip {clip_metadata['span_uuid']} is invalid, skipping ...")
        if verbose:
            logger.warning(clip_metadata)
        return None


def extract_shard_tasks(  # noqa: PLR0913
    input_path: str,
    output_path: str,
    input_s3_profile_name: str,
    output_s3_profile_name: str,
    version: str,
    *,
    verbose: bool = False,
) -> list[ClipSample]:
    """Extract list of clip paths from the input S3 or local path."""
    clip_samples = []
    client_input = get_storage_client(input_path, profile_name=input_s3_profile_name)
    client_output = get_storage_client(output_path, profile_name=output_s3_profile_name)
    # TODO: support fail-restart
    # verify output path is empty
    objects_in_output = get_files_relative(output_path, client_output)
    if len(objects_in_output) > 0:
        error_msg = f"Expect output path {output_path} to be empty"
        raise ValueError(error_msg)
    # extract clip metadata paths
    logger.info(f"Extracting clip metadata from {input_path} ...")
    clip_metadata_paths = _get_clip_metadata_paths_from_summary(input_path, client_input, version)
    # read clip metadata in parallel
    logger.info(f"Reading {len(clip_metadata_paths)} clip metadata from {input_path} ...")
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(
                _worker_read_clip_metadata,
                clip_metadata_path,
                client_input,
                verbose=verbose,
            )
            for clip_metadata_path in clip_metadata_paths
        ]
        for future in futures:
            clip_sample = future.result()
            if clip_sample is not None:
                clip_samples.append(clip_sample)

    return clip_samples


def _load_parquet_ids(
    client: StorageClient | None,
    dedup_path: str | StoragePrefix | pathlib.Path,
    filter_threshold: float | None = None,
) -> tuple[set[str], int, int]:
    """Load IDs from parquet files, optionally applying a filter threshold."""
    ids_to_remove_set = set()
    total_records = 0
    parquet_files_count = 0

    for item in get_files_relative(str(dedup_path), client):
        if not is_parquet_file(item):
            continue

        parquet_file = get_full_path(dedup_path, item)
        bytes_data = read_bytes(parquet_file, client)
        dedup_df = pd.read_parquet(BytesIO(bytes_data))
        total_records += len(dedup_df)

        if filter_threshold is not None:
            dedup_df = dedup_df.loc[dedup_df["cosine_sim_score"] >= filter_threshold]

        ids_to_remove_set.update(dedup_df["id"].tolist())
        parquet_files_count += 1

    return ids_to_remove_set, total_records, parquet_files_count


def filter_shard_tasks_by_semantic_dedup(
    clip_samples: list[ClipSample],
    input_semantic_dedup_path: str,
    input_semantic_dedup_s3_profile_name: str,
    semantic_dedup_epsilon: float,
    *,
    verbose: bool = False,
) -> list[ClipSample]:
    """Filter out clip samples based on semantic dedup results."""
    client = get_storage_client(
        input_semantic_dedup_path,
        profile_name=input_semantic_dedup_s3_profile_name,
    )

    # Attempt to read filtered results first
    formatted_epsilon = f"{semantic_dedup_epsilon:.5f}".rstrip("0").rstrip(".")
    dedup_filtered_path = get_full_path(
        input_semantic_dedup_path,
        "extraction",
        f"unique_ids_{formatted_epsilon}.parquet",
    )
    ids_to_remove_set, total_records, parquet_files_count = _load_parquet_ids(client, dedup_filtered_path)

    if parquet_files_count > 0:
        if verbose:
            logger.info(f"Using filtered results from {dedup_filtered_path}")
            logger.info(f"Loaded {total_records} filtered clips from {parquet_files_count} files.")
    else:
        # Fall back to raw results and filter
        dedup_raw_path = get_full_path(input_semantic_dedup_path, "extraction", "semdedup_pruning_tables")
        if verbose:
            logger.info(f"Filtered parquet not found; falling back to raw results from {dedup_raw_path}")

        similarity_threshold = 1 - semantic_dedup_epsilon
        ids_to_remove_set, total_records, parquet_files_count = _load_parquet_ids(
            client,
            dedup_raw_path,
            filter_threshold=similarity_threshold,
        )

        if parquet_files_count == 0:
            logger.warning(f"No valid parquet files found in {dedup_raw_path}")
            return clip_samples

        if verbose:
            logger.info(f"Processed {parquet_files_count} parquet files with {total_records} records.")
            logger.info(f"Removing {len(ids_to_remove_set)} clips after filtering (epsilon={semantic_dedup_epsilon}).")

    filtered_clip_samples = [sample for sample in clip_samples if sample.uuid not in ids_to_remove_set]

    if verbose:
        logger.info(
            f"Filtered from {len(clip_samples)} to {len(filtered_clip_samples)} clip samples "
            f"after applying semantic dedup from {input_semantic_dedup_path}.",
        )

    return filtered_clip_samples
