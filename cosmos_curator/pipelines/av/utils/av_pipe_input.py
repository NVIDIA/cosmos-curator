# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Module for extracting input tasks for the AV pipeline."""

import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import jinja2
from loguru import logger

from cosmos_curator.core.utils.db.database_types import PostgresDB
from cosmos_curator.core.utils.storage.s3_client import S3Prefix
from cosmos_curator.core.utils.storage.storage_client import StorageClient
from cosmos_curator.core.utils.storage.storage_utils import (
    get_files_relative,
    get_full_path,
    get_storage_client,
    is_remote_path,
    path_exists,
    read_json_file,
    read_text,
)
from cosmos_curator.core.utils.storage.writer_utils import (
    write_json,
)
from cosmos_curator.pipelines.av.captioning.captioning_stages import is_vri_prompt
from cosmos_curator.pipelines.av.utils.av_data_info import (
    CAMERA_MAPPING,
    SQLITE_DB_NAME,
)
from cosmos_curator.pipelines.av.utils.av_data_model import (
    AvSample,
    AvSessionTrajectoryTask,
    AvSessionVideoSplitTask,
    ClipForAnnotation,
    ClipForTrajectory,
)

WINDOWS_PER_CLIP = 2


CAPTION_KEYWORDS = ["airport_pick_up_drop_off"]


def _is_keyword_in_caption(caption: str, keyword: str) -> bool:
    if keyword == "airport_pick_up_drop_off":
        if "airport" not in caption.lower():
            return False
        if any(x in caption.lower() for x in ["pick up", "pick-up", "parking", "drop off", "drop-off"]):
            return True
    return False


def is_pcd_file(filename: str) -> bool:
    """Check if a file is a PCD file.

    Args:
        filename: The name of the file to check.

    Returns:
        True if the file is a PCD file, False otherwise.

    """
    return filename.endswith(".pcd")


def is_video_file(filename: str) -> bool:
    """Check if a file is a video file.

    Args:
        filename: The name of the file to check.

    Returns:
        True if the file is a video file, False otherwise.

    """
    return filename.endswith((".h264", ".mp4"))


def is_sqlite_file(path: str) -> bool:
    """Check if a file is a SQLite file.

    Args:
        path: The path to the file to check.

    Returns:
        True if the file is a SQLite file, False otherwise.

    """
    return Path(path).name == SQLITE_DB_NAME


def _to_s3_or_path(path: str) -> S3Prefix | Path:
    return S3Prefix(path) if is_remote_path(path) else Path(path)


def read_session_file(
    session_filepath: str | None,
) -> list[str]:
    """Expect a file either on cloud storage and local path.

    In the file, there should be one video file per line.
    Then we will extract the video sessions, i.e. the directory containing the video files.

    Args:
        session_filepath: The path to the file to read.

    Returns:
        A list of video sessions.

    """
    if session_filepath is None:
        return []
    client = get_storage_client(target_path=str(session_filepath))
    text = read_text(_to_s3_or_path(session_filepath), client)
    try:
        dset = set()
        for line in text.splitlines():
            if is_video_file(line):
                directory = "/".join(line.strip().split("/")[:-1])
                dset.add(directory)
        return list(dset)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error processing {session_filepath}: {e}")
        return []


def _get_video_sessions(
    input_path: str,
    verbose: bool,  # noqa: FBT001
) -> dict[str, int]:
    client = get_storage_client(target_path=input_path)
    session_names = {}
    for item in get_files_relative(input_path, client):
        if not is_video_file(item):
            continue
        if verbose:
            logger.info(f"Found input video {item}")
        session = "/".join(item.split("/")[:-1])
        if session not in session_names:
            session_names[session] = 0
        session_names[session] += 1
    logger.info(f"Found {len(session_names)} video sessions under {input_path}")
    return session_names


def _get_existing_video_sessions(
    db: PostgresDB,
    version: str,
) -> list[str]:
    query = jinja2.Template(
        """
        select session_name
        from source_data
        where
            version = '{{ version }}'
    """
    ).render(
        version=version,
    )
    results = db.make_query(query, verbose=True)
    existing_sessions = [row.session_name for row in results]
    logger.info(f"Found {len(existing_sessions)} existing sessions in {db.name}")
    return existing_sessions


def extract_source_video_sessions(
    db: PostgresDB,
    input_path: str,
    version: str,
    verbose: bool,  # noqa: FBT001
) -> list[AvSessionVideoSplitTask]:
    """Extract source video sessions from the database.

    Args:
        db: The database to extract the sessions from.
        input_path: The path to the input data.
        version: The version of the data.
        verbose: Whether to print verbose output.

    Returns:
        A list of source video sessions.

    """
    existing_sessions = set(_get_existing_video_sessions(db, version))
    logger.info(f"Found {len(existing_sessions)} unique existing sessions")

    source_video_sessions = []
    num_sessions_found_in_db = 0
    all_sessions = _get_video_sessions(input_path, verbose)

    for session, num_cameras in all_sessions.items():
        if session in existing_sessions:
            num_sessions_found_in_db += 1
            continue

        session_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"{session}/{version}")
        session_url = str(get_full_path(input_path, session))
        source_video_sessions.append(AvSessionVideoSplitTask(session, version, session_uuid, session_url, num_cameras))
    logger.info(f"{num_sessions_found_in_db}/{len(all_sessions)} sessions found in S3 already exists in DB")

    return source_video_sessions


def _extract_session_split_tasks_from_db(  # noqa: PLR0913
    db: PostgresDB,
    source_version: str,
    encoder: str,
    target_version: str,
    selected_sessions: list[str] | None = None,
    limit: int = 0,
) -> list[AvSessionVideoSplitTask]:
    selected_sessions = [] if selected_sessions is None else selected_sessions
    query = jinja2.Template(
        """
        select session_name, session_url, num_cameras
        from source_data
        where
            version = '{{ source_version }}' and
            not exists (
                select 1
                from clipped_session cs
                where
                    cs.encoder = '{{ encoder }}' and
                    cs.version = '{{ target_version }}' and
                    source_data.session_name = cs.source_video_session_name
            )
        {% if limit > 0 %}
        limit {{ limit }}
        {% endif %}
    """
    ).render(
        source_version=source_version,
        encoder=encoder,
        target_version=target_version,
        limit=limit,
    )

    tasks = []
    results = db.make_query(query, verbose=True)
    logger.info(f"Found {len(results)} sessions from database")
    debug_output = True
    for row in results:
        if len(selected_sessions) > 0 and row.session_name not in selected_sessions:
            if debug_output:
                logger.debug(f"Skipping session {row.session_name} that is not in selected list")
                debug_output = False
            continue
        tasks.append(
            AvSessionVideoSplitTask(
                row.session_name,
                source_version,
                uuid.uuid5(uuid.NAMESPACE_URL, f"{row.session_name}/{source_version}"),
                row.session_url,
                row.num_cameras,
            )
        )
    return tasks


def _check_output_path(output_path: str, client: StorageClient | None) -> None:
    logger.info(f"Checking output path {output_path}")
    if path_exists(get_full_path(output_path, "summary.json"), client):
        logger.warning(f"Output path {output_path} already concluded with a summary.json file.")


def _find_processed_video_sessions(
    output_path: str,
    client: StorageClient | None,
    verbose: bool = False,  # noqa: FBT001, FBT002
) -> list[str]:
    processed_session_jsons = get_files_relative(str(get_full_path(output_path, "processed_sessions")), client)
    logger.info(f"Found {len(processed_session_jsons)} processed sessions in {output_path}")
    if verbose:
        for session in processed_session_jsons:
            logger.debug(session)
    return processed_session_jsons


def _worker_verify_processed_sessions(
    processed_session_json: str,
    output_path: str,
    client: StorageClient | None,
) -> str | None:
    # read the processed session json
    processed_session_json_path = get_full_path(output_path, "processed_sessions", processed_session_json)

    def _extract(data: dict[str, Any]) -> int:
        num_session_chunks_obj = data.get("num_session_chunks")
        if num_session_chunks_obj is None:
            error = f"`num_session_chunks` key not found in {processed_session_json_path}"
            raise ValueError(error)
        return int(num_session_chunks_obj)

    try:
        data = read_json_file(processed_session_json_path, client)
        num_session_chunks = _extract(data)
        for idx in range(num_session_chunks):
            processed_session_chunk_path = get_full_path(
                output_path,
                "processed_session_chunks",
                processed_session_json.removesuffix(".json") + f"_{idx}.json",
            )
            if not path_exists(processed_session_chunk_path, client):
                logger.debug(f"Partial-processed session {processed_session_json} missing chunk-{idx}")
                return None
        return processed_session_json  # noqa: TRY300
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to read processed session json {processed_session_json_path}: {e}")
        return None


def _extract_session_split_tasks_from_cloud_storage(
    input_path: str,
    input_sessions: list[str],
    output_path: str,
) -> list[AvSessionVideoSplitTask]:
    # clients
    client_output = get_storage_client(target_path=output_path)
    # check output path
    _check_output_path(output_path, client_output)

    # find already processed sessions
    processed_session_jsons = _find_processed_video_sessions(output_path, client_output, verbose=False)
    # verify if all chunks are processed
    fully_processed_sessions = set()
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(
                _worker_verify_processed_sessions,
                session_json,
                output_path,
                client_output,
            )
            for session_json in processed_session_jsons
        ]
        for future in futures:
            session_json = future.result()
            if session_json is not None:
                fully_processed_sessions.add(session_json.removesuffix(".json"))
    logger.info(f"Found {len(fully_processed_sessions)} fully processed sessions")
    # build input list
    tasks = []
    for session in input_sessions:
        if session in fully_processed_sessions:
            continue
        tasks.append(
            AvSessionVideoSplitTask(
                source_video_session_name=session,
                source_video_version="v0",
                session_uuid=uuid.uuid5(uuid.NAMESPACE_URL, f"{session}/v0"),
                session_url=str(get_full_path(input_path, session)),
            )
        )
    return tasks


def extract_session_split_tasks(  # noqa: PLR0913
    db: PostgresDB | None,
    input_prefix: str | None,
    output_prefix: str,
    source_version: str,
    encoder: str,
    target_version: str,
    sessions: list[str] | None = None,
    limit: int = 0,
) -> list[AvSessionVideoSplitTask]:
    """Extract session split tasks from the database or cloud storage.

    Args:
        db: The database to extract the sessions from.
        input_prefix: The path to the input data.
        output_prefix: The path to the output data.
        source_version: The version of the source data.
        encoder: The encoder to use.
        target_version: The version of the target data.
        sessions: The sessions to extract.
        limit: The limit on the number of sessions to extract.

    Returns:
        A list of session split tasks.

    """
    sessions = [] if sessions is None else sessions
    if db is None:
        if input_prefix is None:
            error = "input_path must be provided"
            raise ValueError(error)
        return _extract_session_split_tasks_from_cloud_storage(
            input_path=input_prefix,
            input_sessions=sessions,
            output_path=output_prefix,
        )
    return _extract_session_split_tasks_from_db(
        db,
        source_version,
        encoder,
        target_version,
        selected_sessions=sessions,
        limit=limit,
    )


def extract_clip_caption_tasks(  # noqa: PLR0913
    db: PostgresDB,
    camera_format_id: str,
    prompt_types: list[str],
    source_version: str,
    encoder: str,
    target_version: str,
    sessions: list[str] | None = None,
    limit: int = 0,
) -> list[ClipForAnnotation]:
    """Extract clip caption tasks from the database.

    Args:
        db: The database to extract the tasks from.
        camera_format_id: The camera format ID.
        prompt_types: The prompt types to extract.
        source_version: The version of the source data.
        encoder: The encoder to use.
        target_version: The version of the target data.
        sessions: The sessions to extract.
        limit: The limit on the number of tasks to extract.

    Returns:
        A list of clip caption tasks.

    """
    sessions = [] if sessions is None else sessions
    camera_id: list[int] | None = None
    if any(is_vri_prompt(prompt_type) for prompt_type in prompt_types):
        camera_id = CAMERA_MAPPING[camera_format_id].get("camera_id_for_vri_caption")

        if camera_id is None:
            logger.warning(f"No `camera_id_for_vri_caption` found for camera format ID {camera_format_id}")

    prompt_types_quoted = [f"'{x}'" for x in prompt_types]
    query = jinja2.Template(
        """
        select
            cs.source_video_session_name as video_session_name,
            vs.clip_uuid as clip_uuid,
            vs.clip_session_uuid as clip_session_uuid,
            vs.camera_id as camera_id,
            vs.span_index as span_index,
            vs.url as url
        from video_span vs
        inner join
            clipped_session cs
            on vs.session_uuid = cs.session_uuid
        where
            vs.version = '{{ source_version }}' and
            vs.encoder = '{{ encoder }}' and
            {% if camera_id %}
            vs.camera_id in ({{camera_id | join(',')}}) and
            {% endif %}
            cs.version = '{{ source_version }}' and
            cs.encoder = '{{ encoder }}' and
            not exists (
                select 1
                from clip_caption cc
                where
                    vs.clip_uuid = cc.clip_uuid and
                    cc.version = '{{ target_version }}' and
                    cc.prompt_type in ({{ prompt_types_quoted | join(',') }})
            )
        {% if limit > 0 %}
        limit {{ limit }}
        {% endif %}
        """
    ).render(
        prompt_types_quoted=prompt_types_quoted,
        limit=limit,
        source_version=source_version,
        encoder=encoder,
        target_version=target_version,
        camera_id=camera_id,
    )

    tasks = []
    results = db.make_query(query, verbose=True)
    for row in results:
        if len(sessions) > 0 and row.video_session_name not in sessions:
            continue
        if row.camera_id not in CAMERA_MAPPING[camera_format_id]["camera_name_mapping_cosmos"]:
            continue
        tasks.append(
            ClipForAnnotation(
                video_session_name=row.video_session_name,
                clip_session_uuid=row.clip_session_uuid,
                uuid=row.clip_uuid,
                camera_id=row.camera_id,
                span_index=row.span_index,
                url=row.url,
            )
        )
    return tasks


def extract_clip_trajectory_tasks(  # noqa: PLR0913
    db: PostgresDB,
    camera_format_id: str,
    source_version: str,
    encoder: str,
    target_version: str,
    sessions: list[str] | None = None,
    limit: int = 0,
) -> list[AvSessionTrajectoryTask]:
    """Extract clip trajectory tasks from the database.

    Args:
        db: The database to extract the tasks from.
        camera_format_id: The camera format ID.
        source_version: The version of the source data.
        encoder: The encoder to use.
        target_version: The version of the target data.
        sessions: The sessions to extract.
        limit: The limit on the number of tasks to extract.

    Returns:
        A list of clip trajectory tasks.

    """
    sessions = [] if sessions is None else sessions
    query = jinja2.Template(
        """
        select
            cs.source_video_session_name as session_name,
            sd.session_url as session_url,
            vs.clip_uuid as clip_uuid,
            vs.camera_id as camera_id,
            vs.span_index as span_index,
            vs.timestamps as timestamps
        from video_span vs
        inner join
            clipped_session cs
            on vs.session_uuid = cs.session_uuid
        inner join
            source_data sd
            on cs.source_video_session_name = sd.session_name
            and cs.source_video_version = sd.version
        where
            vs.version = '{{ source_version }}' and
            vs.encoder = '{{ encoder }}' and
            cs.version = '{{ source_version }}' and
            cs.encoder = '{{ encoder }}' and
            not exists (
                select 1
                from clip_trajectory ct
                where
                    vs.clip_uuid = ct.clip_uuid and
                    ct.version = '{{ target_version }}'
            )
        order by session_name
        {% if limit > 0 %}
        limit {{ limit }}
        {% endif %}
        """
    ).render(
        limit=limit,
        source_version=source_version,
        encoder=encoder,
        target_version=target_version,
    )

    tasks: list[AvSessionTrajectoryTask] = []
    current_session = None
    current_url = None
    current_clips: list[ClipForTrajectory] = []
    results = db.make_query(query, verbose=True)
    for row in results:
        if len(sessions) > 0 and row.session_name not in sessions:
            continue
        if row.camera_id not in CAMERA_MAPPING[camera_format_id]["camera_name_mapping_cosmos"]:
            continue
        if current_session != row.session_name:
            if current_clips:
                tasks.append(
                    AvSessionTrajectoryTask(
                        current_url,  # type: ignore[arg-type]
                        current_clips,
                    )
                )
            current_session = row.session_name
            current_url = row.session_url
            current_clips = []
        current_clips.append(
            ClipForTrajectory(
                row.clip_uuid,
                row.camera_id,
                row.span_index,
                bytes(row.timestamps),
            )
        )
    if current_clips:
        tasks.append(
            AvSessionTrajectoryTask(
                current_url,  # type: ignore[arg-type]
                current_clips,
            )
        )

    return tasks


def extract_sharding_tasks(  # noqa: PLR0912, PLR0913, PLR0915, C901
    db: PostgresDB,
    camera_format_id: str,
    clip_version: str,
    split_algo_name: str,
    encoder: str,
    caption_version: str,
    prompt_type: str,
    sessions: list[str] | None = None,
    keyword: str | None = None,
    limit: int = 0,
) -> list[AvSample]:
    """Extract sharding tasks from the database.

    Args:
        db: The database to extract the tasks from.
        camera_format_id: The camera format ID.
        clip_version: The version of the clip data.
        split_algo_name: The name of the split algorithm.
        encoder: The encoder to use.
        caption_version: The version of the caption data.
        prompt_type: The prompt type to use.
        sessions: The sessions to extract.
        keyword: The keyword to filter by.
        limit: The limit on the number of tasks to extract.

    Returns:
        A list of sharding tasks.

    """
    sessions = [] if sessions is None else sessions
    query = jinja2.Template(
        """
        select
            session_name,
            clip_session_uuid,
            array_agg(camera_id order by camera_id) as camera_ids,
            array_agg(clip_uuid order by camera_id) as clip_uuids,
            array_agg(clip_url order by camera_id) as clip_urls,
            array_agg(clip_timestamps order by camera_id) as clip_timestampss,
            array_agg(window_caption order by camera_id) as window_captions,
            array_agg(window_start_frame order by camera_id) as window_start_frames,
            array_agg(window_end_frame order by camera_id) as window_end_frames,
            array_agg(t5_embedding_url order by camera_id) as t5_urls
        from (
            select
                cs.source_video_session_name as session_name,
                vs.clip_session_uuid as clip_session_uuid,
                vs.camera_id as camera_id,
                vs.clip_uuid as clip_uuid,
                vs.url as clip_url,
                vs.timestamps as clip_timestamps,
                cc.window_caption as window_caption,
                cc.window_start_frame as window_start_frame,
                cc.window_end_frame as window_end_frame,
                cc.t5_embedding_url as t5_embedding_url
            from video_span vs
            inner join clipped_session cs
                on vs.session_uuid = cs.session_uuid
            inner join clip_caption cc
                on vs.clip_uuid = cc.clip_uuid
            where
                vs.version = '{{ clip_version }}' and
                vs.split_algo_name = '{{ split_algo_name }}' and
                vs.encoder = '{{ encoder }}' and
                cs.version = '{{ clip_version }}' and
                cs.encoder = '{{ encoder }}' and
                cc.version = '{{ caption_version }}' and
                cc.prompt_type = '{{ prompt_type }}'
        ) as tmp
        group by session_name, clip_session_uuid
        {% if limit > 0 %}
        limit {{ limit }}
        {% endif %}
        """
    ).render(
        limit=limit,
        clip_version=clip_version,
        split_algo_name=split_algo_name,
        encoder=encoder,
        caption_version=caption_version,
        prompt_type=prompt_type,
    )

    tasks = []
    results = db.make_query(query, verbose=True)
    logger.info(f"Found {len(results)} clip-sessions from database")
    num_missing_clip_uuids = 0
    num_missing_clip_urls = 0
    num_missing_timestamps = 0
    num_missing_captions = 0
    num_missing_t5s = 0
    num_missing_windows = 0
    num_not_all_cameras_available = 0
    num_filtered_by_sessions = 0
    num_filtered_by_keyword = 0
    for row in results:
        num_cameras = len(row.camera_ids)
        if num_cameras != len(row.clip_uuids):
            logger.error(f"Number of cameras ({num_cameras}) != number of clip urls ({len(row.clip_uuids)})")
            num_missing_clip_uuids += 1
            continue
        if num_cameras != len(row.clip_urls):
            logger.error(f"Number of cameras ({num_cameras}) != number of clip urls ({len(row.clip_urls)})")
            num_missing_clip_urls += 1
            continue
        if num_cameras != len(row.clip_timestampss):
            logger.error(
                f"Number of cameras ({num_cameras}) != number of clip timestamps ({len(row.clip_timestampss)})"
            )
            num_missing_timestamps += 1
            continue
        if num_cameras != len(row.window_captions):
            logger.error(f"Number of cameras ({num_cameras}) != number of window captions ({len(row.window_captions)})")
            num_missing_captions += 1
            continue
        if num_cameras != len(row.t5_urls):
            logger.error(f"Number of cameras ({num_cameras}) != number of t5 urls ({len(row.t5_urls)})")
            num_missing_t5s += 1
            continue
        if any(len(x) != WINDOWS_PER_CLIP for x in row.window_captions):
            logger.error(f"Window captions must have {WINDOWS_PER_CLIP} elements")
            num_missing_windows += 1
            continue
        if any(len(x) != WINDOWS_PER_CLIP for x in row.window_start_frames):
            logger.error(f"Window start-frames must have {WINDOWS_PER_CLIP} elements")
            num_missing_windows += 1
            continue
        if any(len(x) != WINDOWS_PER_CLIP for x in row.window_end_frames):
            logger.error(f"Window end-frames must have {WINDOWS_PER_CLIP} elements")
            num_missing_windows += 1
            continue
        # all required cameras must be present
        if not all(
            camera_id in row.camera_ids for camera_id in CAMERA_MAPPING[camera_format_id]["camera_name_mapping_cosmos"]
        ):
            num_not_all_cameras_available += 1
            continue
        # filter sessions
        if len(sessions) > 0 and row.session_name not in sessions:
            num_filtered_by_sessions += 1
            continue
        # filter by keyword
        if keyword is not None:
            keyword_found = False
            for window_caption in row.window_captions:
                for caption in window_caption:
                    if _is_keyword_in_caption(caption, keyword):
                        keyword_found = True
            if not keyword_found:
                num_filtered_by_keyword += 1
                continue
        # build up the task
        tasks.append(
            AvSample(
                row.clip_session_uuid,
                row.camera_ids,
                row.clip_uuids,
                row.clip_urls,
                [bytes(x) for x in row.clip_timestampss],
                row.window_captions,
                row.window_start_frames,
                row.window_end_frames,
                row.t5_urls,
            )
        )
    logger.info(
        f"{num_missing_clip_uuids=} {num_missing_clip_urls=} {num_missing_timestamps=} "
        f"{num_missing_captions=} {num_missing_t5s=} {num_missing_windows=}"
    )
    logger.info(f"{num_not_all_cameras_available=} {num_filtered_by_sessions=} {num_filtered_by_keyword=}")
    return tasks


def add_trajectory_to_samples(
    db: PostgresDB,
    samples: list[AvSample],
    trajectory_version: str,
    batch_size: int = 5000,
) -> list[AvSample]:
    """Add trajectory data to samples.

    Args:
        db: The database to extract the trajectories from.
        samples: The samples to add the trajectories to.
        trajectory_version: The version of the trajectory data.
        batch_size: The batch size to use for the query.

    Returns:
        A list of samples with trajectory data.

    """
    # Get all clip UUIDs across all samples
    all_clip_uuids = []
    for sample in samples:
        all_clip_uuids.extend([str(uuid) for uuid in sample.clip_uuids])

    # Create an empty trajectories dictionary to collect results
    trajectories = {}

    # Process in batches
    for i in range(0, len(all_clip_uuids), batch_size):
        batch_uuids = all_clip_uuids[i : i + batch_size]

        # Format the UUIDs with explicit UUID casting for PostgreSQL
        formatted_uuids = "ARRAY[" + ", ".join(f"'{uuid}'::uuid" for uuid in batch_uuids) + "]"

        # Query trajectories for this batch of clips
        query = jinja2.Template(
            """
            select
                clip_uuid,
                trajectory_url
            from clip_trajectory
            WHERE clip_uuid = ANY({{ clip_uuids }})
            AND version = '{{ trajectory_version }}'
        """
        ).render(
            clip_uuids=formatted_uuids,
            trajectory_version=trajectory_version,
        )

        batch_results = db.make_query(query, verbose=True)

        # Add results to the trajectories dictionary
        for row in batch_results:
            trajectories[row.clip_uuid] = row.trajectory_url

    logger.info(f"Total: Found {len(trajectories)} clip-trajectories from database")

    # Create new samples with trajectory data (this part remains unchanged)
    enhanced_samples = []
    for sample in samples:
        sample_trajectory_urls = []
        for clip_uuid in sample.clip_uuids:
            if clip_uuid in trajectories:
                sample_trajectory_urls.append(trajectories[clip_uuid])
            else:
                # Handle missing trajectory data
                logger.error(f"Missing trajectory data for clip_uuid={clip_uuid}")
                sample_trajectory_urls.append(None)

        # Create a new sample with trajectory data
        enhanced_sample = AvSample(
            sample.clip_session_uuid,
            sample.camera_ids,
            sample.clip_uuids,
            sample.clip_urls,
            sample.clip_timestampss_ms,
            sample.window_captions,
            sample.window_start_frames,
            sample.window_end_frames,
            sample.t5_urls,
            sample_trajectory_urls,
        )
        enhanced_samples.append(enhanced_sample)

    return enhanced_samples


def _summarize_video_session(  # noqa: C901, PLR0915
    session_json: str, output_path: str, client_output: StorageClient | None
) -> dict[str, Any] | None:
    """Summarize a video session. Takes in a path to a video session JSON file.

    Read in the associated processed_session_chunks, and then summarize the
    metadata and clip_uuids in the video session.

    Args:
        session_json: The path to the session JSON file.
        output_path: The path to the output directory.
        client_output: The S3 client to use for output.

    Returns:
        session_summary: Dict[str, Any]

    """
    try:
        _session_json = get_full_path(_to_s3_or_path(output_path), "processed_sessions", session_json)
        session = read_json_file(_session_json, client_output)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error reading video session json {session_json}: {e}")
        return None

    def _extract(session: dict[str, Any]) -> tuple[str, int, int, int]:
        session_name_obj = session.get("video_session_name")
        num_session_chunks_obj = session.get("num_session_chunks")
        height_obj = session.get("height")
        width_obj = session.get("width")

        if session_name_obj is None:
            error = f"`video_session_name` key not found in {session_json}"
            raise ValueError(error)
        if num_session_chunks_obj is None:
            error = f"`num_session_chunks` key not found in {session_json}"
            raise ValueError(error)
        if height_obj is None:
            error = f"`height` key not found in {session_json}"
            raise ValueError(error)
        if width_obj is None:
            error = f"`width` key not found in {session_json}"
            raise ValueError(error)

        session_name = str(session_name_obj)
        num_chunks = int(num_session_chunks_obj)
        height = int(height_obj)
        width = int(width_obj)

        return session_name, num_chunks, height, width

    try:
        session_name, num_chunks, height, width = _extract(session)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error parsing video session json {session_json}: {e}")
        return None

    source_video_duration_s: float = 0.0
    clips: list[str] = []

    def _extract_clips(chunk_json: dict[str, Any]) -> list[str]:
        clips_obj = chunk_json.get("clips", [])
        if not isinstance(clips_obj, list):
            error = f"Expected `clips` value to be a list in {chunk_json_path}, not type {type(clips_obj)}"
            raise TypeError(error)
        return clips_obj

    for chunk_idx in range(num_chunks):
        chunk_json_path = get_full_path(output_path, "processed_session_chunks", f"{session_name}_{chunk_idx}.json")
        try:
            chunk_json = read_json_file(chunk_json_path, client_output)
            chunk_session_name = chunk_json.get("video_session_name")
            if chunk_session_name != session_name:
                logger.error(
                    f"Chunk session name {chunk_session_name} does not match "
                    f"session name {session_name} in {chunk_json_path}"
                )
                continue
            session_chunk_index = chunk_json.get("session_chunk_index", -1)
            if session_chunk_index != chunk_idx:
                logger.error(
                    f"Chunk session index {session_chunk_index} does not match "
                    f"chunk_idx {chunk_idx} in {chunk_json_path}"
                )
                continue
            if chunk_idx == 0:
                source_video_duration_s = float(chunk_json.get("source_video_duration_s", 0.0))

            clips.extend(_extract_clips(chunk_json))
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error reading chunk_json_path={chunk_json_path}: {e}")
            continue

    return {
        "session_name": session_name,
        "source_video_duration_s": source_video_duration_s,
        "height": height,
        "width": width,
        "clips": clips,
    }


def _get_summary(
    output_path: str,
    num_threads: int,
) -> tuple[dict[str, dict[str, Any]], float]:
    """Generate a summary of processed video sessions from the output directory.

    Process all video sessions in the output directory, collecting metadata
    about each session including duration, dimensions, and clip uuids.

    Args:
        output_path: Path to the directory containing processed video sessions
        num_threads: Number of threads to use for parallel processing

    Returns:
        Tuple containing:
            - Dictionary mapping session names to their summaries. Each summary contains:
                - session_name: Name of the video session
                - source_video_duration_s: Total duration in seconds
                - height: Video height in pixels
                - width: Video width in pixels
                - clips: List of clip UUIDs
            - Total video length across all sessions in seconds

    """
    client_output = get_storage_client(target_path=output_path)
    processed_session_jsons = _find_processed_video_sessions(output_path, client_output, verbose=False)

    total_video_length: float = 0.0  # seconds
    summary: dict[str, dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(
                _summarize_video_session,
                session_json,
                output_path,
                client_output,
            )
            for session_json in processed_session_jsons
        ]
        for future in futures:
            session_summary = future.result()
            if session_summary is not None:
                try:
                    session_name = session_summary["session_name"]
                    session_summary.pop("session_name")
                    total_video_length += session_summary["source_video_duration_s"]
                    summary[session_name] = session_summary
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Error parsing session summary {session_summary}: {e}")
                    continue

    return summary, total_video_length


def write_summary(
    output_path: str,
    num_threads: int = 32,
) -> float:
    """Write a summary of processed video sessions to a JSON file.

    Generates a summary of all video sessions in the input directory, including
    metadata about each session (duration, dimensions, clips) and writes it to
    a summary.json file in the output directory.

    Args:
        output_path: Path where the summary.json file will be written
        num_threads: Number of threads to use for parallel processing

    Returns:
        float: Total video length across all sessions in seconds

    """
    summary, total_video_length = _get_summary(output_path, num_threads)
    client_output = get_storage_client(target_path=output_path, can_overwrite=True, can_delete=True)
    _check_output_path(output_path, client_output)
    write_json(
        summary,
        get_full_path(output_path, "summary.json"),
        "summary",
        "summary",
        verbose=True,
        client=client_output,
        backup_and_overwrite=True,
    )
    return total_video_length
