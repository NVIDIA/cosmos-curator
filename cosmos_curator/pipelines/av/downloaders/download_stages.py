# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Download stages for AV pipelines."""

import pathlib
import subprocess

import numpy as np
import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.config.operation_context import (
    make_pipeline_named_temporary_file,
)
from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.core.utils.infra.tracing import traced_span
from cosmos_curator.core.utils.storage import s3_client
from cosmos_curator.core.utils.storage.storage_client import StorageClient
from cosmos_curator.core.utils.storage.storage_utils import (
    get_files_relative,
    get_full_path,
    get_storage_client,
    read_bytes,
)
from cosmos_curator.pipelines.av.captioning.captioning_stages import is_vri_prompt
from cosmos_curator.pipelines.av.utils.av_data_info import (
    CAMERA_MAPPING,
    CameraMapping,
    get_camera_id,
)
from cosmos_curator.pipelines.av.utils.av_data_model import (
    AvClipAnnotationTask,
    AvSessionTrajectoryTask,
    AvSessionVideoSplitTask,
    AvVideo,
    ClipForAnnotation,
)
from cosmos_curator.pipelines.av.utils.av_pipe_input import (
    is_sqlite_file,
    is_video_file,
)


def is_h264_file(file_name: str) -> bool:
    """Check if a file is an H.264 file.

    Args:
        file_name: The name of the file to check.

    Returns:
        True if the file is an H.264 file, False otherwise.

    """
    return file_name.endswith(".h264")


class VideoDownloader(CuratorStage):
    """VideoDownloader class that downloads videos from S3.

    This class downloads videos from S3 and populates the timestamps for each video.
    """

    def __init__(
        self,
        output_prefix: str,
        camera_format_id: str,
        prompt_variants: list[str],
        verbose: bool = False,  # noqa: FBT001, FBT002
        log_stats: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the VideoDownloader.

        Args:
            output_prefix: The output prefix.
            camera_format_id: The camera format ID.
            prompt_variants: The prompt variants.
            verbose: If True, log verbose information.
            log_stats: If True, log statistics.

        """
        self._timer = StageTimer(self)
        self._output_prefix = output_prefix.rstrip("/")
        self._camera_mapping_entry: CameraMapping = CAMERA_MAPPING[camera_format_id]
        self._camera_id_extractor = self._camera_mapping_entry["camera_id_extractor"]
        self._vri_camera_ids = self._camera_mapping_entry["camera_id_for_vri_caption"]
        self._extract_timestamp_from_video = bool(self._camera_mapping_entry.get("extract_timestamp_from_video", False))
        self._prompt_variants = prompt_variants if prompt_variants is not None else []
        self._verbose = verbose
        self._log_stats = log_stats
        self._client: StorageClient | None = None

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        return CuratorStageResource(
            cpus=0.25,
        )

    @property
    def conda_env_name(self) -> str | None:
        """Get the conda environment name.

        Returns:
            The conda environment name.

        """
        return "unified"

    def stage_setup(self) -> None:
        """Set up the VideoDownloader.

        This method sets up the VideoDownloader.

        """
        self._client = get_storage_client(target_path=self._output_prefix)

    @staticmethod
    def _convert_h264_to_mp4(mp4_file: pathlib.Path, h264_bytes: bytes) -> None:
        with make_pipeline_named_temporary_file("download") as h264_file:
            h264_file.write_bytes(h264_bytes)
            cmd = [
                "ffmpeg",
                "-loglevel",
                "panic",
                "-f",
                "h264",
                "-i",
                h264_file.as_posix(),
                "-c",
                "copy",
                "-y",
                "-f",
                "mp4",
                mp4_file.as_posix(),
            ]
            subprocess.check_call(cmd)  # noqa: S603

    @staticmethod
    def _log_video_info(video: AvVideo) -> None:
        size_bytes = video.metadata.size if video.metadata.size is not None else 0
        size_mb = size_bytes / (1024**2)
        num_timestamps = len(video.timestamps_ms) if video.timestamps_ms is not None else 0
        framerate = video.metadata.framerate if video.metadata.framerate is not None else 0
        height = video.metadata.height if video.metadata.height is not None else 0
        duration = video.metadata.duration if video.metadata.duration is not None else 0
        num_frames = video.metadata.num_frames if video.metadata.num_frames is not None else 0

        with traced_span(
            "VideoDownloader.video_info",
            attributes={
                "source_video": video.source_video,
                "size_mb": round(size_mb, 1),
                "height": height,
                "framerate": round(framerate, 1),
                "duration_s": round(duration, 1),
                "num_frames": num_frames,
                "num_timestamps": num_timestamps,
            },
        ):
            logger.info(
                f"Downloaded {video.source_video} size={size_mb:,.0f}MB "
                f"height={height} fps={framerate:.1f} "
                f"duration={duration:.0f}s "
                f"#-frames={num_frames} "
                f"#-timestamps={num_timestamps} "
                f"{video.metadata.video_codec}-{video.metadata.pixel_format}"
            )

    def _read_timestamps(self, task: AvSessionVideoSplitTask) -> None:
        timestamp_camera_id_mapping: dict[int, int] = self._camera_mapping_entry.get(
            "timestamp_camera_id_mapping",
            {x: x for x in self._camera_mapping_entry["camera_name_mapping_cosmos"]},
        )
        timestamps_ms: dict[int, list[int]] = {x: [] for x in timestamp_camera_id_mapping.values()}
        all_timestamp_files = self._camera_mapping_entry["all_timestamp_files"]
        for timestamp_file in all_timestamp_files:
            timestamp_url = get_full_path(task.session_url, timestamp_file)
            data = read_bytes(timestamp_url, self._client)
            for line in data.decode("utf-8").splitlines():
                items = line.split(",")
                try:
                    camera_id = timestamp_camera_id_mapping[int(items[0])]
                    timestamps_ms[camera_id].append(int(float(items[1])))
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Error parsing line {line} in {timestamp_url}: {e!s}")
                    continue
        for video in task.videos:
            video.timestamps_ms = np.array(timestamps_ms.get(video.camera_id, []), dtype=np.int64)

    @nvtx.annotate("VideoDownloader")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[AvSessionVideoSplitTask]) -> list[AvSessionVideoSplitTask] | None:
        """Process the data.

        This method processes the data.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed task.

        """
        output_tasks = [self._process_data(task) for task in tasks]
        return [task for task in output_tasks if task is not None]

    def _process_data(self, task: AvSessionVideoSplitTask) -> AvSessionVideoSplitTask | None:  # noqa: C901
        self._timer.reinit(self, task.get_major_size())
        logger.info(f"Finding videos under {task.session_url}")
        expected_camera_ids = (
            self._vri_camera_ids
            if any(is_vri_prompt(x) for x in self._prompt_variants)
            else list(self._camera_mapping_entry["camera_name_mapping_cosmos"].keys())
        )
        for item in get_files_relative(task.session_url, self._client):
            if not is_video_file(item):
                continue
            camera_id = get_camera_id(
                item,
                self._camera_id_extractor["delimiter"],
                self._camera_id_extractor["index"],
            )
            if camera_id not in expected_camera_ids:
                continue
            video_url = get_full_path(task.session_url, item)
            video = AvVideo(str(video_url), camera_id)
            try:
                encoded_data = read_bytes(video_url, self._client)
                with make_pipeline_named_temporary_file("download") as mp4_file:
                    if is_h264_file(str(video_url)):
                        self._convert_h264_to_mp4(mp4_file, encoded_data)
                    else:
                        mp4_file.write_bytes(encoded_data)
                    video.encoded_data = bytes_to_numpy(mp4_file.read_bytes())  # type: ignore[assignment]
                    video.populate_metadata()
                    # TODO(LazyData): re-enable when batch-mode ObjectRef ownership is
                    # resolved.  In batch mode, pool.stop() kills actor -> OwnerDiedError.
                    # video.encoded_data.store()  # noqa: ERA001
            except Exception as e:  # noqa: BLE001
                logger.error(f"Got an exception {e!s} when trying to read {video_url}")
                continue

            task.videos.append(video)
        logger.info(f"Downloaded {len(task.videos)} videos from {task.session_url}")

        # get timestamps
        if not self._extract_timestamp_from_video:
            self._read_timestamps(task)

        for video in task.videos:
            self._log_video_info(video)

        if self._log_stats:
            stage_name, stage_perf_stats = self._timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats

        if len(task.videos) != len(expected_camera_ids):
            logger.error(
                f"Expected {len(expected_camera_ids)} videos but downloaded "
                f"{len(task.videos)}; drop session {task.session_url}"
            )
            return None
        if all(video.timestamps_ms is None or len(video.timestamps_ms) == 0 for video in task.videos):
            logger.error(f"Not all videos have timestamps; drop session {task.session_url}")
            return None
        return task


class SqliteDownloader(CuratorStage):
    """SqliteDownloader class that downloads sqlite db from S3.

    This class downloads sqlite db from S3 and populates the sqlite db for each session.
    """

    def __init__(
        self,
        output_prefix: str,
        verbose: bool = False,  # noqa: FBT001, FBT002
        log_stats: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the SqliteDownloader.

        Args:
            output_prefix: The output prefix.
            verbose: If True, log verbose information.
            log_stats: If True, log statistics.

        """
        self._timer = StageTimer(self)
        self._output_prefix = output_prefix.rstrip("/")
        self._verbose = verbose
        self._log_stats = log_stats
        self._client: s3_client.S3Client | None = None

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        return CuratorStageResource(
            cpus=0.25,
        )

    def stage_setup(self) -> None:
        """Set up the SqliteDownloader.

        This method sets up the SqliteDownloader.

        """
        self._client = s3_client.create_s3_client(self._output_prefix)

    @nvtx.annotate("SqliteDownloader")  # type: ignore[untyped-decorator]
    def process_data(self, task: AvSessionTrajectoryTask) -> list[AvSessionTrajectoryTask] | None:
        """Process the data.

        This method processes the data.

        Args:
            task: The task to process.

        Returns:
            The processed task.

        """
        self._timer.reinit(self, task.get_major_size())
        logger.info(f"Finding sqlite db under {task.session_url}")
        for item in get_files_relative(task.session_url, self._client):
            if not is_sqlite_file(item):
                continue
            sqlite_url = get_full_path(task.session_url, item)
            try:
                raw_source_bytes = read_bytes(sqlite_url, self._client)
                task.sqlite_db = raw_source_bytes
                if self._verbose:
                    logger.info(f"Downloaded {sqlite_url} size={len(raw_source_bytes):,}B")
            except Exception as e:  # noqa: BLE001
                logger.error(f"Got an exception {e!s} when trying to read {sqlite_url}")
                continue

        if self._log_stats:
            stage_name, stage_perf_stats = self._timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats

        return [task]


class ClipDownloader(CuratorStage):
    """ClipDownloader class that downloads clips from S3.

    This class downloads clips from S3 and populates the clips for each session.
    """

    def __init__(
        self,
        output_prefix: str,
        verbose: bool = False,  # noqa: FBT001, FBT002
        log_stats: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the ClipDownloader.

        Args:
            output_prefix: The output prefix.
            verbose: If True, log verbose information.
            log_stats: If True, log statistics.

        """
        self._timer = StageTimer(self)
        self._output_prefix = output_prefix.rstrip("/")
        self._verbose = verbose
        self._log_stats = log_stats
        self._client: s3_client.S3Client | None = None

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        return CuratorStageResource(
            cpus=0.25,
        )

    def stage_setup(self) -> None:
        """Set up the ClipDownloader.

        This method sets up the ClipDownloader.

        """
        self._client = s3_client.create_s3_client(self._output_prefix)

    def _process_data(self, task: AvClipAnnotationTask) -> AvClipAnnotationTask:
        self._timer.reinit(self, task.get_major_size())
        num_clips_downloaded = 0

        def download_clip(clip: ClipForAnnotation) -> int:
            try:
                clip.encoded_data = bytes_to_numpy(read_bytes(clip.url, self._client))  # type: ignore[assignment]
                # TODO(LazyData): re-enable when batch-mode ObjectRef ownership is
                # resolved.  In batch mode, pool.stop() kills actor -> OwnerDiedError.
                # clip.encoded_data.store()  # noqa: ERA001
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error downloading {clip.url}: {e!s}")
                return 0
            return 1

        with self._timer.time_process(len(task.clips)):
            num_clips_downloaded = sum(download_clip(clip) for clip in task.clips)

        logger.info(f"Downloaded {num_clips_downloaded}/{len(task.clips)} clips")

        if self._log_stats:
            stage_name, stage_perf_stats = self._timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats

        return task

    @nvtx.annotate("ClipDownloader")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[AvClipAnnotationTask]) -> list[AvClipAnnotationTask] | None:
        """Process the data.

        This method processes the data.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed task.

        """
        return [self._process_data(task) for task in tasks]
