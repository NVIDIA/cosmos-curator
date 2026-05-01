# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Clip extraction stages."""

import pathlib
import subprocess
import uuid

import numpy as np
import numpy.typing as npt
import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.config.operation_context import make_pipeline_temporary_dir
from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.pipelines.av.utils.av_data_info import CAMERA_MAPPING
from cosmos_curator.pipelines.av.utils.av_data_model import (
    AvSessionVideoSplitTask,
    AvVideo,
    ClipForTranscode,
)


class ClipTranscodingStage(CuratorStage):
    """ClipTranscodingStage class that transcodes clips.

    This class transcodes clips using ffmpeg.
    """

    def __init__(  # noqa: PLR0913
        self,
        encoder: str = "libopenh264",
        openh264_bitrate: int = 10,
        encoder_threads: int = 4,
        encode_batch_size: int = 8,
        nb_streams_per_gpu: int = 3,
        verbose: bool = False,  # noqa: FBT001, FBT002
        log_stats: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the ClipTranscodingStage.

        Args:
            encoder: The encoder to use.
            openh264_bitrate: The bitrate for the openh264 encoder.
            encoder_threads: The number of threads for the encoder.
            encode_batch_size: The number of clips to encode in a batch.
            nb_streams_per_gpu: The number of streams per GPU.
            verbose: If True, log verbose information.
            log_stats: If True, log statistics.

        """
        self._timer = StageTimer(self)
        self._encoder = encoder
        self._openh264_bitrate = openh264_bitrate
        self._encoder_threads = encoder_threads
        self._encode_batch_size = encode_batch_size
        self._nb_streams_per_gpu = nb_streams_per_gpu
        self._use_hwaccel = encoder == "h264_nvenc"
        self._verbose = verbose
        self._log_stats = log_stats
        if encoder not in {"libopenh264", "h264_nvenc"}:
            error = f"Expected encoder of `libopenh264` or `h264_nvenc`. Got {encoder}"
            raise ValueError(error)

    @nvtx.annotate("ClipTranscodingStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[AvSessionVideoSplitTask]) -> list[AvSessionVideoSplitTask]:
        """Process the data.

        This method processes the data.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed task.

        """
        return [self._process_data(task) for task in tasks]

    def _process_data(self, task: AvSessionVideoSplitTask) -> AvSessionVideoSplitTask:
        self._timer.reinit(self, task.get_major_size())
        task.encoder = self._encoder
        for video in task.videos:
            data = video.encoded_data.resolve()
            if data is None:
                error = "Please load video!"
                raise ValueError(error)
            with self._timer.time_process(
                len(video.clips),
                video.metadata.duration if video.metadata.duration else 0,
            ):
                if not video.clips:
                    logger.warning(f"No clips to transcode for {video.source_video}. Skipping...")
                    video.encoded_data.drop()
                    continue
                with make_pipeline_temporary_dir(sub_dir="transcode") as tmp_dir:
                    # write video to file
                    video_file = tmp_dir / "input.mp4"
                    with video_file.open("wb") as f:
                        f.write(data)
                    force_pix_fmt = video.is_10_bit_color() or False

                    # extract clips in batches
                    for i in range(0, len(video.clips), self._encode_batch_size):
                        batch = video.clips[i : i + self._encode_batch_size]
                        self._extract_clips(
                            tmp_dir,
                            video_file.name,
                            force_pix_fmt,
                            batch,
                            str(video.source_video),
                        )
                    logger.info(f"Finished transcoding {len(video.clips)} clips from {video.source_video}")
            # we are done with encoded_data -- release inline and ObjectRef
            video.encoded_data.drop()
        if self._log_stats:
            stage_name, stage_perf_stats = self._timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats

        return task

    @property
    def num_gpus_per_worker(self) -> float | None:
        """Get the number of GPUs per worker.

        Returns:
            The number of GPUs per worker.

        """
        if self._encoder == "h264_nvenc" or self._use_hwaccel:
            if self._nb_streams_per_gpu > 0:
                return 1.0 / self._nb_streams_per_gpu
            return 1.0
        return None

    @property
    def num_cpus_per_worker(self) -> float | None:
        """Get the number of CPUs per worker.

        Returns:
            The number of CPUs per worker.

        """
        if self.num_gpus_per_worker is None:
            return self._encoder_threads
        return None

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        kwargs: dict[str, float] = {}
        gpus = self.num_gpus_per_worker
        if gpus is not None:
            kwargs["gpus"] = gpus

        cpus = self.num_cpus_per_worker
        if cpus is not None:
            kwargs["cpus"] = cpus

        return CuratorStageResource(**kwargs)

    @nvtx.annotate("ClipLoadingStage:_extract_clips")  # type: ignore[untyped-decorator]
    def _extract_clips(  # noqa: PLR0912, C901
        self,
        working_dir: pathlib.Path,
        video_filename: str,
        force_pix_fmt: bool,  # noqa: FBT001
        clips: list[ClipForTranscode],
        input_video: str,
    ) -> None:
        # construct ffmpeg command
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning" if self._verbose else "error",
        ]

        for i, clip in enumerate(clips):
            # set decoder threads
            if self.num_cpus_per_worker is not None:
                command.extend(["-threads", str(self._encoder_threads)])
            else:
                command.extend(["-threads", str(1)])
            # hwaccel needs to specified before each input
            if self._use_hwaccel:
                if self._encoder == "h264_nvenc":
                    command.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
                else:
                    command.extend(["-hwaccel", "auto"])
            command.extend(
                [
                    "-i",
                    video_filename,
                    "-ss",
                    str(clip.span_start),
                    "-to",
                    str(clip.span_end),
                    "-map",
                    f"{i}:v:0",
                    "-c:v",
                    self._encoder,
                ]
            )
            if self._encoder == "h264_nvenc":
                # IMPORTANT! these settings are necessary for high quality!
                command.extend(
                    [
                        "-rc:v",
                        "vbr",
                        "-cq:v",
                        "21",
                        "-tune",
                        "hq",
                        "-b_ref_mode",
                        "middle",
                        "-temporal-aq",
                        "1",
                        "-rc-lookahead",
                        "20",
                        "-spatial-aq",
                        "1",
                    ]
                )
                # To fix `10 bit encode not supported` error
                if force_pix_fmt:
                    command.extend(["-pix_fmt", "yuv420p"])
            else:
                command.extend(["-b:v", f"{self._openh264_bitrate}M"])

            if self.num_cpus_per_worker is not None:
                command.extend(["-threads", str(self._encoder_threads)])
            else:
                command.extend(["-threads", str(1)])

            command.extend(
                [
                    "-map",
                    f"{i}:a:0?",
                    "-c:a",
                    "copy",
                    f"{clip.uuid}.mp4",
                ]
            )

        # run ffmpeg command
        if self._verbose:
            logger.info(f"Running ffmpeg command: {' '.join(command)}")

        try:
            output = subprocess.check_output(  # noqa: S603
                command, cwd=working_dir, stderr=subprocess.STDOUT
            )
            if output and self._verbose:
                logger.warning(f"ffmpeg output: {output.decode('utf-8')}")
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg command failed with return code {e.returncode} on {input_video}")
            logger.warning(f"Command: {' '.join(command)}")
            if e.output:
                logger.warning(f"Error output: {e.output.decode('utf-8')}")
            return

        # read clips back into memory
        for clip in clips:
            clip.encoded_data = bytes_to_numpy((working_dir / f"{clip.uuid}.mp4").read_bytes())  # type: ignore[assignment]
            # TODO(LazyData): re-enable when batch-mode ObjectRef ownership is
            # resolved.  In batch mode, pool.stop() kills actor -> OwnerDiedError.
            # clip.encoded_data.store()  # noqa: ERA001


class FixedStrideExtractorStage(CuratorStage):
    """FixedStrideExtractorStage class that extracts clips with a fixed stride.

    This class extracts clips with a fixed stride from a video.
    """

    def __init__(  # noqa: PLR0913
        self,
        camera_format_id: str,
        clip_len_frames: int = 256,
        clip_stride_frames: int = 256,
        limit_clips: int = 0,
        verbose: bool = False,  # noqa: FBT001, FBT002
        log_stats: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize a FixedStrideExtractorStage.

        Args:
            camera_format_id: The camera format ID.
            clip_len_frames: Expected clip length in # of frames.
            clip_stride_frames: Stride length in # of frames
            limit_clips: Maximum number of clips to extract.
            verbose: If True, log verbose information.
            log_stats: If True, log statistics.

        """
        self._timer = StageTimer(self)
        self._align_every_frame = CAMERA_MAPPING[camera_format_id].get("align_every_frame", False)
        self._clip_len_frames = clip_len_frames
        self._clip_stride_frames = clip_stride_frames
        self._limit_clips = limit_clips
        self._verbose = verbose
        self._log_stats = log_stats

    @nvtx.annotate("FixedStrideExtractorStage")  # type: ignore[untyped-decorator]
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

    def _process_data(self, task: AvSessionVideoSplitTask) -> AvSessionVideoSplitTask | None:  # noqa: PLR0912, C901
        self._timer.reinit(self, task.get_major_size())
        task.split_algo_name = "fixed-stride"

        # verify very video is valid
        for video in task.videos:
            if not self._is_video_valid(video):
                return None

        start_timestamp_ms = min(video.timestamps_ms[0] for video in task.videos)  # type: ignore[index]

        clip_start_timestamp_ms = start_timestamp_ms
        index = 0

        with self._timer.time_process():
            while True:
                all_clips: dict[int, ClipForTranscode | None] = {video.camera_id: None for video in task.videos}
                any_camera_done = False

                for video in task.videos:
                    if clip_start_timestamp_ms >= video.timestamps_ms[-1]:  # type: ignore[index]
                        # one of the video is done - whole session is done
                        any_camera_done = True
                        break

                    start_frame_idx = self._find_closest_frame_index(
                        video.timestamps_ms,  # type: ignore[arg-type]
                        clip_start_timestamp_ms,
                    )
                    end_frame_idx = start_frame_idx + self._clip_len_frames

                    if self._verbose:
                        logger.debug(
                            f"camera-{video.camera_id} clip-{index=} "
                            f"{start_timestamp_ms=:,} {start_frame_idx=} {end_frame_idx=}"
                        )

                    if not self._is_end_frame_valid(video, end_frame_idx, index):
                        any_camera_done = True
                        break

                    # now make a clip
                    clip = self._create_clip(task, video, index, start_frame_idx, end_frame_idx)

                    if self._align_every_frame:
                        if self._verify_clip_timestamps(clip, start_frame_idx, end_frame_idx, video):
                            all_clips[video.camera_id] = clip
                    else:
                        all_clips[video.camera_id] = clip

                if any_camera_done:
                    break

                if all(clip is not None for clip in all_clips.values()):
                    for video in task.videos:
                        video.clips.append(all_clips[video.camera_id])  # type: ignore[arg-type]

                # advance to next clip
                clip_start_timestamp_ms += (
                    self._clip_stride_frames / video.metadata.framerate  # type: ignore[operator]
                ) * 1e3
                index += 1

                if self._limit_clips > 0 and index >= self._limit_clips:
                    break

            for video in task.videos:
                if video.clips:
                    logger.info(f"Extracted {len(video.clips)} clips from {video.source_video}")
                else:
                    logger.warning(f"No clips extracted from {video.source_video} duration={video.metadata.duration}")

            # done with video-level timestamps
            video.timestamps_ms = None

        if self._log_stats:
            stage_name, stage_perf_stats = self._timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats
        return task

    @staticmethod
    def _is_video_valid(video: AvVideo) -> bool:
        """Check if the video is valid.

        Args:
            video: The video to check.

        Returns:
            True if the video is valid, False otherwise.

        """
        if not video.encoded_data:
            logger.warning(f"Empty encoded_data for {video.source_video}. Skipping entire session.")
            return False
        if not video.has_metadata():
            logger.warning(f"Incomplete metadata for {video.source_video}. Skipping entire session.")
            return False
        if video.timestamps_ms is None or len(video.timestamps_ms) == 0:
            logger.warning(f"No timestamps for {video.source_video}. Skipping entire session.")
            return False

        return True

    @staticmethod
    def _is_end_frame_valid(video: AvVideo, end_frame_idx: int, clip_index: int) -> bool:
        """Check if the end frame is valid.

        Args:
            video: The video to check.
            end_frame_idx: The end frame index.
            clip_index: The clip index.

        Returns:
            True if the end frame is valid, False otherwise.

        """
        if end_frame_idx >= video.metadata.num_frames:  # type: ignore[operator]
            # to make sure all clips in same clip session have same length
            # require each clip to have exact self._clip_len_frames
            # so whole session is done in this case
            return False

        if end_frame_idx >= len(video.timestamps_ms):  # type: ignore[arg-type]
            logger.warning(
                f"clip-{clip_index} end-frame-idx={end_frame_idx} is beyond "
                f"timestamp list for {video.source_video}; skipping ..."
            )
            return False

        return True

    @staticmethod
    def _create_clip(
        task: AvSessionVideoSplitTask,
        video: AvVideo,
        clip_idx: int,
        start_frame_idx: int,
        end_frame_idx: int,
    ) -> ClipForTranscode:
        """Create a clip object."""
        span_start_s = start_frame_idx / video.metadata.framerate  # type: ignore[operator]
        span_end_s = end_frame_idx / video.metadata.framerate  # type: ignore[operator]

        return ClipForTranscode(
            uuid=uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{video.source_video}_{start_frame_idx}_{end_frame_idx}",
            ),
            clip_session_uuid=uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{task.session_url}_{clip_idx}",
            ),
            span_index=clip_idx,
            span_start=span_start_s,
            span_end=span_end_s,
            timestamps_ms=video.timestamps_ms[start_frame_idx:end_frame_idx],  # type: ignore[index]
        )

    @staticmethod
    def _find_closest_frame_index(timestamps_ms: npt.NDArray[np.int64], target_timestamp_ms: int) -> int:
        """Find the closest frame index to the target time."""
        diff = np.abs(timestamps_ms - target_timestamp_ms)
        return int(np.argmin(diff))

    @staticmethod
    def _verify_clip_timestamps(
        clip: ClipForTranscode,
        start_frame_idx: int,
        end_frame_idx: int,
        video: AvVideo,
    ) -> bool:
        """Verify the clip against timestamps to make sure time flows at constant rate.

        Args:
            clip: The clip to verify.
            start_frame_idx: The start frame index.
            end_frame_idx: The end frame index.
            video: The video to verify.

        Returns:
            True if the clip is valid, False otherwise.

        """
        frame_time = np.diff(clip.timestamps_ms)  # type: ignore[arg-type]
        if video.metadata.framerate is None:
            logger.error(f"Framerate is not set for {video.source_video}")
            return False

        target_frame_time_ms = 1e3 / video.metadata.framerate
        if frame_time.max() > target_frame_time_ms * 2.0 or frame_time.min() < target_frame_time_ms * 0.5:
            logger.warning(
                f"clip-{clip.span_index} of {video.source_video} has frame time "
                f"variation of max={frame_time.max():.1f} min={frame_time.min():.1f} "
                f"vs. {target_frame_time_ms:.1f} ms"
            )
            return False

        clip_time_ms = (
            video.timestamps_ms[end_frame_idx] - video.timestamps_ms[start_frame_idx]  # type: ignore[index]
        )
        target_clip_time_ms = (clip.span_end - clip.span_start) * 1e3
        if clip_time_ms > target_clip_time_ms * 1.05 or clip_time_ms < target_clip_time_ms * 0.95:
            logger.warning(
                f"clip-{clip.span_index} {video.source_video} has clip time "
                f"variation of {clip_time_ms:.0f} vs. {target_clip_time_ms:.0f} ms"
            )
            return False

        return True
