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
"""Clip extraction stages."""

import copy
import pathlib
import subprocess
import uuid
from uuid import UUID

import numpy as np
import numpy.typing as npt
import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.config.operation_context import make_pipeline_temporary_dir
from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.core.utils.misc import grouping
from cosmos_curator.pipelines.video.utils.data_model import (
    Clip,
    SplitPipeTask,
    Video,
    assert_time_alignment,
)
from cosmos_curator.pipelines.video.utils.decoder_utils import DEFAULT_TRANSCODE_BITRATE_M


def slice_video_clips(
    video: Video,
    start: int,
    end: int,
    chunk_index: int,
    num_chunks: int,
) -> Video:
    """Slice a video's clips and return a new Video for use in chunked subtasks.

    Helper used by chunk_tasks (ClipTranscodingStage). Only slices ``video.clips``;
    filtered_clips are not used, since this stage comes before filtering.

    Args:
        video: The source Video.
        start: Start index of the clips to slice.
        end: End index of the clips to slice.
        chunk_index: Chunk index for the sliced video.
        num_chunks: Total number of chunks for the video.

    Returns:
        A new Video with clips=video.clips[start:end] and other fields copied.

    Raises:
        ValueError: If start/end are invalid.

    """
    if end < start:
        msg = f"End index {end} is less than start index {start}"
        raise ValueError(msg)

    if start < 0 or end > len(video.clips):
        msg = f"Start index {start} or end index {end} is out of range [0, {len(video.clips)})"
        raise ValueError(msg)

    return Video(
        input_video=video.input_video,
        relative_path=video.relative_path,
        encoded_data=video.encoded_data,
        metadata=video.metadata,
        frame_array=video.frame_array,
        timestamps=video.timestamps,
        clips=video.clips[start:end],
        num_total_clips=len(video.clips),
        num_clip_chunks=num_chunks,
        clip_chunk_index=chunk_index,
        clip_stats=video.clip_stats,
        errors=copy.deepcopy(video.errors),
    )


def chunk_tasks(tasks: list[SplitPipeTask], num_clips_per_chunk: int, *, verbose: bool = False) -> list[SplitPipeTask]:
    """Split each task into subtasks by chunking clip indices across all videos.

    Think of videos/clips as a table (rows = videos, columns = clip indices)::

        |           | clip_0 | clip_1 | ... | clip_N-1 |
        | video_0   |        |        |     |          |
        | video_1   |        |        |     |          |
        | ...       |        |        |     |          |
        | video_M-1 |        |        |     |          |

    Tasks are chunked into subtasks by chunking the columns into contiguous ranges.
    Each chunk contains all M videos with the same clip index range; clips at the
    same index have the same span.

    Args:
        tasks: The tasks to chunk.
        num_clips_per_chunk: The number of clips per chunk.
        verbose: Whether to print verbose logs.

    Returns:
        Time-aligned chunked tasks.

    Raises:
        ValueError: If the number of clips per chunk is not positive.

    """
    output_tasks: list[SplitPipeTask] = []
    for task in tasks:
        clip_durations = [c.duration for v in task.videos for c in v.clips]
        total_clips = sum(len(v.clips) for v in task.videos)
        if len(clip_durations) > 0:
            logger.info(
                f"{len(task.videos)} video(s) with {total_clips} total clips and weight={task.weight:.2f}; "
                f"min-clip={min(clip_durations):.2f}s, "
                f"max-clip={max(clip_durations):.1f}s.",
            )

        # Chunk the clips for the primary video. This chunking strategy
        # will be applied to all the other videos.
        primary_clips = task.videos[0].clips
        primary_chunks = list(
            grouping.split_by_chunk_size(
                primary_clips,
                num_clips_per_chunk * 8,
                lambda x: int(x.span[1] - x.span[0]),
            ),
        )
        num_chunks = len(primary_chunks)

        start = 0
        for idx in range(num_chunks):
            chunk_clips = primary_chunks[idx]
            end = start + len(chunk_clips)
            chunk_videos = [slice_video_clips(v, start, end, idx, num_chunks) for v in task.videos]
            subtask = SplitPipeTask(
                session_id=task.session_id,
                videos=chunk_videos,
                stage_perf=copy.deepcopy(task.stage_perf),
            )
            start = end
            if idx > 0:
                for stats in subtask.stage_perf.values():
                    stats.reset()
            if verbose:
                logger.info(
                    f"Spawning subtask {idx} with {len(subtask.video.clips)} clips and weight={subtask.weight:.2f}",
                )
            output_tasks.append(subtask)
        logger.info(f"Creating {num_chunks} tasks for downstream from session {task.session_id}.")

    assert_time_alignment(output_tasks)
    return output_tasks


class ClipTranscodingStage(CuratorStage):
    """Stage that transcodes video clips into a standardized format.

    This stage handles the conversion of video clips using FFmpeg, supporting both
    software (libopenh264) and hardware (NVENC) encoding with configurable parameters.
    """

    def __init__(  # noqa: PLR0913
        self,
        num_cpus_per_worker: float = 6.0,
        encoder: str = "libopenh264",
        encoder_threads: int = 1,
        encode_batch_size: int = 16,
        nb_streams_per_gpu: int = 3,
        *,
        use_hwaccel: bool = False,
        use_input_bit_rate: bool = False,
        num_clips_per_chunk: int = 32,
        max_output_frames: int | None = None,
        verbose: bool = False,
        ffmpeg_verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the clip transcoding stage.

        Args:
            num_cpus_per_worker: Number of CPUs per worker.
            encoder: Video encoder to use.
            encoder_threads: Number of threads per encoder.
            encode_batch_size: Number of clips to encode in parallel.
            nb_streams_per_gpu: Number of streams per GPU.
            use_hwaccel: Whether to use hardware acceleration.
            use_input_bit_rate: Whether to use input video bit rate.
            num_clips_per_chunk: Number of clips per chunk.
            max_output_frames: If set, limit each clip's output frame count to this value
                by reducing FPS during transcoding. Source FPS is never increased.
            verbose: Whether to print verbose logs.
            ffmpeg_verbose: Whether to print FFmpeg verbose logs.
            log_stats: Whether to log performance statistics.

        """
        self._timer = StageTimer(self)
        self._num_cpus_per_worker = num_cpus_per_worker
        self._encoder = encoder
        self._encoder_threads = encoder_threads
        self._encode_batch_size = encode_batch_size
        self._nb_streams_per_gpu = nb_streams_per_gpu
        self._use_hwaccel = use_hwaccel
        self._use_input_bit_rate = use_input_bit_rate
        self._num_clips_per_chunk = num_clips_per_chunk
        self._max_output_frames = max_output_frames
        self._verbose = verbose
        self._ffmpeg_verbose = ffmpeg_verbose
        self._log_stats = log_stats
        if encoder not in {"libopenh264", "h264_nvenc"}:
            error_msg = f"Expected encoder of `libopenh264` or `h264_nvenc`. Got {encoder}"
            raise ValueError(error_msg)
        if max_output_frames is not None and max_output_frames <= 0:
            error_msg = f"max_output_frames must be a positive integer, got {max_output_frames}"
            raise ValueError(error_msg)

    def _process_video(self, video: Video) -> None:
        if not video.encoded_data:
            error_msg = "Please load video!"
            raise ValueError(error_msg)

        if not video.clips:
            logger.warning(f"No clips to transcode for {video.input_video}. Skipping...")
            video.encoded_data.drop()
            return

        if self._verbose:
            logger.info(f"Processing video {video.input_video} with {len(video.clips)} clips")

        with make_pipeline_temporary_dir(sub_dir="transcode") as tmp_dir:
            # write video to file
            video_file = tmp_dir / "input.mp4"
            video_data = video.encoded_data.resolve()
            if video_data is None:
                msg = f"Video {video.input_video} has no encoded_data after resolve"
                raise ValueError(msg)
            with video_file.open("wb") as f:
                f.write(video_data)
            force_pix_fmt = video.is_10_bit_color() or False

            # use input video bit-rate
            use_bit_rate = None
            if self._use_input_bit_rate:
                use_bit_rate = str(video.metadata.bit_rate_k) + "K"

            # extract clips in batches
            for i in range(0, len(video.clips), self._encode_batch_size):
                batch = video.clips[i : i + self._encode_batch_size]
                self._extract_clips(
                    tmp_dir,
                    video_file.name,
                    force_pix_fmt=force_pix_fmt,
                    use_bit_rate=use_bit_rate,
                    clips=batch,
                    input_video=str(video.input_video),
                    source_fps=video.metadata.framerate,
                )

    @nvtx.annotate("ClipTranscodingStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Process the data for the clip transcoding stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed task.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            for video in task.videos:
                with self._timer.time_process(
                    len(video.clips),
                    video.metadata.duration or 0,
                ):
                    try:
                        self._process_video(video)
                    except Exception as e:  # noqa: BLE001
                        logger.exception(f"Error processing video {video.input_video}")
                        video.errors[self.__class__.__name__] = str(e)

                video.encoded_data.drop()

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        # chunk tasks into subtasks, guaranteed to be time-aligned
        return chunk_tasks(tasks, self._num_clips_per_chunk, verbose=self._verbose)

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        if self._encoder == "h264_nvenc" or self._use_hwaccel:
            if self._nb_streams_per_gpu > 0:
                return CuratorStageResource(gpus=1.0 / self._nb_streams_per_gpu)
            return CuratorStageResource(gpus=1.0)
        return CuratorStageResource(cpus=self._num_cpus_per_worker)

    @nvtx.annotate("ClipLoadingStage:_extract_clips")  # type: ignore[untyped-decorator]
    def _extract_clips(  # noqa: C901, PLR0912, PLR0913
        self,
        working_dir: pathlib.Path,
        video_filename: str,
        *,
        force_pix_fmt: bool,
        use_bit_rate: str | None,
        clips: list[Clip],
        input_video: str,
        source_fps: float | None = None,
    ) -> None:
        # construct ffmpeg command
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning" if self._ffmpeg_verbose else "error",
        ]

        for i, clip in enumerate(clips):
            # set decoder threads
            if self.resources.gpus > 0:
                command.extend(["-threads", str(1)])
            else:
                command.extend(["-threads", str(self._encoder_threads)])
            # hwaccel needs to specified before each input
            if self._use_hwaccel:
                if self._encoder == "h264_nvenc":
                    command.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
                else:
                    command.extend(["-hwaccel", "auto"])
            start_s, end_s = clip.span
            command.extend(
                [
                    "-ss",
                    str(start_s),
                    "-to",
                    str(end_s),
                    "-i",
                    video_filename,
                    "-map",
                    f"{i}:v:0",
                    "-c:v",
                    self._encoder,
                ],
            )
            if use_bit_rate is not None:
                command.extend(["-b:v", use_bit_rate])
            else:
                command.extend(["-b:v", f"{DEFAULT_TRANSCODE_BITRATE_M}M"])
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
                    ],
                )
                # To fix `10 bit encode not supported` error
                if force_pix_fmt:
                    command.extend(["-pix_fmt", "yuv420p"])
            if self.resources.gpus > 0:
                command.extend(["-threads", str(1)])
            else:
                command.extend(["-threads", str(self._encoder_threads)])
            # Limit output frame count by reducing FPS when max_output_frames is set.
            # We use both -r (target FPS) and -frames:v (hard cap) because ffmpeg's
            # FPS resampling can produce 1-2 extra frames due to timestamp rounding.
            if self._max_output_frames is not None and source_fps is not None:
                duration = end_s - start_s
                if duration > 0 and source_fps * duration > self._max_output_frames:
                    target_fps = self._max_output_frames / duration
                    command.extend(["-r", f"{target_fps:.4f}", "-frames:v", str(self._max_output_frames)])
            command.extend(
                [
                    "-map",
                    f"{i}:a:0?",
                    "-c:a",
                    "copy",
                    f"{clip.uuid}.mp4",
                ],
            )

        # run ffmpeg command
        try:
            output = subprocess.check_output(  # noqa: S603
                command, cwd=working_dir, stderr=subprocess.STDOUT
            )
            if output and self._ffmpeg_verbose:
                logger.warning(f"ffmpeg output: {output.decode('utf-8')}")
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg command failed with return code {e.returncode} on {input_video}")
            logger.warning(f"Command: {' '.join(command)}")
            if e.output:
                logger.warning(f"Error output: {e.output.decode('utf-8')}")
            for clip in clips:
                clip.errors["transcode"] = e.output.decode("utf-8") if e.output else str(e)
            return

        # read clips back into memory
        for clip in clips:
            clip.encoded_data = bytes_to_numpy((working_dir / f"{clip.uuid}.mp4").read_bytes())  # type: ignore[assignment]
            try:
                clip.extract_metadata()
            except Exception as e:  # noqa: BLE001
                logger.exception(f"Failed to extract metadata for {clip.source_video=} {clip.uuid=} {clip.span=}")
                clip.errors["extract_metadata"] = str(e)
                clip.encoded_data.drop()
                continue
            # TODO(LazyData): re-enable when batch-mode ObjectRef ownership is
            # resolved.  In batch mode, pool.stop() kills actor -> OwnerDiedError.
            # clip.encoded_data.store()  # noqa: ERA001


def _validate_video_timestamps(video_timestamps: list[npt.NDArray[np.float32]]) -> None:
    """Validate the timestamps for each video.

    Args:
        video_timestamps: The timestamps for each video.

    Raises:
        ValueError: If any video has no timestamps.

    """
    if len(video_timestamps) == 0:
        msg = "No timestamps found for videos"
        raise ValueError(msg)

    if any(len(ts) == 0 for ts in video_timestamps):
        msg = "Some videos have no timestamps"
        raise ValueError(msg)


def _get_videos_timestamps(videos: list[Video]) -> list[npt.NDArray[np.float32]]:
    """Get timestamps for each video in seconds.

    Args:
        videos: The videos to get timestamps for.

    Returns:
        List of timestamp arrays (in seconds) for each video.

    Raises:
        ValueError: If any video has missing or empty timestamps.

    """
    for video in videos:
        if (video.timestamps is None or len(video.timestamps) == 0) and "timestamps" not in video.errors:
            video.errors["timestamps"] = "missing"
    missing = [v for v in videos if v.timestamps is None or len(v.timestamps) == 0]
    if missing:
        missing_paths = [str(v.input_video) for v in missing]
        msg = f"Videos missing timestamps: {missing_paths}"
        raise ValueError(msg)
    return [v.timestamps for v in videos]  # type: ignore[misc]


def _get_videos_durations(videos: list[Video]) -> list[float]:
    """Get the duration of a videos in seconds.

    Args:
        videos: The videos to get the durations of.

    Returns:
        The durations of the videos in seconds.

    """

    # Note: this is technically not correct, see CVC-690 for more details.
    # The correct duration is the difference between the last and first
    # timestamps. However, this decision was made early in development, and
    # correcting the behavior may be more complicated than expected.
    def _video_duration(video: Video) -> float:
        num_frames = video.metadata.num_frames
        framerate = video.metadata.framerate
        if num_frames is None or framerate is None or framerate <= 0:
            return -1.0
        return float(num_frames / framerate)

    return [_video_duration(video) for video in videos]


def _make_spans_fixed_stride(
    start_s: float,
    end_s: float,
    clip_len_s: float,
    clip_stride_s: float,
    min_clip_length_s: float,
) -> list[tuple[float, float]]:
    """Make a single set of spans for a list of videos.

    Each span represents a temporal window for clip extraction. Because videos
    share spans, this allows for easy grouping of clips by video when writing
    to disk.

    Assumes there is a shared temporal overlap across all sets of timestamps
    and that timestamps are sorted in ascending order.

    The caller of this helper function is expected to send in a non-empty list
    of video timestamps, using _validate_video_timestamps.

    Args:
        start_s: The start time in seconds.
        end_s: The end time in seconds.
        clip_len_s: The clip length.
        clip_stride_s: The clip stride.
        min_clip_length_s: The minimum clip length.

    Returns:
        List of (start_time, end_time) tuples in seconds.

    """
    spans: list[tuple[float, float]] = []
    start_span_s = start_s

    while start_span_s < end_s:
        end_span_s = min(start_span_s + clip_len_s, end_s)
        if (end_span_s - start_span_s) >= min_clip_length_s:
            spans.append((start_span_s, end_span_s))
        start_span_s += clip_stride_s

    return spans


def _make_clip_uuids(session_id: str, spans: list[tuple[float, float]]) -> list[UUID]:
    """Make deterministic clip uuids for session and a list of spans.

    Args:
        session_id: The session id for the group of videos
        spans: The spans to make clip uuids for.

    Returns:
        The clip uuids.

    """
    return [uuid.uuid5(uuid.NAMESPACE_URL, f"{session_id}_{span[0]}_{span[1]}") for span in spans]


def _populate_clips_fixed_stride(  # noqa: PLR0913
    videos: list[Video],
    session_id: str,
    clip_len_s: float,
    clip_stride_s: float,
    min_clip_length_s: float,
    *,
    limit_clips: int = 0,
) -> None:
    """Extract and populate clips for a list of videos using fixed stride.

    This mutates the videos in place and populates the clips list.

    Args:
        videos: The videos to populate clips for.
        session_id: The session id for the group of videos
        clip_len_s: The clip length in seconds.
        clip_stride_s: The clip stride in seconds.
        min_clip_length_s: The minimum clip length in seconds.
        limit_clips: If positive, only the first limit_clips spans are used. 0 means no limit.

    Notes:
    Previously, this logic resided inside the FixedStrideExtractorStage class.
    The original logic treated all videos as starting at timestamp 0 and
    ending at the "duration".

    The correct behavior is to use the actual start and end timestamps of the
    videos.

    However, correcting this behavior is not as straightforward as it seems.
    Using the actual start and end timestamps of the videos changes the output of
    the pipeline, and adds additional complexity, and it isn't clear how to handle
    that complexity and remain backwards compatible.

    The problem is the last clip in the video. It is likely that the last clip
    will be shorter than the clip length, and this will cause the pipeline to
    drop the last clip of each video, which is not the desired behavior.

    """
    durations = _get_videos_durations(videos)
    positive_durations = [d for d in durations if d > 0]
    if len(positive_durations) < len(videos):
        msg = "Some videos have invalid (zero or negative) duration"
        raise ValueError(msg)

    end_s = min(positive_durations)

    video_timestamps = _get_videos_timestamps(videos)
    _validate_video_timestamps(video_timestamps)  # secondary guard: catches empty-list edge case

    starts_s = [float(ts[0]) for ts in video_timestamps]
    ends_s = [t + duration for t, duration in zip(starts_s, durations, strict=True)]

    # See note above. This preserves the original behavior of the pipeline.
    # When this specific version was added, the goal was to add multi-cam
    # support, not to change behavior.
    start_s = max(t for t in starts_s)

    # Log warning if videos don't start near zero, as end_s assumes start=0
    if start_s > 0.1:  # noqa: PLR2004
        logger.warning(
            f"Videos start at {start_s:.2f}s (not 0), but duration-based end_s "
            f"assumes start=0. This may cause unexpected span boundaries."
        )

    # Original version assumed start=0.0. See note above.
    end_s = min(t for t in ends_s) - start_s
    start_s = 0.0

    # Generate spans
    spans = _make_spans_fixed_stride(
        start_s,
        end_s,
        clip_len_s,
        clip_stride_s,
        min_clip_length_s,
    )

    # Apply clip limit before creating clips
    if limit_clips > 0:
        spans = spans[:limit_clips]

    # Generate deterministic UUIDs for a session and its spans
    clip_uuids = _make_clip_uuids(session_id, spans)

    # Populate clips for each video
    for span, clip_uuid in zip(spans, clip_uuids, strict=True):
        for video in videos:
            clip = Clip(
                uuid=clip_uuid,
                source_video=str(video.input_video),
                span=span,
            )
            video.clips.append(clip)


class FixedStrideExtractorStage(CuratorStage):
    """Stage that extracts video clips using fixed-length intervals.

    This stage splits videos into clips of specified length and stride, ensuring
    each clip meets minimum length requirements and optionally limiting total clips.
    """

    def __init__(  # noqa: PLR0913
        self,
        clip_len_s: float = 10,
        clip_stride_s: float = 10,
        min_clip_length_s: float = 10,
        limit_clips: int = 0,
        *,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the fixed stride extractor stage.

        Args:
            clip_len_s: clip length.
            clip_stride_s: Stride length.
            min_clip_length_s: Minimum clip length. If raw video is smaller, will yield no spans.
            log_stats: Whether to log statistics. Default False.
            limit_clips: limit clips
            verbose: verbose

        """
        self._timer = StageTimer(self)
        self.clip_stride_s = clip_stride_s
        assert clip_stride_s
        self.clip_len_s = clip_len_s
        self.min_clip_length_s = min_clip_length_s
        self._limit_clips = limit_clips
        self._verbose = verbose
        self._log_stats = log_stats

    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # type: ignore[override]
        """Process the data for the fixed stride extractor stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """

        def _require_metadata(v: Video) -> None:
            if not v.has_metadata():
                v.errors["metadata"] = "incomplete"
                error_msg = f"Incomplete metadata for {v.input_video}. Skipping"
                raise ValueError(error_msg)

        def _validate_videos(videos: list[Video]) -> None:
            for video in videos:
                _require_metadata(video)

        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            with self._timer.time_process():
                try:
                    _validate_videos(task.videos)
                    _populate_clips_fixed_stride(
                        task.videos,
                        task.session_id,
                        self.clip_len_s,
                        self.clip_stride_s,
                        self.min_clip_length_s,
                        limit_clips=self._limit_clips,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.exception(f"Failed to populate clips for {task.session_id}")
                    task.errors["FixedStrideExtractorStage"] = f"failed to populate clips: {e}"

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        assert_time_alignment(tasks)
        return tasks
