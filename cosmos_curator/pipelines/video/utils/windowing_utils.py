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
"""Utilities which are used in multiple places in the pipeline and/or are unit-tested."""

import subprocess

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger

from cosmos_curator.core.utils.config.operation_context import make_pipeline_named_temporary_file
from cosmos_curator.core.utils.model import pixi_utils
from cosmos_curator.pipelines.common.model_constraints import PreprocessMode
from cosmos_curator.pipelines.video.utils.data_model import (
    Clip,
    Video,
    Window,
    WindowConfig,
)
from cosmos_curator.pipelines.video.utils.decoder_utils import DEFAULT_TRANSCODE_BITRATE_M, get_frame_count
from cosmos_curator.pipelines.video.utils.windowing_types import WindowFrameInfo

if pixi_utils.is_running_in_env("default"):
    from cosmos_curator.pipelines.video.utils.vision_process import fetch_video


WINDOW_MIN_FRAMES = 4


def compute_windows(total_frames: int, window_size: int = 128, remainder_threshold: int = 64) -> list[WindowFrameInfo]:
    """Generate windows by splitting the video into segments of the specified size.

    Args:
        total_frames: total frames
        window_size: The size of each window in number of frames.
        remainder_threshold: The minimum number of frames required to create a new window from the remainder.

    Returns:
        List of ``WindowFrameInfo`` items, each representing an inclusive
        ``(start_frame, end_frame)`` window.

    """
    if not total_frames or total_frames < WINDOW_MIN_FRAMES:
        return []
    if total_frames <= window_size:
        return [WindowFrameInfo(0, total_frames - 1)]
    # Calculate the number of full window_size windows
    num_full_windows = total_frames // window_size

    # Calculate the remainder frames after filling in window_size windows
    remainder = total_frames % window_size

    out: list[WindowFrameInfo] = []
    # Yield each full window
    for i in range(num_full_windows):
        start_frame = i * window_size
        end_frame = start_frame + window_size - 1
        out.append(WindowFrameInfo(start_frame, end_frame))

    # Handle the remainder
    if remainder >= remainder_threshold:
        out.append(WindowFrameInfo(total_frames - remainder, total_frames - 1))
    elif remainder > 0 and num_full_windows > 0:
        # Expand the last window with the remainder if it exists
        out[-1] = WindowFrameInfo(out[-1].start, total_frames - 1)
    return out


def estimate_native_frame_count(clip: Clip, fallback_window: Window | None = None) -> int:
    """Estimate the native frame count ``N`` for a clip segment.

    When ``clip.windows`` is populated (normal vLLM async prep path), windows
    partition ``0 .. total_native-1``; ``max(end_frame) + 1`` recovers ``N``.

    When ``clip.windows`` is empty, uses ``fallback_window.end_frame + 1`` if
    provided; otherwise returns ``1`` (degenerate single-frame case).

    ::

        clip.windows:  [w0: 0..9] [w1: 10..19]
                        |---------+---------| N = 20

    Assumes native frame indices are contiguous and uniformly spaced across
    ``clip.span``. If only a subset of windows is present the estimate may be
    low, and mapped source times become approximate.

    Args:
        clip: Clip whose ``windows`` list defines the partition when non-empty.
        fallback_window: Used only when ``clip.windows`` is empty.

    Returns:
        Estimated ``N >= 1``.

    """
    if clip.windows:
        max_end: int = max(w.end_frame for w in clip.windows)
        return max_end + 1
    if fallback_window is not None:
        n: int = max(fallback_window.end_frame + 1, 1)
        return n
    return 1


def frame_index_to_source_time_s(
    clip_span: tuple[float, float],
    frame_index: int,
    native_frame_count: int,
) -> float:
    """Map a single native frame index to an estimated source time (seconds).

    Linearly interpolates ``frame_index`` between ``clip_span[0]`` and
    ``clip_span[1]`` over native indices ``0 .. native_frame_count - 1``.

    ::

        source time
        clip_span[0] |---------+---------| clip_span[1]
        frame index   0        f        N-1

        result = clip_span[0] + (f / max(N-1, 1)) * (clip_span[1] - clip_span[0])

    Args:
        clip_span: ``(t0, t1)`` seconds on the original source video.
        frame_index: Native frame index to map.
        native_frame_count: Estimated ``N``; denominator is ``max(N - 1, 1)``.

    Returns:
        Estimated source time in seconds for ``frame_index``.

    """
    span_s = max(clip_span[1] - clip_span[0], 0.0)
    denom = max(native_frame_count - 1, 1)
    return clip_span[0] + (frame_index / denom) * span_s


def window_source_time_bounds_s(
    clip_span: tuple[float, float],
    start_frame: int,
    end_frame: int,
    native_frame_count: int,
) -> tuple[float, float]:
    """Map inclusive native frame indices to estimated source times (seconds).

    Convenience wrapper: calls :func:`frame_index_to_source_time_s` for both
    ``start_frame`` and ``end_frame``.

    ::

        source timeline
        clip_span[0] |----[start_frame ... end_frame]----| clip_span[1]
                          ^                          ^
                     source_start_s            source_end_s

    Assumes uniform spacing of native frames across the clip time range.
    Matches the current pipeline where windows are built from a contiguous
    native decode range.

    Args:
        clip_span: ``(t0, t1)`` seconds of this clip on the original source video.
        start_frame: Inclusive native start index for the window.
        end_frame: Inclusive native end index for the window.
        native_frame_count: Estimated ``N``; denominator is ``max(N - 1, 1)``.

    Returns:
        ``(source_start_s, source_end_s)`` estimated bounds on the source timeline.

    """
    return (
        frame_index_to_source_time_s(clip_span, start_frame, native_frame_count),
        frame_index_to_source_time_s(clip_span, end_frame, native_frame_count),
    )


def window_source_time_bounds_from_clip(
    clip: Clip,
    window: Window,
) -> tuple[float, float]:
    """Estimate source-timeline bounds for a window using clip context.

    Combines :func:`estimate_native_frame_count` and
    :func:`window_source_time_bounds_s` into a single call for the common
    case where a ``Clip`` and ``Window`` are both available.

    Args:
        clip: Clip providing ``span`` and ``windows`` for ``N`` estimation.
        window: Window whose ``start_frame`` / ``end_frame`` are mapped.

    Returns:
        ``(source_start_s, source_end_s)`` estimated bounds on the source timeline.

    """
    n = estimate_native_frame_count(clip, fallback_window=window)
    return window_source_time_bounds_s(clip.span, window.start_frame, window.end_frame, n)


def window_source_time_trace_attributes(clip: Clip, window: Window) -> dict[str, str | float]:
    """Build trace attributes mapping a window's native frames to estimated source times.

    Returns a dict of OTel-safe key/value pairs suitable for ``traced_span(..., attributes={})``.
    Never raises -- returns an empty dict on failure so tracing cannot break the pipeline.

    Returned keys::

        window.source_start_s   -- window start on source timeline (seconds)
        window.source_end_s     -- window end on source timeline (seconds)
        window.clip_span_start_s -- parent clip start on source timeline
        window.clip_span_end_s   -- parent clip end on source timeline
        window.source_bounds     -- human-scannable summary, e.g. "12.345s-15.678s"

    Args:
        clip: Clip providing ``span`` and ``windows`` for N estimation.
        window: Window whose ``start_frame`` / ``end_frame`` are mapped.

    Returns:
        Dict of trace attributes, or empty dict on error.

    """
    try:
        source_start_s, source_end_s = window_source_time_bounds_from_clip(clip, window)
        return {
            # Window's estimated position on the original source timeline (seconds).
            "window.source_start_s": source_start_s,
            "window.source_end_s": source_end_s,
            # Parent clip's full time range -- context for where this window sits.
            "window.clip_span_start_s": clip.span[0],
            "window.clip_span_end_s": clip.span[1],
            # Human-scannable summary, e.g. "12.345s-15.678s".
            "window.source_bounds": f"{source_start_s:.3f}s-{source_end_s:.3f}s",
        }
    except Exception:  # noqa: BLE001 -- tracing must never break the pipeline
        return {}


def split_video_into_windows(  # noqa: PLR0913
    mp4_bytes: bytes | npt.NDArray[np.uint8],
    window_size: int = 256,
    remainder_threshold: int = 128,
    sampling_fps: float = 2.0,
    *,
    preprocess_mode: PreprocessMode = PreprocessMode.CURATOR,
    flip_input: bool = False,
    num_frames_to_use: int = 0,
    return_bytes: bool = False,
    target_bit_rate: str = f"{DEFAULT_TRANSCODE_BITRATE_M}M",
    return_video_frames: bool = True,
    num_threads: int = 1,
    max_pixels_per_frame: int | None = None,
) -> tuple[list[bytes | None], list[torch.Tensor | None], list[WindowFrameInfo]]:
    """Calculate windows and return video inputs for the Qwen language model from input clips.

    Processes video to determine the windows for a clip, decode in one shot and return processed frames
    for each window in a format suitable for consumption by the Qwen model.

    All three returned lists are guaranteed to have the same length.  When
    ``return_bytes`` or ``return_video_frames`` is ``False``, the
    corresponding list is padded with ``None`` so that callers can safely
    ``zip`` the results without length checks.

    Args:
        mp4_bytes: input video in bytes
        window_size: window size
        remainder_threshold: threshold for remainder
        sampling_fps: sampling fps when generating frames
        preprocess_mode: Owner of resize/rescale/normalize for returned video frames.
        flip_input: Whether to flip the input video/image horizontally.
        num_frames_to_use: Number of frames to extract from the video. If 0, uses all frames.
        return_bytes: Whether to extract mp4 bytes for each window for use by PreviewStage
        target_bit_rate: Target bit rate for the output window mp4 bytes.
        return_video_frames: whether to return video frames
        num_threads: number of threads
        max_pixels_per_frame: Optional fixed per-frame resize upper bound.

    Returns:
        Tuple containing three lists of equal length:
            - "window_mp4_bytes": mp4 bytes per window (``None`` when *return_bytes* is False)
            - "window_frames": Decoded per-window frames (``None`` when *return_video_frames* is False)
            - "window_info": start and end frame indices for each window in a clip

    """
    # TODO(ep): Consider migrating to ``cosmos_curator.core.utils.misc.memfd.buffer_as_memfd_path``
    # to avoid disk I/O for the temporary video file.  memfd provides a memory-backed
    # /proc/self/fd/<fd> path, eliminating the write-to-disk round-trip.  Note: memfd_create
    # may be blocked by seccomp on NVCF, in which case it falls back to tempfile anyway.
    with make_pipeline_named_temporary_file(sub_dir="windowing") as input_file:
        with input_file.open("wb") as f:
            f.write(mp4_bytes)
        total_frames = get_frame_count(mp4_bytes)
        windows = compute_windows(total_frames, window_size, remainder_threshold)
        video_frames: list[torch.Tensor | None] = []
        mp4_bytes_list: list[bytes | None] = []

        if not windows:
            return mp4_bytes_list, video_frames, windows

        if return_video_frames:
            video, frame_counts = fetch_video(
                str(input_file),
                sampling_fps=sampling_fps,
                window_range=windows,
                preprocess_mode=preprocess_mode,
                num_frames_to_use=num_frames_to_use,
                flip_input=flip_input,
                max_pixels_per_frame=max_pixels_per_frame,
            )

            index = 0
            for count in frame_counts:
                video_frames.append(video[index : index + count])
                index += count

        if return_bytes:
            if len(windows) == 1:
                raw = mp4_bytes.tobytes() if not isinstance(mp4_bytes, bytes) else mp4_bytes
                mp4_bytes_list.append(raw)
            else:
                for window in windows:
                    with make_pipeline_named_temporary_file(sub_dir="windowing") as tmp_file:
                        command = [
                            "ffmpeg",
                            "-threads",
                            str(num_threads),
                            "-y",
                            "-i",
                            str(input_file),
                            "-loglevel",
                            "error",
                            "-vf",
                            f"select='between(n\\,{window.start}\\,{window.end})',setpts=PTS-STARTPTS",
                            "-b:v",
                            str(target_bit_rate),
                            "-threads",
                            str(num_threads),
                            "-f",
                            "mp4",
                            "-an",
                            str(tmp_file),
                        ]
                        subprocess.check_call(command)  # noqa: S603
                        mp4_bytes_list.append(tmp_file.read_bytes())

        n = len(windows)
        video_frames.extend([None] * (n - len(video_frames)))
        mp4_bytes_list.extend([None] * (n - len(mp4_bytes_list)))

        return mp4_bytes_list, video_frames, windows


def _make_windows_for_clip(  # noqa: PLR0913
    clip: Clip,
    config: WindowConfig,
    target_bit_rate: str,
    num_decode_threads: int,
    *,
    preprocess_mode: PreprocessMode = PreprocessMode.CURATOR,
    keep_mp4: bool = False,
    return_frames: bool = True,
) -> tuple[list[Window], list[torch.Tensor]]:
    """Make windows for a clip.

    Args:
        clip: The clip to create windows for.
        config: The configuration for the windowing.
        target_bit_rate: The target bit rate.
        num_decode_threads: The number of threads to use.
        preprocess_mode: Owner of resize/rescale/normalize for returned video frames.
        keep_mp4: Whether to keep the MP4.
        return_frames: Whether to decode and return frame tensors.

    Returns:
        A tuple of lists of windows and frames.

    """
    windows: list[Window] = []
    frames: list[torch.Tensor] = []

    data = clip.encoded_data.resolve()
    if data is None:
        logger.error(f"clip {clip.uuid} does not have a encoded_data")
        clip.errors["clip_windowing"] = "clip.encoded_data is None"
        return windows, frames

    window_mp4_bytes, window_frames, window_infos = split_video_into_windows(
        data,
        window_size=config.window_size,
        remainder_threshold=config.remainder_threshold,
        sampling_fps=config.sampling_fps,
        preprocess_mode=preprocess_mode,
        return_bytes=keep_mp4,
        target_bit_rate=target_bit_rate,
        return_video_frames=return_frames,
        num_threads=num_decode_threads,
        max_pixels_per_frame=config.video_max_pixels_per_frame,
    )

    for window_bytes, window_frames_tensor, window_frame_info in zip(
        window_mp4_bytes, window_frames, window_infos, strict=True
    ):
        if return_frames and window_frames_tensor is None:
            logger.error(f"Window frames are None for window {window_frame_info.start} to {window_frame_info.end}")
            continue
        try:
            window = Window(
                start_frame=window_frame_info.start,
                end_frame=window_frame_info.end,
                mp4_bytes=window_bytes,
            )
            clip.windows.append(window)
            windows.append(window)
            if return_frames and window_frames_tensor is not None:
                frames.append(window_frames_tensor)
        except Exception as e:  # noqa: BLE001
            logger.exception("Error when splitting a video into windows")
            clip.errors["clip_windowing"] = str(e)

    return windows, frames


def make_windows_for_video(  # noqa: PLR0913
    video: Video,
    config: WindowConfig,
    num_decode_threads: int,
    *,
    preprocess_mode: PreprocessMode = PreprocessMode.CURATOR,
    keep_mp4: bool = False,
    return_frames: bool = True,
) -> tuple[list[Window], list[torch.Tensor]]:
    """Add windows to each clip in a video.

    Args:
        video: The video to make vLLM inputs for.
        config: The configuration for the windowing.
        num_decode_threads: The number of threads to use when decoding the video.
        preprocess_mode: Owner of resize/rescale/normalize for returned video frames.
        keep_mp4: Whether to keep the MP4.
        return_frames: Whether to decode and return frame tensors.

    Returns:
        A tuple of lists of windows and frames.

    """
    target_bit_rate = (
        f"{video.metadata.bit_rate_k}K" if config.use_input_bit_rate else f"{DEFAULT_TRANSCODE_BITRATE_M}M"
    )

    windows: list[Window] = []
    frames: list[torch.Tensor] = []

    for clip in video.clips:
        if not clip.encoded_data:
            logger.warning(f"Clip {clip.uuid} has no encoded_data.")
            clip.errors["encoded_data"] = "empty"
            continue

        _windows, _frames = _make_windows_for_clip(
            clip,
            config,
            target_bit_rate,
            num_decode_threads,
            preprocess_mode=preprocess_mode,
            keep_mp4=keep_mp4,
            return_frames=return_frames,
        )

        windows.extend(_windows)
        frames.extend(_frames)

    return windows, frames
