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

"""Model-card endpoint sampling and sequential, bounded-frame-memory MP4 decoding."""

import math
from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO
from typing import Final, cast

import av
import numpy as np
import numpy.typing as npt

_NS_PER_SECOND = 1_000_000_000
_DECODE_ERROR: Final = "decode_error"
_INVALID_TIMESTAMPS: Final = "invalid_timestamps"
_DURATION_MISMATCH: Final = "duration_mismatch"


class VideoMediaError(ValueError):
    """A terminal item error caused by this episode's media."""

    def __init__(self, code: str, message: str, *, media_duration_ns: int | None = None) -> None:
        """Keep the stable error code and any measured media duration."""
        super().__init__(message)
        self.code = code
        self.media_duration_ns = media_duration_ns


@dataclass(frozen=True)
class VideoTimeline:
    """Stream timing and model-card sample indices, without retained decoded frames."""

    start_pts_ns: int
    duration_ns: int
    average_rate: Fraction
    frame_count: int
    selected_indices: tuple[int, ...]

    @property
    def duration_tolerance_ns(self) -> int:
        """Return two nominal frame periods, rounded upward to nanoseconds."""
        return math.ceil(Fraction(2 * _NS_PER_SECOND, 1) / self.average_rate)


@dataclass(frozen=True)
class DecodedVideo:
    """Only the selected RGB arrays plus the sequential decode-work count."""

    frames: tuple[npt.NDArray[np.uint8], ...]
    decoded_frame_count: int


def _video_stream(container: av.container.InputContainer) -> av.video.stream.VideoStream:
    streams = container.streams.video
    if len(streams) != 1:
        msg = f"Expected exactly one video stream, got {len(streams)}"
        raise VideoMediaError(_DECODE_ERROR, msg)
    return streams[0]


def _stream_timing(stream: av.video.stream.VideoStream) -> tuple[Fraction, int, int, Fraction]:
    """Read the stream interval and rate for AV duration validation, independent of sampling."""
    if stream.start_time is None or stream.duration is None or stream.time_base is None:
        msg = "Video stream lacks start_time, duration, or time_base"
        raise VideoMediaError(_INVALID_TIMESTAMPS, msg)
    time_base = Fraction(stream.time_base)
    if time_base <= 0:
        msg = "Video stream time_base must be positive"
        raise VideoMediaError(_INVALID_TIMESTAMPS, msg)
    start_ns = round(Fraction(stream.start_time) * time_base * _NS_PER_SECOND)
    duration_ns = round(Fraction(stream.duration) * time_base * _NS_PER_SECOND)
    if duration_ns <= 0:
        msg = "Video stream duration must be positive"
        raise VideoMediaError(_INVALID_TIMESTAMPS, msg)
    if stream.average_rate is None or Fraction(stream.average_rate) <= 0:
        msg = "Video stream average_rate must be positive"
        raise VideoMediaError(_INVALID_TIMESTAMPS, msg, media_duration_ns=duration_ns)
    return time_base, start_ns, duration_ns, Fraction(stream.average_rate)


def select_frame_indices(frame_count: int, *, sample_count: int = 8) -> tuple[int, ...]:
    """Use endpoint-inclusive integer linspace, including repeats for short videos."""
    if frame_count < 1:
        msg = "Video contains no frames"
        raise VideoMediaError(_DECODE_ERROR, msg)
    return tuple(int(index) for index in np.linspace(0, frame_count - 1, sample_count, dtype=np.int64))


def inspect_video(payload: bytes, *, frame_count: int, threads: int = 1) -> VideoTimeline:
    """Use the declared frame count, or count once and reopen for sampling."""
    try:
        with av.open(BytesIO(payload), mode="r") as container:
            stream = _video_stream(container)
            stream.thread_count = threads
            _time_base, start_ns, duration_ns, rate = _stream_timing(stream)
            source_count = stream.frames
            if not source_count:
                source_count = 0
                for frame in container.decode(stream):
                    if frame.is_corrupt:
                        msg = "Decoder reported a corrupt video frame"
                        raise VideoMediaError(_DECODE_ERROR, msg, media_duration_ns=duration_ns)
                    source_count += 1
            indices = select_frame_indices(source_count, sample_count=frame_count)
    except (av.FFmpegError, OSError) as exc:
        msg = f"Could not inspect video: {type(exc).__name__}"
        raise VideoMediaError(_DECODE_ERROR, msg) from exc
    return VideoTimeline(start_ns, duration_ns, rate, source_count, indices)


def check_episode_duration(timeline: VideoTimeline, *, episode_duration_ns: int) -> None:
    """Reject a media/episode span difference above two stream-frame periods."""
    if abs(timeline.duration_ns - episode_duration_ns) > timeline.duration_tolerance_ns:
        msg = (
            f"Media duration {timeline.duration_ns} ns differs from episode duration {episode_duration_ns} ns "
            f"by more than {timeline.duration_tolerance_ns} ns"
        )
        raise VideoMediaError(_DURATION_MISMATCH, msg, media_duration_ns=timeline.duration_ns)


def decode_selected_frames(payload: bytes, timeline: VideoTimeline, *, threads: int = 1) -> DecodedVideo:
    """Decode in presentation order, retain selected RGB only, and verify frame coverage."""
    selected: dict[int, npt.NDArray[np.uint8]] = {}
    wanted = set(timeline.selected_indices)
    decoded_count = 0
    try:
        with av.open(BytesIO(payload), mode="r") as container:
            stream = _video_stream(container)
            stream.thread_count = threads
            for index, frame in enumerate(container.decode(stream)):
                if frame.is_corrupt:
                    msg = "Decoder reported a corrupt video frame"
                    raise VideoMediaError(_DECODE_ERROR, msg, media_duration_ns=timeline.duration_ns)
                if index in wanted:
                    selected[index] = cast("npt.NDArray[np.uint8]", frame.to_ndarray(format="rgb24"))
                decoded_count += 1
    except (av.FFmpegError, OSError) as exc:
        msg = f"Could not decode video: {type(exc).__name__}"
        raise VideoMediaError(_DECODE_ERROR, msg, media_duration_ns=timeline.duration_ns) from exc
    if decoded_count != timeline.frame_count or set(selected) != wanted:
        msg = f"Decoded {decoded_count} frames but sampling expected {timeline.frame_count}"
        raise VideoMediaError(_DECODE_ERROR, msg, media_duration_ns=timeline.duration_ns)
    if len({frame.shape for frame in selected.values()}) != 1:
        msg = "Sampled frames have inconsistent dimensions"
        raise VideoMediaError(_DECODE_ERROR, msg, media_duration_ns=timeline.duration_ns)
    return DecodedVideo(tuple(selected[index] for index in timeline.selected_indices), decoded_count)
