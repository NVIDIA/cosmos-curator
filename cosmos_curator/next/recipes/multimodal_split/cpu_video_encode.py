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

"""Write decoded RGB frames to an MP4.

This is the frame-wise counterpart to :mod:`cosmos_curator.next.media.ffmpeg`,
which cuts spans out of an existing file. A recipe that has already selected
frames onto a sampling grid cannot cut spans -- the frames it wants are not a
contiguous range of the source -- so it hands the pixels over instead.

It lives in this recipe rather than in ``next.media`` because it encodes through
PyAV and takes ``libopenh264``'s settings directly. Promoting it to shared code
is a decision for whoever needs a second caller, not for the first one.

``cpu`` names the intent rather than a hard limit: the signature takes an encoder
name, so nothing here refuses a GPU one. It is spelled this way to leave room for
a GPU sibling, matching the ``CpuVideoDecoder``/``GpuVideoDecoder`` split the
sensor library already uses on the decode side.

Frames are consumed one at a time from an iterable, never collected. A caller
that batches its decode to bound peak memory would lose that bound if this
materialized the clip.
"""

from collections.abc import Iterable
from fractions import Fraction
from pathlib import Path
from typing import cast

import av
import numpy as np
import numpy.typing as npt

_RGB_CHANNELS = 3
_RGB_NDIM = 3
_PIXEL_FORMAT = "yuv420p"
# ``rate`` takes a Fraction; every rate this recipe accepts is a whole number of
# frames per second, so the denominator is always 1.
_RATE_DENOMINATOR = 1


def encode_frames_to_path(  # noqa: PLR0913 -- the encode settings are one flat group, not a nested one
    frames: Iterable[npt.NDArray[np.uint8]],
    path: Path,
    *,
    fps: int,
    encoder: str,
    bitrate: str,
    gop_size: int,
    faststart: bool,
) -> int:
    """Encode ``frames`` to ``path`` as an MP4 and return how many were written.

    Geometry is taken from the first frame and every later frame must match it,
    because a stream's dimensions are fixed once its header is written. A
    mismatch means the caller mixed sources, which is a defect rather than a
    resize request.

    Writing nothing is an error. A clip with no frames is not a shorter clip, it
    is a caller that should have stopped earlier, and leaving a zero-frame MP4
    behind would let it look like output.

    Args:
        frames: RGB frames of identical shape ``(H, W, 3)`` and ``uint8`` dtype.
        path: destination file, which is left absent if nothing is written.
        fps: whole frames per second of the written timeline.
        encoder: FFmpeg video encoder name.
        bitrate: FFmpeg target video bitrate, such as ``4M``.
        gop_size: frames between keyframes.
        faststart: write the MP4 index ahead of the media rather than after it.

    Returns:
        The number of frames written.

    Raises:
        ValueError: if no frames are supplied, or a frame's shape or dtype does
            not match the first one.

    """
    iterator = iter(frames)
    first = next(iterator, None)
    if first is None:
        msg = f"Refusing to write {path}: no frames were supplied"
        raise ValueError(msg)
    _validate_frame(first, expected_shape=None)

    try:
        return _encode(
            _chain(first, iterator),
            path,
            fps=fps,
            encoder=encoder,
            bitrate=bitrate,
            gop_size=gop_size,
            faststart=faststart,
            shape=first.shape,
        )
    except BaseException:
        # A half-written MP4 is indistinguishable from a finished one to whatever
        # reads the output tree next, so a clip that fails leaves nothing behind.
        path.unlink(missing_ok=True)
        raise


def _encode(  # noqa: PLR0913
    frames: Iterable[npt.NDArray[np.uint8]],
    path: Path,
    *,
    fps: int,
    encoder: str,
    bitrate: str,
    gop_size: int,
    faststart: bool,
    shape: tuple[int, ...],
) -> int:
    height, width = int(shape[0]), int(shape[1])
    written = 0
    # faststart is a muxer setting rather than an encoder one, so it belongs to
    # the container and not to the stream below.
    options = {"movflags": "faststart"} if faststart else {}
    with av.open(str(path), mode="w", format="mp4", options=options) as container:
        # add_stream widens to a Video|Audio|Subtitle union for a non-literal
        # encoder name; only video encoders are ever requested here.
        stream = cast(
            "av.video.stream.VideoStream",
            container.add_stream(encoder, rate=Fraction(fps, _RATE_DENOMINATOR)),
        )
        stream.width = width
        stream.height = height
        stream.pix_fmt = _PIXEL_FORMAT
        stream.bit_rate = _parse_bitrate(bitrate)
        stream.gop_size = gop_size
        for frame in frames:
            _validate_frame(frame, expected_shape=shape)
            # ``from_numpy_buffer`` wraps the array rather than copying it into a
            # fresh AVFrame, which at 4K is a 24 MB copy per frame -- measured at
            # 10.8 ms against 0.6 ms, roughly 7% of a clip's wall clock. The
            # AVFrame holds a reference to the array, so it outlives this line for
            # as long as the encoder needs it. ``ascontiguousarray`` stays because
            # the wrap requires a contiguous buffer, and is a no-op when the frame
            # already is one.
            video_frame = av.VideoFrame.from_numpy_buffer(np.ascontiguousarray(frame), format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
            written += 1
        for packet in stream.encode(None):
            container.mux(packet)
    return written


def _chain(first: npt.NDArray[np.uint8], rest: Iterable[npt.NDArray[np.uint8]]) -> Iterable[npt.NDArray[np.uint8]]:
    yield first
    yield from rest


def _validate_frame(frame: npt.NDArray[np.uint8], *, expected_shape: tuple[int, ...] | None) -> None:
    if expected_shape is None:
        if frame.ndim != _RGB_NDIM or frame.shape[2] != _RGB_CHANNELS:
            msg = f"frames must have shape (H, W, {_RGB_CHANNELS}), got shape {frame.shape}"
            raise ValueError(msg)
    elif frame.shape != expected_shape:
        msg = f"every frame in one clip must have shape {expected_shape}, got shape {frame.shape}"
        raise ValueError(msg)
    if frame.dtype != np.uint8:
        msg = f"frames must be uint8, got {frame.dtype}"
        raise ValueError(msg)


def _parse_bitrate(bitrate: str) -> int:
    """Convert an FFmpeg-style bitrate such as ``4M`` or ``800K`` to bits per second."""
    multipliers = {"K": 1_000, "M": 1_000_000}
    suffix = bitrate[-1].upper()
    if suffix in multipliers:
        return int(float(bitrate[:-1]) * multipliers[suffix])
    return int(float(bitrate))
