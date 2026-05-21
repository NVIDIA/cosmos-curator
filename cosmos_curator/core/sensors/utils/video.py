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
"""Video utilities for the sensor library.

Given stream timestamps / keyframes / decoder state, these utilities decode
frames at the requested timestamps.
"""

import time
from collections.abc import Generator
from contextlib import contextmanager
from fractions import Fraction
from typing import (
    Any,
    BinaryIO,
    Protocol,
    Self,
    cast,
    overload,
)

import attrs
import av
import numpy as np
import numpy.typing as npt
from av.container import InputContainer
from loguru import logger

from cosmos_curator.core.sensors.data.camera_data import MotionVectorData, MotionVectorFrameData
from cosmos_curator.core.sensors.data.video import VideoIndex, VideoMetadata
from cosmos_curator.core.sensors.types.types import DataSource, VideoIndexCreationMethod
from cosmos_curator.core.sensors.utils.io import open_data_source

type DecodePlan = list[tuple[int, list[tuple[int, int]]]]
type DecodeResult = tuple[npt.NDArray[np.uint8], MotionVectorData | None]

_VALID_THREAD_TYPES = {"NONE", "FRAME", "SLICE", "AUTO"}
_CPU_OUTPUT_FORMATS: dict[str, tuple[np.dtype[np.uint8], int]] = {
    "rgb24": (np.dtype(np.uint8), 3),
}


@attrs.define(frozen=True)
class VideoDecodeConfig:
    """Base class for video decoder backend configuration."""


@attrs.define(frozen=True)
class CpuVideoDecodeConfig(VideoDecodeConfig):
    """Configuration for the CPU video decoder backend."""

    thread_type: str = "AUTO"
    thread_count: int = 4
    dest_format: str = "rgb24"
    export_mvs: bool = False

    def __attrs_post_init__(self) -> None:
        """Validate CPU decoder threading parameters."""
        if self.thread_type not in _VALID_THREAD_TYPES:
            msg = f"thread_type must be one of {sorted(_VALID_THREAD_TYPES)}, got {self.thread_type!r}"
            raise ValueError(msg)
        if self.thread_count < 0:
            msg = f"thread_count must be non-negative, got {self.thread_count}"
            raise ValueError(msg)
        if self.dest_format not in _CPU_OUTPUT_FORMATS:
            msg = f"unsupported dest_format {self.dest_format!r}"
            raise ValueError(msg)

    @property
    def dest_dtype(self) -> np.dtype[np.uint8]:
        """Return the NumPy dtype for ``dest_format``."""
        return _CPU_OUTPUT_FORMATS[self.dest_format][0]

    @property
    def channels(self) -> int:
        """Return the number of output channels for ``dest_format``."""
        return _CPU_OUTPUT_FORMATS[self.dest_format][1]


@attrs.define(frozen=True)
class GpuVideoDecodeConfig(VideoDecodeConfig):
    """Configuration for the GPU video decoder backend."""


DEFAULT_VIDEO_DECODE_CONFIG = CpuVideoDecodeConfig()


class VideoDecoder(Protocol):
    """Decoder interface for exact frame extraction from a video stream."""

    @property
    def time_base(self) -> Fraction:
        """Return the stream time base for the opened decoder session."""

    def decode(self, decode_plan: DecodePlan) -> DecodeResult:
        """Decode frames described by ``decode_plan``."""


class CpuVideoDecoder:
    """CPU implementation of exact frame extraction for one opened video stream."""

    def __init__(
        self,
        container: InputContainer,
        stream: av.video.stream.VideoStream,
        config: CpuVideoDecodeConfig | None = None,
        stats: dict[str, float] | None = None,
    ) -> None:
        """Initialize the decoder for one already-opened video stream.

        Preferred method for creating a decoder is via the :meth:`open`
        class method.

        Args:
            container: The PyAV container containing the video stream.
            stream: The PyAV video stream to decode.
            config: The configuration for the decoder.
            stats: Optional dict for benchmarking instrumentation.

        Raises:
            ValueError: If the selected video stream has no ``time_base``.

        """
        if stream.time_base is None:
            msg = "Time base is None for the opened video stream"
            raise ValueError(msg)
        self.container = container
        self.stream = stream
        self._time_base = stream.time_base
        self.config = config or CpuVideoDecodeConfig()
        self.stats = stats
        self.stream.thread_type = self.config.thread_type
        self.stream.thread_count = self.config.thread_count
        if self.config.export_mvs:
            self.stream.codec_context.flags2 |= av.codec.context.Flags2.export_mvs
        self.last_decoded_pts: int | None = None

    @property
    def time_base(self) -> Fraction:
        """Return the stream time base for the opened decoder session."""
        return self._time_base

    @classmethod
    @contextmanager
    def open(
        cls,
        source: DataSource,
        stream_idx: int = 0,
        config: CpuVideoDecodeConfig | None = None,
        stats: dict[str, float] | None = None,
    ) -> Generator[Self, None, None]:
        """Open a CPU decoder session that owns the underlying video source.

        The returned session owns the source/container/stream lifetime for the
        duration of the context manager and exposes the validated stream
        ``time_base`` through ``decoder.time_base``.

        Args:
            source: Video data source to decode. See
                :data:`cosmos_curator.core.sensors.types.types.DataSource`.
                Cloud-backed inputs must be wrapped as a
                :class:`io.BufferedIOBase` by the caller.
            stream_idx: PyAV index of the video stream to decode, usually 0.
            config: Optional CPU decoder configuration. Defaults to
                :class:`CpuVideoDecodeConfig` with FFmpeg-controlled threading.
            stats: Optional dict for benchmarking instrumentation. When
                provided, ``t_seek`` (seek + flush_buffers wall time),
                ``t_convert`` (``to_ndarray`` wall time), ``t_copy``
                (output-buffer write time), and ``frames_decoded`` are
                accumulated. Pass ``None`` (default) in production.

        Yields:
            An open :class:`CpuVideoDecoder` session ready for repeated
            :meth:`decode` calls.

        Raises:
            ValueError: If the selected video stream has no ``time_base``.

        """
        with (
            open_data_source(source, mode="rb") as stream,
            open_video_container(stream, stream_idx=stream_idx) as (container, video_stream),
        ):
            yield cls(container, video_stream, config, stats)

    def _accumulate_stat(self, key: str, delta: float) -> None:
        """Add ``delta`` to ``stats[key]`` if instrumentation is enabled."""
        if self.stats is not None:
            self.stats[key] = self.stats.get(key, 0.0) + delta

    def _increment_frames_decoded(self) -> None:
        """Increment the decoded-frame counter if instrumentation is enabled."""
        if self.stats is not None:
            self.stats["frames_decoded"] = self.stats.get("frames_decoded", 0) + 1

    def _seek_and_flush(self, kf_pts_stream: int) -> None:
        """Seek to one governing keyframe and flush decoder state."""
        t0 = time.perf_counter()
        self.container.seek(kf_pts_stream, stream=self.stream)
        self.stream.codec_context.flush_buffers()
        self._accumulate_stat("t_seek", time.perf_counter() - t0)

    def _validate_monotonic_frame_pts(self, frame_pts: int, kf_pts_stream: int) -> None:
        """Ensure decoded frame PTS values stay strictly increasing across seeks."""
        if self.last_decoded_pts is not None and frame_pts <= self.last_decoded_pts:
            msg = (
                f"Non-monotonic frame pts={frame_pts} decoded after {self.last_decoded_pts} "
                f"(GOP kf_pts_stream={kf_pts_stream}) — "
                "seek landed before the keyframe or decoder output regressed in presentation order. "
                "This indicates a bug in the decode plan or a malformed video stream."
            )
            raise RuntimeError(msg)
        self.last_decoded_pts = frame_pts

    def _copy_decoded_frame(
        self,
        dest: npt.NDArray[np.uint8],
        dest_idx: int,
        frame: av.VideoFrame,
        current_count: int,
    ) -> int:
        """Convert one decoded frame to rgb24 and copy it into the output buffer."""
        t0 = time.perf_counter()
        arr = frame.to_ndarray(format=self.config.dest_format)
        self._accumulate_stat("t_convert", time.perf_counter() - t0)

        t0 = time.perf_counter()
        if current_count == 1:
            dest[dest_idx] = arr
        else:
            dest[dest_idx : dest_idx + current_count] = np.broadcast_to(
                arr, (current_count, self.stream.height, self.stream.width, self.config.channels)
            )
        self._accumulate_stat("t_copy", time.perf_counter() - t0)
        return dest_idx + current_count

    @staticmethod
    def _motion_vector_frame_from_video_frame(frame: av.VideoFrame) -> MotionVectorFrameData:
        """Convert PyAV motion-vector side data into the sensor data model."""
        for side_data in frame.side_data:
            if side_data.type != av.sidedata.sidedata.Type.MOTION_VECTORS:  # type: ignore[attr-defined]
                continue

            raw_motion_vectors = side_data.to_ndarray()  # type: ignore[attr-defined]
            field_names = raw_motion_vectors.dtype.names
            if field_names is None:
                msg = "PyAV motion-vector side data did not expose structured fields"
                raise ValueError(msg)

            return MotionVectorFrameData(
                source=np.asarray(raw_motion_vectors["source"], dtype=np.int32),
                w=np.asarray(raw_motion_vectors["w"], dtype=np.int32),
                h=np.asarray(raw_motion_vectors["h"], dtype=np.int32),
                src_x=np.asarray(raw_motion_vectors["src_x"], dtype=np.int32),
                src_y=np.asarray(raw_motion_vectors["src_y"], dtype=np.int32),
                dst_x=np.asarray(raw_motion_vectors["dst_x"], dtype=np.int32),
                dst_y=np.asarray(raw_motion_vectors["dst_y"], dtype=np.int32),
                flags=np.asarray(raw_motion_vectors["flags"], dtype=np.int64),
                motion_x=np.asarray(raw_motion_vectors["motion_x"], dtype=np.int32),
                motion_y=np.asarray(raw_motion_vectors["motion_y"], dtype=np.int32),
                motion_scale=np.asarray(raw_motion_vectors["motion_scale"], dtype=np.int32),
            )

        return MotionVectorFrameData.empty()

    def _decode_group(
        self,
        dest: npt.NDArray[np.uint8],
        dest_idx: int,
        kf_pts_stream: int,
        group_targets: list[tuple[int, int]],
        motion_vector_frames: list[MotionVectorFrameData] | None = None,
    ) -> int:
        """Decode one GOP-worth of target frames after seeking to its keyframe."""
        self._seek_and_flush(kf_pts_stream)

        target_idx = 0
        for packet in self.container.demux(self.stream):
            try:
                frames_in_packet = list(packet.decode())
            except av.error.EOFError:
                break

            for frame in frames_in_packet:
                if frame.pts is None:
                    continue

                self._validate_monotonic_frame_pts(frame.pts, kf_pts_stream)
                self._increment_frames_decoded()

                if target_idx >= len(group_targets) or frame.pts != group_targets[target_idx][0]:
                    continue

                current_count = group_targets[target_idx][1]
                if self.config.export_mvs and motion_vector_frames is not None:
                    motion_vector_frame = self._motion_vector_frame_from_video_frame(frame)
                    motion_vector_frames.extend([motion_vector_frame] * current_count)

                dest_idx = self._copy_decoded_frame(dest, dest_idx, frame, current_count)
                target_idx += 1
                if target_idx >= len(group_targets):
                    return dest_idx

            if packet.size == 0:
                break

        if target_idx < len(group_targets):
            missing = [pts for pts, _ in group_targets[target_idx:]]
            msg = (
                f"GOP kf_pts_stream={kf_pts_stream}: {len(missing)} target(s) not found in "
                f"decoded frames — container index may not match packet stream. "
                f"Missing pts_stream: {missing[:5]}{'...' if len(missing) > 5 else ''}"  # noqa: PLR2004
            )
            raise ValueError(msg)
        return dest_idx

    def decode(self, decode_plan: DecodePlan) -> DecodeResult:
        """Decode the exact frame for each target timestamp in ``decode_plan``.

        For each ``(kf_pts_stream, group)`` entry in ``decode_plan``:

        - Seeks to ``kf_pts_stream`` (stream-native pts, no conversion needed).
        - Flushes the codec buffer.
        - Decodes forward. When a frame whose PTS exactly matches a target is
          reached, copies it to the output and advances to the next target.
        - Breaks out of demux as soon as all targets in the group are resolved,
          avoiding reading the rest of that GOP from disk.

        Every target PTS must be an exact frame PTS from the container index.
        Since callers build targets from ``VideoIndex.pts_stream``, this is
        always true. All comparisons are done in stream-native pts units to
        avoid the lossy ns↔stream_pts round-trip (which causes precision errors
        for fps-rate time_bases such as ``Fraction(1, 30)``).

        Across GOP boundaries, this implementation assumes decoded frame PTS
        values remain strictly increasing after each seek. If decoder output
        regresses or re-emits earlier presentation timestamps at a GOP seam,
        that is treated as malformed stream data or unsupported seek behavior
        and raises ``RuntimeError``.

        Args:
            decode_plan: List of ``(kf_pts_stream, group)`` as returned by
                :func:`make_decode_plan`. Each group is a list of
                ``(pts_stream, count)`` pairs, all in stream time_base units.

        Returns:
            Tuple of ``(frames, motion_vectors)``. ``frames`` is a fresh
            ``uint8`` array of shape ``(total_count, height, width, 3)`` in
            rgb24 order. ``motion_vectors`` is ``None`` unless
            ``CpuVideoDecodeConfig.export_mvs`` is enabled, in which case it
            contains one aligned per-frame payload per decoded RGB frame.

            ``total_count = sum(count for _, group in decode_plan for _, count in group)``.

        Raises:
            RuntimeError: If decoded frame PTS values are duplicated or move
                backward. This usually indicates a bug in the seek/decode plan
                logic, but can also occur when the container index or decoded
                stream timestamps are malformed.
            ValueError: If a target PTS in the plan was not found in the
                decoded frames (container index / packet stream mismatch).

        """
        total_count = sum(count for _, group in decode_plan for _, count in group)
        frames_np = np.empty(
            (total_count, self.stream.height, self.stream.width, self.config.channels), dtype=self.config.dest_dtype
        )
        frames_np_idx = 0
        self.last_decoded_pts = None
        motion_vector_frames: list[MotionVectorFrameData] | None = [] if self.config.export_mvs else None
        for kf_pts_stream, group in decode_plan:
            frames_np_idx = self._decode_group(frames_np, frames_np_idx, kf_pts_stream, group, motion_vector_frames)

        if frames_np_idx != total_count:
            msg = (
                f"Decode produced {frames_np_idx} frame slots but expected {total_count} — "
                "internal accounting error (bug in decode logic)."
            )
            raise RuntimeError(msg)

        motion_vectors = (
            None
            if motion_vector_frames is None
            else MotionVectorData(
                frames=motion_vector_frames,  # type: ignore[arg-type]  # attrs converter stores an immutable tuple
            )
        )
        return frames_np, motion_vectors


class GpuVideoDecoder:
    """GPU decoder placeholder for future exact frame extraction."""

    def __init__(
        self,
        container: InputContainer,
        stream: av.video.stream.VideoStream,
        config: GpuVideoDecodeConfig | None = None,
        stats: dict[str, float] | None = None,
    ) -> None:
        """Initialize the decoder for one already-opened video stream."""
        if stream.time_base is None:
            msg = "Time base is None for the opened video stream"
            raise ValueError(msg)
        self.container = container
        self.stream = stream
        self._time_base = stream.time_base
        self.config = config or GpuVideoDecodeConfig()
        self.stats = stats

    @property
    def time_base(self) -> Fraction:
        """Return the stream time base for the opened decoder session."""
        return self._time_base

    @classmethod
    @contextmanager
    def open(
        cls,
        source: DataSource,
        stream_idx: int = 0,
        config: GpuVideoDecodeConfig | None = None,
        stats: dict[str, float] | None = None,
    ) -> Generator["GpuVideoDecoder", None, None]:
        """Open a decoder session that owns the underlying video source."""
        with (
            open_data_source(source, mode="rb") as stream,
            open_video_container(stream, stream_idx=stream_idx) as (container, video_stream),
        ):
            yield cls(container, video_stream, config, stats)

    def decode(self, decode_plan: DecodePlan) -> DecodeResult:
        """Decode the frame closest to each target timestamp using the GPU."""
        del decode_plan
        msg = "GPU decode mode not implemented"
        raise NotImplementedError(msg)


class _HeaderIndexUnavailableError(ValueError):
    """Header-based packet indexing is unavailable for this stream."""


@overload
def pts_to_ns(pts: int, time_base: Fraction) -> int: ...


@overload
def pts_to_ns(pts: npt.NDArray[np.int64], time_base: Fraction) -> npt.NDArray[np.int64]: ...


def pts_to_ns(pts: int | npt.NDArray[np.int64], time_base: Fraction) -> int | npt.NDArray[np.int64]:
    """Convert a packet PTS value to nanoseconds (MCAP-compatible time base).

    Args:
        pts: packet PTS value in stream-native time_base units.
        time_base: time base of the video stream.

    Returns:
        PTS value in nanoseconds.

    """
    # time_base.denominator is guaranteed non-zero by the Fraction initializer.
    #
    # Overflow note: the intermediate product  pts * 1_000_000_000 * numerator
    # exceeds int64_max for pts > ~9.22e9 when numerator == 1 (e.g. H.264 at
    # 90 kHz overflows for recordings longer than ~28.5 hours).  The scalar
    # path is already safe because Python ints are arbitrary precision.  For
    # the array path we promote to object dtype (Python ints) for the
    # intermediate multiply then cast back, mirroring the i128 widening cast
    # used in Rust/C++:
    #
    # >>>  // Rust equivalent
    # >>>  fn pts_to_ns(pts: i64, numerator: i64, denominator: i64) -> i64 {
    # >>>      (pts as i128 * 1_000_000_000 * numerator as i128 / denominator as i128) as i64
    # >>>  }
    if isinstance(pts, np.ndarray):
        pts_wide = pts.astype(object)  # promotes each element to a Python int
        return (pts_wide * 1_000_000_000 * time_base.numerator // time_base.denominator).astype(pts.dtype)
    return int(pts * 1_000_000_000 * time_base.numerator // time_base.denominator)


@contextmanager
def open_video_container(
    stream: BinaryIO,
    stream_idx: int = 0,
    video_format: str | None = None,
) -> Generator[tuple[InputContainer, av.video.stream.VideoStream], None, None]:
    """Context manager for an already-open binary stream containing video data.

    This provides a common pattern for opening video containers and extracting
    one video stream. The caller retains ownership of the underlying binary stream.

    Args:
        stream: already-open binary stream containing video data. The caller
            retains ownership and is responsible for closing it.
        stream_idx: PyAv index of the video stream to decode, usually 0.
        video_format: Format of the video stream, like "mp4", "mkv", etc.
            None is probably best

    Yields:
        Tuple of ``(container, video_stream)``.

    Example:
        with (
            video_path.open("rb") as stream:
            open_video_container(stream) as (container, video_stream)
        ):
            for packet in container.demux(video=0):
                if packet.pts is not None and video_stream.time_base is not None:
                    ts = float(packet.pts) * video_stream.time_base

    """
    # PyAV's ``av.open`` reads from the stream's current position rather than
    # from absolute 0. The sensor library's DataSource contract for
    # ``io.BufferedIOBase`` inputs is "use absolute offsets starting from 0";
    # we realise that here by seeking to 0 before handing the stream to PyAV.
    # ``open_data_source`` itself remains a thin pass-through (no seek / no
    # restore); positioning at the container origin is the responsibility of
    # the consumer wrapper (this function, for video, or the image sensor's
    # load helper for PIL).
    if stream.seekable():
        stream.seek(0)
    with av.open(stream, format=video_format) as container:
        container = cast("InputContainer", container)
        video_stream = container.streams.video[stream_idx]
        yield container, video_stream


def _get_video_index_full_demux(
    container: InputContainer, stream_idx: int
) -> tuple[list[int], list[int], list[int], list[bool], list[bool]]:
    """Build packet lists by demuxing the stream (``VideoIndexCreationMethod.FULL_DEMUX``)."""
    offset: list[int] = []
    size: list[int] = []
    pts: list[int] = []
    is_keyframe: list[bool] = []
    is_discard: list[bool] = []

    for packet in container.demux(video=stream_idx):
        if packet.pts is None or packet.pos is None:
            continue

        pts.append(packet.pts)
        offset.append(packet.pos)
        size.append(packet.size)
        is_keyframe.append(packet.is_keyframe)
        is_discard.append(packet.is_discard)

    return offset, size, pts, is_keyframe, is_discard


def _get_video_index_from_header(
    video_stream: av.video.stream.VideoStream,
) -> tuple[list[int], list[int], list[int], list[bool], list[bool]]:
    """Build packet lists from header index entries (``VideoIndexCreationMethod.FROM_HEADER``).

    Raises:
        _HeaderIndexUnavailableError: If the stream exposes no usable
            ``index_entries`` and callers should retry with full demux.

    """
    offset: list[int] = []
    size: list[int] = []
    pts: list[int] = []
    is_keyframe: list[bool] = []
    is_discard: list[bool] = []

    # PyAV exposes CFFI index entries at runtime; stubs omit ``index_entries``.
    # Note: PyAV did not provide index_entries until version 17.
    stream_index: Any = video_stream
    index_entries = getattr(stream_index, "index_entries", None)
    if index_entries is None:
        msg = "stream does not expose header index entries; retry with FULL_DEMUX"
        raise _HeaderIndexUnavailableError(msg)

    for entry in index_entries:
        pts.append(entry.timestamp)
        offset.append(entry.pos)
        size.append(entry.size)
        is_keyframe.append(entry.is_keyframe)
        is_discard.append(entry.is_discard)

    if len(pts) == 0:
        msg = "stream header index is empty; retry with FULL_DEMUX"
        raise _HeaderIndexUnavailableError(msg)

    return offset, size, pts, is_keyframe, is_discard


def make_index_and_metadata(
    data: DataSource,
    stream_idx: int = 0,
    video_format: str | None = None,
    index_method: VideoIndexCreationMethod = VideoIndexCreationMethod.FROM_HEADER,
    allow_header_fallback: bool = True,  # noqa: FBT001, FBT002
) -> tuple[VideoIndex, VideoMetadata]:
    """Build a :class:`VideoIndex` and :class:`VideoMetadata` from a video source.

    Packets are indexed in presentation timestamp (PTS) order, not file/decode
    order.  For B-frame video (e.g. H.264 main/high profile), the container
    stores packets in decode order (DTS order), which produces a non-monotonic
    PTS sequence such as ``[0, 3, 1, 2, 6, 4, 5, ...]``.  All arrays are
    argsorted by PTS so that ``VideoIndex.pts_ns`` is always monotonically
    increasing.  As a consequence, ``VideoIndex.offset`` is **not** monotonically
    increasing for B-frame video — it is a per-packet lookup, not a scan order.

    Args:
        data: Video data source. See
            :data:`cosmos_curator.core.sensors.types.types.DataSource`.
            Cloud-backed inputs must be wrapped as a
            :class:`io.BufferedIOBase` by the caller.
        stream_idx: index of the video stream to use (default: 0).
        video_format: container format hint (default: ``None``; let libav detect).
        index_method: how to collect per-packet metadata.  ``FROM_HEADER`` (default)
            reads from the container index (fast).  ``FULL_DEMUX`` scans every
            packet (slow; for tests or rare validation).
        allow_header_fallback: If ``True`` and ``FROM_HEADER`` is unavailable,
            transparently fall back to ``FULL_DEMUX``. If ``False``, raise the
            header-index error instead. Diagnostics should set this to ``False``
            when they need to know whether the embedded header index is usable.

    Returns:
        ``(index, metadata)`` where ``index`` holds per-packet arrays and
        ``time_base``, and ``metadata`` holds scalar stream properties
        (codec, resolution, container format).

    Raises:
        ValueError: if the stream contains no packets with valid PTS.
        ValueError: if the stream contains no keyframes.

    """
    with (
        open_data_source(data, mode="rb") as stream,
        open_video_container(stream, stream_idx=stream_idx, video_format=video_format) as (
            container,
            video_stream,
        ),
    ):
        if video_stream.time_base is None:
            error_msg = f"Time base is None for video stream {stream_idx}"
            raise ValueError(error_msg)
        time_base = video_stream.time_base
        match index_method:
            case VideoIndexCreationMethod.FULL_DEMUX:
                offset, size, pts, is_keyframe, is_discard = _get_video_index_full_demux(container, stream_idx)
            case VideoIndexCreationMethod.FROM_HEADER:
                try:
                    offset, size, pts, is_keyframe, is_discard = _get_video_index_from_header(video_stream)
                except _HeaderIndexUnavailableError as e:
                    if not allow_header_fallback:
                        raise
                    logger.warning(
                        "FROM_HEADER unavailable for stream {} ({}); falling back to FULL_DEMUX",
                        stream_idx,
                        e,
                    )
                    offset, size, pts, is_keyframe, is_discard = _get_video_index_full_demux(container, stream_idx)
            case _:
                error_msg = f"unsupported index_method: {index_method!r}"  # type: ignore[unreachable]
                raise ValueError(error_msg)

        pts_stream_np = np.array(pts, dtype=np.int64)
        pts_ns_np = pts_to_ns(pts_stream_np, time_base)
        offset_np = np.array(offset, dtype=np.int64)
        size_np = np.array(size, dtype=np.int64)
        is_keyframe_np = np.array(is_keyframe, dtype=np.bool_)
        is_discard_np = np.array(is_discard, dtype=np.bool_)

        # Sort all arrays by PTS so that pts_ns is monotonically increasing.
        # For non-B-frame video this is a no-op.  For B-frame video, packets
        # arrive in decode order (DTS order) which is not PTS order, so without
        # this sort pts_ns[-1] would not be the maximum PTS and any sorted
        # assumption (e.g. searchsorted in make_decode_plan) would be violated.
        sort_idx = np.argsort(pts_ns_np, kind="stable")
        pts_ns_np = pts_ns_np[sort_idx]
        pts_stream_np = pts_stream_np[sort_idx]
        offset_np = offset_np[sort_idx]
        size_np = size_np[sort_idx]
        is_keyframe_np = is_keyframe_np[sort_idx]
        is_discard_np = is_discard_np[sort_idx]

        if len(pts_ns_np) == 0:
            error_msg = f"video stream {stream_idx} contains no packets with valid PTS"
            raise ValueError(error_msg)

        if not is_keyframe_np.any():
            error_msg = f"video stream {stream_idx} contains no keyframes"
            raise ValueError(error_msg)

        kf_pts_ns_np = pts_ns_np[is_keyframe_np]
        kf_pts_stream_np = pts_stream_np[is_keyframe_np]

        index = VideoIndex(
            offset=offset_np,
            size=size_np,
            pts_ns=pts_ns_np,
            pts_stream=pts_stream_np,
            is_keyframe=is_keyframe_np,
            is_discard=is_discard_np,
            kf_pts_ns=kf_pts_ns_np,
            kf_pts_stream=kf_pts_stream_np,
            time_base=time_base,
        )

        duration_s = (pts_ns_np[-1] - pts_ns_np[0]) / 1_000_000_000.0
        bit_rate_bps = int(size_np.sum() * 8 / duration_s) if duration_s > 0 else 0
        avg_rate = video_stream.average_rate

        metadata = VideoMetadata(
            codec_name=video_stream.codec_context.name,
            codec_max_bframes=video_stream.codec_context.max_b_frames,
            codec_profile=video_stream.codec_context.profile or "",
            container_format=container.format.name,
            height=video_stream.height,
            width=video_stream.width,
            # Defensive code, denominator should never be zero, handle it anyway
            avg_frame_rate=(
                Fraction(avg_rate.numerator, avg_rate.denominator)
                if avg_rate and avg_rate.denominator != 0
                else Fraction(0)
            ),
            pix_fmt=str(video_stream.codec_context.pix_fmt) if video_stream.codec_context.pix_fmt else "",
            bit_rate_bps=bit_rate_bps,
        )

        return index, metadata


def _validate_decode_plan_timestamp_inputs(
    kf_pts_stream: npt.NDArray[np.int64],
    pts_stream: npt.NDArray[np.int64],
) -> None:
    """Validate keyframe/target timestamp inputs for make_decode_plan."""
    if kf_pts_stream.ndim != 1:
        error_msg = f"kf_pts_stream must be 1-D, got ndim={kf_pts_stream.ndim}"
        raise ValueError(error_msg)

    if pts_stream.ndim != 1:
        error_msg = f"pts_stream must be 1-D, got ndim={pts_stream.ndim}"
        raise ValueError(error_msg)

    if len(kf_pts_stream) == 0:
        error_msg = "kf_pts_stream is empty (no keyframes)"
        raise ValueError(error_msg)

    if len(kf_pts_stream) > 1 and not np.all(kf_pts_stream[:-1] < kf_pts_stream[1:]):
        error_msg = "kf_pts_stream must be sorted in ascending order"
        raise ValueError(error_msg)

    if len(pts_stream) == 0:
        return

    if len(pts_stream) > 1 and not np.all(pts_stream[:-1] < pts_stream[1:]):
        error_msg = "pts_stream must be sorted in ascending order"
        raise ValueError(error_msg)

    before_first_kf = pts_stream[pts_stream < kf_pts_stream[0]]
    if len(before_first_kf) > 0:
        error_msg = (
            f"pts_stream contains timestamps before the first keyframe ({kf_pts_stream[0]}): {before_first_kf.tolist()}"
        )
        raise ValueError(error_msg)


def _validate_decode_plan_counts(
    pts_stream: npt.NDArray[np.int64],
    counts: npt.NDArray[np.int64],
) -> None:
    """Validate per-target multiplicities for make_decode_plan."""
    if len(pts_stream) != len(counts):
        error_msg = f"pts_stream and counts must have the same length: {len(pts_stream)} != {len(counts)}"
        raise ValueError(error_msg)

    if counts.ndim != 1:
        error_msg = f"counts must be 1-D, got ndim={counts.ndim}"
        raise ValueError(error_msg)

    if np.any(counts <= 0):
        error_msg = f"counts must be strictly positive, got {counts.tolist()}"
        raise ValueError(error_msg)


def make_decode_plan(
    kf_pts_stream: npt.NDArray[np.int64],
    pts_stream: npt.NDArray[np.int64],
    counts: npt.NDArray[np.int64],
) -> DecodePlan:
    """Pre-compute all seek targets and their associated frame targets.

    Groups target timestamps by their governing keyframe so that multiple
    targets in the same GOP require only one seek.  Returns entries in
    ascending keyframe order so all seeks are strictly forward.

    All timestamp values are in stream-native time_base units (not nanoseconds).
    Passing stream pts directly avoids lossy ns↔stream_pts round-trips for
    fps-rate time_bases (e.g. ``Fraction(1, 30)``).

    Args:
        kf_pts_stream: Monotonically increasing keyframe presentation timestamps
            in stream time_base units.  Callers typically obtain this from
            ``VideoIndex.kf_pts_stream``.
        pts_stream: Sorted array of target frame timestamps in stream time_base
            units.  Each entry is a frame the decoder must produce.
        counts: Number of times each target frame appears in the output.
            Must be the same length as ``pts_stream``.

    Returns:
        List of ``(kf_pts_stream, group)`` pairs, one per unique governing
        keyframe.  ``group`` is a list of ``(pts_stream, count)`` pairs for
        all targets whose GOP starts at ``kf_pts_stream``, in ascending
        timestamp order.

        kf_pts_stream -> list of (pts_stream, count) pairs

    Raises:
        ValueError: If ``kf_pts_stream`` is empty.
        ValueError: If ``kf_pts_stream`` is not sorted in ascending order.
        ValueError: If ``pts_stream`` and ``counts`` differ in length.
        ValueError: If ``counts`` is not 1-D or contains non-positive values.
        ValueError: If ``pts_stream`` is not sorted in ascending order.
        ValueError: If any target in ``pts_stream`` is before the first keyframe.

    """
    _validate_decode_plan_timestamp_inputs(kf_pts_stream, pts_stream)
    _validate_decode_plan_counts(pts_stream, counts)

    if len(pts_stream) == 0:
        return []

    # For each target find the preceding keyframe (the I-frame that opens the
    # GOP containing that target).
    # searchsorted(..., side='right') - 1  →  last kf_pts_stream <= target
    insert = np.searchsorted(kf_pts_stream, pts_stream, side="right") - 1
    np.clip(insert, 0, len(kf_pts_stream) - 1, out=insert)
    governing_kf_pts = kf_pts_stream[insert]

    # Group targets by governing keyframe, preserving ascending order.
    unique_kf_pts = np.unique(governing_kf_pts)
    plan: list[tuple[int, list[tuple[int, int]]]] = []
    for kf in unique_kf_pts:
        mask = governing_kf_pts == kf
        group: list[tuple[int, int]] = list(zip(pts_stream[mask].tolist(), counts[mask].tolist(), strict=True))
        plan.append((int(kf), group))

    return plan
