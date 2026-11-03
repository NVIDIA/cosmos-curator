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
"""Forward-only streaming camera sensor."""

import itertools
from collections.abc import Generator, Iterator
from fractions import Fraction

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.data.mp4 import Mp4FrameIndex, Mp4Metadata, TimestampArray
from cosmos_curator.core.sensors.data.streaming_camera_data import StreamingCameraData, StreamingFrame
from cosmos_curator.core.sensors.mp4.indexing import Mp4Indexer, index_video
from cosmos_curator.core.sensors.sampling.grid import SamplingWindow
from cosmos_curator.core.sensors.sampling.policy import NearestTimestampPolicy
from cosmos_curator.core.sensors.sampling.sampler import find_closest_indices
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.types.types import DataSource, TimestampSource, VideoFrameAlignmentField
from cosmos_curator.core.sensors.utils.validation import require_1d, require_strictly_increasing
from cosmos_curator.core.sensors.utils.video import (
    DEFAULT_VIDEO_DECODE_CONFIG,
    CpuVideoDecodeConfig,
    StreamingDecodePlan,
    StreamingVideoDecoder,
    execute_streaming_decode_plan,
    make_streaming_decode_plan,
)

_MIN_FOR_AN_INTERVAL = 2
_INT64_MAX = int(np.iinfo(np.int64).max)

_INDEX_TIMESTAMP_SOURCES: dict[VideoFrameAlignmentField, TimestampSource] = {
    VideoFrameAlignmentField.DTS_NS: TimestampSource.CONTAINER_DTS,
    VideoFrameAlignmentField.PTS_NS: TimestampSource.CONTAINER_PTS,
    VideoFrameAlignmentField.EDITED_PTS_NS: TimestampSource.CONTAINER_PTS,
    VideoFrameAlignmentField.EOF_NS: TimestampSource.SENSOR,
    VideoFrameAlignmentField.SOF_NS: TimestampSource.SENSOR,
    VideoFrameAlignmentField.MOE_NS: TimestampSource.SENSOR,
}


class IndexRecordUnusableError(ValueError):
    """A frame index cannot describe this stream or the selected timeline.

    Raised when the index has no frames, its frame count disagrees with
    metadata, or the selected timestamp array has missing values.
    """


def _validate_timestamp_array(timestamps_ns: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Validate and return a nonempty 1-D int64 timestamp array.

    Raises:
        ValueError: If the array is not 1-D, does not have dtype int64, or is empty.

    """
    require_1d("timestamps_ns", timestamps_ns, np.int64)
    if timestamps_ns.size == 0:
        msg = "timestamps_ns must describe at least one observation"
        raise ValueError(msg)
    return timestamps_ns


def _avg_frame_rate(metadata: Mp4Metadata) -> Fraction:
    """Rebuild the nominal frame rate, or ``Fraction(0)`` when the container declares none."""
    if metadata.avg_frame_rate_den == 0:
        return Fraction(0)
    return Fraction(metadata.avg_frame_rate_num, metadata.avg_frame_rate_den)


def _serving_timestamps(
    timeline_ns: npt.NDArray[np.int64], references_ns: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """Return the timeline timestamp selected for each reference.

    Which observation serves a reference is decided by the supplied timestamps alone, so
    the answer exists before anything is decoded. The walk reaches it by
    stepping and this by lookup, and they agree because ``find_closest_indices``
    is the same rule the indexed sensor samples by.
    """
    if references_ns.size == 0:
        return references_ns
    return timeline_ns[find_closest_indices(timeline_ns, references_ns)]


@attrs.define(slots=True)
class _WindowValidationState:
    """Carries the previous window's bounds so overlap is caught as it happens."""

    previous_start: int | None = None
    previous_end: int | None = None


def _validate_next_window(window: SamplingWindow, state: _WindowValidationState) -> None:
    """Reject overlapping or non-monotonic sampling windows incrementally.

    A forward-only walk cannot serve a window it has already passed, and
    buffering the overlap would reintroduce the memory this sensor exists to
    avoid. ``McapCameraSensor`` refuses overlap for the same reason. Checked as
    windows arrive rather than up front, because the grid is iterated lazily.
    """
    if state.previous_start is not None and window.start_ns < state.previous_start:
        msg = "SamplingSpec.grid windows must be monotonically increasing"
        raise ValueError(msg)
    if state.previous_end is not None and window.start_ns < state.previous_end:
        msg = "StreamingCameraSensor requires non-overlapping sampling windows"
        raise ValueError(msg)
    state.previous_start = int(window.start_ns)
    state.previous_end = int(window.exclusive_end_ns)


def _indexed_timeline(index: Mp4FrameIndex, alignment_field: VideoFrameAlignmentField) -> npt.NDArray[np.int64]:
    """Select an index timeline, using presentation order for frame alignment."""
    timestamp_field = alignment_field.value
    timestamps_ns: TimestampArray | None = getattr(index, timestamp_field)
    if timestamps_ns is None:
        msg = f"frame index has no {timestamp_field} timestamps"
        raise IndexRecordUnusableError(msg)
    if isinstance(timestamps_ns, np.ma.MaskedArray) and np.any(timestamps_ns.mask):
        msg = f"frame index {timestamp_field} has missing timestamps"
        raise IndexRecordUnusableError(msg)
    timeline_ns = np.asarray(timestamps_ns)
    return timeline_ns if alignment_field is VideoFrameAlignmentField.DTS_NS else timeline_ns[index.pts_sort_order]


class StreamingCameraSensor:
    """A forward-only camera sensor backed by an MP4 frame index.

    Metadata and the frame index may be supplied together or built lazily from the
    source. The selected alignment field supplies the frame timestamps.
    Frames are materialized on demand, retaining one selected RGB frame
    independently of window length. Codec buffers, timestamps, indexes, and
    caller-retained outputs contribute additional memory.

    Each reference timestamp in ``spec.grid`` selects its nearest source frame,
    even outside the window's bounds. With the same source timeline, reference
    grid, and policy, changing window boundaries preserves selected timestamps
    and pixels. Windows must not overlap, and their payloads must be consumed in
    order through the shared decode pass; several windows can supply one logical clip.
    """

    def __init__(  # noqa: PLR0913 -- source, indexing, alignment, and precomputed metadata are independent options
        self,
        source: DataSource | None = None,
        stream_idx: int = 0,
        decode_config: CpuVideoDecodeConfig = DEFAULT_VIDEO_DECODE_CONFIG,
        *,
        metadata: Mp4Metadata | None = None,
        frame_index: Mp4FrameIndex | None = None,
        alignment_field: VideoFrameAlignmentField = VideoFrameAlignmentField.PTS_NS,
        indexer: Mp4Indexer = Mp4Indexer.PYAV,
        moov_bytes: bytes | None = None,
    ) -> None:
        """Initialize the streaming camera sensor.

        Args:
            source: Video source, or None for a supplied index used without decoding.
            stream_idx: Video stream position used for indexing and decoding.
            decode_config: Frame decoder configuration.
            metadata: MP4 metadata. Must be supplied together with frame_index.
            frame_index: MP4 frame index. Must be supplied together with metadata.
            alignment_field: Frame-index timestamp field used for alignment.
            indexer: Backend used when metadata and the frame index are not supplied.
            moov_bytes: The source's ``moov`` box, header included, which the supplied
                metadata and frame index describe. Decoding then reads only packet bytes
                from the source; otherwise it also reads the ``moov`` from the source. An
                index built from the source keeps the ``moov`` it read.

        Raises:
            ValueError: Only one of metadata and frame_index is supplied, no source
                is available to build them, or moov_bytes is supplied without them.
            TypeError: The alignment field or indexer is not an enum member.
            IndexRecordUnusableError: The supplied index is empty or disagrees with metadata.

        """
        if (metadata is None) != (frame_index is None):
            msg = "metadata and frame_index must be supplied together"
            raise ValueError(msg)
        if source is None and frame_index is None:
            msg = "source is required when metadata and frame_index are not supplied"
            raise ValueError(msg)
        if moov_bytes is not None and frame_index is None:
            msg = "moov_bytes requires the metadata and frame_index it was indexed with"
            raise ValueError(msg)
        alignment_value: object = alignment_field
        indexer_value: object = indexer
        if not isinstance(alignment_value, VideoFrameAlignmentField):
            msg = "alignment_field must be a VideoFrameAlignmentField"
            raise TypeError(msg)
        if not isinstance(indexer_value, Mp4Indexer):
            msg = "indexer must be an Mp4Indexer"
            raise TypeError(msg)
        if frame_index is not None and metadata is not None:
            if len(frame_index) == 0:
                msg = "frame index contains no frames"
                raise IndexRecordUnusableError(msg)
            if len(frame_index) != metadata.frame_count:
                msg = "frame index count disagrees with metadata.frame_count"
                raise IndexRecordUnusableError(msg)
        self._source = source
        self._stream_idx = stream_idx
        self._decode_config = decode_config
        self._metadata = metadata
        self._frame_index = frame_index
        self._alignment_field = alignment_field
        self._indexer = indexer
        self._moov_bytes = moov_bytes

    def _ensure_index(self) -> tuple[Mp4Metadata, Mp4FrameIndex]:
        """Return metadata and the frame index, building both if needed."""
        metadata, frame_index = self._metadata, self._frame_index
        if metadata is None or frame_index is None:
            source = self._source
            if source is None:
                msg = "source is required to build metadata and the frame index"
                raise ValueError(msg)
            video = index_video(source, indexer=self._indexer, stream_idx=self._stream_idx)
            metadata, frame_index = video.metadata, video.frame_index
            self._metadata, self._frame_index, self._moov_bytes = metadata, frame_index, video.moov_bytes
        return metadata, frame_index

    @property
    def frame_index(self) -> Mp4FrameIndex:
        """Return the frame index, building it from the source if needed."""
        _, frame_index = self._ensure_index()
        return frame_index

    @property
    def metadata(self) -> Mp4Metadata:
        """Return the MP4 metadata, building it from the source if needed."""
        metadata, _ = self._ensure_index()
        return metadata

    @property
    def alignment_field(self) -> VideoFrameAlignmentField:
        """Return the timestamp field selected for alignment."""
        return self._alignment_field

    @property
    def codec_name(self) -> str:
        """Video codec name as the container's own boxes spell it."""
        return self.metadata.codec_name

    @property
    def has_bframes(self) -> bool:
        """Whether the container carries composition offsets, i.e. signals frame reordering."""
        # An all-null column means the ctts box is absent; an all-zero one means it is
        # present and says there is no reordering.
        offsets = self.frame_index.composition_offset_ticks
        return offsets is not None and bool(np.any(offsets != 0))

    @property
    def avg_frame_rate(self) -> Fraction:
        """The container's nominal frame rate, or ``Fraction(0)`` when it declares none."""
        return _avg_frame_rate(self.metadata)

    @property
    def timestamp_source(self) -> TimestampSource:
        """Which clock :attr:`timestamps_ns` is on."""
        return _INDEX_TIMESTAMP_SOURCES[self._alignment_field]

    @property
    def timestamps_ns(self) -> npt.NDArray[np.int64]:
        """Return the selected frame-index timestamps in int64 nanoseconds.

        Raises:
            IndexRecordUnusableError: The selected field is absent or partially missing.

        """
        return _validate_timestamp_array(_indexed_timeline(self.frame_index, self._alignment_field))

    @property
    def start_ns(self) -> int:
        """Return the first timeline timestamp, in nanoseconds."""
        return int(self.timestamps_ns[0])

    @property
    def end_ns(self) -> int:
        """Return the last timeline timestamp, in nanoseconds."""
        return int(self.timestamps_ns[-1])

    def supports_sampling_policy(self, policy: object) -> bool:
        """Return whether this sensor can sample with *policy*."""
        return isinstance(policy, NearestTimestampPolicy)

    def stream_timestamps(self, batch_size: int = 0) -> Iterator[npt.NDArray[np.int64]]:
        """Yield timestamps in timeline order, without resampling.

        Args:
            batch_size: ``0`` yields every timestamp in one batch; ``> 0`` yields
                consecutive batches of that size.

        Yields:
            Consecutive, non-overlapping int64 nanosecond arrays whose union is
            the whole timeline.

        Raises:
            ValueError: If *batch_size* is negative.

        """
        if batch_size < 0:
            msg = f"batch_size must be non-negative, got {batch_size}"
            raise ValueError(msg)
        timestamps_ns = self.timestamps_ns
        if batch_size == 0:
            yield timestamps_ns
            return
        for start in range(0, len(timestamps_ns), batch_size):
            yield timestamps_ns[start : start + batch_size]

    def sample(self, spec: SamplingSpec, *, policy: NearestTimestampPolicy) -> Generator[StreamingCameraData]:
        """Walk the video once, yielding one payload per window of ``spec.grid``.

        One payload per window including empty ones, because ``SensorGroup``
        advances every sensor in lockstep and a skipped window would hand the
        next window's frames to the wrong one.

        Every payload draws on one shared decode, so they must be consumed in the
        order they were yielded. Skipping a payload is fine; going back to one
        raises, because the walk cannot rewind to serve it.

        Sampling can start partway through a recording without decoding from the
        beginning. Sparse sampling may still require decoding intermediate frames,
        so a lower output frame rate does not guarantee proportionally less
        decoding work.

        Decoding is where the timeline has to ascend, so that is where the order
        is checked. A disordered timeline is a bad argument to a forward-only
        walk, but to a caller that only measures timestamps it is the defect
        worth reporting, and rejecting one at construction would deny that caller
        the measurement. ``CameraSensor`` still refuses one in its constructor,
        so the two disagree on purpose.

        Args:
            spec: The sampling grid to serve.
            policy: Its ``max_delta_ns`` bounds how far a reference may sit from
                the frame chosen for it.

        Returns:
            One :class:`StreamingCameraData` per window, in grid order.

        Raises:
            RuntimeError: If the sensor was built without a source, or if a
                payload is consumed after a later one has been.
            ValueError: If DTS alignment would require changing frame order, if
                ``timestamps_ns`` does not strictly ascend, if the
                stream has no displayable frames, if the grid's windows overlap
                or run backwards, if the recording's origin shifts its
                timestamps outside signed int64, or if a served frame disagrees
                with the timeline the payload published.

        """
        source = self._source
        if source is None:
            msg = "this sensor was built from an index and has no media to decode"
            raise RuntimeError(msg)
        timestamps_ns = self.timestamps_ns
        require_strictly_increasing("timestamps_ns", timestamps_ns)
        if self._alignment_field is VideoFrameAlignmentField.DTS_NS:
            presentation_order = self.frame_index.pts_sort_order
            if np.any(presentation_order[1:] < presentation_order[:-1]):
                msg = "DTS_NS sampling requires matching decode and presentation order; select PTS_NS or another field"
                raise ValueError(msg)
        return self._walk(source, spec, timestamps_ns, policy=policy)

    def _walk(
        self,
        source: DataSource,
        spec: SamplingSpec,
        timeline_ns: npt.NDArray[np.int64],
        *,
        policy: NearestTimestampPolicy,
    ) -> Generator[StreamingCameraData]:
        """Serve ``spec.grid`` off one traversal of *source*.

        Split from :meth:`sample` so that method's checks run when it is called
        rather than when its first payload is pulled.
        """
        metadata, frame_index = self._ensure_index()
        decoder = StreamingVideoDecoder(source, metadata, frame_index, self._decode_config, moov_bytes=self._moov_bytes)
        yield from _stream_indexed_camera_windows(decoder, timeline_ns, spec, policy=policy)


@attrs.define
class _IndexedWindowState:
    """Window ordering and lifetime, independent of the codec's target position."""

    decoder: StreamingVideoDecoder
    claimed_sequence: int = -1
    last_sequence: int = -1
    completed_sequence: int = -1
    metadata_done: bool = False


def _stream_indexed_camera_windows(
    decoder: StreamingVideoDecoder,
    timeline_ns: npt.NDArray[np.int64],
    spec: SamplingSpec,
    *,
    policy: NearestTimestampPolicy,
) -> Generator[StreamingCameraData]:
    """Publish index-selected windows without opening or advancing the decoder."""
    state = _IndexedWindowState(decoder)
    validation = _WindowValidationState()
    try:
        for sequence, window in enumerate(spec.grid):
            _validate_next_window(window, validation)
            positions = (
                find_closest_indices(timeline_ns, window.timestamps_ns)
                if len(window.timestamps_ns)
                else np.empty(0, dtype=np.int64)
            )
            plan = make_streaming_decode_plan(decoder.frame_index, positions)
            source_ns = timeline_ns[positions]
            state.last_sequence = sequence
            yield StreamingCameraData(
                align_timestamps_ns=window.timestamps_ns,
                sensor_timestamps_ns=source_ns,
                walk=_serve_indexed_window(state, plan, window, source_ns, sequence, policy.max_delta_ns),
            )
    except Exception:
        decoder.close()
        raise
    finally:
        # SensorGroup closes this iterator while retained payloads can still need it.
        state.metadata_done = True
        if state.completed_sequence == state.last_sequence:
            decoder.close()


def _serve_indexed_window(  # noqa: PLR0913 -- one window's plan, timestamps, ownership, and tolerance
    state: _IndexedWindowState,
    plan: StreamingDecodePlan,
    window: SamplingWindow,
    source_ns: npt.NDArray[np.int64],
    sequence: int,
    max_delta_ns: int | None,
) -> Generator[StreamingFrame]:
    """Execute a window only when consumed; abandoning it permits later windows."""
    frames = execute_streaming_decode_plan(state.decoder, plan)
    try:
        for reference, timestamp in zip(window.timestamps_ns, source_ns, strict=True):
            if sequence < state.claimed_sequence:
                msg = "StreamingCameraData payloads must be consumed in the order they were yielded"
                raise RuntimeError(msg)  # noqa: TRY301 -- errors must close the shared pass
            state.claimed_sequence = sequence
            reference_ns, timestamp_ns = int(reference), int(timestamp)
            delta_ns = abs(timestamp_ns - reference_ns)
            if max_delta_ns is not None and delta_ns > max_delta_ns:
                msg = (
                    f"max_delta_ns={max_delta_ns} exceeded: "
                    f"delta was {delta_ns} ns for reference={reference_ns}, timestamp={timestamp_ns}"
                )
                raise ValueError(msg)  # noqa: TRY301 -- errors must close the shared pass
            _position, pixels = next(frames)
            yield reference_ns, timestamp_ns, pixels
        state.completed_sequence = sequence
        if state.metadata_done and sequence == state.last_sequence:
            state.decoder.close()
    except Exception:
        state.decoder.close()
        raise
    finally:
        frames.close()


def stream_camera_windows(
    frames: Iterator[tuple[int, npt.NDArray[np.uint8]]],
    timeline_ns: npt.NDArray[np.int64],
    spec: SamplingSpec,
    *,
    policy: NearestTimestampPolicy,
) -> Generator[StreamingCameraData]:
    """Align a decoded frame iterator to reference windows using its supplied timeline."""
    _validate_timestamp_array(timeline_ns)
    require_strictly_increasing("timestamps_ns", timeline_ns)
    cursor = _Cursor(frames, timeline_ns=timeline_ns)
    state = _WindowValidationState()
    sequence = itertools.count()
    for window in spec.grid:
        _validate_next_window(window, state)
        expected_ns = _serving_timestamps(timeline_ns, window.timestamps_ns)
        yield StreamingCameraData(
            align_timestamps_ns=window.timestamps_ns,
            sensor_timestamps_ns=expected_ns,
            walk=cursor.serve(window, policy.max_delta_ns, next(sequence)),
        )


@attrs.define(slots=True)
class _Cursor:
    """The shared decode position, advanced by whichever window is being served."""

    _frames: Iterator[tuple[int, npt.NDArray[np.uint8]]] = attrs.field(alias="frames")
    _timeline_ns: npt.NDArray[np.int64] = attrs.field(alias="timeline_ns")
    _current: tuple[int, npt.NDArray[np.uint8]] | None = attrs.field(init=False, default=None)
    _upcoming: tuple[int, npt.NDArray[np.uint8]] | None = attrs.field(init=False, default=None)
    _started: bool = attrs.field(init=False, default=False)
    _index: int = attrs.field(init=False, default=0)
    _drawn_from: int = attrs.field(init=False, default=-1)

    def serve(
        self,
        window: SamplingWindow,
        max_delta_ns: int | None,
        sequence: int,
    ) -> Generator[StreamingFrame]:
        """Yield one row per reference timestamp in *window*."""
        for reference_ns in window.timestamps_ns:
            self._claim(sequence)
            self._start()
            if self._current is None:
                msg = "video stream contains no displayable frames"
                raise ValueError(msg)
            reference = int(reference_ns)
            while self._upcoming is not None and abs(self._upcoming[0] - reference) < abs(self._current[0] - reference):
                self._current = self._upcoming
                self._upcoming = self._advance()

            timestamp_ns, frame = self._current
            delta_ns = abs(timestamp_ns - reference)
            if max_delta_ns is not None and delta_ns > max_delta_ns:
                msg = (
                    f"max_delta_ns={max_delta_ns} exceeded: "
                    f"delta was {delta_ns} ns for reference={reference}, timestamp={timestamp_ns}"
                )
                raise ValueError(msg)
            yield reference, timestamp_ns, frame

    def _claim(self, sequence: int) -> None:
        """Refuse a payload that would draw from behind the walk's position.

        Every payload draws from one decode position that cannot rewind, so a
        window can only be served while nothing later has been. Skipping ahead is
        fine -- the walk reaches later references by advancing -- but going back
        would pick real frames for the wrong timestamps, which a tolerance check
        catches only by luck. Checked per row, so interleaving is caught too.
        """
        if sequence < self._drawn_from:
            msg = (
                "StreamingCameraData payloads must be consumed in the order they were yielded; "
                f"window {sequence} was asked for after window {self._drawn_from}, and one forward "
                "walk cannot go back"
            )
            raise RuntimeError(msg)
        self._drawn_from = sequence

    def _start(self) -> None:
        """Prime the first two frames on first use, so decode is lazy."""
        if self._started:
            return
        self._started = True
        first = next(self._frames, None)
        if first is None:
            return
        self._current = self._observed(first)
        self._upcoming = self._advance()

    def _advance(self) -> tuple[int, npt.NDArray[np.uint8]] | None:
        """Take the next frame, timestamped by the timeline rather than the container.

        A container ending before the timeline does is indistinguishable from a
        clean end unless checked here: left alone, later references would reuse
        the final decoded frame while still publishing the timeline's timestamps
        for observations no frame ever backed.
        """
        row = next(self._frames, None)
        if row is None:
            if self._index != len(self._timeline_ns):
                msg = (
                    f"container ended after {self._index} frames but timestamps_ns describes "
                    f"{len(self._timeline_ns)} observations"
                )
                raise ValueError(msg)
            return None
        return self._observed(row)

    def _observed(self, row: tuple[int, npt.NDArray[np.uint8]]) -> tuple[int, npt.NDArray[np.uint8]]:
        """Timestamp one frame from the timeline.

        Each frame is paired with the next selected timestamp. Nothing compares
        their times: a container written at a declared rate spaces its frames
        evenly and cannot express a camera pausing, so a timeline that records
        one disagrees with it by a frame period from then on, while still
        describing the same frames in the same order.

        How many observations there are is a different matter, and is checked --
        here for a recording that outruns its timeline, and at the end of the
        walk for a timeline that outruns its recording.
        """
        _pts_ns, frame = row
        index = self._index
        self._index += 1
        if index >= len(self._timeline_ns):
            msg = (
                f"timestamps_ns describes {len(self._timeline_ns)} observations "
                "but the recording holds more frames than that"
            )
            raise ValueError(msg)
        return int(self._timeline_ns[index]), frame
