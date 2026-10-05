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
"""MCAP-backed LiDAR sensor for decode-late (schemaless) LiDAR channels.

MCAP LiDAR channels carry undecoded vendor packet bytes verbatim (``schema_id=0``,
a vendor-specific ``message_encoding``); no protobuf schema describes the
payload. Decoding those bytes into points is vendor-specific, and some
vendor decoders are proprietary. ``core.sensors`` stays license-agnostic, so
this module never imports a concrete decoder. Instead it depends on the
small :class:`PacketDecoder` protocol below; callers hand in whatever
vendor-specific decoder instance they have that satisfies it structurally.
"""

from collections.abc import Generator, Iterable, Iterator
from typing import Final, Protocol

import attrs
import numpy as np
import numpy.typing as npt
from mcap.reader import McapReader

from cosmos_curator.core.sensors.data.extrinsics import SensorExtrinsics
from cosmos_curator.core.sensors.data.lidar_data import LidarData, LidarMetadata
from cosmos_curator.core.sensors.sampling.grid import SamplingWindow
from cosmos_curator.core.sensors.sampling.policy import NoSamplingPolicy, require_no_sampling_policy
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.types.types import DataSource
from cosmos_curator.core.sensors.utils.mcap import McapTopicAccessor, channel_metadata_for_topic

DEFAULT_TOPIC = "/lidar"
DEFAULT_REFERENCE_FRAME = "sensor"

# points_align_index is uint16 (LidarData's own field type); a row index at or
# past this many rows would silently wrap and alias its points onto an earlier
# row instead of erroring.
_MAX_ALIGN_ROWS: Final[int] = np.iinfo(np.uint16).max + 1


class DecodedSweep(Protocol):
    """Structural shape of one batch of decoded LiDAR points sharing one sweep.

    A ``PacketDecoder`` is expected to emit whole sweeps -- for a decoder
    whose azimuth-wraparound detection has no way to flush a partial
    rotation, each sweep is a genuine full 360-degree turn. This protocol
    itself does not require completeness, though: it is also satisfied
    structurally by this module's own internal :class:`_SweepPointsView`,
    which holds only the subset of one sweep's points that fell inside a
    sampling window and is not a full rotation.

    Members are declared as read-only properties, not plain attributes, so
    any frozen (immutable) vendor-decoder sweep type -- as well as
    :class:`_SweepPointsView` -- can satisfy this protocol structurally,
    without importing it.
    """

    @property
    def points_xyz(self) -> npt.NDArray[np.float32]:
        """Per-point ``(x, y, z)`` coordinates, shape ``(P, 3)``."""
        ...

    @property
    def intensity(self) -> npt.NDArray[np.float32]:
        """Per-point return intensity, shape ``(P,)``."""
        ...

    @property
    def ring(self) -> npt.NDArray[np.uint16]:
        """Per-point channel/ring index, shape ``(P,)``."""
        ...

    @property
    def sensor_timestamps_ns(self) -> npt.NDArray[np.int64]:
        """Per-point sensor-clock timestamps in nanoseconds, shape ``(P,)``."""
        ...

    @property
    def sweep_timestamp_ns(self) -> int:
        """Sensor-clock reference instant for this batch of points, in nanoseconds."""
        ...


class PacketDecoder(Protocol):
    """Structural shape of a vendor packet decoder's ``parse`` method, without importing it.

    The decoder is expected to be stateful across calls (sweep boundaries
    are detected from the packet stream as a whole), so ``LidarSensor``
    feeds it every packet on the topic in one pass and does not construct
    more than one generator from it.
    """

    def parse(self, packets: Iterable[tuple[bytes, int]]) -> Iterator[DecodedSweep]:
        """Decode vendor ``(payload_bytes, arrival_timestamp_ns)`` packets into sweeps.

        ``arrival_timestamp_ns`` is the recorder/host clock (MCAP's own
        ``log_time``), not necessarily whatever device clock the decoder
        extracts from ``payload_bytes`` itself for
        ``DecodedSweep.sensor_timestamps_ns``.
        """
        ...


@attrs.frozen
class _SweepPointsView:
    """A subset of one sweep's points, filtered to those inside one sampling window.

    ``sweep_timestamp_ns`` here is the *earliest surviving point's* own
    timestamp within the window, not the original sweep's overall
    ``sweep_timestamp_ns`` -- a sweep whose own timestamp range spans
    multiple windows contributes a different, valid-for-that-window instant
    to each one. Satisfies :class:`DecodedSweep` structurally.
    """

    points_xyz: npt.NDArray[np.float32]
    intensity: npt.NDArray[np.float32]
    ring: npt.NDArray[np.uint16]
    sensor_timestamps_ns: npt.NDArray[np.int64]
    sweep_timestamp_ns: int


def _sweep_bound_ns(sweep: DecodedSweep, *, use_max: bool) -> int:
    """Return the min or max of *sweep*'s own per-point timestamps, or its sweep timestamp if empty."""
    if not sweep.sensor_timestamps_ns.size:
        return sweep.sweep_timestamp_ns
    return int(sweep.sensor_timestamps_ns.max()) if use_max else int(sweep.sensor_timestamps_ns.min())


def _sweep_points_in_window(sweep: DecodedSweep, window: SamplingWindow) -> _SweepPointsView | None:
    """Return the subset of *sweep*'s points whose own timestamp falls in *window*, or None."""
    mask = (sweep.sensor_timestamps_ns >= window.start_ns) & (sweep.sensor_timestamps_ns < window.exclusive_end_ns)
    if not mask.any():
        return None
    filtered_timestamps_ns = sweep.sensor_timestamps_ns[mask]
    return _SweepPointsView(
        points_xyz=sweep.points_xyz[mask],
        intensity=sweep.intensity[mask],
        ring=sweep.ring[mask],
        sensor_timestamps_ns=filtered_timestamps_ns,
        sweep_timestamp_ns=int(filtered_timestamps_ns.min()),
    )


def _lidar_data_from_sweeps(sweeps: list[DecodedSweep], metadata: LidarMetadata) -> LidarData:
    """Build one ``LidarData`` batch from *sweeps*, one row per element.

    Each element of *sweeps* becomes one row; it may be a whole sweep (from
    ``read_all()``/``stream()``) or a window-filtered fragment of one sweep's
    points (from ``sample()``).

    Raises:
        ValueError: if *sweeps* has more rows than ``points_align_index``'s
            ``uint16`` dtype can represent (65536), which would otherwise
            silently wrap and alias later points onto an earlier row.

    """
    if len(sweeps) > _MAX_ALIGN_ROWS:
        msg = f"cannot build LidarData from {len(sweeps)} rows: points_align_index is uint16, max {_MAX_ALIGN_ROWS}"
        raise ValueError(msg)
    align_timestamps_ns = np.array([sweep.sweep_timestamp_ns for sweep in sweeps], dtype=np.int64)
    points_align_index = np.concatenate(
        [np.full(len(sweep.points_xyz), i, dtype=np.uint16) for i, sweep in enumerate(sweeps)],
    )
    points_xyz = np.concatenate([sweep.points_xyz for sweep in sweeps], axis=0)
    points_timestamps_ns = np.concatenate([sweep.sensor_timestamps_ns for sweep in sweeps])
    points_intensity = np.concatenate([sweep.intensity for sweep in sweeps])
    points_ring = np.concatenate([sweep.ring for sweep in sweeps])

    # A decoder is not required to emit one sweep's points in ascending firing-time
    # order -- some vendor decoders group points by channel/block/return
    # type, not by timestamp -- but LidarData requires points_timestamps_ns
    # nondecreasing across the whole batch. Stable-sort every per-point array
    # together by timestamp. points_align_index is assigned per-point before
    # the sort and permuted by this same `order`, so each point keeps its true
    # originating-sweep index regardless of how the sort reorders points.
    order = np.argsort(points_timestamps_ns, kind="stable")

    return LidarData(
        align_timestamps_ns=align_timestamps_ns,
        sensor_timestamps_ns=align_timestamps_ns,
        points_xyz=points_xyz[order],
        points_timestamps_ns=points_timestamps_ns[order],
        metadata=metadata,
        points_intensity=points_intensity[order],
        points_ring=points_ring[order],
        points_align_index=points_align_index[order],
    )


def _empty_lidar_data(metadata: LidarMetadata) -> LidarData:
    """Build an empty LiDAR batch with the same optional-array presence as populated batches."""
    empty_ts = np.empty(0, dtype=np.int64)
    return LidarData(
        align_timestamps_ns=empty_ts,
        sensor_timestamps_ns=empty_ts,
        points_xyz=np.empty((0, 3), dtype=np.float32),
        points_timestamps_ns=empty_ts,
        metadata=metadata,
        points_intensity=np.empty(0, dtype=np.float32),
        points_ring=np.empty(0, dtype=np.uint16),
        points_align_index=np.empty(0, dtype=np.uint16),
    )


class LidarSensor:
    """MCAP-backed, decode-late LiDAR sensor.

    Reads undecoded vendor packets from a schemaless MCAP topic and decodes them
    into ``LidarData`` batches using a caller-supplied :class:`PacketDecoder`.
    Each output row (``LidarData`` alignment index) is one sweep, or one
    window's worth of one sweep's points when windowed (see below).
    ``align_timestamps_ns`` and ``sensor_timestamps_ns`` both carry a
    sensor-clock timestamp -- the decoder's own ``sweep_timestamp_ns`` for
    whole-sweep reads (``read_all()``/``stream()``), or the earliest
    in-window point's own timestamp for a windowed row (``sample()``) --
    since ``metadata.motion_compensated=False`` and no independent
    recorder-clock alignment is established at this layer.

    **Clock domains are not reconciled.** ``start_ns``/``end_ns``/
    ``timestamps_ns`` are recorder/host clock (MCAP's own ``log_time``);
    ``sensor_timestamps_ns``, ``align_timestamps_ns``, and everything
    ``sample()`` filters points by are device clock, straight from the
    decoder. Whether -- and by how much -- those two clocks differ for a
    given recording is an open question tracked separately, and this sensor
    does not attempt a conversion. A caller who builds a ``sample()`` window
    grid from this sensor's own ``start_ns``/``end_ns`` is therefore
    comparing across an unknown, unreconciled clock offset: at best every
    window silently comes back empty, at worst points land in the wrong
    window without any error. Treat every timestamp this class exposes as
    sensor-clock or recorder-clock as documented per field, never as
    wall-clock, and do not mix the two domains in one comparison.

    Decoding a vendor packet stream is inherently stateful: sweep boundaries
    are detected across the whole stream, and a decoder generally cannot be
    restarted mid-stream without dropping points. This sensor therefore
    decodes the full topic once per ``LidarSensor`` instance (on first
    ``sample()``/``stream()``/``read_all()`` call) and caches the resulting
    sweeps, rather than decoding per sampling window. The trailing partial
    sweep at end-of-stream may be dropped by the decoder; that is a decoder
    contract, not something this sensor works around.

    Sampling windows filter by each *point's own* ``sensor_timestamps_ns``,
    not by a whole sweep's single reference timestamp. A sweep's real-world
    rotation can span one window, two adjacent windows, or -- if the grid is
    finer than the sweep's own duration -- arbitrarily many; it contributes a
    partial row (only its in-window points) to every window its own
    timestamp range overlaps, rather than being assigned atomically to one
    window. ``read_all()`` and ``stream()`` are unwindowed and always return
    whole, untruncated sweeps.
    """

    def __init__(  # noqa: PLR0913
        self,
        source: DataSource,
        topic: str = DEFAULT_TOPIC,
        *,
        packet_decoder: PacketDecoder,
        reference_frame: str = DEFAULT_REFERENCE_FRAME,
        extrinsics: SensorExtrinsics | None = None,
        sensor_model: str | None = None,
    ) -> None:
        """Initialize the MCAP LiDAR sensor."""
        self._topic = topic
        self._mcap = McapTopicAccessor(source, topic)
        self._packet_decoder = packet_decoder
        self._metadata = LidarMetadata(
            motion_compensated=False,
            reference_frame=reference_frame,
            extrinsics=extrinsics,
            sensor_model=sensor_model,
        )
        self._sweeps: list[DecodedSweep] | None = None
        self._sweep_starts_ns: npt.NDArray[np.int64] | None = None
        self._sweep_ends_ns: npt.NDArray[np.int64] | None = None
        self._empty_lidar_data: LidarData | None = None

    @property
    def start_ns(self) -> int:
        """Earliest LiDAR packet time on this topic, in nanoseconds.

        This is a recorder/host-clock timestamp (MCAP's own ``log_time``), a
        different clock domain than the device-clock ``sensor_timestamps_ns``
        that ``sample()`` filters points against. Do not build a ``sample()``
        window grid from ``start_ns``/``end_ns`` without first reconciling
        the two clocks; an unreconciled offset can silently shift or empty
        out every window.
        """
        return self._mcap.start_ns

    @property
    def end_ns(self) -> int:
        """Latest LiDAR packet time on this topic, in nanoseconds.

        Recorder/host-clock timestamp, like ``start_ns`` -- do not mix with
        device-clock ``sensor_timestamps_ns`` in one comparison, and do not
        build a ``sample()`` window grid from it without first reconciling
        the two clocks.
        """
        return self._mcap.end_ns

    @property
    def max_gap_ns(self) -> int:
        """Return maximum expected gap duration in nanoseconds."""
        return self._mcap.max_gap_ns

    @property
    def timestamps_ns(self) -> npt.NDArray[np.int64]:
        """MCAP's own packet ``log_time`` values in nanoseconds (not sweep timestamps)."""
        return self._mcap.timestamps_ns

    def _get_empty_lidar_data(self) -> LidarData:
        """Return a cached empty LiDAR batch."""
        if self._empty_lidar_data is None:
            self._empty_lidar_data = _empty_lidar_data(self._metadata)
        return self._empty_lidar_data

    def _decode_all_sweeps(
        self,
        reader: McapReader,
    ) -> tuple[list[DecodedSweep], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Decode every packet on the topic into sweeps, in one pass, and cache the result.

        Also returns each sweep's own ``[min, max]`` per-point timestamp
        bounds (derived from its points, not from the decoder-reported
        ``sweep_timestamp_ns``), for fast window-overlap lookups in
        :meth:`_read_window_sweeps`.

        Raises:
            ValueError: if the decoder yields sweeps whose per-point
                timestamp ranges are not strictly disjoint
                (``sweep_starts_ns[i] > sweep_ends_ns[i-1]`` for every
                consecutive pair), or whose
                ``sweep_timestamp_ns`` is not strictly increasing.
                ``_read_window_sweeps`` uses ``np.searchsorted`` over the
                per-point bounds, which silently returns wrong (not merely
                incomplete) results on unsorted input; strict disjointness
                also guarantees that any sweeps landing in the same
                ``sample()`` window keep their relative order when their
                in-window points are filtered, so ``_read_window_sweeps``
                never needs to re-sort ``window_views`` by timestamp.
                ``sweep_timestamp_ns`` becomes ``align_timestamps_ns`` in
                ``read_all()``/``_lidar_data_from_sweeps``, which requires
                strict increase -- but ``stream()`` yields one sweep at a
                time as its own single-row batch, where that per-batch check
                alone cannot see a repeated or decreasing value across
                yields. The ``PacketDecoder`` protocol does not itself
                require ordered or disjoint output, so all of this is
                checked rather than assumed.

        """
        if self._sweeps is not None and self._sweep_starts_ns is not None and self._sweep_ends_ns is not None:
            return self._sweeps, self._sweep_starts_ns, self._sweep_ends_ns

        packets = (
            (bytes(message.data), int(message.log_time))
            for _schema, _channel, message in self._mcap.iter_messages(
                reader,
                self.start_ns,
                self.end_ns + 1,
                log_time_order=True,
            )
        )
        sweeps = list(self._packet_decoder.parse(packets))
        sweep_starts_ns = np.array([_sweep_bound_ns(sweep, use_max=False) for sweep in sweeps], dtype=np.int64)
        sweep_ends_ns = np.array([_sweep_bound_ns(sweep, use_max=True) for sweep in sweeps], dtype=np.int64)
        sweep_timestamps_ns = np.array([sweep.sweep_timestamp_ns for sweep in sweeps], dtype=np.int64)
        if sweep_starts_ns.size and (
            np.any(sweep_starts_ns[1:] <= sweep_ends_ns[:-1])
            or np.any(sweep_timestamps_ns[1:] <= sweep_timestamps_ns[:-1])
        ):
            msg = (
                f"{type(self._packet_decoder).__name__}.parse must yield sweeps with "
                "strictly disjoint (non-overlapping, non-touching) point-timestamp ranges "
                "and strictly increasing sweep timestamps"
            )
            raise ValueError(msg)
        self._sweeps = sweeps
        self._sweep_starts_ns = sweep_starts_ns
        self._sweep_ends_ns = sweep_ends_ns
        return sweeps, sweep_starts_ns, sweep_ends_ns

    def _read_window_sweeps(self, reader: McapReader, window: SamplingWindow) -> LidarData:
        """Build one ``LidarData`` batch from the in-window points of every overlapping sweep."""
        sweeps, sweep_starts_ns, sweep_ends_ns = self._decode_all_sweeps(reader)

        # Candidate sweeps are those whose own [start, end] timestamp range can
        # possibly overlap the half-open window; exact inclusion is then decided
        # per point in _sweep_points_in_window.
        lo = int(np.searchsorted(sweep_ends_ns, window.start_ns, side="left"))
        hi = int(np.searchsorted(sweep_starts_ns, window.exclusive_end_ns, side="left"))

        window_views: list[DecodedSweep] = [
            view for sweep in sweeps[lo:hi] if (view := _sweep_points_in_window(sweep, window)) is not None
        ]
        if not window_views:
            return self._get_empty_lidar_data()
        return _lidar_data_from_sweeps(window_views, self._metadata)

    def supports_sampling_policy(self, policy: object) -> bool:
        """Return whether this sensor can sample with *policy*."""
        return isinstance(policy, NoSamplingPolicy)

    def read_all(self) -> LidarData:
        """Decode every sweep on the configured LiDAR topic into one batch."""
        with self._mcap.open_reader() as reader:
            sweeps, _sweep_starts_ns, _sweep_ends_ns = self._decode_all_sweeps(reader)
            if not sweeps:
                return self._get_empty_lidar_data()
            return _lidar_data_from_sweeps(sweeps, self._metadata)

    def sample(self, spec: SamplingSpec, *, policy: NoSamplingPolicy) -> Generator[LidarData]:
        """Yield LiDAR point batches for each sampling window.

        Each yielded batch contains, for every sweep overlapping the current
        half-open sampling window, only the points whose own
        ``sensor_timestamps_ns`` falls inside that window. A sweep whose own
        timestamp range spans multiple windows contributes a partial row to
        each one it overlaps -- not just the two windows on either side of a
        single boundary, but however many windows that range actually spans.

        ``spec``'s windows are compared directly against device-clock point
        timestamps with no conversion. Build ``spec`` in that same
        device-clock domain, not from this sensor's own (recorder-clock)
        ``start_ns``/``end_ns``.
        """
        require_no_sampling_policy(policy, sensor_name=type(self).__name__)
        with self._mcap.open_reader() as reader:
            for window in spec.grid:
                yield self._read_window_sweeps(reader, window)

    def stream(self) -> Generator[LidarData]:
        """Yield every decoded sweep as its own single-row ``LidarData`` batch, in time order."""
        with self._mcap.open_reader() as reader:
            sweeps, _sweep_starts_ns, _sweep_ends_ns = self._decode_all_sweeps(reader)
            for sweep in sweeps:
                yield _lidar_data_from_sweeps([sweep], self._metadata)


def calibration_csv_for_topic(reader: McapReader, topic: str) -> str:
    """Return the ``calibration_csv`` channel-metadata entry that a calibration-aware writer attaches to *topic*.

    Raises:
        ValueError: if the topic's channel has no ``calibration_csv`` entry.

    """
    metadata = channel_metadata_for_topic(reader, topic)
    calibration_csv = metadata.get("calibration_csv")
    if calibration_csv is None:
        msg = f"MCAP channel for topic {topic!r} has no 'calibration_csv' metadata entry"
        raise ValueError(msg)
    return calibration_csv
