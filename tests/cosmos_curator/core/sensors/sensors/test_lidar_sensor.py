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
"""Tests for MCAP-backed ``LidarSensor``.

Uses a synthetic in-test :class:`_FakePacketDecoder` rather than a real
vendor decoder: ``LidarSensor`` depends only on the ``PacketDecoder``
protocol, and its own license-agnostic ``core.sensors`` tests should not
need a proprietary, optional-feature dependency to exercise
windowing/mapping logic. Decoder correctness itself is a vendor-specific
decoder's own test responsibility.

Unlike the real decoder (which silently drops a trailing partial sweep at
end-of-stream because it has no ``flush()``), :class:`_FakePacketDecoder`
flushes every accumulated sweep, including the last one -- that quirk is a
decoder-specific contract, not something under test here.
"""

import struct
from collections.abc import Iterable, Iterator
from pathlib import Path

import attrs
import numpy as np
import numpy.typing as npt
import pytest
from mcap.reader import make_reader as mcap_make_reader

from cosmos_curator.core.sensors.sampling.policy import NearestTimestampPolicy, NoSamplingPolicy
from cosmos_curator.core.sensors.sensors.lidar_sensor import (
    _MAX_ALIGN_ROWS,
    DEFAULT_REFERENCE_FRAME,
    DEFAULT_TOPIC,
    LidarSensor,
    calibration_csv_for_topic,
)
from cosmos_curator.core.sensors.utils import mcap as sensor_mcap
from tests.cosmos_curator.core.sensors.test_utils import one_window_spec

_MESSAGE_ENCODING = "test.lidar.packet"
_PACKET_STRUCT = struct.Struct("<IffffHqq")  # sweep_id, x, y, z, intensity, ring, sensor_ts_ns, sweep_ts_ns


@attrs.frozen
class _FakeSweep:
    """Minimal stand-in satisfying the ``DecodedSweep`` protocol."""

    points_xyz: npt.NDArray[np.float32]
    intensity: npt.NDArray[np.float32]
    ring: npt.NDArray[np.uint16]
    sensor_timestamps_ns: npt.NDArray[np.int64]
    sweep_timestamp_ns: int


class _FakePacketDecoder:
    """Groups ``_PACKET_STRUCT``-encoded test packets into sweeps by ``sweep_id``.

    Assumes packets for a given sweep arrive contiguously (in ``sweep_id``
    order) -- the same assumption a real vendor decoder makes about azimuth
    wraparound on an ordered packet stream.
    """

    def parse(self, packets: Iterable[tuple[bytes, int]]) -> Iterator[_FakeSweep]:
        """Decode test packets into one sweep per contiguous run of ``sweep_id``."""
        current_id: int | None = None
        rows: list[tuple[float, float, float, float, int, int]] = []
        sweep_timestamp_ns = 0
        for payload, _arrival_ns in packets:
            sweep_id, x, y, z, intensity, ring, sensor_ts_ns, sweep_ts_ns = _PACKET_STRUCT.unpack(payload)
            if current_id is not None and sweep_id != current_id:
                yield _rows_to_sweep(rows, sweep_timestamp_ns)
                rows = []
            current_id = sweep_id
            sweep_timestamp_ns = sweep_ts_ns
            rows.append((x, y, z, intensity, ring, sensor_ts_ns))
        if rows:
            yield _rows_to_sweep(rows, sweep_timestamp_ns)


class _FixedSweepsPacketDecoder:
    """Ignores its input packets and always yields a caller-supplied list of sweeps.

    Used to exercise sweep-count/ordering edge cases without needing to write
    and parse an MCAP file with a matching number of real packets.
    """

    def __init__(self, sweeps: list[_FakeSweep]) -> None:
        self._sweeps = sweeps

    def parse(self, packets: Iterable[tuple[bytes, int]]) -> Iterator[_FakeSweep]:
        """Consume *packets* (to reach EOF) and yield the fixed sweep list."""
        for _ in packets:
            pass
        yield from self._sweeps


def _one_point_sweep(sweep_timestamp_ns: int) -> _FakeSweep:
    return _FakeSweep(
        points_xyz=np.zeros((1, 3), dtype=np.float32),
        intensity=np.zeros(1, dtype=np.float32),
        ring=np.zeros(1, dtype=np.uint16),
        sensor_timestamps_ns=np.array([sweep_timestamp_ns], dtype=np.int64),
        sweep_timestamp_ns=sweep_timestamp_ns,
    )


def _empty_sweep(sweep_timestamp_ns: int) -> _FakeSweep:
    return _FakeSweep(
        points_xyz=np.zeros((0, 3), dtype=np.float32),
        intensity=np.zeros(0, dtype=np.float32),
        ring=np.zeros(0, dtype=np.uint16),
        sensor_timestamps_ns=np.array([], dtype=np.int64),
        sweep_timestamp_ns=sweep_timestamp_ns,
    )


def _rows_to_sweep(rows: list[tuple[float, float, float, float, int, int]], sweep_timestamp_ns: int) -> _FakeSweep:
    xyz = np.array([[r[0], r[1], r[2]] for r in rows], dtype=np.float32)
    intensity = np.array([r[3] for r in rows], dtype=np.float32)
    ring = np.array([r[4] for r in rows], dtype=np.uint16)
    sensor_timestamps_ns = np.array([r[5] for r in rows], dtype=np.int64)
    return _FakeSweep(
        points_xyz=xyz,
        intensity=intensity,
        ring=ring,
        sensor_timestamps_ns=sensor_timestamps_ns,
        sweep_timestamp_ns=sweep_timestamp_ns,
    )


def _make_packet(
    *,
    sweep_id: int,
    point: tuple[float, float, float] = (1.0, 2.0, 3.0),
    intensity: float = 10.0,
    ring: int = 0,
    sensor_ts_ns: int,
    sweep_ts_ns: int,
) -> bytes:
    x, y, z = point
    return _PACKET_STRUCT.pack(sweep_id, x, y, z, intensity, ring, sensor_ts_ns, sweep_ts_ns)


def _write_schemaless_lidar_mcap(
    path: Path,
    records: list[sensor_mcap.McapMessageRecord],
    *,
    topic: str = DEFAULT_TOPIC,
    calibration_csv: str | None = "Channel,Elevation,Azimuth\n1,0,0\n",
) -> None:
    channel_metadata = {"calibration_csv": calibration_csv} if calibration_csv is not None else {}
    sensor_mcap.write_mcap_messages(
        path,
        stream=sensor_mcap.McapStreamSpec(
            topic=topic,
            library="cosmos_curator lidar sensor test",
            message_encoding=_MESSAGE_ENCODING,
            channel_metadata=channel_metadata,
        ),
        records=records,
    )


def _record(payload: bytes, log_time_ns: int, sequence: int) -> sensor_mcap.McapMessageRecord:
    return sensor_mcap.McapMessageRecord(
        data=payload,
        log_time_ns=log_time_ns,
        publish_time_ns=log_time_ns,
        sequence=sequence,
    )


def _two_sweep_mcap(tmp_path: Path) -> Path:
    """Write an MCAP with two two-point sweeps at sweep_timestamp_ns 1_000 and 5_000."""
    records = [
        _record(_make_packet(sweep_id=0, sensor_ts_ns=900, sweep_ts_ns=1_000, point=(1.0, 0.0, 0.0)), 900, 0),
        _record(_make_packet(sweep_id=0, sensor_ts_ns=950, sweep_ts_ns=1_000, point=(2.0, 0.0, 0.0)), 950, 1),
        _record(_make_packet(sweep_id=1, sensor_ts_ns=4_900, sweep_ts_ns=5_000, point=(3.0, 0.0, 0.0)), 4_900, 2),
        _record(_make_packet(sweep_id=1, sensor_ts_ns=4_950, sweep_ts_ns=5_000, point=(4.0, 0.0, 0.0)), 4_950, 3),
    ]
    path = tmp_path / "lidar.mcap"
    _write_schemaless_lidar_mcap(path, records)
    return path


def test_lidar_sensor_read_all_concatenates_every_sweep(tmp_path: Path) -> None:
    """read_all() decodes every packet on the topic into one batch, one row per sweep."""
    path = _two_sweep_mcap(tmp_path)
    sensor = LidarSensor(path, packet_decoder=_FakePacketDecoder())

    data = sensor.read_all()

    np.testing.assert_array_equal(data.align_timestamps_ns, [1_000, 5_000])
    np.testing.assert_array_equal(data.sensor_timestamps_ns, [1_000, 5_000])
    assert data.points_xyz.shape == (4, 3)
    np.testing.assert_array_equal(data.points_align_index, [0, 0, 1, 1])
    np.testing.assert_array_equal(data.points_timestamps_ns, [900, 950, 4_900, 4_950])
    assert data.metadata.motion_compensated is False
    assert data.metadata.reference_frame == DEFAULT_REFERENCE_FRAME


def test_lidar_sensor_sample_buckets_points_by_own_timestamp(tmp_path: Path) -> None:
    """sample() includes a sweep's points in a window based on each point's own timestamp."""
    path = _two_sweep_mcap(tmp_path)
    sensor = LidarSensor(path, packet_decoder=_FakePacketDecoder())
    spec = one_window_spec(0, 3_000)

    batches = list(sensor.sample(spec, policy=NoSamplingPolicy()))

    assert len(batches) == 1
    # Row's align timestamp is the earliest surviving point's own timestamp (900),
    # not the sweep's original sweep_timestamp_ns (1_000).
    np.testing.assert_array_equal(batches[0].align_timestamps_ns, [900])
    assert batches[0].points_xyz.shape == (2, 3)


def test_lidar_sensor_sample_splits_a_sweep_straddling_a_window_boundary(tmp_path: Path) -> None:
    """A sweep whose points span a window boundary contributes a partial row to each window.

    Includes a point at exactly the boundary timestamp (1_000) to exercise
    the half-open ``[start, exclusive_end)`` tie behavior in
    ``_sweep_points_in_window``'s mask: it must land only in the window that
    starts at 1_000, not the one that ends there.
    """
    records = [
        _record(_make_packet(sweep_id=0, sensor_ts_ns=800, sweep_ts_ns=800, point=(1.0, 0.0, 0.0)), 800, 0),
        _record(_make_packet(sweep_id=0, sensor_ts_ns=1_000, sweep_ts_ns=800, point=(1.5, 0.0, 0.0)), 1_000, 1),
        _record(_make_packet(sweep_id=0, sensor_ts_ns=1_200, sweep_ts_ns=800, point=(2.0, 0.0, 0.0)), 1_200, 2),
        _record(_make_packet(sweep_id=0, sensor_ts_ns=1_800, sweep_ts_ns=800, point=(3.0, 0.0, 0.0)), 1_800, 3),
    ]
    path = tmp_path / "straddling.mcap"
    _write_schemaless_lidar_mcap(path, records)
    sensor = LidarSensor(path, packet_decoder=_FakePacketDecoder())

    (first_batch,) = list(sensor.sample(one_window_spec(0, 1_000), policy=NoSamplingPolicy()))
    (second_batch,) = list(sensor.sample(one_window_spec(1_000, 2_000), policy=NoSamplingPolicy()))

    np.testing.assert_array_equal(first_batch.points_timestamps_ns, [800])
    np.testing.assert_array_equal(first_batch.align_timestamps_ns, [800])
    np.testing.assert_array_equal(second_batch.points_timestamps_ns, [1_000, 1_200, 1_800])
    np.testing.assert_array_equal(second_batch.align_timestamps_ns, [1_000])


def test_lidar_sensor_sample_returns_empty_batch_for_windows_with_no_overlapping_points(tmp_path: Path) -> None:
    """A window covering no sweep's points yields an empty, correctly shaped LidarData."""
    path = _two_sweep_mcap(tmp_path)
    sensor = LidarSensor(path, packet_decoder=_FakePacketDecoder())
    spec = one_window_spec(1_500, 4_000)

    (batch,) = list(sensor.sample(spec, policy=NoSamplingPolicy()))

    assert len(batch.align_timestamps_ns) == 0
    assert batch.points_xyz.shape == (0, 3)
    assert batch.points_intensity is not None
    assert batch.points_intensity.shape == (0,)
    assert batch.points_align_index is not None
    assert batch.points_align_index.dtype == np.uint16


def test_lidar_sensor_read_all_raises_past_uint16_row_limit(tmp_path: Path) -> None:
    """More than 65536 sweeps would overflow points_align_index's uint16 dtype and must raise."""
    too_many = _MAX_ALIGN_ROWS + 1
    decoder = _FixedSweepsPacketDecoder([_one_point_sweep(i) for i in range(too_many)])
    sensor = LidarSensor(_two_sweep_mcap(tmp_path), packet_decoder=decoder)

    with pytest.raises(ValueError, match="uint16"):
        sensor.read_all()


def test_lidar_sensor_read_all_succeeds_at_uint16_row_limit(tmp_path: Path) -> None:
    """Exactly 65536 sweeps (the valid max) must not raise -- only the limit itself is a one-off boundary."""
    decoder = _FixedSweepsPacketDecoder([_one_point_sweep(i) for i in range(_MAX_ALIGN_ROWS)])
    sensor = LidarSensor(_two_sweep_mcap(tmp_path), packet_decoder=decoder)

    data = sensor.read_all()

    assert len(data.align_timestamps_ns) == _MAX_ALIGN_ROWS


def test_lidar_sensor_rejects_decoder_yielding_sweeps_out_of_order(tmp_path: Path) -> None:
    """A PacketDecoder that yields sweeps out of order raises, not silently misbehaves."""
    decoder = _FixedSweepsPacketDecoder([_one_point_sweep(5_000), _one_point_sweep(1_000)])
    sensor = LidarSensor(_two_sweep_mcap(tmp_path), packet_decoder=decoder)

    with pytest.raises(ValueError, match="strictly disjoint"):
        sensor.read_all()


def test_lidar_sensor_rejects_decoder_yielding_overlapping_sweeps(tmp_path: Path) -> None:
    """Two sweeps whose point-timestamp ranges overlap raise at decode time, not deep inside sample().

    Regression test for a real bug: previously, _decode_all_sweeps only
    checked starts_ns/ends_ns independently nondecreasing, not disjoint, so
    overlapping sweeps could pass that check. sample()'s windowed path then
    built each window's align_timestamps_ns from the window-filtered
    (local) min of each candidate sweep, in original sweep order -- which
    is only guaranteed nondecreasing if sweep ranges never overlap. With
    overlap, two sweeps' filtered mins could come out in the wrong relative
    order, producing a non-monotonic align_timestamps_ns that crashed
    LidarData's own strict-increase validator several calls removed from
    the actual bad input. Rejecting overlap here, at the source, prevents
    that: it's the earliest point such a decoder's bad output can be
    caught deterministically, regardless of which window would have
    eventually tripped over it.
    """
    # Sweep 0 spans [100, 990]; sweep 1 spans [150, 1_000] -- overlapping,
    # not merely touching.
    sweep_a = _FakeSweep(
        points_xyz=np.zeros((2, 3), dtype=np.float32),
        intensity=np.zeros(2, dtype=np.float32),
        ring=np.zeros(2, dtype=np.uint16),
        sensor_timestamps_ns=np.array([100, 990], dtype=np.int64),
        sweep_timestamp_ns=100,
    )
    sweep_b = _FakeSweep(
        points_xyz=np.zeros((2, 3), dtype=np.float32),
        intensity=np.zeros(2, dtype=np.float32),
        ring=np.zeros(2, dtype=np.uint16),
        sensor_timestamps_ns=np.array([150, 1_000], dtype=np.int64),
        sweep_timestamp_ns=101,
    )
    decoder = _FixedSweepsPacketDecoder([sweep_a, sweep_b])
    sensor = LidarSensor(_two_sweep_mcap(tmp_path), packet_decoder=decoder)

    with pytest.raises(ValueError, match="strictly disjoint"):
        sensor.read_all()


def test_lidar_sensor_rejects_decoder_with_nondecreasing_bounds_but_tied_sweep_timestamps(
    tmp_path: Path,
) -> None:
    """Nondecreasing per-point bounds alone aren't enough -- sweep_timestamp_ns itself must strictly increase.

    A decoder could report sweep_timestamp_ns independently of its own points'
    timestamps (e.g. a bug in how it computes the reference instant). Two
    sweeps here have correctly nondecreasing point bounds but the same
    (tied) sweep_timestamp_ns -- read_all() would feed that tie into
    LidarData.align_timestamps_ns, which requires strict increase, and
    stream() yields one sweep at a time as its own single-row batch, where
    that per-batch check alone can't see the tie across yields. This must be
    caught here instead.
    """
    tied_sweep_timestamp_ns = 999_000
    sweep_a = _FakeSweep(
        points_xyz=np.zeros((2, 3), dtype=np.float32),
        intensity=np.zeros(2, dtype=np.float32),
        ring=np.zeros(2, dtype=np.uint16),
        sensor_timestamps_ns=np.array([100, 200], dtype=np.int64),
        sweep_timestamp_ns=tied_sweep_timestamp_ns,
    )
    sweep_b = _FakeSweep(
        points_xyz=np.zeros((2, 3), dtype=np.float32),
        intensity=np.zeros(2, dtype=np.float32),
        ring=np.zeros(2, dtype=np.uint16),
        sensor_timestamps_ns=np.array([300, 400], dtype=np.int64),
        sweep_timestamp_ns=tied_sweep_timestamp_ns,
    )
    decoder = _FixedSweepsPacketDecoder([sweep_a, sweep_b])
    sensor = LidarSensor(_two_sweep_mcap(tmp_path), packet_decoder=decoder)

    with pytest.raises(ValueError, match="strictly increasing sweep timestamps"):
        sensor.read_all()


def test_lidar_sensor_read_all_handles_decoder_yielding_an_empty_sweep(tmp_path: Path) -> None:
    """A decoder-yielded sweep with zero points contributes an empty row, not a crash."""
    decoder = _FixedSweepsPacketDecoder([_one_point_sweep(100), _empty_sweep(500), _one_point_sweep(900)])
    sensor = LidarSensor(_two_sweep_mcap(tmp_path), packet_decoder=decoder)

    data = sensor.read_all()

    np.testing.assert_array_equal(data.align_timestamps_ns, [100, 500, 900])
    np.testing.assert_array_equal(data.points_align_index, [0, 2])
    assert data.points_xyz.shape == (2, 3)


def test_lidar_sensor_sample_excludes_an_empty_sweep_from_every_window(tmp_path: Path) -> None:
    """A decoder-yielded sweep with zero points is cleanly excluded from any window it would overlap."""
    decoder = _FixedSweepsPacketDecoder([_one_point_sweep(100), _empty_sweep(500), _one_point_sweep(900)])
    sensor = LidarSensor(_two_sweep_mcap(tmp_path), packet_decoder=decoder)
    # [400, 600) only brackets the empty sweep's own sweep_timestamp_ns (500); neither
    # of the other two sweeps' single points fall in this window.
    spec = one_window_spec(400, 600)

    (batch,) = list(sensor.sample(spec, policy=NoSamplingPolicy()))

    assert len(batch.align_timestamps_ns) == 0
    assert batch.points_xyz.shape == (0, 3)


def test_lidar_sensor_sorts_points_within_a_sweep_by_timestamp(tmp_path: Path) -> None:
    """A decoder that emits one sweep's points out of firing-time order still produces valid LidarData."""
    unordered_ts = [100, 50, 200, 150]
    sweep = _FakeSweep(
        points_xyz=np.array([[i, 0.0, 0.0] for i in range(4)], dtype=np.float32),
        intensity=np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        ring=np.array([1, 2, 3, 4], dtype=np.uint16),
        sensor_timestamps_ns=np.array(unordered_ts, dtype=np.int64),
        sweep_timestamp_ns=min(unordered_ts),
    )
    decoder = _FixedSweepsPacketDecoder([sweep])
    sensor = LidarSensor(_two_sweep_mcap(tmp_path), packet_decoder=decoder)

    data = sensor.read_all()

    np.testing.assert_array_equal(data.points_timestamps_ns, [50, 100, 150, 200])
    # x-coordinate was set equal to the point's original index (0..3); sorting by
    # timestamp [100, 50, 200, 150] ascending should reorder to indices [1, 0, 3, 2].
    np.testing.assert_array_equal(data.points_xyz[:, 0], [1, 0, 3, 2])
    np.testing.assert_array_equal(data.points_intensity, [20.0, 10.0, 40.0, 30.0])
    np.testing.assert_array_equal(data.points_ring, [2, 1, 4, 3])


def test_lidar_sensor_read_all_preserves_points_align_index_across_internally_disordered_sweeps(
    tmp_path: Path,
) -> None:
    """points_align_index tracks true origin sweep correctly when each sweep's own points are out of order.

    Two sweeps with disjoint point-timestamp ranges (the only shape
    _decode_all_sweeps now accepts, see the strict-disjointness fix above),
    each internally out of firing-time order. The batch-wide stable sort
    must reorder each sweep's points without ever letting points_align_index
    drift from their true originating sweep.
    """
    sweep_a = _FakeSweep(
        points_xyz=np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]], dtype=np.float32),
        intensity=np.zeros(2, dtype=np.float32),
        ring=np.zeros(2, dtype=np.uint16),
        sensor_timestamps_ns=np.array([50, 10], dtype=np.int64),
        sweep_timestamp_ns=10,
    )
    sweep_b = _FakeSweep(
        points_xyz=np.array([[20.0, 0.0, 0.0], [21.0, 0.0, 0.0]], dtype=np.float32),
        intensity=np.zeros(2, dtype=np.float32),
        ring=np.zeros(2, dtype=np.uint16),
        sensor_timestamps_ns=np.array([250, 200], dtype=np.int64),
        sweep_timestamp_ns=200,
    )
    decoder = _FixedSweepsPacketDecoder([sweep_a, sweep_b])
    sensor = LidarSensor(_two_sweep_mcap(tmp_path), packet_decoder=decoder)

    data = sensor.read_all()

    np.testing.assert_array_equal(data.points_timestamps_ns, [10, 50, 200, 250])
    np.testing.assert_array_equal(data.points_align_index, [0, 0, 1, 1])
    np.testing.assert_array_equal(data.points_xyz[:, 0], [11, 10, 21, 20])


def test_lidar_sensor_sample_requires_no_sampling_policy(tmp_path: Path) -> None:
    """sample() rejects any policy other than NoSamplingPolicy."""
    path = _two_sweep_mcap(tmp_path)
    sensor = LidarSensor(path, packet_decoder=_FakePacketDecoder())
    spec = one_window_spec(0, 3_000)

    with pytest.raises(TypeError, match="NoSamplingPolicy"):
        list(sensor.sample(spec, policy=NearestTimestampPolicy()))


def test_lidar_sensor_supports_sampling_policy() -> None:
    """Only NoSamplingPolicy is supported."""
    sensor = LidarSensor(b"", packet_decoder=_FakePacketDecoder())
    assert sensor.supports_sampling_policy(NoSamplingPolicy()) is True
    assert sensor.supports_sampling_policy(NearestTimestampPolicy()) is False


def test_lidar_sensor_stream_yields_one_row_per_sweep_in_order(tmp_path: Path) -> None:
    """stream() yields each sweep as its own single-row batch, in time order."""
    path = _two_sweep_mcap(tmp_path)
    sensor = LidarSensor(path, packet_decoder=_FakePacketDecoder())

    batches = list(sensor.stream())

    assert len(batches) == 2
    assert batches[0].align_timestamps_ns.tolist() == [1_000]
    assert batches[1].align_timestamps_ns.tolist() == [5_000]


def test_lidar_sensor_start_end_and_timestamps_pass_through(tmp_path: Path) -> None:
    """start_ns/end_ns/timestamps_ns pass through MCAP's own packet log times."""
    path = _two_sweep_mcap(tmp_path)
    sensor = LidarSensor(path, packet_decoder=_FakePacketDecoder())

    assert sensor.start_ns == 900
    assert sensor.end_ns == 4_950
    np.testing.assert_array_equal(sensor.timestamps_ns, [900, 950, 4_900, 4_950])


def test_calibration_csv_for_topic_reads_channel_metadata(tmp_path: Path) -> None:
    """calibration_csv_for_topic reads back the calibration CSV written into channel metadata."""
    path = _two_sweep_mcap(tmp_path)
    with path.open("rb") as stream:
        reader = mcap_make_reader(stream)  # type: ignore[no-untyped-call]
        csv = calibration_csv_for_topic(reader, DEFAULT_TOPIC)

    assert csv == "Channel,Elevation,Azimuth\n1,0,0\n"


def test_calibration_csv_for_topic_raises_when_missing(tmp_path: Path) -> None:
    """calibration_csv_for_topic raises when the channel has no calibration_csv entry."""
    records = [_record(_make_packet(sweep_id=0, sensor_ts_ns=0, sweep_ts_ns=0), 0, 0)]
    path = tmp_path / "no_calibration.mcap"
    _write_schemaless_lidar_mcap(path, records, calibration_csv=None)

    with path.open("rb") as stream:
        reader = mcap_make_reader(stream)  # type: ignore[no-untyped-call]
        with pytest.raises(ValueError, match="calibration_csv"):
            calibration_csv_for_topic(reader, DEFAULT_TOPIC)
