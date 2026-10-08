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
"""Decode Hesai Pandar128E3X UDP packets into per-sweep point clouds.

Satisfies the ``PacketDecoder`` protocol from
:mod:`cosmos_curator.core.sensors.sensors.lidar_sensor`: ``parse`` takes
``(payload_bytes, arrival_timestamp_ns)`` tuples -- the vendor's undecoded
UDP payload verbatim, plus a recorder/host-clock arrival stamp -- and hands the
payload bytes to hydra4's OT-128 decoder (wire-format compatible with
Pandar128E3X).

hydra4 is *stateful across* ``to_pypcd4`` calls: it accumulates decoded blocks
into an internal buffer and emits one ``PointCloud`` per full 360° sweep it
detects via azimuth wrap-around. This parser is therefore a thin batching
wrapper -- it forwards packets in fixed-size chunks and yields whatever
PointClouds hydra4 emits, converted into :class:`HesaiLidarSweep`.

The trailing partial sweep at end-of-stream is silently dropped: hydra4 has no
``flush()`` API and there is no wrap event to trigger emission of the tail
accumulator. Recovering or accounting for those points is tracked separately.

The parser owns a per-instance temporary CSV file carrying per-unit factory
calibration, because hydra4 coerces ``calibration_path`` via ``Path(...)``
internally and cannot accept an in-memory ``StringIO``.
"""

import tempfile
import weakref
from collections.abc import Iterable, Iterator
from itertools import batched
from pathlib import Path
from types import TracebackType
from typing import Final, Self

import attrs
import numpy as np
from hydra4 import Hydra
from hydra4.msgs import (
    builtin_interfaces__msg__Time as _Time,
)
from hydra4.msgs import (
    pandar_msgs__msg__PandarPacket as _PandarPacket,
)
from hydra4.msgs import (
    pandar_msgs__msg__PandarScan as _PandarScan,
)
from hydra4.msgs import (
    std_msgs__msg__Header as _Header,
)
from numpy.typing import NDArray
from pypcd4 import PointCloud  # type: ignore[attr-defined]

HESAI_P128_CODEC: Final[str] = "HESAI_P_128_V4P5"

_NS_PER_S: Final[int] = 1_000_000_000

# Sensor frame convention — decoded points live in Pandar128E3X's own frame,
# no rig transform applied. Callers apply frame transforms downstream.
_FRAME_ID: Final[str] = "pandar128_top"

# Packets per ``to_pypcd4`` call. Batching is purely a perf lever; correctness
# is unchanged for any batch size because hydra4's sweep detection is stateful
# across calls. 500 covers ~50° of azimuth so at most one wrap is possible per
# call — keeps behaviour predictable if we ever want to inspect emissions.
_DEFAULT_PACKET_BATCH_SIZE: Final[int] = 500


@attrs.frozen
class HesaiLidarSweep:
    """Decoded point cloud for one full 360° Pandar128E3X sweep.

    Fields concatenate all points hydra4 emits for the sweep. In dual-return
    modes hydra4 drops the redundant "duplicate" return within its
    ``dual_return_distance_threshold`` (0.1 m by default) but does not tag
    surviving points with a return-type label; downstream code that needs a
    per-point return-index reconstructs it by grouping points that
    share ``(ring, sensor_timestamps_ns)`` and applying the source's own
    return-mode metadata to the physical-position ordering -- that is
    not a wire fact the parser can honestly expose.

    ``sweep_timestamp_ns`` is the earliest per-point sensor timestamp in the
    sweep, derived from hydra4's timing model (packet-Tail timestamp + static
    per-block / per-channel firing offsets). It reflects the sensor clock, not
    the recorder host clock.
    """

    points_xyz: NDArray[np.float32]
    intensity: NDArray[np.float32]
    ring: NDArray[np.uint16]
    sensor_timestamps_ns: NDArray[np.int64]
    sweep_timestamp_ns: int


def _parse_device_config(config: str) -> dict[str, str]:
    """Parse a ``device=...,key=value,...`` config string into a dict.

    The config string is a comma-separated ``key=value`` list; the first
    field is conventionally ``device=<SENSOR_MODEL>``. Empty and
    ill-formed entries are silently dropped.
    """
    result: dict[str, str] = {}
    for field in config.split(","):
        key, sep, value = field.partition("=")
        if sep and key:
            result[key.strip()] = value
    return result


def is_hesai_p128_track(device_config: str | dict[str, str]) -> bool:
    """Return True if the device config describes a Pandar128E3X track."""
    parsed = device_config if isinstance(device_config, dict) else _parse_device_config(device_config)
    return parsed.get("device") == HESAI_P128_CODEC


class HesaiLidarParser:
    """Stream Pandar128E3X UDP packets through hydra4 and yield decoded sweeps.

    Construction stages the caller-provided per-unit calibration CSV into a
    per-instance temporary file. The file is deleted when :meth:`close` runs
    (or on ``__exit__`` when used as a context manager); the parser is
    unusable after that. Callers that decode multiple clips should build one
    ``HesaiLidarParser`` per clip so each carries the right calibration.

    The parser is not thread-safe; callers wanting concurrent decoding should
    build one instance per worker.
    """

    def __init__(
        self,
        calibration_csv: str,
        *,
        min_distance: float | None = None,
        max_distance: float | None = None,
        dual_return_distance_threshold: float = 0.1,
        packet_batch_size: int = _DEFAULT_PACKET_BATCH_SIZE,
    ) -> None:
        """Stage ``calibration_csv`` to a per-instance temp file and build the hydra4 decoder."""
        if packet_batch_size < 1:
            msg = f"packet_batch_size must be >= 1, got {packet_batch_size}"
            raise ValueError(msg)
        # Initialise close-critical attributes before any step that can raise, so
        # that :meth:`close` and ``__exit__`` are safe to call on a partially
        # constructed instance and never surface an ``AttributeError``.
        self._closed = False
        self._calibration_path: Path | None = None
        self._finalizer = None
        # hydra4 opens ``calibration_path`` at scan time via ``Path(...)``; the file
        # therefore has to outlive construction, so keeping it open with a
        # context manager here is intentionally wrong for the lifetime we need.
        calibration_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="w",
            suffix=".csv",
            prefix="hesai_p128_calib_",
            delete=False,
        )
        self._calibration_path = Path(calibration_file.name)
        # Safety net for callers that never call close()/use the context manager --
        # e.g. LidarSensor stores an injected PacketDecoder but never closes it, since
        # the protocol itself has no close(). weakref.finalize holds no strong
        # reference back to self (the bound Path.unlink only references the path), so
        # it does not keep this instance alive; it fires once, at GC or interpreter
        # shutdown, whichever comes first. close() still fires it eagerly and
        # deterministically when callers do remember to use it.
        self._finalizer = weakref.finalize(self, self._calibration_path.unlink, missing_ok=True)
        try:
            calibration_file.write(calibration_csv)
            calibration_file.flush()
            calibration_file.close()
            self._hydra = Hydra(
                model=Hydra.Model.PandarOT128,
                calibration_path=self._calibration_path,
                min_distance=min_distance,
                max_distance=max_distance,
                dual_return_distance_threshold=dual_return_distance_threshold,
            )
        except BaseException:
            # Any failure past temp-file creation must unlink the file and
            # close the underlying handle, so callers never see a leaked file.
            if not calibration_file.closed:
                calibration_file.close()
            self.close()
            raise
        self._packet_batch_size = packet_batch_size

    def __enter__(self) -> Self:
        """Return ``self`` so instances may be used as context managers."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Delete the backing calibration file on context exit."""
        self.close()

    def close(self) -> None:
        """Delete the backing calibration file; the parser is unusable afterward."""
        if self._closed:
            return
        if self._finalizer is not None:
            self._finalizer()
        self._closed = True

    def parse(self, packets: Iterable[tuple[bytes, int]]) -> Iterator[HesaiLidarSweep]:
        """Feed ``(payload_bytes, arrival_timestamp_ns)`` tuples to hydra4 and yield sweeps.

        Sweep boundaries are detected internally by hydra4; this method
        forwards batches of packets and yields one :class:`HesaiLidarSweep`
        per :class:`hydra4.PointCloud` emitted. The trailing packets past
        the last wrap remain in hydra4's accumulator and are silently
        discarded when the iterator ends.
        """
        if self._closed:
            msg = "HesaiLidarParser has been closed"
            raise RuntimeError(msg)

        # ``strict`` was added to ``itertools.batched`` in Python 3.13; the
        # project supports 3.12+ (see ``pyproject.toml`` requires-python), so
        # the argument has to stay off here.
        for batch in batched(packets, self._packet_batch_size):  # noqa: B911
            scan = _build_pandar_scan(batch)
            for cloud in self._hydra.to_pypcd4(scan):
                yield _cloud_to_sweep(cloud)


def _build_pandar_scan(batch: tuple[tuple[bytes, int], ...]) -> _PandarScan:
    first_ts_ns = batch[0][1]
    header = _Header(stamp=_ns_to_time(first_ts_ns), frame_id=_FRAME_ID)
    packets = [
        _PandarPacket(
            stamp=_ns_to_time(arrival_ts_ns),
            data=np.frombuffer(payload, dtype=np.uint8),
            size=len(payload),
        )
        for payload, arrival_ts_ns in batch
    ]
    return _PandarScan(header=header, packets=packets)


def _ns_to_time(ts_ns: int) -> _Time:
    return _Time(sec=ts_ns // _NS_PER_S, nanosec=ts_ns % _NS_PER_S)


def _cloud_to_sweep(cloud: PointCloud) -> HesaiLidarSweep:
    data = cloud.pc_data
    xyz = np.stack(
        [
            np.asarray(data["x"], dtype=np.float32),
            np.asarray(data["y"], dtype=np.float32),
            np.asarray(data["z"], dtype=np.float32),
        ],
        axis=1,
    )
    sensor_timestamps_ns = np.rint(np.asarray(data["stamp"], dtype=np.float64) * _NS_PER_S).astype(np.int64)
    sweep_timestamp_ns = int(sensor_timestamps_ns.min()) if sensor_timestamps_ns.size else 0
    return HesaiLidarSweep(
        points_xyz=xyz,
        intensity=np.asarray(data["intensity"], dtype=np.float32),
        ring=np.asarray(data["ring"], dtype=np.uint16),
        sensor_timestamps_ns=sensor_timestamps_ns,
        sweep_timestamp_ns=sweep_timestamp_ns,
    )
