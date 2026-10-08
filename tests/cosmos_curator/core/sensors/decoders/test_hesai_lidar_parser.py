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
"""Tests for :mod:`cosmos_curator.core.sensors.decoders.hesai_lidar_parser`."""

import gc
import math
import struct
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from cosmos_curator.core.sensors.decoders import hesai_lidar_parser
from cosmos_curator.core.sensors.decoders.hesai_lidar_parser import (
    HESAI_P128_CODEC,
    HesaiLidarParser,
    HesaiLidarSweep,
    is_hesai_p128_track,
)

# Pandar128E3X wire-format constants (per Pandar128E3X manual §3.1.2 and the
# hydra4 OT-128 decoder we adapt). Kept local to the tests so a change to the
# parser's own private constants does not silently mask a wire-layout drift.
_PRE_HEADER_SIZE = 6
_HEADER_SIZE = 6
_CHANNEL_COUNT = 128
_BYTES_PER_CHANNEL = 3  # baseline (no Weight Factor)
_BLOCK_DATA_SIZE = _CHANNEL_COUNT * _BYTES_PER_CHANNEL  # 384
_CRC_SIZE = 4
_BODY_SIZE = 2 + _BLOCK_DATA_SIZE + 2 + _BLOCK_DATA_SIZE + _CRC_SIZE  # 776
_FUNCTIONAL_SAFETY_SIZE = 17
_TAIL_SIZE = 56
_BASELINE_PAYLOAD_SIZE = _PRE_HEADER_SIZE + _HEADER_SIZE + _BODY_SIZE + _FUNCTIONAL_SAFETY_SIZE + _TAIL_SIZE  # 861

_HEADER_FLAGS_FUNC_SAFETY = 0x04  # bit[2] — turns on the 17 B Functional Safety block

_TAIL_OFF_AZ_STATE = 9
_TAIL_OFF_OP_STATE = 11
_TAIL_OFF_RET_MODE = 12
_TAIL_OFF_DATETIME = 15
_TAIL_OFF_TIMESTAMP = 21

_OP_STATE_HIGH_RES = 0
_OP_STATE_STANDARD = 2

# Hesai Pandar Tail return-mode wire codes (manual §3.1.2 Tail).
_RETURN_MODE_STRONGEST = 0x37  # single-return
_RETURN_MODE_LAST_STRONGEST = 0x39  # dual-return, matches real corpus data

_DIS_UNIT_M = 0.004  # per manual §3.1.2.2: distance quanta = 4 mm

_HUNDREDTHS_PER_DEGREE = 100
_ONE_REVOLUTION_HUNDREDTHS = 360 * _HUNDREDTHS_PER_DEGREE  # 36000
# Step used by the sweep-of-packets fixture — 0.1° per packet, matching the
# corpus hres value and hydra4's expectation.
_PACKET_AZIMUTH_STEP_HUNDREDTHS = 10
_PACKETS_PER_FULL_SWEEP = _ONE_REVOLUTION_HUNDREDTHS // _PACKET_AZIMUTH_STEP_HUNDREDTHS  # 3600

# Fixed sensor wall-clock for synthetic packets — kept UTC so the derived
# per-point ``stamp`` field is independent of the machine timezone.
_SENSOR_DATETIME = datetime(2026, 1, 1, 0, 0, 0, 0, tzinfo=UTC)


# ─────────────────────────────────────────────────────────────────────────────
# Fixture builders
# ─────────────────────────────────────────────────────────────────────────────


def _make_calibration_csv(
    elevations_deg: Sequence[float] | None = None,
    azimuth_offsets_deg: Sequence[float] | None = None,
) -> str:
    """Return a hydra4-compatible 128-row calibration CSV.

    Defaults to a zeroed calibration (elevation=0°, azimuth offset=0°) so a
    channel firing along the sensor frame's XY plane gives points on that
    plane — makes round-trip geometry checks trivial.
    """
    elevations = elevations_deg if elevations_deg is not None else [0.0] * _CHANNEL_COUNT
    azimuth_offsets = azimuth_offsets_deg if azimuth_offsets_deg is not None else [0.0] * _CHANNEL_COUNT
    if len(elevations) != _CHANNEL_COUNT or len(azimuth_offsets) != _CHANNEL_COUNT:
        msg = f"calibration must have exactly {_CHANNEL_COUNT} rows"
        raise ValueError(msg)
    lines = ["Channel,Elevation,Azimuth"]
    lines.extend(f"{i + 1},{elevations[i]:.9g},{azimuth_offsets[i]:.9g}" for i in range(_CHANNEL_COUNT))
    lines.append("")
    return "\n".join(lines)


def _encode_block(returns: dict[int, tuple[float, int]]) -> bytes:
    """Encode one 128-channel block; distance is metres, intensity is 0..255.

    Channels not mentioned in ``returns`` receive distance=0 (invalid return),
    which hydra4 skips.
    """
    block = bytearray(_BLOCK_DATA_SIZE)
    for channel, (distance_m, intensity) in returns.items():
        if not 0 <= channel < _CHANNEL_COUNT:
            msg = f"channel out of range [0, {_CHANNEL_COUNT}): {channel}"
            raise ValueError(msg)
        distance_quanta = round(distance_m / _DIS_UNIT_M)
        if not 0 <= distance_quanta <= 0xFFFF:
            msg = f"distance out of encodable range: {distance_m} m"
            raise ValueError(msg)
        if not 0 <= intensity <= 0xFF:
            msg = f"intensity out of range: {intensity}"
            raise ValueError(msg)
        offset = channel * _BYTES_PER_CHANNEL
        struct.pack_into("<HB", block, offset, distance_quanta, intensity)
    return bytes(block)


def _make_packet(
    *,
    azimuth_hundredths: int,
    returns: dict[int, tuple[float, int]] | None = None,
    second_block_returns: dict[int, tuple[float, int]] | None = None,
    operational_state: int = _OP_STATE_STANDARD,
    return_mode: int = _RETURN_MODE_STRONGEST,
    timestamp_us: int = 0,
) -> bytes:
    """Build one synthetic 861 B Pandar128E3X UDP body with Functional Safety on.

    Real corpus data emits Functional Safety on every packet, so tests
    mirror that layout to catch regressions on the actual on-wire shape.

    ``returns`` populates block 1; ``second_block_returns`` populates block 2
    (defaults to a copy of ``returns``, matching the block-1/block-2 pairing
    used by dual-return capture).
    """
    if returns is None:
        returns = {}
    if second_block_returns is None:
        second_block_returns = returns

    payload = bytearray(_BASELINE_PAYLOAD_SIZE)

    # Pre-Header.
    payload[0] = 0xEE
    payload[1] = 0xFF
    payload[2] = 0x01  # protocol major
    payload[3] = 0x04  # protocol minor

    # Header — flags byte at offset 11 enables Functional Safety.
    payload[11] = _HEADER_FLAGS_FUNC_SAFETY

    # Body: azimuth_1 | block_1 | azimuth_2 | block_2 | CRC.
    body_offset = _PRE_HEADER_SIZE + _HEADER_SIZE
    struct.pack_into("<H", payload, body_offset, azimuth_hundredths)
    payload[body_offset + 2 : body_offset + 2 + _BLOCK_DATA_SIZE] = _encode_block(returns)
    struct.pack_into("<H", payload, body_offset + 2 + _BLOCK_DATA_SIZE, azimuth_hundredths)
    payload[body_offset + 2 + _BLOCK_DATA_SIZE + 2 : body_offset + 2 + _BLOCK_DATA_SIZE + 2 + _BLOCK_DATA_SIZE] = (
        _encode_block(second_block_returns)
    )
    # CRC bytes intentionally zero — hydra4's decoder does not validate CRC.

    # Functional Safety: 17 zero bytes at offset body_offset + _BODY_SIZE.
    # Already zero from bytearray init.

    # Tail (56 bytes) at offset body_offset + _BODY_SIZE + _FUNCTIONAL_SAFETY_SIZE = 805.
    tail_offset = body_offset + _BODY_SIZE + _FUNCTIONAL_SAFETY_SIZE
    # Azimuth State: both blocks at state 0 (both bits pairs = 00).
    struct.pack_into("<H", payload, tail_offset + _TAIL_OFF_AZ_STATE, 0)
    payload[tail_offset + _TAIL_OFF_OP_STATE] = operational_state
    payload[tail_offset + _TAIL_OFF_RET_MODE] = return_mode
    # Date/Time: Year - 1900, Month, Day, Hour, Minute, Second.
    struct.pack_into(
        "<6B",
        payload,
        tail_offset + _TAIL_OFF_DATETIME,
        _SENSOR_DATETIME.year - 1900,
        _SENSOR_DATETIME.month,
        _SENSOR_DATETIME.day,
        _SENSOR_DATETIME.hour,
        _SENSOR_DATETIME.minute,
        _SENSOR_DATETIME.second,
    )
    struct.pack_into("<I", payload, tail_offset + _TAIL_OFF_TIMESTAMP, timestamp_us)

    assert len(payload) == _BASELINE_PAYLOAD_SIZE
    return bytes(payload)


def _sweep_of_packets(
    start_hundredths: int,
    step_hundredths: int,
    count: int,
    *,
    returns: dict[int, tuple[float, int]] | None = None,
    arrival_start_ns: int = 0,
    arrival_step_ns: int = 100_000,
) -> list[tuple[bytes, int]]:
    """Build ``count`` packets marching azimuth forward, wrapping mod 360°."""
    packets: list[tuple[bytes, int]] = []
    for i in range(count):
        azimuth = (start_hundredths + i * step_hundredths) % _ONE_REVOLUTION_HUNDREDTHS
        payload = _make_packet(azimuth_hundredths=azimuth, returns=returns)
        packets.append((payload, arrival_start_ns + i * arrival_step_ns))
    return packets


# ─────────────────────────────────────────────────────────────────────────────
# Track / codec helpers
# ─────────────────────────────────────────────────────────────────────────────


def test_is_hesai_p128_track_accepts_pandar128_e3x_config() -> None:
    """A comma-separated device configuration with device=HESAI_P_128_V4P5 recognises as a Pandar128 track."""
    assert is_hesai_p128_track(f"device={HESAI_P128_CODEC},ip=10.0.0.1,port=7501")


def test_is_hesai_p128_track_accepts_parsed_dict() -> None:
    """A pre-parsed config dict with the same device value is also recognised."""
    assert is_hesai_p128_track({"device": HESAI_P128_CODEC, "port": "7501"})


def test_is_hesai_p128_track_rejects_other_device() -> None:
    """Config strings advertising a different Hesai model are not accepted."""
    assert not is_hesai_p128_track("device=HESAI_A_128,port=7501")


def test_is_hesai_p128_track_rejects_missing_device_key() -> None:
    """Config strings without any device key resolve to False."""
    assert not is_hesai_p128_track("ip=10.0.0.1,port=7501")


# ─────────────────────────────────────────────────────────────────────────────
# Calibration lifecycle
# ─────────────────────────────────────────────────────────────────────────────


def test_parser_construction_stages_calibration_to_disk() -> None:
    """Construction writes the calibration CSV to a real filesystem path hydra4 can open."""
    parser = HesaiLidarParser(_make_calibration_csv())
    try:
        assert parser._calibration_path.exists()
        assert parser._calibration_path.read_text().startswith("Channel,Elevation,Azimuth")
    finally:
        parser.close()


def test_parser_close_removes_calibration_file() -> None:
    """Explicit close unlinks the backing calibration file."""
    parser = HesaiLidarParser(_make_calibration_csv())
    calibration_path = parser._calibration_path
    parser.close()
    assert not calibration_path.exists()


def test_parser_close_is_idempotent() -> None:
    """Calling close twice does not raise or attempt a second unlink."""
    parser = HesaiLidarParser(_make_calibration_csv())
    parser.close()
    parser.close()


def test_parser_calibration_file_is_cleaned_up_without_explicit_close() -> None:
    """A caller that never calls close()/uses the context manager still doesn't leak the file.

    Mirrors how LidarSensor actually uses a PacketDecoder: it stores the
    instance and never calls close() on it, since the PacketDecoder protocol
    has no close() at all. The weakref.finalize safety net must still clean
    up once the parser itself is garbage collected.
    """
    parser = HesaiLidarParser(_make_calibration_csv())
    calibration_path = parser._calibration_path
    assert calibration_path.exists()

    del parser
    gc.collect()

    assert not calibration_path.exists()


def test_parser_context_manager_cleans_up() -> None:
    """Using the parser as a context manager unlinks the calibration file on exit."""
    with HesaiLidarParser(_make_calibration_csv()) as parser:
        calibration_path = parser._calibration_path
        assert calibration_path.exists()
    assert not calibration_path.exists()


def test_parser_use_after_close_raises() -> None:
    """Parse-after-close raises RuntimeError instead of silently returning nothing."""
    parser = HesaiLidarParser(_make_calibration_csv())
    parser.close()
    with pytest.raises(RuntimeError, match="closed"):
        list(parser.parse([]))


def test_parser_rejects_bad_packet_batch_size() -> None:
    """Non-positive packet_batch_size is rejected at construction."""
    with pytest.raises(ValueError, match="packet_batch_size"):
        HesaiLidarParser(_make_calibration_csv(), packet_batch_size=0)


def test_parser_unlinks_calibration_when_hydra_constructor_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure inside the hydra4 constructor must not leak the calibration temp file."""
    leaked_path_holder: dict[str, Path] = {}

    class _RaisingHydra:
        Model = hesai_lidar_parser.Hydra.Model

        def __init__(self, *_args: object, **kwargs: object) -> None:
            calibration_path = kwargs["calibration_path"]
            assert isinstance(calibration_path, Path)
            leaked_path_holder["path"] = calibration_path
            msg = "simulated hydra4 constructor failure"
            raise RuntimeError(msg)

    monkeypatch.setattr(hesai_lidar_parser, "Hydra", _RaisingHydra)

    with pytest.raises(RuntimeError, match="simulated hydra4 constructor failure"):
        HesaiLidarParser(_make_calibration_csv())

    calibration_path = leaked_path_holder["path"]
    assert not calibration_path.exists(), f"temp file leaked at {calibration_path}"


# ─────────────────────────────────────────────────────────────────────────────
# Empty input
# ─────────────────────────────────────────────────────────────────────────────


def test_parser_empty_iterable_yields_nothing() -> None:
    """An empty packet stream yields no HesaiLidarSweep values."""
    with HesaiLidarParser(_make_calibration_csv()) as parser:
        assert list(parser.parse([])) == []


# ─────────────────────────────────────────────────────────────────────────────
# Sweep emission — the fundamental hydra4 contract
# ─────────────────────────────────────────────────────────────────────────────


def test_parser_no_wrap_emits_nothing() -> None:
    """One partial sweep with no wrap-around must not emit a PointCloud."""
    packets = _sweep_of_packets(
        start_hundredths=0,
        step_hundredths=10,  # 0.1° per packet
        count=_PACKETS_PER_FULL_SWEEP - 1,  # stops at 359.9°, no wrap yet
        returns={0: (5.0, 100)},
    )
    with HesaiLidarParser(_make_calibration_csv()) as parser:
        assert list(parser.parse(packets)) == []


def test_parser_azimuth_wraparound_emits_one_sweep() -> None:
    """A wrap from ~360° back to 0° must trigger hydra4 to emit one PointCloud."""
    # 3601 packets = one full rotation plus the first packet of the next sweep,
    # which triggers hydra4's wrap detection and emission of the first sweep.
    packets = _sweep_of_packets(
        start_hundredths=0,
        step_hundredths=10,
        count=_PACKETS_PER_FULL_SWEEP + 1,
        returns={0: (5.0, 100)},
    )
    with HesaiLidarParser(_make_calibration_csv()) as parser:
        sweeps = list(parser.parse(packets))
    assert len(sweeps) == 1
    assert isinstance(sweeps[0], HesaiLidarSweep)


def test_parser_minimum_sweep_boundary_two_wraps_yields_two_sweeps() -> None:
    """Two full revolutions plus one packet must yield exactly two PointClouds."""
    packets = _sweep_of_packets(
        start_hundredths=0,
        step_hundredths=10,
        count=2 * _PACKETS_PER_FULL_SWEEP + 1,
        returns={0: (5.0, 100)},
    )
    with HesaiLidarParser(_make_calibration_csv()) as parser:
        sweeps = list(parser.parse(packets))
    assert len(sweeps) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Round-trip geometry
# ─────────────────────────────────────────────────────────────────────────────


def _decode_one_sweep(
    packets: list[tuple[bytes, int]],
    calibration_csv: str | None = None,
) -> HesaiLidarSweep:
    with HesaiLidarParser(calibration_csv or _make_calibration_csv()) as parser:
        sweeps = list(parser.parse(packets))
    assert len(sweeps) == 1, f"expected 1 sweep, got {len(sweeps)}"
    return sweeps[0]


def test_parser_single_return_geometry_zero_calibration() -> None:
    """A single return on channel 0 at 90° with zero calibration lands at (10, 0, 0).

    Verifies the end-to-end wire → hydra4 → HesaiLidarSweep geometry pipeline
    without any elevation or azimuth-offset contribution from calibration.
    """
    channel = 0
    distance_m = 10.0
    intensity = 200
    azimuth_deg = 90.0
    azimuth_hundredths = round(azimuth_deg * _HUNDREDTHS_PER_DEGREE)

    # One sweep worth of empty packets, plus one packet at 90° with a real
    # return, plus a wrap-triggering packet to force hydra4 to emit.
    step = 10
    prefix = _sweep_of_packets(
        start_hundredths=0,
        step_hundredths=step,
        count=azimuth_hundredths // step,
    )
    target = (
        _make_packet(azimuth_hundredths=azimuth_hundredths, returns={channel: (distance_m, intensity)}),
        len(prefix) * 100_000,
    )
    suffix = _sweep_of_packets(
        start_hundredths=azimuth_hundredths + step,
        step_hundredths=step,
        count=_PACKETS_PER_FULL_SWEEP - azimuth_hundredths // step,
        arrival_start_ns=(len(prefix) + 1) * 100_000,
    )
    packets = [*prefix, target, *suffix]

    sweep = _decode_one_sweep(packets)

    # There should be exactly one accepted return — two blocks would double it,
    # but block 2 has the same content in our fixture so both fire. For dual
    # return-mode 0x37 (STRONGEST) is single-return, so block 1 only.
    assert sweep.points_xyz.shape[0] == 1, sweep.points_xyz
    x, y, z = sweep.points_xyz[0]
    assert math.isclose(x, distance_m, abs_tol=1e-4)
    assert math.isclose(y, 0.0, abs_tol=1e-4)
    assert math.isclose(z, 0.0, abs_tol=1e-4)
    assert sweep.ring[0] == channel
    assert math.isclose(float(sweep.intensity[0]), intensity, abs_tol=1e-6)


def test_parser_geometry_matches_calibration_elevation() -> None:
    """A non-zero calibration elevation tilts the point off the XY plane.

    Feeds hydra4 a per-unit calibration with one channel at 15° elevation and
    checks the resulting point's Z / XY radius matches the expected trig.
    """
    channel = 0
    elevation_deg = 15.0
    elevations = [0.0] * _CHANNEL_COUNT
    elevations[channel] = elevation_deg

    distance_m = 20.0
    azimuth_deg = 0.0  # Along +Y in the sensor frame under zero azimuth offset.
    azimuth_hundredths = round(azimuth_deg * _HUNDREDTHS_PER_DEGREE)

    step = 10
    target = (
        _make_packet(azimuth_hundredths=azimuth_hundredths, returns={channel: (distance_m, 100)}),
        0,
    )
    rest = _sweep_of_packets(
        start_hundredths=azimuth_hundredths + step,
        step_hundredths=step,
        count=_PACKETS_PER_FULL_SWEEP,
        arrival_start_ns=100_000,
    )
    packets = [target, *rest]

    sweep = _decode_one_sweep(packets, _make_calibration_csv(elevations_deg=elevations))

    assert sweep.points_xyz.shape[0] == 1
    x, y, z = sweep.points_xyz[0]
    # elevation in radians: hydra4 stores calibration in radians internally,
    # even though the CSV is degrees.
    expected_z = distance_m * math.sin(math.radians(elevation_deg))
    expected_xy = distance_m * math.cos(math.radians(elevation_deg))
    assert math.isclose(z, expected_z, abs_tol=1e-3)
    assert math.isclose(math.hypot(x, y), expected_xy, abs_tol=1e-3)


def test_parser_sweep_timestamp_derived_from_sensor_clock() -> None:
    """``sweep_timestamp_ns`` must be the min of per-point ``sensor_timestamps_ns``."""
    channel = 0
    step = 10
    target = (
        _make_packet(azimuth_hundredths=0, returns={channel: (5.0, 100)}),
        0,
    )
    rest = _sweep_of_packets(
        start_hundredths=step,
        step_hundredths=step,
        count=_PACKETS_PER_FULL_SWEEP,
        arrival_start_ns=100_000,
    )
    sweep = _decode_one_sweep([target, *rest])
    assert sweep.sweep_timestamp_ns == int(np.asarray(sweep.sensor_timestamps_ns).min())
