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

"""SensorGroup: top-level coordinator for aligned multi-sensor sampling."""

from collections.abc import Generator, Mapping
from typing import Protocol

from cosmos_curator.core.sensors.data.aligned_frame import AlignedFrame
from cosmos_curator.core.sensors.data.sensor_data import SensorData
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec

_MIN_SENSOR_OVERLAP_PARTICIPANTS = 2


class Sensor(Protocol):  # pragma: no cover
    """Structural interface for all sensor implementations.

    Any object with ``start_ns``, ``end_ns``, and ``sample()`` satisfies this
    protocol; explicit inheritance is not required.
    """

    @property
    def start_ns(self) -> int:
        """Earliest sensor timestamp in nanoseconds."""
        ...

    @property
    def end_ns(self) -> int:
        """Latest sensor timestamp in nanoseconds."""
        ...

    def sample(self, spec: SamplingSpec) -> Generator[SensorData, None, None]:
        """Yield one ``SensorData`` per window in ``spec.grid``."""
        ...


def _sensor_overlap(
    sensor_ranges: Mapping[str, tuple[int, int]],
    frame_start_ns: int,
    frame_end_ns: int,
) -> float:
    """Return the minimum per-sensor coverage fraction over the frame interval."""
    if not sensor_ranges:
        msg = "sensor_ranges must be non-empty"
        raise ValueError(msg)

    duration_ns = frame_end_ns - frame_start_ns
    if duration_ns <= 0:
        msg = f"frame interval duration must be positive, got {frame_start_ns=} {frame_end_ns=}"
        raise ValueError(msg)

    coverages: list[float] = []
    for start_ns, end_ns in sensor_ranges.values():
        overlap_start_ns = max(start_ns, frame_start_ns)
        overlap_end_ns = min(end_ns, frame_end_ns)
        covered_ns = max(0, overlap_end_ns - overlap_start_ns)
        coverages.append(covered_ns / duration_ns)

    return min(coverages)


class SensorGroup:
    """Top-level coordinator for aligned multi-sensor sampling.

    ``SensorGroup`` owns a named collection of sensors, exposes aggregate
    ``start_ns`` / ``end_ns`` bounds, and drives all sensor generators in
    lockstep through a single ``.sample(spec)`` entry point.

    Partial coverage:
        When a sensor has no data for a window it yields empty
        ``SensorData`` (``len(align_timestamps_ns) == 0``). Such sensors are
        omitted from that window's ``AlignedFrame.sensor_data``.  Windows
        where *every* sensor has no data produce an ``AlignedFrame`` with an
        empty ``sensor_data`` mapping.

    Policy enforcement:
        The same ``spec`` — including ``spec.policy`` — is passed to every
        sensor generator.  Each sensor enforces the tolerance independently
        via :func:`~cosmos_curator.core.sensors.sampling.sampler.sample_window_indices`.
        When ``spec.policy.sensor_overlap`` is greater than zero, each
        multi-sensor ``AlignedFrame`` must satisfy that minimum temporal
        overlap fraction across participating sensor payloads. A ``ValueError``
        raised by any sensor or policy check propagates to the caller
        unchanged.
    """

    def __init__(self, sensors: dict[str, Sensor]) -> None:
        """Initialise with a non-empty mapping of named sensors.

        Args:
            sensors: named sensors; must contain at least one entry.

        Raises:
            ValueError: if ``sensors`` is empty.

        """
        if not sensors:
            msg = "sensors must be non-empty"
            raise ValueError(msg)
        self._sensors = dict(sensors)

    @property
    def start_ns(self) -> int:
        """Minimum ``start_ns`` across all sensors."""
        return min(s.start_ns for s in self._sensors.values())

    @property
    def end_ns(self) -> int:
        """Maximum ``end_ns`` across all sensors."""
        return max(s.end_ns for s in self._sensors.values())

    def sample(self, spec: SamplingSpec) -> Generator[AlignedFrame, None, None]:
        """Yield one ``AlignedFrame`` per window in ``spec.grid``.

        All sensor generators are started with the same ``spec`` and advanced
        in lockstep — one step per window.  Each yielded frame carries
        ``align_timestamps_ns == window.timestamps_ns`` and a ``sensor_data``
        mapping that includes only sensors with data for that window.

        Args:
            spec: sampling specification; the same instance is passed to every
                sensor generator.

        Yields:
            ``AlignedFrame`` for each window in ``spec.grid``.

        Raises:
            ValueError: if any sensor's policy tolerance is exceeded, or if
                the aligned frame overlap is below ``policy.sensor_overlap``.

        """
        generators = {name: sensor.sample(spec) for name, sensor in self._sensors.items()}
        for window in spec.grid:
            sensor_data: dict[str, SensorData] = {}
            for name, gen in generators.items():
                data = next(gen)
                if len(data.align_timestamps_ns) > 0:
                    sensor_data[name] = data
            frame = AlignedFrame(
                align_timestamps_ns=window.timestamps_ns,
                sensor_data=sensor_data,
            )
            policy = spec.policy
            if (
                policy is not None
                and policy.sensor_overlap > 0.0
                and len(frame.sensor_data) >= _MIN_SENSOR_OVERLAP_PARTICIPANTS
            ):
                sensor_ranges = {
                    sensor_id: (int(data.sensor_timestamps_ns[0]), int(data.sensor_timestamps_ns[-1]))
                    for sensor_id, data in frame.sensor_data.items()
                }
                frame_start_ns = min(start_ns for start_ns, _end_ns in sensor_ranges.values())
                frame_end_ns = max(end_ns for _start_ns, end_ns in sensor_ranges.values())
                if frame_end_ns <= frame_start_ns:
                    msg = f"frame interval duration must be positive, got {frame_start_ns=} {frame_end_ns=}"
                    raise ValueError(msg)
                overlap_score = _sensor_overlap(sensor_ranges, frame_start_ns, frame_end_ns)
                if overlap_score < policy.sensor_overlap:
                    msg = f"sensor_overlap {overlap_score:.6f} is below required threshold {policy.sensor_overlap:.6f}"
                    raise ValueError(msg)
            yield frame
