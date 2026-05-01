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

"""Aligned frame data structures for cosmos_curator.core.sensors package."""

from collections.abc import Mapping
from types import MappingProxyType

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.data.sensor_data import SensorData
from cosmos_curator.core.sensors.utils.helpers import as_readonly_view
from cosmos_curator.core.sensors.utils.validation import strictly_increasing_int64_array


def _as_immutable_sensor_data(value: Mapping[str, SensorData]) -> Mapping[str, SensorData]:
    """Return an immutable snapshot of the caller-provided sensor mapping."""
    return MappingProxyType(dict(value))


def _validate_sensor_alignment(
    instance: "AlignedFrame",
    _attribute: object,
    value: Mapping[str, SensorData],
) -> None:
    """Validate that each sensor payload matches the aligned frame timeline."""
    expected_len = len(instance.align_timestamps_ns)
    for sensor_id, data in value.items():
        if len(data.align_timestamps_ns) != expected_len:
            msg = (
                f"sensor {sensor_id!r} align_timestamps_ns length {len(data.align_timestamps_ns)} "
                f"!= aligned frame length {expected_len}"
            )
            raise ValueError(msg)
        if not np.array_equal(data.align_timestamps_ns, instance.align_timestamps_ns):
            msg = f"sensor {sensor_id!r} align_timestamps_ns must exactly match aligned frame align_timestamps_ns"
            raise ValueError(msg)
        if len(data.sensor_timestamps_ns) != expected_len:
            msg = (
                f"sensor {sensor_id!r} sensor_timestamps_ns length "
                f"{len(data.sensor_timestamps_ns)} != aligned frame length {expected_len}"
            )
            raise ValueError(msg)


@attrs.define(hash=False, frozen=True)
class AlignedFrame:
    """Single bundle of sensor data aligned to a reference timeline.

    Attributes:
        align_timestamps_ns: shared 1-D alignment timestamps (ns); each sensor's
            ``align_timestamps_ns`` must match this array exactly
        sensor_data: a mapping of sensor data by sensor id

    """

    __hash__ = None  # type: ignore[assignment]

    align_timestamps_ns: npt.NDArray[np.int64] = attrs.field(
        converter=as_readonly_view,
        validator=strictly_increasing_int64_array,
    )
    sensor_data: Mapping[str, SensorData] = attrs.field(
        converter=_as_immutable_sensor_data,
        validator=_validate_sensor_alignment,
    )

    def __getitem__(self, key: str) -> SensorData:
        """Get sensor data by key."""
        return self.sensor_data[key]

    def __contains__(self, key: object) -> bool:
        """Check if sensor data exists by key."""
        return key in self.sensor_data
