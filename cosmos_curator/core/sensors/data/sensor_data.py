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

"""Base sensor data structures for cosmos_curator.core.sensors package."""

from typing import Protocol

import numpy as np
import numpy.typing as npt


class SensorData(Protocol):
    """Minimum interface for all sensor data structures.

    Attributes:
        align_timestamps_ns: 1-D alignment timeline (ns) each row is aligned to;
            same length as ``sensor_timestamps_ns``
        sensor_timestamps_ns: 1-D sensor-reported times (ns); may differ from
            ``align_timestamps_ns`` due to resampling/grid alignment

    """

    align_timestamps_ns: npt.NDArray[np.int64]
    sensor_timestamps_ns: npt.NDArray[np.int64]
