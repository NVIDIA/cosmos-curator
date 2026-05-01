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
"""Types for the sensor library."""

import enum
import io
from pathlib import Path

import numpy as np
import numpy.typing as npt

type DataSource = Path | str | io.BufferedIOBase | bytes | npt.NDArray[np.uint8]


class VideoIndexCreationMethod(enum.Enum):
    """How packet-level metadata is collected when building a video index.

    ``FROM_HEADER`` reads the stream's index entries parsed from the container
    header (fast). ``FULL_DEMUX`` walks every packet via demux (slow, I/O heavy).

    Prefer ``FROM_HEADER`` in production. Reserve ``FULL_DEMUX`` for tests or
    when header-only metadata is proven insufficient for a format (please file
    an issue in that case).
    """

    FROM_HEADER = "from_header"
    FULL_DEMUX = "full_demux"
