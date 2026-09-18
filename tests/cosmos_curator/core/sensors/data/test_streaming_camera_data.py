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
"""Tests for the single-pass streaming camera payload."""

from collections.abc import Generator

import numpy as np
import pytest

from cosmos_curator.core.sensors.data.streaming_camera_data import StreamingCameraData, StreamingFrame


def _stream() -> Generator[StreamingFrame]:
    for k in range(3):
        yield k * 100, k * 100 + 1, np.full((2, 2, 3), k, dtype=np.uint8)


def _payload() -> StreamingCameraData:
    """Return a payload over ``_stream``, carrying the timelines that walk will serve."""
    return StreamingCameraData(
        align_timestamps_ns=np.array([0, 100, 200], dtype=np.int64),
        sensor_timestamps_ns=np.array([1, 101, 201], dtype=np.int64),
        walk=_stream(),
    )


def test_frames_yields_the_underlying_walk() -> None:
    """``frames()`` yields ``(align_timestamp_ns, pts_ns, frame)``."""
    data = _payload()

    rows = [(align_ns, pts_ns, int(frame[0, 0, 0])) for align_ns, pts_ns, frame in data.frames()]

    assert rows == [(0, 1, 0), (100, 101, 1), (200, 201, 2)]


def test_second_walk_raises() -> None:
    """The stream is forward-only, so it can only be walked once."""
    data = _payload()
    for _ in data.frames():
        break

    with pytest.raises(RuntimeError, match="already been consumed"):
        data.frames()
