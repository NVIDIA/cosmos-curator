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
"""Single-pass camera payload for the forward-only streaming sensor."""

from collections.abc import Generator

import attrs
import numpy as np
import numpy.typing as npt

# One aligned row: the reference timestamp being served, the presentation
# timestamp of the source frame chosen for it, and that frame's pixels.
type StreamingFrame = tuple[int, int, npt.NDArray[np.uint8]]


@attrs.define
class StreamingCameraData:
    """One window of camera data, delivered a frame at a time.

    Pixels are decoded lazily as :meth:`frames` is iterated, without collecting
    the window's frames. Payloads from one sampling call share a forward decode
    pass and must be consumed in order.

    Attributes:
        align_timestamps_ns: the reference timestamps this window asked for.
        sensor_timestamps_ns: the source observation serving each reference.

    """

    align_timestamps_ns: npt.NDArray[np.int64] = attrs.field()
    sensor_timestamps_ns: npt.NDArray[np.int64] = attrs.field()
    _walk: Generator[StreamingFrame] = attrs.field(alias="walk")
    _consumed: bool = attrs.field(init=False, default=False)

    def frames(self) -> Generator[StreamingFrame]:
        """Decode this window, yielding ``(align_timestamp_ns, pts_ns, frame)`` per reference.

        The frame in each row is only valid until the generator advances, and
        the same array comes back when one source frame serves several
        consecutive references. Treat it as read-only, and copy it to keep it.

        Returns:
            A generator yielding one row per reference timestamp, in ascending
            ``align_timestamp_ns`` order.

        Raises:
            RuntimeError: If this window was already decoded, or if a later
                window from the same ``sample()`` call has been. One decode is
                shared across every window and only moves forwards, so it can
                neither restart nor go back; ask the sensor for a new one.

        """
        if self._consumed:
            msg = "StreamingCameraData is single-use and has already been consumed"
            raise RuntimeError(msg)
        self._consumed = True
        return self._walk
