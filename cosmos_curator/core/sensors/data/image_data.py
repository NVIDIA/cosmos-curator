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

"""Image data structures for the sensor library."""

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.utils.helpers import as_readonly_view
from cosmos_curator.core.sensors.utils.validation import (
    nondecreasing_int64_array,
    strictly_increasing_int64_array,
    uint8_frame_batch,
)

_RGB_CHANNELS = 3


@attrs.define(hash=False, frozen=True)
class ImageMetadata:
    """Geometry and format properties of sampled still images."""

    height: int = attrs.field(validator=attrs.validators.gt(0))
    width: int = attrs.field(validator=attrs.validators.gt(0))
    image_format: str | None = None


@attrs.define(hash=False, frozen=True)
class ImageData:
    """Static images aligned to a reference timestamp grid."""

    __hash__ = None  # type: ignore[assignment]

    align_timestamps_ns: npt.NDArray[np.int64] = attrs.field(
        converter=as_readonly_view,
        validator=strictly_increasing_int64_array,
    )
    sensor_timestamps_ns: npt.NDArray[np.int64] = attrs.field(
        converter=as_readonly_view,
        validator=nondecreasing_int64_array,
    )
    frames: npt.NDArray[np.uint8] = attrs.field(
        converter=as_readonly_view,
        validator=uint8_frame_batch,
    )
    metadata: ImageMetadata

    @classmethod
    def from_frames(
        cls,
        frames: npt.NDArray[np.uint8],
        metadata: ImageMetadata,
        align_timestamps_ns: npt.NDArray[np.int64] | None = None,
    ) -> "ImageData":
        """Construct ``ImageData`` from frames, synthesizing timestamps when absent."""
        n_frames = len(frames)
        resolved_align_timestamps = (
            np.arange(n_frames, dtype=np.int64)
            if align_timestamps_ns is None
            else np.array(align_timestamps_ns, dtype=np.int64, copy=True)
        )
        return cls(
            align_timestamps_ns=resolved_align_timestamps,
            sensor_timestamps_ns=np.array(resolved_align_timestamps, copy=True),
            frames=frames,
            metadata=metadata,
        )

    def __attrs_post_init__(self) -> None:
        """Validate cross-field invariants."""
        expected_shape = (self.metadata.height, self.metadata.width, _RGB_CHANNELS)
        if self.frames.shape[1:] != expected_shape:
            msg = f"frames must have shape (N, {expected_shape[0]}, {expected_shape[1]}, 3), got {self.frames.shape}"
            raise ValueError(msg)

        if not (len(self.align_timestamps_ns) == len(self.sensor_timestamps_ns) == len(self.frames)):
            msg = (
                "All arrays must be the same length: "
                f"align_timestamps_ns={len(self.align_timestamps_ns)} "
                f"sensor_timestamps_ns={len(self.sensor_timestamps_ns)} "
                f"frames={len(self.frames)}"
            )
            raise ValueError(msg)
