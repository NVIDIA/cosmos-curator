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

"""Choose and validate indexed camera timelines for multimodal splitting."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.data.mp4 import Mp4FrameIndex, Mp4Metadata
from cosmos_curator.core.sensors.sensors.streaming_camera_sensor import StreamingCameraSensor
from cosmos_curator.core.sensors.utils.validation import require_1d, require_strictly_increasing
from cosmos_curator.next.recipes.mp4_index.timestamp_field_selection import first_complete_alignment_field


def indexed_timeline(
    index: Mp4FrameIndex, metadata: Mp4Metadata, *, alignment_fields: Sequence[str]
) -> tuple[npt.NDArray[np.int64], str]:
    """Select the first complete column; invalid available timestamps remain errors."""
    if len(index) == 0:
        msg = "mp4-index record has no frames"
        raise ValueError(msg)
    alignment_field = first_complete_alignment_field(index, alignment_fields)
    if alignment_field is None:
        msg = f"mp4-index record has no complete camera timestamp column from {list(alignment_fields)}"
        raise ValueError(msg)
    sensor = StreamingCameraSensor(metadata=metadata, frame_index=index, alignment_field=alignment_field)
    timeline = sensor.timestamps_ns
    validate_timeline(timeline, frame_count=len(index))
    return timeline, alignment_field.value


def validate_timeline(timeline: npt.NDArray[np.int64], *, frame_count: int) -> None:
    """Require one increasing timeline value per indexed frame."""
    require_1d("camera timeline", timeline, np.int64)
    require_strictly_increasing("camera timeline", timeline)
    if len(timeline) != frame_count:
        msg = f"camera timeline has {len(timeline)} timestamps for {frame_count} indexed frames"
        raise ValueError(msg)
