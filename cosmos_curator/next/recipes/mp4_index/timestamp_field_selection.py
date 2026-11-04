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

"""Selecting which camera timestamp field a consumer of ``mp4-index`` records reads, shared across recipes.

``multimodal-split`` and ``data-integrity`` both let a config name an ordered preference of
camera timestamp fields and take the first one an indexed record fully populates. The
vocabulary and that selection rule live here so the two recipes cannot drift apart. What
each recipe does with the chosen timeline -- reject a non-increasing one, or measure it --
stays with the recipe.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np

from cosmos_curator.core.sensors.data.mp4 import Mp4FrameIndex
from cosmos_curator.core.sensors.types.types import VideoFrameAlignmentField

#: Camera timestamp fields a config may prefer. Sensor clocks (EOF/SOF/MOE) come from an
#: index extension; container PTS is read from the index or the media. ``dts_ns`` is
#: deliberately absent: decode order is always increasing, so it cannot describe timing.
CameraAlignmentField = Literal["eof_ns", "moe_ns", "sof_ns", "pts_ns", "edited_pts_ns"]

#: Fields only an ``mp4-index`` record can supply, because they come from vendor boxes the
#: media readers do not decode.
INDEX_ONLY_ALIGNMENT_FIELDS: frozenset[str] = frozenset({"sof_ns", "moe_ns"})


def first_complete_alignment_field(
    index: Mp4FrameIndex, alignment_fields: Sequence[str]
) -> VideoFrameAlignmentField | None:
    """Return the first field in *alignment_fields* that *index* populates for every frame.

    ``None`` when none is complete. A partly populated column is skipped rather than used,
    since a timeline with holes would be aligned by position and mislabel every frame after
    the first gap.
    """
    for field in alignment_fields:
        timestamps = getattr(index, field)
        if timestamps is None or (isinstance(timestamps, np.ma.MaskedArray) and np.any(timestamps.mask)):
            continue
        return VideoFrameAlignmentField(field)
    return None
