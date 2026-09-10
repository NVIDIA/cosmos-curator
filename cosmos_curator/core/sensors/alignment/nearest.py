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

"""Generic timestamp nearest-neighbour operations for sensor alignment."""

import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.utils.validation import require_strictly_increasing

# Flips the sign bit, which maps int64 onto uint64 in the same order: INT64_MIN
# to 0 and INT64_MAX to 2**64 - 1.
_SIGN_BIT = np.uint64(0x8000000000000000)


def timestamp_distance_ns(left: npt.NDArray[np.int64], right: npt.NDArray[np.int64]) -> npt.NDArray[np.uint64]:
    """Return ``|left - right|`` elementwise, exactly, however far apart they are.

    Subtracting int64 wraps once two timestamps are more than int64 apart, which
    makes the furthest candidate look adjacent and selects it. Ordering survives
    the move to uint64, and a difference between two uint64 values always fits in
    one, so the answer is exact without leaving numpy for Python integers -- which
    measured 18x slower on a window and 98x on a timeline.
    """
    unsigned_left = np.ascontiguousarray(left).view(np.uint64) ^ _SIGN_BIT
    unsigned_right = np.ascontiguousarray(right).view(np.uint64) ^ _SIGN_BIT
    return np.where(unsigned_left > unsigned_right, unsigned_left - unsigned_right, unsigned_right - unsigned_left)


def nearest_timestamp_indices(
    source_timestamps_ns: npt.NDArray[np.int64],
    requested_timestamps_ns: npt.NDArray[np.int64],
    *,
    source_name: str = "source_timestamps_ns",
    requested_name: str = "requested_timestamps_ns",
) -> npt.NDArray[np.int64]:
    """Return the nearest source-timestamp index for every requested timestamp.

    Both timestamp arrays must be strictly increasing. Midpoint ties select the
    earlier source timestamp. This low-level operation has no sensor, clip,
    sampling-window, or policy semantics; callers apply their own locality and
    tolerance rules around it.
    """
    if len(source_timestamps_ns) == 0:
        msg = f"{source_name} must be non-empty"
        raise ValueError(msg)
    if len(requested_timestamps_ns) == 0:
        msg = f"{requested_name} must be non-empty"
        raise ValueError(msg)

    require_strictly_increasing(source_name, source_timestamps_ns)
    require_strictly_increasing(requested_name, requested_timestamps_ns)

    if len(source_timestamps_ns) == 1:
        return np.zeros_like(requested_timestamps_ns, dtype=np.int64)

    right_indices = np.searchsorted(source_timestamps_ns, requested_timestamps_ns)
    right_indices = np.clip(right_indices, 1, len(source_timestamps_ns) - 1)
    nearest_indices = right_indices - 1

    left_timestamps_ns = source_timestamps_ns[nearest_indices]
    right_timestamps_ns = source_timestamps_ns[right_indices]
    right_is_closer = timestamp_distance_ns(requested_timestamps_ns, right_timestamps_ns) < timestamp_distance_ns(
        requested_timestamps_ns, left_timestamps_ns
    )
    nearest_indices[right_is_closer] = right_indices[right_is_closer]
    return nearest_indices.astype(np.int64)
