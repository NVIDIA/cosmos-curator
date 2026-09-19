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

"""Nearest-pose alignment over one decoded ego-trajectory clip."""

from typing import Any

import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.alignment.nearest import nearest_timestamp_indices, timestamp_distance_ns
from cosmos_curator.core.sensors.data.egotrajectory_data import EgoTrajectory
from cosmos_curator.core.sensors.exceptions import AlignmentError, AlignmentFailureReason
from cosmos_curator.core.sensors.sampling.policy import NearestTimestampPolicy, require_nearest_timestamp_policy
from cosmos_curator.core.sensors.utils.validation import require_1d, require_strictly_increasing


def _select_optional(
    values: npt.NDArray[Any] | None,
    indices: npt.NDArray[np.int64],
) -> npt.NDArray[Any] | None:
    """Select rows from an optional ego-trajectory column."""
    return None if values is None else values[indices]


def align_ego_trajectory_nearest(
    source: EgoTrajectory,
    requested_timestamps_ns: npt.NDArray[np.int64],
    *,
    policy: NearestTimestampPolicy,
) -> EgoTrajectory:
    """Select one nearest source pose per requested timestamp from one clip.

    ``source`` is the complete decoded trajectory for one already-split clip.
    Its mapper-produced ``align_timestamps_ns`` and ``requested_timestamps_ns``
    must use the same alignment clock. This function does not inspect MCAP
    metadata directly; a mapper may derive ``align_timestamps_ns`` from MCAP
    ``log_time``.

    Source rows may be reused. Exact midpoint ties choose the earlier source
    row. A selected pose is copied unchanged: position, orientation, validity,
    and optional source metadata are never interpolated or otherwise derived.

    A clip the splitter wrote holds its half-open interior plus the pose
    immediately before it, and every source row stays eligible, that one
    included. So a selected ``sensor_timestamps_ns`` may lie before the clip's
    span: it is source provenance, not a span-boundary marker. No pose after the
    span is carried, so near the end of a clip this returns the nearest pose the
    clip holds even where the unsplit session had a nearer one just beyond it.
    """
    policy = require_nearest_timestamp_policy(policy, sensor_name="align_ego_trajectory_nearest")

    require_1d("requested_timestamps_ns", requested_timestamps_ns, np.int64)
    require_strictly_increasing("requested_timestamps_ns", requested_timestamps_ns)

    if len(requested_timestamps_ns) and not len(source.align_timestamps_ns):
        msg = "no ego-trajectory source pose is available for the requested timestamps"
        raise AlignmentError(
            AlignmentFailureReason.EGO_TRAJECTORY_NO_ELIGIBLE_POSE,
            msg,
            align_timestamps_ns=requested_timestamps_ns[[0]],
            known_source_timestamps_ns=source.align_timestamps_ns,
            max_delta_ns=policy.max_delta_ns,
        )

    if len(requested_timestamps_ns):
        indices = nearest_timestamp_indices(source.align_timestamps_ns, requested_timestamps_ns)
    else:
        indices = np.empty(0, dtype=np.int64)

    selected_align_timestamps_ns = source.align_timestamps_ns[indices]
    deltas_ns = timestamp_distance_ns(requested_timestamps_ns, selected_align_timestamps_ns)
    if policy.max_delta_ns is not None and len(deltas_ns):
        worst_index = int(deltas_ns.argmax())
        worst_delta_ns = int(deltas_ns[worst_index])
        if worst_delta_ns > policy.max_delta_ns:
            msg = (
                f"max_delta_ns={policy.max_delta_ns} exceeded: max delta was {worst_delta_ns} ns "
                f"for requested={int(requested_timestamps_ns[worst_index])}, "
                f"source={int(selected_align_timestamps_ns[worst_index])}"
            )
            raise AlignmentError(
                AlignmentFailureReason.EGO_TRAJECTORY_MAX_DELTA_EXCEEDED,
                msg,
                align_timestamps_ns=requested_timestamps_ns[[worst_index]],
                sensor_timestamps_ns=source.sensor_timestamps_ns[indices[[worst_index]]],
                known_source_timestamps_ns=selected_align_timestamps_ns[[worst_index]],
                delta_ns=worst_delta_ns,
                max_delta_ns=policy.max_delta_ns,
            )

    return EgoTrajectory(
        align_timestamps_ns=np.array(requested_timestamps_ns, copy=True),
        sensor_timestamps_ns=source.sensor_timestamps_ns[indices],
        positions_m=source.positions_m[indices],
        orientations_xyzw=source.orientations_xyzw[indices],
        pose_valid=source.pose_valid[indices],
        host_timestamps_ns=_select_optional(source.host_timestamps_ns, indices),
        sequence_counter=_select_optional(source.sequence_counter, indices),
        frame=source.frame,
    )
