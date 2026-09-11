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

"""Tests for clip-local nearest ego-trajectory alignment."""

import numpy as np
import pytest

from cosmos_curator.core.sensors.alignment.ego_trajectory import align_ego_trajectory_nearest
from cosmos_curator.core.sensors.data.egotrajectory_data import EgoTrajectory
from cosmos_curator.core.sensors.exceptions import AlignmentError, AlignmentFailureReason
from cosmos_curator.core.sensors.sampling.policy import NearestTimestampPolicy


def _source_trajectory() -> EgoTrajectory:
    """Build distinct source poses with separate alignment and device clocks."""
    half_sqrt = float(np.sqrt(0.5))
    return EgoTrajectory(
        align_timestamps_ns=np.array([100, 200, 400], dtype=np.int64),
        sensor_timestamps_ns=np.array([1_100, 1_200, 1_400], dtype=np.int64),
        positions_m=np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [7.0, 8.0, 9.0]], dtype=np.float64),
        orientations_xyzw=np.array(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, half_sqrt, half_sqrt]],
            dtype=np.float64,
        ),
        pose_valid=np.array([True, False, True], dtype=np.bool_),
        host_timestamps_ns=np.array([2_100, 2_200, 2_400], dtype=np.int64),
        sequence_counter=np.array([10, 20, 40], dtype=np.uint64),
        frame="world_enu",
    )


def _uniform_trajectory(align_timestamps_ns: np.ndarray) -> EgoTrajectory:
    """Build distinct valid poses at the supplied alignment timestamps."""
    count = len(align_timestamps_ns)
    return EgoTrajectory(
        align_timestamps_ns=align_timestamps_ns,
        sensor_timestamps_ns=align_timestamps_ns,
        positions_m=np.column_stack(
            [
                align_timestamps_ns.astype(np.float64),
                np.zeros(count, dtype=np.float64),
                np.zeros(count, dtype=np.float64),
            ]
        ),
        orientations_xyzw=np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64), (count, 1)),
        pose_valid=np.ones(count, dtype=np.bool_),
        host_timestamps_ns=align_timestamps_ns + 100,
        sequence_counter=align_timestamps_ns.astype(np.uint64),
        frame="world_enu",
    )


def test_nearest_on_a_clip_with_halo_poses_matches_the_unsplit_session() -> None:
    """Two halo poses make nearest selection independent of the clip boundary."""
    session_timestamps_ns = np.arange(0, 10_001, 1_000, dtype=np.int64)
    clip_start_ns, clip_exclusive_end_ns = 3_200, 6_900
    requested_timestamps_ns = np.arange(clip_start_ns, clip_exclusive_end_ns, 250, dtype=np.int64)

    in_span = session_timestamps_ns[
        (session_timestamps_ns >= clip_start_ns) & (session_timestamps_ns < clip_exclusive_end_ns)
    ]
    predecessor = session_timestamps_ns[session_timestamps_ns < clip_start_ns][-1:]
    successor = session_timestamps_ns[session_timestamps_ns >= clip_exclusive_end_ns][:1]
    policy = NearestTimestampPolicy()

    session = align_ego_trajectory_nearest(
        _uniform_trajectory(session_timestamps_ns), requested_timestamps_ns, policy=policy
    )
    with_halo = align_ego_trajectory_nearest(
        _uniform_trajectory(np.concatenate([predecessor, in_span, successor])),
        requested_timestamps_ns,
        policy=policy,
    )
    without_halo = align_ego_trajectory_nearest(_uniform_trajectory(in_span), requested_timestamps_ns, policy=policy)

    np.testing.assert_array_equal(with_halo.sensor_timestamps_ns, session.sensor_timestamps_ns)
    np.testing.assert_array_equal(with_halo.positions_m, session.positions_m)
    np.testing.assert_array_equal(with_halo.orientations_xyzw, session.orientations_xyzw)
    np.testing.assert_array_equal(with_halo.pose_valid, session.pose_valid)
    assert with_halo.host_timestamps_ns is not None
    assert session.host_timestamps_ns is not None
    np.testing.assert_array_equal(with_halo.host_timestamps_ns, session.host_timestamps_ns)
    assert with_halo.sequence_counter is not None
    assert session.sequence_counter is not None
    np.testing.assert_array_equal(with_halo.sequence_counter, session.sequence_counter)
    assert not np.array_equal(without_halo.sensor_timestamps_ns, session.sensor_timestamps_ns)
    assert np.any(with_halo.sensor_timestamps_ns < clip_start_ns)
    assert np.any(with_halo.sensor_timestamps_ns >= clip_exclusive_end_ns)


def test_nearest_selects_exact_earlier_later_ties_and_reused_rows() -> None:
    """One clip-local source pose can satisfy several requested alignment rows."""
    result = align_ego_trajectory_nearest(
        _source_trajectory(),
        np.array([100, 150, 190, 205, 350], dtype=np.int64),
        policy=NearestTimestampPolicy(),
    )

    np.testing.assert_array_equal(result.align_timestamps_ns, [100, 150, 190, 205, 350])
    np.testing.assert_array_equal(result.sensor_timestamps_ns, [1_100, 1_100, 1_200, 1_200, 1_400])
    np.testing.assert_allclose(
        result.positions_m,
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [7.0, 8.0, 9.0]],
    )
    np.testing.assert_array_equal(result.pose_valid, [True, True, False, False, True])
    assert result.host_timestamps_ns is not None
    np.testing.assert_array_equal(result.host_timestamps_ns, [2_100, 2_100, 2_200, 2_200, 2_400])
    assert result.sequence_counter is not None
    np.testing.assert_array_equal(result.sequence_counter, [10, 10, 20, 20, 40])
    assert result.frame == "world_enu"


def test_nearest_uses_mapped_alignment_time_without_mutating_device_timestamps() -> None:
    """Selection and output alignment use the mapper clock, not the device clock."""
    result = align_ego_trajectory_nearest(
        _source_trajectory(),
        np.array([400], dtype=np.int64),
        policy=NearestTimestampPolicy(),
    )

    half_sqrt = float(np.sqrt(0.5))
    np.testing.assert_array_equal(result.align_timestamps_ns, [400])
    np.testing.assert_array_equal(result.sensor_timestamps_ns, [1_400])
    np.testing.assert_allclose(result.orientations_xyzw, [[0.0, 0.0, half_sqrt, half_sqrt]])


def test_nearest_rejects_an_exceeded_maximum_delta_without_partial_output() -> None:
    """Tolerance errors report the failed request and selected source support."""
    with pytest.raises(AlignmentError) as error_info:
        align_ego_trajectory_nearest(
            _source_trajectory(),
            np.array([100, 260], dtype=np.int64),
            policy=NearestTimestampPolicy(max_delta_ns=50),
        )

    error = error_info.value
    assert error.reason is AlignmentFailureReason.EGO_TRAJECTORY_MAX_DELTA_EXCEEDED
    assert error.delta_ns == 60
    assert error.max_delta_ns == 50
    np.testing.assert_array_equal(error.align_timestamps_ns, [260])
    np.testing.assert_array_equal(error.sensor_timestamps_ns, [1_200])
    np.testing.assert_array_equal(error.known_source_timestamps_ns, [200])


def test_nearest_none_maximum_delta_leaves_selection_unbounded() -> None:
    """None disables only the nearest-distance limit."""
    result = align_ego_trajectory_nearest(
        _source_trajectory(),
        np.array([1_000], dtype=np.int64),
        policy=NearestTimestampPolicy(max_delta_ns=None),
    )

    np.testing.assert_array_equal(result.sensor_timestamps_ns, [1_400])


def test_nearest_zero_maximum_delta_requires_an_exact_match() -> None:
    """A zero nearest tolerance permits exact source alignment timestamps only."""
    exact = align_ego_trajectory_nearest(
        _source_trajectory(),
        np.array([100], dtype=np.int64),
        policy=NearestTimestampPolicy(max_delta_ns=0),
    )
    np.testing.assert_array_equal(exact.sensor_timestamps_ns, [1_100])

    with pytest.raises(AlignmentError) as error_info:
        align_ego_trajectory_nearest(
            _source_trajectory(),
            np.array([101], dtype=np.int64),
            policy=NearestTimestampPolicy(max_delta_ns=0),
        )

    error = error_info.value
    assert error.reason is AlignmentFailureReason.EGO_TRAJECTORY_MAX_DELTA_EXCEEDED
    assert error.delta_ns == 1
    assert error.max_delta_ns == 0


def test_nearest_rejects_nonempty_request_without_clip_source_poses() -> None:
    """A clip without poses cannot satisfy a nonempty alignment request."""
    source = EgoTrajectory(
        align_timestamps_ns=np.empty(0, dtype=np.int64),
        sensor_timestamps_ns=np.empty(0, dtype=np.int64),
        positions_m=np.empty((0, 3), dtype=np.float64),
        orientations_xyzw=np.empty((0, 4), dtype=np.float64),
        pose_valid=np.empty(0, dtype=np.bool_),
        frame="world_enu",
    )

    with pytest.raises(AlignmentError, match="no ego-trajectory source pose") as error_info:
        align_ego_trajectory_nearest(source, np.array([100], dtype=np.int64), policy=NearestTimestampPolicy())

    assert error_info.value.reason is AlignmentFailureReason.EGO_TRAJECTORY_NO_ELIGIBLE_POSE


def test_nearest_returns_a_well_shaped_empty_batch_for_an_empty_request() -> None:
    """An empty request is a valid window with no source rows to select."""
    result = align_ego_trajectory_nearest(
        _source_trajectory(),
        np.empty(0, dtype=np.int64),
        policy=NearestTimestampPolicy(),
    )

    np.testing.assert_array_equal(result.align_timestamps_ns, [])
    np.testing.assert_array_equal(result.sensor_timestamps_ns, [])
    assert result.positions_m.shape == (0, 3)
    assert result.orientations_xyzw.shape == (0, 4)
    assert result.host_timestamps_ns is not None
    assert result.host_timestamps_ns.shape == (0,)
    assert result.sequence_counter is not None
    assert result.sequence_counter.shape == (0,)
    assert result.frame == "world_enu"


def test_nearest_rejects_a_bare_policy() -> None:
    """The pure operation requires the explicit nearest-timestamp policy."""
    with pytest.raises(TypeError, match="NearestTimestampPolicy"):
        align_ego_trajectory_nearest(
            _source_trajectory(),
            np.array([100], dtype=np.int64),
            policy=None,  # type: ignore[arg-type]
        )
