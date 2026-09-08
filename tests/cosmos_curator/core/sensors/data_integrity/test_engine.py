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

"""Unit tests for the session-grain half of the engine.

The per-stream half needs an open sensor and is exercised through the recipe's
fixtures; ``run_session_metrics`` needs nothing but numbers, so it is tested here.
"""

import pytest

from cosmos_curator.core.sensors.data_integrity.engine import run_session_metrics
from cosmos_curator.core.sensors.data_integrity.instruments import (
    NAME_SENSOR_OVERLAP,
    NAME_SENSOR_SPREAD,
    SESSION_INSTRUMENTS,
    Thresholds,
)
from cosmos_curator.core.sensors.data_integrity.results import (
    CheckStatus,
    SensorBounds,
    SessionTimestampSource,
    session_timestamp_source,
)
from cosmos_curator.core.sensors.types.types import TimestampSource


def _capture(*pairs: tuple[int, int]) -> list[SensorBounds]:
    """Build bounds measured on the rigs' own capture clocks.

    Stated at every call site that means to measure, because that is the only
    provenance these metrics run over at all: on container PTS a mux has already put
    every camera on one regenerated grid, and the numbers below would describe the
    encoder. Tests of the *skip* say so with :func:`_container` instead.
    """
    return [SensorBounds(start_ns=start, end_ns=end, timestamp_source=TimestampSource.CAPTURE) for start, end in pairs]


def _container(*pairs: tuple[int, int]) -> list[SensorBounds]:
    """Build bounds measured on container PTS -- what a session without sidecars looks like."""
    return [
        SensorBounds(start_ns=start, end_ns=end, timestamp_source=TimestampSource.CONTAINER_PTS) for start, end in pairs
    ]


@pytest.mark.parametrize(
    ("sources", "expected"),
    [
        ([], None),
        ([TimestampSource.CAPTURE], SessionTimestampSource.CAPTURE),
        ([TimestampSource.CAPTURE, TimestampSource.CAPTURE], SessionTimestampSource.CAPTURE),
        ([TimestampSource.CONTAINER_PTS], SessionTimestampSource.CONTAINER_PTS),
        ([TimestampSource.CAPTURE, TimestampSource.CONTAINER_PTS], SessionTimestampSource.MIXED),
        ([TimestampSource.CONTAINER_PTS, TimestampSource.CAPTURE], SessionTimestampSource.MIXED),
    ],
)
def test_a_sessions_clock_is_its_sensors_clock_only_when_they_all_agree(
    sources: list[TimestampSource], expected: SessionTimestampSource | None
) -> None:
    """The all-or-nothing rule, in the one place both the engine and the store read it from.

    Tested directly because two callers depend on it agreeing with itself: the kernel
    deciding whether to measure, and the store recording why it did not.
    """
    assert session_timestamp_source(sources) is expected


def _by_name(bounds: list[SensorBounds], **kwargs: object) -> dict[str, CheckStatus]:
    thresholds = Thresholds(**kwargs)  # type: ignore[arg-type]
    return {result.name: result.status for result in run_session_metrics(bounds, thresholds=thresholds)}


def test_every_session_metric_is_judged_in_registry_order() -> None:
    """The report's order is the registry's, so a reader sees the same metrics in the same places."""
    results = run_session_metrics(_capture((0, 1_000), (100, 1_100)))
    assert [result.name for result in results] == [spec.name for spec in SESSION_INSTRUMENTS]


def test_an_aligned_session_passes_both_metrics() -> None:
    """The rig every check is measured against: same window, same start, same stop."""
    assert _by_name(_capture((0, 1_000_000_000), (0, 1_000_000_000))) == {
        NAME_SENSOR_SPREAD: CheckStatus.PASS,
        NAME_SENSOR_OVERLAP: CheckStatus.PASS,
    }


def test_a_sensor_that_started_late_fails_the_spread_it_exceeds() -> None:
    """One camera two seconds behind the rest is what the spread threshold is for."""
    late = _capture((0, 10_000_000_000), (2_000_000_000, 10_000_000_000))
    assert _by_name(late, max_sensor_spread_ns=1_000_000_000)[NAME_SENSOR_SPREAD] is CheckStatus.FAIL
    assert _by_name(late, max_sensor_spread_ns=3_000_000_000)[NAME_SENSOR_SPREAD] is CheckStatus.PASS


def test_overlap_is_judged_as_the_share_of_the_session_that_is_not_shared() -> None:
    """Stated as the complement because a spec can only ask for a value to stay below a limit."""
    # 10% of the span has one sensor missing: 1s of a 10s session.
    ragged = _capture((0, 10_000_000_000), (1_000_000_000, 10_000_000_000))
    assert _by_name(ragged, max_non_overlap_percent=5.0)[NAME_SENSOR_OVERLAP] is CheckStatus.FAIL
    assert _by_name(ragged, max_non_overlap_percent=20.0)[NAME_SENSOR_OVERLAP] is CheckStatus.PASS


@pytest.mark.parametrize("bounds", [[], _capture((0, 1_000))])
def test_a_session_with_nothing_to_compare_is_skipped_not_passed(bounds: list[SensorBounds]) -> None:
    """A single-sensor session must not read as a session whose sensors agree."""
    statuses = _by_name(bounds)
    assert set(statuses.values()) == {CheckStatus.SKIPPED}


def test_a_skipped_session_metric_says_how_many_sensors_it_had() -> None:
    """A bare "insufficient data" leaves an operator guessing; the count is the whole explanation."""
    (spread, _) = run_session_metrics(_capture((0, 1_000)))
    assert "num_sensors=1" in spread.reason


def test_bounds_are_taken_in_any_order() -> None:
    """A session runner reports streams as they finish, so the engine cannot require sorting them."""
    forward = run_session_metrics(_capture((0, 1_000), (300, 1_400), (150, 1_250)))
    backward = run_session_metrics(_capture((150, 1_250), (300, 1_400), (0, 1_000)))
    assert [(r.name, r.status, r.measurement) for r in forward] == [(r.name, r.status, r.measurement) for r in backward]


def test_a_session_on_container_pts_is_not_measured_at_all() -> None:
    """The skip this gate exists for, and why it has to be a skip rather than a pass.

    These bounds are perfectly aligned, so on the numbers alone both metrics would
    report a flawless rig. That is exactly the trap: a constant-frame-rate mux puts
    every camera on one regenerated grid, so the alignment is the encoder's and says
    nothing about what the cameras did.
    """
    results = run_session_metrics(_container((0, 1_000_000_000), (0, 1_000_000_000)))

    assert {result.status for result in results} == {CheckStatus.SKIPPED}
    # No measurement, not a measurement judged unusable: a stored zero spread is what
    # a later re-judge would turn into a PASS under any threshold at all.
    assert {result.measurement for result in results} == {None}
    assert all("container PTS" in result.reason for result in results)


def test_a_session_whose_sensors_disagree_on_their_clock_is_not_measured_either() -> None:
    """One camera's real capture times against another's synthesized grid is not a comparison.

    Mixed is not a partial ``capture``: the spread it reports is the offset between two
    unrelated clocks, which is an artifact of the mux in the same way an all-PTS
    session's zero is.
    """
    mixed = [*_capture((0, 1_000_000_000)), *_container((0, 1_000_000_000))]
    results = run_session_metrics(mixed)

    assert {result.status for result in results} == {CheckStatus.SKIPPED}
    assert {result.measurement for result in results} == {None}
    assert all("disagree" in result.reason for result in results)


def test_a_session_with_no_bounds_at_all_says_so_rather_than_naming_a_clock() -> None:
    """An empty or unlisted session was measured on no clock, which is not a bad clock."""
    (spread, _) = run_session_metrics([])
    assert spread.status is CheckStatus.SKIPPED
    assert "no sensor reported a clock" in spread.reason


def test_which_clock_a_lone_camera_was_on_decides_which_skip_it_gets() -> None:
    """The precedence, which is the whole reason the two skips are told apart.

    One camera is one camera either way, but the findings differ: on capture times the
    session *was* measured and had nothing to compare, so its numbers are stored and a
    re-judge rebuilds them. On container PTS it was never measurable, so there is
    nothing to store and a re-judge must not invent it.
    """
    (measured, _) = run_session_metrics(_capture((0, 1_000)))
    (never_ran, _) = run_session_metrics(_container((0, 1_000)))

    assert measured.status is never_ran.status is CheckStatus.SKIPPED
    assert measured.measurement is not None, "one sensor is still a measurement, just an undefined one"
    assert "num_sensors=1" in measured.reason
    assert never_ran.measurement is None
    assert "capture timestamps" in never_ran.reason
