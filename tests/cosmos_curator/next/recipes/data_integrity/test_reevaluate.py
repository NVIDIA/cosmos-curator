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

"""Unit tests for re-judging stored measurements under a new policy.

This is the requirement the whole store exists to satisfy: change a threshold, get
new verdicts, open no source data.
"""

from collections.abc import Callable
from fractions import Fraction

import pytest

from cosmos_curator.core.sensors.data_integrity import identity
from cosmos_curator.core.sensors.data_integrity.engine import run_session_metrics
from cosmos_curator.core.sensors.data_integrity.instruments import (
    DEFAULT_THRESHOLDS,
    SESSION_INSTRUMENTS,
    Thresholds,
)
from cosmos_curator.core.sensors.data_integrity.results import StreamResult
from cosmos_curator.next.recipes.data_integrity import reevaluate, store

NO_HEADER_RATE = Fraction(0, 1)
SESSION = "/data/session"
DRIFT = f"{SESSION}/drift.mp4"

#: Tight enough to fail a 1% drift that the default 5% policy passes.
STRICT = Thresholds(max_rate_deviation_percent=0.1)

#: Tight enough to fail any skew at all, which the one-second default passes.
NO_SKEW = Thresholds(max_sensor_spread_ns=0)

_ONE_MINUTE_NS = 60_000_000_000
_SKEW_NS = 1_000_000


def _rate_verdicts(root: str) -> dict[str, dict[str, object]]:
    """Every stored rate verdict, keyed by the policy that produced it."""
    return {str(row["policy_id"]): row for row in store.read_evaluations(root) if row["metric_name"] == "rate"}


def _spread_verdicts(root: str) -> dict[str, dict[str, object]]:
    """Every stored sensor-spread verdict, keyed by the policy that produced it."""
    spread = SESSION_INSTRUMENTS[0].name
    return {str(row["policy_id"]): row for row in store.read_session_evaluations(root) if row["metric_name"] == spread}


@pytest.fixture
def measured(
    store_root: str, make_stream: Callable[..., StreamResult], drifting: Callable[..., list[int]]
) -> tuple[str, str]:
    """Build a store holding one 1%-drifting stream, measured and judged under the defaults."""
    run_id = store.write_run(
        store_root,
        [make_stream(DRIFT, drifting(percent=1.0))],
        session_path=SESSION,
        thresholds=DEFAULT_THRESHOLDS,
        tool="di-session",
    )
    return store_root, run_id


@pytest.fixture
def session_measured(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> tuple[str, str]:
    """Build a store holding two slightly-skewed sensors, with session metrics measured and judged."""
    streams = [make_stream(DRIFT, perfect()), make_stream(f"{SESSION}/rear.mp4", perfect())]
    run_id = store.write_run(
        store_root,
        streams,
        session_path=SESSION,
        thresholds=DEFAULT_THRESHOLDS,
        tool="di-session",
        session_metrics=run_session_metrics(
            [(0, _ONE_MINUTE_NS), (_SKEW_NS, _ONE_MINUTE_NS + _SKEW_NS)], thresholds=DEFAULT_THRESHOLDS
        ),
    )
    return store_root, run_id


def test_tightening_a_threshold_flips_the_verdict(measured: tuple[str, str]) -> None:
    """The headline case: same facts, stricter policy, different answer."""
    root, _ = measured
    assert _rate_verdicts(root)[store.policy_id(DEFAULT_THRESHOLDS)]["check_status"] == "PASS"

    reevaluate.reevaluate(root, thresholds=STRICT)
    assert _rate_verdicts(root)[store.policy_id(STRICT)]["check_status"] == "FAIL"


def test_re_evaluation_opens_no_source(measured: tuple[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    """No I/O against the data itself -- the sources in this store do not even exist."""

    def _explode(*_args: object, **_kwargs: object) -> None:
        pytest.fail("re-evaluation must not open a source")

    monkeypatch.setattr("cosmos_curator.next.recipes.data_integrity.sources.open_source", _explode)
    monkeypatch.setattr("cosmos_curator.next.recipes.data_integrity.sources.run_checks", _explode)

    root, _ = measured
    reevaluate.reevaluate(root, thresholds=STRICT)
    assert store.policy_id(STRICT) in _rate_verdicts(root)


def test_a_re_judge_commits_like_any_other_run(measured: tuple[str, str]) -> None:
    """Uncommitted rows are invisible, so a re-judge that skipped the commit would vanish."""
    root, measurement_run = measured
    rejudge_run = reevaluate.reevaluate(root, thresholds=STRICT)

    assert store.completed_runs(root) == frozenset({measurement_run, rejudge_run})
    (row,) = [entry for entry in store.read_runs(root) if entry["run_id"] == rejudge_run]
    assert row["tool"] == "di-reevaluate"
    assert row["policy_id"] == store.policy_id(STRICT)
    # A re-judge touches no stream; it re-reads facts that another run measured.
    assert row["num_streams"] is None


def test_a_re_judge_ignores_measurements_that_never_committed(
    measured: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    make_stream: Callable[..., StreamResult],
    drifting: Callable[..., list[int]],
) -> None:
    """Facts from a torn write are not facts, so they must not acquire a fresh verdict."""
    root, measurement_run = measured

    def _boom(*_args: object, **_kwargs: object) -> None:
        msg = "killed before the commit"
        raise RuntimeError(msg)

    monkeypatch.setattr(store, "commit_run", _boom)
    with pytest.raises(RuntimeError, match="killed before the commit"):
        store.write_run(
            root,
            [make_stream("/data/session/torn.mp4", drifting(percent=1.0))],
            session_path="/data/session",
            thresholds=DEFAULT_THRESHOLDS,
            tool="di-session",
        )

    monkeypatch.undo()
    reevaluate.reevaluate(root, thresholds=STRICT)
    rejudged = [row for row in store.read_evaluations(root) if row["policy_id"] == store.policy_id(STRICT)]
    assert {row["measurement_run_id"] for row in rejudged} == {measurement_run}


def test_both_generations_of_verdict_survive(measured: tuple[str, str]) -> None:
    """A re-judge adds rows beside the originals; nothing is superseded."""
    root, _ = measured
    reevaluate.reevaluate(root, thresholds=STRICT)
    policies = _rate_verdicts(root)
    assert set(policies) == {store.policy_id(DEFAULT_THRESHOLDS), store.policy_id(STRICT)}


def test_re_judged_rows_point_back_at_the_measuring_run(measured: tuple[str, str]) -> None:
    """`run_id <> measurement_run_id` is what isolates re-evaluated verdicts."""
    root, measurement_run = measured
    rejudge_run = reevaluate.reevaluate(root, thresholds=STRICT)

    original = _rate_verdicts(root)[store.policy_id(DEFAULT_THRESHOLDS)]
    rejudged = _rate_verdicts(root)[store.policy_id(STRICT)]

    assert original["run_id"] == original["measurement_run_id"] == measurement_run
    assert rejudged["run_id"] == rejudge_run
    assert rejudged["measurement_run_id"] == measurement_run
    assert rejudged["run_id"] != rejudged["measurement_run_id"]


def test_re_judged_row_keeps_the_measuring_instrument_version(measured: tuple[str, str]) -> None:
    """The version records which code produced the *input*; a re-judge does not re-measure."""
    root, _ = measured
    (stored,) = store.read_measurements(root, "rate")
    reevaluate.reevaluate(root, thresholds=STRICT)
    assert _rate_verdicts(root)[store.policy_id(STRICT)]["instrument_version"] == stored["instrument_version"]


def test_re_judging_under_the_same_policy_supersedes_itself(measured: tuple[str, str]) -> None:
    """Identical thresholds collide on purpose, so a repeat does not fork the history."""
    root, _ = measured
    rerun = reevaluate.reevaluate(root, thresholds=DEFAULT_THRESHOLDS)
    verdicts = _rate_verdicts(root)
    assert set(verdicts) == {store.policy_id(DEFAULT_THRESHOLDS)}
    assert verdicts[store.policy_id(DEFAULT_THRESHOLDS)]["run_id"] == rerun


def test_every_metric_is_re_judged(measured: tuple[str, str]) -> None:
    """Not just the one whose threshold moved: a policy is judged as a whole."""
    root, _ = measured
    reevaluate.reevaluate(root, thresholds=STRICT)
    rejudged = {
        row["metric_name"] for row in store.read_evaluations(root) if row["policy_id"] == store.policy_id(STRICT)
    }
    assert rejudged == {
        "timestamp_ordering",
        "rate",
        "timestamp_gap",
        "jitter",
        "frame_reordering_present",
    }


def test_a_metric_that_never_ran_stays_skipped(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """A null row has no measurement to rebuild, and no threshold can change that."""
    stream = make_stream("/data/session/no_rate.mp4", perfect(), expected_hz=None, avg_frame_rate=NO_HEADER_RATE)
    store.write_run(
        store_root, [stream], session_path="/data/session", thresholds=DEFAULT_THRESHOLDS, tool="di-session"
    )
    reevaluate.reevaluate(store_root, thresholds=STRICT)

    rejudged = _rate_verdicts(store_root)[store.policy_id(STRICT)]
    assert rejudged["check_status"] == "SKIPPED"
    assert rejudged["margin"] is None
    assert rejudged["threshold"] is None


def test_an_undefined_measurement_stays_skipped(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """A rebuilt-but-undefined measurement is skipped for the same reason it was the first time."""
    store.write_run(
        store_root,
        [make_stream("/data/session/one_frame.mp4", perfect(1))],
        session_path="/data/session",
        thresholds=DEFAULT_THRESHOLDS,
        tool="di-session",
    )
    reevaluate.reevaluate(store_root, thresholds=STRICT)
    assert _rate_verdicts(store_root)[store.policy_id(STRICT)]["check_status"] == "SKIPPED"


def test_re_evaluating_an_errored_stream_produces_nothing(
    store_root: str, make_errored_stream: Callable[..., StreamResult]
) -> None:
    """An errored stream has no measurements, so there is nothing to re-judge."""
    store.write_run(
        store_root,
        [make_errored_stream("/data/session/broken.mp4")],
        session_path="/data/session",
        thresholds=DEFAULT_THRESHOLDS,
        tool="di-session",
    )
    reevaluate.reevaluate(store_root, thresholds=STRICT)
    assert store.read_evaluations(store_root) == []


def test_a_session_verdict_is_re_judged_too(session_measured: tuple[str, str]) -> None:
    """Session-grain facts are stored to be re-judged; a policy is judged as a whole."""
    root, _ = session_measured
    assert _spread_verdicts(root)[store.policy_id(DEFAULT_THRESHOLDS)]["check_status"] == "PASS"

    reevaluate.reevaluate(root, thresholds=NO_SKEW)
    rejudged = _spread_verdicts(root)[store.policy_id(NO_SKEW)]
    assert rejudged["check_status"] == "FAIL"
    assert rejudged["threshold"] == 0


def test_a_re_judged_session_verdict_points_back_at_the_measuring_run(session_measured: tuple[str, str]) -> None:
    """The provenance rule holds for session rows: they name the run that measured them."""
    root, measurement_run = session_measured
    rejudge_run = reevaluate.reevaluate(root, thresholds=NO_SKEW)
    rejudged = _spread_verdicts(root)[store.policy_id(NO_SKEW)]
    assert rejudged["run_id"] == rejudge_run
    assert rejudged["measurement_run_id"] == measurement_run
    assert rejudged["session_id"] == identity.session_id(SESSION)


def test_a_re_judged_session_verdict_moves_the_rollup(session_measured: tuple[str, str]) -> None:
    """The point of storing the verdict: a re-judge changes what the store says about the session."""
    root, _ = session_measured
    reevaluate.reevaluate(root, thresholds=NO_SKEW)
    assert store.session_rollup(root, policy=store.policy_id(DEFAULT_THRESHOLDS))["status"] == "PASS"
    assert store.session_rollup(root, policy=store.policy_id(NO_SKEW))["status"] == "FAIL"


def test_every_session_metric_is_re_judged(session_measured: tuple[str, str]) -> None:
    """Every session metric that was measured gets a fresh verdict, not just the one that moved."""
    root, _ = session_measured
    reevaluate.reevaluate(root, thresholds=NO_SKEW)
    rejudged = {
        row["metric_name"]
        for row in store.read_session_evaluations(root)
        if row["policy_id"] == store.policy_id(NO_SKEW)
    }
    assert rejudged == {spec.name for spec in SESSION_INSTRUMENTS}


def test_an_undefined_session_measurement_stays_skipped(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """A session metric needs two sensors, and no policy can supply the second one.

    The session-grain twin of :func:`test_an_undefined_measurement_stays_skipped`, and
    the shape a single-sensor session has: the facts are stored and rebuilt, they are
    simply undefined, so the re-judge has to leave the verdict ``SKIPPED`` with nothing
    to compare rather than invent a threshold and a margin for a measurement that has
    no value. The "never ran" shape below it -- ``is_defined`` null -- no session metric
    can reach, neither of them needing an expected rate.
    """
    store.write_run(
        store_root,
        [make_stream(DRIFT, perfect())],
        session_path=SESSION,
        thresholds=DEFAULT_THRESHOLDS,
        tool="di-session",
        session_metrics=run_session_metrics([(0, _ONE_MINUTE_NS)], thresholds=DEFAULT_THRESHOLDS),
    )
    reevaluate.reevaluate(store_root, thresholds=NO_SKEW)

    rejudged = _spread_verdicts(store_root)[store.policy_id(NO_SKEW)]
    assert rejudged["check_status"] == "SKIPPED"
    assert rejudged["margin"] is None
    assert rejudged["threshold"] is None
    # Every session metric, not just the one this test reads back by name.
    assert {
        row["check_status"]
        for row in store.read_session_evaluations(store_root)
        if row["policy_id"] == store.policy_id(NO_SKEW)
    } == {"SKIPPED"}


def test_a_store_holding_no_session_measurements_re_judges_none(measured: tuple[str, str]) -> None:
    """The per-stream loop must not invent session verdicts out of an older store."""
    root, _ = measured
    reevaluate.reevaluate(root, thresholds=NO_SKEW)
    assert store.read_session_evaluations(root) == []


def test_rollup_reflects_the_policy_it_is_asked_about(measured: tuple[str, str]) -> None:
    """Two verdict generations coexist, so a rollup has to say which one it means."""
    root, _ = measured
    reevaluate.reevaluate(root, thresholds=STRICT)
    assert store.session_rollup(root, policy=store.policy_id(DEFAULT_THRESHOLDS))["status"] == "PASS"
    assert store.session_rollup(root, policy=store.policy_id(STRICT))["status"] == "FAIL"
