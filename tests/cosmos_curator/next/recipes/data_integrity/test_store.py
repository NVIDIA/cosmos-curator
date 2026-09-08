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

"""Unit tests for the Lance-backed data-integrity store."""

import ast
import datetime
import json
import math
import pathlib
from collections.abc import Callable
from fractions import Fraction

import pyarrow as pa
import pytest

from cosmos_curator.core.sensors.data_integrity import identity
from cosmos_curator.core.sensors.data_integrity.engine import run_session_metrics
from cosmos_curator.core.sensors.data_integrity.instruments import (
    DEFAULT_THRESHOLDS,
    INSTRUMENTS,
    SESSION_INSTRUMENTS,
    Thresholds,
    instrument,
)
from cosmos_curator.core.sensors.data_integrity.results import (
    CheckResult,
    SensorBounds,
    SessionTimestampSource,
    StreamResult,
)
from cosmos_curator.core.sensors.types.types import TimestampSource
from cosmos_curator.core.utils.storage.storage_client import StorageStat
from cosmos_curator.core.utils.storage_cli import StorageCliError
from cosmos_curator.next.recipes.data_integrity import storage_io, store, store_schema

#: The zero-numerator sentinel for "the container declares no rate", which is what
#: makes rate / gap / jitter never run at all.
NO_HEADER_RATE = Fraction(0, 1)

SESSION = "/data/session"
FRONT = "/data/session/front.mp4"


def _write(
    root: str,
    streams: list[StreamResult],
    *,
    session_path: str | None = SESSION,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    session_metrics: list[CheckResult] | None = None,
    created_at: datetime.datetime | None = None,
    storage_stats: dict[str, StorageStat] | None = None,
) -> str:
    """Write one session run with the defaults most of these tests want."""
    return store.write_run(
        root,
        streams,
        session_path=session_path,
        thresholds=thresholds,
        tool="di-session",
        session_metrics=session_metrics,
        created_at=created_at,
        storage_stats=storage_stats,
    )


#: One second of skew between two sensors, which is what makes a session-grain
#: measurement have something to say.
_SKEW_NS = 1_000_000_000
_ONE_MINUTE_NS = 60_000_000_000

#: Write stamps for tests that need one run to supersede another. Spelled out rather
#: than taken from the clock so which row wins is a property of the test, not of how
#: fast it ran, and distinct so no two rows ever tie on ``created_at``.
_EARLIER = datetime.datetime(2029, 1, 1, tzinfo=datetime.UTC)
_LATER = datetime.datetime(2030, 1, 1, tzinfo=datetime.UTC)
_LATEST = datetime.datetime(2031, 1, 1, tzinfo=datetime.UTC)


#: The spread metric's registered name, read off the registry rather than spelled out.
SPREAD = SESSION_INSTRUMENTS[0].name


def _metrics_for(
    bounds: list[tuple[int, int]],
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    *,
    source: TimestampSource = TimestampSource.CAPTURE,
) -> list[CheckResult]:
    """Judge one session from its sensors' recording bounds, through the real kernel.

    Straight to :func:`run_session_metrics` rather than through ``session_runner``,
    both because the store sits below it and because bounds stated outright are how a
    test says which session shape it is about.

    ``source`` defaults to ``CAPTURE`` because a store test wants rows with numbers in
    them, and only a session measured on capture times produces any: the two session
    metrics do not run at all on container PTS. Tests of *that* pass it explicitly.
    """
    return run_session_metrics(
        [SensorBounds(start_ns=start, end_ns=end, timestamp_source=source) for start, end in bounds],
        thresholds=thresholds,
    )


def _aligned_metrics(thresholds: Thresholds = DEFAULT_THRESHOLDS) -> list[CheckResult]:
    """Two sensors that started and stopped together: every session metric passes."""
    return _metrics_for([(0, _ONE_MINUTE_NS), (0, _ONE_MINUTE_NS)], thresholds)


def _skewed_metrics(thresholds: Thresholds = DEFAULT_THRESHOLDS) -> list[CheckResult]:
    """Two sensors a second out of step at both ends."""
    return _metrics_for([(0, _ONE_MINUTE_NS), (_SKEW_NS, _ONE_MINUTE_NS + _SKEW_NS)], thresholds)


def _append_session(
    root: str,
    session_path: str,
    *,
    num_streams: int,
    num_unreadable: int = 0,
    error: str | None = None,
    created_at: datetime.datetime = _LATEST,
) -> str:
    """Commit one hand-built ``session.lance`` row, carrying no stream rows of its own.

    :func:`write_run` derives ``num_streams`` from the streams it is handed, so it
    cannot express the two shapes the Ray pipeline produces and the rollup has to
    grade: a session found to hold more streams than the run could reach, and one that
    could not be listed at all. Both are built here directly and committed the way any
    run's rows are, since an uncommitted row is not read.
    """
    run_id = store.new_run_id()
    store.append_rows(
        [
            store.session_row(
                session_id=store.session_key(session_path),
                run_id=run_id,
                created_at=created_at,
                session_path=session_path,
                num_streams=num_streams,
                num_unreadable=num_unreadable,
                error=error,
            )
        ],
        store.join(root, store_schema.SESSION_DATASET),
        store_schema.SESSION_SCHEMA,
        None,
    )
    store.commit_run(
        root,
        run_id=run_id,
        created_at=created_at,
        tool="data-integrity",
        session_path=None,
        thresholds=DEFAULT_THRESHOLDS,
        num_streams=num_streams,
        storage_options=None,
    )
    return run_id


#: Modules the storage layer sits *below*. It shares the result vocabulary with them
#: (:mod:`results`) rather than importing them, so a store write cannot end up
#: depending on how a CLI happens to render or parse anything.
_CLI_MODULES = frozenset(
    {
        "cosmos_curator.next.recipes.data_integrity.cli",
        "cosmos_curator.next.recipes.data_integrity.cli_support",
        "cosmos_curator.next.recipes.data_integrity.session_cli",
        "cosmos_curator.next.recipes.data_integrity.session_runner",
        "cosmos_curator.next.recipes.data_integrity.report",
        "cosmos_curator.next.recipes.data_integrity.sources",
    }
)

#: The two packages the modules under test are spread across: the store and its schema
#: live with the recipe, the vocabulary they serialise with the sensor library.
_MODULE_DIRS = (pathlib.Path(store.__file__).parent, pathlib.Path(identity.__file__).parent)


def _module_source(module: str) -> str:
    """Read a data-integrity module's source from whichever of the two packages holds it."""
    for directory in _MODULE_DIRS:
        path = directory / f"{module}.py"
        if path.exists():
            return path.read_text()
    msg = f"no such data-integrity module: {module}"
    raise AssertionError(msg)


@pytest.mark.parametrize("module", ["store", "store_schema", "reevaluate", "results", "instruments", "identity"])
def test_the_storage_layer_does_not_import_the_cli(module: str) -> None:
    """Layering, checked rather than remembered: results flow up to the CLIs, never down."""
    imported = {
        node.module
        for node in ast.walk(ast.parse(_module_source(module)))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert imported & _CLI_MODULES == set()


def test_the_two_verdict_schemas_agree_on_every_shared_column() -> None:
    """A stream verdict and a session verdict differ only in what they are a verdict *about*.

    ``store._verdict_columns`` fills every column below from one place, so the two
    schemas have to agree about them: if they drift, a row that builder produces fits
    only one of the two tables. What may legitimately differ is the subject --
    ``stream_id`` and ``source`` against ``session_id`` -- and ``session_path``, which
    only a per-stream verdict can leave null, ``di-check`` judging a stream that
    belongs to no session.
    """
    subject = {"stream_id", "source", "session_id", "session_path"}

    def shared(schema: pa.Schema) -> list[tuple[str, pa.DataType, bool]]:
        return [(f.name, f.type, f.nullable) for f in schema if f.name not in subject]

    assert shared(store_schema.EVALUATION_SCHEMA), "the shared block cannot be empty"
    assert shared(store_schema.EVALUATION_SCHEMA) == shared(store_schema.SESSION_EVALUATION_SCHEMA)


def test_first_write_creates_the_store(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """The root does not have to exist; a first run creates every dataset it needs."""
    _write(store_root, [make_stream(FRONT, perfect())], session_metrics=_aligned_metrics())
    root = pathlib.Path(store_root)
    assert (root / store_schema.STREAM_DATASET).exists()
    assert (root / store_schema.EVALUATION_DATASET).exists()
    assert (root / store_schema.RUN_DATASET).exists()
    for spec in INSTRUMENTS:
        assert (root / store_schema.metric_dataset_path(spec.name)).exists()
    assert (root / store_schema.SESSION_DATASET).exists()
    assert (root / store_schema.SESSION_EVALUATION_DATASET).exists()
    for spec in SESSION_INSTRUMENTS:
        assert (root / store_schema.session_metric_dataset_path(spec.name)).exists()
    assert (root / store_schema.MANIFEST_NAME).exists()


def test_stream_row_carries_identity_and_session_context(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """The dedup key, the descriptive columns, and the session context all land together."""
    _write(store_root, [make_stream(FRONT, perfect())])
    (row,) = store.read_streams(store_root)
    assert row["stream_id"] == identity.stream_id(FRONT)
    assert row["source"] == FRONT
    assert row["relative_key"] == "front.mp4"
    assert row["session_id"] == identity.session_id(SESSION)
    assert row["locator_namespace"] == identity.NAMESPACE_LOCAL
    assert row["selector_type"] == identity.SELECTOR_VIDEO_STREAM
    assert row["selector_value"] == "0"


@pytest.mark.parametrize("source", list(TimestampSource))
def test_a_stream_row_records_the_clock_it_was_measured_on(
    source: TimestampSource,
    store_root: str,
    make_stream: Callable[..., StreamResult],
    perfect: Callable[..., list[int]],
) -> None:
    """Provenance is persisted, not re-derived: the row keeps the numbers, not the sidecar.

    Without this column a stored jitter of nearly zero is unreadable -- a rig that is
    genuinely steady and a constant-frame-rate mux that invented the timeline both
    look like that, and only one of them is a measurement.
    """
    _write(store_root, [make_stream(FRONT, perfect(), timestamp_source=source)])
    (row,) = store.read_streams(store_root)
    assert row["timestamp_source"] == source.value


def test_an_errored_stream_row_claims_no_clock(
    store_root: str, make_errored_stream: Callable[..., StreamResult]
) -> None:
    """Null, like every other sensor-level column on a stream that was never read."""
    _write(store_root, [make_errored_stream("/data/session/rear.mp4")])
    (row,) = store.read_streams(store_root)
    assert row["timestamp_source"] is None


def test_two_stream_indices_of_one_file_are_two_streams(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """``--stream-idx 0`` and ``--stream-idx 1`` measured different bytes, so neither may win."""
    _write(store_root, [make_stream(FRONT, perfect(), selector_value="0")])
    _write(store_root, [make_stream(FRONT, perfect(), selector_value="1")])

    rows = {str(row["selector_value"]): row for row in store.read_streams(store_root)}
    assert set(rows) == {"0", "1"}
    assert rows["0"]["stream_id"] != rows["1"]["stream_id"]
    # ...and the metric rows follow the same key rather than collapsing onto one.
    assert len({row["stream_id"] for row in store.read_measurements(store_root, "rate")}) == 2


def test_single_stream_run_has_no_session(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """A null session_path is the only structural difference between the two CLIs' rows."""
    _write(store_root, [make_stream("/data/front.mp4", perfect())], session_path=None)
    (row,) = store.read_streams(store_root)
    assert row["session_path"] is None
    assert row["session_id"] is None
    assert row["relative_key"] is None


def test_errored_stream_is_recorded_with_no_measurements(
    store_root: str, make_errored_stream: Callable[..., StreamResult]
) -> None:
    """The case the store exists for: a stream that could not be opened must not vanish."""
    _write(store_root, [make_errored_stream("/data/session/rear.mp4")])
    (row,) = store.read_streams(store_root)
    assert row["error"] == "moov atom not found"
    assert row["codec_name"] is None
    for spec in INSTRUMENTS:
        assert store.read_measurements(store_root, spec.name) == []
    assert store.read_evaluations(store_root) == []


def test_measured_but_undefined_keeps_its_numbers(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """``is_defined`` False still has real values to save, NaN included where one was produced."""
    _write(store_root, [make_stream("/data/session/one_frame.mp4", perfect(1))])
    (rate,) = store.read_measurements(store_root, "rate")
    assert rate["is_defined"] is False
    # NaN survives as a float64 bit pattern; it is emphatically not null.
    assert rate["period_deviation_percent"] is not None
    assert math.isnan(rate["period_deviation_percent"])  # type: ignore[arg-type]
    # The other metrics carry real values even when undefined, so is_defined False
    # does not imply NaN.
    (ordering,) = store.read_measurements(store_root, "timestamp_ordering")
    assert ordering["is_defined"] is False
    assert ordering["num_samples"] == 1


def test_metric_that_never_ran_is_an_explicit_null_row(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """A metric considered but unable to run stays distinguishable from one not in the run."""
    # No user rate and no header rate, so rate / gap / jitter are never constructed.
    stream = make_stream("/data/session/no_rate.mp4", perfect(), expected_hz=None, avg_frame_rate=NO_HEADER_RATE)
    _write(store_root, [stream])

    (rate,) = store.read_measurements(store_root, "rate")
    assert rate["is_defined"] is None
    assert all(rate[name] is None for name in ("period_deviation_percent", "expected_hz", "num_samples"))

    # The row exists rather than being absent, and its verdict is SKIPPED.
    verdicts = {row["metric_name"]: row for row in store.read_evaluations(store_root)}
    assert verdicts["rate"]["check_status"] == "SKIPPED"
    assert verdicts["rate"]["margin"] is None
    assert verdicts["rate"]["threshold"] is None
    # Ordering does not need a rate, so it still ran.
    assert verdicts["timestamp_ordering"]["check_status"] == "PASS"


def test_verdict_carries_margin_threshold_and_both_run_ids(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """A stored verdict explains itself: what was applied, how far off, and whose facts."""
    run_id = _write(store_root, [make_stream(FRONT, perfect())])
    rate = next(row for row in store.read_evaluations(store_root) if row["metric_name"] == "rate")
    assert rate["run_id"] == run_id
    # Measured and judged in one go, so the two ids agree on a first pass.
    assert rate["measurement_run_id"] == run_id
    assert rate["threshold"] == DEFAULT_THRESHOLDS.max_rate_deviation_percent
    assert rate["margin"] is not None
    assert json.loads(str(rate["thresholds_json"]))["max_gaps"] == DEFAULT_THRESHOLDS.max_gaps


def test_failing_check_is_stored_as_fail(
    store_root: str, make_stream: Callable[..., StreamResult], drifting: Callable[..., list[int]]
) -> None:
    """A verdict outside the policy is FAIL with a negative margin."""
    thresholds = Thresholds(max_rate_deviation_percent=0.5)
    stream = make_stream("/data/session/drift.mp4", drifting(percent=3.0), thresholds=thresholds)
    _write(store_root, [stream], thresholds=thresholds)
    rate = next(row for row in store.read_evaluations(store_root) if row["metric_name"] == "rate")
    assert rate["check_status"] == "FAIL"
    assert float(rate["margin"]) < 0  # type: ignore[arg-type]


def test_re_running_supersedes_rather_than_duplicating(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """Append-only keeps history, but "current state" is one row per key."""
    first = _write(store_root, [make_stream(FRONT, perfect())])
    second = _write(
        store_root,
        [make_stream(FRONT, perfect())],
        created_at=datetime.datetime(2030, 1, 1, tzinfo=datetime.UTC),
    )
    assert first != second

    assert len(store.read_streams(store_root, latest=False)) == 2
    (latest,) = store.read_streams(store_root)
    assert latest["run_id"] == second

    assert len(store.read_measurements(store_root, "rate", latest=False)) == 2
    assert len(store.read_measurements(store_root, "rate")) == 1


def test_run_id_breaks_a_timestamp_tie_deterministically(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """Every row one run writes shares a timestamp, so timestamps alone give no total order."""
    at = datetime.datetime(2026, 5, 1, tzinfo=datetime.UTC)
    written = sorted(_write(store_root, [make_stream(FRONT, perfect())], created_at=at) for _ in range(2))
    (winner,) = store.read_streams(store_root)
    assert winner["run_id"] == written[-1]
    # Deterministic, not merely stable within one call.
    assert store.read_streams(store_root)[0]["run_id"] == winner["run_id"]


def test_instrument_version_is_not_part_of_the_measurement_key(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """Re-measuring with newer code supersedes the old row rather than sitting beside it."""
    _write(store_root, [make_stream(FRONT, perfect())])
    _write(
        store_root,
        [make_stream(FRONT, perfect())],
        created_at=datetime.datetime(2030, 1, 1, tzinfo=datetime.UTC),
    )
    assert len(store.read_measurements(store_root, "rate")) == 1


def _fail_the_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kill a run between its dataset writes and its commit, the way a crash would."""

    def _boom(*_args: object, **_kwargs: object) -> None:
        msg = "killed before the commit"
        raise RuntimeError(msg)

    monkeypatch.setattr(store, "commit_run", _boom)


def test_a_run_that_never_committed_is_not_readable(
    store_root: str,
    monkeypatch: pytest.MonkeyPatch,
    make_stream: Callable[..., StreamResult],
    perfect: Callable[..., list[int]],
) -> None:
    """A write spans several datasets, so rows without a commit describe a torn write."""
    _fail_the_commit(monkeypatch)
    with pytest.raises(RuntimeError, match="killed before the commit"):
        _write(store_root, [make_stream(FRONT, perfect())])

    assert store.read_streams(store_root) == []
    assert store.read_measurements(store_root, "rate") == []
    assert store.read_evaluations(store_root) == []
    # Still on disk, and reachable when the question is "what happened here".
    assert len(store.read_streams(store_root, include_incomplete=True)) == 1


def test_an_uncommitted_run_does_not_hide_the_committed_row_beneath_it(
    store_root: str,
    monkeypatch: pytest.MonkeyPatch,
    make_stream: Callable[..., StreamResult],
    perfect: Callable[..., list[int]],
) -> None:
    """The newest row wins only among committed ones, or a crash would blank the key."""
    good = _write(store_root, [make_stream(FRONT, perfect())])

    _fail_the_commit(monkeypatch)
    with pytest.raises(RuntimeError, match="killed before the commit"):
        _write(
            store_root,
            [make_stream(FRONT, perfect())],
            created_at=datetime.datetime(2030, 1, 1, tzinfo=datetime.UTC),
        )

    (row,) = store.read_streams(store_root)
    assert row["run_id"] == good


def test_the_run_ledger_records_every_finished_run(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """The commit row is the run's provenance: who wrote it, under what policy, and when."""
    run_id = _write(store_root, [make_stream(FRONT, perfect())])
    assert store.completed_runs(store_root) == frozenset({run_id})

    (row,) = store.read_runs(store_root)
    assert row["run_id"] == run_id
    assert row["tool"] == "di-session"
    assert row["session_path"] == SESSION
    assert row["policy_id"] == store.policy_id(DEFAULT_THRESHOLDS)
    assert row["num_streams"] == 1
    assert row["store_schema_version"] == store_schema.STORE_SCHEMA_VERSION
    assert row["committed_at"] >= row["created_at"]


def test_session_rollup_matches_the_live_report_precedence(
    store_root: str,
    make_stream: Callable[..., StreamResult],
    make_errored_stream: Callable[..., StreamResult],
    perfect: Callable[..., list[int]],
    drifting: Callable[..., list[int]],
) -> None:
    """ERROR outranks FAIL, because an errored stream was never measured at all."""
    thresholds = Thresholds(max_rate_deviation_percent=0.5)
    streams = [
        make_stream("/data/session/good.mp4", perfect(), thresholds=thresholds),
        make_stream("/data/session/drift.mp4", drifting(percent=3.0), thresholds=thresholds),
        make_errored_stream("/data/session/broken.mp4"),
    ]
    _write(store_root, streams, thresholds=thresholds)

    rollup = store.session_rollup(store_root)
    assert rollup["status"] == "ERROR"
    assert rollup["num_streams"] == 3
    assert rollup["stream_status_counts"] == {"PASS": 1, "FAIL": 1, "ERROR": 1}


def test_session_rollup_of_an_empty_store_is_an_error(store_root: str) -> None:
    """Nothing measured is not the same as nothing wrong."""
    _write(store_root, [])
    assert store.session_rollup(store_root)["status"] == "ERROR"


def test_session_rollup_under_a_policy_nothing_was_judged_by_is_an_error(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """A store with no verdicts for the requested policy has nothing to call a pass."""
    _write(store_root, [make_stream(FRONT, perfect())])
    stored = store.session_rollup(store_root, policy=store.policy_id(DEFAULT_THRESHOLDS))
    assert stored["status"] == "PASS"

    other = store.session_rollup(store_root, policy=store.policy_id(Thresholds(max_gaps=99)))
    assert other["status"] == "ERROR"
    assert other["stream_status_counts"] == {"PASS": 0, "FAIL": 0, "ERROR": 1}


def test_a_session_run_records_the_session_itself(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """The session's own row: its key, its context, and how much of it was usable."""
    streams = [make_stream(FRONT, perfect()), make_stream("/data/session/rear.mp4", perfect())]
    run_id = _write(store_root, streams, session_metrics=_aligned_metrics())

    (row,) = store.read_sessions(store_root)
    assert row["session_id"] == identity.session_id(SESSION)
    assert row["session_path"] == SESSION
    assert row["locator_namespace"] == identity.NAMESPACE_LOCAL
    assert row["run_id"] == run_id
    assert row["num_streams"] == 2
    assert row["num_unreadable"] == 0
    assert row["error"] is None


@pytest.mark.parametrize(
    ("sources", "expected"),
    [
        ([TimestampSource.CAPTURE, TimestampSource.CAPTURE], "capture"),
        ([TimestampSource.CONTAINER_PTS, TimestampSource.CONTAINER_PTS], "container_pts"),
        ([TimestampSource.CAPTURE, TimestampSource.CONTAINER_PTS], "mixed"),
    ],
)
def test_a_session_records_which_clock_its_streams_shared(
    store_root: str,
    make_stream: Callable[..., StreamResult],
    perfect: Callable[..., list[int]],
    sources: list[TimestampSource],
    expected: str,
) -> None:
    """The column that makes a skipped session-grain metric legible.

    A never-ran measurement row is all nulls and cannot say why it never ran, so the
    session's row carries the cause -- and it carries it as the roll-up of the streams
    beside it, not as an independent claim.
    """
    streams = [
        make_stream(f"/data/session/cam{i}.mp4", perfect(), timestamp_source=source) for i, source in enumerate(sources)
    ]
    _write(store_root, streams)

    (row,) = store.read_sessions(store_root)
    assert row["timestamp_source"] == expected


def test_a_session_no_stream_contributed_to_names_no_clock(
    store_root: str, make_errored_stream: Callable[..., StreamResult]
) -> None:
    """No clock is null, not a bad clock: an unreachable stream reported neither bounds nor provenance."""
    _write(store_root, [make_errored_stream("/data/session/broken.mp4")])

    (row,) = store.read_sessions(store_root)
    assert row["timestamp_source"] is None


def test_a_session_metric_that_never_ran_stores_null_columns(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """The shape that stops a mux's artifact from being re-judged into a pass.

    Two aligned sensors on container PTS: measured, they would store a zero spread
    that every threshold accepts. Stored as never-ran, there is nothing for
    ``di-reevaluate`` to rebuild.
    """
    streams = [make_stream(FRONT, perfect()), make_stream("/data/session/rear.mp4", perfect())]
    _write(
        store_root,
        streams,
        session_metrics=_metrics_for(
            [(0, _ONE_MINUTE_NS), (0, _ONE_MINUTE_NS)],
            source=TimestampSource.CONTAINER_PTS,
        ),
    )

    for spec in SESSION_INSTRUMENTS:
        (measurement,) = store.read_session_measurements(store_root, spec.name)
        assert measurement["is_defined"] is None
        assert all(measurement[name] is None for name in spec.field_names)

    verdicts = {row["metric_name"]: row for row in store.read_session_evaluations(store_root)}
    assert {row["check_status"] for row in verdicts.values()} == {"SKIPPED"}
    assert "container PTS" in str(verdicts[SPREAD]["reason"])
    # And the session's own row names the cause the metric rows cannot.
    (session,) = store.read_sessions(store_root)
    assert session["timestamp_source"] == "container_pts"


def test_a_row_from_a_newer_schema_is_refused_rather_than_half_written(store_root: str) -> None:
    """The whole enforcement of the non-additive version 2 to 3 bump, pinned.

    ``timestamp_source`` joined a schema that :func:`store.append_rows` writes with a
    plain ``mode="append"``, so a store laid down before it cannot take new rows and
    needs a fresh root -- a cost taken deliberately over migration machinery (see
    :data:`store_schema.STORE_SCHEMA_VERSION`). That trade only holds while the
    refusal is loud. A Lance that quietly dropped the column, or wrote the row into a
    partial file first, would leave an operator with a store whose clocks are absent
    rather than one that stopped and asked for a new root.

    Held on ``session.lance``. ``stream.lance`` gained the column through the same
    function and the same call, so one dataset pins the mechanism for both.

    The two runs differ only in ``run_id``, and that is what makes the aftermath
    legible: read newest-per-session, a row half-written under the same key would be
    collapsed into the row it was meant to supersede and the dataset would look
    untouched either way. Read as full history, a partial write has somewhere to show.
    """

    def _run(timestamp_source: SessionTimestampSource | None) -> dict[str, object]:
        return store.session_row(
            session_id=store.session_key(SESSION),
            run_id=store.new_run_id(),
            created_at=_LATEST,
            session_path=SESSION,
            num_streams=2,
            num_unreadable=0,
            timestamp_source=timestamp_source,
        )

    uri = store.join(store_root, store_schema.SESSION_DATASET)
    # ``from_pylist`` keeps only the fields the schema it is handed names, so writing a
    # row under a schema without ``timestamp_source`` lays the dataset down exactly as
    # version 2 left it -- no hand-built legacy row required.
    legacy = store_schema.SESSION_SCHEMA.remove(store_schema.SESSION_SCHEMA.get_field_index("timestamp_source"))
    settled = _run(None)
    store.append_rows([settled], uri, legacy, None)

    with pytest.raises(OSError, match="Append with different schema"):
        store.append_rows([_run(SessionTimestampSource.CAPTURE)], uri, store_schema.SESSION_SCHEMA, None)

    history = store.read_sessions(store_root, latest=False, include_incomplete=True)
    assert [row["run_id"] for row in history] == [settled["run_id"]], "the refused row reached the dataset anyway"
    assert "timestamp_source" not in history[0], "the append was refused, so the column cannot be there"


def test_a_single_stream_run_writes_no_session_rows(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """``di-check`` has no session to key a row on, so it must write none even if handed verdicts."""
    _write(
        store_root,
        [make_stream("/data/front.mp4", perfect())],
        session_path=None,
        session_metrics=_aligned_metrics(),
    )
    assert store.read_sessions(store_root) == []
    assert store.read_session_evaluations(store_root) == []
    assert not (pathlib.Path(store_root) / store_schema.SESSION_DATASET).exists()


def test_an_unreadable_stream_counts_against_its_session(
    store_root: str,
    make_stream: Callable[..., StreamResult],
    make_errored_stream: Callable[..., StreamResult],
    perfect: Callable[..., list[int]],
) -> None:
    """``num_unreadable`` is what makes a session's row legible without joining to its streams."""
    streams = [make_stream(FRONT, perfect()), make_errored_stream("/data/session/rear.mp4")]
    _write(store_root, streams, session_metrics=_aligned_metrics())
    (row,) = store.read_sessions(store_root)
    assert row["num_streams"] == 2
    assert row["num_unreadable"] == 1


def test_session_measurements_round_trip_as_typed_columns(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """A session measurement survives Arrow field for field, like a per-stream one."""
    _write(store_root, [make_stream(FRONT, perfect())], session_metrics=_skewed_metrics())

    (spread,) = store.read_session_measurements(store_root, "multi_sensor_spread")
    assert spread["session_id"] == identity.session_id(SESSION)
    assert spread["session_path"] == SESSION
    assert spread["is_defined"] is True
    assert spread["start_spread_ns"] == _SKEW_NS
    assert spread["num_sensors"] == 2
    # No source column, and deliberately: there is no one stream a session-grain
    # measurement is about.
    assert "source" not in spread

    (overlap,) = store.read_session_measurements(store_root, "multi_sensor_overlap")
    assert overlap["non_overlap_fraction"] is not None
    assert float(overlap["non_overlap_fraction"]) > 0  # type: ignore[arg-type]


def test_a_session_of_one_sensor_is_measured_and_undefined(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """Nothing to compare is a measurement with no verdict, not an absent measurement."""
    _write(store_root, [make_stream(FRONT, perfect())], session_metrics=_metrics_for([(0, 1_000_000_000)]))

    (spread,) = store.read_session_measurements(store_root, "multi_sensor_spread")
    assert spread["is_defined"] is False
    assert spread["num_sensors"] == 1

    verdict = next(row for row in store.read_session_evaluations(store_root) if row["metric_name"] == SPREAD)
    assert verdict["check_status"] == "SKIPPED"
    # Null together, which is what makes it impossible to store a verdict for
    # something that was never judged.
    assert verdict["margin"] is None
    assert verdict["threshold"] is None


def test_a_session_verdict_carries_its_policy_and_both_run_ids(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """A stored session verdict explains itself on the same terms a per-stream one does."""
    run_id = _write(store_root, [make_stream(FRONT, perfect())], session_metrics=_aligned_metrics())
    verdict = next(row for row in store.read_session_evaluations(store_root) if row["metric_name"] == SPREAD)
    assert verdict["session_id"] == identity.session_id(SESSION)
    assert verdict["run_id"] == run_id
    assert verdict["measurement_run_id"] == run_id
    assert verdict["policy_id"] == store.policy_id(DEFAULT_THRESHOLDS)
    assert verdict["check_status"] == "PASS"
    assert verdict["threshold"] == DEFAULT_THRESHOLDS.max_sensor_spread_ns


def test_re_running_a_session_supersedes_its_own_rows(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """The session tables dedup on ``session_id`` by the same rule the stream tables use."""
    first = _write(store_root, [make_stream(FRONT, perfect())], session_metrics=_aligned_metrics())
    second = _write(
        store_root,
        [make_stream(FRONT, perfect())],
        session_metrics=_skewed_metrics(),
        created_at=datetime.datetime(2030, 1, 1, tzinfo=datetime.UTC),
    )
    assert first != second

    assert len(store.read_sessions(store_root, latest=False)) == 2
    (latest,) = store.read_sessions(store_root)
    assert latest["run_id"] == second

    (spread,) = store.read_session_measurements(store_root, "multi_sensor_spread")
    assert spread["start_spread_ns"] == _SKEW_NS


def test_a_session_run_that_never_committed_is_not_readable(
    store_root: str,
    monkeypatch: pytest.MonkeyPatch,
    make_stream: Callable[..., StreamResult],
    perfect: Callable[..., list[int]],
) -> None:
    """Session rows are gated by the same commit marker as everything else."""
    _fail_the_commit(monkeypatch)
    with pytest.raises(RuntimeError, match="killed before the commit"):
        _write(store_root, [make_stream(FRONT, perfect())], session_metrics=_aligned_metrics())

    assert store.read_sessions(store_root) == []
    assert store.read_session_measurements(store_root, "multi_sensor_spread") == []
    assert store.read_session_evaluations(store_root) == []
    assert len(store.read_sessions(store_root, include_incomplete=True)) == 1


def test_session_rollup_fails_on_a_stored_session_verdict(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """The store-side twin of the bug the live report had: every stream passes, the session does not."""
    thresholds = Thresholds(max_sensor_spread_ns=0)
    streams = [make_stream(FRONT, perfect()), make_stream("/data/session/rear.mp4", perfect())]
    _write(store_root, streams, thresholds=thresholds, session_metrics=_skewed_metrics(thresholds))

    rollup = store.session_rollup(store_root)
    assert rollup["stream_status_counts"] == {"PASS": 2, "FAIL": 0, "ERROR": 0}
    assert rollup["session_status_counts"] == {"PASS": 0, "FAIL": 1, "ERROR": 0}
    assert rollup["status"] == "FAIL"


def test_session_rollup_errors_on_a_session_that_could_not_be_listed(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """A session never listed was never measured, which outranks any failure the way an errored stream does."""
    _write(store_root, [make_stream(FRONT, perfect())], session_metrics=_aligned_metrics())
    _append_session(store_root, "/data/gone", num_streams=0, error="prefix does not exist")

    rollup = store.session_rollup(store_root)
    assert rollup["session_status_counts"] == {"PASS": 1, "FAIL": 0, "ERROR": 1}
    assert rollup["status"] == "ERROR"


def test_session_rollup_errors_on_a_session_the_run_did_not_measure_through(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """A stream nobody could reach leaves no row, and must not leave its session reading PASS.

    The unreachable-stream case: the run found two streams, reached one, and wrote no
    row for the other, so nothing in ``stream.lance`` can force the session to ERROR.
    What does is ``num_streams`` -- what the session was found to *hold* -- against the
    rows the store actually has.
    """
    _write(store_root, [make_stream(FRONT, perfect())], created_at=_EARLIER, session_metrics=_aligned_metrics())
    _append_session(store_root, SESSION, num_streams=2)

    rollup = store.session_rollup(store_root)
    assert rollup["stream_status_counts"] == {"PASS": 1, "FAIL": 0, "ERROR": 0}
    assert rollup["session_status_counts"] == {"PASS": 0, "FAIL": 0, "ERROR": 1}
    assert rollup["status"] == "ERROR"


def test_a_session_two_runs_measured_between_them_rolls_up_pass(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """Counting rows rather than storing a shortfall is what lets a re-run close the gap.

    Coverage is a question about the store, not about any one run: the stream a first
    run could not reach and a second one could is measured either way, and the session
    completes without either run revising a column. A stored ``num_unreachable`` would
    have pinned this session at ``ERROR`` for good.
    """
    _write(store_root, [make_stream(FRONT, perfect())], created_at=_EARLIER, session_metrics=_aligned_metrics())
    _write(store_root, [make_stream("/data/session/rear.mp4", perfect())], created_at=_LATER)
    _append_session(store_root, SESSION, num_streams=2)

    rollup = store.session_rollup(store_root)
    assert rollup["stream_status_counts"] == {"PASS": 2, "FAIL": 0, "ERROR": 0}
    assert rollup["session_status_counts"] == {"PASS": 1, "FAIL": 0, "ERROR": 0}
    assert rollup["status"] == "PASS"


def test_re_measuring_one_stream_under_a_wider_session_strands_neither(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """Coverage counts every row a session ever had, not the rows a reader resolves to now.

    A ``stream_id`` is keyed on the source alone, so re-running at a different scope
    moves that stream's newest row to the wider session. Both runs are legal --
    ``_drop_nested`` only de-duplicates within one run -- and the earlier session was
    genuinely measured through, so counting the resolved rows would have left it at
    ``ERROR`` permanently on the strength of a re-run.
    """
    side = "/data/session/nested/side.mp4"
    _write(store_root, [make_stream(side, perfect())], session_path="/data/session/nested", created_at=_EARLIER)
    _write(
        store_root,
        [make_stream(FRONT, perfect()), make_stream(side, perfect())],
        created_at=_LATER,
        session_metrics=_aligned_metrics(),
    )

    rollup = store.session_rollup(store_root)
    assert rollup["stream_status_counts"] == {"PASS": 2, "FAIL": 0, "ERROR": 0}
    assert rollup["session_status_counts"] == {"PASS": 2, "FAIL": 0, "ERROR": 0}
    assert rollup["status"] == "PASS"


def test_session_rollup_errors_on_a_session_that_held_nothing(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """An empty session is graded the way a live report grades one: nothing measured, so ERROR.

    It costs the run nothing -- an empty session is a finding about the input, not a
    failure -- but the session's own verdict cannot be ``PASS`` when there was nothing
    to judge.
    """
    _write(store_root, [make_stream(FRONT, perfect())], created_at=_EARLIER)
    _append_session(store_root, "/data/empty", num_streams=0)

    rollup = store.session_rollup(store_root)
    assert rollup["session_status_counts"] == {"PASS": 1, "FAIL": 0, "ERROR": 1}
    assert rollup["status"] == "ERROR"


def test_session_rollup_of_a_store_without_session_rows_is_unchanged(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """A store written before these tables existed still rolls up, and to what it did before."""
    _write(store_root, [make_stream(FRONT, perfect())], session_path=None)
    rollup = store.session_rollup(store_root)
    assert rollup["status"] == "PASS"
    assert rollup["num_sessions"] == 0


def test_a_session_with_no_verdicts_is_not_an_error(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """Unlike a stream, a session row is written whether or not its writer had verdicts to offer."""
    _write(store_root, [make_stream(FRONT, perfect())])
    assert store.read_session_evaluations(store_root) == []
    rollup = store.session_rollup(store_root)
    assert rollup["session_status_counts"] == {"PASS": 1, "FAIL": 0, "ERROR": 0}
    assert rollup["status"] == "PASS"


def test_manifest_records_the_run(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """Provenance for the most recent run; per-run detail lives on the rows themselves."""
    run_id = _write(store_root, [make_stream(FRONT, perfect())])
    manifest = store.read_manifest(store_root)
    assert manifest["run_id"] == run_id
    assert manifest["tool"] == "di-session"
    assert manifest["store_schema_version"] == store_schema.STORE_SCHEMA_VERSION
    assert manifest["policy_id"] == store.policy_id(DEFAULT_THRESHOLDS)
    assert manifest["instruments"] == {spec.name: spec.version for spec in (*INSTRUMENTS, *SESSION_INSTRUMENTS)}
    datasets = manifest["datasets"]
    assert isinstance(datasets, dict)
    # Both grains, since a session run writes both.
    assert set(datasets["measurements"]) == {spec.name for spec in INSTRUMENTS}
    assert set(datasets["session_measurements"]) == {spec.name for spec in SESSION_INSTRUMENTS}


def test_a_caller_can_assemble_a_run_from_the_public_builders(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """What a Ray driver needs: rows built one at a time, then appended and committed.

    ``write_run`` gathers a whole session before writing anything, which a driver
    measuring thousands of streams across many sessions cannot do. The builders are
    therefore public API, and this is the sequence that uses them.
    """
    stream = make_stream(FRONT, perfect())
    created_at = datetime.datetime(2026, 8, 20, tzinfo=datetime.UTC)
    run_id = store.new_run_id()
    key = store.stream_key(stream)
    shared = {"stream_id": key, "run_id": run_id, "created_at": created_at, "session_path": SESSION}

    store.append_rows(
        [store.stream_row(stream, content=store.content_identity(FRONT), **shared)],
        store.join(store_root, store_schema.STREAM_DATASET),
        store_schema.STREAM_SCHEMA,
        None,
    )
    for check in stream.metrics:
        spec = instrument(check.name)
        store.append_rows(
            [store.measurement_row(spec, check, source=FRONT, **shared)],
            store.join(store_root, store_schema.metric_dataset_path(spec.name)),
            store_schema.MEASUREMENT_SCHEMAS[spec.name],
            None,
        )
        store.append_rows(
            [
                store.build_evaluation_row(
                    spec,
                    check,
                    source=FRONT,
                    measurement_run_id=run_id,
                    thresholds=DEFAULT_THRESHOLDS,
                    **shared,
                )
            ],
            store.join(store_root, store_schema.EVALUATION_DATASET),
            store_schema.EVALUATION_SCHEMA,
            None,
        )
    store.commit_run(
        store_root,
        run_id=run_id,
        created_at=created_at,
        tool="data-integrity",
        session_path=None,
        thresholds=DEFAULT_THRESHOLDS,
        num_streams=1,
        storage_options=None,
    )

    (row,) = store.read_streams(store_root)
    assert row["stream_id"] == key
    assert row["session_id"] == identity.session_id(SESSION)
    assert len(store.read_evaluations(store_root)) == len(INSTRUMENTS)
    assert [run["tool"] for run in store.read_runs(store_root)] == ["data-integrity"]


def test_policy_id_is_a_function_of_the_thresholds() -> None:
    """Identical policies collide on purpose; a changed one opens a new verdict generation."""
    assert store.policy_id(Thresholds()) == store.policy_id(Thresholds())
    assert store.policy_id(Thresholds()) != store.policy_id(Thresholds(max_gaps=1))


def test_local_content_identity_comes_from_a_stat(tmp_path: pathlib.Path) -> None:
    """Local files have a size and mtime but no ETag."""
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"x" * 1234)
    content = store.content_identity(str(path))
    assert content.size_bytes == 1234
    assert content.last_modified is not None
    assert content.etag is None


def test_content_identity_of_a_missing_file_is_empty(tmp_path: pathlib.Path) -> None:
    """Provenance is best-effort; an unreadable source must not fail a store write."""
    assert store.content_identity(str(tmp_path / "gone.mp4")) == store.ContentIdentity()


def test_storage_stats_are_reused_rather_than_refetched(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """The session CLI already issued the HEAD, so the store must not issue a second one.

    A second HEAD here would also fail outright, since nothing in this test can reach
    S3 -- which is precisely what makes the assertion meaningful.
    """
    source = "s3://bucket/session/front.mp4"
    stat = StorageStat(
        size_bytes=4096,
        etag="d41d8cd98f00b204e9800998ecf8427e",
        last_modified=datetime.datetime(2026, 3, 1, tzinfo=datetime.UTC),
    )
    _write(
        store_root,
        [make_stream(source, perfect())],
        session_path="s3://bucket/session",
        storage_stats={source: stat},
    )
    (row,) = store.read_streams(store_root)
    assert row["content_etag"] == stat.etag
    assert row["content_size_bytes"] == stat.size_bytes
    assert row["content_last_modified"] == stat.last_modified


def test_a_lookup_that_learned_nothing_is_not_mistaken_for_a_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed HEAD upstream must not be recorded as "this object has no ETag".

    ``storage_io.object_stat`` reports a lookup that learned nothing as ``None``, and
    the caller passes that on, so ``None`` has to mean "look again" rather than "there
    is nothing to know".
    """
    fetched: list[str] = []

    def _stat(source: str, **_kwargs: object) -> StorageStat:
        fetched.append(source)
        return StorageStat(size_bytes=7, etag="abc")

    monkeypatch.setattr(storage_io, "object_stat", _stat)
    source = "s3://bucket/session/front.mp4"
    content = store.content_identity(source, stat=None)
    assert fetched == [source]
    assert content.etag == "abc"


def test_staleness_is_flagged_when_the_bytes_change(
    store_root: str,
    tmp_path: pathlib.Path,
    make_stream: Callable[..., StreamResult],
    perfect: Callable[..., list[int]],
) -> None:
    """The one staleness signal neither the instrument version nor the schema can see."""
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x" * 100)
    _write(store_root, [make_stream(str(source), perfect())], session_path=str(tmp_path))
    (stream_row,) = store.read_streams(store_root)
    (rate_row,) = store.read_measurements(store_root, "rate")

    unchanged = store.content_identity(str(source))
    assert store.stale_reasons("rate", rate_row, stream_row=stream_row, content=unchanged) == []

    source.write_bytes(b"y" * 200)
    changed = store.content_identity(str(source))
    assert store.stale_reasons("rate", rate_row, stream_row=stream_row, content=changed) == [
        "content_size_bytes changed"
    ]


def test_staleness_is_flagged_when_the_instrument_moves_on(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """A hand-bumped version is the signal only a human can give."""
    _write(store_root, [make_stream(FRONT, perfect())])
    (row,) = store.read_measurements(store_root, "rate")
    assert store.stale_reasons("rate", row) == []
    assert store.stale_reasons("rate", {**row, "instrument_version": 0}) == ["instrument_version 0 != 1"]


def test_reading_a_store_that_was_never_written_yields_no_rows(tmp_path: pathlib.Path) -> None:
    """A dataset that does not exist is not an error for a reader."""
    empty = str(tmp_path / "never-written")
    assert store.read_measurements(empty, "rate") == []
    assert store.read_streams(empty) == []


def test_unknown_metric_is_rejected_before_any_io(store_root: str) -> None:
    """A typo should name the alternatives, not read an empty dataset and look fine."""
    with pytest.raises(KeyError, match="known per-stream metrics"):
        store.read_measurements(store_root, "no_such_metric")
    with pytest.raises(KeyError, match="known session-grain metrics"):
        store.read_session_measurements(store_root, "no_such_metric")


def test_a_metric_of_the_wrong_grain_is_rejected_by_the_reader(
    store_root: str, make_stream: Callable[..., StreamResult], perfect: Callable[..., list[int]]
) -> None:
    """The grains live in separate directories, so a name from the other one reads back as no rows.

    Which is indistinguishable from a metric nobody has measured yet, and the reason
    the readers cannot lean on :func:`instrument` for validation: it spans both
    registries on purpose, so it accepts either name.
    """
    _write(store_root, [make_stream(FRONT, perfect())], session_metrics=_aligned_metrics())
    assert store.read_measurements(store_root, "rate"), "the per-stream metric is there to be read"
    assert store.read_session_measurements(store_root, SPREAD), "the session-grain one too"

    with pytest.raises(KeyError, match=f"unknown per-stream metric '{SPREAD}'"):
        store.read_measurements(store_root, SPREAD)
    with pytest.raises(KeyError, match=f"session-grain metrics: multi_sensor_overlap, {SPREAD}"):
        store.read_session_measurements(store_root, "rate")


def test_azure_store_is_rejected_explicitly() -> None:
    """Lance's Azure options are a different set of keys; guessing would fail opaquely later."""
    with pytest.raises(StorageCliError, match="az://"):
        store.read_streams("az://container/store")
