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

"""Unit tests for the function the Ray Data stage runs on one session."""

import datetime
import pathlib
import pickle
from collections.abc import Callable
from typing import Any

import pytest

from cosmos_curator.core.sensors.data_integrity.instruments import INSTRUMENTS, SESSION_INSTRUMENTS
from cosmos_curator.core.sensors.data_integrity.results import CheckStatus, StreamResult
from cosmos_curator.next.recipes.data_integrity import processing, store_schema
from cosmos_curator.next.recipes.data_integrity.config import ResolvedDataIntegrityConfig

RUN_ID = "0" * 32
CREATED_AT = datetime.datetime(2026, 8, 20, 12, 0, tzinfo=datetime.UTC)

#: How many metrics a stream with a usable declared rate produces, and therefore how
#: many measurement rows and how many evaluations it contributes.
NUM_METRICS = len(INSTRUMENTS)


def _config(
    session: pathlib.Path,
    *,
    thresholds: dict[str, Any] | None = None,
    **execution: Any,  # noqa: ANN401
) -> ResolvedDataIntegrityConfig:
    """Resolve a config covering one local session."""
    return ResolvedDataIntegrityConfig.model_validate(
        {
            "schema_version": 1,
            "kind": "data-integrity",
            "input": {"sessions": [str(session)]},
            "output": {"store_root": str(session.parent / "di-store")},
            "checks": {"thresholds": thresholds or {}},
            "execution": execution,
        }
    )


@pytest.fixture
def session(tmp_path: pathlib.Path, h264_video: Callable[..., bytes]) -> pathlib.Path:
    """Build a session directory holding one readable video and one unreadable one."""
    path = tmp_path / "clips" / "session"
    path.mkdir(parents=True)
    (path / "front.mp4").write_bytes(h264_video())
    (path / "rear.mp4").write_bytes(b"not a video")
    return path


def _check(
    session: pathlib.Path,
    *,
    thresholds: dict[str, Any] | None = None,
    **execution: Any,  # noqa: ANN401
) -> dict[str, object]:
    """Measure one session the way the Ray stage does."""
    return processing.check_session(
        {"session_path": str(session)},
        config=_config(session, thresholds=thresholds, **execution),
        run_id=RUN_ID,
        created_at=CREATED_AT,
    )


def _rows(record: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    """Unpack the opaque payload the driver appends."""
    return pickle.loads(record["rows"])  # type: ignore[arg-type]  # noqa: S301 -- written by check_session


def test_a_session_lists_its_streams_in_discovery_order(session: pathlib.Path) -> None:
    """Listing is the first half of the task, and it is ordinary discovery."""
    assert processing.discover_session(str(session), config=_config(session)) == [
        str(session / "front.mp4"),
        str(session / "rear.mp4"),
    ]


def test_listing_honours_the_per_session_stream_cap(session: pathlib.Path) -> None:
    """``limit`` caps streams within a session; it never drops a session."""
    config = ResolvedDataIntegrityConfig.model_validate(
        {
            "schema_version": 1,
            "kind": "data-integrity",
            "input": {"sessions": [str(session)], "limit": 1},
            "output": {"store_root": str(session.parent / "di-store")},
        }
    )

    assert processing.discover_session(str(session), config=config) == [str(session / "front.mp4")]


def test_a_session_that_cannot_be_listed_is_reported_rather_than_raised(tmp_path: pathlib.Path) -> None:
    """Raising would abort the dataset and discard every session already measured.

    Ray's ``max_errored_blocks`` is 0 by default and a block holds several sessions, so
    one stale path in a long input would cost the whole run its commit.
    """
    missing = tmp_path / "absent"

    record = processing.check_session(
        {"session_path": str(missing)},
        config=_config(missing),
        run_id=RUN_ID,
        created_at=CREATED_AT,
    )

    assert "absent" in str(record["listing_error"])
    assert record["streams"] == 0
    # One row, and only one: nothing under the session was measured, so it has no
    # measurements and no verdicts -- the shape an errored stream has.
    rows = _rows(record)
    (session_row,) = rows[store_schema.SESSION_DATASET]
    assert "absent" in str(session_row["error"])
    assert session_row["num_streams"] == 0
    assert set(rows) == {store_schema.SESSION_DATASET}


def test_an_unreachable_stream_leaves_no_row_and_does_not_cost_its_siblings(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
    unreachable_video: Callable[[pathlib.Path], pathlib.Path],
) -> None:
    """Counted, never recorded: a row here would supersede a healthy earlier measurement.

    Readers resolve to the newest row per ``stream_id``, so persisting "we could not
    reach this" would read back later as "this stream is broken".

    The failure is a real unreadable file rather than an injected exception, because the
    thing most worth pinning is the one line that asks for this treatment: with
    ``raise_infrastructure_errors`` dropped from the call below, the real runner hands
    back an error row and every assertion here fails.
    """
    path = tmp_path / "clips" / "locked"
    path.mkdir(parents=True)
    (path / "front.mp4").write_bytes(h264_video())
    unreachable_video(path / "rear.mp4")

    record = _check(path)
    stream_rows = _rows(record)[store_schema.STREAM_DATASET]

    assert record["unreachable"] == 1
    assert record["unreadable"] == 0
    assert record["streams"] == 2, "the count of streams found is unchanged by one being unreachable"
    assert [row["source"] for row in stream_rows] == [str(path / "front.mp4")]


def test_every_record_carries_every_column(session: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """One Arrow type per column across blocks, which is why the listing error is "" and not None.

    The driver's ``iter_batches`` concatenates blocks, and a column that is null in one
    and a string in another has no common type to concatenate under.
    """
    measured = _check(session)
    unlistable = processing.check_session(
        {"session_path": str(tmp_path / "absent")},
        config=_config(tmp_path / "absent"),
        run_id=RUN_ID,
        created_at=CREATED_AT,
    )

    assert set(measured) == set(unlistable)
    assert {type(value) for value in (measured["listing_error"], unlistable["listing_error"])} == {str}
    assert measured["listing_error"] == ""


def test_a_session_yields_the_rows_of_every_stream_it_holds(session: pathlib.Path) -> None:
    """One task covers the whole session: a row per stream, plus the readable one's metrics."""
    record = _check(session)
    rows = _rows(record)

    assert record["streams"] == 2
    assert len(rows[store_schema.STREAM_DATASET]) == 2
    assert len(rows[store_schema.EVALUATION_DATASET]) == NUM_METRICS
    assert sum(len(rows[store_schema.metric_dataset_path(spec.name)]) for spec in INSTRUMENTS) == NUM_METRICS


def test_a_failing_session_metric_is_counted_in_the_record(
    tmp_path: pathlib.Path, h264_video: Callable[..., bytes]
) -> None:
    """The counter the driver sums into the run summary, which no store row states.

    Two sensors that stopped a second apart, under a policy allowing no non-overlap at
    all: every stream passes on its own, so the session's own verdict is the only thing
    that fails, and ``failed_metrics`` beside it must not move.
    """
    ragged = tmp_path / "ragged"
    ragged.mkdir()
    (ragged / "front.mp4").write_bytes(h264_video(frames=30))
    (ragged / "rear.mp4").write_bytes(h264_video(frames=60))

    record = _check(ragged, thresholds={"max_non_overlap_percent": 0.0})
    verdicts = {
        row["metric_name"]: row["check_status"] for row in _rows(record)[store_schema.SESSION_EVALUATION_DATASET]
    }

    assert record["failed_metrics"] == 0
    assert record["failed_session_metrics"] == 1
    assert verdicts["multi_sensor_overlap"] == CheckStatus.FAIL.value


def test_an_unreadable_stream_does_not_cost_its_siblings(session: pathlib.Path) -> None:
    """One broken video must not take out the rest of its session, or the run."""
    record = _check(session)
    stream_rows = _rows(record)[store_schema.STREAM_DATASET]

    assert record["unreadable"] == 1
    assert record["failed_metrics"] == 0
    errored = [row for row in stream_rows if row["error"] is not None]
    assert [row["source"] for row in errored] == [str(session / "rear.mp4")]
    # The readable sibling is still measured, and only it is judged: an errored stream
    # was never measured, so there is nothing to judge.
    assert len(stream_rows) == 2


def test_every_row_carries_the_runs_identity_and_its_session(session: pathlib.Path) -> None:
    """One run id across the invocation, one session path on every row of it."""
    for dataset_rows in _rows(_check(session)).values():
        for row in dataset_rows:
            assert row["run_id"] == RUN_ID
            assert row["session_path"] == str(session)
            assert row["created_at"] == CREATED_AT


def test_an_empty_session_produces_a_result_but_no_stream_rows(tmp_path: pathlib.Path) -> None:
    """An empty session is a fact about the input, not a failure of the task."""
    empty = tmp_path / "empty"
    empty.mkdir()
    record = _check(empty)
    rows = _rows(record)

    assert record["streams"] == 0
    assert record["unreadable"] == 0
    assert store_schema.STREAM_DATASET not in rows
    # It was listed, though, and that is worth a row: a session found empty and a
    # session never reached are different findings about the input.
    (session_row,) = rows[store_schema.SESSION_DATASET]
    assert session_row["error"] is None
    assert session_row["num_streams"] == 0
    # Its own metrics are still measured and judged, and come back SKIPPED for the
    # reason a single-sensor session's do: a session metric needs two sensors to have
    # anything to say. Nothing here forks on `num_streams == 0` -- both shapes go
    # through `session_metrics`, which is what stops the pipeline and di-session from
    # judging a session two ways.
    assert record["failed_session_metrics"] == 0
    for spec in SESSION_INSTRUMENTS:
        (measurement,) = rows[store_schema.session_metric_dataset_path(spec.name)]
        assert measurement["is_defined"] is False
        assert measurement["num_sensors"] == 0
    assert [row["check_status"] for row in rows[store_schema.SESSION_EVALUATION_DATASET]] == [
        CheckStatus.SKIPPED.value
    ] * len(SESSION_INSTRUMENTS)


def test_the_configured_attempt_budget_reaches_every_stream(
    session: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``execution.stream_attempts`` is the retry knob, and it is not decorative."""
    seen: list[object] = []

    def _record(source: str, **kwargs: object) -> StreamResult:
        seen.append(kwargs["max_attempts"])
        return StreamResult(
            source=source,
            codec_name=None,
            has_bframes=None,
            num_samples=None,
            start_ns=None,
            end_ns=None,
            metrics=[],
            error="stubbed",
        )

    monkeypatch.setattr(processing, "run_one_stream", _record)

    _check(session, stream_attempts=5)

    assert seen == [5, 5]


def test_the_payload_names_only_datasets_the_driver_can_write(session: pathlib.Path) -> None:
    """The payload's keys and the schemas they are written under are one contract."""
    assert set(_rows(_check(session))) <= set(processing.DATASET_SCHEMAS)
