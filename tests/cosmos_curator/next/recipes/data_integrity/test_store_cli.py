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

"""Unit tests for the ``output.store_root`` wiring on the ``di`` tool.

The tool is stubbed at its I/O seam only -- ``run_session`` -- so the rows under test
are produced by the real engine and the real store, with nothing faked but the source
data.

Whether a store root is *shaped* correctly is ``test_config.py``'s subject now that the
validation lives on the config model. What is left here is what a run actually writes.
"""

import pathlib
from collections.abc import Callable

import pytest
import yaml

from cosmos_curator.core.sensors.data_integrity.engine import run_session_metrics
from cosmos_curator.core.sensors.data_integrity.instruments import DEFAULT_THRESHOLDS, Thresholds
from cosmos_curator.core.sensors.data_integrity.results import (
    CheckResult,
    SensorBounds,
    SessionReport,
    StreamResult,
)
from cosmos_curator.core.sensors.types.types import TimestampSource
from cosmos_curator.next.recipes.data_integrity import session_cli, store
from cosmos_curator.next.recipes.data_integrity.cli_support import (
    ERROR_EXIT_CODE,
    FAIL_EXIT_CODE,
    PASS_EXIT_CODE,
)

SESSION = "/data/session"

#: A session-grain measurement needs sensors that differ, so these name a session whose
#: two sensors started a millisecond apart.
NO_SKEW = Thresholds(max_sensor_spread_ns=0)
_ONE_MINUTE_NS = 60_000_000_000
_SKEW_NS = 1_000_000


@pytest.fixture
def argv(tmp_path: pathlib.Path) -> Callable[..., list[str]]:
    """Write a config naming ``store_root`` and return the argv pointing the tool at it."""

    def _write(store_root: str | None = None, **blocks: object) -> list[str]:
        data: dict[str, object] = {
            "schema_version": 1,
            "kind": "data-integrity",
            "input": {"sessions": [SESSION]},
            "output": {"store_root": store_root},
            **blocks,
        }
        path = tmp_path / "di.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return [str(path), "--no-progress"]

    return _write


@pytest.fixture
def stub_session(
    monkeypatch: pytest.MonkeyPatch,
    make_stream: Callable[..., StreamResult],
    perfect: Callable[..., list[int]],
) -> Callable[..., SessionReport]:
    """Make the tool return a prepared report instead of discovering real streams."""

    def _install(streams: list[StreamResult] | None = None, metrics: list[CheckResult] | None = None) -> SessionReport:
        report = SessionReport(
            session_path=SESSION,
            streams=[make_stream(f"{SESSION}/front.mp4", perfect())] if streams is None else streams,
            metrics=[] if metrics is None else metrics,
        )
        monkeypatch.setattr(session_cli, "run_session", lambda *_args, **_kwargs: report)
        return report

    return _install


def test_a_null_store_root_writes_nothing(
    stub_session: Callable[..., SessionReport], monkeypatch: pytest.MonkeyPatch, argv: Callable[..., list[str]]
) -> None:
    """Persisting is opt-in, and the store pulls in lance, so a plain check must not reach it."""
    stub_session()

    def _explode(*_args: object, **_kwargs: object) -> str:
        pytest.fail("the store was written by a run that never asked for one")

    monkeypatch.setattr(session_cli, "persist_run", _explode)
    assert session_cli.main(argv()) == PASS_EXIT_CODE


def test_a_run_persists_the_session_and_its_streams(
    stub_session: Callable[..., SessionReport], store_root: str, argv: Callable[..., list[str]]
) -> None:
    """One run writes a full set of rows, keyed to the session it measured."""
    stub_session()
    assert session_cli.main(argv(store_root)) == PASS_EXIT_CODE

    (row,) = store.read_streams(store_root)
    assert row["source"] == f"{SESSION}/front.mp4"
    assert row["session_path"] == SESSION
    assert row["relative_key"] == "front.mp4"
    assert len(store.read_evaluations(store_root)) == len(store.INSTRUMENTS)
    assert store.read_manifest(store_root)["tool"] == "di"


def test_a_run_persists_the_session_itself(
    stub_session: Callable[..., SessionReport], store_root: str, argv: Callable[..., list[str]]
) -> None:
    """The session's own row, which is what a reader rolls a verdict up from."""
    stub_session()
    assert session_cli.main(argv(store_root)) == PASS_EXIT_CODE

    (row,) = store.read_sessions(store_root)
    assert row["session_path"] == SESSION
    assert row["num_streams"] == 1
    assert row["num_unreadable"] == 0
    assert row["error"] is None


def test_a_run_stores_the_policy_it_applied(
    stub_session: Callable[..., SessionReport], store_root: str, argv: Callable[..., list[str]]
) -> None:
    """The stored policy has to be the run's, not the registry default."""
    stub_session()
    session_cli.main(argv(store_root, checks={"thresholds": {"max_gaps": 7}}))
    assert store.read_manifest(store_root)["policy_id"] == store.policy_id(Thresholds(max_gaps=7))


def test_a_run_stores_the_defaults_when_the_config_states_no_policy(
    stub_session: Callable[..., SessionReport], store_root: str, argv: Callable[..., list[str]]
) -> None:
    """An omitted thresholds block stores the documented defaults, not an empty policy."""
    stub_session()
    session_cli.main(argv(store_root))
    assert store.read_manifest(store_root)["policy_id"] == store.policy_id(DEFAULT_THRESHOLDS)


def test_a_stream_that_could_not_be_opened_is_persisted_with_its_reason(
    stub_session: Callable[..., SessionReport],
    make_errored_stream: Callable[..., StreamResult],
    store_root: str,
    argv: Callable[..., list[str]],
) -> None:
    """An unreadable stream is the case worth keeping: the row records why."""
    stub_session([make_errored_stream(f"{SESSION}/broken.mp4")])
    assert session_cli.main(argv(store_root)) == ERROR_EXIT_CODE

    (row,) = store.read_streams(store_root)
    assert row["error"] == "moov atom not found"
    assert store.read_evaluations(store_root) == []


def test_two_runs_share_one_store(
    stub_session: Callable[..., SessionReport],
    make_stream: Callable[..., StreamResult],
    perfect: Callable[..., list[int]],
    store_root: str,
    argv: Callable[..., list[str]],
) -> None:
    """The second run appends to the first rather than replacing it."""
    stub_session([make_stream(f"{SESSION}/front.mp4", perfect())])
    session_cli.main(argv(store_root))
    stub_session([make_stream(f"{SESSION}/rear.mp4", perfect())])
    session_cli.main(argv(store_root))

    assert sorted(str(row["source"]) for row in store.read_streams(store_root)) == [
        f"{SESSION}/front.mp4",
        f"{SESSION}/rear.mp4",
    ]


def test_exit_code_still_reflects_the_verdict_when_persisting(
    stub_session: Callable[..., SessionReport],
    make_stream: Callable[..., StreamResult],
    drifting: Callable[..., list[int]],
    store_root: str,
    argv: Callable[..., list[str]],
) -> None:
    """Persisting changes what is kept, not what is reported."""
    strict = Thresholds(max_rate_deviation_percent=0.5)
    stub_session([make_stream(f"{SESSION}/front.mp4", drifting(percent=3.0), thresholds=strict)])
    config = argv(store_root, checks={"thresholds": {"max_rate_deviation_percent": 0.5}})
    assert session_cli.main(config) == FAIL_EXIT_CODE

    rate = next(row for row in store.read_evaluations(store_root) if row["metric_name"] == "rate")
    assert rate["check_status"] == "FAIL"


def test_what_the_tool_exited_on_is_what_reads_back(
    stub_session: Callable[..., SessionReport],
    make_stream: Callable[..., StreamResult],
    perfect: Callable[..., list[int]],
    capsys: pytest.CaptureFixture[str],
    store_root: str,
    argv: Callable[..., list[str]],
) -> None:
    """The regression this table exists to prevent.

    Every stream passes and the session does not, so before session verdicts were
    stored the store read back ``PASS`` for a run the tool exited ``1`` on.
    """
    # On sensor times, because a session on container PTS never reaches a verdict at
    # all -- and it is the verdict, not the skip, that this table exists to record.
    streams = [
        make_stream(f"{SESSION}/front.mp4", perfect(), timestamp_source=TimestampSource.SENSOR),
        make_stream(f"{SESSION}/rear.mp4", perfect(), timestamp_source=TimestampSource.SENSOR),
    ]
    stub_session(
        streams,
        run_session_metrics(
            [
                SensorBounds(start_ns=start, end_ns=end, timestamp_source=TimestampSource.SENSOR)
                for start, end in ((0, _ONE_MINUTE_NS), (_SKEW_NS, _ONE_MINUTE_NS))
            ],
            thresholds=NO_SKEW,
        ),
    )

    config = argv(store_root, checks={"thresholds": {"max_sensor_spread_ns": 0}})
    assert session_cli.main(config) == FAIL_EXIT_CODE
    assert "Session overall: FAIL" in capsys.readouterr().out

    assert {row["check_status"] for row in store.read_evaluations(store_root)} == {"PASS"}
    verdict = next(
        row for row in store.read_session_evaluations(store_root) if row["metric_name"] == "multi_sensor_spread"
    )
    assert verdict["check_status"] == "FAIL"
    assert store.session_rollup(store_root)["status"] == "FAIL"


def test_the_report_survives_a_store_that_could_not_be_written(
    stub_session: Callable[..., SessionReport],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    store_root: str,
    argv: Callable[..., list[str]],
) -> None:
    """The verdict is the primary output and survives a store failure.

    The exit code does not: saving the results was part of what was asked for, so a
    run that could not save them did not finish.
    """

    def _boom(*_args: object, **_kwargs: object) -> str:
        msg = "bucket denied"
        raise OSError(msg)

    monkeypatch.setattr(session_cli, "persist_run", _boom)
    stub_session()

    assert session_cli.main(argv(store_root)) == ERROR_EXIT_CODE
    captured = capsys.readouterr()
    assert "Session overall: PASS" in captured.out
    assert "could not write the store" in captured.err


def test_a_store_root_the_config_rejects_costs_no_reads(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], argv: Callable[..., list[str]]
) -> None:
    """The store is written last, so validating it late means paying for the whole run first.

    Catching it while the config resolves is what turns a typo from a wasted session
    into a usage error -- the same guarantee the old parse-time ``type=`` gave, now
    covering the pipeline as well because it moved onto the shared model.
    """
    monkeypatch.setattr(session_cli, "run_session", lambda *_a, **_k: pytest.fail("read a source anyway"))

    assert session_cli.main(argv("az://container/store")) == ERROR_EXIT_CODE
    assert "does not support az:// yet" in capsys.readouterr().err
