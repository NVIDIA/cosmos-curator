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

"""Unit tests for the ``--store-path`` wiring on both data-integrity CLIs.

Both CLIs are stubbed at their I/O seam only -- ``run_checks`` for the single-video
tool, ``run_session`` for the session one -- so the rows under test are produced by
the real engine and the real store, with nothing faked but the source data.
"""

import argparse
import pathlib
from collections.abc import Callable

import pytest

from cosmos_curator.core.sensors.data_integrity.engine import run_session_metrics
from cosmos_curator.core.sensors.data_integrity.instruments import DEFAULT_THRESHOLDS, Thresholds
from cosmos_curator.core.sensors.data_integrity.results import (
    CheckResult,
    ResolvedConfig,
    SensorBounds,
    SessionReport,
    StreamResult,
    VideoInfo,
)
from cosmos_curator.core.sensors.types.types import TimestampSource
from cosmos_curator.next.recipes.data_integrity import cli, session_cli, store, store_cli
from cosmos_curator.next.recipes.data_integrity.cli_support import (
    ERROR_EXIT_CODE,
    FAIL_EXIT_CODE,
    PASS_EXIT_CODE,
)

SOURCE = "/data/front.mp4"
SESSION = "/data/session"

#: What argparse exits with when it rejects an argument, as opposed to the tool's own
#: ERROR_EXIT_CODE for a run that started and then failed.
USAGE_EXIT_CODE = 2

#: A session-grain measurement needs sensors that differ, so these name a session whose
#: two sensors started a millisecond apart.
NO_SKEW = Thresholds(max_sensor_spread_ns=0)
_ONE_MINUTE_NS = 60_000_000_000
_SKEW_NS = 1_000_000

EngineRun = tuple[list[CheckResult], VideoInfo, ResolvedConfig]


@pytest.fixture
def stub_check(
    monkeypatch: pytest.MonkeyPatch,
    run_engine: Callable[..., EngineRun],
    perfect: Callable[..., list[int]],
) -> Callable[..., None]:
    """Make ``di-check`` run the engine over a synthetic timeline instead of a file.

    The engine runs on each call rather than once up front, so the thresholds the CLI
    parsed are the thresholds the metrics are judged against -- otherwise a policy flag
    would silently have no effect on the verdict under test.
    """

    def _install(timestamps: list[int] | None = None) -> None:
        timeline = perfect() if timestamps is None else timestamps

        def _run_checks(_source: str, **kwargs: object) -> EngineRun:
            return run_engine(
                timeline,
                expected_hz=kwargs["expected_hz"],
                thresholds=kwargs["thresholds"],
            )

        monkeypatch.setattr(cli, "run_checks", _run_checks)
        monkeypatch.setattr(cli, "validate_source", lambda _source: None)

    return _install


@pytest.fixture
def stub_session(
    monkeypatch: pytest.MonkeyPatch,
    make_stream: Callable[..., StreamResult],
    perfect: Callable[..., list[int]],
) -> Callable[..., SessionReport]:
    """Make ``di-session`` return a prepared report instead of discovering real streams."""

    def _install(streams: list[StreamResult] | None = None, metrics: list[CheckResult] | None = None) -> SessionReport:
        report = SessionReport(
            session_path=SESSION,
            streams=[make_stream(f"{SESSION}/front.mp4", perfect())] if streams is None else streams,
            metrics=[] if metrics is None else metrics,
        )
        monkeypatch.setattr(session_cli, "run_session", lambda *_args, **_kwargs: report)
        return report

    return _install


def _session_argv(store_root: str | None = None) -> list[str]:
    argv = ["--session-path", SESSION, "--no-progress"]
    return argv if store_root is None else [*argv, "--store-path", store_root]


def test_check_without_the_flag_writes_nothing(
    stub_check: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persisting is opt-in, and the store pulls in lance, so a plain check must not reach it."""
    stub_check()

    def _explode(*_args: object, **_kwargs: object) -> str:
        pytest.fail("the store was written by a run that never asked for one")

    monkeypatch.setattr(cli, "persist_run", _explode)
    assert cli.main(["--source", SOURCE]) == PASS_EXIT_CODE


def test_check_persists_one_stream_with_no_session(stub_check: Callable[..., None], store_root: str) -> None:
    """A single-video run writes a full set of rows; the null session is its signature."""
    stub_check()
    assert cli.main(["--source", SOURCE, "--store-path", store_root]) == PASS_EXIT_CODE

    (row,) = store.read_streams(store_root)
    assert row["source"] == SOURCE
    assert row["session_path"] is None
    assert row["relative_key"] is None
    assert len(store.read_evaluations(store_root)) == len(store.INSTRUMENTS)
    assert store.read_manifest(store_root)["tool"] == "di-check"


def test_check_stores_the_policy_it_applied(stub_check: Callable[..., None], store_root: str) -> None:
    """The stored policy has to be the run's, not the registry default."""
    stub_check()
    cli.main(["--source", SOURCE, "--store-path", store_root, "--max-gaps", "7"])
    assert store.read_manifest(store_root)["policy_id"] == store.policy_id(Thresholds(max_gaps=7))


def test_check_stores_the_defaults_when_no_policy_flag_is_given(
    stub_check: Callable[..., None], store_root: str
) -> None:
    """A run with no threshold flags stores the documented defaults, not an empty policy."""
    stub_check()
    cli.main(["--source", SOURCE, "--store-path", store_root])
    assert store.read_manifest(store_root)["policy_id"] == store.policy_id(DEFAULT_THRESHOLDS)


def test_check_exit_code_still_reflects_the_verdict(
    stub_check: Callable[..., None], drifting: Callable[..., list[int]], store_root: str
) -> None:
    """Persisting changes what is kept, not what is reported."""
    stub_check(drifting(percent=3.0))
    argv = ["--source", SOURCE, "--store-path", store_root, "--max-rate-deviation-percent", "0.5"]
    assert cli.main(argv) == FAIL_EXIT_CODE

    rate = next(row for row in store.read_evaluations(store_root) if row["metric_name"] == "rate")
    assert rate["check_status"] == "FAIL"


def test_check_prints_its_report_even_when_the_store_fails(
    stub_check: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    store_root: str,
) -> None:
    """The verdict is the primary output and survives a store failure.

    The exit code does not: saving the results was part of what was asked for, so a
    run that could not save them did not finish.
    """

    def _boom(*_args: object, **_kwargs: object) -> str:
        msg = "disk on fire"
        raise OSError(msg)

    monkeypatch.setattr(cli, "persist_run", _boom)
    stub_check()

    assert cli.main(["--source", SOURCE, "--store-path", store_root]) == ERROR_EXIT_CODE
    captured = capsys.readouterr()
    assert "Overall: PASS" in captured.out
    assert "could not write the store" in captured.err


def test_session_persists_the_session_context(stub_session: Callable[..., SessionReport], store_root: str) -> None:
    """The session tool fills in exactly what the single-video one leaves null."""
    stub_session()
    assert session_cli.main(_session_argv(store_root)) == PASS_EXIT_CODE

    (row,) = store.read_streams(store_root)
    assert row["session_path"] == SESSION
    assert row["relative_key"] == "front.mp4"
    assert store.read_manifest(store_root)["tool"] == "di-session"


def test_session_persists_a_stream_it_could_not_open(
    stub_session: Callable[..., SessionReport],
    make_errored_stream: Callable[..., StreamResult],
    store_root: str,
) -> None:
    """An unreadable stream is the case worth keeping: the row records why."""
    stub_session([make_errored_stream(f"{SESSION}/broken.mp4")])
    assert session_cli.main(_session_argv(store_root)) == ERROR_EXIT_CODE

    (row,) = store.read_streams(store_root)
    assert row["error"] == "moov atom not found"
    assert store.read_evaluations(store_root) == []


def test_two_runs_share_one_store(
    stub_session: Callable[..., SessionReport],
    make_stream: Callable[..., StreamResult],
    perfect: Callable[..., list[int]],
    store_root: str,
) -> None:
    """The second run appends to the first rather than replacing it."""
    stub_session([make_stream(f"{SESSION}/front.mp4", perfect())])
    session_cli.main(_session_argv(store_root))
    stub_session([make_stream(f"{SESSION}/rear.mp4", perfect())])
    session_cli.main(_session_argv(store_root))

    assert sorted(str(row["source"]) for row in store.read_streams(store_root)) == [
        f"{SESSION}/front.mp4",
        f"{SESSION}/rear.mp4",
    ]


def test_session_persists_the_session_itself(stub_session: Callable[..., SessionReport], store_root: str) -> None:
    """The session tool writes the session's own row, which the single-video one has none of."""
    stub_session()
    assert session_cli.main(_session_argv(store_root)) == PASS_EXIT_CODE

    (row,) = store.read_sessions(store_root)
    assert row["session_path"] == SESSION
    assert row["num_streams"] == 1
    assert row["num_unreadable"] == 0
    assert row["error"] is None


def test_what_the_session_tool_exited_on_is_what_reads_back(
    stub_session: Callable[..., SessionReport],
    make_stream: Callable[..., StreamResult],
    perfect: Callable[..., list[int]],
    capsys: pytest.CaptureFixture[str],
    store_root: str,
) -> None:
    """The regression this table exists to prevent.

    Every stream passes and the session does not, so before session verdicts were
    stored the store read back ``PASS`` for a run the CLI exited ``1`` on.
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

    assert session_cli.main([*_session_argv(store_root), "--max-sensor-spread-ns", "0"]) == FAIL_EXIT_CODE
    assert "Session overall: FAIL" in capsys.readouterr().out

    assert {row["check_status"] for row in store.read_evaluations(store_root)} == {"PASS"}
    verdict = next(
        row for row in store.read_session_evaluations(store_root) if row["metric_name"] == "multi_sensor_spread"
    )
    assert verdict["check_status"] == "FAIL"
    assert store.session_rollup(store_root)["status"] == "FAIL"


def test_session_prints_its_report_even_when_the_store_fails(
    stub_session: Callable[..., SessionReport],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    store_root: str,
) -> None:
    """Same contract as the single-video CLI: report on stdout, failure on the exit code."""

    def _boom(*_args: object, **_kwargs: object) -> str:
        msg = "bucket denied"
        raise OSError(msg)

    monkeypatch.setattr(session_cli, "persist_run", _boom)
    stub_session()

    assert session_cli.main(_session_argv(store_root)) == ERROR_EXIT_CODE
    captured = capsys.readouterr()
    assert "Session overall: PASS" in captured.out
    assert "could not write the store" in captured.err


def test_a_store_is_a_store_whichever_tool_wrote_it(
    stub_check: Callable[..., None],
    stub_session: Callable[..., SessionReport],
    store_root: str,
) -> None:
    """One flag, one layout: a session run reads back the single-video run's rows."""
    stub_check()
    cli.main(["--source", f"{SESSION}/front.mp4", "--store-path", store_root])
    stub_session()
    session_cli.main(_session_argv(store_root))

    # Same source both times, so the two runs are two generations of one stream.
    (row,) = store.read_streams(store_root)
    assert row["session_path"] == SESSION, "the later, session-aware run should win"
    assert len(store.read_streams(store_root, latest=False)) == 2


def test_both_tools_spell_the_flag_the_same_way() -> None:
    """Absent by default, and one name on both tools, so either can write the same store."""
    assert cli._parse_args(["--source", SOURCE]).store_path is None
    assert session_cli._parse_args(["--session-path", SESSION]).store_path is None
    assert cli._parse_args(["--source", SOURCE, "--store-path", "/s"]).store_path == "/s"
    assert session_cli._parse_args(["--session-path", SESSION, "--store-path", "/s"]).store_path == "/s"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", "store path is empty"),
        ("   ", "store path is empty"),
        ("az://container/store", "does not support az:// yet"),
        ("gs://bucket/store", "unsupported store URI"),
        ("file:///tmp/store", "unsupported store URI"),
        ("s3://", "names no bucket"),
        ("s3:///", "names no bucket"),
    ],
)
def test_an_unusable_store_path_is_refused(value: str, expected: str) -> None:
    """Each of these would otherwise be taken for a local directory and half-work.

    An empty path resolves to the filesystem root, and an unsupported scheme creates a
    directory named after the URI before failing somewhere deeper.
    """
    with pytest.raises(argparse.ArgumentTypeError, match=expected):
        store_cli.validate_store_path(value)


def test_a_home_relative_store_path_is_expanded() -> None:
    """Otherwise the shell's ``~`` survives into a directory literally named ``~``."""
    assert store_cli.validate_store_path("~/di-store") == str(pathlib.Path.home() / "di-store")


def test_an_s3_store_path_is_passed_through_untouched() -> None:
    """A bucket key is opaque, so normalizing it would risk naming a different object."""
    assert store_cli.validate_store_path("s3://bucket/prefix/") == "s3://bucket/prefix/"
    assert store_cli.validate_store_path("s3://bucket") == "s3://bucket", "a bucket root is a valid store"


@pytest.mark.parametrize(
    ("argv", "main"),
    [
        (["--source", SOURCE, "--store-path", "az://c/store"], cli.main),
        (_session_argv("az://c/store"), session_cli.main),
    ],
)
def test_an_unusable_store_path_is_refused_before_any_source_is_read(
    argv: list[str], main: Callable[[list[str]], int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store is written last, so validating it late means paying for the whole run first.

    Catching it at parse time is what turns a typo from a wasted session into a usage
    error.
    """

    def _explode(*_args: object, **_kwargs: object) -> object:
        pytest.fail("a source was read despite an unusable --store-path")

    monkeypatch.setattr(cli, "run_checks", _explode)
    monkeypatch.setattr(session_cli, "run_session", _explode)

    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == USAGE_EXIT_CODE
