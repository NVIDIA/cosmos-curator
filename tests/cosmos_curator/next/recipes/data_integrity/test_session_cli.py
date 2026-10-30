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

"""Unit tests for the ``di`` tool: config resolution, main, exit codes and progress.

The tool takes a config and four output flags, so the surface under test is a resolution
matrix rather than a flag matrix: the cases below pin that each config block reaches the
runner it governs, and that a setting the config cannot express is rejected rather than
ignored.
Policy *values* are the config model's own tests (``test_config.py``); what is tested
here is that this entry point reads them.
"""

import io
import json
import pathlib
import signal
import sys
import threading
from collections.abc import Callable

import pytest
import yaml

from cosmos_curator.core.sensors.data_integrity.instruments import DEFAULT_THRESHOLDS, Thresholds
from cosmos_curator.core.sensors.data_integrity.results import (
    CheckResult,
    CheckStatus,
    SessionReport,
    StreamResult,
)
from cosmos_curator.core.utils.storage.storage_client import StorageStat
from cosmos_curator.core.utils.storage_cli import StorageCliError
from cosmos_curator.next.recipes.data_integrity import cli_support, session_cli
from cosmos_curator.next.recipes.data_integrity.cli_support import (
    ERROR_EXIT_CODE,
    FAIL_EXIT_CODE,
    INTERRUPTED_EXIT_CODE,
    PASS_EXIT_CODE,
    interrupt_guard,
)
from cosmos_curator.next.recipes.data_integrity.config import DataIntegrityExecutionConfig

SESSION = "/data/session"


@pytest.fixture
def config(tmp_path: pathlib.Path) -> Callable[..., list[str]]:
    """Write a config and return the argv that points the tool at it.

    Returns argv rather than a path because every test here goes through ``main``: the
    config file *is* this tool's input, so building one is part of building a call.
    """

    def _write(session: str = SESSION, **blocks: object) -> list[str]:
        data: dict[str, object] = {
            "schema_version": 1,
            "kind": "data-integrity",
            "input": {"sessions": [session]},
            **blocks,
        }
        path = tmp_path / "di.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return [str(path)]

    return _write


def _boom_stat(_source: str) -> StorageStat:
    msg = "HEAD denied"
    raise StorageCliError(msg)


def _stream(source: str, status: CheckStatus, *, error: str | None = None) -> StreamResult:
    if error is not None:
        return StreamResult(
            source=source,
            codec_name=None,
            has_bframes=None,
            num_samples=None,
            start_ns=None,
            end_ns=None,
            metrics=[],
            error=error,
        )
    return StreamResult(
        source=source,
        codec_name="h264",
        has_bframes=False,
        num_samples=10,
        start_ns=0,
        end_ns=1,
        metrics=[CheckResult("rate", status, "rate=ok", measurement=None, evaluation=None)],
    )


def _stub_session(
    monkeypatch: pytest.MonkeyPatch,
    report: SessionReport | None = None,
    *,
    capture: dict[str, object] | None = None,
) -> None:
    """Replace the runner, optionally recording the arguments it was handed."""
    result = SessionReport("s", [_stream("a", CheckStatus.PASS)]) if report is None else report

    def _run(session_path: str, **kwargs: object) -> SessionReport:
        if capture is not None:
            capture["session_path"] = session_path
            capture.update(kwargs)
        return result

    monkeypatch.setattr(session_cli, "run_session", _run)


def test_exit_code_pass(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], config: Callable[..., list[str]]
) -> None:
    """An all-PASS session exits 0 and prints the human report."""
    _stub_session(monkeypatch)
    assert session_cli.main(config()) == PASS_EXIT_CODE
    assert "Session overall: PASS" in capsys.readouterr().out


def test_exit_code_fail(monkeypatch: pytest.MonkeyPatch, config: Callable[..., list[str]]) -> None:
    """A session with a failing stream exits 1."""
    _stub_session(monkeypatch, SessionReport("s", [_stream("a", CheckStatus.FAIL)]))
    assert session_cli.main(config()) == FAIL_EXIT_CODE


def test_a_failing_session_metric_exits_one_on_its_own(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    config: Callable[..., list[str]],
) -> None:
    """No stream is at fault, so nothing but the session verdict can carry this to the exit code."""
    _stub_session(
        monkeypatch,
        SessionReport(
            "s",
            [_stream("a", CheckStatus.PASS), _stream("b", CheckStatus.PASS)],
            metrics=[CheckResult("multi_sensor_overlap", CheckStatus.FAIL, "non_overlap_percent=40", None, None)],
        ),
    )

    assert session_cli.main(config()) == FAIL_EXIT_CODE
    out = capsys.readouterr().out
    assert "multi_sensor_overlap" in out, "an operator handed exit 1 has to be able to see what failed"


def test_unmeasured_stream_exits_error_even_alongside_a_failure(
    monkeypatch: pytest.MonkeyPatch, config: Callable[..., list[str]]
) -> None:
    """A session mixing a FAIL with an unreadable stream exits 2, not 1.

    The exit code has to carry the same precedence as the verdict, so a caller
    scripting on it can tell "measured and bad" (1) from "not fully measured" (2)
    and re-queue only the latter.
    """
    _stub_session(
        monkeypatch,
        SessionReport("s", [_stream("a", CheckStatus.FAIL), _stream("b", CheckStatus.PASS, error="boom")]),
    )
    assert session_cli.main(config()) == ERROR_EXIT_CODE


def test_exit_code_error_on_empty_session(monkeypatch: pytest.MonkeyPatch, config: Callable[..., list[str]]) -> None:
    """A session with no discovered streams exits 2 (ERROR)."""
    _stub_session(monkeypatch, SessionReport("s", []))
    assert session_cli.main(config()) == ERROR_EXIT_CODE


def test_the_config_blocks_reach_the_runner(monkeypatch: pytest.MonkeyPatch, config: Callable[..., list[str]]) -> None:
    """Every setting the runner takes comes from the config, and lands where it belongs.

    One config feeds both executors, so a block that stopped being forwarded here would
    make the same file mean two different runs. That is the drift this test rules out.
    """
    seen: dict[str, object] = {}
    _stub_session(monkeypatch, capture=seen)

    argv = config(
        session="s3://bucket/clips/uuid/",
        input={
            "sessions": ["s3://bucket/clips/uuid/"],
            "limit": 5,
            "stream_globs": ["recorder-*/*", "*.mp4"],
        },
        checks={"expected_hz": 30.0, "batch_size": 128},
        execution={
            "s3_profile_name": "prof",
            "azure_profile_name": "azprof",
            "endpoint_url": "https://endpoint.io",
            "stream_concurrency": 4,
            "stream_attempts": 5,
        },
    )
    assert session_cli.main(argv) == PASS_EXIT_CODE

    # Normalized, because one session measured here and by a pipeline run has to carry
    # one session_id in the store; the trailing slash is what normalization drops.
    assert seen["session_path"] == "s3://bucket/clips/uuid"
    assert seen["expected_hz"] == 30.0
    assert seen["batch_size"] == 128
    assert seen["limit"] == 5
    assert seen["stream_globs"] == ("recorder-*/*", "*.mp4")
    assert seen["s3_profile_name"] == "prof"
    assert seen["azure_profile_name"] == "azprof"
    assert seen["endpoint_url"] == "https://endpoint.io"
    assert seen["max_workers"] == 4
    assert seen["max_attempts"] == 5


def test_configured_retries_are_honoured_without_being_asked_for(
    monkeypatch: pytest.MonkeyPatch, config: Callable[..., list[str]]
) -> None:
    """A default config retries here exactly as it does under the pipeline.

    Worth its own test rather than only the explicit value above: the runner's own
    default is 1, so forgetting to forward the field leaves a config that reads as
    three attempts behaving as one, and nothing in the output says so.
    """
    seen: dict[str, object] = {}
    _stub_session(monkeypatch, capture=seen)

    assert session_cli.main(config()) == PASS_EXIT_CODE
    assert seen["max_attempts"] == DataIntegrityExecutionConfig().stream_attempts


def test_an_omitted_limit_means_no_cap_rather_than_zero_streams(
    monkeypatch: pytest.MonkeyPatch, config: Callable[..., list[str]]
) -> None:
    """The config spells "no cap" as null and the runner as 0, so the two have to be bridged.

    Forwarding the null unchanged would be a type error; forwarding it as a falsy cap
    would measure nothing and report an empty session as a finding about the data.
    """
    seen: dict[str, object] = {}
    _stub_session(monkeypatch, capture=seen)

    assert session_cli.main(config()) == PASS_EXIT_CODE
    assert seen["limit"] == 0
    assert seen["stream_globs"] == ()


def test_thresholds_come_from_the_config_including_the_session_limits(
    monkeypatch: pytest.MonkeyPatch, config: Callable[..., list[str]]
) -> None:
    """One policy object, built from one block, covering both grains.

    The session-grain limits are the half worth pinning: they are easy to leave out of
    one grain's policy object, and a ``getattr`` fallback would then answer with the
    default rather than report that nothing supplied them.
    """
    seen: dict[str, object] = {}
    _stub_session(monkeypatch, capture=seen)

    assert session_cli.main(config()) == PASS_EXIT_CODE
    assert seen["thresholds"] == DEFAULT_THRESHOLDS

    argv = config(
        checks={
            "thresholds": {
                "max_gaps": 4,
                "allow_frame_reordering": True,
                "max_sensor_spread_ns": 250_000_000,
                "max_non_overlap_percent": 2.5,
            }
        }
    )
    assert session_cli.main(argv) == PASS_EXIT_CODE
    assert seen["thresholds"] == Thresholds(
        max_gaps=4,
        allow_frame_reordering=True,
        max_sensor_spread_ns=250_000_000,
        max_non_overlap_percent=2.5,
    )


def test_a_set_override_reaches_the_runner(monkeypatch: pytest.MonkeyPatch, config: Callable[..., list[str]]) -> None:
    """``--set`` is the whole of the per-invocation surface, so it has to actually apply."""
    seen: dict[str, object] = {}
    _stub_session(monkeypatch, capture=seen)

    argv = [*config(), "--set", "checks.expected_hz=30", "--set", "execution.stream_concurrency=8"]
    assert session_cli.main(argv) == PASS_EXIT_CODE
    assert seen["expected_hz"] == 30.0
    assert seen["max_workers"] == 8


def test_an_invalid_config_value_is_a_clean_error_before_anything_is_read(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], config: Callable[..., list[str]]
) -> None:
    """A rejected setting exits 2 with the model's own message, not a traceback.

    Validation happens before the interrupt guard and before any source is opened, so a
    typo costs nothing; that is what the flag-level ``type=`` validators used to buy and
    what the config model has to keep buying now they are gone.
    """
    monkeypatch.setattr(session_cli, "run_session", lambda *_a, **_k: pytest.fail("measured anyway"))

    argv = [*config(), "--set", "checks.thresholds.max_gaps=-1"]
    assert session_cli.main(argv) == ERROR_EXIT_CODE
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "max_gaps" in captured.err


def test_a_missing_config_is_a_clean_error(capsys: pytest.CaptureFixture[str]) -> None:
    """The config is the input, so a wrong path reads like one rather than like a crash."""
    assert session_cli.main(["/no/such/config.yaml"]) == ERROR_EXIT_CODE
    assert "not found" in capsys.readouterr().err


def test_an_unreadable_config_is_a_clean_error_too(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A directory passed as the config is a mistyped path, not a bug to report.

    "Missing" is only the most common way a config path is wrong; a directory, a broken
    symlink or a file without read permission all fail at ``open`` instead of at the
    existence check, which is the gap a bare ``FileNotFoundError`` left open.
    """
    assert session_cli.main([str(tmp_path)]) == ERROR_EXIT_CODE

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert str(tmp_path) in captured.err


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        ({"sessions": [SESSION], "session_roots": ["/data"]}, "session_roots"),
        ({"sessions": [SESSION], "session_list_uri": "/data/list.txt"}, "session_list_uri"),
        (
            {"sessions": [SESSION], "session_roots": ["/data"], "session_list_uri": "/data/list.txt"},
            "session_list_uri and session_roots",
        ),
        ({"sessions": [SESSION, "/data/other"]}, "names 2"),
    ],
)
def test_an_input_naming_more_than_one_session_is_refused_without_expanding_it(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    config: Callable[..., list[str]],
    block: dict[str, object],
    expected: str,
) -> None:
    """Refused on the config, not on what it expands to.

    Expanding first would cost a listing to say the same thing, and would *pass* for a
    root that happens to hold one session today -- then start failing the day a second
    one lands. The message names ``run-pipeline`` because that is the tool being asked for.
    """
    monkeypatch.setattr(session_cli, "run_session", lambda *_a, **_k: pytest.fail("measured anyway"))

    assert session_cli.main(config(input=block)) == ERROR_EXIT_CODE
    stderr = capsys.readouterr().err
    assert expected in stderr
    assert "run-pipeline" in stderr


def test_a_lone_video_file_is_a_session_of_one_stream(
    monkeypatch: pytest.MonkeyPatch, config: Callable[..., list[str]]
) -> None:
    """Why there is no separate single-video tool: discovery resolves a file like a prefix."""
    seen: dict[str, object] = {}
    _stub_session(monkeypatch, capture=seen)

    assert session_cli.main(config(session="/data/front.mp4")) == PASS_EXIT_CODE
    assert seen["session_path"] == "/data/front.mp4"


def test_the_stream_index_reaches_the_runner(monkeypatch: pytest.MonkeyPatch, config: Callable[..., list[str]]) -> None:
    """--stream-idx has no config twin, and the store keys on it, so it must not be dropped."""
    seen: dict[str, object] = {}
    _stub_session(monkeypatch, capture=seen)

    assert session_cli.main(config()) == PASS_EXIT_CODE
    assert seen["stream_idx"] == 0

    assert session_cli.main([*config(), "--stream-idx", "1"]) == PASS_EXIT_CODE
    assert seen["stream_idx"] == 1


def test_a_negative_stream_index_is_rejected_at_parse_time(
    config: Callable[..., list[str]], capsys: pytest.CaptureFixture[str]
) -> None:
    """One of the four flags left, so it keeps its own validator rather than the model's."""
    with pytest.raises(SystemExit) as excinfo:
        session_cli.main([*config(), "--stream-idx", "-1"])
    assert excinfo.value.code == 2  # argparse's own usage-error status
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "argument --stream-idx:" in captured.err


def test_a_config_without_a_manifest_never_reaches_for_one(
    monkeypatch: pytest.MonkeyPatch, config: Callable[..., list[str]]
) -> None:
    """Reading from the index is opt-in, so an unset ``mp4_index_uri`` must not open a dataset."""
    seen: dict[str, object] = {}
    _stub_session(monkeypatch, capture=seen)
    monkeypatch.setattr(
        cli_support.Mp4IndexLookup, "open", staticmethod(lambda *_a, **_k: pytest.fail("opened a manifest unasked"))
    )

    assert session_cli.main(config()) == PASS_EXIT_CODE
    assert seen["index_lookup"] is None


def test_the_manifest_is_opened_once_and_handed_down(
    monkeypatch: pytest.MonkeyPatch, config: Callable[..., list[str]]
) -> None:
    """Opened here and shared across the session; the runner decides per stream."""
    lookup = object()
    seen: dict[str, object] = {}
    _stub_session(monkeypatch, capture=seen)
    monkeypatch.setattr(cli_support.Mp4IndexLookup, "open", staticmethod(lambda *_a, **_k: lookup))

    argv = config(input={"sessions": [SESSION], "mp4_index_uri": "s3://bucket/mp4-index/manifest.lance"})
    assert session_cli.main(argv) == PASS_EXIT_CODE
    assert seen["index_lookup"] is lookup


def test_an_unreadable_manifest_is_reported_as_the_manifest_failing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], config: Callable[..., list[str]]
) -> None:
    """The session is not what failed, and "could not check session" reads as a data finding."""

    def _explode(*_args: object, **_kwargs: object) -> object:
        msg = "manifest.lance has 3 row(s) at a record_schema_version other than 5"
        raise ValueError(msg)

    monkeypatch.setattr(cli_support.Mp4IndexLookup, "open", staticmethod(_explode))
    monkeypatch.setattr(session_cli, "run_session", lambda *_a, **_k: pytest.fail("measured anyway"))

    argv = config(input={"sessions": [SESSION], "mp4_index_uri": "s3://bucket/mp4-index/manifest.lance"})
    assert session_cli.main(argv) == ERROR_EXIT_CODE

    stderr = capsys.readouterr().err
    assert "input.mp4_index_uri s3://bucket/mp4-index/manifest.lance could not be read" in stderr
    assert "record_schema_version" in stderr, "the underlying reason has to survive the rewrap"
    assert "could not check session" not in stderr


def test_interrupt_exits_without_a_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], config: Callable[..., list[str]]
) -> None:
    """Ctrl-C exits 130 and writes no report.

    A session cut short has an unknown verdict, so emitting a partial report would
    invite reading it as the whole session; the exit code carries the outcome instead.
    """

    def _interrupt(*_a: object, **_k: object) -> SessionReport:
        raise KeyboardInterrupt

    monkeypatch.setattr(session_cli, "run_session", _interrupt)

    assert session_cli.main(config()) == INTERRUPTED_EXIT_CODE
    captured = capsys.readouterr()
    assert "interrupted" in captured.err
    assert captured.out == ""


def test_interrupt_masked_as_a_decode_error_still_reports_an_interruption(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], config: Callable[..., list[str]]
) -> None:
    """A Ctrl-C that libav reports as a decode error is still reported as an abort.

    Abandoning a read mid-stream leaves libav with a truncated source, which it reports
    as "Invalid data found when processing input". Passing that on verbatim would blame
    the file for the operator's Ctrl-C and hand back exit 2, so a wrapper would retry a
    deliberate cancellation.
    """

    def _fail_as_libav_would(*_a: object, **_k: object) -> SessionReport:
        signal.raise_signal(signal.SIGINT)
        msg = "Invalid data found when processing input"
        raise ValueError(msg)

    monkeypatch.setattr(session_cli, "run_session", _fail_as_libav_would)

    assert session_cli.main(config()) == INTERRUPTED_EXIT_CODE
    captured = capsys.readouterr()
    assert "interrupted" in captured.err
    assert "Invalid data" not in captured.err
    assert captured.out == ""


def test_interrupt_swallowed_by_libav_still_reports_an_interruption(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], config: Callable[..., list[str]]
) -> None:
    """A session that returns a report despite the Ctrl-C reports the abort instead.

    libav can absorb the abort in its seek callback, retry, and finish the stream, and a
    discovery listing never reaches a read so cannot notice the event at all. In neither
    case is a returned report evidence that the operator still wants the answer.
    """

    def _finish_anyway(*_a: object, **_k: object) -> SessionReport:
        signal.raise_signal(signal.SIGINT)  # cooperative: sets the flag, does not raise
        return SessionReport("s", [_stream("a", CheckStatus.PASS)])

    monkeypatch.setattr(session_cli, "run_session", _finish_anyway)

    assert session_cli.main(config()) == INTERRUPTED_EXIT_CODE
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "interrupted" in captured.err


def test_second_interrupt_forces_an_unwind() -> None:
    """A second Ctrl-C raises, so a phase that never reaches a read can still be escaped.

    The first signal is only a request, which nothing outside the read path is obliged
    to notice -- a stalled bucket listing, say. The escalation is what keeps the tool
    from becoming unkillable in exchange for its quieter first press.
    """
    with interrupt_guard() as interrupted:
        signal.raise_signal(signal.SIGINT)
        assert interrupted.is_set()
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGINT)


def test_render_failure_exits_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], config: Callable[..., list[str]]
) -> None:
    """A failure while rendering or writing the report yields exit code 2, not a traceback."""
    _stub_session(monkeypatch)

    def _boom(_report: object) -> str:
        msg = "unserializable measurement"
        raise ValueError(msg)

    monkeypatch.setattr(session_cli, "render_text", _boom)
    assert session_cli.main(config()) == ERROR_EXIT_CODE
    assert "error:" in capsys.readouterr().err


def test_json_output_is_valid(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], config: Callable[..., list[str]]
) -> None:
    """--json emits a parseable document with the session status."""
    _stub_session(monkeypatch)
    session_cli.main([*config(), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"
    assert payload["num_streams"] == 1


def test_missing_local_path_exits_error(config: Callable[..., list[str]], capsys: pytest.CaptureFixture[str]) -> None:
    """A nonexistent local session path is a clean ERROR exit, not a traceback."""
    assert session_cli.main(config(session="/no/such/session/dir")) == ERROR_EXIT_CODE
    assert "error:" in capsys.readouterr().err


def test_bogus_video_dir_is_error(tmp_path: pathlib.Path, config: Callable[..., list[str]]) -> None:
    """End-to-end (no mocks): a dir of undecodable mp4s discovers streams but exits ERROR."""
    session = tmp_path / "session"
    session.mkdir()
    (session / "a.mp4").write_bytes(b"not a real video")
    assert session_cli.main(config(session=str(session))) == ERROR_EXIT_CODE


def _one_bad_video(tmp_path: pathlib.Path, count: int = 1) -> str:
    session = tmp_path / "session"
    session.mkdir(parents=True)
    for index in range(count):
        (session / f"{index}.mp4").write_bytes(b"not a real video")
    return str(session)


def test_progress_is_off_for_a_single_stream_and_on_for_several(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], config: Callable[..., list[str]]
) -> None:
    """The default follows the stream count, which is only known after the listing.

    One stream's start line is immediately followed by the report itself, so it says
    nothing the operator is not about to be told; with several, knowing which stream is
    being waited on is the whole value.
    """
    session_cli.main(config(session=_one_bad_video(tmp_path)))
    assert "[1/1]" not in capsys.readouterr().err

    session_cli.main(config(session=_one_bad_video(tmp_path / "many", count=3)))
    assert "[1/3]" in capsys.readouterr().err


@pytest.mark.parametrize(("flag", "shown"), [("--progress", True), ("--no-progress", False)])
def test_an_explicit_progress_flag_overrides_the_stream_count(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    config: Callable[..., list[str]],
    flag: str,
    shown: bool,  # noqa: FBT001 - a parametrized expectation, not a mode switch
) -> None:
    """Either direction, so the default is a default rather than a rule."""
    session_cli.main([*config(session=_one_bad_video(tmp_path)), flag])
    assert ("[1/1]" in capsys.readouterr().err) is shown


def test_progress_lines_never_leak_onto_stdout(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], config: Callable[..., list[str]]
) -> None:
    """Stdout stays a single report / JSON document, whatever progress is doing."""
    session_cli.main([*config(session=_one_bad_video(tmp_path)), "--progress"])
    captured = capsys.readouterr()
    assert "-> ERROR" in captured.err
    assert "[1/1]" not in captured.out


def test_silencing_progress_does_not_cost_the_store_its_content_identity(
    monkeypatch: pytest.MonkeyPatch, config: Callable[..., list[str]]
) -> None:
    """--no-progress is a display choice, and the ``HEAD`` responses are not for display.

    They are what the store records as content identity, and it reuses the caller's
    rather than issuing a second round of its own -- so dropping them here would cost a
    silent run either its identity columns or a duplicate lookup per stream.
    """
    seen: dict[str, object] = {}
    _stub_session(monkeypatch, capture=seen)
    monkeypatch.setattr(session_cli, "persist_run", lambda *_a, **_k: "run-id")

    argv = [*config(session="s3://bucket/clips/uuid/", output={"store_root": "/data/di-store"}), "--no-progress"]
    assert session_cli.main(argv) == PASS_EXIT_CODE

    factory = seen["make_stream_wrapper"]
    assert factory is not None, "the factory is what carries the lookup, display or no display"


def test_a_config_without_a_store_never_reaches_for_one(
    monkeypatch: pytest.MonkeyPatch, config: Callable[..., list[str]]
) -> None:
    """A null ``store_root`` is the Ray-free case, and it must not touch lance at all."""
    _stub_session(monkeypatch)
    monkeypatch.setattr(session_cli, "persist_run", lambda *_a, **_k: pytest.fail("wrote a store nobody asked for"))
    assert session_cli.main(config(output={"store_root": None})) == PASS_EXIT_CODE
    assert session_cli.main(config()) == PASS_EXIT_CODE, "an absent output block means the same thing"


def test_the_store_records_this_tool_as_the_writer(
    monkeypatch: pytest.MonkeyPatch, config: Callable[..., list[str]]
) -> None:
    """One name now that there is one tool. Write-only, so older rows keep theirs."""
    seen: dict[str, object] = {}
    _stub_session(monkeypatch)

    def _persist(store_root: str, _streams: object, **kwargs: object) -> str:
        seen["store_root"] = store_root
        seen.update(kwargs)
        return "run-id"

    monkeypatch.setattr(session_cli, "persist_run", _persist)
    assert session_cli.main(config(output={"store_root": "/data/di-store"})) == PASS_EXIT_CODE
    assert seen["store_root"] == "/data/di-store"
    assert seen["tool"] == "di"


def test_perf_profile_reports_each_stream_and_the_session_around_them(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], config: Callable[..., list[str]]
) -> None:
    """Phases are per stream because that is the grain they are measured at.

    ``render_ms`` and the total are session-grain: one report is rendered, and the total
    has to be the session's elapsed time rather than a sum of phases, which would exceed
    it as soon as two streams overlap.
    """
    report = SessionReport("s", [_stream("a", CheckStatus.PASS), _stream("b", CheckStatus.PASS)])

    def _run(_session_path: str, **kwargs: object) -> SessionReport:
        stats = kwargs["stats"]
        assert isinstance(stats, dict)
        stats["a"] = {"sensor_init_ms": 1.0, "stream_ms": 2.0, "evaluate_ms": 3.0}
        stats["b"] = {"sensor_init_ms": 4.0, "stream_ms": 5.0, "evaluate_ms": 6.0}
        return report

    monkeypatch.setattr(session_cli, "run_session", _run)
    assert session_cli.main([*config(), "--perf-profile"]) == PASS_EXIT_CODE

    err = capsys.readouterr().err
    assert "[1/2] a" in err
    assert "[2/2] b" in err
    assert "sensor_init_ms = 4.000" in err
    assert "render_ms" in err
    assert "total_ms" in err
    assert "stream_concurrency = 1" in err


def test_perf_profile_says_when_the_phases_cannot_be_added_up(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], config: Callable[..., list[str]]
) -> None:
    """Above concurrency 1 the per-stream phases overlap, so a profile has to say so."""
    _stub_session(monkeypatch)
    argv = [*config(execution={"stream_concurrency": 4}), "--perf-profile"]
    assert session_cli.main(argv) == PASS_EXIT_CODE
    assert "streams overlap" in capsys.readouterr().err


def test_perf_profile_distinguishes_an_unmeasured_stream_from_an_instant_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], config: Callable[..., list[str]]
) -> None:
    """A stream that errored was never timed, and zeros would read as a stream that flew."""
    _stub_session(monkeypatch, SessionReport("s", [_stream("a", CheckStatus.PASS, error="moov atom not found")]))
    assert session_cli.main([*config(), "--perf-profile"]) == ERROR_EXIT_CODE

    err = capsys.readouterr().err
    assert "not measured  = ERROR" in err
    assert "sensor_init_ms" not in err


def test_no_perf_profile_means_no_stats_are_collected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], config: Callable[..., list[str]]
) -> None:
    """The runner skips the bookkeeping when nothing will print it."""
    seen: dict[str, object] = {}
    _stub_session(monkeypatch, capture=seen)
    assert session_cli.main(config()) == PASS_EXIT_CODE
    assert seen["stats"] is None
    assert "Per-phase" not in capsys.readouterr().err


def test_counting_reader_counts_and_delegates() -> None:
    """_CountingReader tallies bytes read and passes seek/tell through to the raw stream."""
    seen: list[int] = []
    raw = io.BytesIO(b"abcdefghij")
    reader = session_cli._CountingReader(raw, seen.append)
    assert reader.read(4) == b"abcd"
    assert reader.read() == b"efghij"
    assert seen == [4, 10]  # cumulative counts
    assert reader.seekable()
    assert reader.readable()
    reader.seek(0)
    assert reader.tell() == 0
    # readinto routes through read, so it is counted too.
    buf = bytearray(3)
    assert reader.readinto(buf) == 3
    assert bytes(buf) == b"abc"
    assert seen[-1] == 13


def test_progress_shows_total_and_percent_when_size_known(capsys: pytest.CaptureFixture[str]) -> None:
    """When a size lookup resolves, the counter renders read/total (pct%)."""
    progress = session_cli._Progress(stat_lookup=lambda _s: StorageStat(size_bytes=100))
    wrapper = progress.make_wrapper(1, 1, "s3://bucket/clip.mp4")
    stream = wrapper(io.BytesIO(b"x" * 50))
    stream.read()  # 50 of 100 bytes -> 50%
    assert "(50%)" in capsys.readouterr().err


def test_progress_survives_a_raising_size_lookup(capsys: pytest.CaptureFixture[str]) -> None:
    """A size lookup that raises degrades to the unknown-size counter, it does not propagate."""
    progress = session_cli._Progress(stat_lookup=_boom_stat)
    wrapper = progress.make_wrapper(1, 1, "s3://bucket/clip.mp4")
    stream = wrapper(io.BytesIO(b"x" * 50))
    stream.read()
    assert "MB read" in capsys.readouterr().err


def test_progress_drops_live_counter_when_streams_overlap(capsys: pytest.CaptureFixture[str]) -> None:
    """Concurrent streams cannot share one redrawn line, so the in-place counter is off.

    The byte total still has to reach the finish line, and nothing may emit a carriage
    return that would stomp another stream's output.
    """
    progress = session_cli._Progress(stat_lookup=lambda _s: StorageStat(size_bytes=100), live_byte_counter=False)
    wrapper = progress.make_wrapper(1, 2, "s3://bucket/clip.mp4")
    stream = wrapper(io.BytesIO(b"x" * 50))
    stream.read()
    assert capsys.readouterr().err == ""

    progress.finish(1, 2, _stream("s3://bucket/clip.mp4", CheckStatus.PASS))
    err = capsys.readouterr().err
    assert "\r" not in err
    assert "MB read" in err


def test_progress_skips_the_size_lookup_without_the_live_counter() -> None:
    """No size lookup when nothing will display the total, which is the concurrent default.

    Only the live counter reads the total; the finish line reports bytes actually read.
    On a cloud source the lookup is a HEAD behind a freshly built client per stream, and
    that construction holds the interpreter lock, so a dozen concurrent streams queue up
    behind it -- measurably, for a value that would be thrown away.
    """
    looked_up: list[str] = []

    def _record(source: str) -> StorageStat:
        looked_up.append(source)
        return StorageStat(size_bytes=100)

    progress = session_cli._Progress(stat_lookup=_record, live_byte_counter=False)
    progress.make_wrapper(1, 2, "s3://bucket/clip.mp4")(io.BytesIO(b"x" * 50)).read()
    assert looked_up == []

    # Still consulted when the counter is there to show it.
    live = session_cli._Progress(stat_lookup=_record, live_byte_counter=True)
    live.make_wrapper(1, 1, "s3://bucket/clip.mp4")(io.BytesIO(b"x" * 50)).read()
    assert looked_up == ["s3://bucket/clip.mp4"]

    # ...and when the store needs the response as content identity, even though the
    # counter itself is off. One HEAD per stream, two consumers.
    storing = session_cli._Progress(stat_lookup=_record, live_byte_counter=False, collect_stats=True)
    storing.make_wrapper(1, 2, "s3://bucket/other.mp4")(io.BytesIO(b"x" * 50)).read()
    assert looked_up == ["s3://bucket/clip.mp4", "s3://bucket/other.mp4"]
    assert storing.stats["s3://bucket/other.mp4"].size_bytes == 100


def test_a_silenced_progress_still_looks_up_what_the_store_needs() -> None:
    """``display=False`` must not reach the lookup, only the drawing.

    The counting wrapper does go away -- nothing reads the count -- but the one ``HEAD``
    per stream has a second consumer that does not care what is on screen.
    """
    looked_up: list[str] = []

    def _record(source: str) -> StorageStat:
        looked_up.append(source)
        return StorageStat(size_bytes=100)

    progress = session_cli._Progress(stat_lookup=_record, display=False, collect_stats=True)
    wrapper = progress.make_wrapper(1, 4, "s3://bucket/clip.mp4")
    assert wrapper(io.BytesIO(b"x" * 50)).read() == b"x" * 50, "the stream itself is untouched"
    assert looked_up == ["s3://bucket/clip.mp4"]
    assert progress.stats["s3://bucket/clip.mp4"].size_bytes == 100


@pytest.mark.parametrize("lookup", [lambda _s: None, _boom_stat])
def test_a_lookup_that_learned_nothing_is_not_cached_for_the_store(
    lookup: Callable[[str], StorageStat | None],
) -> None:
    """``None`` is a HEAD that learned nothing, and the store must be free to retry it.

    Caching it would satisfy the store's "reuse the caller's stat" path and persist a
    null ETag and size for an object that is perfectly readable.
    """
    progress = session_cli._Progress(stat_lookup=lookup, live_byte_counter=False, collect_stats=True)
    progress.make_wrapper(1, 2, "s3://bucket/clip.mp4")(io.BytesIO(b"x" * 50)).read()
    assert progress.stats == {}


def test_progress_writes_whole_lines_under_concurrent_hooks() -> None:
    """Hooks run on worker threads, so a line must never be interleaved with another's."""
    progress = session_cli._Progress(live_byte_counter=False)
    written: list[str] = []
    barrier = threading.Barrier(4, timeout=30)

    def _start(index: int) -> None:
        barrier.wait()
        progress.start(index, 4, f"s3://bucket/clip{index}.mp4")

    with io.StringIO() as buffer:
        real_stderr, sys.stderr = sys.stderr, buffer
        try:
            threads = [threading.Thread(target=_start, args=(i,)) for i in range(1, 5)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            written = buffer.getvalue().splitlines()
        finally:
            sys.stderr = real_stderr

    assert sorted(written) == sorted(f"[{i}/4] s3://bucket/clip{i}.mp4" for i in range(1, 5))


def test_progress_falls_back_to_read_only_when_size_unknown(capsys: pytest.CaptureFixture[str]) -> None:
    """With no resolvable size, the counter shows a bare 'MB read' figure.

    A stat carrying an ETag but no size, rather than an empty one: ``object_stat``
    collapses a lookup that learned nothing at all to ``None``, so an all-fields-empty
    stat is not an input the progress display can actually receive.
    """
    progress = session_cli._Progress(stat_lookup=lambda _s: StorageStat(etag="abc"))
    wrapper = progress.make_wrapper(1, 1, "s3://bucket/clip.mp4")
    stream = wrapper(io.BytesIO(b"x" * 50))
    stream.read()
    err = capsys.readouterr().err
    assert "MB read" in err
    assert "%" not in err
