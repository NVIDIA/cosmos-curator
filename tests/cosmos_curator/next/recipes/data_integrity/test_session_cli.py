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

"""Unit tests for the session-level data-integrity CLI (argparse + main + exit codes)."""

import io
import json
import pathlib
import signal
import sys
import threading
from collections.abc import Callable

import pytest

from cosmos_curator.core.sensors.data_integrity.instruments import MAX_PERCENT
from cosmos_curator.core.sensors.data_integrity.results import (
    CheckResult,
    CheckStatus,
    SessionReport,
    StreamResult,
)
from cosmos_curator.core.utils.storage.storage_client import StorageStat
from cosmos_curator.core.utils.storage_cli import StorageCliError
from cosmos_curator.next.recipes.data_integrity import session_cli
from cosmos_curator.next.recipes.data_integrity.cli_support import (
    DEFAULT_THRESHOLDS,
    ERROR_EXIT_CODE,
    FAIL_EXIT_CODE,
    INTERRUPTED_EXIT_CODE,
    PASS_EXIT_CODE,
    Thresholds,
)


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


def test_exit_code_pass(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """An all-PASS session exits 0 and prints the human report."""
    monkeypatch.setattr(
        session_cli, "run_session", lambda *_a, **_k: SessionReport("s", [_stream("a", CheckStatus.PASS)])
    )
    code = session_cli.main(["--session-path", "s"])
    assert code == PASS_EXIT_CODE
    assert "Session overall: PASS" in capsys.readouterr().out


def test_exit_code_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """A session with a failing stream exits 1."""
    monkeypatch.setattr(
        session_cli, "run_session", lambda *_a, **_k: SessionReport("s", [_stream("a", CheckStatus.FAIL)])
    )
    assert session_cli.main(["--session-path", "s"]) == FAIL_EXIT_CODE


def test_a_failing_session_metric_exits_one_on_its_own(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No stream is at fault, so nothing but the session verdict can carry this to the exit code."""
    report = SessionReport(
        "s",
        [_stream("a", CheckStatus.PASS), _stream("b", CheckStatus.PASS)],
        metrics=[CheckResult("multi_sensor_overlap", CheckStatus.FAIL, "non_overlap_percent=40", None, None)],
    )
    monkeypatch.setattr(session_cli, "run_session", lambda *_a, **_k: report)

    assert session_cli.main(["--session-path", "s"]) == FAIL_EXIT_CODE
    out = capsys.readouterr().out
    assert "multi_sensor_overlap" in out, "an operator handed exit 1 has to be able to see what failed"


def test_unmeasured_stream_exits_error_even_alongside_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A session mixing a FAIL with an unreadable stream exits 2, not 1.

    The exit code has to carry the same precedence as the verdict, so a caller
    scripting on it can tell "measured and bad" (1) from "not fully measured" (2)
    and re-queue only the latter.
    """
    report = SessionReport("s", [_stream("a", CheckStatus.FAIL), _stream("b", CheckStatus.PASS, error="boom")])
    monkeypatch.setattr(session_cli, "run_session", lambda *_a, **_k: report)
    assert session_cli.main(["--session-path", "s"]) == ERROR_EXIT_CODE


def test_threshold_flags_reach_the_session_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Policy flags are forwarded to the runner, and default to the neutral policy."""
    seen: dict[str, object] = {}

    def _run(*_a: object, **kwargs: object) -> SessionReport:
        seen.update(kwargs)
        return SessionReport("s", [_stream("a", CheckStatus.PASS)])

    monkeypatch.setattr(session_cli, "run_session", _run)

    assert session_cli.main(["--session-path", "s"]) == PASS_EXIT_CODE
    assert seen["thresholds"] == DEFAULT_THRESHOLDS

    assert session_cli.main(["--session-path", "s", "--max-gaps", "4", "--allow-frame-reordering"]) == 0
    assert seen["thresholds"] == Thresholds(max_gaps=4, allow_frame_reordering=True)


def test_session_threshold_flags_reach_the_session_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """The session limits thread through too.

    Worth pinning separately from the per-stream flags: ``thresholds_from_args`` reads these
    two with a ``getattr`` fallback, so that a CLI offering only the per-stream flags still
    gets a whole policy. That fallback would quietly answer with the default for a renamed
    dest or dropped ``add_session_threshold_args`` call, leaving the flags accepted and inert.
    """
    seen: dict[str, object] = {}

    def _run(*_a: object, **kwargs: object) -> SessionReport:
        seen.update(kwargs)
        return SessionReport("s", [_stream("a", CheckStatus.PASS)])

    monkeypatch.setattr(session_cli, "run_session", _run)

    argv = ["--session-path", "s", "--max-sensor-spread-ns", "250000000", "--max-non-overlap-percent", "2.5"]
    assert session_cli.main(argv) == PASS_EXIT_CODE
    assert seen["thresholds"] == Thresholds(max_sensor_spread_ns=250_000_000, max_non_overlap_percent=2.5)

    # The ceiling is a legal policy, not a rejected one. Worth a case of its own
    # because it is the only value that tells `> MAX_PERCENT` from `>= MAX_PERCENT`,
    # and the rejections below can only ever pin the wrong side of that line.
    assert session_cli.main(["--session-path", "s", "--max-non-overlap-percent", "100"]) == PASS_EXIT_CODE
    assert seen["thresholds"] == Thresholds(max_non_overlap_percent=MAX_PERCENT)
    assert session_cli.main(["--session-path", "s", "--max-non-overlap-percent", "0"]) == PASS_EXIT_CODE
    assert seen["thresholds"] == Thresholds(max_non_overlap_percent=0.0)


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--max-sensor-spread-ns", "-1"),
        ("--max-sensor-spread-ns", "nan"),
        ("--max-non-overlap-percent", "-0.5"),
        ("--max-non-overlap-percent", "inf"),
        # A share cannot exceed 100, so this is a typo rather than a laxer policy:
        # 100 already tolerates every session there is.
        ("--max-non-overlap-percent", "101"),
    ],
)
def test_invalid_session_threshold_values_are_rejected_by_argparse(
    flag: str, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Bad policy values fail at parse time rather than producing a meaningless verdict."""
    with pytest.raises(SystemExit) as excinfo:
        session_cli.main(["--session-path", "s", flag, value])
    assert excinfo.value.code == 2  # argparse's own usage-error status
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"argument {flag}:" in captured.err
    assert value in captured.err


def test_interrupt_exits_without_a_report(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Ctrl-C exits 130 and writes no report.

    A session cut short has an unknown verdict, so emitting a partial report would
    invite reading it as the whole session; the exit code carries the outcome instead.
    """

    def _interrupt(*_a: object, **_k: object) -> SessionReport:
        raise KeyboardInterrupt

    monkeypatch.setattr(session_cli, "run_session", _interrupt)

    assert session_cli.main(["--session-path", "s"]) == INTERRUPTED_EXIT_CODE
    captured = capsys.readouterr()
    assert "interrupted" in captured.err
    assert captured.out == ""


def test_interrupt_masked_as_a_decode_error_still_reports_an_interruption(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A Ctrl-C that libav reports as a decode error is still reported as an abort."""

    def _fail_as_libav_would(*_a: object, **_k: object) -> SessionReport:
        signal.raise_signal(signal.SIGINT)
        msg = "Invalid data found when processing input"
        raise ValueError(msg)

    monkeypatch.setattr(session_cli, "run_session", _fail_as_libav_would)

    assert session_cli.main(["--session-path", "s"]) == INTERRUPTED_EXIT_CODE
    captured = capsys.readouterr()
    assert "interrupted" in captured.err
    assert "Invalid data" not in captured.err
    assert captured.out == ""


def test_interrupt_swallowed_by_libav_still_reports_an_interruption(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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

    assert session_cli.main(["--session-path", "s"]) == INTERRUPTED_EXIT_CODE
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "interrupted" in captured.err


def test_render_failure_exits_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """A failure while rendering or writing the report yields exit code 2, not a traceback."""
    monkeypatch.setattr(
        session_cli, "run_session", lambda *_a, **_k: SessionReport("s", [_stream("a", CheckStatus.PASS)])
    )

    def _boom(_report: object) -> str:
        msg = "unserializable measurement"
        raise ValueError(msg)

    monkeypatch.setattr(session_cli, "render_text", _boom)
    assert session_cli.main(["--session-path", "s"]) == ERROR_EXIT_CODE
    assert "error:" in capsys.readouterr().err


def test_exit_code_error_on_empty_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """A session with no discovered streams exits 2 (ERROR)."""
    monkeypatch.setattr(session_cli, "run_session", lambda *_a, **_k: SessionReport("s", []))
    assert session_cli.main(["--session-path", "s"]) == ERROR_EXIT_CODE


def test_json_output_is_valid(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """--json emits a parseable document with the session status."""
    monkeypatch.setattr(
        session_cli, "run_session", lambda *_a, **_k: SessionReport("s", [_stream("a", CheckStatus.PASS)])
    )
    session_cli.main(["--session-path", "s", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"
    assert payload["num_streams"] == 1


def test_missing_local_path_exits_error(capsys: pytest.CaptureFixture[str]) -> None:
    """A nonexistent local session path is a clean ERROR exit, not a traceback."""
    code = session_cli.main(["--session-path", "/no/such/session/dir"])
    assert code == ERROR_EXIT_CODE
    assert "error:" in capsys.readouterr().err


def test_forwards_cloud_and_sampling_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI args are threaded through to run_session (endpoint, profile, limit, hz)."""
    captured: dict[str, object] = {}

    def _fake_run_session(session_path: str, **kwargs: object) -> SessionReport:
        captured["session_path"] = session_path
        captured.update(kwargs)
        return SessionReport(session_path, [_stream("a", CheckStatus.PASS)])

    monkeypatch.setattr(session_cli, "run_session", _fake_run_session)
    session_cli.main(
        [
            "--session-path",
            "s3://bucket/clips/uuid/",
            "--expected-hz",
            "30",
            "--limit",
            "5",
            "--endpoint-url",
            "https://endpoint.io",
            "--s3-profile-name",
            "prof",
        ]
    )
    assert captured["session_path"] == "s3://bucket/clips/uuid/"
    assert captured["expected_hz"] == 30.0
    assert captured["limit"] == 5
    assert captured["endpoint_url"] == "https://endpoint.io"
    assert captured["s3_profile_name"] == "prof"


def test_max_workers_defaults_to_the_usable_cpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting --max-workers scales concurrency to the host rather than a fixed number."""
    captured: dict[str, object] = {}

    def _fake_run_session(session_path: str, **kwargs: object) -> SessionReport:
        captured.update(kwargs)
        return SessionReport(session_path, [_stream("a", CheckStatus.PASS)])

    monkeypatch.setattr(session_cli, "run_session", _fake_run_session)
    monkeypatch.setattr(session_cli, "available_cpu_count", lambda: 12)
    session_cli.main(["--session-path", "s"])
    assert captured["max_workers"] == 12


def test_explicit_max_workers_overrides_the_cpu_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit --max-workers wins over the detected CPU count."""
    captured: dict[str, object] = {}

    def _fake_run_session(session_path: str, **kwargs: object) -> SessionReport:
        captured.update(kwargs)
        return SessionReport(session_path, [_stream("a", CheckStatus.PASS)])

    monkeypatch.setattr(session_cli, "run_session", _fake_run_session)
    monkeypatch.setattr(session_cli, "available_cpu_count", lambda: 12)
    session_cli.main(["--session-path", "s", "--max-workers", "2"])
    assert captured["max_workers"] == 2


def test_bogus_video_dir_is_error(tmp_path: pathlib.Path) -> None:
    """End-to-end (no mocks): a dir of undecodable mp4s discovers streams but exits ERROR."""
    (tmp_path / "a.mp4").write_bytes(b"not a real video")
    assert session_cli.main(["--session-path", str(tmp_path)]) == ERROR_EXIT_CODE


def test_progress_lines_go_to_stderr(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--progress (default) streams a per-stream start/finish line to stderr, not stdout."""
    (tmp_path / "a.mp4").write_bytes(b"not a real video")
    session_cli.main(["--session-path", str(tmp_path)])
    captured = capsys.readouterr()
    assert "[1/1]" in captured.err
    assert "-> ERROR" in captured.err
    # The report itself stays on stdout; progress never leaks there.
    assert "[1/1]" not in captured.out


def test_no_progress_is_silent_on_stderr(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--no-progress suppresses the per-stream progress lines."""
    (tmp_path / "a.mp4").write_bytes(b"not a real video")
    session_cli.main(["--session-path", str(tmp_path), "--no-progress"])
    assert "[1/1]" not in capsys.readouterr().err


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


def test_wrapper_factory_threaded_to_run_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """With progress on, main passes a make_stream_wrapper factory into run_session."""
    captured: dict[str, object] = {}

    def _fake_run_session(session_path: str, **kwargs: object) -> SessionReport:
        captured.update(kwargs)
        return SessionReport(session_path, [_stream("a", CheckStatus.PASS)])

    monkeypatch.setattr(session_cli, "run_session", _fake_run_session)
    session_cli.main(["--session-path", "s3://bucket/clips/uuid/"])
    assert captured["make_stream_wrapper"] is not None
    # The factory produces a stream wrapper that counts an in-memory stream.
    factory = captured["make_stream_wrapper"]
    wrapper = factory(1, 1, "s3://bucket/clips/uuid/a.mp4")  # type: ignore[operator]
    wrapped = wrapper(io.BytesIO(b"xyz"))
    assert wrapped.read() == b"xyz"
