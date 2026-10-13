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

"""Unit tests for the single-video data-integrity CLI (main + exit codes)."""

import json
import signal
from collections.abc import Callable
from typing import Any

import pytest

from cosmos_curator.core.sensors.data_integrity.results import CheckResult, ResolvedConfig, VideoInfo
from cosmos_curator.core.sensors.types.types import TimestampSource
from cosmos_curator.next.recipes.data_integrity import cli, cli_support
from cosmos_curator.next.recipes.data_integrity.cli_support import (
    DEFAULT_THRESHOLDS,
    ERROR_EXIT_CODE,
    INTERRUPTED_EXIT_CODE,
    PASS_EXIT_CODE,
    Thresholds,
    interrupt_guard,
)

EngineRun = tuple[list[CheckResult], VideoInfo, ResolvedConfig]


def _stub_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the check phase succeed so a test can isolate the reporting phase."""
    monkeypatch.setattr(cli, "validate_source", lambda _source: None)
    monkeypatch.setattr(cli, "run_checks", lambda *_a, **_k: ([], None, None))


def test_a_run_without_the_index_flag_never_reaches_for_a_manifest(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reading from the index is opt-in, so an unflagged run must not open a dataset."""
    seen: dict[str, object] = {}

    def _run_checks(_source: str, **kwargs: object) -> EngineRun:
        seen.update(kwargs)
        return ([], None, None)

    monkeypatch.setattr(cli, "validate_source", lambda _source: None)
    monkeypatch.setattr(cli, "run_checks", _run_checks)
    monkeypatch.setattr(cli, "_render_human", lambda *_a, **_k: "")
    monkeypatch.setattr(
        cli_support.Mp4IndexLookup, "open", staticmethod(lambda *_a, **_k: pytest.fail("opened a manifest unasked"))
    )

    assert cli.main(["--source", "video.mp4"]) == PASS_EXIT_CODE
    assert seen["index_lookup"] is None
    capsys.readouterr()


def test_the_index_flag_hands_run_checks_the_opened_manifest(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI opens the manifest once and passes it down; run_checks decides per stream."""
    lookup = object()
    seen: dict[str, object] = {}

    def _run_checks(_source: str, **kwargs: object) -> EngineRun:
        seen.update(kwargs)
        return ([], None, None)

    monkeypatch.setattr(cli, "validate_source", lambda _source: None)
    monkeypatch.setattr(cli, "run_checks", _run_checks)
    monkeypatch.setattr(cli, "_render_human", lambda *_a, **_k: "")
    monkeypatch.setattr(cli_support.Mp4IndexLookup, "open", staticmethod(lambda *_a, **_k: lookup))

    assert (
        cli.main(["--source", "video.mp4", "--index-manifest", "s3://bucket/mp4-index/manifest.lance"])
        == PASS_EXIT_CODE
    )
    assert seen["index_lookup"] is lookup
    capsys.readouterr()


def test_an_unreadable_manifest_is_reported_as_the_manifest_failing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The run still fails, but it names the manifest rather than the video."""

    def _explode(*_args: object, **_kwargs: object) -> object:
        msg = "manifest.lance has 3 row(s) at a record_schema_version other than 5"
        raise ValueError(msg)

    monkeypatch.setattr(cli, "validate_source", lambda _source: None)
    monkeypatch.setattr(cli_support.Mp4IndexLookup, "open", staticmethod(_explode))

    exit_code = cli.main(["--source", "video.mp4", "--index-manifest", "s3://bucket/mp4-index/manifest.lance"])

    assert exit_code == ERROR_EXIT_CODE
    stderr = capsys.readouterr().err
    assert "--index-manifest s3://bucket/mp4-index/manifest.lance could not be read" in stderr
    assert "record_schema_version" in stderr, "the underlying reason has to survive the rewrap"
    assert "video.mp4" not in stderr, "the source is not what failed"


def _raise_render(*_args: object, **_kwargs: object) -> str:
    msg = "unserializable measurement"
    raise ValueError(msg)


def _stub_engine(
    monkeypatch: pytest.MonkeyPatch,
    run_engine: Callable[..., EngineRun],
    perfect: Callable[..., list[int]],
    *,
    timestamp_source: TimestampSource,
) -> None:
    """Run the real engine over a synthetic timeline measured on *timestamp_source*."""

    def _run_checks(_source: str, **kwargs: object) -> EngineRun:
        return run_engine(perfect(), expected_hz=kwargs["expected_hz"], timestamp_source=timestamp_source)

    monkeypatch.setattr(cli, "validate_source", lambda _source: None)
    monkeypatch.setattr(cli, "run_checks", _run_checks)


@pytest.mark.parametrize("timestamp_source", list(TimestampSource))
def test_the_human_report_names_the_clock_it_measured_on(
    timestamp_source: TimestampSource,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    run_engine: Callable[..., EngineRun],
    perfect: Callable[..., list[int]],
) -> None:
    """Beside the rate's origin, in the same header and for the same reason.

    A perfect cadence reads as PASS either way, so the verdict lines cannot tell the
    reader whether that cadence was recorded or manufactured by the mux.
    """
    _stub_engine(monkeypatch, run_engine, perfect, timestamp_source=timestamp_source)

    assert cli.main(["--source", "clip.mp4"]) == PASS_EXIT_CODE
    assert f"timestamps: {timestamp_source.value}" in capsys.readouterr().out


def test_the_json_report_carries_the_clock_it_measured_on(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    run_engine: Callable[..., EngineRun],
    perfect: Callable[..., list[int]],
) -> None:
    """Under ``video``, so the field name matches the session report and the store column."""
    _stub_engine(monkeypatch, run_engine, perfect, timestamp_source=TimestampSource.SENSOR)

    assert cli.main(["--source", "clip.mp4", "--json"]) == PASS_EXIT_CODE
    payload = json.loads(capsys.readouterr().out)
    assert payload["video"]["timestamp_source"] == "sensor"


@pytest.mark.parametrize(
    ("renderer", "argv"),
    [
        ("_render_human", ["--source", "clip.mp4"]),
        ("_render_json", ["--source", "clip.mp4", "--json"]),
    ],
)
def test_render_failure_exits_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    renderer: str,
    argv: list[str],
) -> None:
    """A failure while rendering the report yields exit code 2, not a traceback.

    Reporting runs after the checks have already succeeded, so it has to sit inside the
    same handler; otherwise a non-finite measurement or a closed stdout escapes as a
    traceback and breaks the documented exit-code contract.
    """
    _stub_checks(monkeypatch)
    monkeypatch.setattr(cli, renderer, _raise_render)

    assert cli.main(argv) == ERROR_EXIT_CODE
    assert "error:" in capsys.readouterr().err


def _capture_thresholds(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub the check phase and capture the keyword arguments it was handed."""
    seen: dict[str, Any] = {}

    def _run(*_a: object, **kwargs: object) -> tuple[list[object], None, None]:
        seen.update(kwargs)
        return ([], None, None)

    monkeypatch.setattr(cli, "validate_source", lambda _source: None)
    monkeypatch.setattr(cli, "run_checks", _run)
    monkeypatch.setattr(cli, "_render_human", lambda *_a, **_k: "")
    return seen


def test_thresholds_default_to_the_neutral_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without threshold flags the CLI applies the same policy as the library default."""
    seen = _capture_thresholds(monkeypatch)

    assert cli.main(["--source", "clip.mp4"]) == PASS_EXIT_CODE
    assert seen["thresholds"] == DEFAULT_THRESHOLDS


def test_threshold_flags_reach_the_check_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every threshold flag is forwarded, so the policy the user asked for is the one applied."""
    seen = _capture_thresholds(monkeypatch)
    argv = [
        "--source",
        "clip.mp4",
        "--max-strict-violations",
        "2",
        "--max-rate-deviation-percent",
        "1.5",
        "--max-gaps",
        "3",
        "--max-jitter-percent",
        "0.25",
        "--allow-frame-reordering",
    ]

    assert cli.main(argv) == PASS_EXIT_CODE
    assert seen["thresholds"] == Thresholds(
        max_strict_violations=2,
        max_rate_deviation_percent=1.5,
        max_gaps=3,
        max_jitter_percent=0.25,
        allow_frame_reordering=True,
    )


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--max-strict-violations", "-1"),
        ("--max-gaps", "nan"),
        ("--max-rate-deviation-percent", "-0.5"),
        ("--max-jitter-percent", "inf"),
    ],
)
def test_invalid_threshold_values_are_rejected_by_argparse(
    flag: str, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Bad policy values fail at parse time rather than producing a meaningless verdict."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--source", "clip.mp4", flag, value])
    assert excinfo.value.code == 2  # argparse's own usage-error status
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"argument {flag}:" in captured.err
    assert value in captured.err


def test_interrupt_exits_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ctrl-C during the checks reports an interruption and exits 130, not 2.

    A distinct code matters because an operator abort is not a failure to evaluate:
    a wrapper retrying on exit 2 should not retry a deliberate cancellation.
    """

    def _interrupt(*_a: object, **_k: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "validate_source", lambda _source: None)
    monkeypatch.setattr(cli, "run_checks", _interrupt)

    assert cli.main(["--source", "clip.mp4"]) == INTERRUPTED_EXIT_CODE
    assert "interrupted" in capsys.readouterr().err


def test_interrupt_swallowed_by_libav_still_reports_an_interruption(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run that finishes despite the Ctrl-C reports the abort, not the report.

    libav can absorb the abort entirely -- notably in its seek callback, where it
    retries and completes the run -- so a successful return is not evidence that the
    operator still wants the answer. Printing one here would also mean exit 0 for a
    session the operator believes they cancelled.
    """

    def _finish_anyway(*_a: object, **_k: object) -> tuple[list[object], None, None]:
        signal.raise_signal(signal.SIGINT)  # cooperative: sets the flag, does not raise
        return ([], None, None)

    monkeypatch.setattr(cli, "validate_source", lambda _source: None)
    monkeypatch.setattr(cli, "run_checks", _finish_anyway)
    monkeypatch.setattr(cli, "_render_human", lambda *_a, **_k: "a report nobody asked for\n")

    assert cli.main(["--source", "clip.mp4"]) == INTERRUPTED_EXIT_CODE
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "interrupted" in captured.err


def test_second_interrupt_forces_an_unwind() -> None:
    """A second Ctrl-C raises, so a phase that never reaches a read can still be escaped.

    The first signal is only a request, which nothing outside the read path is obliged
    to notice -- a stalled bucket listing, say. The escalation is what keeps the CLI
    from becoming unkillable in exchange for its quieter first press.
    """
    with interrupt_guard() as interrupted:
        signal.raise_signal(signal.SIGINT)
        assert interrupted.is_set()
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGINT)


def test_interrupt_masked_as_a_decode_error_still_reports_an_interruption(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A Ctrl-C that libav reports as a decode error is still reported as an abort.

    Abandoning a read mid-stream leaves libav with a truncated source, which it reports
    as "Invalid data found when processing input". Passing that on verbatim would blame
    the file for the operator's Ctrl-C and, worse, hand back exit 2 so a wrapper would
    retry a deliberate cancellation.
    """

    def _fail_as_libav_would(*_a: object, **_k: object) -> None:
        signal.raise_signal(signal.SIGINT)
        msg = "Invalid data found when processing input"
        raise ValueError(msg)

    monkeypatch.setattr(cli, "validate_source", lambda _source: None)
    monkeypatch.setattr(cli, "run_checks", _fail_as_libav_would)

    assert cli.main(["--source", "clip.mp4"]) == INTERRUPTED_EXIT_CODE
    err = capsys.readouterr().err
    assert "interrupted" in err
    assert "Invalid data" not in err
