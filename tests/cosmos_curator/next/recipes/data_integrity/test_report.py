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

"""Unit tests for the session data-integrity result model and rendering."""

import json
from typing import cast

import pytest

from cosmos_curator.core.sensors.data_integrity.results import (
    CheckResult,
    CheckStatus,
    ExpectedHzSource,
    OverallStatus,
    SessionReport,
    StreamResult,
)
from cosmos_curator.core.sensors.types.types import TimestampSource
from cosmos_curator.next.recipes.data_integrity.report import render_text, report_to_dict, to_json


def _metric(name: str, status: CheckStatus) -> CheckResult:
    return CheckResult(
        name,
        status,
        f"{name}=ok",
        measurement={"value": 0},
        evaluation={"status": status.value, "margin": 0},
    )


def _stream(
    source: str,
    *,
    statuses: list[CheckStatus] | None = None,
    error: str | None = None,
    expected_hz: float | None = 30.0,
    expected_hz_source: ExpectedHzSource | None = ExpectedHzSource.HEADER,
    timestamp_source: TimestampSource = TimestampSource.CONTAINER_PTS,
) -> StreamResult:
    metrics = [] if error is not None else [_metric(f"m{i}", s) for i, s in enumerate(statuses or [])]
    return StreamResult(
        source=source,
        codec_name=None if error else "h264",
        has_bframes=None if error else False,
        num_samples=None if error else 100,
        start_ns=None if error else 0,
        end_ns=None if error else 1000,
        metrics=metrics,
        error=error,
        expected_hz=None if error else expected_hz,
        expected_hz_source=None if error else expected_hz_source,
        timestamp_source=None if error else timestamp_source,
    )


def test_stream_status_pass_when_no_metric_fails() -> None:
    """A stream with only PASS/SKIPPED metrics is PASS."""
    stream = _stream("a", statuses=[CheckStatus.PASS, CheckStatus.SKIPPED])
    assert stream.status is OverallStatus.PASS


def test_stream_status_fail_when_any_metric_fails() -> None:
    """A single failing metric fails the stream; SKIPPED does not."""
    stream = _stream("a", statuses=[CheckStatus.PASS, CheckStatus.FAIL, CheckStatus.SKIPPED])
    assert stream.status is OverallStatus.FAIL


def test_stream_status_error_when_open_failed() -> None:
    """A stream that could not be opened is ERROR, not FAIL."""
    stream = _stream("a", error="could not open")
    assert stream.status is OverallStatus.ERROR


def test_session_error_outranks_fail() -> None:
    """An unmeasured stream outranks a FAIL: the session verdict is incomplete, not merely bad.

    A FAIL can be re-judged against new thresholds; an ERROR stream has no
    measurement to re-judge, so the session stays partial until it is read again.
    """
    report = SessionReport(
        session_path="s",
        streams=[_stream("a", statuses=[CheckStatus.FAIL]), _stream("b", error="boom")],
    )
    assert report.status is OverallStatus.ERROR


def test_session_error_when_only_errors() -> None:
    """With no FAILs but at least one ERROR, the session is ERROR."""
    report = SessionReport(
        session_path="s",
        streams=[_stream("a", statuses=[CheckStatus.PASS]), _stream("b", error="boom")],
    )
    assert report.status is OverallStatus.ERROR


def test_session_pass_when_all_streams_pass() -> None:
    """All-passing streams roll up to PASS."""
    report = SessionReport(session_path="s", streams=[_stream("a", statuses=[CheckStatus.PASS])])
    assert report.status is OverallStatus.PASS


def test_empty_session_is_error() -> None:
    """A session with no discovered streams is ERROR rather than a vacuous PASS."""
    report = SessionReport(session_path="s", streams=[])
    assert report.status is OverallStatus.ERROR


def test_a_failing_session_metric_fails_a_session_of_passing_streams() -> None:
    """The whole point of a session-grain verdict: no one stream is at fault, the set is."""
    report = SessionReport(
        session_path="s",
        streams=[_stream("a", statuses=[CheckStatus.PASS]), _stream("b", statuses=[CheckStatus.PASS])],
        metrics=[_metric("multi_sensor_overlap", CheckStatus.FAIL)],
    )
    assert report.status is OverallStatus.FAIL


def test_a_skipped_session_metric_leaves_the_verdict_alone() -> None:
    """Most sessions have too few sensors to compare, and that is not a finding."""
    report = SessionReport(
        session_path="s",
        streams=[_stream("a", statuses=[CheckStatus.PASS])],
        metrics=[_metric("multi_sensor_overlap", CheckStatus.SKIPPED)],
    )
    assert report.status is OverallStatus.PASS


def test_an_errored_stream_still_outranks_a_failing_session_metric() -> None:
    """ERROR first, because a session with an unmeasured stream has a partial verdict to re-queue."""
    report = SessionReport(
        session_path="s",
        streams=[_stream("a", statuses=[CheckStatus.PASS]), _stream("b", error="boom")],
        metrics=[_metric("multi_sensor_overlap", CheckStatus.FAIL)],
    )
    assert report.status is OverallStatus.ERROR


def test_render_text_puts_the_session_metrics_under_the_session_heading() -> None:
    """Beneath the roll-up, not beside a stream: attributing one to a stream would misplace the blame."""
    report = SessionReport(
        session_path="s",
        streams=[_stream("a", statuses=[CheckStatus.PASS])],
        metrics=[_metric("multi_sensor_overlap", CheckStatus.FAIL)],
    )
    lines = [line for line in render_text(report).splitlines() if line]

    assert lines[-5].startswith("Data-integrity report for session:")
    assert lines[-4].startswith("  streams: 1")
    assert lines[-3].startswith("  timestamps:")
    assert "multi_sensor_overlap" in lines[-2]
    assert lines[-1] == "Session overall: FAIL"


def test_report_to_dict_keeps_session_metrics_out_of_the_streams() -> None:
    """A consumer walking "streams" must not find a session's verdict among them."""
    report = SessionReport(
        session_path="s",
        streams=[_stream("a", statuses=[CheckStatus.PASS])],
        metrics=[_metric("multi_sensor_overlap", CheckStatus.FAIL)],
    )
    as_dict = report_to_dict(report)

    assert [m["name"] for m in cast("list[dict[str, object]]", as_dict["metrics"])] == ["multi_sensor_overlap"]
    streams = cast("list[dict[str, object]]", as_dict["streams"])
    assert [m["name"] for m in cast("list[dict[str, object]]", streams[0]["metrics"])] == ["m0"]
    assert json.loads(to_json(report)) == as_dict


def test_report_to_dict_counts_and_roundtrips_json() -> None:
    """report_to_dict captures counts and to_json is a faithful JSON encoding of it."""
    report = SessionReport(
        session_path="s3://bucket/clips/uuid/",
        streams=[
            _stream("a", statuses=[CheckStatus.PASS, CheckStatus.SKIPPED]),
            _stream("b", statuses=[CheckStatus.FAIL]),
            _stream("c", error="boom"),
        ],
    )
    as_dict = report_to_dict(report)
    assert as_dict["status"] == OverallStatus.ERROR.value
    assert as_dict["num_streams"] == 3
    assert as_dict["stream_status_counts"] == {"PASS": 1, "FAIL": 1, "ERROR": 1}
    assert json.loads(to_json(report)) == as_dict


def test_render_text_includes_session_verdict_and_error() -> None:
    """The human-readable report surfaces the overall verdict and per-stream error."""
    report = SessionReport(session_path="s", streams=[_stream("b", error="could not open")])
    text = render_text(report)
    assert "Session overall: ERROR" in text
    assert "could not open" in text


def test_render_text_puts_the_session_summary_after_the_streams() -> None:
    """The roll-up trails the per-stream output so it survives a long session's scrollback."""
    report = SessionReport(
        session_path="s3://bucket/clips/uuid/",
        streams=[_stream("a", statuses=[CheckStatus.PASS]), _stream("b", statuses=[CheckStatus.PASS])],
    )
    lines = [line for line in render_text(report).splitlines() if line]

    assert lines[0] == "Stream: a"
    assert lines[-4:] == [
        "Data-integrity report for session: s3://bucket/clips/uuid/",
        "  streams: 2   pass: 2   fail: 0   error: 0",
        "  timestamps: container_pts",
        "Session overall: PASS",
    ]


def test_render_text_reports_expected_hz_per_stream() -> None:
    """Each stream shows its effective rate and origin, matching the single-video report."""
    report = SessionReport(session_path="s", streams=[_stream("a", statuses=[CheckStatus.PASS], expected_hz=29.97)])
    assert "expected_hz: 29.970 (source: header)" in render_text(report)


def test_render_text_reports_unavailable_expected_hz() -> None:
    """An unresolvable rate is spelled out, since it is why the rate/gap/jitter checks SKIP."""
    report = SessionReport(
        session_path="s",
        streams=[
            _stream(
                "a",
                statuses=[CheckStatus.SKIPPED],
                expected_hz=None,
                expected_hz_source=ExpectedHzSource.UNAVAILABLE,
            )
        ],
    )
    assert "expected_hz: N/A (source: unavailable)" in render_text(report)


@pytest.mark.parametrize("source", list(TimestampSource))
def test_render_text_reports_the_clock_each_stream_was_measured_on(source: TimestampSource) -> None:
    """The reader's only way to tell a steady rig from a re-timestamped mux.

    Both produce near-zero jitter, so the metric lines below cannot distinguish them
    and the report has to say which it is.
    """
    report = SessionReport(
        session_path="s",
        streams=[_stream("a", statuses=[CheckStatus.PASS], timestamp_source=source)],
    )
    assert f"timestamps: {source.value}" in render_text(report)


def test_errored_stream_omits_the_timestamp_source_line() -> None:
    """Nothing was measured, so there is no clock to name -- as with the rate line.

    Neither per stream nor for the session: with no stream contributing, the session
    was measured on no clock rather than on a bad one.
    """
    report = SessionReport(session_path="s", streams=[_stream("b", error="could not open")])
    assert "timestamps:" not in render_text(report)


@pytest.mark.parametrize(
    ("sources", "expected"),
    [
        ([TimestampSource.CAPTURE, TimestampSource.CAPTURE], "capture"),
        ([TimestampSource.CONTAINER_PTS, TimestampSource.CONTAINER_PTS], "container_pts"),
        ([TimestampSource.CAPTURE, TimestampSource.CONTAINER_PTS], "mixed"),
    ],
)
def test_render_text_reports_the_clock_the_session_as_a_whole_was_measured_on(
    sources: list[TimestampSource], expected: str
) -> None:
    """Under the session heading because it decides whether the session metrics ran at all.

    Rolled up from the streams rather than carried on the report, so it cannot disagree
    with the per-stream lines above it or with the store row written beside it.
    """
    report = SessionReport(
        session_path="s",
        streams=[
            _stream(f"cam{i}", statuses=[CheckStatus.PASS], timestamp_source=source) for i, source in enumerate(sources)
        ],
    )
    lines = [line for line in render_text(report).splitlines() if line]

    assert f"  timestamps: {expected}" in lines
    assert cast("str", report_to_dict(report)["timestamp_source"]) == expected


def test_a_session_no_stream_contributed_to_reports_no_clock() -> None:
    """Null rather than a name, on the same terms the store column uses."""
    report = SessionReport(session_path="s", streams=[_stream("b", error="could not open")])
    assert report_to_dict(report)["timestamp_source"] is None


def test_report_to_dict_exposes_the_timestamp_source() -> None:
    """The JSON payload carries provenance under the same name the store column uses."""
    report = SessionReport(
        session_path="s",
        streams=[_stream("a", statuses=[CheckStatus.PASS], timestamp_source=TimestampSource.CAPTURE)],
    )
    stream = cast("list[dict[str, object]]", report_to_dict(report)["streams"])[0]
    assert stream["timestamp_source"] == "capture"


def test_errored_stream_omits_expected_hz_line() -> None:
    """A stream that never opened has no rate to report, so the line is left out entirely."""
    report = SessionReport(session_path="s", streams=[_stream("b", error="could not open")])
    assert "expected_hz" not in render_text(report)


def test_report_to_dict_exposes_expected_hz() -> None:
    """The JSON payload carries the rate and its origin under the single-video field names."""
    report = SessionReport(session_path="s", streams=[_stream("a", statuses=[CheckStatus.PASS], expected_hz=29.97)])
    stream = cast("list[dict[str, object]]", report_to_dict(report)["streams"])[0]
    assert stream["expected_hz"] == 29.97
    assert stream["expected_hz_source"] == "header"
