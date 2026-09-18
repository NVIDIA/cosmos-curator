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

"""Session-level rendering for the data-integrity tool.

The results themselves -- :class:`~.results.StreamResult`,
:class:`~.results.SessionReport` and the verdict enums they roll up -- are defined in
:mod:`.results`, because the store persists the same objects this renders. What is
left here is the presentation: a human-readable block via :func:`render_text` and a
machine-readable one via :func:`report_to_dict` / :func:`to_json`.
"""

import json

from cosmos_curator.core.sensors.data_integrity.results import (
    CheckResult,
    OverallStatus,
    SessionReport,
    StreamResult,
    session_clock,
)


def _count_by_status(streams: list[StreamResult]) -> dict[str, int]:
    counts = {status.value: 0 for status in OverallStatus}
    for stream in streams:
        counts[stream.status.value] += 1
    return counts


def _metrics_to_dicts(metrics: list[CheckResult]) -> list[dict[str, object]]:
    return [
        {
            "name": metric.name,
            "status": metric.status.value,
            "reason": metric.reason,
            "measurement": metric.measurement,
            "evaluation": metric.evaluation,
        }
        for metric in metrics
    ]


def _session_clock_value(report: SessionReport) -> str | None:
    """Name the clock the session's contributing streams shared, if they shared one.

    The same roll-up the store records, so a report and its ``session.lance`` row
    cannot disagree about which clock a session was measured on.
    """
    clock = session_clock(report.streams)
    return None if clock is None else clock.value


def report_to_dict(report: SessionReport) -> dict[str, object]:
    """Convert a :class:`SessionReport` to a plain JSON-serializable dict."""
    return {
        "session_path": report.session_path,
        "status": report.status.value,
        "num_streams": len(report.streams),
        "stream_status_counts": _count_by_status(report.streams),
        # Derived from the streams below rather than carried on the report, because it
        # is not an independent fact: it is the roll-up of their clocks, and a field
        # could disagree with them.
        "timestamp_source": _session_clock_value(report),
        # Beside the streams rather than inside them: these judge the session, and a
        # consumer that walks "streams" must not see them as one stream's verdict.
        "metrics": _metrics_to_dicts(report.metrics),
        "streams": [
            {
                "source": stream.source,
                "status": stream.status.value,
                "codec_name": stream.codec_name,
                "has_bframes": stream.has_bframes,
                "num_samples": stream.num_samples,
                "start_ns": stream.start_ns,
                "end_ns": stream.end_ns,
                "expected_hz": stream.expected_hz,
                "expected_hz_source": (
                    stream.expected_hz_source.value if stream.expected_hz_source is not None else None
                ),
                "timestamp_source": (stream.timestamp_source.value if stream.timestamp_source is not None else None),
                "error": stream.error,
                "metrics": _metrics_to_dicts(stream.metrics),
            }
            for stream in report.streams
        ],
    }


def to_json(report: SessionReport, *, indent: int = 2) -> str:
    """Render a :class:`SessionReport` as a machine-readable JSON string."""
    return json.dumps(report_to_dict(report), indent=indent, allow_nan=False)


_METRIC_NAME_WIDTH = 26
_METRIC_STATUS_WIDTH = 10


def _render_stream(stream: StreamResult) -> list[str]:
    lines = [f"Stream: {stream.source}"]
    if stream.error is not None:
        lines.append(f"  ERROR: {stream.error}")
        return lines
    lines.append(
        f"  codec: {stream.codec_name}   has_bframes: {str(stream.has_bframes).lower()}   "
        f"num_samples: {stream.num_samples}   start_ns: {stream.start_ns}   end_ns: {stream.end_ns}"
    )
    if stream.expected_hz_source is not None:
        # Same wording as the single-video report so the two outputs stay comparable.
        value = f"{stream.expected_hz:.3f}" if stream.expected_hz is not None else "N/A"
        lines.append(f"  expected_hz: {value} (source: {stream.expected_hz_source.value})")
    if stream.timestamp_source is not None:
        # Beside the rate's provenance rather than among the numbers above, because it
        # qualifies all of them: on container PTS every timing metric below describes
        # the mux rather than the recording.
        lines.append(f"  timestamps: {stream.timestamp_source.value}")
    lines.append("")
    lines.extend(
        f"  {metric.name:<{_METRIC_NAME_WIDTH}}{metric.status.value:<{_METRIC_STATUS_WIDTH}}{metric.reason}"
        for metric in stream.metrics
    )
    lines.append(f"  -> {stream.status.value}")
    return lines


def render_text(report: SessionReport) -> str:
    """Render a :class:`SessionReport` as a human-readable multi-line string."""
    counts = _count_by_status(report.streams)
    lines: list[str] = []
    for stream in report.streams:
        lines.extend(_render_stream(stream))
        lines.append("")
    lines.append(f"Data-integrity report for session: {report.session_path}")
    lines.append(
        f"  streams: {len(report.streams)}   "
        + "   ".join(f"{name.lower()}: {counts[name]}" for name in (s.value for s in OverallStatus))
    )
    clock = _session_clock_value(report)
    if clock is not None:
        # Above the session metrics because it decides whether they ran at all: the two
        # cross-sensor checks skip unless every contributing stream reported `sensor`.
        lines.append(f"  timestamps: {clock}")
    # Under the session heading, in the same columns the per-stream metrics use, so the
    # two read alike -- what differs is only which line they sit beneath.
    lines.extend(
        f"  {metric.name:<{_METRIC_NAME_WIDTH}}{metric.status.value:<{_METRIC_STATUS_WIDTH}}{metric.reason}"
        for metric in report.metrics
    )
    lines.append(f"Session overall: {report.status.value}")
    return "\n".join(lines)
