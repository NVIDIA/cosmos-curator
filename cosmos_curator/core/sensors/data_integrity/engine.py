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

"""Engine for the data-integrity metrics: run them all and judge them.

This is the single source of truth for *running every data-integrity metric against
its subject and evaluating the results*. What it adds on top of the metric kernel
(:mod:`.metrics` / :mod:`.evaluation`) is the running: :func:`run_metrics` streams one
open sensor's timeline into every per-stream metric and resolves the effective
expected rate (:func:`resolve_expected_hz`), while :func:`run_session_metrics` folds a
session's sensor bounds into every session-grain metric.

Everything here is pure compute over an already-open sensor. There is no I/O, no
URI handling and no argparse: opening a source is the caller's job, which is what
keeps this module usable by the sensor library itself. The value-level validators
(:func:`validate_expected_hz` and friends) are the library-boundary counterpart of
a CLI's argparse ``type=`` layer, so direct callers of these APIs get the same
fail-fast contract an operator would.

The vocabulary the results are expressed in lives in :mod:`.results`, and which
threshold applies to which measurement in :mod:`.instruments`. Rendering and
aggregation live with the callers.
"""

import math
import time
from collections.abc import Iterable
from typing import Protocol

from cosmos_curator.core.sensors.data_integrity.instruments import (
    DEFAULT_THRESHOLDS,
    INSTRUMENTS,
    NAME_GAP,
    NAME_JITTER,
    NAME_ORDERING,
    NAME_RATE,
    NAME_REORDERING,
    NAME_SENSOR_OVERLAP,
    NAME_SENSOR_SPREAD,
    SESSION_INSTRUMENTS,
    Thresholds,
    evaluate_metric,
)
from cosmos_curator.core.sensors.data_integrity.metrics import (
    FrameReorderingPresentMetric,
    JitterMetric,
    Measurement,
    MultiSensorOverlapMetric,
    MultiSensorSpreadMetric,
    RateMetric,
    TimestampGapMetric,
    TimestampOrderingMetric,
)
from cosmos_curator.core.sensors.data_integrity.results import (
    CheckResult,
    CheckStatus,
    ExpectedHzSource,
    IntegritySensor,
    ResolvedConfig,
    SensorBounds,
    SessionTimestampSource,
    VideoInfo,
    session_timestamp_source,
)

# Reason attached to a rate-dependent check that has no usable expected rate.
REASON_MISSING_HZ = "skipped: --expected-hz not provided and container header lacks a nominal frame rate"

#: What a session was measured on, for the reason on a check that needed capture times.
#: ``None`` is its own phrase because "no sensor reported a clock" is not a clock.
#: ``CAPTURE`` is absent because it is not a cause: a session on capture times runs the
#: check rather than skipping it.
_SESSION_SOURCE_PHRASES: dict[SessionTimestampSource | None, str] = {
    SessionTimestampSource.CONTAINER_PTS: "every sensor was measured on container PTS",
    SessionTimestampSource.MIXED: "the sensors disagree on which clock they were measured on",
    None: "no sensor reported a clock",
}

#: Fallback for a caller that has no cause to name. A single run cannot produce one:
#: its session row and its measurements are built from the same streams through
#: :func:`~.results.session_clock`, so they agree by construction. Two runs can --
#: one that measures a session and a later one that records it without measuring it --
#: which is why a re-judge looks the clock up per measuring run rather than per session.
_UNEXPLAINED_SOURCE_PHRASE = "the session's record does not name a clock that explains it"


def validate_expected_hz(value: float | None) -> float | None:
    """Validate an already-typed expected sample rate at a library boundary.

    The argparse ``type=`` layer only guards the CLIs; this is the value-level
    counterpart so direct callers of the runner / resolver APIs get the same
    fail-fast contract. ``None`` (auto-detect from the container header) is
    allowed; ``0``, negatives, ``NaN``, and ``inf`` are not, since they would feed
    a meaningless cadence into the rate/gap/jitter metrics.
    """
    if value is not None and (not math.isfinite(value) or value <= 0.0):
        msg = f"expected_hz must be a positive finite number or None, got {value!r}"
        raise ValueError(msg)
    return value


def validate_non_negative_int(name: str, value: int) -> int:
    """Validate that an already-typed integer flag (e.g. ``batch_size`` / ``limit``) is non-negative."""
    if value < 0:
        msg = f"{name} must be >= 0, got {value!r}"
        raise ValueError(msg)
    return value


def validate_positive_int(name: str, value: int) -> int:
    """Validate that an already-typed integer count (e.g. ``max_workers``) is at least one."""
    if value < 1:
        msg = f"{name} must be >= 1, got {value!r}"
        raise ValueError(msg)
    return value


def resolve_expected_hz(user_hz: float | None, sensor: IntegritySensor) -> ResolvedConfig:
    """Resolve the effective expected sample rate and record its origin.

    Priority: explicit ``user_hz`` > container header ``avg_frame_rate`` > unavailable.

    An explicit ``user_hz`` is the authoritative baseline: it states independently
    what the capture was supposed to be, which is the only way to judge whether it
    was right.

    The header value is best-effort and weaker than it looks. It comes from
    libav's ``AVStream.avg_frame_rate``, which is not guaranteed to preserve an
    encoder-declared cadence -- for MP4 it is commonly derived from the
    container's own sample-duration table, the same timing data the checked
    timestamps come from. Judged against such a value, gap and jitter still bite,
    because they measure variation *between* samples: an irregular cadence shows
    up whatever the baseline. The rate check does not, and cannot be relied on to
    catch a uniformly wrong capture rate, since a stream that ran entirely at the
    wrong speed tends to declare a header rate that matches it. Pass
    ``--expected-hz`` whenever that distinction matters.

    ``Fraction(0)`` -- the sentinel used in
    :mod:`cosmos_curator.core.sensors.utils.video` when the header lacks a usable
    rate -- collapses to ``UNAVAILABLE`` and rate-dependent metrics SKIP.

    Raises:
        ValueError: if ``user_hz`` is non-positive or non-finite (see
            :func:`validate_expected_hz`).

    """
    user_hz = validate_expected_hz(user_hz)
    if user_hz is not None:
        return ResolvedConfig(expected_hz=user_hz, expected_hz_source=ExpectedHzSource.USER)
    nominal = sensor.video_metadata.avg_frame_rate
    if nominal.numerator == 0:
        return ResolvedConfig(expected_hz=None, expected_hz_source=ExpectedHzSource.UNAVAILABLE)
    return ResolvedConfig(expected_hz=float(nominal), expected_hz_source=ExpectedHzSource.HEADER)


def video_info(sensor: IntegritySensor) -> VideoInfo:
    """Snapshot the sensor-level facts shared by both reports."""
    has_samples = bool(sensor.timestamps_ns.size)
    return VideoInfo(
        codec_name=sensor.codec_name,
        has_bframes=sensor.has_bframes,
        num_samples=sensor.timestamps_ns.size,
        start_ns=sensor.start_ns if has_samples else None,
        end_ns=sensor.end_ns if has_samples else None,
        timestamp_source=sensor.timestamp_source,
    )


def _skipped_missing_hz(name: str) -> CheckResult:
    """Build a SKIPPED result for a rate-dependent metric with no usable rate."""
    return CheckResult(
        name=name, status=CheckStatus.SKIPPED, reason=REASON_MISSING_HZ, measurement=None, evaluation=None
    )


def reason_without_capture(session_source: SessionTimestampSource | None) -> str:
    """Phrase why a session metric needing capture times could not run.

    Public because a re-judge has to say the same thing. Nothing about a stored
    never-ran row explains itself -- every column on it is null -- so
    ``di-reevaluate`` rebuilds the reason from ``session.lance``'s recorded clock,
    and rebuilding it here rather than beside the reader is what keeps the two
    invocations describing one session the same way.

    The cause is named rather than elided because the fix differs by cause: a session
    on container PTS needs its capture-timestamp sidecars, while a mixed one has some
    already and is missing the rest.
    """
    phrase = _SESSION_SOURCE_PHRASES.get(session_source, _UNEXPLAINED_SOURCE_PHRASE)
    return f"skipped: comparing sensors needs capture timestamps, but {phrase}"


def _skipped_without_capture(name: str, session_source: SessionTimestampSource | None) -> CheckResult:
    """Build a SKIPPED result for a session metric whose sensors lacked a shared capture clock."""
    return CheckResult(
        name=name,
        status=CheckStatus.SKIPPED,
        reason=reason_without_capture(session_source),
        measurement=None,
        evaluation=None,
    )


class _MetricInstrument(Protocol):  # pragma: no cover
    """The one thing :func:`run_metrics` needs of a metric once it has been fed."""

    def measurement(self) -> Measurement:
        """Finalize the immutable measurement."""
        ...


def run_metrics(
    sensor: IntegritySensor,
    *,
    expected_hz: float | None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    batch_size: int = 0,
    stats: dict[str, float] | None = None,
) -> tuple[list[CheckResult], VideoInfo, ResolvedConfig]:
    """Run every data-integrity metric over one open sensor and evaluate them.

    Streams the sensor's timestamps once (``stream_timestamps(batch_size)``),
    folding them into the ordering / rate / gap / jitter instruments, records the
    frame-reordering flag, then evaluates each metric against ``thresholds``. A
    metric whose measurement is undefined -- or whose expected rate is unavailable
    -- is reported ``SKIPPED`` and never handed to an evaluator.

    Pure compute: the sensor is already open, so this performs no I/O and is the
    unit both CLIs share.

    Args:
        sensor: an opened sensor exposing the :class:`IntegritySensor` surface.
        expected_hz: expected sample rate in Hz; ``None`` falls back to the
            sensor's nominal ``avg_frame_rate`` (see :func:`resolve_expected_hz`).
        thresholds: pass/fail policy (see :class:`Thresholds`).
        batch_size: window size for streaming timestamps; ``0`` = one batch. The
            kernel guarantees the streaming result matches the one-shot result.
        stats: optional out-parameter; when provided, ``stream_ms`` and
            ``evaluate_ms`` wall-clock phase timings (milliseconds) are recorded.

    Returns:
        ``(results, video_info, resolved_cfg)`` with ``results`` in the fixed
        metric order (ordering, rate, gap, jitter, reordering).

    Raises:
        ValueError: if ``expected_hz`` or ``batch_size`` is invalid (see
            :func:`validate_expected_hz` / :func:`validate_non_negative_int`).

    """
    batch_size = validate_non_negative_int("batch_size", batch_size)
    resolved_cfg = resolve_expected_hz(expected_hz, sensor)
    hz = resolved_cfg.expected_hz

    ordering = TimestampOrderingMetric()
    rate = RateMetric(expected_hz=hz) if hz is not None else None
    gap = TimestampGapMetric(expected_hz=hz) if hz is not None else None
    jitter = JitterMetric(expected_hz=hz) if hz is not None else None
    timestamp_driven = [m for m in (ordering, rate, gap, jitter) if m is not None]

    t0 = time.perf_counter()
    for window in sensor.stream_timestamps(batch_size):
        for metric in timestamp_driven:
            metric.update(timestamps_ns=window)
    reordering = FrameReorderingPresentMetric()
    reordering.update(has_reordering=sensor.has_bframes)
    if stats is not None:
        stats["stream_ms"] = (time.perf_counter() - t0) * 1000

    # Report order comes from the registry: ordering (correctness of the timeline
    # itself) -> rate/gap/jitter (need a rate to judge) -> codec-level reordering, so
    # timeline defects read first. A rate-dependent metric with no usable rate was
    # never constructed above and so has no measurement to judge at all.
    built: dict[str, _MetricInstrument | None] = {
        NAME_ORDERING: ordering,
        NAME_RATE: rate,
        NAME_GAP: gap,
        NAME_JITTER: jitter,
        NAME_REORDERING: reordering,
    }
    t0 = time.perf_counter()
    results: list[CheckResult] = []
    for spec in INSTRUMENTS:
        instrument = built[spec.name]
        if instrument is None:
            results.append(_skipped_missing_hz(spec.name))
        else:
            results.append(evaluate_metric(spec, instrument.measurement(), thresholds))
    if stats is not None:
        stats["evaluate_ms"] = (time.perf_counter() - t0) * 1000
    return results, video_info(sensor), resolved_cfg


def run_session_metrics(
    bounds: Iterable[SensorBounds],
    *,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> list[CheckResult]:
    """Run every session-grain metric over one session's sensor bounds and evaluate them.

    The counterpart of :func:`run_metrics` for the metrics whose subject is a session
    rather than a stream. It takes the bounds and nothing else -- no sensors, no
    sources, no session path -- because that is all these metrics read, and taking less
    keeps a caller free to feed bounds it has already collected (a session runner
    holding finished per-stream results) rather than sensors it would have to reopen.

    A sensor that produced no timestamps has no bounds and is the caller's to leave
    out: an absent sensor is a different finding from a late one, and only the caller
    knows which sensors it expected. With fewer than two bounds every session metric
    reports ``SKIPPED``, having nothing to compare.

    Comparing sensors is only meaningful when they share a clock, so a metric declaring
    ``requires_capture_timestamps`` does not run at all unless every sensor here
    reported real capture times -- see :func:`session_timestamp_source`. That check
    comes *first*, ahead of the two-sensor minimum, and the two skips are deliberately
    different: one lone camera measured on capture times is a session that was measured
    and found to have nothing to compare, while one measured on container PTS was never
    measurable in the first place.

    Args:
        bounds: one :class:`SensorBounds` per sensor, in any order -- these metrics are
            symmetric in their inputs.
        thresholds: pass/fail policy (see :class:`Thresholds`).

    Returns:
        One :class:`CheckResult` per session metric, in registry order.

    Raises:
        TypeError: if a bound is not an integer.
        ValueError: if a bound is outside int64 nanoseconds.

    """
    sensors = list(bounds)
    session_source = session_timestamp_source(sensor.timestamp_source for sensor in sensors)
    on_capture = session_source is SessionTimestampSource.CAPTURE

    spread = MultiSensorSpreadMetric()
    overlap = MultiSensorOverlapMetric()
    for sensor in sensors:
        spread.update(start_ns=sensor.start_ns, end_ns=sensor.end_ns)
        overlap.update(start_ns=sensor.start_ns, end_ns=sensor.end_ns)

    built: dict[str, _MetricInstrument] = {
        NAME_SENSOR_SPREAD: spread,
        NAME_SENSOR_OVERLAP: overlap,
    }
    # A gated metric's accumulator was fed above but is never read: its numbers are an
    # artifact of the mux, and a stored artifact is one a re-judge can turn into a pass.
    return [
        _skipped_without_capture(spec.name, session_source)
        if spec.requires_capture_timestamps and not on_capture
        else evaluate_metric(spec, built[spec.name].measurement(), thresholds)
        for spec in SESSION_INSTRUMENTS
    ]
