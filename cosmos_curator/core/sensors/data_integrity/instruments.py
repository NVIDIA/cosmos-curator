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

"""One registry entry per data-integrity metric: policy, wording, and row shape.

The kernel (:mod:`.metrics` / :mod:`.evaluation`) deliberately knows nothing about
which threshold applies to which measurement, or how a verdict is worded. That
knowledge used to be spelled out once per metric in the CLI and would have had to
be spelled out again by the store and again by re-evaluation. Here it is declared
once, so a new metric is a new :class:`InstrumentSpec` rather than a new branch in
three places -- and so a re-evaluated verdict is produced by exactly the same code
that produced the original: :func:`evaluate_metric`, the tool's one path from a
measurement to a verdict.

Two registries, because a metric's *subject* is part of its shape:
:data:`INSTRUMENTS` judges one stream, :data:`SESSION_INSTRUMENTS` judges a whole
session. The spec type is the same for both; what differs is who iterates it and what
they feed it.

:class:`Thresholds` lives here rather than with the CLI because a threshold is only
meaningful next to the accessor that reads it.

No ``pyarrow`` import: field *shapes* are declared as :class:`FieldSpec` and
translated to Arrow types by :mod:`.store_schema`, so importing the registry (which
the CLIs always do) never pulls in the storage stack (which they only do with
``--store-path``).
"""

import enum
from collections.abc import Callable
from typing import Any, cast

import attrs

from cosmos_curator.core.sensors.data_integrity.evaluation import (
    EvaluationResult,
    EvaluationStatus,
    below_threshold,
)
from cosmos_curator.core.sensors.data_integrity.metrics import (
    FrameReorderingPresentMeasurement,
    JitterMeasurement,
    Measurement,
    MultiSensorOverlapMeasurement,
    MultiSensorSpreadMeasurement,
    RateMeasurement,
    TimestampGapMeasurement,
    TimestampOrderingMeasurement,
)
from cosmos_curator.core.sensors.data_integrity.results import (
    CheckResult,
    CheckStatus,
    evaluation_to_dict,
    measurement_to_dict,
)

# Metric identifiers used verbatim in both CLIs' human and JSON reports, and as the
# name of each metric's stored dataset.
NAME_ORDERING = "timestamp_ordering"
NAME_RATE = "rate"
NAME_GAP = "timestamp_gap"
NAME_JITTER = "jitter"
NAME_REORDERING = "frame_reordering_present"

# Session-grain metric identifiers, kept distinct from every name above because the
# two grains share a namespace in reports even though they live in separate datasets.
# "spread" rather than the design doc's "gap": what it measures is how far apart the
# sensors start and stop, and a second "gap" beside ``timestamp_gap`` -- which is about
# missing samples within one stream -- would name two unrelated things alike.
NAME_SENSOR_SPREAD = "multi_sensor_spread"
NAME_SENSOR_OVERLAP = "multi_sensor_overlap"

#: Ceiling for a threshold whose measurement is a *share* of something, and so cannot
#: exceed 100 however bad the data is. Only ``max_non_overlap_percent`` is such a
#: threshold: the deviation limits beside it (``max_rate_deviation_percent``,
#: ``max_jitter_percent``) measure how far a value strayed from an expected one, which
#: is genuinely unbounded -- a stream at a tenth of its nominal rate deviates by 900%.
#: Stated here, next to the field it constrains, because both surfaces that accept a
#: policy have to enforce it and neither should be the one that decides what it is.
MAX_PERCENT = 100.0


@attrs.define(frozen=True)
class Thresholds:
    """Pass/fail policy applied by ``run_metrics``.

    Defaults are neutral, first-principles limits (an ideal stream is strictly
    increasing, on-cadence, gap-free), not values tuned on any dataset.

    Attributes:
        max_strict_violations: max allowed ordering violations (backward +
            duplicate steps); default 0 (require strictly increasing).
        max_rate_deviation_percent: max mean-period deviation from the expected
            cadence, in percent.
        max_gaps: max allowed inferred gaps; default 0.
        max_jitter_percent: max inter-sample jitter, in percent of the period.
        allow_frame_reordering: when False (default), a B-frame / frame-reordering
            flag fails the frame-reordering metric.
        max_sensor_spread_ns: max allowed spread between the sensors of one session
            starting, or stopping, whichever end is worse. Unlike the limits above it
            has no first-principles value -- a perfectly synchronised rig spreads by
            zero, but no real one does -- so the default is a placeholder to be set
            from what the rigs actually do.
        max_non_overlap_percent: max allowed share of a session during which some
            sensor was not recording, in percent. Stated as the complement because
            every limit here is a ceiling, and an overlap is a floor. Being a share it
            is meaningless above :data:`MAX_PERCENT`, where ``100`` already tolerates
            everything -- a bound the surfaces that accept a policy enforce, since this
            class validates nothing and takes whatever it is handed.

    """

    max_strict_violations: int = 0
    max_rate_deviation_percent: float = 5.0
    max_gaps: int = 0
    max_jitter_percent: float = 10.0
    allow_frame_reordering: bool = False
    max_sensor_spread_ns: int = 1_000_000_000
    max_non_overlap_percent: float = 5.0


DEFAULT_THRESHOLDS = Thresholds()


class FieldKind(enum.Enum):
    """Storage shape of one measurement field.

    Coarser than the Python annotation on purpose: the store only needs to know
    which Arrow column type holds the value, and every discrete field in the kernel
    is an ``int64``-range integer (see the time-domain note in :mod:`.metrics`).
    """

    INT = "int"
    FLOAT = "float"
    BOOL = "bool"


@attrs.define(frozen=True)
class FieldSpec:
    """One measured field: its name, its storage shape, and whether it can be null.

    ``nullable`` describes the *measurement*, not the store. Every metric column is
    additionally null on a "never ran" row, which :mod:`.store_schema` handles by
    making all metric columns nullable in Arrow regardless of this flag; the flag is
    what tells a reader whether ``None`` is a value the metric can legitimately
    produce (an absent ``first_gap_index`` means "no gaps", not "no measurement").
    """

    name: str
    kind: FieldKind
    nullable: bool = False


@attrs.define(frozen=True)
class InstrumentSpec:
    """Everything outside the kernel that one metric needs, declared once.

    Attributes:
        name: metric identifier, also its dataset name (one of the ``NAME_*``
            constants).
        version: hand-bumped when the metric's math changes meaning. The honest
            staleness signal -- only a human knows whether a code change altered
            what a number means. Bump it when it does.
        measurement_cls: the frozen measurement type this metric finalizes.
        fields: the measured fields, in declaration order.
        requires_expected_hz: whether the metric can only be built when an expected
            rate is known. Rate, gap and jitter can't; without one they never run at
            all, which is the ``is_defined = None`` case in the store.
        requires_capture_timestamps: whether the metric can only be built when every
            contributing sensor reported real capture times. The two session-grain
            metrics can't: a constant-frame-rate mux rewrites each camera onto one
            regenerated grid, so comparing those grids reports a flawless rig by
            construction. Same consequence as the flag above -- the metric never runs
            and the store records ``is_defined = None`` -- because a stored zero
            spread would otherwise be re-judged into a ``PASS`` it never earned.
        threshold: reads this metric's limit out of a :class:`Thresholds`.
        accessor: reads the judged value out of a measurement.
        reason: renders the one-line summary the CLI prints, from the measurement
            and the applied threshold.
        margin_is_int: whether the judged value (and so the margin) is an integer.
            Kept explicit because the store's ``margin`` column is ``float64`` for
            every metric, and reads have to restore the integer-ness rather than
            silently reporting ``0.0`` where the CLI said ``0``.

    """

    name: str
    version: int
    measurement_cls: type
    fields: tuple[FieldSpec, ...]
    requires_expected_hz: bool
    requires_capture_timestamps: bool
    threshold: Callable[[Thresholds], Any]
    accessor: Callable[[Any], Any]
    reason: Callable[[Any, Any], str]
    margin_is_int: bool

    @property
    def field_names(self) -> tuple[str, ...]:
        """The measured field names, in declaration order."""
        return tuple(field.name for field in self.fields)

    def to_row(self, measurement: Measurement) -> dict[str, object]:
        """Project ``measurement`` onto its declared fields as a plain row dict.

        Values pass through as-is apart from a coercion to the declared kind, so a
        ``NaN`` stays ``NaN`` -- Arrow stores it natively, and flattening it to
        ``None`` here is exactly the loss the typed columns exist to avoid.
        """
        row: dict[str, object] = {}
        for field in self.fields:
            value = getattr(measurement, field.name)
            row[field.name] = None if value is None else _coerce(value, field.kind)
        return row

    def from_row(self, row: dict[str, object]) -> Measurement:
        """Rebuild the frozen measurement from a stored row.

        Only declared fields are read, so a row carrying the store's identity
        columns alongside them can be handed over unfiltered.
        """
        kwargs = {
            field.name: (None if row[field.name] is None else _coerce(row[field.name], field.kind))
            for field in self.fields
        }
        return cast("Measurement", self.measurement_cls(**kwargs))

    def evaluate(self, measurement: Measurement, thresholds: Thresholds) -> EvaluationResult[Any]:
        """Judge ``measurement`` under ``thresholds``, preserving the margin's type.

        The int and float paths are separate calls rather than one widened to
        ``float`` because the margin is reported verbatim: an ordering margin of
        ``0`` must not read back as ``0.0``.

        Precondition: ``measurement.is_defined``. An undefined measurement has no
        honest margin and the kernel refuses to judge it; callers check first (see
        :func:`evaluate_metric`).
        """
        threshold = self.threshold(thresholds)
        if self.margin_is_int:
            return below_threshold(
                threshold=int(threshold),
                measurement=measurement,
                accessor=cast("Callable[[Measurement], int]", self.accessor),
            )
        return below_threshold(
            threshold=float(threshold),
            measurement=measurement,
            accessor=cast("Callable[[Measurement], float]", self.accessor),
        )


def _coerce(value: object, kind: FieldKind) -> object:
    """Coerce a stored or measured value to its declared kind.

    Guards both directions: numpy scalars leaking out of a metric become Python
    scalars on the way in, and Arrow's ``int`` for a ``float64`` column (or vice
    versa) becomes the measurement's declared type on the way back out, so a
    round-tripped measurement compares equal to the original field by field.
    """
    match kind:
        case FieldKind.INT:
            return int(cast("int", value))
        case FieldKind.FLOAT:
            return float(cast("float", value))
        case FieldKind.BOOL:
            return bool(value)


_ORDERING_SPEC = InstrumentSpec(
    name=NAME_ORDERING,
    version=1,
    measurement_cls=TimestampOrderingMeasurement,
    fields=(
        FieldSpec("decreasing_count", FieldKind.INT),
        FieldSpec("duplicate_count", FieldKind.INT),
        FieldSpec("first_decreasing_index", FieldKind.INT, nullable=True),
        FieldSpec("first_duplicate_index", FieldKind.INT, nullable=True),
        FieldSpec("num_samples", FieldKind.INT),
    ),
    requires_expected_hz=False,
    requires_capture_timestamps=False,
    threshold=lambda t: t.max_strict_violations,
    accessor=lambda m: m.strict_violation_count,
    reason=lambda m, threshold: f"strict_violation_count={m.strict_violation_count} (threshold={threshold})",
    margin_is_int=True,
)

_RATE_SPEC = InstrumentSpec(
    name=NAME_RATE,
    version=1,
    measurement_cls=RateMeasurement,
    fields=(
        FieldSpec("period_deviation_percent", FieldKind.FLOAT),
        FieldSpec("expected_hz", FieldKind.FLOAT),
        FieldSpec("expected_period_ns", FieldKind.INT),
        FieldSpec("actual_mean_hz", FieldKind.FLOAT),
        FieldSpec("actual_mean_period_ns", FieldKind.FLOAT),
        FieldSpec("num_samples", FieldKind.INT),
        FieldSpec("num_intervals", FieldKind.INT),
        FieldSpec("num_filtered", FieldKind.INT),
    ),
    requires_expected_hz=True,
    requires_capture_timestamps=False,
    threshold=lambda t: t.max_rate_deviation_percent,
    accessor=lambda m: m.period_deviation_percent,
    reason=(
        lambda m, threshold: f"period_deviation_percent={m.period_deviation_percent:.4f} (threshold={threshold:.4f}%)"
    ),
    margin_is_int=False,
)

_GAP_SPEC = InstrumentSpec(
    name=NAME_GAP,
    version=1,
    measurement_cls=TimestampGapMeasurement,
    fields=(
        FieldSpec("max_gap_ns", FieldKind.INT),
        FieldSpec("expected_period_ns", FieldKind.INT),
        FieldSpec("expected_hz", FieldKind.FLOAT),
        FieldSpec("num_samples", FieldKind.INT),
        FieldSpec("num_gaps", FieldKind.INT),
        FieldSpec("first_gap_index", FieldKind.INT, nullable=True),
    ),
    requires_expected_hz=True,
    requires_capture_timestamps=False,
    threshold=lambda t: t.max_gaps,
    accessor=lambda m: m.num_gaps,
    reason=lambda m, threshold: f"num_gaps={m.num_gaps} (threshold={threshold})",
    margin_is_int=True,
)

_JITTER_SPEC = InstrumentSpec(
    name=NAME_JITTER,
    version=1,
    measurement_cls=JitterMeasurement,
    fields=(
        FieldSpec("jitter_percent", FieldKind.FLOAT),
        FieldSpec("expected_hz", FieldKind.FLOAT),
        FieldSpec("num_samples", FieldKind.INT),
        FieldSpec("num_intervals", FieldKind.INT),
        FieldSpec("num_filtered", FieldKind.INT),
    ),
    requires_expected_hz=True,
    requires_capture_timestamps=False,
    threshold=lambda t: t.max_jitter_percent,
    accessor=lambda m: m.jitter_percent,
    reason=lambda m, threshold: f"jitter_percent={m.jitter_percent:.4f} (threshold={threshold:.4f}%)",
    margin_is_int=False,
)

_REORDERING_SPEC = InstrumentSpec(
    name=NAME_REORDERING,
    version=1,
    measurement_cls=FrameReorderingPresentMeasurement,
    fields=(FieldSpec("has_reordering", FieldKind.BOOL, nullable=True),),
    requires_expected_hz=False,
    requires_capture_timestamps=False,
    # threshold 0 fails when a reordering flag is set; 1 permits it.
    threshold=lambda t: 1 if t.allow_frame_reordering else 0,
    accessor=lambda m: int(bool(m.has_reordering)),
    reason=lambda m, threshold: f"has_reordering={bool(m.has_reordering)} (threshold={threshold})",
    margin_is_int=True,
)

_SENSOR_SPREAD_SPEC = InstrumentSpec(
    name=NAME_SENSOR_SPREAD,
    version=1,
    measurement_cls=MultiSensorSpreadMeasurement,
    fields=(
        FieldSpec("start_spread_ns", FieldKind.INT),
        FieldSpec("stop_spread_ns", FieldKind.INT),
        FieldSpec("num_sensors", FieldKind.INT),
    ),
    requires_expected_hz=False,
    requires_capture_timestamps=True,
    threshold=lambda t: t.max_sensor_spread_ns,
    # The worse end of the rig: a session is as badly aligned as its furthest-out
    # sensor, whether that sensor was late to start or late to stop.
    accessor=lambda m: m.max_spread_ns,
    reason=lambda m, threshold: f"max_spread_ns={m.max_spread_ns} over {m.num_sensors} sensors (threshold={threshold})",
    margin_is_int=True,
)

_SENSOR_OVERLAP_SPEC = InstrumentSpec(
    name=NAME_SENSOR_OVERLAP,
    version=1,
    measurement_cls=MultiSensorOverlapMeasurement,
    fields=(
        FieldSpec("overlap_fraction", FieldKind.FLOAT),
        FieldSpec("non_overlap_fraction", FieldKind.FLOAT),
        FieldSpec("effective_duration_ns", FieldKind.INT),
        FieldSpec("total_duration_ns", FieldKind.INT),
        FieldSpec("num_sensors", FieldKind.INT),
    ),
    requires_expected_hz=False,
    requires_capture_timestamps=True,
    threshold=lambda t: t.max_non_overlap_percent,
    # The complement, because :func:`below_threshold` is the only comparison a spec can
    # express and an overlap is the one quantity here that a session wants *more* of.
    accessor=lambda m: m.non_overlap_percent,
    reason=(
        lambda m, threshold: (
            f"non_overlap_percent={m.non_overlap_percent:.4f} over {m.num_sensors} sensors (threshold={threshold:.4f}%)"
        )
    ),
    margin_is_int=False,
)

#: Every per-stream metric, in report order: the timeline's own correctness first, then
#: the three checks that need a rate to judge, then the codec-level flag.
INSTRUMENTS: tuple[InstrumentSpec, ...] = (
    _ORDERING_SPEC,
    _RATE_SPEC,
    _GAP_SPEC,
    _JITTER_SPEC,
    _REORDERING_SPEC,
)

#: Every session-grain metric, whose subject is a whole session rather than one stream.
#:
#: A separate tuple rather than more entries in :data:`INSTRUMENTS`, because everything
#: that iterates that one -- the per-stream engine, the store's row builders and its
#: schema map, re-evaluation -- does so once per stream. A session-arity spec in there
#: would have every one of them try to measure a session per stream.
SESSION_INSTRUMENTS: tuple[InstrumentSpec, ...] = (
    _SENSOR_SPREAD_SPEC,
    _SENSOR_OVERLAP_SPEC,
)

_BY_NAME = {spec.name: spec for spec in (*INSTRUMENTS, *SESSION_INSTRUMENTS)}


def instrument(name: str) -> InstrumentSpec:
    """Look up one metric's spec by name, of either grain.

    One lookup across both registries because a name is unique across them, and a
    caller holding a stored metric name -- re-evaluation, a staleness check -- wants
    the spec, not a prior answer about which grain it came from.

    Raises:
        KeyError: if ``name`` is not a registered metric, with the known names
            listed -- a stored dataset naming a metric this build has never heard of
            is a real situation (an older store, a reverted metric) and deserves
            better than a bare key error.

    """
    try:
        return _BY_NAME[name]
    except KeyError:
        msg = f"unknown metric {name!r}; known metrics: {', '.join(sorted(_BY_NAME))}"
        raise KeyError(msg) from None


def instrument_versions() -> dict[str, int]:
    """Map every metric name, of either grain, to its current instrument version.

    For the store's manifest, whose ``instruments`` field records every metric the
    build knows rather than the subset whose rows a run happened to write -- so a
    ``di-check`` run, which has no session to measure, still lists the session-grain
    metrics. Flat rather than split by grain, because a name is unique across the two
    registries and the datasets they land in already say which is which.
    """
    return {spec.name: spec.version for spec in (*INSTRUMENTS, *SESSION_INSTRUMENTS)}


def evaluate_metric(spec: InstrumentSpec, measurement: Measurement, thresholds: Thresholds) -> CheckResult:
    """Judge one finalized measurement under ``thresholds`` and package the verdict.

    The single evaluation path for the whole tool: both CLIs and re-evaluation of a
    stored measurement come through here, so a re-judged verdict cannot drift from
    the original by running slightly different policy code. It lives with the registry
    for that reason -- policy and wording are declared here, and re-evaluation needs
    them without needing anything a CLI owns.

    SKIPPED when the measurement is undefined -- the kernel refuses to judge one,
    having no honest margin to report. The detail names whichever count the metric
    counts, falling back to ``insufficient data`` for a metric that counts nothing
    (:class:`FrameReorderingPresentMeasurement`).
    """
    if not measurement.is_defined:
        detail = "insufficient data"
        for counted in ("num_samples", "num_sensors"):
            n = getattr(measurement, counted, None)
            if n is not None:
                detail = f"{counted}={n}"
                break
        return CheckResult(
            name=spec.name,
            status=CheckStatus.SKIPPED,
            reason=f"measurement undefined ({detail})",
            measurement=measurement_to_dict(measurement),
            evaluation=None,
            raw_measurement=measurement,
        )
    result = spec.evaluate(measurement, thresholds)
    status = CheckStatus.PASS if result.status is EvaluationStatus.PASS else CheckStatus.FAIL
    return CheckResult(
        name=spec.name,
        status=status,
        reason=spec.reason(measurement, spec.threshold(thresholds)),
        measurement=measurement_to_dict(measurement),
        evaluation=evaluation_to_dict(result),
        raw_measurement=measurement,
    )
