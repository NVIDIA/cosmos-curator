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

"""Unit tests for the instrument registry: field parity, row round-trips, policy wiring."""

import math

import attrs
import pytest

from cosmos_curator.core.sensors.data_integrity.instruments import (
    DEFAULT_THRESHOLDS,
    INSTRUMENTS,
    SESSION_INSTRUMENTS,
    FieldKind,
    InstrumentSpec,
    Thresholds,
    instrument,
    instrument_versions,
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

#: Both registries, for the structural checks that hold of any spec whatever its grain.
ALL_INSTRUMENTS = (*INSTRUMENTS, *SESSION_INSTRUMENTS)


def _same_fields(left: Measurement, right: Measurement) -> bool:
    """Compare two measurements field by field, treating NaN as equal to NaN.

    ``==`` will not do: ``TimestampGapMeasurement`` is declared ``eq=False`` and so
    compares by identity, which would make an equality assertion pass vacuously. And
    even for the rest, ``NaN != NaN`` -- which is exactly the value these round-trips
    exist to protect.
    """
    a, b = attrs.asdict(left), attrs.asdict(right)  # type: ignore[arg-type]
    if a.keys() != b.keys():
        return False
    return all(_same_value(a[key], b[key]) for key in a)


def _same_value(left: object, right: object) -> bool:
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left) and math.isnan(right):
        return True
    return bool(left == right)


#: One instance per metric, each carrying a value that has to survive storage: a
#: nullable index that is set, a nullable index that is not, and a NaN.
_ROUND_TRIP_CASES: list[Measurement] = [
    TimestampOrderingMeasurement(
        decreasing_count=2,
        duplicate_count=1,
        first_decreasing_index=7,
        first_duplicate_index=None,
        num_samples=600,
    ),
    RateMeasurement(
        period_deviation_percent=0.42,
        expected_hz=30.0,
        expected_period_ns=33_333_333,
        actual_mean_hz=29.87,
        actual_mean_period_ns=33_478_260.5,
        num_samples=600,
        num_intervals=599,
        num_filtered=0,
    ),
    # Undefined: the fields are NaN, not zero, and that distinction has to survive.
    RateMeasurement(
        period_deviation_percent=float("nan"),
        expected_hz=30.0,
        expected_period_ns=33_333_333,
        actual_mean_hz=float("nan"),
        actual_mean_period_ns=float("nan"),
        num_samples=1,
        num_intervals=0,
        num_filtered=0,
    ),
    TimestampGapMeasurement(
        max_gap_ns=90_000_000,
        expected_period_ns=33_333_333,
        expected_hz=30.0,
        num_samples=600,
        num_gaps=3,
        first_gap_index=41,
    ),
    JitterMeasurement(
        jitter_percent=float("nan"),
        expected_hz=30.0,
        num_samples=1,
        num_intervals=0,
        num_filtered=0,
    ),
    FrameReorderingPresentMeasurement(has_reordering=True),
    FrameReorderingPresentMeasurement(has_reordering=None),
    MultiSensorSpreadMeasurement(start_spread_ns=250, stop_spread_ns=400, num_sensors=3),
    MultiSensorOverlapMeasurement(
        overlap_fraction=0.6,
        non_overlap_fraction=0.4,
        effective_duration_ns=750,
        total_duration_ns=1_250,
        num_sensors=2,
    ),
    # Undefined: a session of one shared instant has no ratio, and the NaN saying so
    # has to survive storage rather than reading back as a measured zero overlap.
    MultiSensorOverlapMeasurement(
        overlap_fraction=float("nan"),
        non_overlap_fraction=float("nan"),
        effective_duration_ns=0,
        total_duration_ns=0,
        num_sensors=2,
    ),
]


@pytest.mark.parametrize("measurement", _ROUND_TRIP_CASES, ids=lambda m: type(m).__name__)
def test_row_round_trip_preserves_every_field(measurement: Measurement) -> None:
    """A measurement survives to_row / from_row unchanged, NaN and None included."""
    spec = next(s for s in ALL_INSTRUMENTS if isinstance(measurement, s.measurement_cls))
    assert _same_fields(spec.from_row(spec.to_row(measurement)), measurement)


def test_round_trip_keeps_nan_distinct_from_null() -> None:
    """NaN means "measured, undefined"; null means "not measured". Conflating them loses the difference."""
    spec = instrument("rate")
    undefined = RateMeasurement(
        period_deviation_percent=float("nan"),
        expected_hz=30.0,
        expected_period_ns=33_333_333,
        actual_mean_hz=float("nan"),
        actual_mean_period_ns=float("nan"),
        num_samples=1,
        num_intervals=0,
        num_filtered=0,
    )
    row = spec.to_row(undefined)
    assert math.isnan(row["period_deviation_percent"])  # type: ignore[arg-type]
    assert row["period_deviation_percent"] is not None


def test_from_row_ignores_columns_it_does_not_own() -> None:
    """A stored row carries the identity prefix too, and must not need filtering first."""
    spec = instrument("frame_reordering_present")
    row = {"stream_id": "abc", "run_id": "def", "is_defined": True, "has_reordering": False}
    assert spec.from_row(row).has_reordering is False  # type: ignore[attr-defined]


@pytest.mark.parametrize("spec", ALL_INSTRUMENTS, ids=lambda s: s.name)
def test_declared_fields_match_the_measurement_class(spec: InstrumentSpec) -> None:
    """The registry's field list must not drift from the attrs class it describes.

    Field *shapes* are declared by hand so the store never depends on annotation
    introspection; this is the check that keeps that declaration honest. ClassVars
    (MIN_SAMPLES and friends) are excluded by attrs and stay unpersisted.
    """
    declared = spec.field_names
    actual = tuple(field.name for field in attrs.fields(spec.measurement_cls))
    assert declared == actual


@pytest.mark.parametrize("spec", ALL_INSTRUMENTS, ids=lambda s: s.name)
def test_nullable_declaration_matches_the_annotation(spec: InstrumentSpec) -> None:
    """A field declared non-nullable must not be an ``| None`` on the measurement."""
    annotations = {field.name: str(field.type) for field in attrs.fields(spec.measurement_cls)}
    for field in spec.fields:
        assert field.nullable == ("None" in annotations[field.name]), field.name


@pytest.mark.parametrize("spec", ALL_INSTRUMENTS, ids=lambda s: s.name)
def test_threshold_reads_the_policy(spec: InstrumentSpec) -> None:
    """Every metric resolves a threshold out of a Thresholds without raising."""
    assert isinstance(spec.threshold(DEFAULT_THRESHOLDS), (int, float))


def test_margin_type_survives_evaluation() -> None:
    """An integer margin must not read back as a float; the CLI prints it verbatim."""
    ordering = instrument("timestamp_ordering")
    measurement = TimestampOrderingMeasurement(
        decreasing_count=0, duplicate_count=0, first_decreasing_index=None, first_duplicate_index=None, num_samples=10
    )
    result = ordering.evaluate(measurement, DEFAULT_THRESHOLDS)
    assert isinstance(result.margin, int)
    assert not isinstance(result.margin, bool)

    rate = instrument("rate")
    rate_measurement = RateMeasurement(
        period_deviation_percent=1.0,
        expected_hz=30.0,
        expected_period_ns=33_333_333,
        actual_mean_hz=30.0,
        actual_mean_period_ns=33_333_333.0,
        num_samples=10,
        num_intervals=9,
        num_filtered=0,
    )
    assert isinstance(rate.evaluate(rate_measurement, DEFAULT_THRESHOLDS).margin, float)


def test_frame_reordering_threshold_follows_the_allow_flag() -> None:
    """The flag is expressed as a threshold so it goes through the same evaluator as the rest."""
    spec = instrument("frame_reordering_present")
    assert spec.threshold(Thresholds(allow_frame_reordering=False)) == 0
    assert spec.threshold(Thresholds(allow_frame_reordering=True)) == 1


def test_unknown_metric_names_its_alternatives() -> None:
    """An older store can name a metric this build has never heard of; say which exist."""
    with pytest.raises(KeyError, match="known metrics"):
        instrument("no_such_metric")


def test_registry_is_indexed_consistently() -> None:
    """Lookup by name returns the same spec objects the ordered registries hold, of either grain."""
    assert [instrument(spec.name) for spec in ALL_INSTRUMENTS] == list(ALL_INSTRUMENTS)


def test_the_manifest_versions_every_metric_a_run_writes() -> None:
    """A run stores both grains, so a manifest that named one would understate what it describes."""
    assert instrument_versions() == {spec.name: spec.version for spec in ALL_INSTRUMENTS}


def test_the_two_grains_stay_in_separate_registries() -> None:
    """Everything iterating INSTRUMENTS does so per stream, so a session spec in there measures a session per stream."""
    assert not set(INSTRUMENTS) & set(SESSION_INSTRUMENTS)
    names = [spec.name for spec in ALL_INSTRUMENTS]
    assert len(names) == len(set(names)), "one lookup spans both registries, so a shared name would shadow a metric"


def test_every_field_kind_maps_to_a_python_type() -> None:
    """A new FieldKind must be given storage meaning, not silently default to one."""
    kinds = {field.kind for spec in ALL_INSTRUMENTS for field in spec.fields}
    assert kinds <= set(FieldKind)
