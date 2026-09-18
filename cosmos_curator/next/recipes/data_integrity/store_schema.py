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

"""Arrow schemas for the data-integrity store.

Four kinds of dataset, described in full in
``docs/curator/design/data-integrity-store-schema.md``:

* ``stream.lance`` -- one row per stream a run touched, *including* ones that could
  not be opened. A stream that fails to open produces no measurements, so without
  this dataset the broken stream would vanish from the store entirely.
* ``measurements/<metric>.lance`` -- one row per stream, one dataset per metric,
  with the measured fields as native typed columns.
* ``evaluation.lance`` -- one row per policy x stream x metric. The one tall
  dataset, because verdicts have a uniform shape whatever produced them.
* ``run.lance`` -- one row per *finished* run. Written last, so it is what makes a
  run's rows visible; see :func:`~.store.commit_run`.

Each of the first three has a session-grain twin -- ``session.lance``,
``session_measurements/<metric>.lance`` and ``session_evaluation.lance`` -- carrying
``session_id`` where its per-stream original carries ``stream_id``. Parallel datasets
rather than a widened identity prefix, because ``stream_id`` being ``nullable=False``
is what spares every existing reader from having to ask which grain a row is. The
two grains also key differently, so one dataset would carry two rules for resolving
its newest row.

Measurement columns are typed rather than packed into a JSON payload so that
``NaN`` needs no encoding: JSON cannot represent it, but Arrow stores it natively
as a float64 bit pattern and keeps null in a separate validity bitmap. That gives
three distinct states for free -- null (the metric never ran), ``NaN`` (it ran and
the value is undefined), and a real value -- where a JSON payload would need a
hand-written sentinel contract and would quietly degrade the middle one into the
first.

Every metric column is nullable regardless of whether the measurement itself can
produce a null, because a "never ran" row carries the identity columns and nothing
else. ``InstrumentSpec.fields`` is what records whether a null is a value the
metric can legitimately produce.
"""

import pyarrow as pa  # type: ignore[import-untyped]

from cosmos_curator.core.sensors.data_integrity.instruments import (
    INSTRUMENTS,
    SESSION_INSTRUMENTS,
    FieldKind,
    FieldSpec,
    InstrumentSpec,
)

#: Bumped when the layout of the store as a whole changes (a dataset added, moved,
#: or renamed). Per-metric schema changes are visible in the dataset's own schema,
#: which Lance records, so they do not need a number here.
#:
#: 2 added the three session-grain datasets. Additive: a version-1 store is read by
#: this code unchanged, since a dataset that was never written reads as no rows.
#:
#: 3 added ``timestamp_source`` to :data:`STREAM_SCHEMA` and to :data:`SESSION_SCHEMA`
#: -- one clock per stream, and the roll-up of them per session. Unlike 2, this one is *not*
#: additive: :func:`~.store.append_rows` appends with a plain
#: ``lance.write_dataset(..., mode="append")``, which requires the incoming schema to
#: match the dataset's, so a store written at version 1 or 2 cannot take new rows and
#: needs a fresh root. Deliberate rather than mitigated -- migration machinery for a
#: store whose only readers are the two CLIs and the rollup would cost more than
#: re-measuring, and re-measuring is cheap because the sources are unchanged.
#:
#: Version 3 rows may contain historical ``timestamp_source="capture"`` or the
#: canonical ``"sensor"`` value. Current enum readers normalize ``"capture"`` to
#: ``"sensor"``, and writers emit only ``"sensor"``. This vocabulary migration
#: does not change Arrow layout, so it deliberately does not bump this version.
STORE_SCHEMA_VERSION = 3

STREAM_DATASET = "stream.lance"
EVALUATION_DATASET = "evaluation.lance"
RUN_DATASET = "run.lance"
MEASUREMENTS_DIR = "measurements"
SESSION_DATASET = "session.lance"
SESSION_EVALUATION_DATASET = "session_evaluation.lance"
SESSION_MEASUREMENTS_DIR = "session_measurements"
MANIFEST_NAME = "manifest.json"

# Microseconds, not nanoseconds: this is wall-clock provenance ("when did we write
# this row"), not the sample timeline, which stays int64 ns in its own columns.
_TIMESTAMP = pa.timestamp("us", tz="UTC")

_ARROW_TYPES = {
    FieldKind.INT: pa.int64(),
    FieldKind.FLOAT: pa.float64(),
    FieldKind.BOOL: pa.bool_(),
}


def metric_dataset_path(metric_name: str) -> str:
    """Build the store-relative path of one per-stream metric's dataset."""
    return f"{MEASUREMENTS_DIR}/{metric_name}.lance"


def session_metric_dataset_path(metric_name: str) -> str:
    """Build the store-relative path of one session-grain metric's dataset.

    A separate directory rather than a suffix inside ``measurements/``, so the two
    grains cannot collide on a name and so listing either one is a directory read.
    """
    return f"{SESSION_MEASUREMENTS_DIR}/{metric_name}.lance"


STREAM_SCHEMA: pa.Schema = pa.schema(
    [
        # Identity. stream_id is the dedup key; source is the readable form of the
        # same thing and is deliberately not a key (see identity.py).
        pa.field("stream_id", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("created_at", _TIMESTAMP, nullable=False),
        # Session context. Null for a single-stream run, which is the only structural
        # difference between what di-check and di-session write.
        pa.field("session_id", pa.string()),
        pa.field("session_path", pa.string()),
        pa.field("locator_namespace", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("relative_key", pa.string()),
        pa.field("selector_type", pa.string(), nullable=False),
        pa.field("selector_value", pa.string(), nullable=False),
        # Content identity, for the staleness question the other signals cannot see:
        # the code is unchanged but the bytes behind the path were replaced. Free --
        # the HEAD that fills these is already issued for the progress display.
        pa.field("content_etag", pa.string()),
        pa.field("content_size_bytes", pa.int64()),
        pa.field("content_last_modified", _TIMESTAMP),
        # Sensor-level facts. Nullable both because an errored stream has none and
        # because they are video-shaped: an IMU or GPS stream has no codec.
        pa.field("codec_name", pa.string()),
        pa.field("has_bframes", pa.bool_()),
        pa.field("num_samples", pa.int64()),
        pa.field("start_ns", pa.int64()),
        pa.field("end_ns", pa.int64()),
        pa.field("expected_hz", pa.float64()),
        pa.field("expected_hz_source", pa.string()),
        # Which clock every timing column above was measured on. Persisted rather than
        # derived at read time because it cannot be recovered later: the row keeps the
        # numbers, not the sidecar that explains them. A query for streams whose
        # cadence is real has to filter on this.
        pa.field("timestamp_source", pa.string()),
        # Non-null means the source could not be read; this stream has no rows in any
        # metric dataset.
        pa.field("error", pa.string()),
    ]
)

#: What a verdict row records about its own provenance, shared by both grains' verdict
#: schemas. Factored out for the reason :data:`MEASUREMENT_IDENTITY_FIELDS` is, and with
#: one more: :func:`~.store._verdict_columns` already fills these columns from a single
#: place, so two hand-written field lists could drift from each other while each still
#: matched the builder.
_VERDICT_PROVENANCE_FIELDS: list[pa.Field] = [
    # run_id is who wrote this verdict; measurement_run_id is whose facts were
    # judged. Equal on the first pass, different after a re-judge, so
    # "run_id <> measurement_run_id" isolates re-evaluated rows.
    pa.field("run_id", pa.string(), nullable=False),
    pa.field("measurement_run_id", pa.string(), nullable=False),
    pa.field("policy_id", pa.string(), nullable=False),
    pa.field("thresholds_json", pa.string(), nullable=False),
    pa.field("created_at", _TIMESTAMP, nullable=False),
]

#: The verdict itself, on the same terms: identical whether the subject is a stream or
#: a session, because the evaluation path that produces it is the same one.
_VERDICT_OUTCOME_FIELDS: list[pa.Field] = [
    pa.field("metric_name", pa.string(), nullable=False),
    # The single verdict column. An earlier draft also carried the kernel's
    # PASS/FAIL as a separate column, but CheckStatus is derived from it, so the
    # two could never disagree and only invited the question of which to filter on.
    pa.field("check_status", pa.string(), nullable=False),
    # Null together when nothing was judged, which is what makes it structurally
    # impossible to store a verdict for a measurement that was never taken.
    pa.field("margin", pa.float64()),
    pa.field("threshold", pa.float64()),
    pa.field("reason", pa.string()),
    pa.field("instrument_version", pa.int32(), nullable=False),
]

EVALUATION_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("stream_id", pa.string(), nullable=False),
        *_VERDICT_PROVENANCE_FIELDS,
        # Nullable, unlike its session-grain twin: di-check judges a stream that
        # belongs to no session.
        pa.field("session_path", pa.string()),
        pa.field("source", pa.string(), nullable=False),
        *_VERDICT_OUTCOME_FIELDS,
    ]
)

RUN_SCHEMA: pa.Schema = pa.schema(
    [
        # The presence of a row *is* the statement: a run whose rows are all written
        # commits itself here, and one that died partway through never gets a row, so
        # readers can tell a finished run from an abandoned one without a status flag.
        pa.field("run_id", pa.string(), nullable=False),
        # When the run's rows are stamped, versus when it finished. The gap between
        # them is how long the run took to write.
        pa.field("created_at", _TIMESTAMP, nullable=False),
        pa.field("committed_at", _TIMESTAMP, nullable=False),
        # Which entry point wrote it: "di-check", "di-session", "di-reevaluate", or
        # "data-integrity" for the Ray Data pipeline, which writes many sessions per run.
        pa.field("tool", pa.string(), nullable=False),
        pa.field("session_path", pa.string()),
        pa.field("policy_id", pa.string(), nullable=False),
        # Null for a re-evaluation, which writes verdicts without touching a stream.
        pa.field("num_streams", pa.int64()),
        pa.field("store_schema_version", pa.int32(), nullable=False),
    ]
)

#: Carried by every metric dataset ahead of that metric's own columns. Named
#: explicitly (rather than reusing a slice of STREAM_SCHEMA) because these are the
#: columns a metric row needs to stand on its own: the dedup key, the run that wrote
#: it, and the version of the code that produced the numbers.
MEASUREMENT_IDENTITY_FIELDS: list[pa.Field] = [
    pa.field("stream_id", pa.string(), nullable=False),
    pa.field("run_id", pa.string(), nullable=False),
    pa.field("created_at", _TIMESTAMP, nullable=False),
    pa.field("session_path", pa.string()),
    pa.field("source", pa.string(), nullable=False),
    pa.field("instrument_version", pa.int32(), nullable=False),
    # Three states, not two: true / false = the metric ran and its result is
    # defined / undefined; null = the metric never ran (no usable expected rate), in
    # which case every metric column below is null too.
    pa.field("is_defined", pa.bool_()),
]

MEASUREMENT_IDENTITY_NAMES: tuple[str, ...] = tuple(field.name for field in MEASUREMENT_IDENTITY_FIELDS)


def _metric_field(spec: FieldSpec) -> pa.Field:
    """Translate one declared measurement field into an Arrow field."""
    return pa.field(spec.name, _ARROW_TYPES[spec.kind])


def measurement_schema(spec: InstrumentSpec) -> pa.Schema:
    """Build one metric's Arrow schema: the shared identity prefix plus its own fields."""
    return pa.schema([*MEASUREMENT_IDENTITY_FIELDS, *(_metric_field(field) for field in spec.fields)])


#: Every metric's schema, by metric name.
MEASUREMENT_SCHEMAS: dict[str, pa.Schema] = {spec.name: measurement_schema(spec) for spec in INSTRUMENTS}


SESSION_SCHEMA: pa.Schema = pa.schema(
    [
        # session_id is the dedup key; session_path is the readable form of the same
        # thing, on the same terms stream_id and source are (see identity.py).
        pa.field("session_id", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("created_at", _TIMESTAMP, nullable=False),
        pa.field("session_path", pa.string(), nullable=False),
        pa.field("locator_namespace", pa.string(), nullable=False),
        # What the run found under this session, and how much of it was not usable.
        # A session listed but empty is 0 and 0, which is a finding about the input.
        pa.field("num_streams", pa.int64(), nullable=False),
        pa.field("num_unreadable", pa.int64(), nullable=False),
        # Which clock this session's streams were measured on, rolled up by
        # :func:`~cosmos_curator.core.sensors.data_integrity.results.session_timestamp_source`:
        # ``sensor`` only when every contributing stream had one, else ``container_pts``
        # or ``mixed``. Null means no stream contributed bounds -- an unlisted or empty
        # session has no clock rather than a bad one. This is what makes a never-ran
        # session-grain measurement legible: the metric row says only that it did not
        # run, and this column says why.
        pa.field("timestamp_source", pa.string()),
        # Non-null means the session itself could not be listed: no streams were
        # found, so it has no rows in any other dataset. The session-level analogue
        # of stream.error, and the reason one stale path in a large manifest costs a
        # row rather than every measurement the run had already taken.
        pa.field("error", pa.string()),
    ]
)

#: Carried by every session-grain metric dataset, ahead of that metric's own columns.
#: :data:`MEASUREMENT_IDENTITY_FIELDS` with ``session_id`` where ``stream_id`` sits and
#: no ``source``: a session's readable name *is* its ``session_path``, which is already
#: here, and there is no one source to name -- picking one of the session's streams
#: would credit it for a measurement over all of them.
SESSION_MEASUREMENT_IDENTITY_FIELDS: list[pa.Field] = [
    pa.field("session_id", pa.string(), nullable=False),
    pa.field("run_id", pa.string(), nullable=False),
    pa.field("created_at", _TIMESTAMP, nullable=False),
    pa.field("session_path", pa.string(), nullable=False),
    pa.field("instrument_version", pa.int32(), nullable=False),
    # Three states, as in MEASUREMENT_IDENTITY_FIELDS. Here false is the common one:
    # a session of a single sensor has nothing to compare, which is measured and
    # undefined rather than never measured.
    pa.field("is_defined", pa.bool_()),
]


def session_measurement_schema(spec: InstrumentSpec) -> pa.Schema:
    """Build one session-grain metric's Arrow schema."""
    return pa.schema([*SESSION_MEASUREMENT_IDENTITY_FIELDS, *(_metric_field(field) for field in spec.fields)])


#: Every session-grain metric's schema, by metric name.
SESSION_MEASUREMENT_SCHEMAS: dict[str, pa.Schema] = {
    spec.name: session_measurement_schema(spec) for spec in SESSION_INSTRUMENTS
}

#: :data:`EVALUATION_SCHEMA` column for column, with ``session_id`` for ``stream_id``
#: and no ``source``. Everything section 4 of the schema doc says about a verdict row
#: holds here unchanged, including ``margin`` and ``threshold`` being null together on
#: a SKIPPED row.
SESSION_EVALUATION_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("session_id", pa.string(), nullable=False),
        *_VERDICT_PROVENANCE_FIELDS,
        # Not nullable, unlike its per-stream twin: a session verdict is keyed on the
        # session it judges, so there is always a path.
        pa.field("session_path", pa.string(), nullable=False),
        *_VERDICT_OUTCOME_FIELDS,
    ]
)
