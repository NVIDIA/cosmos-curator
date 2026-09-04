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

"""Re-judge stored measurements under a new policy, without re-reading any source.

This is what the store is for. A measurement is a fact about the data and does not
change; a verdict is a judgment about that fact and changes whenever the thresholds
do. So tightening a threshold reads the stored facts, runs the *same* evaluation
path the CLIs use (:func:`~.instruments.evaluate_metric`), and appends a new
generation of verdicts under a new ``policy_id``. No source is opened, and the
previous verdicts stay exactly where they were.

The new rows carry ``run_id`` for the re-judging invocation and
``measurement_run_id`` for the run whose facts were judged, so
``run_id <> measurement_run_id`` isolates everything that was re-evaluated rather
than measured.
"""

import datetime
from typing import cast

from cosmos_curator.core.sensors.data_integrity.instruments import (
    INSTRUMENTS,
    SESSION_INSTRUMENTS,
    InstrumentSpec,
    Thresholds,
    evaluate_metric,
)
from cosmos_curator.core.sensors.data_integrity.results import CheckResult, CheckStatus
from cosmos_curator.core.utils.storage_cli import get_lance_storage_options
from cosmos_curator.next.recipes.data_integrity import store, store_schema

# Mirrors the reason the CLIs attach to a rate-dependent metric that never ran. Held
# here as well because a stored "never ran" row has no measurement to describe, and
# re-judging it must produce the same wording as the original run did.
REASON_NEVER_RAN = "skipped: metric did not run (no usable expected rate recorded)"


def _skipped_never_ran(spec: InstrumentSpec) -> CheckResult:
    """Build the verdict for a stored row whose metric never ran."""
    return CheckResult(
        name=spec.name,
        status=CheckStatus.SKIPPED,
        reason=REASON_NEVER_RAN,
        measurement=None,
        evaluation=None,
        raw_measurement=None,
    )


def _reevaluate_row(spec: InstrumentSpec, row: dict[str, object], thresholds: Thresholds) -> CheckResult:
    """Rebuild one stored measurement and judge it under ``thresholds``.

    ``is_defined is None`` marks a metric that never ran, which has no measurement
    to rebuild -- every metric column on that row is null. A ``False`` still has real
    numbers stored, so it is rebuilt and handed to the evaluator, which skips it for
    the same reason it did the first time.
    """
    if row.get("is_defined") is None:
        return _skipped_never_ran(spec)
    measurement = spec.from_row(row)
    return evaluate_metric(spec, measurement, thresholds)


def reevaluate(  # noqa: PLR0913 -- credentials plus provenance, all independent
    root: str,
    *,
    thresholds: Thresholds,
    run_id: str | None = None,
    created_at: datetime.datetime | None = None,
    s3_profile_name: str | None = None,
    endpoint_url: str | None = None,
) -> str:
    """Re-judge every stored measurement under ``thresholds`` and append the verdicts.

    Both grains: per-stream measurements become ``evaluation.lance`` rows and
    session-grain ones become ``session_evaluation.lance`` rows, so tightening a
    policy moves a session's verdict and its streams' verdicts together rather than
    leaving the two under different rules.

    Opens no source data: every input comes from the metric datasets. The written
    rows land under ``policy_id(thresholds)``, so re-running with an unchanged policy
    supersedes its own previous rows while a changed one opens a new generation
    beside the existing verdicts.

    Reads only committed measurements and commits itself the same way (see
    :func:`~.store.commit_run`), so a re-judge interrupted halfway leaves no partial
    generation of verdicts behind.

    Args:
        root: the store root to read from and append to.
        thresholds: the new policy.
        run_id: id for this re-judging invocation; minted when omitted.
        created_at: timestamp shared by every appended row; now (UTC) when omitted.
        s3_profile_name: AWS profile for an ``s3://`` store.
        endpoint_url: S3 endpoint override for S3-compatible stores.

    Returns:
        The ``run_id`` the new verdicts were written under.

    """
    run_id = run_id or store.new_run_id()
    created_at = created_at or datetime.datetime.now(datetime.UTC)
    storage_options = get_lance_storage_options(root, s3_profile_name=s3_profile_name, endpoint_url=endpoint_url)

    # One ledger read for every metric dataset, rather than one per read below.
    finished = store.completed_runs(root, s3_profile_name=s3_profile_name, endpoint_url=endpoint_url)

    rows: list[dict[str, object]] = []
    for spec in INSTRUMENTS:
        for measurement_row in store.read_measurements(
            root, spec.name, completed=finished, s3_profile_name=s3_profile_name, endpoint_url=endpoint_url
        ):
            result = _reevaluate_row(spec, measurement_row, thresholds)
            rows.append(
                store.build_evaluation_row(
                    spec,
                    result,
                    stream_id=str(measurement_row["stream_id"]),
                    source=str(measurement_row["source"]),
                    run_id=run_id,
                    # The facts belong to whichever run measured them, not to this one.
                    measurement_run_id=str(measurement_row["run_id"]),
                    created_at=created_at,
                    session_path=_optional_str(measurement_row.get("session_path")),
                    thresholds=thresholds,
                    instrument_version=cast("int", measurement_row["instrument_version"]),
                )
            )

    # The session-grain half, on the same terms. It is why the session verdict is a
    # table of its own rather than a column on the session row: a verdict welded to
    # the row it judges cannot be re-judged without rewriting the facts beside it.
    session_rows: list[dict[str, object]] = []
    for spec in SESSION_INSTRUMENTS:
        for measurement_row in store.read_session_measurements(
            root, spec.name, completed=finished, s3_profile_name=s3_profile_name, endpoint_url=endpoint_url
        ):
            result = _reevaluate_row(spec, measurement_row, thresholds)
            session_rows.append(
                store.build_session_evaluation_row(
                    spec,
                    result,
                    session_id=str(measurement_row["session_id"]),
                    run_id=run_id,
                    measurement_run_id=str(measurement_row["run_id"]),
                    created_at=created_at,
                    session_path=str(measurement_row["session_path"]),
                    thresholds=thresholds,
                    instrument_version=cast("int", measurement_row["instrument_version"]),
                )
            )

    store.append_rows(
        rows,
        store.join(root, store_schema.EVALUATION_DATASET),
        store_schema.EVALUATION_SCHEMA,
        storage_options,
    )
    store.append_rows(
        session_rows,
        store.join(root, store_schema.SESSION_EVALUATION_DATASET),
        store_schema.SESSION_EVALUATION_SCHEMA,
        storage_options,
    )
    # A re-judge is a run like any other, so it commits like one: without this its
    # verdicts would be written but invisible.
    store.commit_run(
        root,
        run_id=run_id,
        created_at=created_at,
        tool="di-reevaluate",
        session_path=None,
        thresholds=thresholds,
        # Null rather than zero: a re-judge touches no stream, it re-reads facts.
        num_streams=None,
        storage_options=storage_options,
    )
    return run_id


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)
