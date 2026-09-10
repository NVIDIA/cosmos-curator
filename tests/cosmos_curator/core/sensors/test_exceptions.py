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

"""Tests for the public structured alignment-failure contract."""

import pickle

import numpy as np

from cosmos_curator.core.sensors.exceptions import AlignmentError, AlignmentFailureReason


def test_reason_values_are_the_machine_readable_vocabulary() -> None:
    """The reason vocabulary is a stable public contract, so pin its wire values.

    Every reason names a case where the recording cannot serve the request, which
    is what makes ``AlignmentError`` a quarantine signal. A sensor returning a
    payload no recording could produce is a defect and keeps its own exception
    type, so it has no reason here.
    """
    assert [reason.value for reason in AlignmentFailureReason] == [
        "empty_batch",
        "tolerance_exceeded",
        "gps_no_eligible_fix",
        "gps_max_delta_exceeded",
        "gps_missing_bracket",
        "gps_invalid_support_position",
        "gps_max_gap_exceeded",
    ]


def test_alignment_error_is_not_a_value_error() -> None:
    """An alignment-policy outcome must not be swallowed by a caller's ``except ValueError``."""
    error = AlignmentError(AlignmentFailureReason.EMPTY_BATCH, "no rows")

    assert isinstance(error, Exception)
    assert not isinstance(error, ValueError)


def test_structured_fields_are_readable_without_payload_inspection() -> None:
    """Callers diagnose a failure from the exception alone, not from the modality payload."""
    requested = np.array([100, 200], dtype=np.int64)
    source = np.array([100, 190], dtype=np.int64)
    error = AlignmentError(
        AlignmentFailureReason.TOLERANCE_EXCEEDED,
        "max_delta_ns=5 exceeded",
        sensor_id="front",
        align_timestamps_ns=requested,
        sensor_timestamps_ns=source,
        delta_ns=10,
        max_delta_ns=5,
        known_source_timestamps_ns=np.array([90, 190], dtype=np.int64),
        gap_ns=100,
        max_gap_ns=50,
    )

    assert error.reason is AlignmentFailureReason.TOLERANCE_EXCEEDED
    assert error.detail == "max_delta_ns=5 exceeded"
    assert error.sensor_id == "front"
    np.testing.assert_array_equal(error.align_timestamps_ns, requested)
    np.testing.assert_array_equal(error.sensor_timestamps_ns, source)
    assert error.delta_ns == 10
    assert error.max_delta_ns == 5
    np.testing.assert_array_equal(error.known_source_timestamps_ns, [90, 190])
    assert error.gap_ns == 100
    assert error.max_gap_ns == 50


def test_optional_fields_default_to_none() -> None:
    """Only ``reason`` and ``detail`` are required; every structured field is optional."""
    error = AlignmentError(AlignmentFailureReason.EMPTY_BATCH, "no rows")

    assert error.sensor_id is None
    assert error.align_timestamps_ns is None
    assert error.sensor_timestamps_ns is None
    assert error.delta_ns is None
    assert error.max_delta_ns is None
    assert error.known_source_timestamps_ns is None
    assert error.gap_ns is None
    assert error.max_gap_ns is None


def test_str_reports_reason_detail_and_sensor_id() -> None:
    """The rendered message carries the sensor id so logs identify the failing sensor."""
    assert str(AlignmentError(AlignmentFailureReason.EMPTY_BATCH, "no rows", sensor_id="imu")) == (
        "imu: empty_batch: no rows"
    )
    assert str(AlignmentError(AlignmentFailureReason.EMPTY_BATCH, "no rows")) == "empty_batch: no rows"


def test_attached_timelines_are_copies_the_error_owns() -> None:
    """Window timestamps are a read-only view over caller memory; an error outlives that frame."""
    requested = np.array([100, 200], dtype=np.int64)
    selected = np.array([100, 190], dtype=np.int64)
    error = AlignmentError(
        AlignmentFailureReason.TOLERANCE_EXCEEDED,
        "mismatch",
        align_timestamps_ns=requested,
        sensor_timestamps_ns=selected,
    )

    requested[0] = -1
    selected[0] = -1

    np.testing.assert_array_equal(error.align_timestamps_ns, [100, 200])
    np.testing.assert_array_equal(error.sensor_timestamps_ns, [100, 190])


def test_pickling_preserves_a_sensor_id_stamped_after_construction() -> None:
    """Pickling must carry the instance dict, not the ``args`` snapshot taken at construction.

    These two tests exist to guard one specific footgun, not to verify a
    transport requirement. attrs takes a special path for any exception class
    (it keys on ``BaseException``, which ``Exception`` is a subclass of), and on
    that path hand-rolling ``__reduce__`` or flipping ``slots`` silently drops
    state instead of failing loudly. This branch shipped exactly that bug once.
    Plain pickling of an untouched class needs no test — that is CPython's job —
    so there is no bare round-trip case here.

    ``SensorGroup`` stamps ``sensor_id`` after construction, which ``args`` never
    sees. Mutation-checked: ``slots=True`` loses the stamp.
    """
    error = AlignmentError(AlignmentFailureReason.EMPTY_BATCH, "no rows")
    error.sensor_id = "front"

    assert pickle.loads(pickle.dumps(error)).sensor_id == "front"  # noqa: S301


def test_pickling_preserves_attributes_a_caller_attached() -> None:
    """A hand-written ``__reduce__`` drops the instance dict; this is what catches that.

    Mutation-checked against both a custom ``__reduce__`` and ``slots=True``.
    """
    error = AlignmentError(AlignmentFailureReason.EMPTY_BATCH, "no rows")
    error.window_index = 7  # type: ignore[attr-defined]

    restored = pickle.loads(pickle.dumps(error))  # noqa: S301

    assert restored.window_index == 7  # type: ignore[attr-defined]
