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

"""Public structured exceptions for expected sensor alignment failures."""

import enum

import attrs
import numpy as np
import numpy.typing as npt


def _owned_copy(value: npt.NDArray[np.int64] | None) -> npt.NDArray[np.int64] | None:
    """Copy an attached timeline so the error owns its data.

    ``SamplingWindow.timestamps_ns`` is a read-only *view* over an array the
    caller still owns, and a grid's windows all share one base. An error is a
    diagnostic that outlives the frame it describes, so it must not report
    timestamps that can change afterwards or pin a whole grid.
    """
    return None if value is None else np.array(value, copy=True)


class AlignmentFailureReason(enum.StrEnum):
    """Machine-readable vocabulary for a recording that cannot serve a request.

    Every reason here means the same kind of thing: nothing is broken, the data
    simply does not contain what the sampling grid asked for, and the right
    response is to drop that input and keep processing. That is what makes
    ``AlignmentError`` usable as a quarantine signal.

    Sensor defects are deliberately absent. A payload no recording could produce
    — the wrong row count, a timeline that was never requested — means the sensor
    is wrong, and the fix is to the code rather than to the dataset. Those keep
    their own exception types so they fail loudly; quarantining them would
    discard good recordings while hiding the defect. Programming errors are
    excluded for the same reason.

    This vocabulary is partial, not exhaustive over bad data. Corrupt media and
    unreadable sources are data conditions no code fix repairs, but the group
    cannot recognise them, so they keep their own types too — because classifying
    them is out of scope here, not because they mean "fix the code". A caller
    therefore needs an unclassified bucket alongside these reasons, and should
    not read "not an AlignmentError" as "not a data problem".
    """

    EMPTY_BATCH = "empty_batch"
    TOLERANCE_EXCEEDED = "tolerance_exceeded"
    GPS_NO_ELIGIBLE_FIX = "gps_no_eligible_fix"
    GPS_MAX_DELTA_EXCEEDED = "gps_max_delta_exceeded"
    GPS_MISSING_BRACKET = "gps_missing_bracket"
    GPS_INVALID_SUPPORT_POSITION = "gps_invalid_support_position"
    GPS_MAX_GAP_EXCEEDED = "gps_max_gap_exceeded"
    EGO_TRAJECTORY_NO_ELIGIBLE_POSE = "ego_trajectory_no_eligible_pose"
    EGO_TRAJECTORY_MAX_DELTA_EXCEEDED = "ego_trajectory_max_delta_exceeded"


@attrs.define(slots=False, eq=False)
class AlignmentError(Exception):
    """An expected alignment failure for one sampling window.

    Deliberately not a ``ValueError``: a caller's ``except ValueError`` around
    argument validation should not silently absorb a sensor dropout. Callers
    handle the first failed window with one ``try/except AlignmentError`` around
    iteration and branch on ``reason``, never on the message text.

    ``sensor_id`` is mutable so that ``SensorGroup`` can attribute a propagated
    error without losing the original traceback. It attributes unconditionally,
    so the field is always the group's key for that sensor.

    Attributes:
        reason: machine-readable failure category.
        detail: human-readable description; not part of the machine contract.
        sensor_id: the failing sensor's id. Set by ``SensorGroup`` to its
            configured key, so it can be used to look that sensor up. ``None``
            only when raised below the sensor layer, as the sampler does.
        align_timestamps_ns: the reference timeline the window asked to be served.
        sensor_timestamps_ns: selected sensor-clock timestamps, when a failure
            has a selected source row.
        delta_ns: observed nearest-selection distance.
        max_delta_ns: the policy's configured tolerance, the limit ``delta_ns``
            broke.
        known_source_timestamps_ns: source alignment timestamps relevant to the
            failure: selected timestamps, interpolation brackets, or all known
            source support when no narrower diagnostic exists.
        gap_ns: observed source-bracket duration.
        max_gap_ns: configured interpolation gap limit.

    These reuse the package's own names on purpose. ``align_timestamps_ns`` and
    ``sensor_timestamps_ns`` mean here exactly what they mean on a ``SensorData``
    payload, and ``max_delta_ns`` is the ``NearestTimestampPolicy`` field of that
    name, so no translation is needed to read a failure against the data it
    describes.

    """

    reason: AlignmentFailureReason
    detail: str
    sensor_id: str | None = None
    align_timestamps_ns: npt.NDArray[np.int64] | None = attrs.field(default=None, converter=_owned_copy)
    sensor_timestamps_ns: npt.NDArray[np.int64] | None = attrs.field(default=None, converter=_owned_copy)
    delta_ns: int | None = None
    max_delta_ns: int | None = None
    known_source_timestamps_ns: npt.NDArray[np.int64] | None = attrs.field(default=None, converter=_owned_copy)
    gap_ns: int | None = None
    max_gap_ns: int | None = None

    def __str__(self) -> str:
        """Render as ``[sensor_id: ]reason: detail``."""
        prefix = f"{self.sensor_id}: " if self.sensor_id else ""
        return f"{prefix}{self.reason}: {self.detail}"
