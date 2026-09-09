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

"""What a run leaves behind about itself: manifests and failure reports.

Failures are artifacts rather than rows. A failure is not a clip and shares none
of its fields, so carrying both down one stream makes every later stage sort
them apart -- and makes the run summary a second consumer of the stream the next
stage is reading.

One file per writer, never a shared one. Planning knows which spans no rig could
serve; splitting knows which encodes failed. They run in different tasks, so a
single report per session would be two writers racing for one object.
"""

import json
from typing import Any

from cosmos_curator.core.sensors.exceptions import AlignmentError
from cosmos_curator.next.media.spans import Span
from cosmos_curator.next.recipes.multimodal_split.uris import clip_dir_uri
from cosmos_curator.next.utils.storage import write_media

_ERRORS_PREFIX = "errors"
# Named for the stage that wrote it, so a directory of them says which step of a
# run went wrong without opening one.
_PLAN_ERROR_NAME = "span_planner.json"
_MANIFEST_NAME = "clip.json"
# Long enough to name a cause, short enough that a thousand of them stay readable.
_MESSAGE_LIMIT = 500
_FAIL_CLIP = "fail_clip"
_UNCLASSIFIED = "unclassified"


def errors_prefix(output_path: str) -> str:
    """Return the tree every failure report is written under.

    One listing of it finds them all, which is what makes reading them back cost
    a request rather than a request per session.
    """
    return f"{output_path.rstrip('/')}/{_ERRORS_PREFIX}"


def plan_error_uri(output_path: str, session_id: str) -> str:
    """Return where a session's planning failures are written, for a reader with the ID."""
    return f"{output_path.rstrip('/')}/{_ERRORS_PREFIX}/{session_id}/{_PLAN_ERROR_NAME}"


def sensor_error_uri(output_path: str, session_id: str, sensor_key: str) -> str:
    """Return where one sensor's split failures are written for a session."""
    return f"{output_path.rstrip('/')}/{_ERRORS_PREFIX}/{session_id}/split_session.{sensor_key}.json"


def write_plan_errors(
    session_id: str,
    failures: list[dict[str, Any]],
    *,
    output_path: str,
    storage_profile: str,
) -> None:
    """Record why planning refused what it refused, and write nothing otherwise.

    Only on failure. A receipt per healthy session is a write per session before
    any media exists, and the thing it defended against -- a session vanishing
    unnoticed -- is not something Ray Data allows: a task that raises kills the
    dataset, so a run that lost a session fails rather than under-reporting one.

    Each entry carries the stage that failed and the exception behind it,
    because "this session produced nothing" is not a reason.
    """
    if not failures:
        return
    payload = {
        "source_session_id": session_id,
        "stage": "span_planner",
        "failures": failures,
    }
    write_media(
        plan_error_uri(output_path, session_id),
        json.dumps(payload, indent=2).encode(),
        storage_profile=storage_profile,
    )


def write_sensor_errors(
    session_id: str,
    sensor_key: str,
    failures: list[dict[str, Any]],
    *,
    output_path: str,
    storage_profile: str,
) -> None:
    """Record one sensor's failures for a session, and write nothing when it had none.

    Sparse on purpose: a healthy run of ten thousand sessions should not leave a
    hundred and twenty thousand empty objects behind to be listed.
    """
    if not failures:
        return
    payload = {
        "source_session_id": session_id,
        "sensor_key": sensor_key,
        "failures": failures,
    }
    write_media(
        sensor_error_uri(output_path, session_id, sensor_key),
        json.dumps(payload, indent=2).encode(),
        storage_profile=storage_profile,
    )


def failure(  # noqa: PLR0913 -- a diagnostic is exactly this many independent facts
    session_id: str,
    sensor_key: str | None,
    span: Span | None,
    stage: str,
    error: BaseException,
    clip_id: str | None = None,
) -> dict[str, Any]:
    """Describe one failure in the terms a quarantine reader needs.

    ``sensor_key`` is absent for a failure that belongs to the session rather
    than to one of its sensors -- listing it, or finding no window its views
    share. Naming a sensor there would blame one for a fault none of them had.

    ``terminal_policy`` is a name rather than a boolean because the two things a
    boolean would conflate are not opposites. An :class:`AlignmentError` says the
    session does not contain what the grid asked for, which is deterministic:
    re-running decodes the same file against the same grid and fails the same
    way, so the clip is finished. Any other exception says nothing at all -- a
    decoder fault and an S3 timeout arrive identically here, and one is worth
    retrying while the other is not. ``exceptions.py`` warns against exactly the
    inference a boolean invites, that "not an AlignmentError" means "not a data
    problem", so the unclassified case gets its own name instead of being folded
    into either answer.
    """
    return {
        "source_session_id": session_id,
        "sensor_key": sensor_key,
        "clip_id": clip_id,
        "start_ns": span.start_ns if span is not None else None,
        "exclusive_end_ns": span.end_ns if span is not None else None,
        "stage": stage,
        "category": type(error).__name__,
        "message": str(error)[:_MESSAGE_LIMIT],
        "terminal_policy": _FAIL_CLIP if isinstance(error, AlignmentError) else _UNCLASSIFIED,
    }


def write_clip_manifest(  # noqa: PLR0913 -- a clip is placed by exactly these facts
    session_id: str,
    clip_id: str,
    span: Span,
    cameras: list[str],
    *,
    output_path: str,
    sampling_fps: int,
    storage_profile: str,
) -> None:
    """Write what a clip needs to place itself, for a reader holding only the directory.

    A clip's timeline says where its own frames sit; nothing else in the
    directory says which session was cut, or which part of it. Written as an
    artifact rather than left to the row stream so that a consumer given a path
    does not also need the table that produced it.

    ``planned_cameras`` is named for what it is. The manifest is written during
    planning, before any camera has encoded, because no single camera task knows
    which other views a clip holds -- so the field is what a clip was meant to
    hold, and the media beside it is what it holds. They differ when an encode
    fails, which that camera reports under ``errors/``. A consumer that needs
    completeness compares the two, which it can do with the directory it already
    has; one that reads the field as a promise would be reading a lie.
    """
    manifest = {
        "clip_id": clip_id,
        "parent_id": session_id,
        "start_ns": span.start_ns,
        "exclusive_end_ns": span.end_ns,
        "sampling_fps": sampling_fps,
        "planned_cameras": sorted(cameras),
    }
    destination = f"{clip_dir_uri(output_path, session_id, clip_id)}/{_MANIFEST_NAME}"
    write_media(destination, json.dumps(manifest, indent=2).encode(), storage_profile=storage_profile)
