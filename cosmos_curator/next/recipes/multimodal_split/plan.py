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

"""Decide which spans of a session every camera can serve.

The only stage that holds a whole rig at once. Deciding what a clip *is* needs
every camera's timeline together -- a moment no view can serve is not a clip --
while producing one is a single camera's work. Separating them means a session
that cannot be covered never occupies a splitting slot, and the unit of split
work becomes one camera rather than a whole rig.

A camera's timeline and resolved URI travel on the row. Both were paid for here,
and reading them again in the next stage would be a second high-latency round
trip per camera for bytes this stage already holds.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from cosmos_curator.core.sensors.exceptions import AlignmentError, AlignmentFailureReason
from cosmos_curator.core.sensors.sampling.sampler import find_closest_indices
from cosmos_curator.next.media.spans import Span
from cosmos_curator.next.recipes.multimodal_split.config import ResolvedMultimodalSplitConfig
from cosmos_curator.next.recipes.multimodal_split.grids import (
    clip_duration_ns,
    plan_clip_spans,
    sampling_grid_ns,
    timeline_end_ns,
)
from cosmos_curator.next.recipes.multimodal_split.identities import make_clip_id, make_source_id
from cosmos_curator.next.recipes.multimodal_split.reports import (
    failure,
    write_clip_manifest,
    write_plan_errors,
)
from cosmos_curator.next.recipes.multimodal_split.storage import (
    list_session_paths,
    load_timeline_ns,
    resolve_sensor_pattern,
)

SENSOR_TYPE_CAMERA = "camera"

SPAN_PLAN_SCHEMA = pa.schema(
    [
        pa.field("source_session_id", pa.large_string(), nullable=False),
        pa.field("sensor_key", pa.large_string(), nullable=False),
        pa.field("sensor_uri", pa.large_string(), nullable=False),
        # Every sensor is a camera today. Named anyway, because splitting a rig
        # of cameras and an IMU is the same job on different media, and the
        # stage that reads this row has to know which it was handed.
        pa.field("sensor_type", pa.large_string(), nullable=False),
        pa.field("timeline_ns", pa.large_list(pa.int64()), nullable=False),
        pa.field(
            "spans",
            pa.large_list(
                pa.struct(
                    [
                        pa.field("clip_id", pa.large_string(), nullable=False),
                        pa.field("start_ns", pa.int64(), nullable=False),
                        pa.field("exclusive_end_ns", pa.int64(), nullable=False),
                    ]
                )
            ),
            nullable=False,
        ),
    ]
)


def span_planner(
    item: dict[str, Any],
    *,
    config: ResolvedMultimodalSplitConfig,
    storage_profile: str = "default",
) -> list[dict[str, Any]]:
    """Plan one session into spans, and emit one row per camera that will serve them.

    Emits nothing for a session no camera set can cover, so splitting is never
    handed work that was already known to be impossible. The plan report is
    written either way: a session that produced no rows because it was refused
    and one that produced none because a worker died look identical otherwise.
    """
    session_id = str(item["source_session_id"])
    session_uri = str(item["session_uri"])
    cameras = config.sensors.cameras

    # One listing for the whole rig. Resolving twelve wildcard patterns one at a
    # time lists the same session twelve times, which the previous comment here
    # claimed it did not.
    try:
        session_paths = list_session_paths(session_uri, storage_profile=storage_profile)
    except Exception as error:  # noqa: BLE001 -- a session that cannot be listed is one failed session
        write_plan_errors(
            session_id,
            [failure(session_id, None, None, "list_session", error)],
            output_path=config.output.path,
            storage_profile=storage_profile,
        )
        return []

    # Per sensor, so a failure names the sensor that failed. Attributing all of
    # them to the first configured one describes the wrong camera to whoever
    # reads the report.
    sensor_uris: dict[str, str] = {}
    timelines: dict[str, npt.NDArray[np.int64]] = {}
    failures: list[dict[str, Any]] = []
    for key, camera in cameras.items():
        try:
            uri = resolve_sensor_pattern(
                session_uri, camera.uri_pattern, storage_profile=storage_profile, session_paths=session_paths
            )
            timelines[key] = load_timeline_ns(
                uri, storage_profile=storage_profile, read_buffer_bytes=config.execution.read_buffer_bytes
            )
        except Exception as error:  # noqa: BLE001 -- one unreadable sensor is one failed sensor
            failures.append(failure(session_id, key, None, "open_sensor", error))
            continue
        sensor_uris[key] = uri

    # All or nothing: a clip is one moment across every view, so a rig missing a
    # sensor produces no clips rather than clips missing a view.
    if failures:
        write_plan_errors(session_id, failures, output_path=config.output.path, storage_profile=storage_profile)
        return []

    try:
        spans = plan_common_spans(timelines, config)
    except Exception as error:  # noqa: BLE001 -- a rig whose views never overlap is one failed session
        write_plan_errors(
            session_id,
            [failure(session_id, None, None, "plan_spans", error)],
            output_path=config.output.path,
            storage_profile=storage_profile,
        )
        return []

    # Where the session is, not what it is called: two prefixes can each hold a
    # ``session-000``, and the name alone would key both into the same clips.
    source_id = make_source_id(session_uri)
    servable, refused = partition_by_coverage(spans, timelines, config, session_id, source_id)
    write_plan_errors(session_id, refused, output_path=config.output.path, storage_profile=storage_profile)
    if not servable:
        return []

    planned = [
        {
            "clip_id": make_clip_id(source_id, span, config),
            "start_ns": span.start_ns,
            "exclusive_end_ns": span.end_ns,
        }
        for span in servable
    ]
    # Written here rather than after the media, because no single camera task
    # knows which other views a clip holds. Coverage is all-or-nothing, so the
    # planned set is the set -- unless an encode fails, which its camera reports.
    for span, entry in zip(servable, planned, strict=True):
        write_clip_manifest(
            session_id,
            str(entry["clip_id"]),
            span,
            sorted(cameras),
            output_path=config.output.path,
            output_fps=config.clip.output_fps,
            storage_profile=storage_profile,
        )

    return [
        {
            "source_session_id": session_id,
            "sensor_key": key,
            "sensor_uri": sensor_uris[key],
            "sensor_type": SENSOR_TYPE_CAMERA,
            "timeline_ns": timelines[key].tolist(),
            "spans": planned,
        }
        for key in cameras
    ]


def plan_common_spans(
    timelines: dict[str, npt.NDArray[np.int64]], config: ResolvedMultimodalSplitConfig
) -> tuple[Span, ...]:
    """Tile the window every camera was capturing into clip spans.

    The latest start and the earliest end, not one camera's own bounds. Cameras
    on a rig begin within a few frames of each other, so planning from one of
    them puts the first references before another view existed.
    """
    return plan_clip_spans(
        start_ns=max(int(timeline[0]) for timeline in timelines.values()),
        end_ns=min(timeline_end_ns(timeline) for timeline in timelines.values()),
        duration_ns=clip_duration_ns(config),
    )


def partition_by_coverage(
    spans: tuple[Span, ...],
    timelines: dict[str, npt.NDArray[np.int64]],
    config: ResolvedMultimodalSplitConfig,
    session_id: str,
    source_id: str,
) -> tuple[tuple[Span, ...], list[dict[str, Any]]]:
    """Split planned spans into those every camera can serve and those none may.

    Both identifiers, because they answer different questions: the report names
    the session a reader will go looking for, and the clip id keys on where that
    session is, so two prefixes holding the same directory name stay apart.

    Refused before anything is written, because a clip is one moment across every
    view: a span one camera cannot reach is not a clip missing a view, it is not
    a clip.
    """
    servable: list[Span] = []
    refused: list[dict[str, Any]] = []
    for span in spans:
        references = sampling_grid_ns(span, config)
        missing = first_uncovered(references, timelines, config.sensors.max_delta_ns)
        if missing is None:
            servable.append(span)
            continue
        key, worst_ns = missing
        error = AlignmentError(
            AlignmentFailureReason.TOLERANCE_EXCEEDED,
            f"camera {key!r} cannot serve this moment: its nearest observation is {worst_ns} ns from a "
            f"reference, further than the {config.sensors.max_delta_ns} ns the rig was given",
            sensor_id=key,
            align_timestamps_ns=references,
        )
        refused.append(failure(session_id, key, span, "coverage", error, make_clip_id(source_id, span, config)))
    return tuple(servable), refused


def first_uncovered(
    references_ns: npt.NDArray[np.int64],
    timelines: dict[str, npt.NDArray[np.int64]],
    max_delta_ns: int,
) -> tuple[str, int] | None:
    """Return the first camera that cannot reach every reference, and by how far."""
    for key, timeline in timelines.items():
        serving = timeline[find_closest_indices(timeline, references_ns)]
        worst_ns = max(int(abs(int(left) - int(right))) for left, right in zip(serving, references_ns, strict=True))
        if worst_ns > max_delta_ns:
            return key, worst_ns
    return None
