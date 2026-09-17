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
import pyarrow.compute as pc

from cosmos_curator.core.sensors.exceptions import AlignmentError, AlignmentFailureReason
from cosmos_curator.core.sensors.sampling.sampler import find_closest_indices
from cosmos_curator.core.sensors.types.types import TimestampSource
from cosmos_curator.next.media.spans import Span
from cosmos_curator.next.recipes.multimodal_split.config import SENSOR_TYPE_CAMERA, ResolvedMultimodalSplitConfig
from cosmos_curator.next.recipes.multimodal_split.grids import (
    clip_duration_ns,
    plan_clip_spans,
    sampling_grid_ns,
    timeline_end_ns,
)
from cosmos_curator.next.recipes.multimodal_split.identities import (
    MEDIA_CONTRACT_VERSION,
    make_clip_id,
    make_episode_id,
    make_source_id,
    make_split_contract_digest,
)
from cosmos_curator.next.recipes.multimodal_split.records import RECORD_SCHEMA_VERSION
from cosmos_curator.next.recipes.multimodal_split.recovery import ReconciliationMap, reconcile_planned_episode
from cosmos_curator.next.recipes.multimodal_split.reports import (
    failure,
    write_clip_manifest,
    write_plan_errors,
)
from cosmos_curator.next.recipes.multimodal_split.storage import (
    list_session_paths,
    load_container_pts_ns,
    load_mcap_bounds,
    load_timeline_ns,
    resolve_sensor_pattern,
)
from cosmos_curator.next.recipes.multimodal_split.uris import timeline_uri

CAMERA_PLAN = "camera_plan"
MCAP_PLAN = "mcap_plan"
ALREADY_CANONICAL = "already_canonical"

_PLANNED_SPAN_STRUCT = pa.struct(
    [
        pa.field("clip_id", pa.string(), nullable=False),
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_session_id", pa.large_string(), nullable=False),
        pa.field("source_session_location", pa.large_string(), nullable=False),
        pa.field("start_ns", pa.int64(), nullable=False),
        pa.field("exclusive_end_ns", pa.int64(), nullable=False),
        pa.field("sampling_fps", pa.int32(), nullable=False),
        pa.field("expected_row_count", pa.int64(), nullable=False),
        pa.field("timestamp_source", pa.string(), nullable=False),
        pa.field("record_schema_version", pa.int32(), nullable=False),
        pa.field("media_contract_version", pa.int32(), nullable=False),
        pa.field("split_contract_digest", pa.string(), nullable=False),
        pa.field(
            "planned_camera_keys",
            pa.large_list(pa.field("item", pa.string(), nullable=False)),
            nullable=False,
        ),
        pa.field(
            "planned_sensor_keys",
            pa.large_list(pa.field("item", pa.string(), nullable=False)),
            nullable=False,
        ),
    ]
)

PLANNING_RESULT_SCHEMA = pa.schema(
    [
        pa.field("result_kind", pa.string(), nullable=False),
        pa.field("source_session_id", pa.large_string(), nullable=False),
        pa.field("sensor_key", pa.large_string()),
        pa.field("sensor_uri", pa.large_string()),
        # Every sensor is a camera today. Named anyway, because splitting a rig
        # of cameras and an IMU is the same job on different media, and the
        # stage that reads this row has to know which it was handed.
        pa.field("sensor_type", pa.large_string()),
        pa.field("timeline_ns", pa.large_list(pa.int64())),
        pa.field(
            "spans",
            pa.large_list(pa.field("item", _PLANNED_SPAN_STRUCT, nullable=False)),
        ),
        pa.field("episode_id", pa.string()),
    ]
)
# Compatibility name for callers that treated this as the planner's output schema.
SPAN_PLAN_SCHEMA = PLANNING_RESULT_SCHEMA

REMAINING_EPISODE_ID_SCHEMA = pa.schema([pa.field("split_clip_id", pa.string(), nullable=False)])


def remaining_episode_ids_batch(plans: pa.Table) -> pa.Table:
    """Project the unique episode IDs represented by a camera-plan block."""
    spans = pc.list_flatten(plans["spans"])
    clip_ids = pc.struct_field(spans, "clip_id").to_pylist()
    episode_ids = sorted({str(clip_id) for clip_id in clip_ids})
    return pa.Table.from_pydict(
        {"split_clip_id": episode_ids},
        schema=REMAINING_EPISODE_ID_SCHEMA,
    )


def open_cameras(
    session_id: str,
    session_uri: str,
    session_paths: list[str],
    *,
    config: ResolvedMultimodalSplitConfig,
    storage_profile: str,
) -> tuple[dict[str, str], dict[str, npt.NDArray[np.int64]], set[str], list[dict[str, Any]]]:
    """Resolve and read every camera in *session_uri*, and say which ones failed.

    Failures are per sensor, so one names the camera that failed. Attributing all
    of them to the first configured one describes the wrong camera to whoever
    reads the report.

    A camera configured ``required: false`` that the session does not carry is
    left out rather than reported. Its absence is not a fault, and the episode's
    resolved key list is what records it -- there is no separate absence to write.
    Every camera optional and every camera missing is still a session that
    produced nothing, though, and producing nothing with an empty error file is
    the one outcome that file exists to prevent, so that case does become a
    failure.

    Absence means the session's listing does not hold the artifact, which is the
    only statement of absence a storage backend makes uniformly. Deciding it from
    the exception instead would let a timeout read as an absence, and a dropped
    sensor changes the window every clip is planned from -- so two runs of one
    immutable session could plan different episode sets, and the second would
    then fail reconciliation against the first.

    A camera missing its capture timeline is still a fault unless its config sets
    ``allow_pts_fallback``, in which case its timeline is read from the
    container's own presentation timestamps instead -- returned in the third
    element, the keys whose timeline came from that fallback rather than from a
    capture sidecar. "Missing" is decided the same way a camera's own absence
    is: against the listing, not from whatever ``load_timeline_ns`` raises. A
    sidecar the listing does carry is read, and if that read fails -- a
    truncated write, a wrong dtype, a non-ascending timeline -- it is a fault,
    the same as a broken video would be, and never falls back: the fallback
    exists for a sidecar that was never written, not for one that was written
    badly.
    """
    sensor_uris: dict[str, str] = {}
    timelines: dict[str, npt.NDArray[np.int64]] = {}
    pts_fallback_keys: set[str] = set()
    failures: list[dict[str, Any]] = []
    for key, camera in config.cameras.items():
        # Resolving and reading are separate because only the first can say the
        # camera is not in this session. A camera that resolved and then failed
        # to read is a fault whatever ``required`` says: a recording with video
        # and no timestamps is a broken camera, not an absent one.
        try:
            uri = resolve_sensor_pattern(
                session_uri, camera.uri_pattern, storage_profile=storage_profile, session_paths=session_paths
            )
        except FileNotFoundError as error:
            if camera.required:
                failures.append(failure(session_id, key, None, "open_sensor", error))
            continue
        except Exception as error:  # noqa: BLE001 -- an ambiguous pattern is one failed sensor
            failures.append(failure(session_id, key, None, "open_sensor", error))
            continue
        # The listing, not an exception type, decides whether the sidecar is
        # missing -- the same reason ``resolve_sensor_pattern`` above is given
        # the listing rather than being let to raise and be sniffed. Relative to
        # session_uri because session_paths is: resolve_sensor_uri's own join,
        # reversed.
        relative_match = uri.removeprefix(f"{session_uri.rstrip('/')}/")
        if timeline_uri(relative_match) not in session_paths:
            if not camera.allow_pts_fallback:
                missing_timeline = FileNotFoundError(f"no capture timeline sidecar for {uri}")
                failures.append(failure(session_id, key, None, "open_sensor", missing_timeline))
                continue
            try:
                timelines[key] = load_container_pts_ns(
                    uri, storage_profile=storage_profile, read_buffer_bytes=config.execution.read_buffer_bytes
                )
            except Exception as pts_error:  # noqa: BLE001 -- one unreadable sensor is one failed sensor
                failures.append(failure(session_id, key, None, "open_sensor", pts_error))
                continue
            pts_fallback_keys.add(key)
        else:
            try:
                timelines[key] = load_timeline_ns(
                    uri, storage_profile=storage_profile, read_buffer_bytes=config.execution.read_buffer_bytes
                )
            except Exception as timeline_error:  # noqa: BLE001 -- a listed-but-broken sidecar is one failed sensor
                failures.append(failure(session_id, key, None, "open_sensor", timeline_error))
                continue
        sensor_uris[key] = uri

    if not sensor_uris and not failures:
        no_camera = ValueError("no configured camera could be opened")
        failures.append(failure(session_id, None, None, "open_sensor", no_camera))
    return sensor_uris, timelines, pts_fallback_keys, failures


def open_mcap_sensors(
    session_id: str,
    session_uri: str,
    session_paths: list[str],
    *,
    config: ResolvedMultimodalSplitConfig,
    storage_profile: str,
) -> tuple[dict[str, str], dict[str, tuple[int, int]], list[dict[str, Any]]]:
    """Resolve every MCAP sensor in *session_uri* and read only its bounds.

    Bounds, not the messages. An MCAP carries an index, so the first and last
    time on a topic cost two seeks whatever the recording's length -- and the
    messages themselves are the splitter's business, which reads them once in a
    forward pass rather than shipping a three-hour topic through a plan row.

    ``required`` decides a miss here exactly as it does for a camera, and on the
    same evidence: the artifact is absent from the session's listing.
    """
    sensor_uris: dict[str, str] = {}
    bounds: dict[str, tuple[int, int]] = {}
    failures: list[dict[str, Any]] = []
    for key, sensor in config.mcap_sensors.items():
        try:
            uri = resolve_sensor_pattern(
                session_uri, sensor.uri_pattern, storage_profile=storage_profile, session_paths=session_paths
            )
        except FileNotFoundError as error:
            if sensor.required:
                failures.append(failure(session_id, key, None, "open_sensor", error))
            continue
        except Exception as error:  # noqa: BLE001 -- an ambiguous pattern is one failed sensor
            failures.append(failure(session_id, key, None, "open_sensor", error))
            continue
        try:
            bounds[key] = load_mcap_bounds(
                uri,
                sensor.topic,
                storage_profile=storage_profile,
                read_buffer_bytes=config.execution.read_buffer_bytes,
            )
        except Exception as error:  # noqa: BLE001 -- one unreadable sensor is one failed sensor
            failures.append(failure(session_id, key, None, "open_sensor", error))
            continue
        sensor_uris[key] = uri
    return sensor_uris, bounds, failures


def span_planner(
    item: dict[str, Any],
    *,
    config: ResolvedMultimodalSplitConfig,
    storage_profile: str = "default",
    reconciliation: ReconciliationMap | None = None,
) -> list[dict[str, Any]]:
    """Plan one session into spans, and emit one row per camera that will serve them.

    Emits nothing for a session no camera set can cover, so splitting is never
    handed work that was already known to be impossible. The plan report is
    written either way: a session that produced no rows because it was refused
    and one that produced none because a worker died look identical otherwise.
    """
    session_id = str(item["source_session_id"])
    session_uri = str(item["session_uri"])

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

    sensor_uris, timelines, pts_fallback_keys, failures = open_cameras(
        session_id, session_uri, session_paths, config=config, storage_profile=storage_profile
    )
    mcap_uris, mcap_bounds, mcap_failures = open_mcap_sensors(
        session_id, session_uri, session_paths, config=config, storage_profile=storage_profile
    )
    failures.extend(mcap_failures)

    # All or nothing across the sensors declared required: a clip is one moment
    # across everything the rig promised, so a session missing one produces no
    # clips rather than clips missing a modality.
    if failures:
        write_plan_errors(session_id, failures, output_path=config.output.path, storage_profile=storage_profile)
        return []

    try:
        spans = plan_common_spans(timelines, config, bounds=mcap_bounds)
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

    # The cameras that resolved, not the cameras configured. An optional one that
    # did not open is absent from the clip, and naming it here would promise a
    # view no manifest or episode can deliver.
    camera_keys = sorted(sensor_uris)
    mcap_keys = sorted(mcap_uris)
    camera_sources = {key: sensor_uris[key] for key in camera_keys}
    sensor_sources = {key: mcap_uris[key] for key in mcap_keys}
    split_contract_digest = make_split_contract_digest(config)
    # A clip is one moment across everything the rig promised, so one camera
    # reading its timeline from container PTS instead of a capture sidecar marks
    # the whole episode CONTAINER_PTS -- there is no per-camera timestamp_source
    # on the row for a mixed episode to carry.
    timestamp_source = (
        TimestampSource.CONTAINER_PTS if pts_fallback_keys & set(camera_keys) else TimestampSource.CAPTURE
    )
    planned: list[dict[str, Any]] = []
    already_canonical: list[dict[str, Any]] = []
    existing_by_episode_id = (reconciliation or {}).get(source_id, {})
    for span in servable:
        entry: dict[str, Any] = {
            "clip_id": make_clip_id(source_id, span),
            "episode_id": make_episode_id(source_id, span),
            "source_id": source_id,
            "source_session_id": session_id,
            "source_session_location": session_uri,
            "start_ns": span.start_ns,
            "exclusive_end_ns": span.end_ns,
            "sampling_fps": config.clip.sampling_fps,
            # The delivered half-open grid is authoritative. Publication
            # regenerates it so a carried row count cannot diverge silently.
            "expected_row_count": len(sampling_grid_ns(span, config)),
            "timestamp_source": timestamp_source.value,
            "record_schema_version": RECORD_SCHEMA_VERSION,
            "media_contract_version": MEDIA_CONTRACT_VERSION,
            "split_contract_digest": split_contract_digest,
            "planned_camera_keys": camera_keys,
            # The non-camera sensors this clip is expected to carry, so assembly
            # can tell a trajectory that was never asked for from one that was
            # asked for and never written.
            "planned_sensor_keys": mcap_keys,
        }
        if reconcile_planned_episode(
            entry,
            existing_by_episode_id=existing_by_episode_id,
            camera_sources=camera_sources,
            sensor_sources=sensor_sources,
        ):
            already_canonical.append(
                {
                    "result_kind": ALREADY_CANONICAL,
                    "source_session_id": session_id,
                    "sensor_key": None,
                    "sensor_uri": None,
                    "sensor_type": None,
                    "timeline_ns": None,
                    "spans": None,
                    "episode_id": entry["episode_id"],
                }
            )
        else:
            planned.append(entry)
    # Written here rather than after the media, because no single camera task
    # knows which other views a clip holds. Coverage is all-or-nothing, so the
    # planned set is the set -- unless an encode fails, which its camera reports.
    for entry in planned:
        span = Span(start_ns=int(entry["start_ns"]), end_ns=int(entry["exclusive_end_ns"]))
        write_clip_manifest(
            session_id,
            str(entry["clip_id"]),
            span,
            camera_keys,
            mcap_keys,
            output_path=config.output.path,
            sampling_fps=config.clip.sampling_fps,
            storage_profile=storage_profile,
        )

    camera_plans = [
        {
            "result_kind": CAMERA_PLAN,
            "source_session_id": session_id,
            "sensor_key": key,
            "sensor_uri": sensor_uris[key],
            "sensor_type": SENSOR_TYPE_CAMERA,
            "timeline_ns": timelines[key].tolist(),
            "spans": planned,
            "episode_id": None,
        }
        for key in camera_keys
        if planned
    ]
    # No timeline: the splitter reads the topic itself, in one forward pass over
    # the same spans. Copying the messages here to carry them would move the
    # whole recording through a plan row to save nothing.
    mcap_plans = [
        {
            "result_kind": MCAP_PLAN,
            "source_session_id": session_id,
            "sensor_key": key,
            "sensor_uri": uri,
            "sensor_type": config.sensors[key].sensor_type,
            "timeline_ns": None,
            "spans": planned,
            "episode_id": None,
        }
        for key, uri in sorted(mcap_uris.items())
        if planned
    ]
    return [*camera_plans, *mcap_plans, *already_canonical]


def span_planner_batch(
    sessions: pa.Table,
    *,
    config: ResolvedMultimodalSplitConfig,
    storage_profile: str = "default",
    reconciliation: ReconciliationMap | None = None,
) -> pa.Table:
    """Plan one Arrow block while preserving the fixed transient result schema."""
    rows = [
        result
        for session in sessions.to_pylist()
        for result in span_planner(
            session,
            config=config,
            storage_profile=storage_profile,
            reconciliation=reconciliation,
        )
    ]
    return pa.Table.from_pylist(rows, schema=PLANNING_RESULT_SCHEMA)


def plan_common_spans(
    timelines: dict[str, npt.NDArray[np.int64]],
    config: ResolvedMultimodalSplitConfig,
    *,
    bounds: dict[str, tuple[int, int]] | None = None,
) -> tuple[Span, ...]:
    """Tile the window every sensor was recording into clip spans.

    The latest start and the earliest end, not one sensor's own bounds. Cameras
    on a rig begin within a few frames of each other, so planning from one of
    them puts the first references before another view existed.

    An MCAP sensor narrows the same window from its bounds rather than a
    timeline, and for the same reason: a clip running past the last sample is a
    clip whose sensor stops partway through. It contributes only the window,
    never coverage. A sensor placed on the grid does have a tolerance, but it
    applies per row while the clip is cut, not here -- this decides which spans
    exist, and that one decides which of them a sensor can actually serve.
    """
    starts = [int(timeline[0]) for timeline in timelines.values()]
    ends = [timeline_end_ns(timeline) for timeline in timelines.values()]
    for first_ns, last_ns in (bounds or {}).values():
        starts.append(first_ns)
        # Exclusive, like a timeline's end: the last message is an instant, and a
        # span ending on it would be asking for one that has not arrived.
        ends.append(last_ns + 1)
    return plan_clip_spans(
        start_ns=max(starts),
        end_ns=min(ends),
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
        missing = first_uncovered(references, timelines, config.video.max_delta_ns)
        if missing is None:
            servable.append(span)
            continue
        key, worst_ns = missing
        error = AlignmentError(
            AlignmentFailureReason.TOLERANCE_EXCEEDED,
            f"camera {key!r} cannot serve this moment: its nearest observation is {worst_ns} ns from a "
            f"reference, further than the {config.video.max_delta_ns} ns the rig was given",
            sensor_id=key,
            align_timestamps_ns=references,
        )
        refused.append(failure(session_id, key, span, "coverage", error, make_clip_id(source_id, span)))
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
