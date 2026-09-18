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

"""Episode assembly and bounded canonical Lance publication."""

from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from cosmos_curator.core.sensors.types.types import TimestampSource
from cosmos_curator.next.recipes.multimodal_split.config import SENSOR_TYPE_CAMERA
from cosmos_curator.next.recipes.multimodal_split.lance_sink import write_episode_fragment
from cosmos_curator.next.recipes.multimodal_split.records import EPISODE_SCHEMA, EpisodeFacts, build_episode_record

_COMPLETE = "complete"
_DROPPED = "dropped"

ASSEMBLY_RESULT_SCHEMA = pa.schema(
    [
        pa.field("assembly_status", pa.string(), nullable=False),
        pa.field("camera_clips_written", pa.int64(), nullable=False),
        pa.field("camera_clip_bytes", pa.int64(), nullable=False),
        # Counted apart from the cameras rather than folded in. A trajectory is
        # not a clip of its own -- it rides along with one -- so adding it to the
        # camera total would report more clips than the run produced, and
        # dividing the bytes would hide that video is nearly all of them.
        pa.field("sensor_clips_written", pa.int64(), nullable=False),
        pa.field("sensor_clip_bytes", pa.int64(), nullable=False),
        *[pa.field(field.name, field.type, nullable=True, metadata=field.metadata) for field in EPISODE_SCHEMA],
    ]
)

PUBLICATION_RESULT_SCHEMA = pa.schema(
    [
        pa.field("candidate_json", pa.large_string(), nullable=False),
        pa.field("episode_count", pa.int64(), nullable=False),
    ]
)


@dataclass(frozen=True)
class PublicationSummary:
    """Canonical publication results for one invocation."""

    episodes_newly_published: int
    episodes_dropped: int
    camera_clips_written: int
    clip_bytes: int
    sensor_clips_written: int = 0
    sensor_clip_bytes: int = 0


def assemble_episode_group(outcomes: pa.Table) -> pa.Table:  # noqa: C901 -- one exact camera-set state machine
    """Validate one split-clip group and emit one complete or dropped control row."""
    rows = outcomes.to_pylist()
    if not rows:
        msg = "Episode assembly received an empty outcome group"
        raise ValueError(msg)
    first = rows[0]
    shared = _shared_fact_tuple(first)
    for row in rows[1:]:
        if _shared_fact_tuple(row) != shared:
            msg = f"Camera outcomes for split clip {first['split_clip_id']} carry conflicting episode facts"
            raise ValueError(msg)

    # Split by what the outcome is, not by which stage produced it. Both stages
    # write the same terminal schema, and the rows differ in which artifacts they
    # carry: a camera clip has sidecars, a copied sensor clip is one file.
    camera_rows = [row for row in rows if _is_camera(row)]
    sensor_rows = [row for row in rows if not _is_camera(row)]

    planned = _canonical_key_set(first, "planned_camera_keys", first["episode_id"], "camera")
    if not planned:
        msg = f"Episode {first['episode_id']} has a non-canonical planned camera set"
        raise ValueError(msg)
    planned_sensors = _canonical_key_set(first, "planned_sensor_keys", first["episode_id"], "sensor")

    # A planned sensor absent from the group drops the episode exactly as a
    # planned camera does. It was planned because the session carried it, so its
    # absence here is a clip that was not written, not one never asked for.
    if _group_is_incomplete(first, "camera", planned, camera_rows) or _group_is_incomplete(
        first, "sensor", planned_sensors, sensor_rows
    ):
        return _assembly_table(_dropped_row(first, rows))

    camera_media: list[dict[str, Any]] = []
    sensor_media: list[dict[str, Any]] = []
    for row in rows:
        status = str(row["status"])
        if status == "failure":
            return _assembly_table(_dropped_row(first, rows))
        if status != "success":
            msg = f"Episode {first['episode_id']} has unexpected outcome status {status!r}"
            raise ValueError(msg)
        is_camera = _is_camera(row)
        descriptor_fields = (
            "source_media_location",
            "media_location",
            "media_size_bytes",
            "media_sha256",
        ) + (
            (
                "timeline_location",
                "timeline_sha256",
                "sensor_timestamps_location",
                "sensor_timestamps_sha256",
            )
            if is_camera
            else ()
        )
        if any(row[field] is None for field in descriptor_fields):
            msg = f"Successful outcome {row['sensor_key']!r} is missing artifact descriptors"
            raise ValueError(msg)
        descriptors = {field: row[field] for field in descriptor_fields}
        if is_camera:
            camera_media.append({"camera_key": str(row["sensor_key"]), **descriptors})
        else:
            sensor_media.append(
                {"sensor_key": str(row["sensor_key"]), "sensor_type": str(row["sensor_type"]), **descriptors}
            )

    facts = EpisodeFacts(
        record_schema_version=int(first["record_schema_version"]),
        media_contract_version=int(first["media_contract_version"]),
        episode_id=str(first["episode_id"]),
        split_clip_id=str(first["split_clip_id"]),
        source_id=str(first["source_id"]),
        source_session_id=str(first["source_session_id"]),
        source_session_location=str(first["source_session_location"]),
        start_ns=int(first["start_ns"]),
        exclusive_end_ns=int(first["exclusive_end_ns"]),
        sampling_fps=int(first["sampling_fps"]),
        row_count=int(first["expected_row_count"]),
        timestamp_source=TimestampSource(str(first["timestamp_source"])),
        split_contract_digest=str(first["split_contract_digest"]),
    )
    episode = build_episode_record(facts, camera_media, sensor_media)
    return _assembly_table(
        {
            "assembly_status": _COMPLETE,
            "camera_clips_written": len(camera_rows),
            "camera_clip_bytes": sum(int(row["media_size_bytes"]) for row in camera_rows),
            "sensor_clips_written": len(sensor_rows),
            "sensor_clip_bytes": sum(int(row["media_size_bytes"]) for row in sensor_rows),
            **episode,
        }
    )


def _group_is_incomplete(
    first: dict[str, Any],
    label: str,
    planned: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> bool:
    """Say whether a planned set is short, refusing a set that is not the planned one.

    Short is an ordinary outcome -- a clip some sensor could not write -- and
    drops the episode. A key nobody planned, or one delivered twice, is not an
    outcome but a broken stream, and raises.
    """
    received = [str(row["sensor_key"]) for row in rows]
    if len(set(received)) != len(received):
        msg = f"Episode {first['episode_id']} has duplicate {label} outcomes"
        raise ValueError(msg)
    unexpected = sorted(set(received) - set(planned))
    if unexpected:
        msg = f"Episode {first['episode_id']} has unexpected {label} outcomes: {', '.join(unexpected)}"
        raise ValueError(msg)
    return bool(set(planned) - set(received))


def _canonical_key_set(row: dict[str, Any], field: str, episode_id: object, label: str) -> tuple[str, ...]:
    """Return a planned key list, insisting it is sorted and free of duplicates.

    The planner writes it sorted. Reading it back unchecked would let a
    hand-edited or replayed row make "every planned view is present" true by
    naming one twice.
    """
    keys = tuple(str(key) for key in row[field])
    if tuple(sorted(set(keys))) != keys:
        msg = f"Episode {episode_id} has a non-canonical planned {label} set"
        raise ValueError(msg)
    return keys


def _shared_fact_tuple(row: dict[str, Any]) -> tuple[object, ...]:
    """Return every episode fact that all camera outcomes must share."""
    return (
        int(row["record_schema_version"]),
        int(row["media_contract_version"]),
        str(row["episode_id"]),
        str(row["split_clip_id"]),
        str(row["source_id"]),
        str(row["source_session_id"]),
        str(row["source_session_location"]),
        int(row["start_ns"]),
        int(row["exclusive_end_ns"]),
        int(row["sampling_fps"]),
        int(row["expected_row_count"]),
        str(row["timestamp_source"]),
        str(row["split_contract_digest"]),
        tuple(str(key) for key in row["planned_camera_keys"]),
        tuple(str(key) for key in row["planned_sensor_keys"]),
    )


def _dropped_row(first: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Retain accounting for an incomplete episode without constructing a base row."""
    row: dict[str, Any] = {field.name: None for field in EPISODE_SCHEMA}
    row["episode_id"] = str(first["episode_id"])
    return {
        "assembly_status": _DROPPED,
        "camera_clips_written": _written(rows, camera=True),
        "camera_clip_bytes": _written_bytes(rows, camera=True),
        "sensor_clips_written": _written(rows, camera=False),
        "sensor_clip_bytes": _written_bytes(rows, camera=False),
        **row,
    }


def _is_camera(outcome: dict[str, Any]) -> bool:
    return str(outcome["sensor_type"]) == SENSOR_TYPE_CAMERA


def _written(rows: list[dict[str, Any]], *, camera: bool) -> int:
    """Count the artifacts a dropped episode did write, so its cost is still accounted."""
    return sum(str(outcome["status"]) == "success" for outcome in rows if _is_camera(outcome) is camera)


def _written_bytes(rows: list[dict[str, Any]], *, camera: bool) -> int:
    return sum(
        int(outcome["media_size_bytes"])
        for outcome in rows
        if _is_camera(outcome) is camera and str(outcome["status"]) == "success"
    )


def _assembly_table(row: dict[str, Any]) -> pa.Table:
    return pa.Table.from_pylist([row], schema=ASSEMBLY_RESULT_SCHEMA)


def stage_episode_batch(batch: pa.Table, *, uri: str, storage_profile: str) -> pa.Table:
    """Stage one bounded block of complete episode rows on a Ray worker."""
    statuses = {str(value) for value in batch["assembly_status"].to_pylist()}
    if statuses != {_COMPLETE}:
        msg = f"Episode publication received assembly states {sorted(statuses)}"
        raise ValueError(msg)
    episodes = batch.select(EPISODE_SCHEMA.names).cast(EPISODE_SCHEMA)
    candidate = write_episode_fragment(episodes, uri=uri, storage_profile=storage_profile)
    if candidate is None:
        msg = "Episode publication received an empty complete block"
        raise ValueError(msg)
    return pa.Table.from_pylist(
        [{"candidate_json": candidate, "episode_count": episodes.num_rows}],
        schema=PUBLICATION_RESULT_SCHEMA,
    )
