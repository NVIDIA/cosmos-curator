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

"""Canonical camera-only episode records for ``multimodal-split``."""

from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from cosmos_curator.core.sensors.sampling.grid import make_ts_grid
from cosmos_curator.core.sensors.types.types import TimestampSource
from cosmos_curator.next.recipes.multimodal_split.identities import (
    MEDIA_CONTRACT_VERSION,
    make_materialization_digest,
)

RECORD_SCHEMA_VERSION = 1

CAMERA_MEDIA_STRUCT = pa.struct(
    [
        pa.field("camera_key", pa.string(), nullable=False),
        pa.field("source_media_location", pa.large_string(), nullable=False),
        pa.field("media_location", pa.large_string(), nullable=False),
        pa.field("media_size_bytes", pa.int64(), nullable=False),
        pa.field("media_sha256", pa.string(), nullable=False),
        pa.field("timeline_location", pa.large_string(), nullable=False),
        pa.field("timeline_sha256", pa.string(), nullable=False),
        pa.field("observations_location", pa.large_string(), nullable=False),
        pa.field("observations_sha256", pa.string(), nullable=False),
    ]
)

# A sensor whose clip is one file and nothing else. A camera's clip carries two
# sidecars because it was resampled onto the episode's grid and the record of
# which frame served which row is otherwise unrecoverable; a trajectory is copied
# at its own rate, so there is no grid to record and no selection to account for.
# Kept separate from ``CAMERA_MEDIA_STRUCT`` rather than widening it with fields
# that would be null for every row: a reader asking "what is in this clip" gets
# two lists whose shapes mean what they say.
SENSOR_MEDIA_STRUCT = pa.struct(
    [
        pa.field("sensor_key", pa.string(), nullable=False),
        pa.field("sensor_type", pa.string(), nullable=False),
        pa.field("source_media_location", pa.large_string(), nullable=False),
        pa.field("media_location", pa.large_string(), nullable=False),
        pa.field("media_size_bytes", pa.int64(), nullable=False),
        pa.field("media_sha256", pa.string(), nullable=False),
    ]
)

EPISODE_SCHEMA = pa.schema(
    [
        pa.field("record_schema_version", pa.int32(), nullable=False),
        pa.field("media_contract_version", pa.int32(), nullable=False),
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("split_clip_id", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_session_id", pa.large_string(), nullable=False),
        pa.field("source_session_location", pa.large_string(), nullable=False),
        pa.field("start_ns", pa.int64(), nullable=False),
        pa.field("exclusive_end_ns", pa.int64(), nullable=False),
        pa.field("sampling_fps", pa.int32(), nullable=False),
        pa.field("row_count", pa.int64(), nullable=False),
        pa.field("timestamp_source", pa.string(), nullable=False),
        pa.field("split_contract_digest", pa.string(), nullable=False),
        pa.field("materialization_digest", pa.string(), nullable=False),
        pa.field(
            "camera_media",
            pa.large_list(pa.field("item", CAMERA_MEDIA_STRUCT, nullable=False)),
            nullable=False,
        ),
        # Empty when the config asked for no such sensor, which is a different
        # statement from a camera list being empty -- an episode with no views is
        # not an episode, while an episode with no trajectory is an ordinary one.
        pa.field(
            "sensor_media",
            pa.large_list(pa.field("item", SENSOR_MEDIA_STRUCT, nullable=False)),
            nullable=False,
        ),
    ]
)


@dataclass(frozen=True)
class EpisodeFacts:
    """Shared facts carried by every camera outcome for one episode."""

    episode_id: str
    split_clip_id: str
    source_id: str
    source_session_id: str
    source_session_location: str
    start_ns: int
    exclusive_end_ns: int
    sampling_fps: int
    row_count: int
    timestamp_source: TimestampSource
    split_contract_digest: str
    record_schema_version: int = RECORD_SCHEMA_VERSION
    media_contract_version: int = MEDIA_CONTRACT_VERSION


def build_episode_record(
    facts: EpisodeFacts,
    camera_media: list[dict[str, Any]],
    sensor_media: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one canonical row, enforcing the capture-grid and camera contracts."""
    if facts.timestamp_source is not TimestampSource.CAPTURE:
        msg = f"Canonical multimodal episodes require capture timestamps, got {facts.timestamp_source.value!r}"
        raise ValueError(msg)
    _, _, timestamps_ns = make_ts_grid(
        facts.start_ns,
        exclusive_end_ns=facts.exclusive_end_ns,
        sample_rate_hz=facts.sampling_fps,
    )
    expected_rows = len(timestamps_ns)
    if expected_rows != facts.row_count:
        msg = (
            f"Episode {facts.episode_id} has row_count={facts.row_count}, but its span and sampling rate require "
            f"{expected_rows} grid rows"
        )
        raise ValueError(msg)
    ordered = sorted(camera_media, key=lambda camera: str(camera["camera_key"]))
    camera_keys = [str(camera["camera_key"]) for camera in ordered]
    if not camera_keys or len(set(camera_keys)) != len(camera_keys):
        msg = "Canonical episode camera keys must be nonempty and unique"
        raise ValueError(msg)
    ordered_sensors = sorted(sensor_media or [], key=lambda sensor: str(sensor["sensor_key"]))
    sensor_keys = [str(sensor["sensor_key"]) for sensor in ordered_sensors]
    if len(set(sensor_keys)) != len(sensor_keys):
        msg = "Canonical episode sensor keys must be unique"
        raise ValueError(msg)
    materialization_digest = make_materialization_digest(
        episode_id=facts.episode_id,
        split_clip_id=facts.split_clip_id,
        split_contract_digest=facts.split_contract_digest,
        sampling_fps=facts.sampling_fps,
        row_count=facts.row_count,
        camera_media=ordered,
        sensor_media=ordered_sensors,
    )
    return {
        "record_schema_version": facts.record_schema_version,
        "media_contract_version": facts.media_contract_version,
        "episode_id": facts.episode_id,
        "split_clip_id": facts.split_clip_id,
        "source_id": facts.source_id,
        "source_session_id": facts.source_session_id,
        "source_session_location": facts.source_session_location,
        "start_ns": facts.start_ns,
        "exclusive_end_ns": facts.exclusive_end_ns,
        "sampling_fps": facts.sampling_fps,
        "row_count": facts.row_count,
        "timestamp_source": facts.timestamp_source.value,
        "split_contract_digest": facts.split_contract_digest,
        "materialization_digest": materialization_digest,
        "camera_media": ordered,
        "sensor_media": ordered_sensors,
    }


def episode_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Cast canonical episode mappings to the exact publisher-owned schema."""
    return pa.Table.from_pylist(rows, schema=EPISODE_SCHEMA)


def resolve_camera_media(record: dict[str, Any], camera_key: str) -> dict[str, Any]:
    """Return exactly one camera descriptor by key."""
    matches = [camera for camera in record["camera_media"] if str(camera["camera_key"]) == camera_key]
    if len(matches) != 1:
        msg = f"Episode {record.get('episode_id')!r} has {len(matches)} camera descriptors for {camera_key!r}"
        raise ValueError(msg)
    return dict(matches[0])
