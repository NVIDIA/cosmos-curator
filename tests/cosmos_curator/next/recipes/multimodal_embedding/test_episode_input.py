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

"""Tests for selected-camera episode projection used by video embedding."""

from copy import deepcopy
from pathlib import Path

import lance
import pyarrow as pa
import pytest

from cosmos_curator.core.sensors.sampling.grid import make_ts_grid
from cosmos_curator.core.sensors.types.types import TimestampSource
from cosmos_curator.next.recipes.multimodal_embedding.episode_input import (
    EPISODE_MAPPED_INPUT_SCHEMA,
    EPISODE_SOURCE_SCHEMA,
    project_episode_camera,
    validate_episode_input,
)
from cosmos_curator.next.recipes.multimodal_split.records import EPISODE_SCHEMA, EpisodeFacts, build_episode_record


def _episode() -> dict[str, object]:
    start_ns = 10
    end_ns = 1_000_000_010
    _sample_start, _sample_end, timestamps_ns = make_ts_grid(
        start_ns,
        exclusive_end_ns=end_ns,
        sample_rate_hz=30,
    )
    facts = EpisodeFacts(
        episode_id="episode-1",
        split_clip_id="split-1",
        source_id="source-1",
        source_session_id="session-1",
        source_session_location="/input/session-1",
        start_ns=start_ns,
        exclusive_end_ns=end_ns,
        sampling_fps=30,
        row_count=len(timestamps_ns),
        timestamp_source=TimestampSource.SENSOR,
        split_contract_digest="split-contract",
    )
    return build_episode_record(
        facts,
        [
            {
                "camera_key": "front_wide",
                "source_media_location": "/source/front_wide.mp4",
                "media_location": "/output/front_wide.mp4",
                "media_size_bytes": 11,
                "media_sha256": "a" * 64,
                "timeline_location": "/output/front_wide.mp4.timestamps.npy",
                "timeline_sha256": "b" * 64,
                "sensor_timestamps_location": "/output/front_wide.mp4.sensor_timestamps.npy",
                "sensor_timestamps_sha256": "c" * 64,
            }
        ],
        [],
    )


def test_projection_binds_physical_row_identity_camera_media_and_episode_bounds() -> None:
    """The inference phase receives only its selected camera and enough binding for safe publication."""
    source = pa.Table.from_pylist([_episode()], schema=EPISODE_SOURCE_SCHEMA).append_column(
        "_rowaddr", pa.array([(7 << 32) | 3], type=pa.uint64())
    )

    projected = project_episode_camera(source, camera_key="front_wide")
    row = projected.to_pylist()[0]

    assert projected.schema == EPISODE_MAPPED_INPUT_SCHEMA
    assert (row["fragment_id"], row["row_offset"], row["episode_id"]) == (7, 3, "episode-1")
    assert row["camera_key"] == "front_wide"
    assert row["media_location"] == "/output/front_wide.mp4"
    assert (row["start_ns"], row["exclusive_end_ns"]) == (10, 1_000_000_010)


def test_projection_rejects_non_positive_episode_duration() -> None:
    """A malformed source cannot create a vector with an invalid duration binding."""
    source = _episode()
    source["exclusive_end_ns"] = source["start_ns"]
    table = pa.Table.from_pylist([source], schema=EPISODE_SOURCE_SCHEMA).append_column(
        "_rowaddr", pa.array([0], type=pa.uint64())
    )

    with pytest.raises(ValueError, match="positive start_ns/exclusive_end_ns duration"):
        project_episode_camera(table, camera_key="front_wide")


def _dataset(tmp_path: Path, rows: list[dict[str, object]]) -> tuple[str, lance.LanceDataset]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    uri = str(tmp_path / "episodes.lance")
    table = pa.Table.from_pylist(rows, schema=EPISODE_SCHEMA)
    return uri, lance.write_dataset(table, uri, data_storage_version="2.2")


def test_preflight_rejects_missing_and_duplicate_camera(tmp_path: Path) -> None:
    """Camera resolution must be unambiguous before result fields are registered."""
    episode = _episode()
    uri, dataset = _dataset(tmp_path, [episode])
    assert validate_episode_input(dataset, uri=uri, camera_key="front_wide") is None
    with pytest.raises(ValueError, match="0 camera descriptors"):
        validate_episode_input(dataset, uri=uri, camera_key="missing")

    duplicate = deepcopy(episode)
    duplicate["camera_media"].append(deepcopy(duplicate["camera_media"][0]))
    uri, dataset = _dataset(tmp_path / "duplicate", [duplicate])
    with pytest.raises(ValueError, match="2 camera descriptors"):
        validate_episode_input(dataset, uri=uri, camera_key="front_wide")


def test_preflight_rejects_duplicate_identity_and_contract_version(tmp_path: Path) -> None:
    """Shape-compatible records with duplicate IDs or stale Split contracts fail preflight."""
    episode = _episode()
    uri, dataset = _dataset(tmp_path / "duplicate-id", [episode, deepcopy(episode)])
    with pytest.raises(ValueError, match="not globally unique"):
        validate_episode_input(dataset, uri=uri, camera_key="front_wide")

    invalid_version = deepcopy(episode)
    invalid_version["media_contract_version"] = 999
    uri, dataset = _dataset(tmp_path / "invalid-version", [invalid_version])
    with pytest.raises(ValueError, match="media_contract_version"):
        validate_episode_input(dataset, uri=uri, camera_key="front_wide")
