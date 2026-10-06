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
from unittest.mock import Mock

import lance
import pyarrow as pa
import pytest

from cosmos_curator.core.sensors.sampling.grid import make_ts_grid
from cosmos_curator.core.sensors.types.types import TimestampSource
from cosmos_curator.next.recipes.multimodal_embedding.episode_input import (
    CAMERA_WORK_ID,
    EPISODE_MAPPED_INPUT_SCHEMA,
    EPISODE_SOURCE_SCHEMA,
    camera_work_id,
    project_camera_work,
    project_episode_camera,
    validate_episode_cameras,
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


def test_all_cameras_and_work_identity_preserve_independent_pairs(tmp_path: Path) -> None:
    """Camera expansion retains each media binding and distinct checkpoint IDs for the same episode."""
    row = _episode()
    front = row["camera_media"][0]
    row["camera_media"] = [front, {**front, "camera_key": "rear", "media_sha256": "d" * 64}]
    uri, dataset = _dataset(tmp_path, [row])
    keys = tuple(validate_episode_cameras(dataset, uri=uri, selection="all"))
    assert keys == ("front_wide", "rear")
    source = dataset.scanner(columns=EPISODE_SOURCE_SCHEMA.names, with_row_address=True).to_table()
    expanded = project_camera_work(source, camera_keys=keys).to_pylist()
    assert [item["media_sha256"] for item in expanded] == ["a" * 64, "d" * 64]
    assert len({item[CAMERA_WORK_ID] for item in expanded}) == 2
    assert camera_work_id("a_b", "c") != camera_work_id("a", "b_c")


def test_all_on_empty_table_selects_no_cameras(tmp_path: Path) -> None:
    """An empty canonical table has no work or camera namespaces to register."""
    uri, dataset = _dataset(tmp_path, [])
    assert validate_episode_cameras(dataset, uri=uri, selection="all") == {}


@pytest.mark.parametrize("camera_count", [1, 8, 16])
@pytest.mark.parametrize("selection_mode", ["all", "list"])
def test_camera_preflight_scans_once_and_returns_first_s3_locations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, camera_count: int, selection_mode: str
) -> None:
    """Adding cameras does not multiply source scans or global identity aggregations."""
    keys = [f"camera_{index:02}" for index in range(camera_count)]
    rows = []
    for episode_index in range(2):
        row = _episode()
        row["episode_id"] = f"episode-{episode_index}"
        descriptor = row["camera_media"][0]
        row["camera_media"] = [
            {
                **descriptor,
                "camera_key": key,
                "media_location": (
                    f"s3://media/episode-{episode_index}/{key}.mp4"
                    if episode_index == 1 or key == keys[0]
                    else f"/local/{key}.mp4"
                ),
            }
            for key in keys
        ]
        rows.append(row)
    uri, dataset = _dataset(tmp_path, rows)
    scanner, sql = Mock(wraps=dataset.scanner), Mock(wraps=dataset.sql)
    monkeypatch.setattr(dataset, "scanner", scanner)
    monkeypatch.setattr(dataset, "sql", sql)

    locations = validate_episode_cameras(dataset, uri=uri, selection="all" if selection_mode == "all" else keys[::-1])

    assert tuple(locations) == tuple(keys)
    assert locations == {key: f"s3://media/episode-{0 if key == keys[0] else 1}/{key}.mp4" for key in keys}
    scanner.assert_called_once_with(columns=list(EPISODE_SOURCE_SCHEMA.names))
    assert sql.call_count == 1


def test_camera_list_preflight_ignores_unselected_duplicate_cameras(tmp_path: Path) -> None:
    """Only the cameras in the selected list must resolve exactly once."""
    row = _episode()
    front = row["camera_media"][0]
    row["camera_media"] = [front, {**front, "camera_key": "rear"}, {**front, "camera_key": "rear"}]
    uri, dataset = _dataset(tmp_path, [row])
    assert validate_episode_cameras(dataset, uri=uri, selection=["front_wide"]) == {"front_wide": None}
    with pytest.raises(ValueError, match="2 camera descriptors for 'rear'"):
        validate_episode_cameras(dataset, uri=uri, selection="all")


def test_all_preflight_rejects_a_camera_missing_from_a_later_episode(tmp_path: Path) -> None:
    """Discovering cameras in the first row still checks their presence in every later row."""
    first, second = _episode(), _episode()
    second["episode_id"] = "episode-2"
    first["camera_media"].append({**first["camera_media"][0], "camera_key": "rear"})
    uri, dataset = _dataset(tmp_path, [first, second])
    with pytest.raises(ValueError, match="Episode 'episode-2' has 0 camera descriptors for 'rear'"):
        validate_episode_cameras(dataset, uri=uri, selection="all")


def test_all_rejects_heterogeneous_camera_sets_before_registration(tmp_path: Path) -> None:
    """All means every discovered camera; it never silently selects the intersection."""
    first = _episode()
    second = _episode()
    second["episode_id"] = "episode-2"
    front = second["camera_media"][0]
    second["camera_media"] = [front, {**front, "camera_key": "rear"}]
    uri, dataset = _dataset(tmp_path, [first, second])
    with pytest.raises(ValueError, match="rear"):
        validate_episode_cameras(dataset, uri=uri, selection="all")


def test_preflight_rejects_missing_and_duplicate_camera(tmp_path: Path) -> None:
    """Camera resolution must be unambiguous before result fields are registered."""
    episode = _episode()
    uri, dataset = _dataset(tmp_path, [episode])
    assert validate_episode_cameras(dataset, uri=uri, selection=["front_wide"]) == {"front_wide": None}
    with pytest.raises(ValueError, match="0 camera descriptors"):
        validate_episode_cameras(dataset, uri=uri, selection=["missing"])

    duplicate = deepcopy(episode)
    duplicate["camera_media"].append(deepcopy(duplicate["camera_media"][0]))
    uri, dataset = _dataset(tmp_path / "duplicate", [duplicate])
    with pytest.raises(ValueError, match="2 camera descriptors"):
        validate_episode_cameras(dataset, uri=uri, selection=["front_wide"])


def test_preflight_rejects_duplicate_identity_and_contract_version(tmp_path: Path) -> None:
    """Shape-compatible records with duplicate IDs or stale Split contracts fail preflight."""
    episode = _episode()
    uri, dataset = _dataset(tmp_path / "duplicate-id", [episode, deepcopy(episode)])
    with pytest.raises(ValueError, match="not globally unique"):
        validate_episode_cameras(dataset, uri=uri, selection=["front_wide"])

    invalid_version = deepcopy(episode)
    invalid_version["media_contract_version"] = 999
    uri, dataset = _dataset(tmp_path / "invalid-version", [invalid_version])
    with pytest.raises(ValueError, match="media_contract_version"):
        validate_episode_cameras(dataset, uri=uri, selection=["front_wide"])
