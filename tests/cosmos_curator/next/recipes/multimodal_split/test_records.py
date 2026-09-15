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

"""Tests for the canonical camera-only episode record contract."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pytest

from cosmos_curator.core.sensors.types.types import TimestampSource
from cosmos_curator.next.media.spans import Span
from cosmos_curator.next.recipes.multimodal_split.config import ResolvedMultimodalSplitConfig
from cosmos_curator.next.recipes.multimodal_split.grids import clip_duration_ns, sampling_grid_ns
from cosmos_curator.next.recipes.multimodal_split.records import (
    CAMERA_MEDIA_STRUCT,
    EPISODE_SCHEMA,
    SENSOR_MEDIA_STRUCT,
    EpisodeFacts,
    build_episode_record,
    episode_table,
    resolve_camera_media,
)


def _camera(key: str, marker: str) -> dict[str, object]:
    return {
        "camera_key": key,
        "source_media_location": f"/source/{key}.mp4",
        "media_location": f"/output/{key}.mp4",
        "media_size_bytes": 100,
        "media_sha256": marker * 64,
        "timeline_location": f"/output/{key}.mp4.timestamps.npy",
        "timeline_sha256": "a" * 64,
        "observations_location": f"/output/{key}.mp4.observations.npy",
        "observations_sha256": "b" * 64,
    }


def _facts() -> EpisodeFacts:
    return EpisodeFacts(
        episode_id="episode",
        split_clip_id="split",
        source_id="source",
        source_session_id="session",
        source_session_location="/input/session",
        start_ns=0,
        exclusive_end_ns=1_000_000_000,
        sampling_fps=30,
        row_count=30,
        timestamp_source=TimestampSource.CAPTURE,
        split_contract_digest="contract",
    )


def test_episode_schema_is_exact_and_nested_items_are_non_nullable() -> None:
    """Base field order and nested nullability are publication compatibility gates."""
    expected_names = [
        "record_schema_version",
        "media_contract_version",
        "episode_id",
        "split_clip_id",
        "source_id",
        "source_session_id",
        "source_session_location",
        "start_ns",
        "exclusive_end_ns",
        "sampling_fps",
        "row_count",
        "timestamp_source",
        "split_contract_digest",
        "materialization_digest",
        "camera_media",
        "sensor_media",
    ]

    assert EPISODE_SCHEMA.names == expected_names
    assert all(not field.nullable for field in EPISODE_SCHEMA)
    camera_list = EPISODE_SCHEMA.field("camera_media").type
    assert pa.types.is_large_list(camera_list)
    assert not camera_list.value_field.nullable
    assert camera_list.value_type == CAMERA_MEDIA_STRUCT
    assert all(not field.nullable for field in CAMERA_MEDIA_STRUCT)
    sensor_list = EPISODE_SCHEMA.field("sensor_media").type
    assert pa.types.is_large_list(sensor_list)
    assert not sensor_list.value_field.nullable
    assert sensor_list.value_type == SENSOR_MEDIA_STRUCT
    assert all(not field.nullable for field in SENSOR_MEDIA_STRUCT)


def test_record_builder_sorts_cameras_and_camera_resolution_is_exact() -> None:
    """Storage order is canonical while consumers must still name their camera."""
    record = build_episode_record(_facts(), [_camera("left", "2"), _camera("front", "1")])
    table = episode_table([record])

    assert table.schema == EPISODE_SCHEMA
    assert [camera["camera_key"] for camera in record["camera_media"]] == ["front", "left"]
    assert resolve_camera_media(record, "front")["media_location"] == "/output/front.mp4"
    with pytest.raises(ValueError, match="0 camera descriptors"):
        resolve_camera_media(record, "rear")


def test_record_builder_rejects_the_wrong_clock_or_row_count() -> None:
    """The base schema version pins capture-clock and delivered-grid semantics."""
    facts = _facts()
    with pytest.raises(ValueError, match="capture timestamps"):
        build_episode_record(
            replace(facts, timestamp_source=TimestampSource.CONTAINER_PTS),
            [_camera("front", "1")],
        )
    with pytest.raises(ValueError, match="row_count=29"):
        build_episode_record(replace(facts, row_count=29), [_camera("front", "1")])


def test_record_builder_uses_the_delivered_grid_row_count(
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    tmp_path: Path,
) -> None:
    """A whole-frame v1 duration may have a fractional-nanosecond period."""
    config = make_config(
        tmp_path / "input",
        tmp_path / "output",
        duration_s=0.0009765625,
        output_fps=1024,
    )
    duration_ns = clip_duration_ns(config)
    grid = sampling_grid_ns(Span(start_ns=0, end_ns=duration_ns), config)
    facts = replace(
        _facts(),
        exclusive_end_ns=duration_ns,
        sampling_fps=config.clip.sampling_fps,
        row_count=len(grid),
    )

    record = build_episode_record(facts, [_camera("front", "1")])

    assert duration_ns == 976_562
    assert record["row_count"] == 1


def test_delivered_grid_rounds_each_float64_offset(
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    tmp_path: Path,
) -> None:
    """At 30 FPS index two distinguishes nearest rounding from the stale floor formula."""
    config = make_config(tmp_path / "input", tmp_path / "output")
    grid = sampling_grid_ns(Span(start_ns=0, end_ns=1_000_000_000), config)

    assert int(grid[2]) == 66_666_667


def _sensor_media(key: str, marker: str) -> dict[str, object]:
    return {
        "sensor_key": key,
        "sensor_type": "egotrajectory",
        "source_media_location": f"/input/session/{key}.mcap",
        "media_location": f"/output/{key}.mcap",
        "media_size_bytes": 10,
        "media_sha256": marker * 64,
    }


def test_sensor_media_is_stored_in_a_canonical_order() -> None:
    """Two runs that assembled the same clip must produce the same row.

    Outcome order is whatever the grouping handed over, and a row whose list
    order follows that is a row whose bytes depend on scheduling -- so the same
    clip could be published twice under two different serializations.
    """
    facts = _facts()

    record = build_episode_record(
        facts,
        [_camera("front", "1")],
        [_sensor_media("imu", "4"), _sensor_media("ego", "5")],
    )

    assert [sensor["sensor_key"] for sensor in record["sensor_media"]] == ["ego", "imu"]
    assert episode_table([record]).schema == EPISODE_SCHEMA


def test_a_duplicate_sensor_key_is_refused() -> None:
    """One clip cannot hold the same sensor twice, whatever produced the outcomes."""
    facts = _facts()

    with pytest.raises(ValueError, match="sensor keys must be unique"):
        build_episode_record(facts, [_camera("front", "1")], [_sensor_media("ego", "4"), _sensor_media("ego", "5")])
