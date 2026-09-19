# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused canonical episode fixtures for multimodal caption tests."""

from collections.abc import Callable
from pathlib import Path

import lance
import pyarrow as pa
import pytest

from cosmos_curator.core.sensors.sampling.grid import make_ts_grid
from cosmos_curator.core.sensors.types.types import TimestampSource
from cosmos_curator.next.recipes.multimodal_split.records import EPISODE_SCHEMA, EpisodeFacts, build_episode_record


def camera(
    key: str,
    marker: str,
    *,
    location: str | None = None,
    size: int = 10,
    media_sha256: str | None = None,
) -> dict[str, object]:
    """Build one canonical camera descriptor."""
    return {
        "camera_key": key,
        "source_media_location": f"/source/{key}.mp4",
        "media_location": location or f"/output/{key}.mp4",
        "media_size_bytes": size,
        "media_sha256": marker * 64 if media_sha256 is None else media_sha256,
        "timeline_location": f"/output/{key}.mp4.timestamps.npy",
        "timeline_sha256": "a" * 64,
        "sensor_timestamps_location": f"/output/{key}.mp4.sensor_timestamps.npy",
        "sensor_timestamps_sha256": "b" * 64,
    }


def sensor(
    key: str = "ego",
    *,
    location: str = "/output/ego.mcap",
    size: int = 10,
    media_sha256: str = "e" * 64,
) -> dict[str, object]:
    """Build one canonical EgoTrajectory descriptor."""
    return {
        "sensor_key": key,
        "sensor_type": "egotrajectory",
        "source_media_location": "/source/ego.mcap",
        "media_location": location,
        "media_size_bytes": size,
        "media_sha256": media_sha256,
    }


def episode(
    index: int,
    *,
    episode_id: str | None = None,
    camera_media: list[dict[str, object]] | None = None,
    sensor_media: list[dict[str, object]] | None = None,
    start_ns: int = 0,
    exclusive_end_ns: int = 1_000_000_000,
    timestamp_source: TimestampSource = TimestampSource.SENSOR,
) -> dict[str, object]:
    """Build one canonical two-camera episode."""
    sampling_fps = 30
    _, _, timestamps_ns = make_ts_grid(
        start_ns,
        exclusive_end_ns=exclusive_end_ns,
        sample_rate_hz=sampling_fps,
    )
    facts = EpisodeFacts(
        episode_id=episode_id or f"episode-{index}",
        split_clip_id=f"split-{index}",
        source_id=f"source-{index}",
        source_session_id=f"session-{index}",
        source_session_location=f"/input/session-{index}",
        start_ns=start_ns,
        exclusive_end_ns=exclusive_end_ns,
        sampling_fps=sampling_fps,
        row_count=len(timestamps_ns),
        timestamp_source=timestamp_source,
        split_contract_digest="split-contract",
    )
    if camera_media is None:
        camera_media = [camera("front_left", "1"), camera("front_wide", "2")]
    return build_episode_record(facts, camera_media, sensor_media)


@pytest.fixture
def camera_factory() -> Callable[..., dict[str, object]]:
    """Expose the canonical camera descriptor builder."""
    return camera


@pytest.fixture
def episode_factory() -> Callable[..., dict[str, object]]:
    """Expose the canonical episode builder."""
    return episode


@pytest.fixture
def sensor_factory() -> Callable[..., dict[str, object]]:
    """Expose the canonical sensor descriptor builder."""
    return sensor


@pytest.fixture
def episode_dataset_factory(
    tmp_path: Path,
) -> Callable[..., tuple[str, lance.LanceDataset]]:
    """Create canonical local episode tables."""
    created = 0

    def create(*, rows: list[dict[str, object]] | None = None) -> tuple[str, lance.LanceDataset]:
        nonlocal created
        created += 1
        uri = str(tmp_path / f"episodes-{created}.lance")
        dataset = lance.write_dataset(
            pa.Table.from_pylist(rows or [episode(0), episode(1)], schema=EPISODE_SCHEMA),
            uri,
            data_storage_version="2.2",
            max_rows_per_file=2,
        )
        return uri, dataset

    return create
