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

"""Real media cutting, episode assembly, and Lance publication without Ray execution."""

import hashlib
from collections.abc import Callable
from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import ray

from cosmos_curator.next.recipes.multimodal_split.config import ResolvedMultimodalSplitConfig
from cosmos_curator.next.recipes.multimodal_split.discovery import discover_candidate_sessions
from cosmos_curator.next.recipes.multimodal_split.lance_sink import (
    append_episode_fragment,
    open_or_create_episode_table,
)
from cosmos_curator.next.recipes.multimodal_split.plan import CAMERA_PLAN, MCAP_PLAN, span_planner_batch
from cosmos_curator.next.recipes.multimodal_split.publication import assemble_episode_group, stage_episode_batch
from cosmos_curator.next.recipes.multimodal_split.split import split_camera_batch, split_mcap_batch


@pytest.fixture(autouse=True)
def forbid_ray_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject accidental Ray initialization in local batch tests."""

    def unexpected_start(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Local publication tests must not initialize Ray")

    monkeypatch.setattr(ray, "init", unexpected_start)


def _assemble_and_publish(config: ResolvedMultimodalSplitConfig) -> pa.Table:
    """Compose real batch stages locally and publish only complete episode groups.

    This tests the batch and storage contracts. Ray routing and aggregation are
    covered separately by the pipeline integration smoke test.
    """
    profile = config.execution.storage_profile
    sessions = discover_candidate_sessions(config.input, storage_profile=profile)
    plans = span_planner_batch(sessions, config=config, storage_profile=profile)
    camera_plans = plans.filter(pc.equal(plans["result_kind"], CAMERA_PLAN))
    sensor_plans = plans.filter(pc.equal(plans["result_kind"], MCAP_PLAN))
    outcomes = pa.concat_tables(
        [
            split_camera_batch(camera_plans, config=config, storage_profile=profile),
            split_mcap_batch(sensor_plans, config=config, storage_profile=profile),
        ]
    )
    assembled = pa.concat_tables(
        [
            assemble_episode_group(outcomes.filter(pc.equal(outcomes["split_clip_id"], clip_id)))
            for clip_id in pc.unique(outcomes["split_clip_id"]).to_pylist()
        ]
    )
    uri = config.output.episodes_lance_uri
    open_or_create_episode_table(uri=uri, storage_profile=profile)
    complete = assembled.filter(pc.equal(assembled["assembly_status"], "complete"))
    staged = stage_episode_batch(complete, uri=uri, storage_profile=profile)
    for candidate in staged["candidate_json"].to_pylist():
        append_episode_fragment(candidate, uri=uri, storage_profile=profile, attempts=3)
    return assembled


def test_local_publication_keeps_one_episode_carrying_both_its_video_and_its_trajectory(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_ego_mcap: Callable[..., Path],
    make_imu_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A clip is one moment across every modality, and the catalogue says so in one row.

    The whole point of the join: a consumer reads an episode and finds both the
    MP4 and the MCAP under one clip id, rather than having to pair two datasets
    on a span it has to reconstruct. Two copied MCAP sensors at once (ego, imu)
    proves assembly and publication distinguish them rather than one clobbering the
    other -- unit coverage already checks each in isolation.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    make_ego_mcap(session)
    make_imu_mcap(session)
    media = tmp_path / "media"

    assembled = _assemble_and_publish(
        make_config(
            root,
            media,
            mcap_sensors={"ego": "ego/egotrajectory.mcap"},
            imu_sensors={"imu": "imu/imu.mcap"},
        )
    )

    assert assembled["assembly_status"].to_pylist() == ["complete"] * 3
    # Counted apart from the cameras: a trajectory rides along with a clip
    # rather than being a clip of its own, so folding it into the camera total
    # would report six clips for a run that produced three.
    assert pc.sum(assembled["camera_clips_written"]).as_py() == 3
    assert pc.sum(assembled["sensor_clips_written"]).as_py() == 6
    # Each total is the bytes of its own kind of artifact, kept apart so a run
    # can say what its video cost separately from what its trajectories did.
    mcaps = sorted(media.rglob("*.mcap"))
    assert len(mcaps) == 6
    assert pc.sum(assembled["sensor_clip_bytes"]).as_py() == sum(path.stat().st_size for path in mcaps)
    assert pc.sum(assembled["camera_clip_bytes"]).as_py() == sum(path.stat().st_size for path in media.rglob("*.mp4"))

    episodes = lance.dataset(str(media / "episodes.lance")).to_table().to_pylist()
    assert len(episodes) == 3
    for episode in episodes:
        assert [camera["camera_key"] for camera in episode["camera_media"]] == ["front"]
        sensors_by_key = {sensor["sensor_key"]: sensor for sensor in episode["sensor_media"]}
        assert set(sensors_by_key) == {"ego", "imu"}
        assert sensors_by_key["ego"]["sensor_type"] == "egotrajectory"
        assert sensors_by_key["imu"]["sensor_type"] == "imu"
        for sensor in sensors_by_key.values():
            payload = Path(sensor["media_location"]).read_bytes()
            assert len(payload) == sensor["media_size_bytes"]
            assert hashlib.sha256(payload).hexdigest() == sensor["media_sha256"]
            # Beside the video it belongs to, under the one clip id.
            assert Path(sensor["media_location"]).parent == Path(episode["camera_media"][0]["media_location"]).parent


def test_a_trajectory_that_fails_to_cut_drops_the_episode_rather_than_publishing_a_partial_one(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_ego_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A planned trajectory that produced no clip costs the episode its video too.

    Symmetric with a camera that failed to encode. An episode published without
    it would claim a clip is complete while quietly holding less than the run
    was asked for, and nothing downstream could tell the difference.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    # A gap across the middle clip, which is the one span that cannot be cut.
    make_ego_mcap(session, gap=(100, 200))
    media = tmp_path / "media"

    assembled = _assemble_and_publish(make_config(root, media, mcap_sensors={"ego": "ego/egotrajectory.mcap"}))

    assert assembled.num_rows == 3
    assert assembled["assembly_status"].to_pylist().count("complete") == 2
    assert assembled["assembly_status"].to_pylist().count("dropped") == 1
    episodes = lance.dataset(str(media / "episodes.lance")).to_table().to_pylist()
    assert len(episodes) == 2
    assert all(len(episode["sensor_media"]) == 1 for episode in episodes)
    dropped = assembled.filter(pc.equal(assembled["assembly_status"], "dropped"))["episode_id"].to_pylist()
    assert not set(dropped) & {episode["episode_id"] for episode in episodes}
