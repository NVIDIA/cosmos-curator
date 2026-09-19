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

"""Tests for deciding which spans of a session every sensor can serve."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.multimodal_split import plan as plan_module
from cosmos_curator.next.recipes.multimodal_split.config import ResolvedMultimodalSplitConfig
from cosmos_curator.next.recipes.multimodal_split.identities import make_source_id
from cosmos_curator.next.recipes.multimodal_split.plan import (
    ALREADY_CANONICAL,
    CAMERA_PLAN,
    MCAP_PLAN,
    PLANNING_RESULT_SCHEMA,
    first_uncovered,
    plan_common_spans,
    remaining_episode_ids_batch,
    span_planner,
)
from cosmos_curator.next.recipes.multimodal_split.reports import plan_error_uri
from cosmos_curator.next.recipes.multimodal_split.storage import resolve_sensor_pattern

from .conftest import DEVICE_ORIGIN_NS

_MS = 1_000_000
_EPOCH = 1_700_000_000 * 1_000_000_000


@pytest.mark.parametrize(
    ("offset_ns", "covered"),
    [
        (49 * _MS, True),
        # Exactly at the tolerance. The comparison is ``worst > max``, so equal
        # is covered -- and a flip to ``>=`` would pass every other test here,
        # which all sit well clear of the boundary.
        (50 * _MS, True),
        (50 * _MS + 1, False),
        (51 * _MS, False),
    ],
)
def test_the_tolerance_is_inclusive_at_its_boundary(offset_ns: int, *, covered: bool) -> None:
    """A frame exactly ``max_delta_s`` from a reference still serves that moment.

    Which way the boundary falls decides whether a rig at the edge of its
    tolerance yields clips or is refused, and nothing else in the suite pins it:
    every other case is a comfortable margin either side.
    """
    max_delta_ns = 50 * _MS
    references = np.array([_EPOCH, _EPOCH + 100 * _MS], dtype=np.int64)
    timeline = references + offset_ns

    missing = first_uncovered(references, {"front": timeline}, max_delta_ns)

    assert (missing is None) is covered
    if missing is not None:
        assert missing == ("front", offset_ns)


def _existing_row(
    entry: dict[str, object],
    source_media_location: str,
    sensor_media_location: str | None = None,
) -> dict[str, object]:
    return {
        "record_schema_version": entry["record_schema_version"],
        "media_contract_version": entry["media_contract_version"],
        "episode_id": entry["episode_id"],
        "source_id": entry["source_id"],
        "sampling_fps": entry["sampling_fps"],
        "timestamp_source": entry["timestamp_source"],
        "split_contract_digest": entry["split_contract_digest"],
        "camera_media": [{"camera_key": "front", "source_media_location": source_media_location}],
        "sensor_media": (
            []
            if sensor_media_location is None
            else [{"sensor_key": "ego", "source_media_location": sensor_media_location}]
        ),
    }


def test_exact_canonical_episodes_skip_before_manifest_writes(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The inert reconciliation map removes exact episodes before any artifact writer."""
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    item = {"source_session_id": "session-000", "session_uri": str(session)}
    config = make_config(root, tmp_path / "first")
    first = span_planner(item, config=config)
    camera_plan = first[0]
    source_id = make_source_id(str(session))
    existing = {
        str(entry["episode_id"]): _existing_row(entry, str(camera_plan["sensor_uri"])) for entry in camera_plan["spans"]
    }
    monkeypatch.setattr(
        plan_module,
        "write_clip_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("manifest rewritten")),
    )

    reconciled = span_planner(
        item,
        config=make_config(root, tmp_path / "second"),
        reconciliation={source_id: existing},
    )

    assert len(reconciled) == len(existing)
    assert {row["result_kind"] for row in reconciled} == {ALREADY_CANONICAL}
    assert len({row["episode_id"] for row in reconciled}) == len(existing)


def test_remaining_episode_projection_deduplicates_multicamera_spans(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """One planned episode is counted once even though every camera carries its span."""
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    camera_plan = span_planner(
        {"source_session_id": "session-000", "session_uri": str(session)},
        config=make_config(root, tmp_path / "media"),
    )[0]
    second_camera = {**camera_plan, "sensor_key": "left"}
    plans = pa.Table.from_pylist([camera_plan, second_camera], schema=PLANNING_RESULT_SCHEMA)

    episode_ids = remaining_episode_ids_batch(plans)["split_clip_id"].to_pylist()

    assert episode_ids == sorted(span["clip_id"] for span in camera_plan["spans"])


def test_changed_camera_source_conflicts_before_manifest_writes(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A logical retry cannot replace bytes sourced from a different camera object."""
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    item = {"source_session_id": "session-000", "session_uri": str(session)}
    config = make_config(root, tmp_path / "first")
    camera_plan = span_planner(item, config=config)[0]
    entry = camera_plan["spans"][0]
    source_id = make_source_id(str(session))
    existing = _existing_row(entry, "/different/front.mp4")
    monkeypatch.setattr(
        plan_module,
        "write_clip_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("manifest rewritten")),
    )

    with pytest.raises(ValueError, match="conflicts"):
        span_planner(item, config=config, reconciliation={source_id: {str(entry["episode_id"]): existing}})


def test_a_missing_required_camera_refuses_the_whole_session(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A rig that promised a view and did not record it produces no clips at all.

    Not clips without that view: a clip is one moment across every view the rig
    promised, and a consumer reading a short one cannot tell a rig that has no
    left camera from a session where the left camera died.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    config = make_config(root, tmp_path / "media", cameras={"front": "camera/front.mp4", "left": "camera/left.mp4"})

    results = span_planner({"source_session_id": "session-000", "session_uri": str(session)}, config=config)

    assert results == []
    report = json.loads(Path(plan_error_uri(config.output.path, "session-000")).read_bytes())
    assert [entry["sensor_key"] for entry in report["failures"]] == ["left"]


def test_a_missing_optional_camera_leaves_the_session_planning(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A view declared optional is left out of the clip rather than refusing it.

    The episode's planned key list is what records the absence, so a consumer
    reads which views the clip actually holds instead of inferring it.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    config = make_config(
        root,
        tmp_path / "media",
        cameras={"front": "camera/front.mp4", "left": "camera/left.mp4"},
        optional_cameras=frozenset({"left"}),
    )

    results = span_planner({"source_session_id": "session-000", "session_uri": str(session)}, config=config)

    assert [row["sensor_key"] for row in results] == ["front"]
    assert all(span["planned_camera_keys"] == ["front"] for span in results[0]["spans"])
    assert not Path(plan_error_uri(config.output.path, "session-000")).exists()


def test_a_camera_missing_its_timeline_is_refused_without_the_fallback_flag(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A recording with video and no sensor timeline stays a broken camera by default."""
    root = tmp_path / "recordings"
    session = make_session(root, "session-000", write_timestamps=False)
    config = make_config(root, tmp_path / "media")

    results = span_planner({"source_session_id": "session-000", "session_uri": str(session)}, config=config)

    assert results == []
    report = json.loads(Path(plan_error_uri(config.output.path, "session-000")).read_bytes())
    assert [entry["sensor_key"] for entry in report["failures"]] == ["front"]


def test_a_camera_missing_its_timeline_falls_back_to_container_pts_when_allowed(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """``allow_pts_fallback`` reads a camera's timeline from its own container instead of refusing it."""
    root = tmp_path / "recordings"
    session = make_session(root, "session-000", write_timestamps=False)
    config = make_config(root, tmp_path / "media", pts_fallback_cameras=frozenset({"front"}))

    results = span_planner({"source_session_id": "session-000", "session_uri": str(session)}, config=config)

    assert not Path(plan_error_uri(config.output.path, "session-000")).exists()
    assert [row["sensor_key"] for row in results] == ["front"]
    assert results[0]["spans"]
    assert all(span["timestamp_source"] == "container_pts" for span in results[0]["spans"])


def test_a_camera_whose_container_pts_also_fails_is_an_open_sensor_failure(
    tmp_path: Path,
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """``allow_pts_fallback`` is a fallback, not a second guarantee: a camera that fails both stays failed.

    The session's video is unreadable garbage rather than missing, so
    ``load_sensor_timestamps_ns`` fails for the usual reason (no sidecar) and the
    fallback then fails too, for a different one (no parseable container).
    Both are camera-read faults, and the second must not escape as anything
    other than the same per-camera ``open_sensor`` failure the first is.
    """
    root = tmp_path / "recordings"
    camera_path = root / "session-000" / "camera" / "front.mp4"
    camera_path.parent.mkdir(parents=True)
    camera_path.write_bytes(b"not a video container")
    config = make_config(root, tmp_path / "media", pts_fallback_cameras=frozenset({"front"}))

    results = span_planner(
        {"source_session_id": "session-000", "session_uri": str(root / "session-000")}, config=config
    )

    assert results == []
    report = json.loads(Path(plan_error_uri(config.output.path, "session-000")).read_bytes())
    assert [entry["sensor_key"] for entry in report["failures"]] == ["front"]
    assert [entry["stage"] for entry in report["failures"]] == ["open_sensor"]


def test_a_malformed_timeline_sidecar_never_falls_back(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A present-but-corrupt sidecar is a fault, not a missing one, even with the flag set.

    ``allow_pts_fallback`` exists for a sidecar that was never written, not one
    that was written badly: a truncated or malformed ``timestamps.npy`` in the
    listing must fail the camera outright, the same as it would with the flag
    off, rather than being quietly reinterpreted as container PTS.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    (session / "camera" / "front.mp4.timestamps.npy").write_bytes(b"not a valid npy timeline")
    config = make_config(root, tmp_path / "media", pts_fallback_cameras=frozenset({"front"}))

    results = span_planner({"source_session_id": "session-000", "session_uri": str(session)}, config=config)

    assert results == []
    report = json.loads(Path(plan_error_uri(config.output.path, "session-000")).read_bytes())
    assert [entry["sensor_key"] for entry in report["failures"]] == ["front"]


def test_a_mixed_session_marks_the_whole_episode_container_pts(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One camera falling back to container PTS taints the whole episode's timestamp_source.

    A clip is one moment across every view the rig promised, so a reader must
    not be able to read ``timestamp_source`` as "sensor" for an episode where
    only one of two cameras actually has a sensor timeline.

    ``load_container_pts_ns`` is faked to land on the same clock as ``left``'s
    real sensor timeline: a real container's own PTS restarts at 0 per file
    (see ``CameraSensor``'s docstring on why it's an inferior fallback), so two
    cameras on genuinely unrelated clocks would cover no common window at all --
    a real planning failure, but not the one this test is pinning.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000", uri_pattern="camera/front.mp4", write_timestamps=False)
    make_session(root, "session-000", uri_pattern="camera/left.mp4", write_timestamps=True)
    config = make_config(
        root,
        tmp_path / "media",
        cameras={"front": "camera/front.mp4", "left": "camera/left.mp4"},
        pts_fallback_cameras=frozenset({"front"}),
    )

    def fake_load_container_pts_ns(*_args: object, **_kwargs: object) -> np.ndarray:
        period_ns = round(1_000_000_000 / 30)
        return np.array([DEVICE_ORIGIN_NS + index * period_ns for index in range(90)], dtype=np.int64)

    monkeypatch.setattr(plan_module, "load_container_pts_ns", fake_load_container_pts_ns)

    results = span_planner(
        {"source_session_id": "session-000", "session_uri": str(root / "session-000")}, config=config
    )

    assert not Path(plan_error_uri(config.output.path, "session-000")).exists()
    assert sorted(row["sensor_key"] for row in results) == ["front", "left"]
    for row in results:
        assert row["spans"]
        assert all(span["timestamp_source"] == "container_pts" for span in row["spans"])


def test_every_camera_optional_and_absent_still_records_a_reason(
    tmp_path: Path,
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A session where nothing opened is refused even though nothing was required.

    Optional means its absence does not refuse the session, not that a session
    holding none of its views quietly produces nothing: the error file exists so
    that "this session yielded no clips" always has a reason beside it.
    """
    root = tmp_path / "recordings"
    session = root / "session-000"
    session.mkdir(parents=True)
    config = make_config(root, tmp_path / "media", optional_cameras=frozenset({"front"}))

    results = span_planner({"source_session_id": "session-000", "session_uri": str(session)}, config=config)

    assert results == []
    report = json.loads(Path(plan_error_uri(config.output.path, "session-000")).read_bytes())
    assert [entry["sensor_key"] for entry in report["failures"]] == [None]


_SECOND = 1_000_000_000


def test_an_mcap_sensor_narrows_the_planned_window(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_ego_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A clip cannot run past the last pose, so the trajectory's end ends the session.

    The camera records three seconds and the trajectory stops after 2.2, which
    leaves two whole clips rather than three. The third would hold video for a
    second the trajectory says nothing about.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    make_ego_mcap(session, count=220)
    config = make_config(root, tmp_path / "media", mcap_sensors={"ego": "ego/egotrajectory.mcap"})

    results = span_planner({"source_session_id": "session-000", "session_uri": str(session)}, config=config)

    spans = next(row for row in results if row["sensor_key"] == "front")["spans"]
    assert [span["start_ns"] - DEVICE_ORIGIN_NS for span in spans] == [0, _SECOND]


def test_an_mcap_sensor_is_planned_without_its_messages(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_ego_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """The plan row names the topic to cut and carries no timeline for it.

    Planning reads two index entries; the splitter reads the messages. Carrying
    them here would move a three-hour recording through a plan row to save a read
    the splitter makes anyway.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    ego = make_ego_mcap(session)
    config = make_config(root, tmp_path / "media", mcap_sensors={"ego": "ego/egotrajectory.mcap"})

    results = span_planner({"source_session_id": "session-000", "session_uri": str(session)}, config=config)

    ego_plans = [row for row in results if row["result_kind"] == MCAP_PLAN]
    assert len(ego_plans) == 1
    assert ego_plans[0]["sensor_key"] == "ego"
    assert ego_plans[0]["sensor_uri"] == str(ego)
    assert ego_plans[0]["sensor_type"] == "egotrajectory"
    assert ego_plans[0]["timeline_ns"] is None
    # By value against the bounds themselves, not against the camera plan's span
    # list -- the planner hands both sensors the same list object, so comparing
    # the two is ``x == x`` and passes however the spans were built.
    assert [
        (span["start_ns"] - DEVICE_ORIGIN_NS, span["exclusive_end_ns"] - DEVICE_ORIGIN_NS)
        for span in ego_plans[0]["spans"]
    ] == [
        (0, _SECOND),
        (_SECOND, 2 * _SECOND),
        (2 * _SECOND, 3 * _SECOND),
    ]


def test_an_imu_sensor_plans_alongside_egotrajectory(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_ego_mcap: Callable[..., Path],
    make_imu_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """Two copied MCAP sensors plan independently, each correctly typed.

    IMU shares egotrajectory's whole code path (COPIED_SENSOR_TYPES, generic
    open_mcap_sensors) -- what's worth a dedicated check is that a session
    naming both at once tags each plan row with its own sensor_type rather
    than one clobbering the other.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    ego = make_ego_mcap(session)
    imu = make_imu_mcap(session)
    config = make_config(
        root,
        tmp_path / "media",
        mcap_sensors={"ego": "ego/egotrajectory.mcap"},
        imu_sensors={"imu": "imu/imu.mcap"},
    )

    results = span_planner({"source_session_id": "session-000", "session_uri": str(session)}, config=config)

    mcap_plans = {row["sensor_key"]: row for row in results if row["result_kind"] == MCAP_PLAN}
    assert set(mcap_plans) == {"ego", "imu"}
    assert mcap_plans["ego"]["sensor_type"] == "egotrajectory"
    assert mcap_plans["ego"]["sensor_uri"] == str(ego)
    assert mcap_plans["imu"]["sensor_type"] == "imu"
    assert mcap_plans["imu"]["sensor_uri"] == str(imu)


def test_a_missing_required_mcap_sensor_refuses_the_session(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A trajectory the rig promised and did not record costs the session its video too."""
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    config = make_config(root, tmp_path / "media", mcap_sensors={"ego": "ego/egotrajectory.mcap"})

    results = span_planner({"source_session_id": "session-000", "session_uri": str(session)}, config=config)

    assert results == []
    report = json.loads(Path(plan_error_uri(config.output.path, "session-000")).read_bytes())
    assert [entry["sensor_key"] for entry in report["failures"]] == ["ego"]


def test_a_missing_optional_mcap_sensor_leaves_the_cameras_planning(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """An optional trajectory that was not recorded neither narrows nor refuses."""
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    config = make_config(
        root,
        tmp_path / "media",
        mcap_sensors={"ego": "ego/egotrajectory.mcap"},
        optional_sensors=frozenset({"ego"}),
    )

    results = span_planner({"source_session_id": "session-000", "session_uri": str(session)}, config=config)

    assert [row["result_kind"] for row in results] == [CAMERA_PLAN]
    assert len(results[0]["spans"]) == 3
    assert not Path(plan_error_uri(config.output.path, "session-000")).exists()


def test_an_mcap_sensor_starting_late_moves_the_first_clip(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_ego_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """The window opens where the last sensor started, which can be the trajectory.

    The twin of the narrowing test, and not implied by it: a trajectory that
    begins after the cameras would otherwise leave the first clip asking for
    poses from before the recording, and every fixture elsewhere starts both at
    the same instant, which hides it.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    make_ego_mcap(session, first_ns=DEVICE_ORIGIN_NS + _SECOND // 2, count=301)
    config = make_config(root, tmp_path / "media", mcap_sensors={"ego": "ego/egotrajectory.mcap"})

    results = span_planner({"source_session_id": "session-000", "session_uri": str(session)}, config=config)

    spans = next(row for row in results if row["sensor_key"] == "front")["spans"]
    assert [span["start_ns"] - DEVICE_ORIGIN_NS for span in spans] == [_SECOND // 2, _SECOND + _SECOND // 2]


def test_an_mcap_bound_is_converted_from_inclusive_to_exclusive(
    tmp_path: Path,
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A trajectory's last message is an instant inside its window, not the window's end.

    An MCAP reports both bounds inclusively -- ``last_ns`` is a message that
    exists -- while a span ends exclusively. Planning converts by adding a
    nanosecond, and the conversion only changes the tiling when the last message
    falls one nanosecond short of a clip boundary, because whole clips are what
    gets tiled. So it is exercised here directly rather than through a session:
    every realistic recording rounds the difference away, which is exactly why
    a wrong conversion would sit unnoticed.
    """
    config = make_config(tmp_path / "input", tmp_path / "media", duration_s=1.0)
    camera = np.arange(0, 4 * _SECOND, _SECOND // 30, dtype=np.int64)

    spans = plan_common_spans({"front": camera}, config, bounds={"ego": (0, 2 * _SECOND - 1)})

    # The message at ``2s - 1`` is inside the second clip. Treating that bound as
    # already exclusive would end the window one nanosecond early and drop the
    # clip that contains it.
    assert [(span.start_ns, span.end_ns) for span in spans] == [(0, _SECOND), (_SECOND, 2 * _SECOND)]


def test_whole_session_mode_ignores_duration_s_and_emits_one_span(
    tmp_path: Path,
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A whole-session clip is one span covering everything the rig recorded.

    ``clip.whole_session`` trades the fixed-duration tiling for a single span
    the length of the common window, so a short ``duration_s`` here -- which
    would normally tile the same window into several clips -- must not change
    the outcome.
    """
    config = make_config(tmp_path / "input", tmp_path / "media", duration_s=1.0, whole_session=True)
    camera = np.arange(0, 4 * _SECOND, _SECOND // 30, dtype=np.int64)

    spans = plan_common_spans({"front": camera}, config, bounds={"ego": (0, 2 * _SECOND - 1)})

    assert [(span.start_ns, span.end_ns) for span in spans] == [(0, 2 * _SECOND)]


def test_a_transient_failure_on_an_optional_sensor_still_refuses_the_session(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_ego_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional means the sensor is not in the session, not that reading it failed.

    A timeout and an absence both arrive as exceptions, and only absence is a
    property of the recording. Swallowing a timeout would let two runs of one
    immutable session plan different windows and publish different episode sets.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    make_ego_mcap(session)
    config = make_config(
        root,
        tmp_path / "media",
        mcap_sensors={"ego": "ego/egotrajectory.mcap"},
        optional_sensors=frozenset({"ego"}),
    )
    monkeypatch.setattr(
        plan_module,
        "load_mcap_bounds",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("connection reset")),
    )

    results = span_planner({"source_session_id": "session-000", "session_uri": str(session)}, config=config)

    assert results == []
    report = json.loads(Path(plan_error_uri(config.output.path, "session-000")).read_bytes())
    assert [entry["sensor_key"] for entry in report["failures"]] == ["ego"]


def test_an_optional_sensor_that_is_present_is_planned_like_any_other(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_ego_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """Declaring a sensor optional changes what its absence costs, nothing else."""
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    make_ego_mcap(session, count=220)
    config = make_config(
        root,
        tmp_path / "media",
        mcap_sensors={"ego": "ego/egotrajectory.mcap"},
        optional_sensors=frozenset({"ego"}),
    )

    results = span_planner({"source_session_id": "session-000", "session_uri": str(session)}, config=config)

    assert [row["sensor_key"] for row in results if row["result_kind"] == MCAP_PLAN] == ["ego"]
    # Still narrows the window, exactly as a required one does.
    assert len(next(row for row in results if row["result_kind"] == CAMERA_PLAN)["spans"]) == 2


def test_a_present_camera_that_cannot_be_read_refuses_the_session_even_when_optional(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """Optional says the session may not carry it, not that a broken one is fine.

    A camera whose video is there and whose timestamps are not is a fault in the
    recording, and dropping it would quietly plan a session against fewer views
    than it has -- changing the window, and so the episodes, on the strength of a
    corrupt file.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    (session / "camera" / "front.mp4.timestamps.npy").unlink()
    config = make_config(root, tmp_path / "media", optional_cameras=frozenset({"front"}))

    results = span_planner({"source_session_id": "session-000", "session_uri": str(session)}, config=config)

    assert results == []
    report = json.loads(Path(plan_error_uri(config.output.path, "session-000")).read_bytes())
    assert [entry["sensor_key"] for entry in report["failures"]] == ["front"]


def test_an_exact_pattern_reports_absence_as_absence(tmp_path: Path) -> None:
    """A sensor the session does not carry raises FileNotFoundError from resolution.

    Deciding absence from whatever opening the file raises works only where a
    missing file raises FileNotFoundError. Object storage does not: smart_open
    surfaces a missing S3 key as a plain OSError, indistinguishable from a
    timeout -- so an optional sensor would refuse every session on S3 while
    passing every test run against a local path. Resolving against the listing
    is the one answer every backend gives alike, and it costs nothing because
    planning already holds the listing.
    """
    with pytest.raises(FileNotFoundError, match="no artifact"):
        resolve_sensor_pattern(str(tmp_path), "ego/egotrajectory.mcap", session_paths=["camera/front.mp4"])

    resolved = resolve_sensor_pattern(str(tmp_path), "camera/front.mp4", session_paths=["camera/front.mp4"])

    assert resolved == f"{tmp_path}/camera/front.mp4"


def test_a_changed_trajectory_source_conflicts_like_a_changed_camera(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_ego_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A published episode is not canonical if its trajectory came from another file.

    A wildcard resolves against the session's listing, so the same key and topic
    can name a different artifact between runs. Reconciling on the cameras alone
    would keep the trajectory cut from the old one under a canonical identity,
    and nothing downstream could tell -- the episode id is geometry, and the
    split contract digest covers the config rather than what it resolved to.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    ego = make_ego_mcap(session)
    config = make_config(root, tmp_path / "media", mcap_sensors={"ego": "ego/egotrajectory.mcap"})
    item = {"source_session_id": "session-000", "session_uri": str(session)}
    plans = span_planner(item, config=config)
    camera_plan = next(row for row in plans if row["result_kind"] == CAMERA_PLAN)
    entry = camera_plan["spans"][0]
    source_id = make_source_id(str(session))

    # Same cameras, same geometry, same contract -- only the trajectory moved.
    stale = _existing_row(entry, str(camera_plan["sensor_uri"]), "/elsewhere/egotrajectory.mcap")
    with pytest.raises(ValueError, match="conflicts"):
        span_planner(item, config=config, reconciliation={source_id: {str(entry["episode_id"]): stale}})

    # And the unchanged one still reconciles, so the check is not simply refusing.
    current = _existing_row(entry, str(camera_plan["sensor_uri"]), str(ego))
    reconciled = span_planner(item, config=config, reconciliation={source_id: {str(entry["episode_id"]): current}})
    assert any(row["result_kind"] == ALREADY_CANONICAL for row in reconciled)
