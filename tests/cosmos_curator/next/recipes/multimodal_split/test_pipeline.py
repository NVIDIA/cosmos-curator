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

"""End-to-end tests for the ``multimodal-split`` run path."""

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lance
import numpy as np
import pyarrow as pa
import pytest
import ray

from cosmos_curator.core.sensors.utils.timestamps import write_timestamps_ns
from cosmos_curator.next.media.spans import Span
from cosmos_curator.next.recipes.multimodal_split import pipeline
from cosmos_curator.next.recipes.multimodal_split import split as split_module
from cosmos_curator.next.recipes.multimodal_split.config import (
    MultimodalSplitExecutionConfig,
    ResolvedMultimodalSplitConfig,
)
from cosmos_curator.next.recipes.multimodal_split.pipeline import run_config
from cosmos_curator.next.recipes.multimodal_split.plan import plan_common_spans
from cosmos_curator.next.recipes.multimodal_split.reports import (
    failure,
    plan_error_uri,
    sensor_error_uri,
    write_sensor_errors,
)


def test_episode_accounting_rejects_a_missing_terminal_outcome() -> None:
    """The independently planned count must equal the three terminal states."""
    with pytest.raises(RuntimeError, match=r"planned=1.*already_canonical=0.*newly_published=0.*dropped=0"):
        pipeline.validate_episode_accounting(
            episodes_planned=1,
            episodes_already_canonical=0,
            episodes_newly_published=0,
            episodes_dropped=0,
        )


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_episode_id_count_is_global_across_ray_blocks() -> None:
    """Real Ray blocks are the only way to prove the count is block-layout independent."""
    episode_ids = ray.data.from_arrow(
        [
            pa.table({"split_clip_id": ["a", "b"]}),
            pa.table({"split_clip_id": ["a", "b"]}),
        ]
    )

    assert episode_ids.num_blocks() == 2
    assert pipeline._count_unique_episode_ids(episode_ids) == 2


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_a_run_writes_one_clip_per_span_across_every_session(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """Two sessions of three whole clips each produce six playable files and a matching count."""
    root = tmp_path / "recordings"
    make_session(root, "session-000")
    make_session(root, "session-001")
    media = tmp_path / "media"

    summary = run_config(make_config(root, media))

    assert summary["sessions"] == 2
    assert summary["camera_clips_written"] == 6
    assert summary["episodes_planned"] == 6
    assert summary["episodes_already_canonical"] == 0
    assert summary["episodes_newly_published"] == 6
    assert summary["episodes_dropped"] == 0
    assert summary["failures"] == 0
    assert len(list(media.rglob("*.mp4"))) == 6
    episodes = lance.dataset(str(media / "episodes.lance")).to_table().to_pylist()
    assert len(episodes) == 6
    selected = episodes[0]["camera_media"][0]
    payload = Path(selected["media_location"]).read_bytes()
    assert len(payload) == selected["media_size_bytes"]
    assert hashlib.sha256(payload).hexdigest() == selected["media_sha256"]


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_whole_session_mode_publishes_one_episode_covering_the_full_session(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A whole-session run collapses what would tile into three clips into one.

    The same session that ``test_a_run_writes_one_clip_per_span_across_every_session``
    turns into three clips at ``duration_s=1.0`` becomes exactly one here, and
    that one clip's window is the camera's own recorded span rather than a
    ``duration_s`` multiple of it.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    media = tmp_path / "media"
    config = make_config(root, media, whole_session=True)

    summary = run_config(config)

    assert summary["sessions"] == 1
    assert summary["camera_clips_written"] == 1
    assert summary["episodes_planned"] == 1
    assert summary["episodes_already_canonical"] == 0
    assert summary["episodes_newly_published"] == 1
    assert summary["episodes_dropped"] == 0
    assert summary["failures"] == 0
    assert len(list(media.rglob("*.mp4"))) == 1

    timestamps_ns = np.load(session / "camera" / "front.mp4.timestamps.npy")
    spans = plan_common_spans({"front": timestamps_ns}, config)
    assert len(spans) == 1

    episodes = lance.dataset(str(media / "episodes.lance")).to_table().to_pylist()
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode["start_ns"] == spans[0].start_ns
    assert episode["exclusive_end_ns"] == spans[0].end_ns
    assert [camera["camera_key"] for camera in episode["camera_media"]] == ["front"]
    selected = episode["camera_media"][0]
    payload = Path(selected["media_location"]).read_bytes()
    assert len(payload) == selected["media_size_bytes"]
    assert hashlib.sha256(payload).hexdigest() == selected["media_sha256"]


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_one_unusable_session_does_not_end_the_run(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A recording that cannot be opened is quarantined, and its neighbours still publish."""
    root = tmp_path / "recordings"
    make_session(root, "session-000")
    (root / "session-001").mkdir(parents=True)
    media = tmp_path / "media"

    summary = run_config(make_config(root, media))

    assert summary["sessions"] == 2
    assert summary["camera_clips_written"] == 3
    assert summary["failures"] == 1
    assert len(list(media.rglob("*.mp4"))) == 3


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_a_fully_reconciled_rerun_is_a_zero_encode_noop(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """Canonical rows remove their camera work before any stable artifact path is touched."""
    root = tmp_path / "recordings"
    make_session(root, "session-000")
    config = make_config(root, tmp_path / "media")
    first = run_config(config)

    second = run_config(config)

    assert second["episodes_planned"] == first["episodes_planned"] == 3
    assert second["episodes_already_canonical"] == 3
    assert second["episodes_newly_published"] == 0
    assert second["episodes_dropped"] == 0
    assert second["camera_clips_written"] == 0
    assert second["episodes_lance_version"] == first["episodes_lance_version"]
    assert lance.dataset(config.output.episodes_lance_uri).count_rows() == 3


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_a_post_planning_camera_failure_drops_only_the_incomplete_episodes(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful sibling artifacts survive while no partial episode row is published."""
    root = tmp_path / "recordings"
    make_session(root, "session-000", uri_pattern="camera/front.mp4")
    make_session(root, "session-000", uri_pattern="camera/left.mp4")
    media = tmp_path / "media"
    config = make_config(
        root,
        media,
        cameras={"front": "camera/front.mp4", "left": "camera/left.mp4"},
    )
    real_split = split_module.split_session

    def fail_left_camera(
        item: dict[str, Any],
        *,
        config: ResolvedMultimodalSplitConfig,
        storage_profile: str = "default",
    ) -> list[dict[str, Any]]:
        if item["sensor_key"] != "left":
            return real_split(item, config=config, storage_profile=storage_profile)
        failures = [
            failure(
                str(item["source_session_id"]),
                "left",
                Span(start_ns=int(span["start_ns"]), end_ns=int(span["exclusive_end_ns"])),
                "sample_and_encode",
                OSError("controlled post-planning camera failure"),
                str(span["clip_id"]),
            )
            for span in item["spans"]
        ]
        write_sensor_errors(
            str(item["source_session_id"]),
            "left",
            failures,
            output_path=config.output.path,
            storage_profile=storage_profile,
        )
        return [split_module._sensor_outcome(item, span, None) for span in item["spans"]]

    def fail_left_batch(
        plans: pa.Table,
        *,
        config: ResolvedMultimodalSplitConfig,
        storage_profile: str = "default",
    ) -> pa.Table:
        outcomes = [
            outcome
            for item in plans.to_pylist()
            for outcome in fail_left_camera(item, config=config, storage_profile=storage_profile)
        ]
        return pa.Table.from_pylist(outcomes, schema=split_module.SENSOR_OUTCOME_SCHEMA)

    monkeypatch.setattr(pipeline, "split_camera_batch", fail_left_batch)

    summary = run_config(config)

    assert summary["episodes_planned"] == 3
    assert summary["episodes_newly_published"] == 0
    assert summary["episodes_dropped"] == 3
    assert summary["camera_clips_written"] == 3
    assert lance.dataset(config.output.episodes_lance_uri).count_rows() == 0
    assert len(list(media.rglob("front.mp4"))) == 3
    assert Path(sensor_error_uri(str(media), "session-000", "left")).is_file()


def test_a_run_that_finds_no_sessions_is_refused(
    tmp_path: Path,
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """An empty prefix is far more often a wrong path than a genuinely empty dataset."""
    root = tmp_path / "recordings"
    root.mkdir()

    with pytest.raises(ValueError, match="0 candidate session"):
        run_config(make_config(root, tmp_path / "media"))


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_execution_settings_reach_the_runtime_rather_than_being_accepted_and_ignored(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A setting that never reaches what it configures is worse than no setting at all.

    Both of these are execution-only: neither changes a written byte, which is
    why they live outside the settings that decide clip identity.
    """
    seen: dict[str, object] = {}
    monkeypatch.setattr(pipeline, "configure_ray_data_progress", lambda *, progress: seen.update(progress=progress))
    root = tmp_path / "recordings"
    make_session(root, "session-000")
    config = make_config(root, tmp_path / "media", progress=True, storage_profile="hyperion")

    run_config(config)

    assert seen["progress"] is True
    assert config.execution.storage_profile == "hyperion"


def test_the_progress_bar_is_off_unless_asked_for() -> None:
    """A batch run writes to a log, where a redrawing progress bar is noise."""
    assert MultimodalSplitExecutionConfig().progress is False
    assert MultimodalSplitExecutionConfig().storage_profile == "default"


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_a_run_reports_elapsed_time_and_bytes_written(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """Bytes are counted from what was written, not estimated from the config.

    A run that reports a plausible total it did not measure is worse than one
    that reports nothing, so this compares against the files on disk.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000")
    media = tmp_path / "media"

    summary = run_config(make_config(root, media))

    on_disk = sum(path.stat().st_size for path in media.rglob("*.mp4"))
    assert summary["clip_bytes"] == on_disk
    assert summary["elapsed_seconds"] > 0


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_failures_are_summarized_by_category_rather_than_one_line_each(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A bad run produces thousands of failures, and a per-failure log buries the shape of them."""
    root = tmp_path / "recordings"
    make_session(root, "session-000")
    for name in ("session-001", "session-002"):
        (root / name).mkdir(parents=True)

    summary = run_config(make_config(root, tmp_path / "media"))

    assert summary["failures"] == 2
    assert summary["failures_by_category"] == {"FileNotFoundError": 2}


def _split_stage_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    config: ResolvedMultimodalSplitConfig,
) -> dict[str, object]:
    """Return the keyword arguments the recipe hands Ray Data, keyed by stage function."""
    seen: dict[str, object] = {}

    class _FakeDataset:
        def filter(self, **_kwargs: object) -> object:
            return self

        def map_batches(self, fn: object, **kwargs: object) -> object:
            # Keyed by the stage, not merged. Merging let a reservation moved from
            # one stage to the other pass unnoticed, which is the single thing
            # these tests exist to catch.
            seen[getattr(fn, "__name__", str(fn))] = dict(kwargs)
            return self

        def union(self, *_others: object) -> object:
            return self

    monkeypatch.setattr(pipeline.ray.data, "from_arrow", lambda *_a, **_k: _FakeDataset())
    sessions = pa.table({"source_session_id": ["s"], "session_uri": ["u"]})
    pipeline.clip_dataset(pipeline.plan_dataset(sessions, config), config)
    return seen


def _stage_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    config: ResolvedMultimodalSplitConfig,
) -> list[dict[str, object]]:
    """Return the keyword arguments each stage hands Ray Data, in stage order."""
    calls: list[dict[str, object]] = []

    class _FakeDataset:
        def filter(self, **_kwargs: object) -> object:
            return self

        def map_batches(self, _fn: object, **kwargs: object) -> object:
            calls.append(dict(kwargs))
            return self

        def union(self, *_others: object) -> object:
            return self

    monkeypatch.setattr(pipeline.ray.data, "from_arrow", lambda *_a, **_k: _FakeDataset())
    sessions = pa.table({"source_session_id": ["s"], "session_uri": ["u"]})
    pipeline.clip_dataset(pipeline.plan_dataset(sessions, config), config)
    return calls


@pytest.mark.parametrize("num_cpus", [0.125, 0.5])
def test_the_planner_is_scheduled_on_the_cpu_the_config_states(
    tmp_path: Path,
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
    num_cpus: float,
) -> None:
    """The setting has to reach Ray, not merely exist.

    It once did not: the field, its default, its docstring and the template all
    described a figure an operator could tune per storage backend, while the
    planner scheduled on a constant. Two values here, so a reintroduced constant
    fails rather than matching whichever one it happened to be.
    """
    config = make_config(tmp_path / "recordings", tmp_path / "media", planner_num_cpus=num_cpus)

    planner, *_splitters = _stage_kwargs(monkeypatch, config)

    assert planner["num_cpus"] == num_cpus


def test_a_declared_session_heap_is_what_ray_reserves(
    tmp_path: Path,
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ray schedules on declared memory, so an undeclared task gets a number invented for it.

    Ray Data fills in 2.4 GiB per CPU for any map task that does not say, which
    on a 75 GiB node stops the run at 31 concurrent sessions however high the
    session cap is set. The number is a hyperscaler rule of thumb, not a
    measurement of this recipe, and it is nowhere near what a streamed session
    actually holds.
    """
    config = make_config(tmp_path / "recordings", tmp_path / "media", memory_gb=1.5)

    seen = _split_stage_kwargs(monkeypatch, config)

    # The camera stage specifically. The two split stages are scheduled apart so
    # the copied sensor is not priced at the decoder's rate, which only holds if
    # the reservation lands on the stage the setting names.
    assert seen["split_camera_batch"]["memory"] == int(1.5 * 1024**3)
    assert "memory" not in seen["split_mcap_batch"]


def test_an_undeclared_session_heap_leaves_ray_to_its_own_default(
    tmp_path: Path,
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No measurement, no reservation: a guess of ours would be worth no more than Ray's."""
    config = make_config(tmp_path / "recordings", tmp_path / "media")

    seen = _split_stage_kwargs(monkeypatch, config)

    assert seen["split_camera_batch"]["memory"] is None


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_a_flood_of_failures_is_counted_without_being_collected(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A bad run must not hold every failure to count them, nor name them all.

    The failure set is bounded by what the data did, not by anything this
    process chose: a run whose every span was refused would otherwise assemble
    one row per span in the driver and print a warning for each. The summary
    streams them, keeping a few examples to make the cause actionable.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000")
    for index in range(8):
        (root / f"session-{index + 1:03d}").mkdir(parents=True)
    config = make_config(root, tmp_path / "media")

    summary = run_config(config)
    failures = pipeline.read_failures(config)

    assert summary["failures"] == 8
    assert summary["failures_by_category"] == {"FileNotFoundError": 8}
    assert failures.total == 8
    # Named examples are capped; the count carries the rest.
    assert len(failures.examples) == 5


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_a_damaged_plan_report_is_named_rather_than_counted_as_a_clean_session(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """An unreadable report and an absent one mean opposite things, so they cannot share a return.

    A session that plans cleanly writes no report, so absence is the ordinary
    case and says nothing. A report that exists but will not parse is a session
    whose recorded failures are sitting on disk and missing from the total: the
    summary under-reports, and the warning is the only thing that says so.
    Returning None for both is what makes the log line load-bearing.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000")
    for index in (1, 2):
        (root / f"session-{index:03d}").mkdir(parents=True)
    config = make_config(root, tmp_path / "media")
    run_config(config)

    damaged = Path(plan_error_uri(config.output.path, "session-001"))
    assert damaged.is_file(), "the planner must have written the report this test then damages"
    damaged.write_text('{"failures": [', encoding="utf-8")

    failures = pipeline.read_failures(config)

    # The intact report still counts; the damaged one is short by its own.
    assert failures.total == 1


def test_an_unreadable_report_is_logged_where_an_absent_one_is_not(
    tmp_path: Path,
    loguru_records: list[dict[str, Any]],
) -> None:
    """Both return nothing, so the log line is the only thing that tells them apart.

    The count alone cannot: a session short by its own failures looks exactly
    like a session that had none. Whoever is reconciling a total against the
    errors tree gets the path and the parse error, or gets nothing at all.
    """
    damaged = tmp_path / "span_planner.json"
    damaged.write_text('{"failures": [', encoding="utf-8")

    assert pipeline._read_json(str(damaged), storage_profile="default") is None
    assert pipeline._read_json(str(tmp_path / "absent.json"), storage_profile="default") is None

    (warning,) = [record for record in loguru_records if "Could not read" in record["message"]]
    assert warning["level"].name == "WARNING"
    assert str(damaged) in warning["message"]
    assert "JSONDecodeError" in warning["message"]


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_a_split_failure_reaches_the_summary(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A run that lost clips must not summarise as clean.

    Planning and splitting write their reports into one tree, so counting what
    is in it counts both. Asking each session for a planning report instead
    would find only planning's, and a run whose spans died in the split would
    report zero failures beside a clip count nobody could reconcile.

    The failure is made rather than mocked, because splitting happens in a Ray
    worker and a patch in this process would never reach it: a timeline
    describing more observations than its container holds is discovered by the
    walk, which is the one kind of split failure a test can arrange from outside.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000", frames=90)
    config = make_config(root, tmp_path / "media", duration_s=1.0)

    timeline_path = next((root / "session-000").rglob("*.timestamps.npy"))
    written = np.load(timeline_path)
    step = int(written[1] - written[0])
    with timeline_path.open("wb") as handle:
        write_timestamps_ns(np.arange(int(written[0]), int(written[0]) + 180 * step, step, dtype=np.int64), handle)

    summary = run_config(config)

    assert summary["camera_clips_written"] > 0, "the servable spans should still have been written"
    assert summary["failures"] > 0, "the spans the media could not serve are missing from the summary"
    assert summary["failures_by_category"]


def test_unreachable_storage_stops_the_run_before_ray_starts(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The point of checking is that it happens before a cluster exists.

    A credential that cannot reach the input otherwise fails once per session
    inside Ray: a thousand identical reports and a run that writes nothing and
    exits successfully. Nothing else pins the ordering, so dropping the checks
    or moving them after ``ensure_ray_initialized`` would pass every other test.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000")
    started = {"ray": False}

    def refuse(location: str, *, storage_profile: str) -> None:
        del storage_profile
        msg = f"Cannot read the input prefix {location}: no credentials"
        raise RuntimeError(msg)

    def note_ray_start(*args: object, **kwargs: object) -> None:
        del args, kwargs
        started["ray"] = True

    monkeypatch.setattr(pipeline, "check_readable", refuse)
    monkeypatch.setattr(pipeline, "ensure_ray_initialized", note_ray_start)

    with pytest.raises(RuntimeError, match="Cannot read the input prefix"):
        run_config(make_config(root, tmp_path / "media"))

    assert not started["ray"], "storage was checked after Ray had already started"


def test_a_run_publishes_one_episode_carrying_both_its_video_and_its_trajectory(
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
    proves the real pipeline distinguishes them rather than one clobbering the
    other -- unit coverage already checks each in isolation.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    make_ego_mcap(session)
    make_imu_mcap(session)
    media = tmp_path / "media"

    summary = run_config(
        make_config(
            root,
            media,
            mcap_sensors={"ego": "ego/egotrajectory.mcap"},
            imu_sensors={"imu": "imu/imu.mcap"},
        )
    )

    assert summary["episodes_newly_published"] == 3
    assert summary["episodes_dropped"] == 0
    assert summary["failures"] == 0
    # Counted apart from the cameras: a trajectory rides along with a clip
    # rather than being a clip of its own, so folding it into the camera total
    # would report six clips for a run that produced three.
    assert summary["camera_clips_written"] == 3
    assert summary["sensor_clips_written"] == 6
    # Each total is the bytes of its own kind of artifact, kept apart so a run
    # can say what its video cost separately from what its trajectories did.
    mcaps = sorted(media.rglob("*.mcap"))
    assert len(mcaps) == 6
    assert summary["sensor_clip_bytes"] == sum(path.stat().st_size for path in mcaps)
    assert summary["clip_bytes"] == sum(path.stat().st_size for path in media.rglob("*.mp4"))

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

    summary = run_config(make_config(root, media, mcap_sensors={"ego": "ego/egotrajectory.mcap"}))

    assert summary["episodes_planned"] == 3
    assert summary["episodes_newly_published"] == 2
    assert summary["episodes_dropped"] == 1
    episodes = lance.dataset(str(media / "episodes.lance")).to_table().to_pylist()
    assert len(episodes) == 2
    assert all(len(episode["sensor_media"]) == 1 for episode in episodes)


@pytest.mark.integration
def test_a_run_publishes_an_episode_carrying_a_gridded_sensor(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_gps_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A placed sensor reaches the catalogue like a copied one, with null grid sidecars.

    A gridded sensor sits on the episode's own timeline, so it has no timeline or
    sensor-timestamps artifact of its own. Its outcome row carries nulls where a
    camera's carries those descriptors, and assembly refuses a successful outcome
    missing any descriptor that applies to its kind -- so this is where a sensor
    kind is published or silently dropped.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    make_gps_mcap(session)
    media = tmp_path / "media"

    summary = run_config(
        make_config(root, media, gps_sensors={"gps": "gps/gps.mcap"}, sampling="interpolate", duration_s=1.0)
    )

    assert summary["episodes_newly_published"] == 3
    assert summary["episodes_dropped"] == 0
    assert summary["failures"] == 0
    assert summary["sensor_clips_written"] == 3

    episodes = lance.dataset(str(media / "episodes.lance")).to_table().to_pylist()
    assert len(episodes) == 3
    for episode in episodes:
        gps = next(sensor for sensor in episode["sensor_media"] if sensor["sensor_key"] == "gps")
        assert gps["sensor_type"] == "gps"
        # A gridded sensor sits on the episode's own timeline, so it has neither a
        # timeline of its own nor a per-row record of which sample served which
        # row -- and those are absent from its catalogue entry rather than null
        # within it, which is the difference from a camera.
        assert set(gps) == {
            "sensor_key",
            "sensor_type",
            "source_media_location",
            "media_location",
            "media_size_bytes",
            "media_sha256",
        }
        sidecars = {
            "timeline_location",
            "timeline_sha256",
            "sensor_timestamps_location",
            "sensor_timestamps_sha256",
        }
        assert sidecars.isdisjoint(gps)
        assert sidecars <= set(episode["camera_media"][0])
        payload = Path(gps["media_location"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == gps["media_sha256"]
        assert Path(gps["media_location"]).parent == Path(episode["camera_media"][0]["media_location"]).parent


@pytest.mark.integration
def test_a_gps_sensor_that_cannot_be_aligned_drops_its_episode(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_gps_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A clip whose position cannot be placed is not published without it."""
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    make_gps_mcap(session, dropped=frozenset({1}))
    media = tmp_path / "media"

    summary = run_config(
        make_config(
            root,
            media,
            gps_sensors={"gps": "gps/gps.mcap"},
            sampling="interpolate",
            max_gap_ms=1200,
            duration_s=1.0,
        )
    )

    assert summary["episodes_dropped"] > 0
    assert summary["episodes_newly_published"] == summary["episodes_planned"] - summary["episodes_dropped"]
    episodes = lance.dataset(str(media / "episodes.lance")).to_table().to_pylist()
    # Non-empty first: ``all`` over nothing is true, so a run that dropped every
    # episode would otherwise satisfy the line below.
    assert episodes
    assert all(any(s["sensor_key"] == "gps" for s in e["sensor_media"]) for e in episodes)
