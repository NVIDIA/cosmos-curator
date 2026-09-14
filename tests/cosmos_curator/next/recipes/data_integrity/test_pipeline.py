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

"""End-to-end tests for the data-integrity Ray Data recipe.

These run a real local Ray cluster over real (tiny) H.264 files and publish a real
Lance store: the per-session fan-out, the nesting guard and the one-commit rule are
the code under test, so nothing about them is stubbed.
"""

import json
import pathlib
from collections.abc import Callable
from typing import Any

import pytest
import yaml

from cosmos_curator.core.sensors.data_integrity.instruments import (
    INSTRUMENTS,
    NAME_ORDERING,
    SESSION_INSTRUMENTS,
)
from cosmos_curator.next.recipes.data_integrity import pipeline, store
from cosmos_curator.next.recipes.data_integrity.config import TOOL_NAME, resolve_config_data

#: The two surviving sessions hold four streams between them -- one unreadable, and
#: one that only the recursive listing of ``b`` reaches, since the nested session
#: naming it directly is dropped before the run.
NUM_SESSIONS = 2
NUM_STREAMS = 4
NUM_MEASURED = 3
NUM_METRICS = len(INSTRUMENTS)


def _nothing_readable(store_root: pathlib.Path) -> bool:
    """Report whether a store holds nothing a reader will believe.

    Not the same as an absent directory, which is what this used to check: a run that
    fails before its commit may already have appended rows -- an unlisted session's
    ``session.lance`` row, for one. Uncommitted rows are the point of the commit
    marker, so what has to hold is that no reader sees them.
    """
    return not (
        store.read_streams(str(store_root)) or store.read_sessions(str(store_root)) or store.read_runs(str(store_root))
    )


@pytest.fixture
def dataset(tmp_path: pathlib.Path, h264_video: Callable[..., bytes]) -> pathlib.Path:
    """Two sessions, one nested inside the other, and one unreadable video.

    ``b/nested`` is named as its own session as well as being part of ``b``, which is
    how one source comes to be reachable twice in one run without any cloud plumbing.
    """
    clips = tmp_path / "clips"
    (clips / "a").mkdir(parents=True)
    (clips / "b" / "nested").mkdir(parents=True)
    (clips / "a" / "front.mp4").write_bytes(h264_video())
    (clips / "a" / "broken.mp4").write_bytes(b"not a video")
    (clips / "b" / "rear.mp4").write_bytes(h264_video())
    (clips / "b" / "nested" / "side.mp4").write_bytes(h264_video())
    return clips


def _config_data(dataset: pathlib.Path, store_root: pathlib.Path, **execution: Any) -> dict[str, Any]:  # noqa: ANN401
    """Config naming all three session paths of the fixture dataset."""
    return {
        "schema_version": 1,
        "kind": "data-integrity",
        "input": {
            "sessions": [
                str(dataset / "a"),
                str(dataset / "b"),
                str(dataset / "b" / "nested"),
            ]
        },
        "output": {"store_root": str(store_root)},
        "execution": {"append_batch_size": 2, **execution},
    }


@pytest.fixture
def summary(dataset: pathlib.Path, tmp_path: pathlib.Path) -> dict[str, object]:
    """Run the pipeline once, since every store assertion reads the same run."""
    return pipeline.run_config(resolve_config_data(_config_data(dataset, tmp_path / "di-store")))


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_the_run_summarizes_what_it_covered(summary: dict[str, object]) -> None:
    """Two surviving sessions, four streams counted from the workers, one unreadable."""
    assert summary["sessions"] == NUM_SESSIONS
    assert summary["streams"] == NUM_STREAMS
    assert summary["unreadable"] == 1
    # Nothing went unmeasured here, which is what lets this run exit 0.
    assert summary["unreachable"] == 0
    assert summary["unlisted_sessions"] == 0
    assert summary["empty_sessions"] == 0


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_the_whole_invocation_commits_exactly_once(summary: dict[str, object], tmp_path: pathlib.Path) -> None:
    """Many sessions, one run id, one commit -- the rule this recipe inherits."""
    runs = store.read_runs(str(tmp_path / "di-store"))

    assert [row["run_id"] for row in runs] == [summary["run_id"]]
    assert runs[0]["tool"] == TOOL_NAME
    assert runs[0]["session_path"] is None
    assert runs[0]["num_streams"] == NUM_STREAMS


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_every_stream_is_stored_once_under_this_run(summary: dict[str, object], tmp_path: pathlib.Path) -> None:
    """Reading without ``include_incomplete`` proves the rows are believed, not just present."""
    streams = store.read_streams(str(tmp_path / "di-store"))

    assert len(streams) == NUM_STREAMS
    assert {row["run_id"] for row in streams} == {summary["run_id"]}


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_a_nested_session_is_measured_once_under_the_session_enclosing_it(
    summary: dict[str, object],
    dataset: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Dropping the nested session is what keeps one source from being stored twice.

    Each session is measured on its own, so nothing downstream could reconcile two
    rows carrying one ``stream_id``; the guard in ``expand_sessions`` is the only
    thing standing between this dataset and that collision.
    """
    del summary
    rows = store.read_streams(str(tmp_path / "di-store"))
    shared = str(dataset / "b" / "nested" / "side.mp4")
    attributions = [row["session_path"] for row in rows if row["source"] == shared]

    assert attributions == [str(dataset / "b")]
    assert str(dataset / "b" / "nested") not in {row["session_path"] for row in rows}


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_each_stream_keeps_the_session_it_was_found_under(
    summary: dict[str, object],
    dataset: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """One run id for the invocation, but provenance stays per session."""
    del summary
    by_source = {row["source"]: row["session_path"] for row in store.read_streams(str(tmp_path / "di-store"))}

    assert by_source[str(dataset / "a" / "front.mp4")] == str(dataset / "a")
    assert by_source[str(dataset / "a" / "broken.mp4")] == str(dataset / "a")
    assert by_source[str(dataset / "b" / "rear.mp4")] == str(dataset / "b")


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_an_unreadable_stream_is_recorded_and_never_judged(
    summary: dict[str, object],
    dataset: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """One corrupt file is a finding about the data, not a failure of the run."""
    del summary
    root = str(tmp_path / "di-store")
    broken = str(dataset / "a" / "broken.mp4")
    streams = {row["source"]: row for row in store.read_streams(root)}

    assert streams[broken]["error"]
    assert broken not in {row["source"] for row in store.read_measurements(root, NAME_ORDERING)}
    assert broken not in {row["source"] for row in store.read_evaluations(root)}


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_every_measured_stream_contributes_a_row_per_metric(summary: dict[str, object], tmp_path: pathlib.Path) -> None:
    """Five metrics, three measured streams, and one verdict for each pair."""
    del summary
    root = str(tmp_path / "di-store")

    for spec in INSTRUMENTS:
        assert len(store.read_measurements(root, spec.name)) == NUM_MEASURED, spec.name
    assert len(store.read_evaluations(root)) == NUM_MEASURED * NUM_METRICS


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_every_measured_session_is_stored_once(
    summary: dict[str, object], dataset: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A session's own row is what makes it countable without joining to its streams."""
    sessions = {row["session_path"]: row for row in store.read_sessions(str(tmp_path / "di-store"))}

    assert set(sessions) == {str(dataset / "a"), str(dataset / "b")}
    assert {row["run_id"] for row in sessions.values()} == {summary["run_id"]}
    # Session ``a`` is where the corrupt file lives, and the nested session's stream
    # is counted under ``b``, which enclosed it.
    assert (sessions[str(dataset / "a")]["num_streams"], sessions[str(dataset / "a")]["num_unreadable"]) == (2, 1)
    assert (sessions[str(dataset / "b")]["num_streams"], sessions[str(dataset / "b")]["num_unreadable"]) == (2, 0)
    assert all(row["error"] is None for row in sessions.values())


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_every_session_contributes_a_row_per_session_metric(summary: dict[str, object], tmp_path: pathlib.Path) -> None:
    """The pipeline measures the session as well as its streams, and stores both."""
    del summary
    root = str(tmp_path / "di-store")

    for spec in SESSION_INSTRUMENTS:
        assert len(store.read_session_measurements(root, spec.name)) == NUM_SESSIONS, spec.name
    assert len(store.read_session_evaluations(root)) == NUM_SESSIONS * len(SESSION_INSTRUMENTS)


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_a_failing_session_metric_reaches_the_store(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
    capture_timeline: Callable[..., pathlib.Path],
) -> None:
    """The regression this MR exists to prevent: a session verdict that only the report knew.

    Two sensors that stopped a second apart, judged by a policy that allows no
    non-overlap at all. Every stream passes on its own, so a store reconstructing the
    session from stream rows alone would read back ``PASS``. Capture timelines beside
    the clips are what let the session be judged rather than skipped.
    """
    clips = tmp_path / "clips"
    (clips / "ragged").mkdir(parents=True)
    for name, frames in (("front.mp4", 30), ("rear.mp4", 60)):
        (clips / "ragged" / name).write_bytes(h264_video(frames=frames))
        capture_timeline(clips / "ragged" / name, frames=frames)
    store_root = tmp_path / "di-store"

    summary = pipeline.run_config(
        resolve_config_data(
            {
                "schema_version": 1,
                "kind": "data-integrity",
                "input": {"sessions": [str(clips / "ragged")]},
                "output": {"store_root": str(store_root)},
                "checks": {"thresholds": {"max_non_overlap_percent": 0.0}},
            }
        )
    )

    assert {row["check_status"] for row in store.read_evaluations(str(store_root))} == {"PASS"}
    verdicts = {row["metric_name"]: row for row in store.read_session_evaluations(str(store_root))}
    assert verdicts["multi_sensor_overlap"]["check_status"] == "FAIL"
    assert store.session_rollup(str(store_root))["status"] == "FAIL"
    # And the run says so without a reader having to roll the store up: the per-session
    # counters are summed on the driver, which nothing else here exercises.
    assert summary["failed_session_metrics"] == 1
    assert summary["failed_metrics"] == 0


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_the_manifest_describes_the_run_rather_than_a_session(
    summary: dict[str, object], tmp_path: pathlib.Path
) -> None:
    """``session_path`` is null because this run covers many sessions."""
    manifest = store.read_manifest(str(tmp_path / "di-store"))

    assert manifest["run_id"] == summary["run_id"]
    assert manifest["tool"] == TOOL_NAME
    assert manifest["session_path"] is None
    assert manifest["num_streams"] == NUM_STREAMS


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_a_run_with_findings_still_exits_zero(dataset: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """The runtime's contract, not ``di-session``'s: findings are results, not failures."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config_data(dataset, tmp_path / "di-store")))

    assert pipeline.main([str(config_path)]) == 0


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_a_config_override_reaches_the_run(dataset: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """``--set`` is resolved like any other input before the cluster starts."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config_data(dataset, tmp_path / "di-store")))

    assert pipeline.main([str(config_path), "--set", "input.limit=1"]) == 0
    # One stream from each of the two surviving sessions.
    assert len(store.read_streams(str(tmp_path / "di-store"))) == NUM_SESSIONS


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_the_run_widens_rays_no_progress_window_to_the_configured_one(
    dataset: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Left at Ray's own 30 minutes, a slow session fails the run after everything else was measured.

    Asserted at this call site rather than only on the helper: the widening is a
    property of what this recipe asks for, and a call that stopped passing the window
    would restore the default silently.
    """
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        pipeline,
        "configure_ray_data_stability",
        lambda **kwargs: captured.update(kwargs),
    )

    pipeline.run_config(resolve_config_data(_config_data(dataset, tmp_path / "di-store")))

    assert captured["no_progress_timeout_s"] == 2 * 60 * 60


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_an_operator_can_widen_the_no_progress_window_further(
    dataset: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default fits the sessions seen so far, not every session anyone will point this at."""
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        pipeline,
        "configure_ray_data_stability",
        lambda **kwargs: captured.update(kwargs),
    )

    pipeline.run_config(resolve_config_data(_config_data(dataset, tmp_path / "di-store", no_progress_timeout_s=-1)))

    assert captured["no_progress_timeout_s"] == -1


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_a_stale_session_path_costs_the_run_its_exit_status_but_not_its_rows(
    dataset: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """The measurements survive the failure, which is the whole point of failing after the commit.

    One stale entry in a long input is close to inevitable, and aborting would discard
    every session already measured -- for a session that was never going to contribute
    a row anyway.
    """
    store_root = tmp_path / "di-store"
    config = _config_data(dataset, store_root)
    stale = str(dataset / "gone")
    config["input"]["sessions"].append(stale)

    with pytest.raises(pipeline.IncompleteRunError, match="could not be listed") as raised:
        pipeline.run_config(resolve_config_data(config))

    assert stale in str(raised.value)
    runs = store.read_runs(str(store_root))
    assert len(runs) == 1, "the run committed, so its rows are readable"
    assert len(store.read_streams(str(store_root))) == NUM_STREAMS

    # The stale session is in the store too, as a row that says why it is empty. The
    # run still failed: the row explains the gap, it does not close it.
    sessions = {row["session_path"]: row for row in store.read_sessions(str(store_root))}
    assert set(sessions) == {str(dataset / "a"), str(dataset / "b"), stale}
    assert sessions[stale]["error"]
    assert (sessions[stale]["num_streams"], sessions[stale]["num_unreadable"]) == (0, 0)
    assert stale not in {row["session_path"] for row in store.read_session_evaluations(str(store_root))}
    assert store.session_rollup(str(store_root))["status"] == "ERROR"


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_an_unreachable_stream_costs_the_run_its_exit_status_but_adds_no_row(
    dataset: pathlib.Path,
    tmp_path: pathlib.Path,
    unreachable_video: Callable[[pathlib.Path], pathlib.Path],
) -> None:
    """Adding a stream nobody can read must leave the store exactly as it was.

    Every session here lists cleanly, so this is the other half of the coverage
    contract: a run can fall short without a single stale path, and the streams it could
    not reach have to be missing from the store rather than present as findings.
    """
    locked = unreachable_video(dataset / "a" / "locked.mp4")
    store_root = tmp_path / "di-store"

    with pytest.raises(pipeline.IncompleteRunError, match="could not be reached"):
        pipeline.run_config(resolve_config_data(_config_data(dataset, store_root)))

    runs = store.read_runs(str(store_root))
    rows = store.read_streams(str(store_root))
    assert len(runs) == 1, "the run committed, so what it did measure is readable"
    # Rows written, not streams found: the run row is a claim about what is in the store.
    assert runs[0]["num_streams"] == NUM_STREAMS
    assert len(rows) == NUM_STREAMS
    assert str(locked) not in {row["source"] for row in rows}

    # The store does not forget the shortfall, though: session "a" was found to hold a
    # stream the store has no row for, so it reads ERROR rather than PASS. Otherwise
    # only the run's exit status knew, and that is gone by the next read.
    sessions = {row["session_path"]: row["status"] for row in store.session_rollup(str(store_root))["sessions"]}
    assert sessions == {str(dataset / "a"): "ERROR", str(dataset / "b"): "PASS"}
    assert store.session_rollup(str(store_root))["status"] == "ERROR"


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_a_run_that_could_list_nothing_at_all_commits_nothing(tmp_path: pathlib.Path) -> None:
    """With no measurement to preserve there is nothing to commit, so this fails the old way."""
    store_root = tmp_path / "di-store"

    with pytest.raises(ValueError, match="could not be listed"):
        pipeline.run_config(
            resolve_config_data(
                {
                    "schema_version": 1,
                    "kind": "data-integrity",
                    "input": {"sessions": [str(tmp_path / "absent-one"), str(tmp_path / "absent-two")]},
                    "output": {"store_root": str(store_root)},
                }
            )
        )

    # The two sessions did each append a row saying they could not be listed, but the
    # run never reached its commit, so nothing above is readable.
    assert _nothing_readable(store_root)
    assert len(store.read_sessions(str(store_root), include_incomplete=True)) == 2


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_a_run_that_measured_nothing_names_every_reason_it_did_not(tmp_path: pathlib.Path) -> None:
    """One unlistable session and one empty one call for different fixes, so both are named.

    Reporting whichever came first would send an operator after the wrong thing: a
    missing path is not an empty session, and ``input.limit`` has nothing to do with
    either of them.
    """
    empty = tmp_path / "clips" / "empty"
    empty.mkdir(parents=True)
    absent = tmp_path / "clips" / "absent"
    store_root = tmp_path / "di-store"

    with pytest.raises(ValueError, match="nothing measured across 2 session") as raised:
        pipeline.run_config(
            resolve_config_data(
                {
                    "schema_version": 1,
                    "kind": "data-integrity",
                    "input": {"sessions": [str(empty), str(absent)]},
                    "output": {"store_root": str(store_root)},
                }
            )
        )

    message = str(raised.value)
    assert "1 session(s) could not be listed" in message
    assert "1 session(s) held no video streams" in message
    assert _nothing_readable(store_root)


#: The three clauses a zero-row run can carry, one per cause.
_UNLISTED_CLAUSE = "session(s) could not be listed"
_UNREACHED_CLAUSE = "stream(s) could not be reached"
_EMPTY_CLAUSE = "session(s) held no video streams"
_ALL_CLAUSES = (_UNLISTED_CLAUSE, _UNREACHED_CLAUSE, _EMPTY_CLAUSE)


@pytest.mark.parametrize(
    ("unlisted", "unreachable", "empty", "expected"),
    [
        (3, 0, 0, {_UNLISTED_CLAUSE}),
        (0, 0, 3, {_EMPTY_CLAUSE}),
        (1, 0, 2, {_UNLISTED_CLAUSE, _EMPTY_CLAUSE}),
        (1, 4, 0, {_UNLISTED_CLAUSE, _UNREACHED_CLAUSE}),
        (0, 2, 1, {_UNREACHED_CLAUSE, _EMPTY_CLAUSE}),
        (1, 2, 1, set(_ALL_CLAUSES)),
    ],
)
def test_every_way_to_measure_nothing_reads_as_itself(
    unlisted: int, unreachable: int, empty: int, expected: set[str]
) -> None:
    """Every combination of causes, at the message rather than through a run.

    Each combination is reachable by a run -- a mode-000 file is a read that fails under
    a listing that does not -- but that would be a fixture per row for a function whose
    whole job is arithmetic on three counters. The counters themselves are covered by
    the runs above; what is left to pin is that the message spends all of them.
    """
    message = pipeline._nothing_measured(3, unreachable=unreachable, unlisted=unlisted, empty=empty)

    for clause in _ALL_CLAUSES:
        named = clause in message
        assert named is (clause in expected), f"{clause!r} {'named' if named else 'missing'} in {message!r}"


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_a_session_that_holds_no_streams_is_reported_without_failing_the_run(
    dataset: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """The quiet case: a cloud prefix that no longer holds videos lists clean and empty.

    Nothing distinguishes it from a session genuinely without video, so it is counted
    and reported rather than treated as a failure.
    """
    (dataset / "c").mkdir()
    config = _config_data(dataset, tmp_path / "di-store")
    config["input"]["sessions"].append(str(dataset / "c"))

    summary = pipeline.run_config(resolve_config_data(config))

    assert summary["empty_sessions"] == 1
    assert summary["streams"] == NUM_STREAMS


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_sessions_holding_no_streams_fail_without_committing(tmp_path: pathlib.Path) -> None:
    """A run that measured nothing is a config mistake, not a passing run.

    Only knowable after every session has been listed, so the error comes after the
    append loop -- which did write the empty session's own rows, and left them
    uncommitted and therefore unreadable, exactly as any run that does not reach its
    commit.
    """
    empty = tmp_path / "clips" / "empty"
    empty.mkdir(parents=True)
    store_root = tmp_path / "di-store"

    with pytest.raises(ValueError, match="no video streams"):
        pipeline.run_config(
            resolve_config_data(
                {
                    "schema_version": 1,
                    "kind": "data-integrity",
                    "input": {"sessions": [str(empty)]},
                    "output": {"store_root": str(store_root)},
                }
            )
        )

    assert _nothing_readable(store_root)


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_an_unwritable_store_root_fails_the_run(dataset: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """An operational failure propagates, so the runtime can exit nonzero on it."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("in the way")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config_data(dataset, blocker / "di-store")))

    with pytest.raises(OSError, match="not-a-directory"):
        pipeline.main([str(config_path)])
