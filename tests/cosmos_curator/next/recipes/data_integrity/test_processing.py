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

"""Unit tests for the function the Ray Data stage runs on one session."""

import datetime
import pathlib
import pickle
from collections.abc import Callable, Iterator
from typing import Any

import lance
import numpy as np
import pyarrow as pa
import pytest
from loguru import logger

from cosmos_curator.core.sensors.data_integrity.instruments import INSTRUMENTS, NAME_ORDERING, SESSION_INSTRUMENTS
from cosmos_curator.core.sensors.data_integrity.results import CheckStatus, StreamResult
from cosmos_curator.next.recipes.data_integrity import processing, store, store_schema
from cosmos_curator.next.recipes.data_integrity.config import ResolvedDataIntegrityConfig
from cosmos_curator.next.recipes.mp4_index import processing as mp4_index_processing
from cosmos_curator.next.recipes.mp4_index.config import ResolvedMp4IndexConfig
from cosmos_curator.next.recipes.mp4_index.contracts import (
    CONTAINER_KIND_NONFRAGMENTED,
    VIDEO_MANIFEST_SCHEMA,
    make_video_id,
)
from cosmos_curator.next.recipes.mp4_index.index_lookup import Mp4IndexLookup
from cosmos_curator.next.utils.uris import normalize_location

RUN_ID = "0" * 32
CREATED_AT = datetime.datetime(2026, 8, 20, 12, 0, tzinfo=datetime.UTC)
#: Write stamps for tests that stack runs, distinct so which row is newest never ties.
_EARLIER = datetime.datetime(2026, 8, 21, tzinfo=datetime.UTC)
_LATER = datetime.datetime(2026, 8, 22, tzinfo=datetime.UTC)
_LATEST = datetime.datetime(2026, 8, 23, tzinfo=datetime.UTC)

#: How many metrics a stream with a usable declared rate produces, and therefore how
#: many measurement rows and how many evaluations it contributes.
NUM_METRICS = len(INSTRUMENTS)

#: Sensor timeline when present, else container PTS. These fixtures are encoded clips with
#: no recorder clock, so the default ``("eof_ns",)`` would error every stream; tests about
#: the default say so by passing ``alignment_fields=None``.
_EOF_OR_PTS = ("eof_ns", "edited_pts_ns")


def _config(
    session: pathlib.Path,
    *,
    thresholds: dict[str, Any] | None = None,
    mp4_index_uri: str | None = None,
    alignment_fields: tuple[str, ...] | None = _EOF_OR_PTS,
    **execution: Any,  # noqa: ANN401
) -> ResolvedDataIntegrityConfig:
    """Resolve a config covering one local session; ``alignment_fields=None`` keeps the default."""
    session_input: dict[str, Any] = {"sessions": [str(session)]}
    if mp4_index_uri is not None:
        session_input["mp4_index_uri"] = mp4_index_uri
    checks: dict[str, Any] = {"thresholds": thresholds or {}}
    if alignment_fields is not None:
        checks["camera_alignment_fields"] = list(alignment_fields)
    return ResolvedDataIntegrityConfig.model_validate(
        {
            "schema_version": 1,
            "kind": "data-integrity",
            "input": session_input,
            "output": {"store_root": str(session.parent / "di-store")},
            "checks": checks,
            "execution": execution,
        }
    )


@pytest.fixture
def session(tmp_path: pathlib.Path, h264_video: Callable[..., bytes]) -> pathlib.Path:
    """Build a session directory holding one readable video and one unreadable one."""
    path = tmp_path / "clips" / "session"
    path.mkdir(parents=True)
    (path / "front.mp4").write_bytes(h264_video())
    (path / "rear.mp4").write_bytes(b"not a video")
    return path


def _check(
    session: pathlib.Path,
    *,
    thresholds: dict[str, Any] | None = None,
    mp4_index_uri: str | None = None,
    alignment_fields: tuple[str, ...] | None = _EOF_OR_PTS,
    **execution: Any,  # noqa: ANN401
) -> dict[str, object]:
    """Measure one session the way the Ray stage does."""
    return processing.check_session(
        {"session_path": str(session)},
        config=_config(
            session,
            thresholds=thresholds,
            mp4_index_uri=mp4_index_uri,
            alignment_fields=alignment_fields,
            **execution,
        ),
        run_id=RUN_ID,
        created_at=CREATED_AT,
    )


def _rows(record: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    """Unpack the opaque payload the driver appends."""
    return pickle.loads(record["rows"])  # type: ignore[arg-type]  # noqa: S301 -- written by check_session


def _timestamp_sources(record: dict[str, object]) -> dict[str, object]:
    """Which clock each measured stream was timed on, keyed by its source."""
    return {str(row["source"]): row["timestamp_source"] for row in _rows(record)[store_schema.STREAM_DATASET]}


def _manifest(session: pathlib.Path, *videos: pathlib.Path) -> str:
    """Index *videos* with the mp4-index recipe and write the manifest a run would be pointed at.

    A real manifest rather than hand-written rows: the key the indexer stores, the size
    it records and its schema version all have to line up for a lookup to hit.
    """
    config = ResolvedMp4IndexConfig.model_validate(
        {
            "schema_version": 1,
            "kind": "mp4-index",
            "input": {
                "path_prefix": str(session),
                "globs": ["*.mp4"],
                "container_kind": CONTAINER_KIND_NONFRAGMENTED,
            },
            "output": {"path": str(session.parent / "mp4-index")},
        }
    )
    tables = []
    for video in videos:
        # Normalized the way discovery spells a source, since that is the key a lookup
        # builds from one and the key this row has to match.
        location = normalize_location(str(video))
        result = mp4_index_processing.process_one_video(
            {"video_id": make_video_id(location), "media_location": location}, config=config
        )
        assert result["status"] == "success", result["error_message"]
        tables.append(mp4_index_processing.result_table(result))
    uri = str(session.parent / "mp4-index" / "manifest.lance")
    lance.write_dataset(pa.concat_tables(tables), uri)
    return uri


#: A recorder epoch, so a bound on this clock cannot be mistaken for container PTS near zero.
_SENSOR_EPOCH_NS = 1_767_225_600_000_000_000  # 2026-01-01T00:00:00Z
_FRAME_PERIOD_NS = 33_333_333


def _add_eof_to_manifest(manifest_uri: str, *, reorder: bool = False) -> dict[str, list[int]]:
    """Fill ``eof_ns`` on every manifest row, as an index extension would, and return it per source.

    Each frame's end-of-frame time follows its presentation rank, not its decode
    position, so on a reordered stream the stored column is out of order and only a
    reader that applies presentation order sees a clean clock.

    ``reorder`` rewrites the composition offsets so decode samples 0, 1, 2 present as
    1, 2, 0 -- the shape B-frames give a stream. Done on the row rather than with a
    B-frame clip, because the checked-in one carries a single constant offset that its
    edit list cancels, which leaves presentation order equal to decode order. The video
    is never opened for an indexed stream, so the row is all that has to say it.
    """
    rows = lance.dataset(manifest_uri).to_table().to_pylist()
    eof_by_source: dict[str, list[int]] = {}
    for row in rows:
        frames = row["frames"]
        if reorder:
            ticks_per_frame = frames[1]["dts_ticks"] - frames[0]["dts_ticks"]
            ranks = [2, 0, 1, *range(3, len(frames))]
            for sample, (frame, rank) in enumerate(zip(frames, ranks, strict=True)):
                frame["composition_offset_ticks"] = (rank - sample + 2) * ticks_per_frame
        pts_ticks = np.array(
            [frame["dts_ticks"] + (frame["composition_offset_ticks"] or 0) for frame in frames], dtype=np.int64
        )
        presentation_rank = np.argsort(np.argsort(pts_ticks, kind="stable"), kind="stable")
        for frame, rank in zip(frames, presentation_rank, strict=True):
            frame["eof_ns"] = _SENSOR_EPOCH_NS + int(rank) * _FRAME_PERIOD_NS
        eof_by_source[str(row["media_location"])] = [frame["eof_ns"] for frame in frames]
    lance.write_dataset(pa.Table.from_pylist(rows, schema=VIDEO_MANIFEST_SCHEMA), manifest_uri, mode="overwrite")
    return eof_by_source


def _stream_rows(record: dict[str, object]) -> dict[str, dict[str, object]]:
    """Return each stream row, keyed by its source."""
    return {str(row["source"]): row for row in _rows(record)[store_schema.STREAM_DATASET]}


def _ordering_status(record: dict[str, object], source: str) -> object:
    """Return the stored ordering verdict for *source*."""
    stream_id = _stream_rows(record)[source]["stream_id"]
    return next(
        row["check_status"]
        for row in _rows(record)[store_schema.EVALUATION_DATASET]
        if row["stream_id"] == stream_id and row["metric_name"] == NAME_ORDERING
    )


@pytest.fixture(autouse=True)
def _forget_opened_manifests() -> Iterator[None]:
    """Clear the per-process manifest cache, since several tests here open different manifests."""
    processing._mp4_lookup_index_open.cache_clear()
    yield
    processing._mp4_lookup_index_open.cache_clear()


def test_a_session_lists_its_streams_in_discovery_order(session: pathlib.Path) -> None:
    """Listing is the first half of the task, and it is ordinary discovery."""
    assert processing.discover_session(str(session), config=_config(session)) == [
        str(session / "front.mp4"),
        str(session / "rear.mp4"),
    ]


def test_listing_honours_the_per_session_stream_cap(session: pathlib.Path) -> None:
    """``limit`` caps streams within a session; it never drops a session."""
    config = ResolvedDataIntegrityConfig.model_validate(
        {
            "schema_version": 1,
            "kind": "data-integrity",
            "input": {"sessions": [str(session)], "limit": 1},
            "output": {"store_root": str(session.parent / "di-store")},
        }
    )

    assert processing.discover_session(str(session), config=config) == [str(session / "front.mp4")]


def test_listing_honours_the_configured_stream_globs(session: pathlib.Path) -> None:
    """A session prefix holding copies of its cameras measures only the ones the config names."""
    (session / "recorder-00").mkdir()
    (session / "recorder-00" / "front.mp4").write_bytes(b"")
    config = ResolvedDataIntegrityConfig.model_validate(
        {
            "schema_version": 1,
            "kind": "data-integrity",
            "input": {"sessions": [str(session)], "stream_globs": ["recorder-*/*"]},
            "output": {"store_root": str(session.parent / "di-store")},
        }
    )

    assert processing.discover_session(str(session), config=config) == [str(session / "recorder-00" / "front.mp4")]


def test_no_configured_manifest_means_no_lookup(session: pathlib.Path) -> None:
    """The index is opt-in: an unconfigured run must not reach for a dataset at all."""
    assert processing.open_mp4_index(_config(session)) is None


def test_a_manifest_that_cannot_be_read_fails_rather_than_reading_the_videos(session: pathlib.Path) -> None:
    """Falling back would measure these sessions differently from the rest of the run, silently."""
    manifest = session / "not-a-manifest.lance"
    # A real Lance dataset that is not a manifest, so opening it raises. A URI with no
    # dataset at all returns None instead, and is the next test.
    lance.write_dataset(pa.table({"unrelated": [1]}), manifest)

    with pytest.raises(ValueError, match="could not be read") as failure:
        processing.open_mp4_index(_config(session, mp4_index_uri=str(manifest)))

    assert str(manifest) in str(failure.value)


def test_a_manifest_that_does_not_exist_fails_rather_than_counting_as_unconfigured(session: pathlib.Path) -> None:
    """``Mp4IndexLookup.open`` reads absence as "not built yet"; a configured run reads it as a typo."""
    manifest = session / "absent.lance"

    with pytest.raises(ValueError, match="holds no mp4-index manifest") as failure:
        processing.open_mp4_index(_config(session, mp4_index_uri=str(manifest)))

    assert str(manifest) in str(failure.value)


def test_a_covered_stream_is_measured_from_the_index_and_an_uncovered_one_from_its_video(
    session: pathlib.Path,
    h264_video: Callable[..., bytes],
) -> None:
    """A stream the manifest covers is measured from it; the rest are opened as usual.

    Plain mp4-index output has no EOF column, so with container PTS allowed both
    streams land on it -- one read from the row, one from the video.
    """
    (session / "side.mp4").write_bytes(h264_video())

    record = _check(session, mp4_index_uri=_manifest(session, session / "front.mp4"))

    assert _timestamp_sources(record) == {
        str(session / "front.mp4"): "container_pts",
        str(session / "side.mp4"): "container_pts",
        str(session / "rear.mp4"): None,  # unreadable, so measured on no clock at all
    }


def test_an_indexed_eof_clock_is_what_the_stored_row_measures(session: pathlib.Path) -> None:
    """End to end from a real manifest with ``eof_ns`` to the stream row, under the default config.

    A sidecar with other times sits beside the video, so the bounds prove the clock
    came from the index rather than from the deprecated sidecar.
    """
    front = session / "front.mp4"
    with pathlib.Path(f"{front}.timestamps.npy").open("wb") as stream:
        np.save(stream, np.arange(30, dtype=np.int64) * 7, allow_pickle=False)
    manifest_uri = _manifest(session, front)
    eof_ns = _add_eof_to_manifest(manifest_uri)[normalize_location(str(front))]

    record = _check(session, mp4_index_uri=manifest_uri, alignment_fields=None)

    row = _stream_rows(record)[str(front)]
    assert row["error"] is None
    assert row["timestamp_source"] == "sensor"
    assert row["num_samples"] == len(eof_ns)
    assert row["start_ns"] == _SENSOR_EPOCH_NS
    assert row["end_ns"] == max(eof_ns)
    assert _ordering_status(record, str(front)) == CheckStatus.PASS.value


def test_an_indexed_eof_clock_on_a_reordered_stream_is_read_in_presentation_order(session: pathlib.Path) -> None:
    """B-frames store samples out of display order, and the EOF column with them.

    Read in decode order, this clock would step backwards and fail ordering; read in
    presentation order, as the frames are shown, it is the clean clock it describes.
    """
    clip = session / "front.mp4"
    manifest_uri = _manifest(session, clip)
    eof_ns = _add_eof_to_manifest(manifest_uri, reorder=True)[normalize_location(str(clip))]
    assert np.any(np.diff(eof_ns) < 0), "the fixture must store EOF out of order, or this proves nothing"

    record = _check(session, mp4_index_uri=manifest_uri, alignment_fields=None)

    row = _stream_rows(record)[str(clip)]
    assert row["timestamp_source"] == "sensor"
    assert row["start_ns"] == _SENSOR_EPOCH_NS
    assert row["end_ns"] == max(eof_ns)
    assert _ordering_status(record, str(clip)) == CheckStatus.PASS.value


def test_by_default_a_stream_without_a_sensor_clock_is_recorded_as_an_error(
    session: pathlib.Path,
    h264_video: Callable[..., bytes],
) -> None:
    """The default asks for EOF; neither a plain index row nor a bare video has one.

    Each becomes that stream's error row, and the session is still measured and written.
    """
    (session / "side.mp4").write_bytes(h264_video())

    record = _check(session, mp4_index_uri=_manifest(session, session / "front.mp4"), alignment_fields=None)

    errors = {str(row["source"]): row["error"] for row in _rows(record)[store_schema.STREAM_DATASET]}
    assert "no complete camera timestamp column" in str(errors[str(session / "front.mp4")])
    assert "no requested camera timestamps" in str(errors[str(session / "side.mp4")])


def test_a_stale_index_row_reads_the_video_it_describes_rather_than_trusting_it(
    session: pathlib.Path,
    h264_video: Callable[..., bytes],
) -> None:
    """A partial or stale index is a miss, not a misconfiguration: this one video is opened."""
    manifest_uri = _manifest(session, session / "front.mp4")
    # Re-encoded longer, so the indexed size no longer matches the bytes there.
    (session / "front.mp4").write_bytes(h264_video(frames=45))
    warnings: list[str] = []
    sink_id = logger.add(warnings.append, level="WARNING")

    try:
        record = _check(session, mp4_index_uri=manifest_uri)
    finally:
        logger.remove(sink_id)

    assert _timestamp_sources(record)[str(session / "front.mp4")] == "container_pts"
    # The clock above does not say why the video was read: any index miss reads it too.
    # The warning is what pins that staleness was the reason.
    assert any("the index is stale for this video" in warning for warning in warnings)


def test_a_session_that_cannot_be_listed_is_reported_rather_than_raised(tmp_path: pathlib.Path) -> None:
    """Raising would abort the dataset and discard every session already measured.

    Ray's ``max_errored_blocks`` is 0 by default and a block holds several sessions, so
    one stale path in a long input would cost the whole run its commit.
    """
    missing = tmp_path / "absent"

    record = processing.check_session(
        {"session_path": str(missing)},
        config=_config(missing),
        run_id=RUN_ID,
        created_at=CREATED_AT,
    )

    assert "absent" in str(record["listing_error"])
    assert record["streams"] == 0
    # One row, and only one: nothing under the session was measured, so it has no
    # measurements and no verdicts -- the shape an errored stream has.
    rows = _rows(record)
    (session_row,) = rows[store_schema.SESSION_DATASET]
    assert "absent" in str(session_row["error"])
    assert session_row["num_streams"] == 0
    assert set(rows) == {store_schema.SESSION_DATASET}


def test_an_unreachable_stream_leaves_no_row_and_does_not_cost_its_siblings(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
    unreachable_video: Callable[[pathlib.Path], pathlib.Path],
) -> None:
    """Counted, never recorded: a row here would supersede a healthy earlier measurement.

    Readers resolve to the newest row per ``stream_id``, so persisting "we could not
    reach this" would read back later as "this stream is broken".

    The failure is a real unreadable file rather than an injected exception, because the
    thing most worth pinning is the one line that asks for this treatment: with
    ``raise_infrastructure_errors`` dropped from the call below, the real runner hands
    back an error row and every assertion here fails.
    """
    path = tmp_path / "clips" / "locked"
    path.mkdir(parents=True)
    (path / "front.mp4").write_bytes(h264_video())
    unreachable_video(path / "rear.mp4")

    record = _check(path)
    stream_rows = _rows(record)[store_schema.STREAM_DATASET]

    assert record["unreachable"] == 1
    assert record["unreadable"] == 0
    assert record["streams"] == 2, "the count of streams found is unchanged by one being unreachable"
    assert [row["source"] for row in stream_rows] == [str(path / "front.mp4")]


def _check_and_commit(
    session: pathlib.Path,
    *,
    created_at: datetime.datetime,
    thresholds: dict[str, Any] | None = None,
    mp4_index_uri: str,
) -> dict[str, object]:
    """Measure one session and commit its rows the way the pipeline driver does.

    The driver's half of a run without Ray: append every dataset the worker's record
    carries, then the ``run.lance`` row that makes them visible. Each call is its own
    run, so a test can stack runs and read back what the store makes of them.
    """
    config = _config(session, thresholds=thresholds, mp4_index_uri=mp4_index_uri)
    run_id = store.new_run_id()
    record = processing.check_session(
        {"session_path": str(session)}, config=config, run_id=run_id, created_at=created_at
    )
    root = str(config.output.store_root)
    for dataset, rows in _rows(record).items():
        store.append_rows(rows, store.join(root, dataset), processing.DATASET_SCHEMAS[dataset], None)
    store.commit_run(
        root,
        run_id=run_id,
        created_at=created_at,
        tool="data-integrity",
        session_path=None,
        thresholds=config.checks.thresholds.to_thresholds(),
        num_streams=int(record["streams"]) - int(record["unreachable"]),
        storage_options=None,
    )
    return record


@pytest.fixture
def offset_rig(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
    sensor_timeline: Callable[..., pathlib.Path],
) -> tuple[pathlib.Path, pathlib.Path]:
    """Three cameras that each look healthy, one of them recording two seconds late.

    Every stream passes on its own; only the session-grain checks see the offset, and
    only when the late camera is among the streams they are judged over.
    """
    rig = tmp_path / "rig"
    rig.mkdir()
    for name, start_ns in (
        ("a.mp4", _SENSOR_EPOCH_NS),
        ("b.mp4", _SENSOR_EPOCH_NS),
        ("c.mp4", _SENSOR_EPOCH_NS + 2_000_000_000),
    ):
        (rig / name).write_bytes(h264_video(frames=30))
        manifest = sensor_timeline(rig / name, frames=30, start_ns=start_ns)
    return rig, manifest


def _deny_camera(monkeypatch: pytest.MonkeyPatch, video: pathlib.Path) -> None:
    """Inject an infrastructure failure at lookup; indexed checks do not open media."""
    original = Mp4IndexLookup.index_for

    def _index_for(lookup: Mp4IndexLookup, source: str, *, media_size_bytes: int | None = None) -> object:
        if source == str(video):
            message = f"access denied: {source}"
            raise PermissionError(message)
        return original(lookup, source, media_size_bytes=media_size_bytes)

    monkeypatch.setattr(Mp4IndexLookup, "index_for", _index_for)


def test_a_partial_rerun_cannot_turn_a_failed_session_into_a_pass(
    offset_rig: tuple[pathlib.Path, pathlib.Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reproduced case, through the real session metrics and the real Lance store.

    The complete run fails the session on the late camera. The rerun cannot reach it,
    so its own cross-camera verdicts -- over the two aligned cameras -- pass and become
    the newest. The session must still not read PASS, and the rerun's good per-stream
    work must still be committed.
    """
    rig, manifest = offset_rig
    root = str(rig.parent / "di-store")
    first = _check_and_commit(rig, mp4_index_uri=str(manifest), created_at=_EARLIER)
    assert first["failed_session_metrics"] > 0
    assert store.session_rollup(root)["status"] == "FAIL"

    with monkeypatch.context() as denied:
        _deny_camera(denied, rig / "c.mp4")
        rerun = _check_and_commit(rig, mp4_index_uri=str(manifest), created_at=_LATER)

    assert rerun["unreachable"] == 1
    assert rerun["failed_session_metrics"] == 0, "judged over a and b alone, the session looks aligned"
    (session_row,) = store.read_sessions(root)
    assert (session_row["num_streams"], session_row["num_unreachable"]) == (3, 1)
    sessions = store.session_rollup(root)["sessions"]
    assert [entry["status"] for entry in sessions] == ["ERROR"]
    rerun_sources = {row["source"] for row in store.read_streams(root) if row["run_id"] == session_row["run_id"]}
    assert rerun_sources == {str(rig / "a.mp4"), str(rig / "b.mp4")}


def test_a_complete_run_after_a_partial_one_restores_the_verdict(
    offset_rig: tuple[pathlib.Path, pathlib.Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that reaches every camera again decides the session, and no run's history is lost."""
    rig, manifest = offset_rig
    root = str(rig.parent / "di-store")
    _check_and_commit(rig, mp4_index_uri=str(manifest), created_at=_EARLIER)
    with monkeypatch.context() as denied:
        _deny_camera(denied, rig / "c.mp4")
        _check_and_commit(rig, mp4_index_uri=str(manifest), created_at=_LATER)
    assert store.session_rollup(root)["status"] == "ERROR"

    _check_and_commit(rig, mp4_index_uri=str(manifest), created_at=_LATEST)

    assert store.session_rollup(root)["status"] == "FAIL"
    history = store.read_sessions(root, latest=False)
    assert sorted(row["num_unreachable"] for row in history) == [0, 0, 1]


def test_a_complete_run_under_another_policy_cannot_vouch_for_a_partial_runs_verdicts(
    offset_rig: tuple[pathlib.Path, pathlib.Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session's verdicts must come from the run that wrote its newest row.

    The partial rerun's passing verdicts are the newest under the default policy. A
    later complete run judged under a different policy writes the newest session row
    but no default-policy verdicts. Asked about the default policy, the rollup must not
    pair that complete row with the partial run's verdicts and call the session PASS.
    """
    rig, manifest = offset_rig
    root = str(rig.parent / "di-store")
    _check_and_commit(rig, mp4_index_uri=str(manifest), created_at=_EARLIER)
    with monkeypatch.context() as denied:
        _deny_camera(denied, rig / "c.mp4")
        _check_and_commit(rig, mp4_index_uri=str(manifest), created_at=_LATER)
    other_policy = {"max_gaps": 99}
    _check_and_commit(rig, mp4_index_uri=str(manifest), created_at=_LATEST, thresholds=other_policy)

    default = store.policy_id(_config(rig).checks.thresholds.to_thresholds())
    other = store.policy_id(_config(rig, thresholds=other_policy).checks.thresholds.to_thresholds())
    assert [entry["status"] for entry in store.session_rollup(root, policy=default)["sessions"]] == ["ERROR"]
    assert [entry["status"] for entry in store.session_rollup(root, policy=other)["sessions"]] == ["FAIL"]


def test_every_record_carries_every_column(session: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """One Arrow type per column across blocks, which is why the listing error is "" and not None.

    The driver's ``iter_batches`` concatenates blocks, and a column that is null in one
    and a string in another has no common type to concatenate under.
    """
    measured = _check(session)
    unlistable = processing.check_session(
        {"session_path": str(tmp_path / "absent")},
        config=_config(tmp_path / "absent"),
        run_id=RUN_ID,
        created_at=CREATED_AT,
    )

    assert set(measured) == set(unlistable)
    assert {type(value) for value in (measured["listing_error"], unlistable["listing_error"])} == {str}
    assert measured["listing_error"] == ""


def test_a_session_yields_the_rows_of_every_stream_it_holds(session: pathlib.Path) -> None:
    """One task covers the whole session: a row per stream, plus the readable one's metrics."""
    record = _check(session)
    rows = _rows(record)

    assert record["streams"] == 2
    assert len(rows[store_schema.STREAM_DATASET]) == 2
    assert len(rows[store_schema.EVALUATION_DATASET]) == NUM_METRICS
    assert sum(len(rows[store_schema.metric_dataset_path(spec.name)]) for spec in INSTRUMENTS) == NUM_METRICS


def test_a_failing_session_metric_is_counted_in_the_record(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
    sensor_timeline: Callable[..., pathlib.Path],
) -> None:
    """The counter the driver sums into the run summary, which no store row states.

    Two sensors that stopped a second apart, under a policy allowing no non-overlap at
    all: every stream passes on its own, so the session's own verdict is the only thing
    that fails, and ``failed_metrics`` beside it must not move. Sensor timelines make
    the session measurable in the first place.
    """
    ragged = tmp_path / "ragged"
    ragged.mkdir()
    for name, frames in (("front.mp4", 30), ("rear.mp4", 60)):
        (ragged / name).write_bytes(h264_video(frames=frames))
        manifest = sensor_timeline(ragged / name, frames=frames)

    record = _check(ragged, mp4_index_uri=str(manifest), thresholds={"max_non_overlap_percent": 0.0})
    verdicts = {
        row["metric_name"]: row["check_status"] for row in _rows(record)[store_schema.SESSION_EVALUATION_DATASET]
    }

    assert record["failed_metrics"] == 0
    assert record["failed_session_metrics"] == 1
    assert verdicts["multi_sensor_overlap"] == CheckStatus.FAIL.value


def test_an_unreadable_stream_does_not_cost_its_siblings(session: pathlib.Path) -> None:
    """One broken video must not take out the rest of its session, or the run."""
    record = _check(session)
    stream_rows = _rows(record)[store_schema.STREAM_DATASET]

    assert record["unreadable"] == 1
    assert record["failed_metrics"] == 0
    errored = [row for row in stream_rows if row["error"] is not None]
    assert [row["source"] for row in errored] == [str(session / "rear.mp4")]
    # The readable sibling is still measured, and only it is judged: an errored stream
    # was never measured, so there is nothing to judge.
    assert len(stream_rows) == 2


def test_every_row_carries_the_runs_identity_and_its_session(session: pathlib.Path) -> None:
    """One run id across the invocation, one session path on every row of it."""
    for dataset_rows in _rows(_check(session)).values():
        for row in dataset_rows:
            assert row["run_id"] == RUN_ID
            assert row["session_path"] == str(session)
            assert row["created_at"] == CREATED_AT


def test_an_empty_session_produces_a_result_but_no_stream_rows(tmp_path: pathlib.Path) -> None:
    """An empty session is a fact about the input, not a failure of the task."""
    empty = tmp_path / "empty"
    empty.mkdir()
    record = _check(empty)
    rows = _rows(record)

    assert record["streams"] == 0
    assert record["unreadable"] == 0
    assert store_schema.STREAM_DATASET not in rows
    # It was listed, though, and that is worth a row: a session found empty and a
    # session never reached are different findings about the input.
    (session_row,) = rows[store_schema.SESSION_DATASET]
    assert session_row["error"] is None
    assert session_row["num_streams"] == 0
    # No stream contributed bounds, so the session was measured on no clock at all --
    # which is a null column rather than a named one.
    assert session_row["timestamp_source"] is None
    # Its own metrics are still considered and judged, and come back SKIPPED. Nothing
    # here forks on `num_streams == 0` -- both shapes go through `session_metrics`,
    # which is what stops the pipeline and ``di`` from judging a session two ways.
    # The rows are the "never ran" shape rather than a measured-but-undefined one:
    # with no clock to compare on, the metrics were never reachable.
    assert record["failed_session_metrics"] == 0
    for spec in SESSION_INSTRUMENTS:
        (measurement,) = rows[store_schema.session_metric_dataset_path(spec.name)]
        assert measurement["is_defined"] is None
        assert measurement["num_sensors"] is None
    assert [row["check_status"] for row in rows[store_schema.SESSION_EVALUATION_DATASET]] == [
        CheckStatus.SKIPPED.value
    ] * len(SESSION_INSTRUMENTS)


def test_the_configured_attempt_budget_reaches_every_stream(
    session: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``execution.stream_attempts`` is the retry knob, and it is not decorative."""
    seen: list[object] = []

    def _record(source: str, **kwargs: object) -> StreamResult:
        seen.append(kwargs["max_attempts"])
        return StreamResult(
            source=source,
            codec_name=None,
            has_bframes=None,
            num_samples=None,
            start_ns=None,
            end_ns=None,
            metrics=[],
            error="stubbed",
        )

    monkeypatch.setattr(processing, "run_one_stream", _record)

    _check(session, stream_attempts=5)

    assert seen == [5, 5]


def test_the_payload_names_only_datasets_the_driver_can_write(session: pathlib.Path) -> None:
    """The payload's keys and the schemas they are written under are one contract."""
    assert set(_rows(_check(session))) <= set(processing.DATASET_SCHEMAS)
