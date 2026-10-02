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
"""Tests for the mp4-index Ray Data driver, including the caching guard end to end.

``process_one_video`` is tested directly (no Ray) in ``test_processing.py``. ``run_config`` is
tested here: the reconciliation/skip-known-failure paths without starting Ray at all (pre-seeded
tables), and the real dispatch/publish/error-log paths through a real small local Ray Data run --
covering discovery, container_kind-declared dispatch, and (the real proof the caching guard works
end to end, not just at the ``process_one_video`` level) a video whose final sidecars are already
present but whose manifest row is not, run against a deliberately corrupted media file: if the
run still succeeds, moov/moof were never touched.

``input.container_kind`` is a whole-run declaration, not a per-video fact, so a single
``run_config`` call here only ever targets one container shape -- there is no "mixed corpus in
one run" scenario to test.
"""

import pathlib
from datetime import UTC, datetime
from typing import Any, Self

import lance  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pytest
import ray

from cosmos_curator.core.utils import environment
from cosmos_curator.next.recipes.mp4_index import config, pipeline
from cosmos_curator.next.recipes.mp4_index.contracts import (
    ERROR_SCHEMA,
    RECORD_SCHEMA_VERSION,
    VIDEO_IDENTITY_SCHEMA,
    VIDEO_METADATA_SCHEMA,
    derive_session_and_camera,
    make_session_camera_id,
    make_video_id,
    read_arrow_sidecar,
    read_bytes_sidecar,
    read_frame_index,
    read_video_metadata,
    write_arrow_sidecar,
    write_bytes_sidecar,
    write_frame_index,
)
from cosmos_curator.next.utils import lance_fragment_recovery
from cosmos_curator.next.utils.lance_utils import LANCE_DATA_STORAGE_VERSION
from tests.cosmos_curator.core.sensors.mp4 import _box_fixtures as bf


def _write_synthetic_nonfragmented_mp4(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    moov_box = bf.synthetic_moov(frame_count=1, keyframe_index=0, ctts_offset=None)
    path.write_bytes(bf.pack_box("ftyp", b"isom") + moov_box + bf.pack_box("mdat", b"data"))


def _write_synthetic_fragmented_mp4(path: pathlib.Path) -> None:
    """Write a real, byte-accurate fragmented file -- fragment_table.py's own parser must accept it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    moov_box = bf.synthetic_fragmented_moov()
    buf = bf.pack_box("ftyp", b"isom") + moov_box
    for i in range(3):
        buf += bf.pack_moof(bf.real_traf(decode_time=i, data_offset=32, sample_size=100, is_keyframe=i == 0))
        buf += bf.pack_box("mdat", b"video-bytes" * 20)
    path.write_bytes(buf)


def _publish_manifest_row(manifest_row: dict[str, object], manifest_lance_uri: str) -> None:
    """Stage and commit one manifest row directly, bypassing Ray entirely -- pre-seeds a table."""
    table = pa.Table.from_pylist([manifest_row], schema=VIDEO_IDENTITY_SCHEMA)
    lance_fragment_recovery.open_or_create_table(
        uri=manifest_lance_uri,
        storage_profile="default",
        schema=VIDEO_IDENTITY_SCHEMA,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
        kind="mp4-index",
    )
    candidate = lance_fragment_recovery.write_row_fragment(
        table,
        uri=manifest_lance_uri,
        storage_profile="default",
        schema=VIDEO_IDENTITY_SCHEMA,
        id_column="video_id",
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
    )
    assert candidate is not None
    lance_fragment_recovery.append_row_fragment(
        candidate,
        uri=manifest_lance_uri,
        storage_profile="default",
        schema=VIDEO_IDENTITY_SCHEMA,
        id_column="video_id",
        kind="mp4-index",
        operation="append-video",
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
        attempts=1,
    )


def _publish_error_row(errors_lance_uri: str, *, video_id: str, error_type: str = "BoxNotFoundError") -> None:
    """Stage one ERROR_SCHEMA row directly -- pre-seeds a known-failure table for a test."""
    table = pa.Table.from_pylist(
        [
            {
                "video_id": video_id,
                "media_location": "s3://unused/unused.mp4",
                "error_type": error_type,
                "error_message": "pre-seeded for a test",
                "attempted_at": datetime.now(UTC),
            }
        ],
        schema=ERROR_SCHEMA,
    )
    lance.write_dataset(table, errors_lance_uri, mode="create", data_storage_version=LANCE_DATA_STORAGE_VERSION)


def _resolved_config(
    tmp_path: pathlib.Path, *, container_kind: str = "nonfragmented", **execution_overrides: object
) -> config.ResolvedMp4IndexConfig:
    return config.ResolvedMp4IndexConfig.model_validate(
        {
            "schema_version": 1,
            "kind": "mp4-index",
            "input": {
                "path_prefix": str(tmp_path / "input"),
                "globs": ["*/unit-*/*.mp4"],
                "container_kind": container_kind,
            },
            "output": {"path": str(tmp_path / "output")},
            "execution": {"session_id_path_depth": 2, **execution_overrides},
        }
    )


def test_run_config_reports_no_sources_without_starting_ray(tmp_path: pathlib.Path) -> None:
    """An empty discovery result returns immediately -- no Ray cluster needed for a no-op run."""
    (tmp_path / "input" / "sess-a" / "unit-00").mkdir(parents=True)
    cfg = _resolved_config(tmp_path)

    summary = pipeline.run_config(cfg)

    assert summary == {
        "videos_discovered": 0,
        "videos_already_extracted": 0,
        "videos_skipped_known_failure": 0,
        "videos_extracted": 0,
        "videos_failed": 0,
        "manifest_lance_uri": str(tmp_path / "output" / "manifest.lance"),
        "errors_lance_uri": str(tmp_path / "output" / "errors.lance"),
        "indices_built": [],
        "error_indices_built": [],
    }


def test_run_config_skips_already_committed_videos_without_starting_ray(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A video already in manifest.lance is excluded before Ray ever starts -- the actual work-skip."""
    media_path = tmp_path / "input" / "sess-a" / "unit-00" / "camera.mp4"
    _write_synthetic_nonfragmented_mp4(media_path)

    manifest_lance_uri = str(tmp_path / "output" / "manifest.lance")
    video_id = make_video_id(str(media_path))
    source_session_id, camera_key = derive_session_and_camera(str(media_path), session_id_path_depth=2)
    cfg = _resolved_config(tmp_path)
    _publish_manifest_row(
        {
            "video_id": video_id,
            "source_session_id": source_session_id,
            "camera_key": camera_key,
            "session_camera_id": make_session_camera_id(source_session_id, camera_key),
            "media_location": str(media_path),
            "media_size_bytes": media_path.stat().st_size,
            "frame_index_location": str(tmp_path / "output" / "frames" / "unused.arrow"),
            "metadata_location": str(tmp_path / "output" / "metadata" / "unused.arrow"),
            "moov_location": str(tmp_path / "output" / "moov" / "unused.moov"),
            "moof_table_location": None,
            "container_kind": "nonfragmented",
            "record_schema_version": RECORD_SCHEMA_VERSION,
            "extraction_contract_id": cfg.extraction_contract_id,
        },
        manifest_lance_uri,
    )

    def _fail_if_called() -> None:
        msg = "Ray must not start when every candidate is already committed"
        raise AssertionError(msg)

    monkeypatch.setattr(pipeline, "ensure_ray_initialized", _fail_if_called)

    summary = pipeline.run_config(cfg)

    assert summary["videos_discovered"] == 1
    assert summary["videos_already_extracted"] == 1
    assert summary["videos_extracted"] == 0
    assert summary["videos_failed"] == 0
    assert lance.dataset(manifest_lance_uri).count_rows() == 1


def test_run_config_skips_known_failures_by_default_without_starting_ray(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A video with a prior errors.lance row is excluded before Ray starts -- skip_known_failures defaults true."""
    media_path = tmp_path / "input" / "sess-a" / "unit-00" / "camera.mp4"
    _write_synthetic_nonfragmented_mp4(media_path)

    errors_lance_uri = str(tmp_path / "output" / "errors.lance")
    _publish_error_row(errors_lance_uri, video_id=make_video_id(str(media_path)))

    def _fail_if_called() -> None:
        msg = "Ray must not start when every remaining candidate has a known failure"
        raise AssertionError(msg)

    monkeypatch.setattr(pipeline, "ensure_ray_initialized", _fail_if_called)

    cfg = _resolved_config(tmp_path)
    summary = pipeline.run_config(cfg)

    assert summary["videos_discovered"] == 1
    assert summary["videos_skipped_known_failure"] == 1
    assert summary["videos_extracted"] == 0
    assert summary["videos_failed"] == 0
    assert lance.dataset(errors_lance_uri).count_rows() == 1


def test_discover_rows_finds_a_video_under_a_file_uri_path_prefix(tmp_path: pathlib.Path) -> None:
    """A file:// path_prefix discovers the same video a bare local path would, not zero silently.

    Without config-time normalization, is_remote_path("file://...") is False, so discovery
    runs pathlib.Path("file://...").rglob("*") -- a literal, nonexistent relative path that
    silently yields nothing, no error, just a run that discovers (and extracts) 0 videos.
    """
    media_path = tmp_path / "input" / "sess-a" / "unit-00" / "camera.mp4"
    _write_synthetic_nonfragmented_mp4(media_path)

    input_cfg = config.Mp4IndexInputConfig.model_validate(
        {
            "path_prefix": f"file://{tmp_path / 'input'}",
            "globs": ["*/unit-*/*.mp4"],
            "container_kind": "nonfragmented",
        }
    )
    assert input_cfg.path_prefix == str(tmp_path / "input")  # normalized away from file:// at config resolution

    rows = pipeline.discover_rows(input_cfg, storage_profile="default", log_prefix="mp4-index")

    assert len(rows) == 1
    assert rows[0]["media_location"] == str(media_path)


def test_run_config_raises_on_two_video_ids_sharing_one_session_camera_id(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two distinct video_ids deriving the same session_camera_id fail the whole run before Ray starts.

    session_id_path_depth=2 ignores any directory above it -- "batch1/sess-a/unit-00/camera.mp4"
    and "batch2/sess-a/unit-00/camera.mp4" derive identical (source_session_id, camera_key), so
    they'd otherwise publish two "success" manifest rows pointing at one frame_index/metadata
    sidecar pair, second write silently winning. Caught upfront instead, since both video_ids are
    fully known on the driver from path strings alone -- no I/O, no Ray needed to detect it.
    """
    first_path = tmp_path / "input" / "batch1" / "sess-a" / "unit-00" / "camera.mp4"
    second_path = tmp_path / "input" / "batch2" / "sess-a" / "unit-00" / "camera.mp4"
    _write_synthetic_nonfragmented_mp4(first_path)
    _write_synthetic_nonfragmented_mp4(second_path)

    def _fail_if_called() -> None:
        msg = "Ray must not start when a session_camera_id collision is detected"
        raise AssertionError(msg)

    monkeypatch.setattr(pipeline, "ensure_ray_initialized", _fail_if_called)

    cfg = config.ResolvedMp4IndexConfig.model_validate(
        {
            "schema_version": 1,
            "kind": "mp4-index",
            "input": {
                "path_prefix": str(tmp_path / "input"),
                "globs": ["*/*/unit-*/*.mp4"],
                "container_kind": "nonfragmented",
            },
            "output": {"path": str(tmp_path / "output")},
            "execution": {"session_id_path_depth": 2},
        }
    )

    with pytest.raises(pipeline.SessionCameraIdCollisionError, match="sess-a/unit-00/camera"):
        pipeline.run_config(cfg)


@pytest.mark.integration
def test_run_config_raises_on_a_session_camera_id_collision_against_an_already_committed_video(
    tmp_path: pathlib.Path, curator_io_cluster: None
) -> None:
    """A new candidate deriving an already-committed session_camera_id under a different video_id is refused.

    Two runs over disjoint batch prefixes can derive the same (source_session_id, camera_key) --
    and therefore the same frame_index/metadata sidecar path -- for two genuinely different
    video_ids. The in-batch check alone misses this: the second run's own candidate set has no
    internal collision, only a collision against history.
    """
    del curator_io_cluster
    first_path = tmp_path / "input" / "batch1" / "sess-a" / "unit-00" / "camera.mp4"
    _write_synthetic_nonfragmented_mp4(first_path)
    first_cfg = config.ResolvedMp4IndexConfig.model_validate(
        {
            "schema_version": 1,
            "kind": "mp4-index",
            "input": {
                "path_prefix": str(tmp_path / "input" / "batch1"),
                "globs": ["*/unit-*/*.mp4"],
                "container_kind": "nonfragmented",
            },
            "output": {"path": str(tmp_path / "output")},
            "execution": {"session_id_path_depth": 2},
        }
    )
    first = pipeline.run_config(first_cfg)
    assert first["videos_extracted"] == 1

    second_path = tmp_path / "input" / "batch2" / "sess-a" / "unit-00" / "camera.mp4"
    _write_synthetic_nonfragmented_mp4(second_path)
    second_cfg = config.ResolvedMp4IndexConfig.model_validate(
        {
            "schema_version": 1,
            "kind": "mp4-index",
            "input": {
                "path_prefix": str(tmp_path / "input" / "batch2"),
                "globs": ["*/unit-*/*.mp4"],
                "container_kind": "nonfragmented",
            },
            "output": {"path": str(tmp_path / "output")},
            "execution": {"session_id_path_depth": 2},
        }
    )

    with pytest.raises(pipeline.SessionCameraIdCollisionError, match="sess-a/unit-00/camera"):
        pipeline.run_config(second_cfg)


@pytest.mark.integration
def test_run_config_end_to_end_over_a_real_local_ray_data_run_nonfragmented(
    tmp_path: pathlib.Path, curator_io_cluster: None
) -> None:
    """A real (local) Ray Data run declared non-fragmented: a moov sidecar, a frame index, and metadata.

    Depends on ``curator_io_cluster`` because ``run_config`` reserves ``curator_io`` unconditionally;
    without it, an integration test elsewhere in this directory that starts Ray first without that
    resource would leave this test's own dispatch unable to schedule (Ray stays up for the session).
    """
    del curator_io_cluster
    media_path = tmp_path / "input" / "sess-a" / "unit-00" / "camera.mp4"
    _write_synthetic_nonfragmented_mp4(media_path)

    cfg = _resolved_config(tmp_path, container_kind="nonfragmented")
    summary = pipeline.run_config(cfg)

    assert summary["videos_discovered"] == 1
    assert summary["videos_extracted"] == 1
    assert summary["videos_failed"] == 0
    assert summary["indices_built"] == ["video_id", "media_location", "session_camera_id"]
    assert lance.dataset(summary["manifest_lance_uri"]).count_rows() == 1
    assert lance.dataset(summary["errors_lance_uri"]).count_rows() == 0

    row = lance.dataset(summary["manifest_lance_uri"]).to_table().to_pylist()[0]
    assert row["video_id"] == make_video_id(str(media_path))
    assert row["container_kind"] == "nonfragmented"
    frame_table = read_frame_index(row["frame_index_location"])
    assert frame_table.num_rows == 1
    metadata_row = read_video_metadata(row["metadata_location"])
    assert set(VIDEO_METADATA_SCHEMA.names) <= set(metadata_row)
    assert read_bytes_sidecar(row["moov_location"])
    assert row["moof_table_location"] is None


@pytest.mark.integration
def test_run_config_end_to_end_over_a_real_local_ray_data_run_fragmented(
    tmp_path: pathlib.Path, curator_io_cluster: None
) -> None:
    """A real (local) Ray Data run declared fragmented: both a moov sidecar and a moof table are extracted.

    Depends on ``curator_io_cluster``; see the nonfragmented test above for why.
    """
    del curator_io_cluster
    media_path = tmp_path / "input" / "sess-a" / "unit-00" / "camera.mp4"
    _write_synthetic_fragmented_mp4(media_path)

    cfg = _resolved_config(tmp_path, container_kind="fragmented")
    summary = pipeline.run_config(cfg)

    assert summary["videos_discovered"] == 1
    assert summary["videos_extracted"] == 1
    assert summary["videos_failed"] == 0
    assert lance.dataset(summary["manifest_lance_uri"]).count_rows() == 1

    row = lance.dataset(summary["manifest_lance_uri"]).to_table().to_pylist()[0]
    assert row["video_id"] == make_video_id(str(media_path))
    assert row["container_kind"] == "fragmented"
    video_id = row["video_id"]
    assert row["moov_location"] == str(tmp_path / "output" / "moov" / f"{video_id}.moov")
    assert row["moof_table_location"] == str(tmp_path / "output" / "moof" / f"{video_id}.arrow")
    assert read_bytes_sidecar(row["moov_location"])
    table = read_arrow_sidecar(row["moof_table_location"])
    assert table.num_rows == 3


@pytest.mark.integration
def test_run_config_rerun_reconciles_and_extracts_nothing_new(tmp_path: pathlib.Path, curator_io_cluster: None) -> None:
    """A second run against the same config extracts nothing new -- the single most important correctness check.

    Depends on ``curator_io_cluster``; see the nonfragmented test above for why.
    """
    del curator_io_cluster
    media_path = tmp_path / "input" / "sess-a" / "unit-00" / "camera.mp4"
    _write_synthetic_nonfragmented_mp4(media_path)
    cfg = _resolved_config(tmp_path)

    first = pipeline.run_config(cfg)
    second = pipeline.run_config(cfg)

    assert first["videos_extracted"] == 1
    assert second["videos_extracted"] == 0
    assert second["videos_already_extracted"] == 1
    assert lance.dataset(first["manifest_lance_uri"]).count_rows() == 1


@pytest.mark.integration
def test_run_config_a_failing_video_lands_in_the_error_log(tmp_path: pathlib.Path, curator_io_cluster: None) -> None:
    """A video with no moov box fails cleanly through the real Ray path and is recorded in errors.lance.

    Depends on ``curator_io_cluster``; see the nonfragmented test above for why.
    """
    del curator_io_cluster
    media_path = tmp_path / "input" / "sess-a" / "unit-00" / "camera.mp4"
    media_path.parent.mkdir(parents=True)
    media_path.write_bytes(bf.pack_box("ftyp", b"isom") + bf.pack_box("mdat", b"data"))  # no moov

    cfg = _resolved_config(tmp_path)
    summary = pipeline.run_config(cfg)

    assert summary["videos_extracted"] == 0
    assert summary["videos_failed"] == 1
    assert lance.dataset(summary["manifest_lance_uri"]).count_rows() == 0

    errors = lance.dataset(summary["errors_lance_uri"]).to_table().to_pylist()
    assert len(errors) == 1
    assert errors[0]["error_type"] == "BoxNotFoundError"
    assert errors[0]["video_id"] == make_video_id(str(media_path))
    assert summary["error_indices_built"] == ["video_id", "error_type"]


@pytest.mark.integration
def test_run_config_reuses_already_extracted_sidecars_via_the_caching_guard(
    tmp_path: pathlib.Path, curator_io_cluster: None
) -> None:
    """A video whose final sidecars already exist, but whose manifest row does not, is never re-extracted.

    This is the real end-to-end proof the caching guard works through actual Ray dispatch, not
    just at the process_one_video level: the media file is deliberately corrupted (no real moov
    box) after its sidecars are pre-populated, so if the run tried to refetch moov it would fail
    with BoxNotFoundError. The run instead succeeds, proving moov/moof were never touched.

    Depends on ``curator_io_cluster``; see the nonfragmented test above for why.
    """
    del curator_io_cluster
    media_path = tmp_path / "input" / "sess-a" / "unit-00" / "camera.mp4"
    _write_synthetic_nonfragmented_mp4(media_path)

    cfg = _resolved_config(tmp_path)
    video_id = make_video_id(str(media_path))
    source_session_id, camera_key = derive_session_and_camera(str(media_path), session_id_path_depth=2)
    session_camera_id = make_session_camera_id(source_session_id, camera_key)

    moov_box = bf.synthetic_moov(frame_count=5, keyframe_index=0, ctts_offset=None)
    write_bytes_sidecar(moov_box, str(tmp_path / "output" / "moov" / f"{video_id}.moov"))
    write_frame_index([], str(tmp_path / "output" / "frames" / f"{session_camera_id}.arrow"))
    write_arrow_sidecar(
        [dict.fromkeys(VIDEO_METADATA_SCHEMA.names)],
        VIDEO_METADATA_SCHEMA,
        str(tmp_path / "output" / "metadata" / f"{session_camera_id}.arrow"),
    )

    # Corrupt the media file after its sidecars are pre-populated -- a real refetch would fail.
    media_path.write_bytes(bf.pack_box("ftyp", b"isom") + bf.pack_box("mdat", b"data"))

    summary = pipeline.run_config(cfg)

    assert summary["videos_discovered"] == 1
    assert summary["videos_extracted"] == 1
    assert summary["videos_failed"] == 0
    row = lance.dataset(summary["manifest_lance_uri"]).to_table().to_pylist()[0]
    assert row["video_id"] == video_id
    assert row["frame_index_location"] == str(tmp_path / "output" / "frames" / f"{session_camera_id}.arrow")
    assert row["metadata_location"] == str(tmp_path / "output" / "metadata" / f"{session_camera_id}.arrow")


def test_empty_summary_shape() -> None:
    """A no-op summary carries the discovered/already/skipped counts and zeroed extraction fields."""
    summary = pipeline._empty_summary(
        manifest_lance_uri="s3://x/manifest.lance",
        errors_lance_uri="s3://x/errors.lance",
        discovered=5,
        already=2,
        skipped=1,
    )

    assert summary == {
        "videos_discovered": 5,
        "videos_already_extracted": 2,
        "videos_skipped_known_failure": 1,
        "videos_extracted": 0,
        "videos_failed": 0,
        "manifest_lance_uri": "s3://x/manifest.lance",
        "errors_lance_uri": "s3://x/errors.lance",
        "indices_built": [],
        "error_indices_built": [],
    }


class _CapturingDataset:
    """A minimal fake Ray Dataset: records .map()'s own kwargs, chains every other call to itself.

    No real Ray cluster needed -- the same style video-transcode's own test_pipeline.py uses to
    assert on a dispatch call's kwargs without depending on cluster-resource state (which a real
    integration-marked Ray run would make this test sensitive to Ray already being initialized
    without curator_io by an earlier test in the same session).
    """

    def __init__(self) -> None:
        self.map_kwargs: dict[str, Any] = {}

    def map(self, *_args: Any, **kwargs: Any) -> Self:  # noqa: ANN401
        self.map_kwargs = kwargs
        return self

    def repartition(self, *_args: Any, **_kwargs: Any) -> Self:  # noqa: ANN401
        return self

    def map_batches(self, *_args: Any, **_kwargs: Any) -> Self:  # noqa: ANN401
        return self

    def iter_batches(self, *_args: Any, **_kwargs: Any) -> list[Any]:  # noqa: ANN401
        return []


def test_run_config_requests_one_curator_io_slot_per_actor(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every dispatch requests exactly one curator_io slot per actor task, not the whole node budget.

    The node-level budget (curator_io_resources(), e.g. 16.0) and the per-task reservation (1.0)
    use the same resource name but are not interchangeable -- this is what actually bounds
    concurrent actors by IO capacity rather than by CPU count (see
    docs/curator/design/ray-data.md's own "Large source downloads and curator_io" section).
    """
    media_path = tmp_path / "input" / "sess-a" / "unit-00" / "camera.mp4"
    _write_synthetic_nonfragmented_mp4(media_path)

    fake_dataset = _CapturingDataset()
    monkeypatch.setattr(pipeline, "ensure_ray_initialized", lambda **_kwargs: None)
    monkeypatch.setattr(ray.data, "from_items", lambda *_args, **_kwargs: fake_dataset)

    cfg = _resolved_config(tmp_path)
    pipeline.run_config(cfg)

    assert fake_dataset.map_kwargs["resources"] == {environment.CURATOR_IO_RESOURCE_NAME: 1.0}


@pytest.mark.integration
def test_run_config_partial_failure_in_one_batch_commits_successes_and_logs_the_rest(
    tmp_path: pathlib.Path, curator_io_cluster: None
) -> None:
    """One dispatch with a mix of outcomes: the good video lands in the manifest, the bad one in the error log."""
    del curator_io_cluster
    good_path = tmp_path / "input" / "sess-a" / "unit-00" / "good.mp4"
    bad_path = tmp_path / "input" / "sess-b" / "unit-00" / "bad.mp4"
    _write_synthetic_nonfragmented_mp4(good_path)
    bad_path.parent.mkdir(parents=True)
    bad_path.write_bytes(bf.pack_box("ftyp", b"isom") + bf.pack_box("mdat", b"data"))  # no moov

    cfg = _resolved_config(tmp_path)
    summary = pipeline.run_config(cfg)

    assert summary["videos_extracted"] == 1
    assert summary["videos_failed"] == 1

    manifest_rows = lance.dataset(summary["manifest_lance_uri"]).to_table().to_pylist()
    assert [row["video_id"] for row in manifest_rows] == [make_video_id(str(good_path))]

    error_rows = lance.dataset(summary["errors_lance_uri"]).to_table().to_pylist()
    assert len(error_rows) == 1
    assert error_rows[0]["video_id"] == make_video_id(str(bad_path))


@pytest.mark.integration
def test_run_config_all_rows_fail_leaves_the_manifest_empty_and_skips_its_index_rebuild(
    tmp_path: pathlib.Path, curator_io_cluster: None
) -> None:
    """extracted=0: the manifest index rebuild is skipped (nothing to index), but the error log still gets both."""
    del curator_io_cluster
    first_path = tmp_path / "input" / "sess-a" / "unit-00" / "bad1.mp4"
    second_path = tmp_path / "input" / "sess-b" / "unit-00" / "bad2.mp4"
    for path in (first_path, second_path):
        path.parent.mkdir(parents=True)
        path.write_bytes(bf.pack_box("ftyp", b"isom") + bf.pack_box("mdat", b"data"))  # no moov

    cfg = _resolved_config(tmp_path)
    summary = pipeline.run_config(cfg)

    assert summary["videos_extracted"] == 0
    assert summary["videos_failed"] == 2
    assert summary["indices_built"] == []
    assert summary["error_indices_built"] == ["video_id", "error_type"]
    assert lance.dataset(summary["manifest_lance_uri"]).count_rows() == 0
    assert lance.dataset(summary["errors_lance_uri"]).count_rows() == 2


@pytest.mark.integration
def test_run_config_skips_a_known_failure_and_still_dispatches_new_work(
    tmp_path: pathlib.Path, curator_io_cluster: None
) -> None:
    """A known-failure video is skipped by reconciliation while a new candidate in the same run still dispatches."""
    del curator_io_cluster
    known_bad_path = tmp_path / "input" / "sess-a" / "unit-00" / "bad.mp4"
    known_bad_path.parent.mkdir(parents=True)
    known_bad_path.write_bytes(bf.pack_box("ftyp", b"isom") + bf.pack_box("mdat", b"data"))
    _publish_error_row(str(tmp_path / "output" / "errors.lance"), video_id=make_video_id(str(known_bad_path)))

    new_path = tmp_path / "input" / "sess-b" / "unit-00" / "good.mp4"
    _write_synthetic_nonfragmented_mp4(new_path)

    cfg = _resolved_config(tmp_path)
    summary = pipeline.run_config(cfg)

    assert summary["videos_discovered"] == 2
    assert summary["videos_skipped_known_failure"] == 1
    assert summary["videos_extracted"] == 1

    manifest_rows = lance.dataset(summary["manifest_lance_uri"]).to_table().to_pylist()
    assert [row["video_id"] for row in manifest_rows] == [make_video_id(str(new_path))]


@pytest.mark.integration
def test_run_config_retries_a_file_not_found_known_failure_even_with_skip_known_failures_true(
    tmp_path: pathlib.Path, curator_io_cluster: None
) -> None:
    """A video whose only prior failure is FileNotFoundError (e.g. a since-fixed bucket) is dispatched again."""
    del curator_io_cluster
    media_path = tmp_path / "input" / "sess-a" / "unit-00" / "camera.mp4"
    _write_synthetic_nonfragmented_mp4(media_path)
    _publish_error_row(
        str(tmp_path / "output" / "errors.lance"),
        video_id=make_video_id(str(media_path)),
        error_type="FileNotFoundError",
    )

    cfg = _resolved_config(tmp_path)
    summary = pipeline.run_config(cfg)

    assert summary["videos_skipped_known_failure"] == 0
    assert summary["videos_extracted"] == 1
    manifest_rows = lance.dataset(summary["manifest_lance_uri"]).to_table().to_pylist()
    assert [row["video_id"] for row in manifest_rows] == [make_video_id(str(media_path))]


@pytest.mark.integration
def test_run_config_retries_a_known_failure_when_skip_known_failures_is_false(
    tmp_path: pathlib.Path, curator_io_cluster: None
) -> None:
    """skip_known_failures=False dispatches a previously-failed video again instead of reconciling it away."""
    del curator_io_cluster
    media_path = tmp_path / "input" / "sess-a" / "unit-00" / "camera.mp4"
    media_path.parent.mkdir(parents=True)
    media_path.write_bytes(bf.pack_box("ftyp", b"isom") + bf.pack_box("mdat", b"data"))  # no moov -- fails

    cfg = _resolved_config(tmp_path)
    first = pipeline.run_config(cfg)
    assert first["videos_failed"] == 1

    _write_synthetic_nonfragmented_mp4(media_path)  # fix the file so a retry actually succeeds
    retry_cfg = _resolved_config(tmp_path, skip_known_failures=False)
    second = pipeline.run_config(retry_cfg)

    assert second["videos_skipped_known_failure"] == 0
    assert second["videos_extracted"] == 1
    manifest_rows = lance.dataset(second["manifest_lance_uri"]).to_table().to_pylist()
    assert [row["video_id"] for row in manifest_rows] == [make_video_id(str(media_path))]
