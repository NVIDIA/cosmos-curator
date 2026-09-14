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

"""End-to-end integration test for robot-action-split.

Builds a minimal synthetic LeRobot/Mecka dataset on disk, runs the full
discovery → cut → Lance write pipeline, and asserts correct outputs at
each stage.

Dataset layout created by this test
-------------------------------------

  {tmp}/dataset/
    meta/
      info.json              fps=24, observation.images.main 64x64
      subtasks.parquet       0="pick up coffee pod"  1="open machine lid"
      tasks.parquet          0="make coffee"
      episodes/
        chunk-000/
          file-000.parquet   episode_index, episode_id, videos/*/from_timestamp
    data/
      chunk-000/
        file-000.parquet     200 frames: 100xsubtask0 + 100xsubtask1, with action data
    videos/
      observation.images.main/
        chunk-000/
          file-000.mp4       reuses tests/.../test_clip_10s.mp4 (24fps, 10s, 240 frames)

Span geometry
-------------
  - fps=24, frames 0-99 = subtask 0 (~4.2 s), frames 100-199 = subtask 1 (~4.2 s)
  - episode_from_timestamp=0.0 so abs_start/abs_end = frame_start/end directly
  - Both spans pass min_duration_s=4.0 and max_duration_s=20.0
"""

import json
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import lance
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from cosmos_curator.next.recipes.robot_action_split.config import ResolvedRobotActionSplitConfig
from cosmos_curator.next.recipes.robot_action_split.contracts import (
    CLIP_RECORD_SCHEMA_VERSION,
    MEDIA_CONTRACT_VERSION,
)
from cosmos_curator.next.recipes.robot_action_split.discovery import SpanWorkItem, discover_spans
from cosmos_curator.next.recipes.robot_action_split.lance_sink import (
    append_clip_fragment,
    open_or_create_clip_table,
    write_clip_fragment,
)
from cosmos_curator.next.recipes.robot_action_split.pipeline import _publish_outcomes, run
from cosmos_curator.next.recipes.robot_action_split.processing import (
    _build_sidecar_bytes,
    _make_base_fields,
    process_batch,
)
from cosmos_curator.next.recipes.robot_action_split.records import clip_table

_SOURCE_TASK_DESCRIPTION = "  A person prepares coffee by loading a pod and opening the machine.\n"


def _write_outcomes(outcomes: list[dict], *, lance_uri: str) -> int:
    """Bootstrap, stage, and commit one fragment of successful outcome rows."""
    open_or_create_clip_table(uri=lance_uri, storage_profile="default")
    table = clip_table(
        outcomes,
        record_schema_version=CLIP_RECORD_SCHEMA_VERSION,
        media_contract_version=MEDIA_CONTRACT_VERSION,
    )
    candidate = write_clip_fragment(table, uri=lance_uri, storage_profile="default")
    assert candidate is not None
    return append_clip_fragment(candidate, uri=lance_uri, storage_profile="default", attempts=1)


def _make_all_intra_mp4(output: Path) -> None:
    """Generate a 10-second all-intra H.264 MP4 at 24 fps using libopenh264.

    Every frame is a keyframe (-g 1), so smart cut can stream-copy the entire
    span without needing libx264 (which may not be present in dev builds).
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    # Check libopenh264 is available
    enc_result = subprocess.run(  # noqa: S603
        [shutil.which("ffmpeg") or "ffmpeg", "-encoders"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if "libopenh264" not in enc_result.stdout:
        pytest.skip("libopenh264 encoder not available")
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run(  # noqa: S603
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration=10:size=64x64:rate={_FPS}",
            "-c:v",
            "libopenh264",
            "-g",
            "1",
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
        capture_output=True,
    )


def _uri_to_path(uri: str) -> Path:
    """Convert a local file:// URI or plain path string to a Path."""
    if uri.startswith("file://"):
        return Path(urlparse(uri).path)
    return Path(uri)


# ---------------------------------------------------------------------------
# Fixture: synthetic dataset
# ---------------------------------------------------------------------------

_FPS = 24
_FRAMES_PER_SUBTASK = 100  # ~4.17 s per subtask
_TOTAL_FRAMES = _FRAMES_PER_SUBTASK * 2  # 200 frames = ~8.33 s


def _write_dataset(root: Path) -> Path:
    """Write a minimal LeRobot/Mecka dataset under *root* and return its path."""
    dataset = root / "dataset"

    # meta/info.json
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "fps": _FPS,
                "features": {
                    "observation.images.main": {"shape": [64, 64, 3]},
                },
            }
        ),
        encoding="utf-8",
    )

    # meta/subtasks.parquet
    pq.write_table(
        pa.table(
            {
                "subtask_index": pa.array([0, 1], type=pa.int64()),
                "subtask": pa.array(["pick up coffee pod", "open machine lid"]),
            }
        ),
        str(meta / "subtasks.parquet"),
    )

    # meta/tasks.parquet
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([0], type=pa.int64()),
                "task": pa.array(["make coffee"]),
            }
        ),
        str(meta / "tasks.parquet"),
    )

    # meta/episodes/chunk-000/file-000.parquet
    ep_dir = meta / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([0], type=pa.int64()),
                "episode_id": pa.array(["ep_000"]),
                "task_description": pa.array(
                    [_SOURCE_TASK_DESCRIPTION],
                    type=pa.large_string(),
                ),
                "videos/observation.images.main/chunk_index": pa.array([0], type=pa.int64()),
                "videos/observation.images.main/file_index": pa.array([0], type=pa.int64()),
                "videos/observation.images.main/from_timestamp": pa.array([0.0]),
                "dataset_from_index": pa.array([0], type=pa.int64()),
            }
        ),
        str(ep_dir / "file-000.parquet"),
    )

    # data/chunk-000/file-000.parquet — 200 frames: 100 subtask0, 100 subtask1
    data_dir = dataset / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    n = _TOTAL_FRAMES
    frame_indices = list(range(n))
    subtask_indices = [0] * _FRAMES_PER_SUBTASK + [1] * _FRAMES_PER_SUBTASK
    action_data = [[float(i), float(i) * 0.1] for i in range(n)]
    # Straight-line dolly trajectory (identity orientation, position drifting
    # along one axis) so process_batch's camera-motion annotation has a
    # non-trivial, deterministic trajectory to describe.
    camera_position = [[0.0, i * 0.02, 0.0] for i in range(n)]
    camera_rotation = [[0.0, 0.0, 0.0, 1.0] for _ in range(n)]
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([0] * n, type=pa.int64()),
                "frame_index": pa.array(frame_indices, type=pa.int64()),
                "subtask_index": pa.array(subtask_indices, type=pa.int64()),
                "task_index": pa.array([0] * n, type=pa.int64()),
                "action": pa.array(action_data),
                "observation.state.camera_position": pa.array(camera_position),
                "observation.state.camera_rotation": pa.array(camera_rotation),
            }
        ),
        str(data_dir / "file-000.parquet"),
    )

    # videos/observation.images.main/chunk-000/file-000.mp4
    # All-intra so smart cut stream-copies without needing libx264.
    vid_dir = dataset / "videos" / "observation.images.main" / "chunk-000"
    vid_dir.mkdir(parents=True)
    _make_all_intra_mp4(vid_dir / "file-000.mp4")

    return dataset


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------


def _make_config(dataset_path: Path, output_path: Path) -> ResolvedRobotActionSplitConfig:
    return ResolvedRobotActionSplitConfig.model_validate(
        {
            "schema_version": 1,
            "kind": "robot-action-split",
            "input": {
                "uris": [str(dataset_path.parent)],  # dataset root parent
                "source_dataset": "test_dataset",
            },
            "split": {
                "min_duration_s": 4.0,
                "max_duration_s": 20.0,
            },
            "output": {
                "media_root": str(output_path),
                "lance_uri": str(output_path / "clips.lance"),
                "action_format": "pickle",
                "views": [],
            },
            "execution": {
                "storage_profile": "default",
                "discovery_workers": 1,
                "max_segments_per_batch": 50,
                "cut_attempts": 1,
                "media_write_attempts": 1,
                "progress": False,
            },
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Write the synthetic dataset and return its path."""
    return _write_dataset(tmp_path)


@pytest.fixture
def output_path(tmp_path: Path) -> Path:
    """Create and return an empty output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture
def span_work_item_fields() -> dict[str, Any]:
    """Return fields for one item carrying a whitespace-sensitive description."""
    return {
        "source_id": "source-0",
        "span_group_id": "span-0",
        "clip_id": "clip-0",
        "view_name": "observation.images.main",
        "chunk_mp4_uri": "chunk.mp4",
        "data_parquet_uri": "data.parquet",
        "episode_index": 0,
        "episode_frame_base": 0,
        "frame_start": 0,
        "frame_end": 100,
        "native_fps": 24.0,
        "episode_from_timestamp": 0.0,
        "subtask_index": 0,
        "subtask_name": "pick up coffee pod",
        "task_index": 0,
        "task_name": "make coffee",
        "task_description": "  Prepare coffee without spilling.\n",
        "subtask_label_resolved": True,
        "episode_id": "ep_000",
        "camera_intrinsics": None,
    }


def test_base_outcome_fields_include_task_description(span_work_item_fields: dict[str, Any]) -> None:
    """Verify common outcome fields carry descriptions to success and failure rows."""
    span_work_item = SpanWorkItem(**span_work_item_fields)
    fields = _make_base_fields(span_work_item, "test_dataset")

    assert fields["task_description"] == span_work_item.task_description


def test_sidecar_places_task_description_after_task_name(span_work_item_fields: dict[str, Any]) -> None:
    """Verify the JSON sidecar preserves the description and its key position."""
    span_work_item = SpanWorkItem(**span_work_item_fields)
    sidecar = json.loads(_build_sidecar_bytes(span_work_item, "clip.mp4", "action.bin"))
    keys = list(sidecar)

    assert sidecar["task_description"] == span_work_item.task_description
    assert keys.index("task_description") == keys.index("task_name") + 1


def test_discovery_finds_two_spans(dataset: Path, output_path: Path) -> None:
    """Stage 1: parquet discovery emits exactly two spans with correct geometry.

    The data parquet has 200 frames split evenly between subtask 0 and subtask 1.
    Both subtasks span ~4.17 s at 24 fps, so both pass the min_duration_s=4.0 filter.
    """
    config = _make_config(dataset, output_path)
    batches = discover_spans(config)

    # Two spans across one batch (same chunk MP4 + data parquet).
    items = [item for batch in batches for item in batch.items]
    assert len(items) == 2, f"Expected 2 spans, got {len(items)}"

    subtask_names = {item.subtask_name for item in items}
    assert subtask_names == {"pick up coffee pod", "open machine lid"}
    assert all(item.task_description == _SOURCE_TASK_DESCRIPTION for item in items)

    # Geometry: each span covers _FRAMES_PER_SUBTASK frames.
    for item in items:
        assert item.frame_end - item.frame_start == _FRAMES_PER_SUBTASK
        assert abs(item.duration_s - _FRAMES_PER_SUBTASK / _FPS) < 0.01
        assert item.native_fps == _FPS
        assert item.episode_from_timestamp == 0.0

    # span_group_id is distinct per span; clip_id is a stable digest distinct from span_group_id.
    group_ids = [item.span_group_id for item in items]
    assert len(set(group_ids)) == 2
    clip_ids = [item.clip_id for item in items]
    assert len(set(clip_ids)) == 2
    # clip_id is always a fresh digest (includes bitrate + contract version), never the raw span_group_id.
    for item in items:
        assert item.clip_id != item.span_group_id
        assert len(item.clip_id) == 64  # SHA-256 hex


def test_span_timestamp_translation(dataset: Path, output_path: Path) -> None:
    """Stage 1b: nanosecond translation uses integer frame-index space.

    With episode_from_timestamp=0.0 and fps=24:
      subtask 0  frames 0-99   -> start_ns=0, end_ns=round(100/24*1e9)
      subtask 1  frames 100-199 -> start_ns=round(100/24*1e9)
    """
    config = _make_config(dataset, output_path)
    items = sorted(
        [item for batch in discover_spans(config) for item in batch.items],
        key=lambda x: x.frame_start,
    )

    subtask0, subtask1 = items
    assert subtask0.start_ns == 0
    expected_mid_ns = round(_FRAMES_PER_SUBTASK / _FPS * 1e9)
    assert subtask0.end_ns == expected_mid_ns
    assert subtask1.start_ns == expected_mid_ns
    assert subtask1.end_ns == round(_TOTAL_FRAMES / _FPS * 1e9)


def test_cut_produces_clips_and_action_files(dataset: Path, output_path: Path) -> None:
    """Stage 2: processing cuts both spans and writes clips + action files.

    Each successful span produces:
      output/video/observation.images.main/<clip_id>.mp4
      output/action/<span_group_id>.bin
      output/temp_metas/<clip_id>.json
    """
    config = _make_config(dataset, output_path)
    batches = discover_spans(config)
    assert len(batches) == 1

    outcomes = process_batch(batches[0], config=config)

    succeeded = [o for o in outcomes if o["status"] == "success"]
    failed = [o for o in outcomes if o["status"] != "success"]
    assert not failed, f"Unexpected failures: {failed}"
    assert len(succeeded) == 2
    assert all(outcome["task_description"] == _SOURCE_TASK_DESCRIPTION for outcome in succeeded)

    for outcome in succeeded:
        # Clip MP4 exists and is non-empty.
        clip_path = _uri_to_path(outcome["clip_uri"])
        assert clip_path.exists(), f"Clip not found: {clip_path}"
        assert clip_path.stat().st_size > 0

        # Action bin exists.
        action_path = _uri_to_path(outcome["action_data_uri"])
        assert action_path.exists(), f"Action file not found: {action_path}"

        # Sidecar JSON is valid and references the correct clip.
        sidecar_path = output_path / "temp_metas" / f"{outcome['clip_id']}.json"
        assert sidecar_path.exists()
        sidecar = json.loads(sidecar_path.read_text())
        assert sidecar["clip_uuid"] == outcome["clip_id"]
        assert sidecar["span_group_uuid"] == outcome["span_group_id"]
        assert sidecar["task_description"] == _SOURCE_TASK_DESCRIPTION

    # Action data contains the correct frame count.
    for outcome in succeeded:
        action_data = pickle.loads(_uri_to_path(outcome["action_data_uri"]).read_bytes())  # noqa: S301
        assert "action" in action_data
        assert len(action_data["action"]) == _FRAMES_PER_SUBTASK


def test_camera_motion_annotation_populated(dataset: Path, output_path: Path) -> None:
    """Stage 2b: a clip with camera trajectory data gets a non-empty motion annotation.

    The fixture's ``camera_position``/``camera_rotation`` columns describe a
    straight-line dolly move, so ``process_batch`` should compute a non-``None``
    ``camera_motion_annotation`` string for both successful spans.
    """
    config = _make_config(dataset, output_path)
    batches = discover_spans(config)
    outcomes = process_batch(batches[0], config=config)

    succeeded = [o for o in outcomes if o["status"] == "success"]
    assert len(succeeded) == 2
    for outcome in succeeded:
        annotation = outcome["camera_motion_annotation"]
        assert isinstance(annotation, str)
        assert annotation
        assert "dollies" in annotation.lower()


def test_lance_write_produces_correct_rows(dataset: Path, output_path: Path) -> None:
    """Stage 3: successful outcomes are committed to Lance with the correct schema.

    Asserts row count and clip_id uniqueness across the two rows.
    """
    config = _make_config(dataset, output_path)
    batches = discover_spans(config)
    outcomes = process_batch(batches[0], config=config)

    lance_uri = str(output_path / "clips.lance")
    version = _write_outcomes(outcomes, lance_uri=lance_uri)

    assert version >= 1

    ds = lance.dataset(lance_uri)
    table = ds.to_table()
    assert len(table) == 2

    # clip_id is unique.
    clip_ids = table.column("clip_id").to_pylist()
    assert len(set(clip_ids)) == 2

    # subtask names are what we put in the parquet.
    subtask_names = set(table.column("subtask_name").to_pylist())
    assert subtask_names == {"pick up coffee pod", "open machine lid"}
    assert table.column("task_description").to_pylist() == [_SOURCE_TASK_DESCRIPTION, _SOURCE_TASK_DESCRIPTION]
    assert table.column("record_schema_version").to_pylist() == [4, 4]

    # The transaction identifies the producing recipe and operation.
    txn = ds.read_transaction(version)
    assert txn is not None
    assert txn.transaction_properties == {"kind": "robot-action-split", "operation": "append-clips"}


def test_lance_write_produces_multiple_fragments(dataset: Path, output_path: Path) -> None:
    """A publish batch smaller than the successful row count commits one fragment per batch.

    Pure Lance I/O (``process_batch`` and ``_publish_outcomes`` called directly,
    not through Ray Data), so this does not need the ``ray_local`` fixture other
    Ray-Data-path tests in this module require.
    """
    config = _make_config(dataset, output_path)
    config = config.model_copy(update={"execution": config.execution.model_copy(update={"clips_per_publish_batch": 1})})
    batches = discover_spans(config)
    outcomes = process_batch(batches[0], config=config)
    success = [o for o in outcomes if o["status"] == "success"]

    lance_uri = str(output_path / "clips_frags.lance")
    open_or_create_clip_table(uri=lance_uri, storage_profile="default")
    config = config.model_copy(update={"output": config.output.model_copy(update={"lance_uri": lance_uri})})
    _publish_outcomes(outcomes, config=config, initial_clips_version=1)

    ds = lance.dataset(lance_uri)
    assert ds.count_rows() == len(success)
    assert len(ds.get_fragments()) == len(success)


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_full_pipeline_run(dataset: Path, output_path: Path) -> None:
    """Full pipeline over the default Ray Data path: run() reports correct counts.

    Declares ``ray_local`` so the run always lands on the small shared cluster
    rather than on whatever cluster an earlier test happened to leave behind.
    Pinning the cluster also pins its CPU budget, so ``execution.cut_cpus`` is
    written explicitly below instead of inheriting the production default, which
    is larger than that budget.
    """
    config_path = output_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "kind": "robot-action-split",
                "input": {
                    "uris": [str(dataset.parent)],
                    "source_dataset": "test_dataset",
                },
                "split": {"min_duration_s": 4.0, "max_duration_s": 20.0},
                "output": {
                    "media_root": str(output_path / "media"),
                    "lance_uri": str(output_path / "clips.lance"),
                    "action_format": "pickle",
                    "views": [],
                },
                "execution": {
                    "storage_profile": "default",
                    "discovery_workers": 1,
                    # Must fit the ``ray_local`` CPU budget. Ray Data does not
                    # reject a task whose request exceeds cluster capacity - it
                    # backpressures it under ``ResourceBudget`` with no error and
                    # no progress - so an oversized value hangs the run instead
                    # of failing it.
                    "cut_cpus": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = run(str(config_path))

    assert summary["succeeded"] == 2
    assert summary["failed"] == 0
    assert summary["total"] == 2
    assert "attempt_id" not in summary

    # Lance table committed with 2 rows.
    ds = lance.dataset(str(output_path / "clips.lance"))
    assert len(ds.to_table()) == 2


def test_sequential_pipeline_run(dataset: Path, output_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit sequential path (ray_data=false) produces same counts as the Ray Data path."""
    import cosmos_curator.next.recipes.robot_action_split.pipeline as _pipeline_mod  # noqa: PLC0415

    _msg = "_iter_ray_data must not be called when ray_data=False"

    def _should_not_be_called(*_a: object, **_kw: object) -> object:
        raise AssertionError(_msg)

    monkeypatch.setattr(_pipeline_mod, "_iter_ray_data", _should_not_be_called)
    config_path = output_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "kind": "robot-action-split",
                "input": {
                    "uris": [str(dataset.parent)],
                    "source_dataset": "test_dataset",
                },
                "split": {"min_duration_s": 4.0, "max_duration_s": 20.0},
                "output": {
                    "media_root": str(output_path / "media"),
                    "lance_uri": str(output_path / "clips.lance"),
                    "action_format": "pickle",
                    "views": [],
                },
                "execution": {
                    "storage_profile": "default",
                    "discovery_workers": 1,
                    "ray_data": False,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = run(str(config_path))

    assert summary["succeeded"] == 2
    assert summary["failed"] == 0
    assert summary["total"] == 2
