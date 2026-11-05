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

"""End-to-end tests for robot-action-split over a synthetic dataset with a real all-intra MP4.

The dataset (see ``conftest.write_dataset``) holds one 24 fps episode of two
subtasks, 100 frames each, recorded by one view: one chunk work item owing two
clips. Real smart cuts run here, so these tests skip without libopenh264.
Publication runs through ``run_config`` with the processing plan run in-process,
except the integration test, which runs the real Ray Data plan.
"""

import json
import pickle
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import lance
import pyarrow as pa
import pytest
import yaml

from cosmos_curator.next.append.execution import ProcessingScope
from cosmos_curator.next.recipes.robot_action_split import pipeline
from cosmos_curator.next.recipes.robot_action_split.config import ResolvedRobotActionSplitConfig, resolve_config
from cosmos_curator.next.recipes.robot_action_split.discovery import SpanWorkItem, discover_items, discover_spans
from cosmos_curator.next.recipes.robot_action_split.processing import _build_sidecar_bytes, clip_fields, cut_chunks
from tests.cosmos_curator.next.append.helpers import replace_ray, stage_results
from tests.cosmos_curator.next.recipes.robot_action_split.conftest import (
    FPS,
    FRAMES_PER_SUBTASK,
    SOURCE_TASK_DESCRIPTION,
    TOTAL_FRAMES,
    make_config,
    write_dataset,
)


def _uri_to_path(uri: str) -> Path:
    """Convert a local file:// URI or plain path string to a Path."""
    if uri.startswith("file://"):
        return Path(urlparse(uri).path)
    return Path(uri)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Write the synthetic dataset with a real MP4 and return its shard path."""
    return write_dataset(tmp_path)


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


def _cut_in_process(processing: ProcessingScope, config: ResolvedRobotActionSplitConfig) -> None:
    """Run the plan's real per-chunk cuts without Ray and stage their outcomes as the plan's writer would."""
    attempt = processing._attempt
    stage_results(attempt, cut_chunks(attempt.selection, config=config), complete=False)


def _run(config: ResolvedRobotActionSplitConfig, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    with monkeypatch.context() as patch:
        replace_ray(pipeline, patch, results=None, process=lambda scope: _cut_in_process(scope, config))
        return pipeline.run_config(config)


def test_clip_fields_include_task_description(span_work_item_fields: dict[str, Any]) -> None:
    """Common clip fields carry descriptions to finished and failed clips."""
    span_work_item = SpanWorkItem(**span_work_item_fields)
    fields = clip_fields(span_work_item, "test_dataset")

    assert fields["task_description"] == span_work_item.task_description


def test_sidecar_places_task_description_after_task_name(span_work_item_fields: dict[str, Any]) -> None:
    """The JSON sidecar preserves the description and its key position."""
    span_work_item = SpanWorkItem(**span_work_item_fields)
    sidecar = json.loads(_build_sidecar_bytes(span_work_item, "clip.mp4", "action.bin"))
    keys = list(sidecar)

    assert sidecar["task_description"] == span_work_item.task_description
    assert keys.index("task_description") == keys.index("task_name") + 1


def test_discovery_finds_two_spans_in_one_chunk(dataset: Path, output_path: Path) -> None:
    """Parquet discovery emits one chunk owing exactly two spans with correct geometry.

    Both subtasks span ~4.17 s at 24 fps, so both pass the min_duration_s=4.0 filter.
    """
    config = make_config(dataset, output_path)
    batches = discover_spans(config)
    assert len(batches) == 1
    items = batches[0].items
    assert len(items) == 2, f"Expected 2 spans, got {len(items)}"

    assert {item.subtask_name for item in items} == {"pick up coffee pod", "open machine lid"}
    assert all(item.task_description == SOURCE_TASK_DESCRIPTION for item in items)
    for item in items:
        assert item.frame_end - item.frame_start == FRAMES_PER_SUBTASK
        assert abs(item.duration_s - FRAMES_PER_SUBTASK / FPS) < 0.01
        assert item.native_fps == FPS
        assert item.episode_from_timestamp == 0.0

    # span_group_id is distinct per span; clip_id is a stable digest distinct from span_group_id.
    assert len({item.span_group_id for item in items}) == 2
    assert len({item.clip_id for item in items}) == 2
    for item in items:
        assert item.clip_id != item.span_group_id
        assert len(item.clip_id) == 64  # SHA-256 hex


def test_span_timestamp_translation(dataset: Path, output_path: Path) -> None:
    """Nanosecond translation uses integer frame-index space.

    With episode_from_timestamp=0.0 and fps=24:
      subtask 0  frames 0-99   -> start_ns=0, end_ns=round(100/24*1e9)
      subtask 1  frames 100-199 -> start_ns=round(100/24*1e9)
    """
    config = make_config(dataset, output_path)
    items = sorted(discover_spans(config)[0].items, key=lambda x: x.frame_start)

    subtask0, subtask1 = items
    assert subtask0.start_ns == 0
    expected_mid_ns = round(FRAMES_PER_SUBTASK / FPS * 1e9)
    assert subtask0.end_ns == expected_mid_ns
    assert subtask1.start_ns == expected_mid_ns
    assert subtask1.end_ns == round(TOTAL_FRAMES / FPS * 1e9)


def test_cut_produces_clips_action_files_and_sidecars(dataset: Path, output_path: Path) -> None:
    """Processing one chunk cuts both spans and writes clips, action files, and sidecars.

    Each finished clip produces:
      output/video/observation.images.main/<clip_id>.mp4
      output/action/<action_id>.pickle
      output/temp_metas/<clip_id>.json
    """
    config = make_config(dataset, output_path)
    outcome = cut_chunks(discover_items(config), config=config).to_pylist()[0]

    assert outcome["error_stage"] is None
    clips = outcome["clips"]
    assert [clip["error_stage"] for clip in clips] == [None, None]
    assert all(clip["task_description"] == SOURCE_TASK_DESCRIPTION for clip in clips)
    for clip in clips:
        clip_path = _uri_to_path(clip["clip_uri"])
        assert clip_path.stat().st_size > 0
        sidecar = json.loads((output_path / "temp_metas" / f"{clip['clip_id']}.json").read_text())
        assert sidecar["clip_uuid"] == clip["clip_id"]
        assert sidecar["span_group_uuid"] == clip["span_group_id"]
        assert sidecar["task_description"] == SOURCE_TASK_DESCRIPTION
        action_data = pickle.loads(_uri_to_path(clip["action_data_uri"]).read_bytes())  # noqa: S301
        assert len(action_data["action"]) == FRAMES_PER_SUBTASK
        # The fixture's straight-line dolly trajectory gets a motion annotation.
        assert "dollies" in clip["camera_motion_annotation"].lower()


def test_run_publishes_one_row_per_clip_once(dataset: Path, output_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run appends both clips with their labels in one commit; a rerun finds nothing to do."""
    config = make_config(dataset, output_path)
    summary = _run(config, monkeypatch)

    assert (summary["input_items"], summary["selected_items"], summary["remaining_items"]) == (1, 1, 0)
    assert (summary["appended_rows"], summary["errors"], summary["failed_clips"]) == (2, 0, 0)
    dataset_table = lance.dataset(config.output.lance_uri)
    table = dataset_table.to_table()
    assert sorted(table["subtask_name"].to_pylist()) == ["open machine lid", "pick up coffee pod"]
    assert table["task_description"].to_pylist() == [SOURCE_TASK_DESCRIPTION, SOURCE_TASK_DESCRIPTION]
    assert table["error_stage"].null_count == 2
    assert len(set(table["chunk_id"].to_pylist())) == 1
    assert summary["clips_lance_version"] == dataset_table.version

    again = _run(config, monkeypatch)
    assert (again["selected_items"], again["appended_rows"], again["failed_clips"]) == (0, 0, 0)
    assert again["clips_lance_version"] == summary["clips_lance_version"]


def test_rows_pack_into_fragments_by_rows_per_fragment(
    dataset: Path, output_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fragment size is clip rows, independent of the chunk that produced them."""
    config = make_config(dataset, output_path, rows_per_fragment=1)
    _run(config, monkeypatch)

    clips = lance.dataset(config.output.lance_uri)
    assert clips.count_rows() == 2
    assert [fragment.count_rows() for fragment in clips.get_fragments()] == [1, 1]


def test_a_missing_chunk_publishes_its_clips_as_errors(
    dataset: Path, output_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every clip of a chunk whose MP4 is gone keeps a row naming the failure, and is not retried."""
    config = make_config(dataset, output_path)
    next(dataset.glob("videos/*/chunk-000/file-000.mp4")).unlink()
    summary = _run(config, monkeypatch)

    assert (summary["appended_rows"], summary["errors"]) == (2, 1)
    assert (summary["failed_clips"], summary["failed_clips_by_stage"]) == (2, {"chunk-stage": 2})
    rows = lance.dataset(config.output.lance_uri).to_table().to_pylist()
    assert {row["error_stage"] for row in rows} == {"chunk-stage"}
    assert all(row["clip_uri"] is None for row in rows)
    assert _run(config, monkeypatch)["selected_items"] == 0


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_full_pipeline_run(dataset: Path, output_path: Path) -> None:
    """The real Ray Data plan cuts, checkpoints, and publishes; a rerun appends nothing.

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
                "input": {"uris": [str(dataset.parent)], "source_dataset": "test_dataset"},
                "split": {"min_duration_s": 4.0, "max_duration_s": 20.0},
                "output": {"media_root": str(output_path / "media"), "action_format": "pickle"},
                "execution": {
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
    config = resolve_config(config_path)

    summary = pipeline.run_config(config)

    assert (summary["selected_items"], summary["appended_rows"], summary["errors"]) == (1, 2, 0)
    assert summary["cleanup_complete"] is True
    clips = lance.dataset(config.output.lance_uri).to_table()
    assert clips.num_rows == 2
    assert all(_uri_to_path(uri).is_file() for uri in clips["clip_uri"].to_pylist())
    assert not Path(config.output.staging_root_uri).exists() or not any(Path(config.output.staging_root_uri).iterdir())

    again = pipeline.run_config(config)
    assert (again["selected_items"], again["appended_rows"]) == (0, 0)
    assert pa.Table.equals(lance.dataset(config.output.lance_uri).to_table(), clips)
