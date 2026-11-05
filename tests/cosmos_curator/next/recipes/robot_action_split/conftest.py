# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A small local LeRobot-style dataset, its chunk work items, and deterministic chunk outcomes.

The dataset has one episode of two subtasks, recorded by every configured camera
view. Each view's chunk MP4 is one work item, so three views give the shared
append suite three chunks. Discovery never decodes video, so the conformance
dataset writes placeholder MP4s; tests that cut real clips encode one instead.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cosmos_curator.next.recipes.robot_action_split import pipeline
from cosmos_curator.next.recipes.robot_action_split.bindings import resolve_bindings
from cosmos_curator.next.recipes.robot_action_split.config import ResolvedRobotActionSplitConfig
from cosmos_curator.next.recipes.robot_action_split.discovery import decode_chunk, discover_items
from cosmos_curator.next.recipes.robot_action_split.processing import (
    chunk_outcome,
    clip_fields,
    failed_chunk_outcome,
    failed_clip,
)
from tests.cosmos_curator.next.append.conformance import RecipeUnderTest

FPS = 24
FRAMES_PER_SUBTASK = 100  # ~4.17 s per subtask
TOTAL_FRAMES = FRAMES_PER_SUBTASK * 2  # 200 frames = ~8.33 s
SOURCE_TASK_DESCRIPTION = "  A person prepares coffee by loading a pod and opening the machine.\n"
MAIN_VIEW = "observation.images.main"
CONFORMANCE_VIEWS = (MAIN_VIEW, "observation.images.side", "observation.images.wrist")


def _encode_all_intra_mp4(output: Path) -> None:
    """Generate a 10-second all-intra H.264 MP4 at 24 fps using libopenh264.

    Every frame is a keyframe (-g 1), so smart cut can stream-copy the entire
    span without needing libx264 (which may not be present in dev builds).
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not available")
    encoders = subprocess.run(  # noqa: S603
        [ffmpeg, "-encoders"], capture_output=True, text=True, timeout=10, check=False
    )
    if "libopenh264" not in encoders.stdout:
        pytest.skip("libopenh264 encoder not available")
    subprocess.run(  # noqa: S603
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration=10:size=64x64:rate={FPS}",
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


def write_dataset(root: Path, *, views: tuple[str, ...] = (MAIN_VIEW,), encode_video: bool = True) -> Path:
    """Write a minimal LeRobot/Mecka dataset under ``root`` and return its shard path.

    ``encode_video`` writes a real all-intra MP4 per view, skipping the test when
    no libopenh264 encoder is available; otherwise each view gets placeholder bytes.
    """
    dataset = root / "dataset"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps({"fps": FPS, "features": {view: {"shape": [64, 64, 3]} for view in views}}), encoding="utf-8"
    )
    pq.write_table(
        pa.table(
            {
                "subtask_index": pa.array([0, 1], type=pa.int64()),
                "subtask": pa.array(["pick up coffee pod", "open machine lid"]),
            }
        ),
        str(meta / "subtasks.parquet"),
    )
    pq.write_table(
        pa.table({"task_index": pa.array([0], type=pa.int64()), "task": pa.array(["make coffee"])}),
        str(meta / "tasks.parquet"),
    )

    ep_dir = meta / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True)
    episode: dict[str, pa.Array] = {
        "episode_index": pa.array([0], type=pa.int64()),
        "episode_id": pa.array(["ep_000"]),
        "task_description": pa.array([SOURCE_TASK_DESCRIPTION], type=pa.large_string()),
        "dataset_from_index": pa.array([0], type=pa.int64()),
    }
    for view in views:
        episode[f"videos/{view}/chunk_index"] = pa.array([0], type=pa.int64())
        episode[f"videos/{view}/file_index"] = pa.array([0], type=pa.int64())
        episode[f"videos/{view}/from_timestamp"] = pa.array([0.0])
    pq.write_table(pa.table(episode), str(ep_dir / "file-000.parquet"))

    # 200 frames: 100 of subtask 0, then 100 of subtask 1, with action data and a
    # straight-line dolly trajectory so camera motion has something to describe.
    data_dir = dataset / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    n = TOTAL_FRAMES
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([0] * n, type=pa.int64()),
                "frame_index": pa.array(list(range(n)), type=pa.int64()),
                "subtask_index": pa.array([0] * FRAMES_PER_SUBTASK + [1] * FRAMES_PER_SUBTASK, type=pa.int64()),
                "task_index": pa.array([0] * n, type=pa.int64()),
                "action": pa.array([[float(i), float(i) * 0.1] for i in range(n)]),
                "observation.state.camera_position": pa.array([[0.0, i * 0.02, 0.0] for i in range(n)]),
                "observation.state.camera_rotation": pa.array([[0.0, 0.0, 0.0, 1.0] for _ in range(n)]),
            }
        ),
        str(data_dir / "file-000.parquet"),
    )

    for index, view in enumerate(views):
        vid_dir = dataset / "videos" / view / "chunk-000"
        vid_dir.mkdir(parents=True)
        target = vid_dir / "file-000.mp4"
        if not encode_video:
            target.write_bytes(b"placeholder mp4")
        elif index == 0:
            _encode_all_intra_mp4(target)
        else:
            shutil.copyfile(dataset / "videos" / views[0] / "chunk-000" / "file-000.mp4", target)
    return dataset


def make_config(dataset: Path, media_root: Path, **execution: object) -> ResolvedRobotActionSplitConfig:
    """Configure a small run over ``dataset`` with pickle actions and execution that fits a test cluster."""
    return ResolvedRobotActionSplitConfig.model_validate(
        {
            "schema_version": 1,
            "kind": "robot-action-split",
            "input": {"uris": [str(dataset.parent)], "source_dataset": "test_dataset"},
            "split": {"min_duration_s": 4.0, "max_duration_s": 20.0},
            "output": {"media_root": str(media_root), "action_format": "pickle"},
            "execution": {
                "discovery_workers": 1,
                "cut_cpus": 1.0,
                "media_write_attempts": 1,
                "rows_per_checkpoint": 1,
                "rows_per_fragment": 2,
                **execution,
            },
        }
    )


@pytest.fixture
def config(tmp_path: Path) -> ResolvedRobotActionSplitConfig:
    """Return a three-chunk config over placeholder media with small checkpoint groups and fragments."""
    return make_config(write_dataset(tmp_path, views=CONFORMANCE_VIEWS, encode_video=False), tmp_path / "media")


def plan_outcome(item: dict[str, Any], config: ResolvedRobotActionSplitConfig) -> dict[str, Any]:
    """Finish a chunk deterministically from its view, without media.

    The main view finishes both clips; the side view finishes one clip and fails
    the other's cut; the wrist view's chunk MP4 is missing.
    """
    chunk = decode_chunk(item)
    source_dataset = config.input.source_dataset
    view = chunk.items[0].view_name
    if view.endswith("wrist"):
        return failed_chunk_outcome(item, chunk.items, source_dataset, "chunk-stage", "synthetic missing chunk")
    clips = []
    for index, span in enumerate(chunk.items):
        if view.endswith("side") and index:
            clips.append(failed_clip(span, source_dataset, "cut", "cut produced 99/100 frames"))
            continue
        clips.append(
            clip_fields(span, source_dataset)
            | {
                "clip_uri": f"file://{config.output.media_root}/video/{view}/{span.clip_id}.mp4",
                "action_data_uri": f"file://{config.output.media_root}/action/{span.span_group_id}.pickle",
                "camera_motion_annotation": None,
                "error_stage": None,
                "error_message": None,
            }
        )
    return chunk_outcome(item, clips)


@pytest.fixture
def results(config: ResolvedRobotActionSplitConfig) -> pa.Table:
    """Build canonical chunk outcomes: one finished, one partly failed, one missing source."""
    return pa.Table.from_pylist(
        [plan_outcome(item, config) for item in discover_items(config).to_pylist()],
        schema=resolve_bindings(config).spec.result_fields,
    )


def _retune(config: ResolvedRobotActionSplitConfig) -> ResolvedRobotActionSplitConfig:
    execution = {"rows_per_checkpoint": 2, "rows_per_fragment": 3, "cut_cpus": 2.0, "discovery_workers": 2}
    return config.model_copy(update={"execution": config.execution.model_copy(update=execution)})


def _recontract(config: ResolvedRobotActionSplitConfig) -> ResolvedRobotActionSplitConfig:
    return config.model_copy(update={"output": config.output.model_copy(update={"action_format": "bin"})})


# How the shared conformance suite and fresh-cluster scenario drive this recipe.
RECIPE = RecipeUnderTest(
    pipeline=pipeline,
    resolve_bindings=resolve_bindings,
    discover=discover_items,
    retune=_retune,
    recontract=_recontract,
)


@pytest.fixture
def recipe() -> RecipeUnderTest:
    """Drive the robot-action-split pipeline with its own bindings and discovery."""
    return RECIPE
