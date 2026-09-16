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
"""Tests for video pipe input (session-based multi-cam extraction)."""

import contextlib
import json
import uuid
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import patch

import lance
import pytest

from cosmos_curator.pipelines.video.read_write.clip_metadata_lance_schema import (
    CLIP_METADATA_LANCE_DATA_STORAGE_VERSION,
    CLIP_METADATA_LANCE_SCHEMA,
    build_clip_metadata_lance_table,
)
from cosmos_curator.pipelines.video.read_write.metadata_writer_stage import ClipWriterStage
from cosmos_curator.pipelines.video.utils.video_pipe_input import (
    _multi_cam_session_to_split_task,
    _order_video_paths,
    extract_multi_cam_split_tasks,
    extract_shard_tasks,
)


@pytest.mark.parametrize(
    ("paths", "video_extensions", "primary_camera_keyword", "expected", "raises"),
    [
        # Success: single primary
        (["front_cam.mp4"], {".mp4"}, "front", ["front_cam.mp4"], nullcontext()),
        # Success: primary first, rest sorted
        (
            ["rear.mp4", "camera_front_wide.mp4", "left.mp4"],
            {".mp4"},
            "front",
            ["camera_front_wide.mp4", "left.mp4", "rear.mp4"],
            nullcontext(),
        ),
        # Success: keyword matches path substring (case-sensitive)
        (["cam_front.mp4", "back.mp4"], {".mp4"}, "front", ["cam_front.mp4", "back.mp4"], nullcontext()),
        # Success: non-video paths filtered out
        (
            ["notes.txt", "front.mp4", "side.mov"],
            {".mp4"},
            "front",
            ["front.mp4"],
            nullcontext(),
        ),
        # Success: multiple extensions (only .mp4 and .h264 included; .mov filtered out)
        (
            ["front.mp4", "rear.h264", "side.mov"],
            {".mp4", ".h264"},
            "front",
            ["front.mp4", "rear.h264"],
            nullcontext(),
        ),
        # Success: primary is only video
        (["only_primary_cam.mp4"], {".mp4"}, "primary", ["only_primary_cam.mp4"], nullcontext()),
        # Success: empty paths returns empty list
        ([], {".mp4"}, "front", [], nullcontext()),
        # Success: no video files (only non-video) returns empty list
        (["notes.txt", "data.json"], {".mp4"}, "front", [], nullcontext()),
        # No primary camera
        (
            ["rear.mp4", "left.mp4"],
            {".mp4"},
            "front",
            None,
            pytest.raises(ValueError, match=r"No primary camera found.*"),
        ),
        # Multiple primary cameras
        (
            ["front_left.mp4", "front_right.mp4", "rear.mp4"],
            {".mp4"},
            "front",
            None,
            pytest.raises(ValueError, match=r"Multiple primary cameras found.*"),
        ),
    ],
)
def test_order_video_paths(
    paths: list[str],
    video_extensions: set[str],
    primary_camera_keyword: str,
    expected: list[str] | None,
    raises: AbstractContextManager[Any],
) -> None:
    """Primary (path containing keyword) first, rest sorted; ValueError if no or multiple primary."""
    with raises:
        result = _order_video_paths(paths, video_extensions, primary_camera_keyword)
        if expected is not None:
            assert result == expected


@pytest.mark.parametrize(
    (
        "session_name",
        "create_files",
        "primary_camera_keyword",
        "verbose",
        "mock_get_files_return",
        "mock_order_return",
        "expected_num_videos",
        "raises",
    ),
    [
        # Success: two videos, primary first, verbose=False
        (
            "550e8400-e29b-41d4-a716-446655440000",
            ["front.mp4", "rear.mp4"],
            "front",
            False,
            None,
            None,
            2,
            nullcontext(),
        ),
        # Success: two videos, verbose=True (covers verbose log)
        (
            "550e8400-e29b-41d4-a716-446655440001",
            ["front.mp4", "rear.mp4"],
            "front",
            True,
            None,
            None,
            2,
            nullcontext(),
        ),
        # Ordered empty (mocked), return None, verbose=False
        (
            "session-any",
            None,
            "front",
            False,
            ["some.mp4"],
            [],
            None,
            nullcontext(),
        ),
        # Ordered empty (mocked), return None, verbose=True (covers "no video files" log)
        (
            "session-any",
            None,
            "front",
            True,
            ["some.mp4"],
            [],
            None,
            nullcontext(),
        ),
        # No primary camera: ValueError
        (
            "550e8400-e29b-41d4-a716-446655440002",
            ["rear.mp4", "left.mp4"],
            "front",
            False,
            None,
            None,
            None,
            pytest.raises(ValueError, match=r"No primary camera found.*"),
        ),
        # Multiple primary cameras: ValueError
        (
            "550e8400-e29b-41d4-a716-446655440003",
            ["front_left.mp4", "front_right.mp4", "rear.mp4"],
            "front",
            False,
            None,
            None,
            None,
            pytest.raises(ValueError, match=r"Multiple primary cameras found.*"),
        ),
    ],
)
def test_multi_cam_session_to_split_task(
    tmp_path: Path,
    session_name: str,
    create_files: list[str] | None,
    primary_camera_keyword: str,
    mock_get_files_return: list[str] | None,
    mock_order_return: list[str] | None,
    expected_num_videos: int | None,
    raises: AbstractContextManager[Any],
    *,
    verbose: bool,
) -> None:
    """Build SplitPipeTask per session, or None when no videos; ValueError when no/multiple primary."""
    sessions_prefix = str(tmp_path)
    client = None
    video_extensions: set[str] = {".mp4"}

    if create_files is not None:
        session_dir = tmp_path / session_name
        session_dir.mkdir()
        for f in create_files:
            (session_dir / f).write_bytes(b"x")

    with raises:
        if mock_get_files_return is not None or mock_order_return is not None:
            patches = []
            if mock_get_files_return is not None:
                patches.append(
                    patch(
                        "cosmos_curator.pipelines.video.utils.video_pipe_input.get_files_relative",
                        return_value=mock_get_files_return,
                    )
                )
            if mock_order_return is not None:
                patches.append(
                    patch(
                        "cosmos_curator.pipelines.video.utils.video_pipe_input._order_video_paths",
                        return_value=mock_order_return,
                    )
                )
            with contextlib.ExitStack() as stack:
                for p in patches:
                    stack.enter_context(p)
                result = _multi_cam_session_to_split_task(
                    session_name,
                    sessions_prefix,
                    client,
                    video_extensions,
                    primary_camera_keyword,
                    verbose=verbose,
                )
        else:
            result = _multi_cam_session_to_split_task(
                session_name,
                sessions_prefix,
                client,
                video_extensions,
                primary_camera_keyword,
                verbose=verbose,
            )

        if expected_num_videos is not None:
            assert result is not None
            assert len(result.videos) == expected_num_videos
        else:
            assert result is None


def _default_session_args(tmp_path: Path) -> dict[str, str | set[str]]:
    return {
        "sessions_prefix": str(tmp_path),
        "video_extensions": {".mp4"},
        "input_s3_profile_name": "default",
    }


def test_extract_multi_cam_split_tasks_empty_prefix(tmp_path: Path) -> None:
    """Empty prefix returns no tasks."""
    tasks = extract_multi_cam_split_tasks(**_default_session_args(tmp_path), primary_camera_keyword="")
    assert tasks == []


def test_extract_multi_cam_split_tasks_one_session_primary_first(tmp_path: Path) -> None:
    """One session with two videos; primary (front) at slot 0."""
    session_id = "550e8400-e29b-41d4-a716-446655440000"
    session_dir = tmp_path / session_id
    session_dir.mkdir()
    (session_dir / "rear_cam.mp4").write_bytes(b"x")
    (session_dir / "camera_front_wide.mp4").write_bytes(b"y")

    tasks = extract_multi_cam_split_tasks(
        **_default_session_args(tmp_path),
        primary_camera_keyword="front",
        verbose=False,
    )
    assert len(tasks) == 1
    task = tasks[0]
    assert len(task.videos) == 2
    # Primary (path containing "front") must be first
    assert "front" in str(task.videos[0].input_video).lower()
    assert task.video is task.videos[0]


def _legacy_clip_metadata(clip_path: Path, clip_uuid: str = "clip-1") -> dict[str, Any]:
    return {
        "span_uuid": clip_uuid,
        "source_video": "input/video.mp4",
        "start_ns": 0,
        "end_ns": 1_000_000_000,
        "width_source": 640,
        "height_source": 360,
        "framerate_source": 24.0,
        "clip_location": str(clip_path),
        "width": 640,
        "height": 360,
        "framerate": 24.0,
        "num_frames": 24,
        "video_codec": "h264",
        "num_bytes": 128,
        "valid": True,
        "has_caption": True,
        "caption_quality_flags_enabled": True,
        "total_prompt_tokens": 7,
        "total_output_tokens": 5,
        "num_caption_windows": 1,
        "windows": [
            {
                "start_ns": 0,
                "end_ns": 1_000_000_000,
                "caption_status": "success",
                "caption_failure_reason": None,
                "flag_length_outlier": False,
                "flag_repetition": False,
                "flag_near_duplicate": False,
                "qwen_caption": "caption text",
                "qwen_prompt_tokens": 7,
                "qwen_output_tokens": 5,
            }
        ],
        "filtered_windows": [],
    }


def test_extract_shard_tasks_reads_lance_metadata(tmp_path: Path) -> None:
    """Shard task extraction can consume Lance metadata without per-clip JSON."""
    input_root = tmp_path / "split"
    output_root = tmp_path / "dataset" / "v0"
    input_root.mkdir()
    clip_path = input_root / "clips" / "clip-1.mp4"
    clip_path.parent.mkdir()
    clip_path.write_bytes(b"clip")

    table = build_clip_metadata_lance_table(
        [_legacy_clip_metadata(clip_path)],
        video_uuid="video-1",
        clip_chunk_index=0,
    )
    lance.write_dataset(
        table,
        input_root / "lance" / "v0",
        schema=CLIP_METADATA_LANCE_SCHEMA,
        mode="overwrite",
        data_storage_version=CLIP_METADATA_LANCE_DATA_STORAGE_VERSION,
    )

    samples = extract_shard_tasks(
        str(input_root),
        str(output_root),
        "default",
        "default",
        "v0",
        metadata_input_format="lance",
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.uuid == "clip-1"
    assert sample.clip_location == clip_path
    assert sample.clip_metadata["span_uuid"] == "clip-1"
    assert sample.clip_metadata["start_ns"] == 0
    assert sample.clip_metadata["end_ns"] == 1_000_000_000
    assert sample.clip_metadata["windows"][0]["start_ns"] == 0
    assert sample.clip_metadata["windows"][0]["end_ns"] == 1_000_000_000
    assert sample.clip_metadata["windows"][0]["qwen_caption"] == "caption text"
    assert sample.clip_metadata["windows"][0]["qwen_prompt_tokens"] == 7


def test_extract_shard_tasks_reads_lance_metadata_with_null_ns(tmp_path: Path) -> None:
    """Lance rows with null ns timing round-trip without error.

    start_ns/end_ns (and window ns) can be null for a clip whose PTS decode failed.
    The reader applies a uniform None-filter, so null timing fields are dropped and
    consumers read them back as None via .get() (keys absent, not present-with-None).
    """
    input_root = tmp_path / "split"
    output_root = tmp_path / "dataset" / "v0"
    input_root.mkdir()
    clip_path = input_root / "clips" / "clip-1.mp4"
    clip_path.parent.mkdir()
    clip_path.write_bytes(b"clip")

    metadata = _legacy_clip_metadata(clip_path)
    metadata["start_ns"] = None
    metadata["end_ns"] = None
    metadata["windows"][0]["start_ns"] = None
    metadata["windows"][0]["end_ns"] = None

    table = build_clip_metadata_lance_table([metadata], video_uuid="video-1", clip_chunk_index=0)
    lance.write_dataset(
        table,
        input_root / "lance" / "v0",
        schema=CLIP_METADATA_LANCE_SCHEMA,
        mode="overwrite",
        data_storage_version=CLIP_METADATA_LANCE_DATA_STORAGE_VERSION,
    )

    samples = extract_shard_tasks(
        str(input_root),
        str(output_root),
        "default",
        "default",
        "v0",
        metadata_input_format="lance",
    )

    assert len(samples) == 1
    clip_metadata = samples[0].clip_metadata
    # Null timing fields are dropped by the reader's uniform None-filter (keys absent).
    assert "start_ns" not in clip_metadata
    assert "end_ns" not in clip_metadata
    window = clip_metadata["windows"][0]
    assert "start_ns" not in window
    assert "end_ns" not in window
    # the rest of the window is intact
    assert window["qwen_caption"] == "caption text"


def test_extract_shard_tasks_auto_falls_back_to_json_metadata(tmp_path: Path) -> None:
    """Auto mode keeps legacy split outputs readable when Lance is absent."""
    input_root = tmp_path / "split"
    output_root = tmp_path / "dataset" / "v0"
    metas_root = input_root / "metas" / "v0"
    metas_root.mkdir(parents=True)
    clip_path = input_root / "clips" / "clip-1.mp4"
    clip_path.parent.mkdir()
    clip_path.write_bytes(b"clip")
    metadata = _legacy_clip_metadata(clip_path)
    (input_root / "summary.json").write_text(json.dumps({"video": {"clips": ["clip-1"]}}))
    (metas_root / "clip-1.json").write_text(json.dumps(metadata))

    samples = extract_shard_tasks(
        str(input_root),
        str(output_root),
        "default",
        "default",
        "v0",
        metadata_input_format="auto",
    )

    assert len(samples) == 1
    assert samples[0].uuid == "clip-1"
    assert samples[0].clip_metadata == metadata


def _write_clip_metadata(input_root: Path, metadata: dict[str, Any], *, metadata_input_format: str) -> None:
    """Write one clip's metadata as either a Lance dataset or a per-clip JSON file.

    ``metadata_input_format`` here is always the concrete format being written (``"lance"`` or
    ``"auto"``, which exercises the JSON fallback path), never ``"json"`` directly -- both
    call sites below also pass the same value straight through to ``extract_shard_tasks``.
    """
    if metadata_input_format == "lance":
        table = build_clip_metadata_lance_table([metadata], video_uuid="video-1", clip_chunk_index=0)
        lance.write_dataset(
            table,
            input_root / "lance" / "v0",
            schema=CLIP_METADATA_LANCE_SCHEMA,
            mode="overwrite",
            data_storage_version=CLIP_METADATA_LANCE_DATA_STORAGE_VERSION,
        )
    else:
        metas_root = input_root / "metas" / "v0"
        metas_root.mkdir(parents=True)
        (input_root / "summary.json").write_text(json.dumps({"video": {"clips": [metadata["span_uuid"]]}}))
        (metas_root / f"{metadata['span_uuid']}.json").write_text(json.dumps(metadata))


def _clip_writer_stage(output_path: str) -> ClipWriterStage:
    return ClipWriterStage(
        output_path=output_path,
        input_path="input",
        output_s3_profile_name="default",
        upload_clips=True,
        upload_clip_info_in_chunks=False,
        upload_clip_info_in_lance=False,
        upload_cds_parquet=False,
        dry_run=False,
        generate_embeddings=False,
        embedding_algorithm="cosmos_embed1",
        embedding_model_version="v1",
        generate_previews=False,
    )


def test_extract_shard_tasks_reads_clip_writer_relative_output_path_clip_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-trip the relative clip location emitted by ``ClipWriterStage``."""
    monkeypatch.chdir(tmp_path)
    output_path = "split_output"
    input_root = tmp_path / output_path
    clip_uuid = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")

    writer = _clip_writer_stage(output_path)
    clip_location = str(writer._get_clip_uri(clip_uuid, writer.get_output_path_clips(writer._output_path), "mp4"))
    assert clip_location == f"{output_path}/clips/{clip_uuid}.mp4"

    clip_path = tmp_path / clip_location
    clip_path.parent.mkdir(parents=True)
    clip_path.write_bytes(b"clip")

    metadata = _legacy_clip_metadata(clip_path, clip_uuid=str(clip_uuid))
    metadata["clip_location"] = clip_location
    output_root = tmp_path / "dataset" / "v0"
    _write_clip_metadata(input_root, metadata, metadata_input_format="auto")

    samples = extract_shard_tasks(
        str(input_root),
        str(output_root),
        "default",
        "default",
        "v0",
        metadata_input_format="auto",
    )

    assert len(samples) == 1
    assert samples[0].clip_location == clip_path.resolve()


@pytest.mark.parametrize("metadata_input_format", ["lance", "auto"])
def test_extract_shard_tasks_skips_clip_outside_input_root(tmp_path: Path, metadata_input_format: str) -> None:
    """A tampered clip_location outside the input root is skipped, not read, for both metadata formats."""
    input_root = tmp_path / "split"
    output_root = tmp_path / "dataset" / "v0"
    input_root.mkdir()
    outside_clip = tmp_path / "outside" / "secret.mp4"
    outside_clip.parent.mkdir()
    outside_clip.write_bytes(b"secret")
    metadata = _legacy_clip_metadata(outside_clip)
    _write_clip_metadata(input_root, metadata, metadata_input_format=metadata_input_format)

    samples = extract_shard_tasks(
        str(input_root),
        str(output_root),
        "default",
        "default",
        "v0",
        metadata_input_format=metadata_input_format,
    )

    assert samples == []


def test_extract_shard_tasks_allow_unsafe_clip_location_restores_legacy_behavior(tmp_path: Path) -> None:
    """The opt-in escape hatch restores the pre-fix behavior of trusting clip_location as-is."""
    input_root = tmp_path / "split"
    output_root = tmp_path / "dataset" / "v0"
    metas_root = input_root / "metas" / "v0"
    metas_root.mkdir(parents=True)
    outside_clip = tmp_path / "outside" / "secret.mp4"
    outside_clip.parent.mkdir()
    outside_clip.write_bytes(b"secret")
    metadata = _legacy_clip_metadata(outside_clip)
    (input_root / "summary.json").write_text(json.dumps({"video": {"clips": ["clip-1"]}}))
    (metas_root / "clip-1.json").write_text(json.dumps(metadata))

    samples = extract_shard_tasks(
        str(input_root),
        str(output_root),
        "default",
        "default",
        "v0",
        metadata_input_format="auto",
        allow_unsafe_clip_location=True,
    )

    assert len(samples) == 1
    assert samples[0].clip_location == outside_clip


def test_extract_shard_tasks_allow_unsafe_clip_location_keeps_relative_path_unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep relative locations unresolved when compatibility mode is enabled."""
    input_root = tmp_path / "split"
    output_root = tmp_path / "dataset" / "v0"
    metas_root = input_root / "metas" / "v0"
    metas_root.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    metadata = _legacy_clip_metadata(Path("clips/other.mp4"))
    (input_root / "summary.json").write_text(json.dumps({"video": {"clips": ["clip-1"]}}))
    (metas_root / "clip-1.json").write_text(json.dumps(metadata))

    samples = extract_shard_tasks(
        str(input_root),
        str(output_root),
        "default",
        "default",
        "v0",
        metadata_input_format="auto",
        allow_unsafe_clip_location=True,
    )

    assert len(samples) == 1
    assert samples[0].clip_location == Path("clips/other.mp4")


def test_extract_multi_cam_split_tasks_only_uuid_dirs_are_sessions(tmp_path: Path) -> None:
    """Only UUID-named subdirs are considered sessions."""
    (tmp_path / "session_abc").mkdir()
    (tmp_path / "session_abc" / "video.mp4").write_bytes(b"x")
    (tmp_path / "550e8400-e29b-41d4-a716-446655440000").mkdir()
    (tmp_path / "550e8400-e29b-41d4-a716-446655440000" / "cam.mp4").write_bytes(b"y")

    tasks = extract_multi_cam_split_tasks(**_default_session_args(tmp_path), primary_camera_keyword="")
    assert len(tasks) == 1
    assert len(tasks[0].videos) == 1


def test_extract_multi_cam_split_tasks_limit(tmp_path: Path) -> None:
    """Limit caps number of sessions returned."""
    for i in range(3):
        uid = f"550e8400-e29b-41d4-a716-44665544000{i}"
        (tmp_path / uid).mkdir()
        (tmp_path / uid / "v.mp4").write_bytes(b"x")

    tasks = extract_multi_cam_split_tasks(**_default_session_args(tmp_path), primary_camera_keyword="", limit=2)
    assert len(tasks) == 2


def test_extract_multi_cam_split_tasks_skips_empty_sessions(tmp_path: Path) -> None:
    """Sessions with no video files are skipped."""
    # Session with video
    uid1 = "550e8400-e29b-41d4-a716-446655440000"
    (tmp_path / uid1).mkdir()
    (tmp_path / uid1 / "cam.mp4").write_bytes(b"x")

    # Session without video (only text file)
    uid2 = "550e8400-e29b-41d4-a716-446655440001"
    (tmp_path / uid2).mkdir()
    (tmp_path / uid2 / "notes.txt").write_bytes(b"notes")

    tasks = extract_multi_cam_split_tasks(**_default_session_args(tmp_path), primary_camera_keyword="")
    # Should only return the session with video
    assert len(tasks) == 1
    assert len(tasks[0].videos) == 1
