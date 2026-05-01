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
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from cosmos_curator.pipelines.video.utils.video_pipe_input import (
    _multi_cam_session_to_split_task,
    _order_video_paths,
    extract_multi_cam_split_tasks,
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
def test_multi_cam_session_to_split_task(  # noqa: PLR0913
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
