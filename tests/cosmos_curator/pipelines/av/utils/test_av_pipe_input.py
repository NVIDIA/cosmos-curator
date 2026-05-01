# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for AV pipeline input helpers."""

import json
from pathlib import Path

import pytest

from cosmos_curator.core.utils.storage.s3_client import S3Prefix
from cosmos_curator.pipelines.av.utils.av_data_info import SQLITE_DB_NAME
from cosmos_curator.pipelines.av.utils.av_pipe_input import (
    _get_video_sessions,
    _is_keyword_in_caption,
    _summarize_video_session,
    _to_s3_or_path,
    _worker_verify_processed_sessions,
    is_pcd_file,
    is_sqlite_file,
    is_video_file,
    read_session_file,
    write_summary,
)


def _create_processed_session(
    root: Path,
    session_name: str,
    chunk_clips: list[list[str]],
    *,
    duration: float,
    dimensions: tuple[int, int] = (720, 1280),
) -> None:
    """Build a minimal processed session layout for exercising summary logic."""
    processed_sessions_dir = root / "processed_sessions"
    processed_session_chunks_dir = root / "processed_session_chunks"
    processed_sessions_dir.mkdir(parents=True, exist_ok=True)
    processed_session_chunks_dir.mkdir(parents=True, exist_ok=True)

    height, width = dimensions
    session_payload = {
        "video_session_name": session_name,
        "num_session_chunks": len(chunk_clips),
        "height": height,
        "width": width,
    }
    (processed_sessions_dir / f"{session_name}.json").write_text(json.dumps(session_payload))

    for idx, clips in enumerate(chunk_clips):
        chunk_payload = {
            "video_session_name": session_name,
            "session_chunk_index": idx,
            "clips": clips,
        }
        if idx == 0:
            chunk_payload["source_video_duration_s"] = duration
        (processed_session_chunks_dir / f"{session_name}_{idx}.json").write_text(json.dumps(chunk_payload))


# Utility function tests


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("data.pcd", True),
        ("pointcloud.PCD", False),
        ("data.txt", False),
        ("data.pcd.backup", False),
    ],
)
def test_is_pcd_file(filename: str, expected: bool) -> None:  # noqa: FBT001
    """PCD file detection checks .pcd extension."""
    assert is_pcd_file(filename) == expected


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("video.mp4", True),
        ("video.h264", True),
        ("video.MP4", False),
        ("video.avi", False),
        ("video.h264.bak", False),
    ],
)
def test_is_video_file(filename: str, expected: bool) -> None:  # noqa: FBT001
    """Video file detection checks .mp4 and .h264 extensions."""
    assert is_video_file(filename) == expected


def test_is_sqlite_file() -> None:
    """SQLite file detection checks exact database name."""
    assert is_sqlite_file(f"path/to/{SQLITE_DB_NAME}")
    assert is_sqlite_file(SQLITE_DB_NAME)
    assert not is_sqlite_file("other.db")
    assert not is_sqlite_file(f"{SQLITE_DB_NAME}.backup")


def test_to_s3_or_path_local(tmp_path: Path) -> None:
    """Local paths convert to Path objects."""
    local_path = tmp_path / "local" / "path"
    result = _to_s3_or_path(str(local_path))
    assert isinstance(result, Path)
    assert str(result) == str(local_path)


def test_to_s3_or_path_s3() -> None:
    """S3 URIs convert to S3Prefix objects."""
    result = _to_s3_or_path("s3://bucket/key")
    assert isinstance(result, S3Prefix)


@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        ("Driving to the airport for pick up", True),
        ("Airport pick-up zone", True),
        ("Airport parking lot", True),
        ("Airport terminal drop-off area", True),
        ("Airport drop off", True),
        ("Airport runway", False),  # has "airport" but not pickup/parking/drop-off
        ("Picking up groceries", False),  # has "pick up" but not "airport"
        ("International Airport", False),  # just "airport", no relevant keywords
    ],
)
def test_is_keyword_in_caption_airport(caption: str, expected: bool) -> None:  # noqa: FBT001
    """Airport keyword matching requires both 'airport' and pickup/parking/drop-off terms."""
    assert _is_keyword_in_caption(caption, "airport_pick_up_drop_off") == expected


def test_is_keyword_in_caption_unknown_keyword() -> None:
    """Unknown keywords return False."""
    assert not _is_keyword_in_caption("any caption", "unknown_keyword")


# Session file tests


def test_read_session_file_handles_none() -> None:
    """None session_filepath returns empty list."""
    assert read_session_file(None) == []


def test_read_session_file_skips_non_video_lines(tmp_path: Path) -> None:
    """Only video files contribute to session extraction."""
    session_file = tmp_path / "sessions.txt"
    lines = [
        "inputs/session-a/readme.txt",
        "inputs/session-a/notes.md",
        "inputs/session-b/video.mp4",
    ]
    session_file.write_text("\n".join(lines))

    sessions = read_session_file(str(session_file))

    assert sessions == ["inputs/session-b"]


def test_read_session_file_returns_unique_video_sessions(tmp_path: Path) -> None:
    """Session manifest with duplicated directories yields unique session names."""
    session_file = tmp_path / "sessions.txt"
    lines = [
        "inputs/session-a/video1.mp4",
        "inputs/session-a/video2.h264",
        "inputs/session-b/notes.txt",
        "inputs/session-b/camera1.mp4",
    ]
    session_file.write_text("\n".join(lines))

    sessions = read_session_file(str(session_file))

    assert set(sessions) == {"inputs/session-a", "inputs/session-b"}


def test_get_video_sessions_counts_videos(tmp_path: Path) -> None:
    """Video discovery counts camera files per session directory."""
    input_dir = tmp_path / "inputs"
    (input_dir / "session-a").mkdir(parents=True)
    (input_dir / "session-b").mkdir(parents=True)

    (input_dir / "session-a" / "front.mp4").write_text("data")
    (input_dir / "session-a" / "rear.h264").write_text("data")
    (input_dir / "session-a" / "notes.txt").write_text("ignored")
    (input_dir / "session-b" / "cam.mp4").write_text("data")

    sessions = _get_video_sessions(str(input_dir), verbose=False)

    assert sessions == {"session-a": 2, "session-b": 1}


def test_worker_verify_processed_sessions_checks_missing_chunks(tmp_path: Path) -> None:
    """Worker verifies chunk completeness and flags missing chunk files."""
    output_dir = tmp_path
    processed_sessions_dir = output_dir / "processed_sessions"
    processed_sessions_dir.mkdir()
    processed_chunks_dir = output_dir / "processed_session_chunks"
    processed_chunks_dir.mkdir()

    (processed_sessions_dir / "session-a.json").write_text(json.dumps({"num_session_chunks": 2}))
    (processed_chunks_dir / "session-a_0.json").write_text(json.dumps({}))
    (processed_chunks_dir / "session-a_1.json").write_text(json.dumps({}))

    assert _worker_verify_processed_sessions("session-a.json", str(output_dir), None) == "session-a.json"

    (processed_sessions_dir / "session-b.json").write_text(json.dumps({"num_session_chunks": 1}))

    assert _worker_verify_processed_sessions("session-b.json", str(output_dir), None) is None


def test_summarize_video_session_combines_chunks(tmp_path: Path) -> None:
    """Summaries gather metadata and flatten all clip identifiers."""
    output_dir = tmp_path
    _create_processed_session(
        output_dir,
        "session-a",
        [["clip-1", "clip-2"], ["clip-3"]],
        duration=12.5,
        dimensions=(960, 1280),
    )

    summary = _summarize_video_session("session-a.json", str(output_dir), None)

    assert summary is not None
    assert summary["session_name"] == "session-a"
    assert summary["source_video_duration_s"] == pytest.approx(12.5)
    assert summary["height"] == 960
    assert summary["width"] == 1280
    assert summary["clips"] == ["clip-1", "clip-2", "clip-3"]


def test_summarize_video_session_handles_missing_json(tmp_path: Path) -> None:
    """Gracefully handles missing session file."""
    summary = _summarize_video_session("nonexistent.json", str(tmp_path), None)
    assert summary is None


def test_write_summary_emits_summary_json(tmp_path: Path) -> None:
    """write_summary aggregates totals and writes the summary manifest."""
    output_dir = tmp_path
    _create_processed_session(
        output_dir,
        "session-a",
        [["clip-1"], ["clip-2"]],
        duration=7.0,
    )
    _create_processed_session(
        output_dir,
        "session-b",
        [["clip-3", "clip-4"]],
        duration=4.5,
        dimensions=(1080, 1920),
    )

    total_length = write_summary(str(output_dir), num_threads=2)

    assert total_length == pytest.approx(11.5)

    summary_path = output_dir / "summary.json"
    assert summary_path.exists()
    summary_data = json.loads(summary_path.read_text())

    assert summary_data["session-a"] == {
        "source_video_duration_s": pytest.approx(7.0),
        "height": 720,
        "width": 1280,
        "clips": ["clip-1", "clip-2"],
    }
    assert summary_data["session-b"] == {
        "source_video_duration_s": pytest.approx(4.5),
        "height": 1080,
        "width": 1920,
        "clips": ["clip-3", "clip-4"],
    }
