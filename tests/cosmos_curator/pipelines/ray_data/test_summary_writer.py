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

"""Tests for the Ray Data splitting-pipeline summary writer."""

import json
from pathlib import Path

import ray

from cosmos_curator.pipelines.ray_data._summary_writer import (
    _relative_path,
    _video_uuid,
    write_summary,
    write_summary_from_rows,
)


def test_relative_path_strips_prefix() -> None:
    """Strip the input prefix; leave non-matching paths untouched."""
    assert _relative_path("s3://bucket/raw/a/b.mp4", "s3://bucket/raw") == "a/b.mp4"
    assert _relative_path("s3://bucket/raw/a/b.mp4", "s3://bucket/raw/") == "a/b.mp4"
    assert _relative_path("s3://other/a.mp4", "s3://bucket/raw") == "s3://other/a.mp4"


def test_write_summary_aggregates_across_videos(tmp_path: Path) -> None:
    """End-to-end: build a clip-row dataset, aggregate on the driver, write summary.json."""
    input_video_path = "s3://bucket/raw"
    rows = [
        {
            "video_path": "s3://bucket/raw/a.mp4",
            "video_size": 1000,
            "duration_s": 30.0,
            "clip_uuid": "a-1",
            "clip_start_s": 0.0,
            "clip_end_s": 10.0,
            "clip_location": str(tmp_path / "clips" / "a-1.mp4"),
        },
        {
            "video_path": "s3://bucket/raw/a.mp4",
            "video_size": 1000,
            "duration_s": 30.0,
            "clip_uuid": "a-2",
            "clip_start_s": 10.0,
            "clip_end_s": 20.0,
            "clip_location": str(tmp_path / "clips" / "a-2.mp4"),
        },
        {
            "video_path": "s3://bucket/raw/b.mp4",
            "video_size": 500,
            "duration_s": 5.0,
            "clip_uuid": "b-1",
            "clip_start_s": 0.0,
            "clip_end_s": 5.0,
            "clip_location": str(tmp_path / "clips" / "b-1.mp4"),
        },
    ]

    ds = ray.data.from_items(rows)

    num_clips = write_summary(
        ds,
        input_video_path=input_video_path,
        output_path=str(tmp_path),
        num_input_videos=2,
    )

    assert num_clips == 3

    summary = json.loads((tmp_path / "summary.json").read_text())

    assert summary["num_input_videos"] == 2
    assert summary["num_input_videos_selected"] == 2
    assert summary["num_processed_videos"] == 2
    assert summary["total_video_duration"] == 35.0
    assert summary["total_clip_duration"] == 25.0
    assert summary["max_clip_duration"] == 10.0
    assert summary["total_video_bytes"] == 1500
    assert summary["total_num_clips_passed"] == 3
    assert summary["total_num_clips_transcoded"] == 3
    assert summary["pipeline_run_time"] >= 0

    a = summary["a.mp4"]
    assert a == {
        "source_video": "s3://bucket/raw/a.mp4",
        "video_uuid": _video_uuid("s3://bucket/raw/a.mp4"),
        "num_clip_chunks": 1,
        "num_total_clips": 2,
        "clips": ["a-1", "a-2"],
        "filtered_clips": [],
        "num_clips_passed": 2,
        "num_clips_transcoded": 2,
    }

    b = summary["b.mp4"]
    assert b["num_total_clips"] == 1
    assert b["clips"] == ["b-1"]


def test_write_summary_from_rows_aggregates_caption_stats(tmp_path: Path) -> None:
    """Caption-aware clip rows emit Xenna-style token totals."""
    rows = [
        {
            "video_path": "s3://bucket/raw/a.mp4",
            "video_size": 1000,
            "duration_s": 30.0,
            "clip_uuid": "a-1",
            "clip_start_s": 0.0,
            "clip_end_s": 10.0,
            "clip_location": str(tmp_path / "clips" / "a-1.mp4"),
            "has_caption": True,
            "num_caption_windows": 2,
            "total_prompt_tokens": 30,
            "total_output_tokens": 12,
        },
        {
            "video_path": "s3://bucket/raw/a.mp4",
            "video_size": 1000,
            "duration_s": 30.0,
            "clip_uuid": "a-2",
            "clip_start_s": 10.0,
            "clip_end_s": 20.0,
            "clip_location": str(tmp_path / "clips" / "a-2.mp4"),
            "has_caption": False,
            "num_caption_windows": 0,
            "total_prompt_tokens": 0,
            "total_output_tokens": 0,
        },
    ]

    num_clips = write_summary_from_rows(
        rows,
        input_video_path="s3://bucket/raw",
        output_path=str(tmp_path),
        num_input_videos=1,
        pipeline_run_time_minutes=0.5,
    )

    assert num_clips == 2
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["total_num_clips_with_caption"] == 1
    assert summary["total_num_caption_windows"] == 2
    assert summary["total_prompt_tokens"] == 30
    assert summary["total_output_tokens"] == 12
    assert summary["output_tokens_per_s"] == 0.4
    assert summary["a.mp4"]["num_clips_with_caption"] == 1
    assert summary["a.mp4"]["num_caption_windows"] == 2
