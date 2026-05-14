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
"""Shared helpers for output comparison tests."""

import json
from pathlib import Path
from typing import Any


def video_summary(
    *,
    video_uuid: str = "video-uuid",
    clips: list[str] | None = None,
    filtered_clips: list[str] | None = None,
    num_total_clips: int = 3,
) -> dict[str, Any]:
    """Build a representative per-video summary entry."""
    return {
        "source_video": "/inputs/video.mp4",
        "processed": True,
        "video_uuid": video_uuid,
        "num_clip_chunks": 1,
        "num_total_clips": num_total_clips,
        "num_clips_filtered_by_motion": 0,
        "num_clips_filtered_by_aesthetic": 0,
        "num_clips_filtered_by_qwen_classifier": 0,
        "num_clips_filtered_by_qwen_semantic": 0,
        "num_clips_filtered_by_artificial_text": 0,
        "num_clips_passed": 2,
        "num_clips_transcoded": 2,
        "num_clips_with_embeddings": 2,
        "num_clips_with_caption": 2,
        "num_caption_windows": 4,
        "num_clips_with_webp": 2,
        "clips": clips if clips is not None else ["clip-a", "clip-b"],
        "filtered_clips": filtered_clips if filtered_clips is not None else ["clip-filtered"],
    }


def summary(**overrides: object) -> dict[str, Any]:
    """Build a representative split pipeline summary with optional field overrides."""
    summary = {
        "num_input_videos": 1,
        "num_input_videos_selected": 1,
        "num_processed_videos": 1,
        "embedding_algorithm": "internvideo2",
        "total_video_duration": 10.0,
        "total_clip_duration": 8.0,
        "max_clip_duration": 4.0,
        "total_video_bytes": 12345,
        "num_remuxed_videos": 0,
        "total_num_clips_filtered_by_motion": 0,
        "total_num_clips_filtered_by_aesthetic": 0,
        "total_num_clips_filtered_by_qwen_classifier": 0,
        "total_num_clips_filtered_by_qwen_semantic": 0,
        "total_num_clips_filtered_by_artificial_text": 0,
        "total_num_clips_passed": 2,
        "total_num_clips_transcoded": 2,
        "total_num_clips_with_embeddings": 2,
        "total_num_clips_with_caption": 2,
        "total_num_caption_windows": 4,
        "total_num_clips_with_webp": 2,
        "total_prompt_tokens": 100,
        "total_output_tokens": 50,
        "video.mp4": video_summary(),
    }
    summary.update(overrides)
    return summary


def write_summary(output_root: Path, summary: dict[str, Any]) -> None:
    """Write a summary JSON file under an output root."""
    output_root.mkdir()
    (output_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
