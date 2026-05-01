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

"""Summary writer for the Ray Data splitting pipeline.

Collects the post-write clip rows to the driver via ``take_all()`` and
aggregates per-video in Python. This mirrors how the Xenna splitting
pipeline builds its summary (driver-side walk of returned tasks) and
avoids a ``groupby`` shuffle operator sitting in the streaming DAG —
Ray Data reserves CPU per operator, so a shuffle reducer pool would
starve the transcode stage even when it has no work yet.
"""

import json
import time
import uuid
from typing import Any

import ray

from cosmos_curator.core.utils.storage.storage_utils import StorageWriter


def _relative_path(full_path: str, input_video_path: str) -> str:
    prefix = input_video_path.rstrip("/") + "/"
    if full_path.startswith(prefix):
        return full_path[len(prefix) :]
    return full_path


def _video_uuid(video_path: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, video_path))


def write_summary(
    ds: ray.data.Dataset,
    *,
    input_video_path: str,
    output_path: str,
    num_input_videos: int,
) -> int:
    """Run the pipeline, aggregate per-video on the driver, and write ``summary.json``.

    Triggers dataset execution via ``take_all()`` on the post-write clip
    rows. Rows are small (``clip_bytes`` is dropped in the writer), so
    driver-side grouping is cheap and avoids the ``groupby`` shuffle.

    Args:
        ds: Dataset after the clip writer stage.
        input_video_path: Base input path; used to derive relative
            keys in the summary (matching Xenna's output shape).
        output_path: Base output path where ``summary.json`` is written.
        num_input_videos: Number of input videos discovered.

    Returns:
        Total number of clips written.

    """
    pipeline_start = time.monotonic()
    clip_rows = ds.take_all()
    pipeline_run_time_minutes = (time.monotonic() - pipeline_start) / 60

    by_video: dict[str, list[dict[str, Any]]] = {}
    for row in clip_rows:
        by_video.setdefault(row["video_path"], []).append(row)

    # Pre-declare top-level totals so they serialize before per-video entries
    # (matches the Xenna summary.json shape). Assignment to existing keys
    # below updates values without changing insertion order.
    summary: dict[str, Any] = {
        "num_input_videos": num_input_videos,
        "num_input_videos_selected": num_input_videos,
        "num_processed_videos": len(by_video),
        "total_video_duration": 0.0,
        "total_clip_duration": 0.0,
        "max_clip_duration": 0.0,
        "pipeline_run_time": pipeline_run_time_minutes,
        "total_video_bytes": 0,
        "total_num_clips_passed": 0,
        "total_num_clips_transcoded": 0,
    }

    total_video_duration = 0.0
    total_clip_duration = 0.0
    max_clip_duration = 0.0
    total_video_bytes = 0
    total_num_clips = 0

    for video_path, clips in by_video.items():
        # Sort by clip_start_s so the emitted clips list is in temporal
        # order — take_all() does not guarantee row order across tasks.
        clips.sort(key=lambda r: r["clip_start_s"])
        duration_s = float(clips[0]["duration_s"])
        video_size = int(clips[0]["video_size"])
        clip_uuids = [r["clip_uuid"] for r in clips]
        clip_durations = [float(r["clip_end_s"]) - float(r["clip_start_s"]) for r in clips]
        num_clips = len(clip_uuids)
        vid_total_clip_duration = sum(clip_durations)
        vid_max_clip_duration = max(clip_durations) if clip_durations else 0.0

        summary[_relative_path(video_path, input_video_path)] = {
            "source_video": video_path,
            "video_uuid": _video_uuid(video_path),
            "num_clip_chunks": 1,
            "num_total_clips": num_clips,
            "clips": clip_uuids,
            "filtered_clips": [],
            "num_clips_passed": num_clips,
            "num_clips_transcoded": num_clips,
        }
        total_video_duration += duration_s
        total_clip_duration += vid_total_clip_duration
        max_clip_duration = max(max_clip_duration, vid_max_clip_duration)
        total_video_bytes += video_size
        total_num_clips += num_clips

    summary["total_video_duration"] = total_video_duration
    summary["total_clip_duration"] = total_clip_duration
    summary["max_clip_duration"] = max_clip_duration
    summary["total_video_bytes"] = total_video_bytes
    summary["total_num_clips_passed"] = total_num_clips
    summary["total_num_clips_transcoded"] = total_num_clips

    writer = StorageWriter(output_path)
    writer.write_str_to("summary.json", json.dumps(summary, indent=4))

    return total_num_clips
