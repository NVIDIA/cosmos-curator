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
import logging
import time
import uuid
from typing import Any

import ray

from cosmos_curator.core.utils.storage.storage_utils import StorageWriter

logger = logging.getLogger(__name__)


def _relative_path(full_path: str, input_video_path: str) -> str:
    prefix = input_video_path.rstrip("/") + "/"
    if full_path.startswith(prefix):
        return full_path[len(prefix) :]
    return full_path


def _video_uuid(video_path: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, video_path))


def _has_caption_stats(rows: list[dict[str, Any]]) -> bool:
    return any(
        "has_caption" in row
        or "num_caption_windows" in row
        or "total_prompt_tokens" in row
        or "total_output_tokens" in row
        for row in rows
    )


def _add_and_log_caption_throughput(summary: dict[str, Any]) -> None:
    total_output = int(summary.get("total_output_tokens", 0))
    total_prompt = int(summary.get("total_prompt_tokens", 0))
    num_caption_windows = int(summary.get("total_num_caption_windows", 0))
    pipeline_run_time = float(summary.get("pipeline_run_time", 0.0))

    if total_output <= 0 or pipeline_run_time <= 0:
        return

    tokens_per_s = round(total_output / (pipeline_run_time * 60), 1)
    summary["output_tokens_per_s"] = tokens_per_s
    avg_prompt = total_prompt // num_caption_windows if num_caption_windows else 0
    avg_output = total_output // num_caption_windows if num_caption_windows else 0
    throughput_message = (
        "\n"
        "  Captioning throughput\n"
        "  -------------------------------------\n"
        f"  total prompt tokens:      {total_prompt:>10,}\n"
        f"  total output tokens:      {total_output:>10,}\n"
        f"  total caption windows:    {num_caption_windows:>10,}\n"
        f"  avg prompt tokens/window: {avg_prompt:>10,}\n"
        f"  avg output tokens/window: {avg_output:>10,}\n"
        f"  output tokens/s:          {tokens_per_s:>10,.1f}\n"
        "  -------------------------------------"
    )
    logger.info(
        "%s",
        throughput_message,
    )


def _add_caption_stats(
    video_summary: dict[str, Any],
    summary: dict[str, Any],
    clips: list[dict[str, Any]],
) -> None:
    num_clips_with_caption = sum(1 for row in clips if row.get("has_caption", False))
    num_caption_windows = sum(int(row.get("num_caption_windows", 0)) for row in clips)
    total_prompt_tokens = sum(int(row.get("total_prompt_tokens", 0)) for row in clips)
    total_output_tokens = sum(int(row.get("total_output_tokens", 0)) for row in clips)
    video_summary["num_clips_with_caption"] = num_clips_with_caption
    video_summary["num_caption_windows"] = num_caption_windows
    summary["total_num_clips_with_caption"] += num_clips_with_caption
    summary["total_num_caption_windows"] += num_caption_windows
    summary["total_prompt_tokens"] += total_prompt_tokens
    summary["total_output_tokens"] += total_output_tokens


def write_summary_from_rows(
    clip_rows: list[dict[str, Any]],
    *,
    input_video_path: str,
    output_path: str,
    num_input_videos: int,
    pipeline_run_time_minutes: float,
) -> int:
    """Aggregate per-video clip rows on the driver and write ``summary.json``.

    Args:
        clip_rows: Small clip rows after clip writing or caption metadata writing.
        input_video_path: Base input path; used to derive relative keys in the
            summary (matching Xenna's output shape).
        output_path: Base output path where ``summary.json`` is written.
        num_input_videos: Number of input videos discovered.
        pipeline_run_time_minutes: Dataset execution time in minutes.

    Returns:
        Total number of clips written.

    """
    include_caption_stats = _has_caption_stats(clip_rows)

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

    if include_caption_stats:
        summary["total_num_clips_with_caption"] = 0
        summary["total_num_caption_windows"] = 0
        summary["total_prompt_tokens"] = 0
        summary["total_output_tokens"] = 0

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

        video_summary = {
            "source_video": video_path,
            "video_uuid": _video_uuid(video_path),
            "num_clip_chunks": 1,
            "num_total_clips": num_clips,
            "clips": clip_uuids,
            "filtered_clips": [],
            "num_clips_passed": num_clips,
            "num_clips_transcoded": num_clips,
        }

        if include_caption_stats:
            _add_caption_stats(video_summary, summary, clips)

        summary[_relative_path(video_path, input_video_path)] = video_summary
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
    if include_caption_stats:
        _add_and_log_caption_throughput(summary)

    writer = StorageWriter(output_path)
    writer.write_str_to("summary.json", json.dumps(summary, indent=4))

    return total_num_clips


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

    return write_summary_from_rows(
        clip_rows,
        input_video_path=input_video_path,
        output_path=output_path,
        num_input_videos=num_input_videos,
        pipeline_run_time_minutes=pipeline_run_time_minutes,
    )
