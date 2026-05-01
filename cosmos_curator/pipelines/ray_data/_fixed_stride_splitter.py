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

"""Fixed-stride clip splitter for Ray Data pipelines.

Computes clip spans from video duration using a sliding window and stores
them as list columns on the video row (no fan-out).
"""

import uuid
from collections.abc import Callable
from typing import Any


def _compute_spans(
    duration_s: float,
    clip_len_s: float,
    clip_stride_s: float,
    min_clip_length_s: float,
    limit_clips: int = 0,
) -> list[tuple[float, float]]:
    """Compute fixed-stride clip spans from a video duration.

    Args:
        duration_s: Video duration in seconds.
        clip_len_s: Clip length in seconds.
        clip_stride_s: Stride between clip starts in seconds.
        min_clip_length_s: Minimum clip length; shorter clips are dropped.
        limit_clips: Maximum number of clips (0 = unlimited).

    Returns:
        List of (start_s, end_s) tuples.

    """
    spans: list[tuple[float, float]] = []
    start_s = 0.0

    while start_s < duration_s:
        end_s = min(start_s + clip_len_s, duration_s)
        if (end_s - start_s) >= min_clip_length_s:
            spans.append((start_s, end_s))
        start_s += clip_stride_s

    if limit_clips > 0:
        spans = spans[:limit_clips]

    return spans


def make_split_fn(
    clip_len_s: float,
    clip_stride_s: float,
    min_clip_length_s: float,
    limit_clips: int = 0,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a ``map`` function that computes clip spans for a video row.

    Spans are stored as parallel list columns (``clip_uuids``,
    ``clip_starts``, ``clip_ends``) on the same row — no fan-out.
    The downstream transcoder performs the fan-out after transcoding.

    Args:
        clip_len_s: Clip length in seconds.
        clip_stride_s: Stride between clip starts in seconds.
        min_clip_length_s: Minimum clip length; shorter clips are dropped.
        limit_clips: Maximum number of clips per video (0 = unlimited).

    Returns:
        A function suitable for ``ray.data.Dataset.map``.

    """
    assert clip_stride_s > 0, f"clip_stride_s must be positive, got {clip_stride_s}"

    def _split(row: dict[str, Any]) -> dict[str, Any]:
        video_path: str = row["video_path"]
        duration_s: float = row["duration_s"]

        spans = _compute_spans(duration_s, clip_len_s, clip_stride_s, min_clip_length_s, limit_clips)

        clip_uuids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_path}_{s}_{e}")) for s, e in spans]
        clip_starts = [s for s, _ in spans]
        clip_ends = [e for _, e in spans]

        return {
            **row,
            "clip_uuids": clip_uuids,
            "clip_starts": clip_starts,
            "clip_ends": clip_ends,
        }

    return _split
