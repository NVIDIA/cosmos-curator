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
"""Build the ``pa.Table`` of clip rows that flows into each stage.

A clip is "discovered" when it appears in at least one output's processed video summary.
Output roots and the storage profile are run-wide and travel on the stage actors;
clip rows carry only what varies per clip:

- ``clip_id``: stable uuid from the split pipeline
- ``video_key``: which input video the clip came from (needed for issue grouping)
- ``in_a`` / ``in_b``: presence flags (a clip can be one-sided)
- ``artifact_kind``: ``"clip"`` (passed filters) vs ``"filtered_clip"`` (filtered out)

Videos that are present on only one output, or are unprocessed on one output, contribute
no clip rows -- that's a summary-level mismatch caught by ``summary_compare``.
"""

import pyarrow as pa

from cosmos_curator.pipelines.video.split_comparison.summary_schema import (
    OutputSummary,
    ProcessedVideoSummary,
)

CLIP_ROW_SCHEMA: pa.Schema = pa.schema(
    [
        ("clip_id", pa.string()),
        ("video_key", pa.string()),
        ("in_a", pa.bool_()),
        ("in_b", pa.bool_()),
        ("artifact_kind", pa.dictionary(pa.int8(), pa.string())),
    ],
)


def discover_clips(summary_a: OutputSummary, summary_b: OutputSummary) -> pa.Table:
    """Return a ``pa.Table`` of clip rows for videos present in both outputs.

    One row per (video, clip, artifact_kind) with ``in_a`` / ``in_b`` reflecting
    presence. Videos present on only one output, or unprocessed on one output,
    contribute nothing -- those are summary-level mismatches caught by
    ``summary_compare``.
    """
    rows: list[dict[str, object]] = []
    shared_keys = sorted(set(summary_a.videos) & set(summary_b.videos))
    for video_key in shared_keys:
        video_a, video_b = summary_a.videos[video_key], summary_b.videos[video_key]
        if not (video_a.processed and video_b.processed):
            # Processed-on-only-one-output is a summary-level mismatch; no per-clip work.
            continue
        rows.extend(_clip_rows_for_video(video_key, video_a, video_b))
    return pa.Table.from_pylist(rows, schema=CLIP_ROW_SCHEMA)


def _clip_rows_for_video(
    video_key: str,
    video_a: ProcessedVideoSummary,
    video_b: ProcessedVideoSummary,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows.extend(_paired_rows(video_key, video_a.clips, video_b.clips, artifact_kind="clip"))
    rows.extend(
        _paired_rows(video_key, video_a.filtered_clips, video_b.filtered_clips, artifact_kind="filtered_clip"),
    )
    return rows


def _paired_rows(
    video_key: str,
    output_a: tuple[str, ...],
    output_b: tuple[str, ...],
    *,
    artifact_kind: str,
) -> list[dict[str, object]]:
    set_a = set(output_a)
    set_b = set(output_b)
    return [
        {
            "clip_id": clip_id,
            "video_key": video_key,
            "in_a": clip_id in set_a,
            "in_b": clip_id in set_b,
            "artifact_kind": artifact_kind,
        }
        for clip_id in sorted(set_a | set_b)
    ]
