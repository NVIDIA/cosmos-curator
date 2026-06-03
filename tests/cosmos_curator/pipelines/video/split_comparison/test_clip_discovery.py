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
"""Tests for clip discovery: walking OutputSummary into a clip row table."""

from collections.abc import Mapping
from typing import Any

from cosmos_curator.pipelines.video.split_comparison.clip_discovery import (
    CLIP_ROW_SCHEMA,
    discover_clips,
)
from cosmos_curator.pipelines.video.split_comparison.summary_schema import (
    OutputSummary,
    ProcessedVideoSummary,
    UnprocessedVideoSummary,
    VideoSummary,
)


def _processed_video(
    key: str,
    *,
    clips: tuple[str, ...] = (),
    filtered_clips: tuple[str, ...] = (),
) -> ProcessedVideoSummary:
    return ProcessedVideoSummary(
        source_video=f"/inputs/{key}",
        video_uuid=f"uuid-{key}",
        num_clip_chunks=1,
        num_total_clips=len(clips) + len(filtered_clips),
        num_clips_filtered_by_motion=0,
        num_clips_filtered_by_aesthetic=0,
        num_clips_filtered_by_qwen_classifier=0,
        num_clips_filtered_by_qwen_semantic=0,
        num_clips_filtered_by_artificial_text=0,
        num_clips_passed=len(clips),
        num_clips_transcoded=len(clips) + len(filtered_clips),
        num_clips_with_embeddings=len(clips),
        num_clips_with_caption=0,
        num_caption_windows=0,
        num_clips_with_webp=len(clips),
        clips=clips,
        filtered_clips=filtered_clips,
    )


def _unprocessed_video(key: str) -> UnprocessedVideoSummary:
    return UnprocessedVideoSummary(processed=False, source_video=f"/inputs/{key}")


def _summary(
    videos: Mapping[str, VideoSummary],
    **overrides: Any,  # noqa: ANN401 -- test fixture, fields are heterogeneous OutputSummary kwargs
) -> OutputSummary:
    """Build a minimal OutputSummary populating only the fields discovery walks."""
    base: dict[str, Any] = {
        "videos": dict(videos),
        "num_input_videos": len(videos),
        "num_input_videos_selected": len(videos),
        "num_processed_videos": sum(1 for v in videos.values() if v.processed),
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
        "total_num_clips_passed": 0,
        "total_num_clips_transcoded": 0,
        "total_num_clips_with_embeddings": 0,
        "total_num_clips_with_caption": 0,
        "total_num_caption_windows": 0,
        "total_num_clips_with_webp": 0,
        "total_prompt_tokens": 0,
        "total_output_tokens": 0,
    }
    base.update(overrides)
    return OutputSummary(**base)


def test_discover_clips_table_uses_canonical_schema() -> None:
    """The output table always carries CLIP_ROW_SCHEMA exactly."""
    summary_a = _summary({"video.mp4": _processed_video("video.mp4")})
    summary_b = _summary({"video.mp4": _processed_video("video.mp4")})

    table = discover_clips(summary_a, summary_b)

    assert table.schema == CLIP_ROW_SCHEMA
    assert table.num_rows == 0


def test_discover_clips_emits_paired_passed_and_filtered_rows() -> None:
    """Each (video, clip_id, artifact_kind) becomes one row with in_a/in_b set."""
    summary_a = _summary(
        {
            "video.mp4": _processed_video(
                "video.mp4",
                clips=("clip-shared", "clip-a-only"),
                filtered_clips=("filtered-shared",),
            ),
        },
    )
    summary_b = _summary(
        {
            "video.mp4": _processed_video(
                "video.mp4",
                clips=("clip-shared", "clip-b-only"),
                filtered_clips=("filtered-shared", "filtered-b-only"),
            ),
        },
    )

    rows = discover_clips(summary_a, summary_b).to_pylist()

    expected = [
        {"clip_id": "clip-a-only", "video_key": "video.mp4", "in_a": True, "in_b": False, "artifact_kind": "clip"},
        {"clip_id": "clip-b-only", "video_key": "video.mp4", "in_a": False, "in_b": True, "artifact_kind": "clip"},
        {"clip_id": "clip-shared", "video_key": "video.mp4", "in_a": True, "in_b": True, "artifact_kind": "clip"},
        {
            "clip_id": "filtered-b-only",
            "video_key": "video.mp4",
            "in_a": False,
            "in_b": True,
            "artifact_kind": "filtered_clip",
        },
        {
            "clip_id": "filtered-shared",
            "video_key": "video.mp4",
            "in_a": True,
            "in_b": True,
            "artifact_kind": "filtered_clip",
        },
    ]
    assert rows == expected


def test_discover_clips_skips_videos_present_on_only_one_side() -> None:
    """One-sided videos are summary-level mismatches, not per-clip work."""
    summary_a = _summary({"a-only.mp4": _processed_video("a-only.mp4", clips=("c1",))})
    summary_b = _summary({"b-only.mp4": _processed_video("b-only.mp4", clips=("c2",))})

    rows = discover_clips(summary_a, summary_b).to_pylist()

    assert rows == []


def test_discover_clips_skips_videos_unprocessed_on_either_side() -> None:
    """If one output reports the video as unprocessed, there's no clip work to do."""
    summary_a = _summary({"video.mp4": _processed_video("video.mp4", clips=("c1",))})
    summary_b = _summary({"video.mp4": _unprocessed_video("video.mp4")})

    rows = discover_clips(summary_a, summary_b).to_pylist()

    assert rows == []


def test_discover_clips_preserves_artifact_kind_dictionary_encoding() -> None:
    """artifact_kind column is dictionary-encoded so the two finite values compress."""
    summary = _summary(
        {
            "video.mp4": _processed_video(
                "video.mp4",
                clips=("clip-a",),
                filtered_clips=("filtered-a",),
            ),
        },
    )

    table = discover_clips(summary, summary)

    assert table.schema.field("artifact_kind").type == CLIP_ROW_SCHEMA.field("artifact_kind").type


def test_discover_clips_rows_sorted_by_clip_id_within_kind() -> None:
    """Row order within a (video, kind) group is sorted by clip_id for deterministic output."""
    summary_a = _summary(
        {
            "video.mp4": _processed_video(
                "video.mp4",
                clips=("zz-clip", "aa-clip", "mm-clip"),
            ),
        },
    )
    summary_b = _summary({"video.mp4": _processed_video("video.mp4", clips=("aa-clip", "mm-clip", "zz-clip"))})

    rows = discover_clips(summary_a, summary_b).to_pylist()

    assert [row["clip_id"] for row in rows] == ["aa-clip", "mm-clip", "zz-clip"]
