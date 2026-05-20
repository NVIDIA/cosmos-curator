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
"""Video-level comparison planning and result helpers."""

from collections.abc import Mapping, Sequence

import attrs

from cosmos_curator.pipelines.video.output_comparison.report import FeatureComparison, Issue
from cosmos_curator.pipelines.video.output_comparison.summary_loader import OutputRoot
from cosmos_curator.pipelines.video.output_comparison.summary_schema import OutputSummary, ProcessedVideoSummary
from cosmos_curator.pipelines.video.output_comparison.video_schema import ClipComparisonSpec, VideoComparisonSpec

DEFAULT_PROFILE_NAME = "default"


@attrs.define(frozen=True)
class VideoComparisonResult:
    """Comparison results emitted by video-level checks."""

    issues: tuple[Issue, ...]
    feature_comparisons: Mapping[str, FeatureComparison]


def build_video_comparison_specs(  # noqa: PLR0913
    output_a: OutputRoot,
    output_b: OutputRoot,
    summary_a: OutputSummary,
    summary_b: OutputSummary,
    *,
    video_limit: int | None = None,
    selected_video_key: str | None = None,
) -> tuple[VideoComparisonSpec, ...]:
    """Build one comparison spec per video summary key."""
    video_clips_a = _video_clips(summary_a)
    video_clips_b = _video_clips(summary_b)
    video_keys = _video_keys_for_comparison(
        video_clips_a,
        video_clips_b,
        video_limit=video_limit,
        selected_video_key=selected_video_key,
    )
    return tuple(
        VideoComparisonSpec(
            video_key=video_key,
            output_a=str(output_a),
            output_b=str(output_b),
            clips_a=video_clips_a.get(video_key, ()),
            clips_b=video_clips_b.get(video_key, ()),
        )
        for video_key in video_keys
    )


def build_clip_comparison_specs(video_specs: Sequence[VideoComparisonSpec]) -> tuple[ClipComparisonSpec, ...]:
    """Expand video-level specs into clip-level artifact work rows."""
    clip_specs: list[ClipComparisonSpec] = []
    for video_spec in video_specs:
        clips_a = set(video_spec.clips_a)
        clips_b = set(video_spec.clips_b)
        clip_specs.extend(
            ClipComparisonSpec(
                video_key=video_spec.video_key,
                clip_id=clip_id,
                output_a=video_spec.output_a,
                output_b=video_spec.output_b,
                in_a=clip_id in clips_a,
                in_b=clip_id in clips_b,
            )
            for clip_id in sorted(clips_a | clips_b)
        )
    return tuple(clip_specs)


def _video_keys_for_comparison(
    video_clips_a: Mapping[str, tuple[str, ...]],
    video_clips_b: Mapping[str, tuple[str, ...]],
    *,
    video_limit: int | None,
    selected_video_key: str | None,
) -> tuple[str, ...]:
    if selected_video_key is not None and video_limit is not None:
        error_msg = "selected_video_key and video_limit are mutually exclusive"
        raise ValueError(error_msg)
    if selected_video_key is not None:
        known_video_keys = set(video_clips_a) | set(video_clips_b)
        if selected_video_key not in known_video_keys:
            error_msg = f"selected_video_key is not present in either summary: {selected_video_key}"
            raise ValueError(error_msg)
        return (selected_video_key,)
    if video_limit is None:
        return tuple(sorted(set(video_clips_a) | set(video_clips_b)))
    if video_limit < 0:
        error_msg = "video_limit must be greater than or equal to 0"
        raise ValueError(error_msg)
    return tuple(video_clips_a)[:video_limit]


def _video_clips(summary: OutputSummary) -> dict[str, tuple[str, ...]]:
    videos: dict[str, tuple[str, ...]] = {}
    for video_key, video in summary.videos.items():
        match video:
            case ProcessedVideoSummary() as processed_video:
                videos[video_key] = processed_video.clips
            case _:
                videos[video_key] = ()
    return videos
