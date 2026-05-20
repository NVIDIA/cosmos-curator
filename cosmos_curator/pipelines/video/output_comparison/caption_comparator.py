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
"""Caption implementation of the video output feature comparator."""

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any, cast

import attrs

from cosmos_curator.pipelines.video.output_comparison.caption_compare import compare_caption_clip_view
from cosmos_curator.pipelines.video.output_comparison.caption_loader import caption_view_from_clip_artifacts
from cosmos_curator.pipelines.video.output_comparison.caption_policy import (
    DEFAULT_CAPTION_POLICY,
    CaptionComparisonPolicy,
)
from cosmos_curator.pipelines.video.output_comparison.caption_reduce import (
    caption_presence_mismatch_result,
    empty_caption_result,
    reduce_caption_clip_results,
)
from cosmos_curator.pipelines.video.output_comparison.caption_result import (
    CAPTIONS_FEATURE_NAME,
    CaptionClipCompareResult,
    CaptionComparisonResult,
)
from cosmos_curator.pipelines.video.output_comparison.caption_schema import CaptionComparisonCounts, ClipCaptionView
from cosmos_curator.pipelines.video.output_comparison.compare_features import (
    ClipFeaturePlan,
    FeatureComparisonContext,
    FeatureComparisonPlan,
    FeatureComparisonResult,
    ResolvedFeaturePlan,
)
from cosmos_curator.pipelines.video.output_comparison.json_types import JsonDictObject, JsonValue
from cosmos_curator.pipelines.video.output_comparison.summary_schema import OutputSummary, ProcessedVideoSummary
from cosmos_curator.pipelines.video.output_comparison.video_artifacts import load_clip_artifacts
from cosmos_curator.pipelines.video.output_comparison.video_planning import build_clip_comparison_specs
from cosmos_curator.pipelines.video.output_comparison.video_schema import ClipComparisonSpec, VideoComparisonSpec


@attrs.define(frozen=True)
class _CaptionComparisonWorkload:
    """Driver-built caption comparison workload."""

    output_a_has_caption_records: bool
    output_b_has_caption_records: bool
    summary_counts: CaptionComparisonCounts
    videos_only_in_a: tuple[str, ...]
    videos_only_in_b: tuple[str, ...]
    video_keys_a: frozenset[str]
    video_keys_b: frozenset[str]

    def caption_video_specs(self, specs: Sequence[VideoComparisonSpec]) -> tuple[VideoComparisonSpec, ...]:
        """Return video specs that may contain caption records on either side."""
        caption_video_keys = self.video_keys_a | self.video_keys_b
        return tuple(spec for spec in specs if spec.video_key in caption_video_keys)


class CaptionClipViewLoadWorker:
    """Callable Ray actor worker that loads metadata and builds caption views."""

    def __init__(self, profile_name: str, policy: CaptionComparisonPolicy = DEFAULT_CAPTION_POLICY) -> None:
        """Create a caption-view loader with per-actor storage client caching."""
        self._profile_name = profile_name
        self._policy = policy
        self._client_params_by_output_root: dict[tuple[str, str], Mapping[str, Any]] = {}

    def __call__(self, row: Mapping[str, JsonValue]) -> JsonDictObject:
        """Load one clip's metadata and return its normalized caption view row."""
        artifacts = load_clip_artifacts(
            ClipComparisonSpec.from_json_dict(row),
            profile_name=self._profile_name,
            metadata_version=self._policy.metadata_version,
            client_params_by_output_root=self._client_params_by_output_root,
        )
        return caption_view_from_clip_artifacts(artifacts, policy=self._policy).to_json_dict()


@attrs.define(frozen=True)
class CaptionFeatureComparator:
    """Compare caption structure using loaded per-clip metadata."""

    policy: CaptionComparisonPolicy = DEFAULT_CAPTION_POLICY

    @property
    def name(self) -> str:
        """Return the report feature name."""
        return CAPTIONS_FEATURE_NAME

    def build_plan(self, context: FeatureComparisonContext) -> FeatureComparisonPlan:
        """Build a caption comparison plan from loaded summaries."""
        caption_videos_a = _caption_video_clips(context.summary_a)
        caption_videos_b = _caption_video_clips(context.summary_b)
        if context.video_limit is not None or context.selected_video_key is not None:
            selected_video_keys = frozenset(spec.video_key for spec in context.specs)
            caption_videos_a = _filter_caption_videos(caption_videos_a, selected_video_keys)
            caption_videos_b = _filter_caption_videos(caption_videos_b, selected_video_keys)
            output_a_has_caption_records = bool(caption_videos_a)
            output_b_has_caption_records = bool(caption_videos_b)
            summary_counts = _selected_summary_counts(
                selected_video_keys,
                context.summary_a,
                context.summary_b,
            )
        else:
            output_a_has_caption_records = _has_caption_records(context.summary_a, caption_videos_a)
            output_b_has_caption_records = _has_caption_records(context.summary_b, caption_videos_b)
            summary_counts = _summary_counts(
                caption_videos_a,
                caption_videos_b,
                context.summary_a,
                context.summary_b,
            )
        video_keys_a = frozenset(caption_videos_a)
        video_keys_b = frozenset(caption_videos_b)
        caption_workload = _CaptionComparisonWorkload(
            output_a_has_caption_records=output_a_has_caption_records,
            output_b_has_caption_records=output_b_has_caption_records,
            summary_counts=summary_counts,
            videos_only_in_a=tuple(sorted(video_keys_a - video_keys_b)),
            videos_only_in_b=tuple(sorted(video_keys_b - video_keys_a)),
            video_keys_a=video_keys_a,
            video_keys_b=video_keys_b,
        )

        if not caption_workload.output_a_has_caption_records and not caption_workload.output_b_has_caption_records:
            return ResolvedFeaturePlan(self.name, _feature_result(empty_caption_result()))
        if caption_workload.output_a_has_caption_records != caption_workload.output_b_has_caption_records:
            return ResolvedFeaturePlan(
                self.name,
                _feature_result(
                    caption_presence_mismatch_result(
                        a_has_caption_records=caption_workload.output_a_has_caption_records,
                        b_has_caption_records=caption_workload.output_b_has_caption_records,
                        counts=caption_workload.summary_counts,
                    )
                ),
            )
        clip_specs = build_clip_comparison_specs(caption_workload.caption_video_specs(context.specs))
        return ClipFeaturePlan(
            feature_name=self.name,
            clip_specs=clip_specs,
            load_worker_class=CaptionClipViewLoadWorker,
            load_worker_constructor_kwargs={"profile_name": context.profile_name, "policy": self.policy},
            compare_row=cast(
                "Callable[[Mapping[str, JsonValue]], JsonDictObject]",
                partial(self._compare_clip_row, caption_workload),
            ),
            reduce_rows=cast(
                "Callable[[Sequence[Mapping[str, JsonValue]]], FeatureComparisonResult]",
                partial(self._reduce_clip_rows, caption_workload),
            ),
        )

    def _compare_clip_row(self, workload: _CaptionComparisonWorkload, row: Mapping[str, JsonValue]) -> JsonDictObject:
        """Compare one normalized caption clip view row."""
        return self.compare_clip_view(ClipCaptionView.from_json_dict(row), workload)

    def compare_clip_view(self, view: ClipCaptionView, workload: _CaptionComparisonWorkload) -> JsonDictObject:
        """Compare caption structure for one normalized clip view."""
        return compare_caption_clip_view(
            view,
            a_has_caption_records=view.video_key in workload.video_keys_a,
            b_has_caption_records=view.video_key in workload.video_keys_b,
        ).to_json_dict()

    def _reduce_clip_rows(
        self,
        workload: _CaptionComparisonWorkload,
        rows: Sequence[Mapping[str, JsonValue]],
    ) -> FeatureComparisonResult:
        """Reduce compact per-clip caption rows into a feature result."""
        caption_result = reduce_caption_clip_results(
            expected_counts=workload.summary_counts,
            videos_only_in_a=workload.videos_only_in_a,
            videos_only_in_b=workload.videos_only_in_b,
            clip_results=tuple(
                sorted(
                    (CaptionClipCompareResult.from_json_dict(row) for row in rows),
                    key=lambda result: (result.video_key, result.clip_id),
                )
            ),
        )
        return _feature_result(caption_result)


def _feature_result(caption_result: CaptionComparisonResult) -> FeatureComparisonResult:
    return FeatureComparisonResult(issues=caption_result.issues, comparison=caption_result.comparison)


def _caption_video_clips(summary: OutputSummary) -> dict[str, tuple[str, ...]]:
    videos: dict[str, tuple[str, ...]] = {}
    for video_key, video in summary.videos.items():
        match video:
            case ProcessedVideoSummary() as processed_video:
                pass
            case _:
                continue
        if processed_video.num_clips_with_caption == 0 and processed_video.num_caption_windows == 0:
            continue
        videos[video_key] = processed_video.clips
    return videos


def _filter_caption_videos(
    caption_videos: Mapping[str, tuple[str, ...]],
    selected_video_keys: frozenset[str],
) -> dict[str, tuple[str, ...]]:
    return {video_key: clips for video_key, clips in caption_videos.items() if video_key in selected_video_keys}


def _has_caption_records(summary: OutputSummary, caption_videos: Mapping[str, tuple[str, ...]]) -> bool:
    return bool(caption_videos) or summary.total_num_clips_with_caption > 0 or summary.total_num_caption_windows > 0


def _selected_summary_counts(
    selected_video_keys: frozenset[str],
    summary_a: OutputSummary,
    summary_b: OutputSummary,
) -> CaptionComparisonCounts:
    videos_with_captions_a, clips_with_captions_a, caption_windows_a = _selected_caption_counts(
        summary_a, selected_video_keys
    )
    videos_with_captions_b, clips_with_captions_b, caption_windows_b = _selected_caption_counts(
        summary_b, selected_video_keys
    )
    return CaptionComparisonCounts(
        videos_with_captions_a=videos_with_captions_a,
        videos_with_captions_b=videos_with_captions_b,
        clips_with_captions_a=clips_with_captions_a,
        clips_with_captions_b=clips_with_captions_b,
        caption_windows_a=caption_windows_a,
        caption_windows_b=caption_windows_b,
    )


def _selected_caption_counts(summary: OutputSummary, selected_video_keys: frozenset[str]) -> tuple[int, int, int]:
    videos_with_captions = 0
    clips_with_captions = 0
    caption_windows = 0
    for video_key in selected_video_keys:
        match summary.videos.get(video_key):
            case ProcessedVideoSummary() as processed_video:
                pass
            case _:
                continue
        clips_with_captions += processed_video.num_clips_with_caption
        caption_windows += processed_video.num_caption_windows
        if processed_video.num_clips_with_caption > 0 or processed_video.num_caption_windows > 0:
            videos_with_captions += 1
    return videos_with_captions, clips_with_captions, caption_windows


def _summary_counts(
    caption_videos_a: Mapping[str, tuple[str, ...]],
    caption_videos_b: Mapping[str, tuple[str, ...]],
    summary_a: OutputSummary,
    summary_b: OutputSummary,
) -> CaptionComparisonCounts:
    return CaptionComparisonCounts(
        videos_with_captions_a=len(caption_videos_a),
        videos_with_captions_b=len(caption_videos_b),
        clips_with_captions_a=summary_a.total_num_clips_with_caption,
        clips_with_captions_b=summary_b.total_num_clips_with_caption,
        caption_windows_a=summary_a.total_num_caption_windows,
        caption_windows_b=summary_b.total_num_caption_windows,
    )
