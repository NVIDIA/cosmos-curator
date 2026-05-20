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
"""Build caption structure from split pipeline caption metadata."""

from cosmos_curator.pipelines.video.output_comparison.caption_policy import CaptionComparisonPolicy
from cosmos_curator.pipelines.video.output_comparison.caption_schema import (
    CaptionWindowRange,
    ClipCaptionView,
)
from cosmos_curator.pipelines.video.output_comparison.json_types import JsonDictObject, JsonValue
from cosmos_curator.pipelines.video.output_comparison.video_artifacts import LoadedClipArtifacts


def caption_view_from_clip_artifacts(
    artifacts: LoadedClipArtifacts,
    *,
    policy: CaptionComparisonPolicy,
) -> ClipCaptionView:
    """Build a normalized caption view from one loaded clip artifact row."""
    return ClipCaptionView(
        video_key=artifacts.spec.video_key,
        clip_id=artifacts.spec.clip_id,
        in_a=artifacts.spec.in_a,
        in_b=artifacts.spec.in_b,
        windows_a=frozenset(_caption_window_ranges(artifacts.metadata_a, policy=policy))
        if artifacts.metadata_a is not None
        else frozenset(),
        windows_b=frozenset(_caption_window_ranges(artifacts.metadata_b, policy=policy))
        if artifacts.metadata_b is not None
        else frozenset(),
        metadata_path_a=artifacts.metadata_path_a,
        metadata_path_b=artifacts.metadata_path_b,
        missing_metadata_a=artifacts.missing_metadata_a,
        missing_metadata_b=artifacts.missing_metadata_b,
        invalid_metadata_a=artifacts.invalid_metadata_a,
        invalid_metadata_b=artifacts.invalid_metadata_b,
    )


def _caption_window_ranges(metadata: JsonDictObject, *, policy: CaptionComparisonPolicy) -> set[CaptionWindowRange]:
    windows_value = metadata.get("windows")
    if not isinstance(windows_value, list):
        return set()
    windows: set[CaptionWindowRange] = set()
    for window_value in windows_value:
        if not isinstance(window_value, dict) or not all(isinstance(key, str) for key in window_value):
            continue
        window_range = _caption_window_range(window_value, policy=policy)
        if window_range is not None:
            windows.add(window_range)
    return windows


def _caption_window_range(window: JsonDictObject, *, policy: CaptionComparisonPolicy) -> CaptionWindowRange | None:
    start_frame = _int_value(window.get("start_frame"))
    end_frame = _int_value(window.get("end_frame"))
    if start_frame is None or end_frame is None or not _has_regular_caption(window, policy=policy):
        return None
    return CaptionWindowRange(start_frame=start_frame, end_frame=end_frame)


def _has_regular_caption(window: JsonDictObject, *, policy: CaptionComparisonPolicy) -> bool:
    if not policy.is_caption_ok_status(window.get("caption_status")):
        return False
    return any(policy.is_regular_caption_field(key) and isinstance(value, str) for key, value in window.items())


def _int_value(value: JsonValue) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
