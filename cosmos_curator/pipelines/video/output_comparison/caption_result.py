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
"""Caption comparison row and result types."""

from collections.abc import Mapping
from typing import Self

import attrs

from cosmos_curator.pipelines.video.output_comparison.caption_schema import CaptionComparisonCounts
from cosmos_curator.pipelines.video.output_comparison.json_types import JsonDictObject, JsonValue
from cosmos_curator.pipelines.video.output_comparison.report import FeatureComparison, Issue

CAPTIONS_FEATURE_NAME = "captions"


@attrs.define(frozen=True)
class CaptionClipCompareResult:
    """Compact comparison result emitted for one clip."""

    video_key: str
    clip_id: str
    in_a: bool
    in_b: bool
    issues: tuple[Issue, ...]
    counts: CaptionComparisonCounts
    missing_path_a: str | None = None
    missing_path_b: str | None = None
    invalid_path_a: str | None = None
    invalid_path_b: str | None = None

    def to_json_dict(self) -> JsonDictObject:
        """Convert the result to a Ray Data row."""
        return {
            "video_key": self.video_key,
            "clip_id": self.clip_id,
            "in_a": self.in_a,
            "in_b": self.in_b,
            "issues": [issue.to_json_dict() for issue in self.issues],
            "counts": self.counts.to_json_dict(),
            "missing_path_a": self.missing_path_a,
            "missing_path_b": self.missing_path_b,
            "invalid_path_a": self.invalid_path_a,
            "invalid_path_b": self.invalid_path_b,
        }

    @classmethod
    def from_json_dict(cls, row: Mapping[str, JsonValue]) -> Self:
        """Build a compact clip comparison result from a Ray Data row."""
        issues_value = row["issues"]
        counts_value = row["counts"]
        if not isinstance(issues_value, list) or not isinstance(counts_value, dict):
            error_msg = "caption clip result row has invalid issues or counts"
            raise TypeError(error_msg)
        return cls(
            video_key=_required_str(row, "video_key"),
            clip_id=_required_str(row, "clip_id"),
            in_a=_required_bool(row, "in_a"),
            in_b=_required_bool(row, "in_b"),
            issues=tuple(Issue.from_json_dict(issue) for issue in issues_value),
            counts=CaptionComparisonCounts.from_json_dict(counts_value),
            missing_path_a=_optional_str(row, "missing_path_a"),
            missing_path_b=_optional_str(row, "missing_path_b"),
            invalid_path_a=_optional_str(row, "invalid_path_a"),
            invalid_path_b=_optional_str(row, "invalid_path_b"),
        )


@attrs.define(frozen=True)
class CaptionComparisonResult:
    """Issues and counters emitted by caption structure comparison."""

    issues: tuple[Issue, ...]
    comparison: FeatureComparison


def _required_str(row: Mapping[str, JsonValue], field: str) -> str:
    value = row[field]
    if not isinstance(value, str):
        error_msg = f"caption result row field {field!r} must be a string"
        raise TypeError(error_msg)
    return value


def _required_bool(row: Mapping[str, JsonValue], field: str) -> bool:
    value = row[field]
    if not isinstance(value, bool):
        error_msg = f"caption result row field {field!r} must be a boolean"
        raise TypeError(error_msg)
    return value


def _optional_str(row: Mapping[str, JsonValue], field: str) -> str | None:
    value = row[field]
    if value is not None and not isinstance(value, str):
        error_msg = f"caption result row field {field!r} must be a string or null"
        raise TypeError(error_msg)
    return value
