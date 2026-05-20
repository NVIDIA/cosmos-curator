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
"""Report helpers for output comparison."""

import json
from collections.abc import Mapping, Sequence
from typing import Literal, Self, cast

import attrs

from cosmos_curator.pipelines.video.output_comparison.json_types import JsonDictObject, JsonValue, json_string_list


@attrs.define(frozen=True)
class Issue:
    """Structured JSON issue emitted by output comparison rules."""

    code: str
    message: str
    details: JsonDictObject | None = None
    output: str | None = None
    feature: str | None = None
    field: str | None = None
    video: str | None = None
    clip: str | None = None

    @classmethod
    def summary_load_failed(
        cls,
        path: str,
        output_label: str,
        error_type: str,
        error: str,
        *,
        field: str | None = None,
    ) -> Self:
        """Build a structured issue for summary loading failures."""
        output_name = output_label.upper()
        return cls(
            code="summary_load_failed",
            message=f"Failed to load output {output_name} summary at {path}: {error}",
            output=output_label,
            field=field,
            details={
                "path": path,
                "error_type": error_type,
                "error": error,
            },
        )

    def to_json_dict(self) -> JsonDictObject:
        """Convert the issue to a JSON-compatible dictionary."""
        return cast("JsonDictObject", attrs.asdict(self, filter=lambda _attribute, value: value is not None))

    @classmethod
    def from_json_dict(cls, row: JsonValue) -> Self:
        """Build an issue from a JSON-compatible dictionary."""
        if not isinstance(row, dict):
            error_msg = "issue row must be an object"
            raise TypeError(error_msg)
        return cls(
            code=_required_str(row, "code"),
            message=_required_str(row, "message"),
            details=cast("JsonDictObject | None", row.get("details")),
            output=_optional_str(row, "output"),
            feature=_optional_str(row, "feature"),
            field=_optional_str(row, "field"),
            video=_optional_str(row, "video"),
            clip=_optional_str(row, "clip"),
        )


@attrs.define(frozen=True)
class SummaryComparison:
    """Summary comparison counters and video set details."""

    videos_in_both: int = 0
    videos_only_in_a: tuple[str, ...] = ()
    videos_only_in_b: tuple[str, ...] = ()
    exact_top_level_fields_compared: int = 0
    token_fields_compared: int = 0
    per_video_fields_compared: int = 0

    def to_json_dict(self) -> JsonDictObject:
        """Convert summary comparison counters to a JSON-compatible dictionary."""
        summary_comparison = attrs.asdict(self)
        summary_comparison["videos_only_in_a"] = json_string_list(self.videos_only_in_a)
        summary_comparison["videos_only_in_b"] = json_string_list(self.videos_only_in_b)
        return cast("JsonDictObject", summary_comparison)


type FeatureComparisonStatus = Literal["skipped", "passed", "failed"]


@attrs.define(frozen=True)
class FeatureComparison:
    """Feature comparison status and metrics."""

    status: FeatureComparisonStatus
    metrics: JsonDictObject = attrs.field(factory=dict)

    def to_json_dict(self) -> JsonDictObject:
        """Convert feature comparison data to a JSON-compatible dictionary."""
        return {"status": self.status, "metrics": self.metrics}


@attrs.define(frozen=True)
class ComparisonReport:
    """Typed comparison report returned by the public comparison API."""

    output_a: str
    output_b: str
    summary_comparison: SummaryComparison
    issues: tuple[Issue, ...]
    feature_comparisons: Mapping[str, FeatureComparison] = attrs.field(factory=dict)

    @classmethod
    def from_issues(
        cls,
        output_a: str,
        output_b: str,
        summary_comparison: SummaryComparison,
        issues: Sequence[Issue],
        feature_comparisons: Mapping[str, FeatureComparison] | None = None,
    ) -> Self:
        """Build a report from comparison issues."""
        return cls(
            output_a=output_a,
            output_b=output_b,
            summary_comparison=summary_comparison,
            issues=tuple(issues),
            feature_comparisons=feature_comparisons or {},
        )

    @property
    def passed(self) -> bool:
        """Return whether comparison completed without issues."""
        return not self.issues

    def to_json_dict(self) -> JsonDictObject:
        """Convert the report to a JSON-compatible dictionary."""
        return cast(
            "JsonDictObject",
            {
                "passed": self.passed,
                "output_a": self.output_a,
                "output_b": self.output_b,
                "summary_comparison": self.summary_comparison.to_json_dict(),
                "feature_comparisons": {
                    name: comparison.to_json_dict() for name, comparison in sorted(self.feature_comparisons.items())
                },
                "issues": [issue.to_json_dict() for issue in self.issues],
            },
        )


def report_to_json(report: ComparisonReport) -> str:
    """Serialize a comparison report to a stable JSON string.

    Args:
        report: Typed report returned by ``compare_split_outputs``.

    Returns:
        Indented JSON string with deterministic key ordering.

    """
    return json.dumps(report.to_json_dict(), indent=2, sort_keys=True)


def _required_str(row: Mapping[str, JsonValue], field: str) -> str:
    value = row[field]
    if not isinstance(value, str):
        error_msg = f"issue row field {field!r} must be a string"
        raise TypeError(error_msg)
    return value


def _optional_str(row: Mapping[str, JsonValue], field: str) -> str | None:
    value = row.get(field)
    if value is None or isinstance(value, str):
        return value
    error_msg = f"issue row field {field!r} must be a string when present"
    raise TypeError(error_msg)
