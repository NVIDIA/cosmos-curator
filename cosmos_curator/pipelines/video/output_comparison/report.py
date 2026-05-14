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
from collections.abc import Sequence
from typing import Self, cast

import attrs

from cosmos_curator.pipelines.video.output_comparison.json_types import JsonDictObject, json_string_list


@attrs.define(frozen=True)
class Issue:
    """Structured JSON issue emitted by output comparison rules."""

    code: str
    message: str
    details: JsonDictObject | None = None
    output: str | None = None
    field: str | None = None
    video: str | None = None

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


@attrs.define(frozen=True)
class ComparisonReport:
    """Typed comparison report returned by the public comparison API."""

    output_a: str
    output_b: str
    summary_comparison: SummaryComparison
    issues: tuple[Issue, ...]

    @classmethod
    def from_issues(
        cls,
        output_a: str,
        output_b: str,
        summary_comparison: SummaryComparison,
        issues: Sequence[Issue],
    ) -> Self:
        """Build a report from comparison issues."""
        return cls(
            output_a=output_a,
            output_b=output_b,
            summary_comparison=summary_comparison,
            issues=tuple(issues),
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
